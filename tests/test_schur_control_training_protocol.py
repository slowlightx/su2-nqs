"""Independent protocol checks before new training outcomes are available."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_schur_control_training import (
    combine_development_metric, ray_residual_concentration, expansion_gate,
)
from su2_nqs.vmc.cutset_training import weighted_state_distance

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_schur_control_training.json').read_text())


def test_metric_is_fixed_equal_normalized_gram_not_unweighted_added_tail_points():
    rng = np.random.default_rng(919)
    corpora = []
    for size in (8, 12):
        c = rng.normal(size=size)+1j*rng.normal(size=size)
        corpora.append(dict(carrier=c,q=np.exp(rng.normal(size=size)),states=np.zeros((size,4)),
                            reference=rng.normal(size=size)+1j*rng.normal(size=size)))
    metric = combine_development_metric(*corpora)
    np.testing.assert_allclose(np.sum(abs(metric['carrier'])**2/metric['q']),1,atol=3e-15)
    expected = np.zeros((2,2),dtype=complex)
    for data in corpora:
        columns = np.column_stack((data['carrier'],data['reference']))
        gram = columns.conj().T@(columns/data['q'][:,None])
        expected += .5*gram/gram[0,0].real
    columns = np.column_stack((metric['carrier'],metric['reference']))
    np.testing.assert_allclose(columns.conj().T@(columns/metric['q'][:,None]),expected,atol=3e-15)
    scaled = [{**d,'q':d['q']*factor} for d,factor in zip(corpora,(3.2,.3),strict=True)]
    repeat = combine_development_metric(*scaled)
    np.testing.assert_allclose(repeat['q'],metric['q'],rtol=2e-15)
    with pytest.raises(ValueError):
        combine_development_metric({**corpora[0],'q':-corpora[0]['q']},corpora[1])


def test_ray_residual_concentration_ignores_carrier_gauge_and_parametric_residual():
    rng = np.random.default_rng(123)
    c = rng.normal(size=16)+1j*rng.normal(size=16)
    psi = c+.1*(rng.normal(size=16)+1j*rng.normal(size=16))
    q = np.exp(rng.normal(size=16)); groups = np.tile(np.arange(4),4)
    result = ray_residual_concentration(c,psi,q,groups)
    invariant = ray_residual_concentration(c*(.3-.7j),2j*psi+(.2+.3j)*c,3*q,groups)
    for key in result: assert invariant[key] == pytest.approx(result[key],abs=2e-14)


def _gate_fixture():
    pair = dict(energy_per_site=dict(value=-.001,chain_bootstrap_95=[-.002,-.0002]),
                variance_per_site=dict(value=-.001,chain_jackknife_se=.0003))
    audit = dict(paired={'control-minus-carrier':pair,'control-minus-diagonal':copy.deepcopy(pair)},
        arms={name:dict(variance_per_site=dict(value=.02),weight_ess_fraction=.7,
                       max_residual_chain_fraction=.3) for name in ('carrier','diagonal','control')})
    training = dict(history=[dict(accepted_pair=True) for _ in range(6)],
                    final_metric_distance={name:dict(fs=.04) for name in ('diagonal','control')})
    distance = {name:dict(fs=.045) for name in ('diagonal','control')}
    concentration = {name:dict(max_chain_fraction=.3) for name in ('diagonal','control')}
    return audit,training,distance,concentration


def test_frozen_gate_separates_operational_release_from_claim_of_estimator_win():
    audit,training,distance,residual = _gate_fixture()
    audit['paired']['control-minus-diagonal']['energy_per_site']['chain_bootstrap_95'][1] = .0001
    result = expansion_gate(audit,training,distance,residual,SPEC)
    assert result['passed'] and not result['CV_beats_diagonal_energy_95']
    assert not result['production_authorized']
    audit['paired']['control-minus-carrier']['energy_per_site']['chain_bootstrap_95'][1] = .0001
    assert not expansion_gate(audit,training,distance,residual,SPEC)['passed']


def test_large_metric_drift_or_rejected_step_cannot_release_N64():
    audit,training,distance,residual = _gate_fixture()
    distance['control']['fs'] = .08
    gate = expansion_gate(audit,training,distance,residual,SPEC)
    assert not gate['passed'] and not gate['checks']['control_metric_transfer']
    assert not gate['checks']['audit_between_arm_distance_match']
    distance['control']['fs'] = .045
    training['history'][0]['accepted_pair'] = False
    assert not expansion_gate(audit,training,distance,residual,SPEC)['passed']
    training['history'][0]['accepted_pair'] = True
    residual['control']['max_chain_fraction'] = .61
    assert not expansion_gate(audit,training,distance,residual,SPEC)['passed']


def test_six_fixed_radius_steps_respect_metric_triangle_inequality():
    c = np.array([1.,0.]); q = np.ones(2); old = c
    for step in range(1,7):
        new = np.array([np.cos(.01*step),np.sin(.01*step)])
        assert weighted_state_distance(old,new,q)['fs'] == pytest.approx(.01,abs=3e-14)
        old = new
    assert weighted_state_distance(c,old,q)['fs'] == pytest.approx(.06,abs=3e-14)
