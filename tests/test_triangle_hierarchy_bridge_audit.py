"""Failed support never releases a quality audit; explicit diagnostics stay separate."""
import json
from pathlib import Path

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_hierarchy_audit import run_h_bridge_endpoint_audit
from su2_nqs.frontier.provenance import file_sha256


def _fixture(tmp_path, *, reached=True, passed=True, parent_j2=.125, bad_bits=False):
    model = dict(length=12, multiplicity_cap=20, maximum_q=10)
    control, candidate = tmp_path/'old.pkl', tmp_path/'new.pkl'
    control.write_bytes(b'not loaded before validation')
    candidate.write_bytes(b'not loaded before validation either')
    spec = dict(model=model, j2_over_j1=.125, active_top_widths=[{'0': 8}]*14+[{'0': 1}],
        control_checkpoint=str(control), candidate_checkpoint=str(candidate),
        control_sha256=file_sha256(control), candidate_sha256=file_sha256(candidate),
        chains=8, records=8, burn_proposals=64, proposals_between=16, seconds_cap=1800)
    states = np.tile(np.repeat([0, 1], 72), (8, 1))
    if bad_bits:
        states[0, 0] = -1; states[0, 1] = 1  # sum is still72, but illegal bits.
    sample = tmp_path/'support.npz'
    np.savez_compressed(sample, terminal_states=states, terminal_psi=np.ones((8, 2)))
    report = tmp_path/'support.json'
    report.write_text(json.dumps(dict(fixed_endpoint_reached=reached,
        coverage=dict(operational_gate_passed=passed),
        primary16_coverage=dict(operational_gate_passed=False))))
    registration = tmp_path/'registration.json'
    registration.write_text(json.dumps(dict(spec=spec | {'j2_over_j1': parent_j2})))
    for label, path in (('support_samples', sample), ('support_results', report),
                        ('support_registration', registration)):
        spec[label+'_path'] = str(path)
        spec[label+'_sha256'] = file_sha256(path)
    return spec


def test_bridge_failed_support_is_not_a_reason_to_query_full_H(tmp_path):
    spec = _fixture(tmp_path, passed=False)
    with pytest.raises(ValueError, match='support checks must pass'):
        run_h_bridge_endpoint_audit(spec, tmp_path/'audit')
    assert not (tmp_path/'audit').exists()


def test_bridge_passed_different_H_support_is_rejected(tmp_path):
    spec = _fixture(tmp_path, parent_j2=0.)
    with pytest.raises(ValueError, match='same endpoints/layout/H'):
        run_h_bridge_endpoint_audit(spec, tmp_path/'audit')
    assert not (tmp_path/'audit').exists()


def test_bridge_preserving_Sz_does_not_make_invalid_bits_legal(tmp_path):
    spec = _fixture(tmp_path, bad_bits=True)
    with pytest.raises(ValueError, match='fixed-Sz support endpoints'):
        run_h_bridge_endpoint_audit(spec, tmp_path/'audit')
    assert not (tmp_path/'audit').exists()


def test_bridge_changed_support_file_rejected_before_audit(tmp_path):
    spec = _fixture(tmp_path)
    spec['support_samples_sha256'] = '0'*64
    with pytest.raises(ValueError, match='support input hash mismatch'):
        run_h_bridge_endpoint_audit(spec, tmp_path/'audit')
    assert not (tmp_path/'audit').exists()


@pytest.mark.parametrize('passed', [False, True])
def test_bridge_incomplete_fixed_endpoint_never_releases_H(tmp_path, passed):
    spec = _fixture(tmp_path, reached=False, passed=passed)
    with pytest.raises(ValueError, match='support checks must pass'):
        run_h_bridge_endpoint_audit(spec, tmp_path/'audit')
    assert not (tmp_path/'audit').exists()


def test_prospective_secondary_gate_is_separate_from_original_primary_result(tmp_path, monkeypatch):
    from projects.clebsch_tree_nqs.workflows import triangle_hierarchy_audit as audit
    spec = _fixture(tmp_path, passed=True)
    class StoppedBeforeNumerics(RuntimeError):
        pass
    def stop(*args):
        raise StoppedBeforeNumerics
    monkeypatch.setattr(audit, 'load_checkpoint', stop)
    with pytest.raises(StoppedBeforeNumerics):
        run_h_bridge_endpoint_audit(spec, tmp_path/'audit')
    assert not (tmp_path/'audit').exists()


def _coverage_spec(spec):
    return spec | dict(audit_purpose='coverage_diagnostic', separately_registered_diagnostic=True,
        no_benchmark=True, no_attention=True, records=32, burn_proposals=128,
        proposals_between=128, seconds_cap=3600, seed=65012)


@pytest.mark.parametrize('change', [dict(separately_registered_diagnostic=False),
    dict(no_benchmark=False), dict(no_attention=False), dict(no_attention=1),
    dict(records=31), dict(burn_proposals=64), dict(proposals_between=64),
    dict(seconds_cap=3601), dict(audit_purpose='force_quality_audit'), dict(seed=None),
    dict(seed=True), dict(seed=-1), dict(seed=2**32)])
def test_diagnostic_guards_fail_before_input_reads_or_physical_queries(tmp_path, change):
    spec = dict(model=dict(length=12,multiplicity_cap=20,maximum_q=10), chains=8)
    with pytest.raises(ValueError):
        run_h_bridge_endpoint_audit(_coverage_spec(spec) | change, tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('invalid', ['incomplete', 'different_H', 'bad_bits', 'changed_samples'])
def test_diagnostic_retains_fixed_endpoint_and_physical_identity_guards(tmp_path, invalid):
    spec = _coverage_spec(_fixture(tmp_path, passed=False, reached=invalid!='incomplete',
        parent_j2=0. if invalid=='different_H' else .125, bad_bits=invalid=='bad_bits'))
    if invalid=='changed_samples': spec['support_samples_sha256']='0'*64
    with pytest.raises(ValueError):
        run_h_bridge_endpoint_audit(spec, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_explicit_diagnostic_may_reach_loading_but_does_not_change_failed_support(tmp_path, monkeypatch):
    from projects.clebsch_tree_nqs.workflows import triangle_hierarchy_audit as audit
    spec = _coverage_spec(_fixture(tmp_path, passed=False))
    before = file_sha256(Path(spec['support_results_path']))
    class StopBeforePhysics(RuntimeError): pass
    def stop(*args): raise StopBeforePhysics
    monkeypatch.setattr(audit, 'load_checkpoint', stop)
    with pytest.raises(StopBeforePhysics):
        run_h_bridge_endpoint_audit(spec, tmp_path/'out')
    assert file_sha256(Path(spec['support_results_path'])) == before
    assert not (tmp_path/'out').exists()


def test_diagnostic_result_preserves_failed_gate_and_cannot_promote(tmp_path, monkeypatch):
    import jax.numpy as jnp
    from projects.clebsch_tree_nqs.workflows import triangle_hierarchy_audit as audit
    from su2_nqs.vmc import triangular_clebsch_cached_action as cached
    spec = _coverage_spec(_fixture(tmp_path, passed=False)) | {'seed':65012}
    monkeypatch.setattr(audit, 'load_checkpoint', lambda path:{'params':{'c':jnp.asarray(1.)}})
    monkeypatch.setattr(audit, 'repack_active_clebsch_bank', lambda b,*args:b)
    monkeypatch.setattr(audit, 'prepare_feature_lut_bank', lambda b,*args:b)
    monkeypatch.setattr(audit, 'triangular_hnf_gamma_a1_group',
        lambda *args:np.tile(np.arange(144,dtype=np.int32),(1728,1)))
    monkeypatch.setattr(audit, 'triangular_j1j2_bonds', lambda *args,**kwargs:[(0,1,1.)])
    monkeypatch.setattr(audit, 'make_feature_lut_projected_amplitude',
        lambda *args,**kwargs:lambda b,x:b['c'])
    def action_factory(*args,**kwargs):
        def action(bank,rows):
            one=jnp.ones(len(rows));return one,-10*one,one
        return action,{'test_mock_only':True}
    monkeypatch.setattr(cached, 'make_cached_projected_heisenberg_bridge', action_factory)
    def sampler_factory(*args,records,**kwargs):
        def sample(bank,key,states):
            return states,key,jnp.repeat(states[None],records,axis=0),{}
        return sample
    monkeypatch.setattr(audit, 'make_record_sampler', sampler_factory)
    result = run_h_bridge_endpoint_audit(spec,tmp_path/'out')
    assert result['status']=='fresh_paired_H_bridge_coverage_diagnostic_complete'
    assert result['full_H_centers']==516
    assert result['diagnostic_only'] and result['original_support_gate_unchanged']
    assert not result['support_stage_operational_gate_passed']
    assert not result['publication_quality_gate_passed']
    assert not result['benchmark_eligible'] and not result['attention_release']
    assert not result['chain_jackknife_covers_unknown_equilibration_bias']
    registration=json.loads((tmp_path/'out/registration.json').read_text())
    assert registration['audit_purpose']=='coverage_diagnostic'
    assert registration['support_stage_operational_gate_passed'] is False
    with np.load(tmp_path/'out/untouched_products.npz') as data:
        assert data['psi'].shape==(256,2)
        np.testing.assert_array_equal(data['groups'],np.tile(np.arange(8),32))
    assert json.loads(Path(spec['support_results_path']).read_text())['coverage']['operational_gate_passed'] is False
