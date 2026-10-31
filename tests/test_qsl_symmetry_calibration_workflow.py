"""Small frozen reference runs and fail-closed provenance, no neural queries."""
import json

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import qsl_symmetry_calibration as workflow
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256


@pytest.fixture
def fixed(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,'ROOT',tmp_path)
    monkeypatch.setattr(workflow,'transitive_local_source_manifest',lambda *args,**kwargs:{})
    config=dict(output='output',seconds_cap=20,angle_points=9,kagome_geometries=[[2,2]],
        physical_regions=[[0,1]],local_algebra_regions=[[0,1]],boundary_lengths=[4],
        vacuum_weight=.65,spin_references=[],render_figure=False)
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    return path


def test_known_family_and_mechanism_remain_separate(fixed):
    result=workflow.analyze(fixed)
    assert result['references'][0]['coverings']==32
    assert not result['references'][0]['finite_representative_topology_certified']
    assert not result['conditional_boundary_mechanism'][0]['known_bulk_positive']
    assert result['new_neural_H_calls']==result['new_samples']==result['new_training']==0
    assert not result['QSL_claim'] and not result['cusp_detected_in_triangular_data']
    assert result['local_tensor_identities']['maximum_SU2_pullthrough_error']<2e-14
    output=fixed.parent/'output'
    for name in ['registration.json','results.json','completion.json']:
        record=json.loads((output/name).read_text())
        assert record['result_sha256']==canonical_sha256({k:v for k,v in record.items() if k!='result_sha256'})
    with pytest.raises(FileExistsError):
        workflow.analyze(fixed)


def add_reference(fixed):
    root=fixed.parent
    c=.75*np.eye(36)
    for i in range(0,36,2):
        c[i,i+1]=c[i+1,i]=-.75
    np.savez(root/'spin.npz',spin_correlations=c)
    report=dict(status='complete',checkpoint_sha256='actual-parent',input_repaired=False,
        checkpoint_unchanged=True)
    (root/'spin.json').write_text(json.dumps(report))
    cfg=json.loads(fixed.read_text())
    cfg['spin_references']=[dict(label='short_range_dimer_NOT_ground',chi=1,
        energy_per_site=-.25,checkpoint_sha256='actual-parent',arrays='spin.npz',report='spin.json',
        arrays_sha256=file_sha256(root/'spin.npz'),report_sha256=file_sha256(root/'spin.json'))]
    fixed.write_text(json.dumps(cfg))


def test_reference_static_ratio_not_promoted_to_spectrum(fixed):
    add_reference(fixed)
    r=workflow.analyze(fixed)['static_oscillator_references'][0]
    assert len(r['momenta'])==36 and not r['H_variance_available']
    assert all(not p['exact_ground_state_gap_upper_bound'] for p in r['momenta'])
    assert r['momenta'][0]['spatially_constant_mode']


def test_changed_input_refused_before_registration(fixed):
    add_reference(fixed)
    p=fixed.parent/'spin.json';p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='input changed'):
        workflow.analyze(fixed)
    assert not (fixed.parent/'output').exists()
