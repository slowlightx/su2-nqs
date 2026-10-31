"""Reusable analysis fails closed on raw seals and retains missing errors."""
import json
from pathlib import Path

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import spin_sector_entanglement_analysis as workflow
from su2_nqs.frontier.provenance import file_sha256


def test_relative_config_controls_and_missing_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    monkeypatch.setattr(workflow, 'transitive_local_source_manifest', lambda *a, **k: {})
    monkeypatch.chdir(tmp_path)
    series = tmp_path/'series'; series.mkdir()
    row = dict(label='dummy',method='NQS',sites=list(range(6)),left=[0,1,2],
               cut='region',problem=dict(geometry='triangular',j2=.125,boundary='PBC'),
               physical_problem_signature='test',energy=None,step=10)
    (series/'results.json').write_text(json.dumps(dict(records=[row])))
    np.savez(series/'readouts.npz', dummy__region__rho=np.eye(64)/64)
    spec = dict(output='analysis',series='series',seconds_cap=10,
                series_results_sha256=file_sha256(series/'results.json'),
                series_readouts_sha256=file_sha256(series/'readouts.npz'))
    config = Path('config.json'); config.write_text(json.dumps(spec))
    result=workflow.analyze(config)
    assert result['records'][0]['whole_chain_SE'] == {}
    assert result['records'][0]['witness_SE_upper_bounds'] == {}
    assert result['new_H_calls'] == result['new_training'] == 0
    assert not result['strict_entanglement_spectrum']
    assert len(result['controls']) == 3
    assert (tmp_path/'analysis/completion.json').is_file()
    with pytest.raises(FileExistsError):workflow.analyze(config)


def test_changed_raw_input_fails_before_output(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow,'ROOT',tmp_path)
    series=tmp_path/'series';series.mkdir()
    (series/'results.json').write_text('{}')
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(output='analysis',series='series',seconds_cap=10,
                                      series_results_sha256='wrong',series_readouts_sha256='wrong')))
    with pytest.raises(ValueError,match='sealed raw'):workflow.analyze(config)
    assert not (tmp_path/'analysis').exists()


def test_linear_witness_error_bound_does_not_assume_independent_weights():
    row={'whole_chain_SE':{'joint_qL1_qR1_qT0':.1,'joint_qL1_qR1_qT2':.2,
          'joint_qL3_qR3_qT0':.1,'joint_qL3_qR3_qT2':.2,
          'joint_qL3_qR3_qT4':.3,'joint_qL3_qR3_qT6':.4}}
    upper=workflow._conservative_witness_errors(row)
    assert upper['1,1__screening_witness'] == pytest.approx(.15)
    assert upper['3,3__screening_witness'] == pytest.approx(.3)


def test_spin_isotropy_preserves_cross_observable_chain_covariance(tmp_path, monkeypatch):
    from projects.clebsch_tree_nqs.workflows import spin_spatial_vit_quality
    metadata = dict(energy=dict(samples=4, independent_chains=2),
                    structure_factors={'K': {}},
                    total_spin_squared=dict(value=.04, chain_SE=.006))
    (tmp_path/'results.json').write_text(json.dumps(metadata))
    # Individual spin estimators fluctuate together. Taking independent
    # marginal errors would miss their EXACT cancellation in this example.
    zz=np.array([[1.],[2.],[4.],[5.]])
    scalar=3*zz+.2
    np.savez(tmp_path/'structure.npz', chain_ids=[0,0,1,1], scalar=scalar, zz=zz)
    (tmp_path/'structure.json').write_text(json.dumps(
        dict(raw_sha256=file_sha256(tmp_path/'structure.npz'))))
    monkeypatch.setattr(spin_spatial_vit_quality, 'read_sealed',
                        lambda path:json.loads(path.read_text()))
    result=workflow.audit_spin_isotropy(tmp_path,n_sites=36)
    assert result['rank_two_correlations']['K']['scalar_minus_3zz'] == pytest.approx(.2)
    assert result['rank_two_correlations']['K']['paired_chain_SE'] < 1e-14
    assert result['nonsinglet_weight_upper_bound_point_estimate'] == pytest.approx(.02)
    assert result['point_estimate_SE'] == pytest.approx(.003)
    assert not result['strict_su2']
    with pytest.raises(ValueError,match='even positive'):
        workflow.audit_spin_isotropy(tmp_path,n_sites=35)
    with pytest.raises(ValueError,match='even positive'):
        workflow.audit_spin_isotropy(tmp_path,n_sites=True)
    (tmp_path/'structure.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='seal changed'):
        workflow.audit_spin_isotropy(tmp_path,n_sites=36)


def test_agp_streams_share_protocol_without_fake_rdm_pooling(tmp_path, monkeypatch):
    from su2_nqs.frontier.artifacts import canonical_sha256
    monkeypatch.setattr(workflow,'ROOT',tmp_path)
    monkeypatch.setattr(workflow,'transitive_local_source_manifest',lambda *a,**k:{})
    def sealed(path, record):
        record={**record,'result_sha256':canonical_sha256(record)}
        path.write_text(json.dumps(record))
        return record
    series=tmp_path/'series';series.mkdir()
    (series/'results.json').write_text(json.dumps({'records':[]}))
    np.savez(series/'readouts.npz')
    agp=tmp_path/'agp';agp.mkdir()
    case=dict(name='toy',lattice='triangular',j2_over_j1=.125)
    region=dict(sites=[12,13,14,18,19,20],left=[0,1,2])
    registration=sealed(agp/'registration.json',
        dict(config=dict(cases=[case],regions=dict(compact6=region))))
    child=agp/'toy';child.mkdir()
    frozen={'step_012':'fixed-checkpoint-hash'}
    sealed(child/'registration.json',dict(case=case,
        suite_registration_sha256=registration['result_sha256']))
    sealed(child/'results.json',dict(case=case,complete_bounded_audit=True,
                                    frozen_checkpoints=frozen))
    sealed(child/'frozen_endpoint.json',dict(checkpoints=frozen))
    for stream in (0,1):
        rho=np.eye(64)/64 if stream==0 else np.diag([1.]+[0.]*63)
        raw=child/f'audit_{stream}.npz'
        np.savez(raw,compact6_step_012_rho=rho,
                 compact6_step_012_jackknife_rhos=np.stack([rho,rho]))
        sealed(child/f'audit_{stream}.json',dict(checkpoints=frozen,
            raw_file_sha256=file_sha256(raw),states=dict(step_012=dict(
                energy_per_site=-.51,energy_chain_SE=.01,
                variance_per_site=.1,variance_chain_SE=.01))))
    sealed(child/'completion.json',dict(status='complete',outputs={
        path.name:file_sha256(path) for path in child.iterdir()}))
    config=tmp_path/'config.json'
    spec=dict(output='analysis',series='series',seconds_cap=10,
        series_results_sha256=file_sha256(series/'results.json'),
        series_readouts_sha256=file_sha256(series/'readouts.npz'),
        agp_cached=[dict(directory='agp',registration_sha256=file_sha256(agp/'registration.json'))])
    config.write_text(json.dumps(spec))
    result=workflow.analyze(config)
    assert len(result['records'])==2
    assert all(row['method']=='AGP' for row in result['records'])
    assert result['records'][0]['report']['sectors']['3,3']['probability'] == pytest.approx(.25)
    assert result['records'][1]['report']['sectors']['3,3']['probability'] == pytest.approx(1)
    assert result['new_samples']==result['new_H_calls']==0
    # Raw artifact corruption must fail closed before a new completion.
    (child/'audit_1.npz').write_bytes(b'changed')
    spec['output']='bad';config.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='sealed raw input'):
        workflow.analyze(config)
    assert not (tmp_path/'bad').exists()
