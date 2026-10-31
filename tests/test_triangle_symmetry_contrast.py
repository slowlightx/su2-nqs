"""Independent physical rotations and frozen offline quality diagnostics."""
import json

import numpy as np
import pytest
from scipy.linalg import expm

from projects.clebsch_tree_nqs.workflows import triangle_symmetry_contrast as workflow
from su2_nqs.benchmarks.spin_operator_information import joint_spin_operator_split
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256


@pytest.mark.parametrize('n',[1,2,4])
def test_rotation_operator_bank_against_independent_spin_generators(n):
    angles=np.array([0.,.43,np.pi,2*np.pi])
    bank=workflow.rotation_operators(n,angles)
    for axis,single in [('z',np.diag([-.5,.5])),('x',np.array([[0.,.5],[.5,0.]]))]:
        g=np.zeros((2**n,2**n))
        for site in range(n):
            term=np.array([[1.]])
            for i in range(n):
                term=np.kron(term,single if i==site else np.eye(2))
            g+=term
        np.testing.assert_allclose(bank[axis],np.array([expm(1j*t*g) for t in angles]),atol=2e-13,rtol=0)
    # The maximally mixed state has the same true axes and scalar moment.
    rho=np.eye(2**n)/2**n
    for ops in bank.values():
        expected=np.cos(angles/2)**n
        np.testing.assert_allclose(workflow.linear_means(rho[None],ops)[0],expected,atol=2e-13,rtol=0)


def test_jackknife_keeps_paired_covariance():
    a=np.array([[1.,4.],[2.,6.],[3.,8.]])
    b=2*a
    np.testing.assert_allclose(workflow.jackknife_se(b-a),workflow.jackknife_se(a),atol=1e-14)
    incorrect=np.hypot(workflow.jackknife_se(a),workflow.jackknife_se(b))
    assert np.all(incorrect>workflow.jackknife_se(b-a))


def test_quality_drift_can_reject_a_statistically_large_endpoint_difference():
    result=workflow.quality_contrast([0.,.3],[.5,.4],.01,.01)
    assert result['difference']==pytest.approx(.1)
    assert result['empirical_threshold']>.4
    assert not result['exceeds_empirical_threshold']
    assert not result['quality_ranges_are_certified_bias_bounds']


@pytest.mark.parametrize('n,angles',[(True,np.array([1.])),(0,np.array([1.])),(2,np.array([.3j])),(2,np.array([]))])
def test_invalid_operator_inputs(n,angles):
    with pytest.raises(ValueError):
        workflow.rotation_operators(n,angles)


@pytest.fixture
def fixed(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,'ROOT',tmp_path)
    monkeypatch.setattr(workflow,'transitive_local_source_manifest',lambda *a,**k:{})
    def sealed(path,payload):
        result={**payload,'result_sha256':canonical_sha256(payload)}
        path.write_text(json.dumps(result))
        return result
    regions={'compact6':[12,13,14,18,19,20],
             'local_hexagon6':[20,21,15,8,7,13],'winding6':[12,13,14,15,16,17]}
    split=joint_spin_operator_split(6,[0,1,2],(3,3,0))
    entries=[]
    for j2,label in [(0,'a'),(.125,'b')]:
        directory=tmp_path/label; directory.mkdir()
        rho=np.eye(64)/64
        arrays={}
        for cut in regions:
            arrays[cut+'__rho']=rho
            arrays[cut+'__jackknife_rhos']=np.repeat(rho[None],3,axis=0)
        np.savez(directory/'physics.npz',**arrays)
        config=dict(params_sha256='frozen-'+label,problem=dict(geometry='triangular',j2=j2,
            boundary='PBC_both_physical',rows=6,columns=6,regions=regions,
            left_sites={k:[0,1,2] for k in regions}))
        reg=sealed(directory/'registration.json',dict(config=config))
        values=split.measure(rho)
        sealed(directory/'results.json',dict(step=1,full_A1=True,strict_SU2=False,
            raw_sha256=file_sha256(directory/'physics.npz'),params_sha256='frozen-'+label,
            registration_sha256=reg['result_sha256'],independent_chains=3,records=12,
            regions={k:dict(scalars=dict(p_qL3_qR3_qT0=values['full'],projector_q3_high=values['high'])) for k in regions}))
        entries.append(dict(label=label,j2=j2,step=1,directory=label,
            results_sha256=file_sha256(directory/'results.json'),
            registration_sha256=file_sha256(directory/'registration.json'),
            arrays_sha256=file_sha256(directory/'physics.npz')))
    settings=dict(output='output',angle_points=5,seconds_cap=20,multiplier_replicates=32,
        multiplier_seed=1,point_threshold_SE=2.,inputs=entries,regions=regions,left_sites=[0,1,2],
        fixed_scalar_readouts=['p_qL3_qR3_qT0','projector_q3_high'])
    path=tmp_path/'config.json';path.write_text(json.dumps(settings))
    return path


def test_frozen_zero_query_workflow(fixed):
    result=workflow.analyze(fixed)
    assert len(result['records'])==6 and len(result['contrasts'])==15
    assert result['new_H_calls']==result['new_samples']==result['new_training']==result['new_amplitude_queries']==0
    assert not result['phase_claim'] and not result['QSL_specific_signal']
    assert result['diagnostics']['core_reference_error']<2e-13
    assert not any(r['exceeds_empirical_threshold'] for r in result['contrasts'])
    with pytest.raises(FileExistsError):
        workflow.analyze(fixed)


def test_corrupted_input_refused_before_output(fixed):
    p=fixed.parent/'a/results.json'
    p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='seal changed'):
        workflow.analyze(fixed)
    assert not (fixed.parent/'output').exists()
