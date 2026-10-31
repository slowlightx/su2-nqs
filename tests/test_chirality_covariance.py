"""Dense complex-state oracles, including intermediate amplitude nodes."""
import numpy as np
import pytest

from su2_nqs.benchmarks.chirality_covariance import (
    chirality_product_connections, symmetric_matrix, normalized_coordinates,
    cluster_jackknife, compare_chirality_matrices,
)
from su2_nqs.benchmarks.replica_spin_response import operator_action_over_sample


def test_frozen_reader_selects_actual_sealed_energy_audit_without_path_escape():
    from pathlib import Path
    from projects.clebsch_tree_nqs.workflows.spin_chirality_covariance import frozen_energy_audit_path
    settings = dict(parent='/trusted/frozen-parent', step=580)
    assert frozen_energy_audit_path(settings) == Path('/trusted/frozen-parent/audits/step00580/results.json')
    settings['energy_audit_filename'] = 'energy.json'
    assert frozen_energy_audit_path(settings) == Path('/trusted/frozen-parent/audits/step00580/energy.json')
    for filename in ('../energy.json', '/tmp/energy.json', 'training.json', None):
        with pytest.raises(ValueError, match='actual sealed'):
            frozen_energy_audit_path(dict(settings, energy_audit_filename=filename))


def test_energy_only_and_full_readout_audits_require_identical_frozen_state_and_streams():
    from copy import deepcopy
    from projects.clebsch_tree_nqs.workflows.spin_chirality_covariance import validate_frozen_energy_audit
    reports = [dict(seed=s, raw_sha256=f'raw{s}', params_sha256='parameters',
        registration_sha256='registration', records=128, full_H_calls=128) for s in (101, 102)]
    settings = dict(step=580, checkpoint_sha256='checkpoint',
        streams={f'stream{s}.npz': f'raw{s}' for s in (101, 102)})
    registration = dict(result_sha256='registration')
    audit = dict(step=580, registration_sha256='registration', checkpoint_sha256='checkpoint',
        params_sha256='parameters', audit_projection='gamma_a1', streams=reports, full_H_calls=256)
    for filename in ('results.json', 'energy.json'):
        settings['energy_audit_filename'] = filename
        validate_frozen_energy_audit(settings, registration, audit)
    mutations = [lambda a: a.update(registration_sha256='other'),
        lambda a: a.update(step=500), lambda a: a.update(audit_projection='none'),
        lambda a: a.update(checkpoint_sha256='other'), lambda a: a.update(full_H_calls=255),
        lambda a: a['streams'][0].update(params_sha256='other'),
        lambda a: a['streams'][0].update(raw_sha256='other'),
        lambda a: a['streams'][0].update(registration_sha256='other'),
        lambda a: a['streams'][0].update(records=127)]
    for mutate in mutations:
        bad = deepcopy(audit); mutate(bad)
        with pytest.raises(ValueError, match='identity changed'):
            validate_frozen_energy_audit(settings, registration, bad)


@pytest.mark.parametrize('filename', ['results.json', 'energy.json'])
def test_replica_stage_uses_explicit_real_audit_and_rejects_wrong_parent_before_queries(tmp_path, monkeypatch, filename):
    import json
    from projects.clebsch_tree_nqs.workflows import spin_replica_response as reader
    parent = tmp_path/'parent'
    settings = dict(parent=str(parent), output=str(tmp_path/'new-output'), step=580,
        checkpoint_sha256='checkpoint', energy_audit_filename=filename, streams={})
    request = tmp_path/'request.json'; request.write_text(json.dumps(settings))
    paths = []
    def sealed(path):
        paths.append(path)
        if path.name == 'registration.json':
            return dict(config={}, result_sha256='correct-parent')
        return dict(streams=[], registration_sha256='wrong-parent')
    monkeypatch.setattr(reader, 'read_sealed', sealed)
    monkeypatch.setattr(reader, 'validate_spec', lambda _: object())
    monkeypatch.setattr(reader, 'file_sha256', lambda _: 'checkpoint')
    monkeypatch.setattr(reader.jax, 'default_backend', lambda: 'gpu')
    with pytest.raises(ValueError, match='identity changed'):
        reader.run(request)
    assert paths == [parent/'registration.json', parent/'audits/step00580'/filename]
    assert not (tmp_path/'new-output').exists()


def dense_kappa(n, edges):
    sx = np.array([[0, 1], [1, 0]], complex)/2
    sy = np.array([[0, 1j], [-1j, 0]], complex)/2
    def embed(a, site):
        result = np.ones((1, 1), complex)
        for i in range(n):result = np.kron(result, a if i == site else np.eye(2))
        return result
    return sum(embed(sx, a)@embed(sy, b)-embed(sy, a)@embed(sx, b) for a,b in edges)


@pytest.mark.parametrize('left,right', [(((0,1),),((0,1),)),
    (((0,1),(1,2)),((1,2),(2,3))), (((0,1),),((2,3),)),
    (((0,1),(1,0)),((0,1),))])
def test_composed_rows_equal_independent_dense_operators(left,right):
    n=4;bits=((np.arange(16)[:,None] >> np.arange(3,-1,-1))&1).astype(np.int8)
    target,source,c=chirality_product_connections(bits,left,right)
    matrix=np.zeros((16,16),complex)
    np.add.at(matrix,(source,target@(1<<np.arange(3,-1,-1))),c)
    a,b=dense_kappa(n,left),dense_kappa(n,right)
    np.testing.assert_allclose(matrix,(a@b+b@a)/2,atol=1e-15,rtol=0)
    assert len(set(map(tuple,np.column_stack((source,target)))))==len(source)


def test_intermediate_nodes_are_kept_and_complex_Born_average_is_exact():
    bits=((np.arange(16)[:,None] >> np.arange(3,-1,-1))&1).astype(np.int8)
    rng=np.random.default_rng(8472);psi=rng.normal(size=16)+1j*rng.normal(size=16)
    psi[[0,2,3,5,8,11]]=0;psi/=np.linalg.norm(psi)
    keep=np.flatnonzero(psi);a,b=((0,1),),((1,2),(2,3))
    target,source,c=chirality_product_connections(bits[keep],a,b)
    packed=target@(1<<np.arange(3,-1,-1))
    logs=np.full(len(packed),-np.inf+0j);nonzero=psi[packed]!=0
    logs[nonzero]=np.log(psi[packed[nonzero]])
    local=operator_action_over_sample(logs,source,c,np.log(psi[keep]))
    ka,kb=dense_kappa(4,a),dense_kappa(4,b)
    expected=np.vdot(psi,(ka@kb+kb@ka)@psi)/2
    np.testing.assert_allclose(abs(psi[keep])**2@local,expected,atol=4e-15,rtol=0)
    # Sharp nodal counterexample: |down,up> has <K^2>=1/4, but its
    # single-action ratio vanishes on the only Born-supported configuration.
    t,s,c=chirality_product_connections(np.array([[0,1]]),((0,1),),((0,1),))
    np.testing.assert_array_equal(t,[[0,1]]);np.testing.assert_allclose(c,[.25])


def test_coordinates_and_paired_jackknife_preserve_covariance():
    matrix=symmetric_matrix(np.array([2.,2.,2.,.2,.1,.2]))
    assert normalized_coordinates(matrix)[0]==pytest.approx(1/3)
    labels=np.repeat(np.arange(8),3)
    records=np.tile([2.,2.,2.,.2,.1,.2],(24,1))
    mean,leave=cluster_jackknife(records,labels)
    np.testing.assert_allclose(mean,records[0]);np.testing.assert_allclose(leave,np.tile(mean,(8,1)))
    probes=records.copy();probes[:,3:]=2*records[:,3:]+records[:,[0,0,1]]+records[:,[1,2,2]]
    result=compare_chirality_matrices(records,probes,np.ones(24),labels)
    np.testing.assert_allclose(result['replica_minus_ordinary_coordinates'],0,atol=1e-15)
    np.testing.assert_allclose(result['replica_minus_ordinary_coordinates_SE'],0,atol=1e-15)
    with pytest.raises(ValueError):normalized_coordinates(-np.eye(3))
    with pytest.raises(ValueError):cluster_jackknife(records,labels[:2])


def test_a_new_replica_matrix_is_not_itself_an_entanglement_witness():
    # Both states are explicit mixtures of product configurations, fixed Sz,
    # invariant under global bit flip and exchange of the two boundary pairs.
    # They have the SAME C but different normalized M. Thus extra information
    # relative to C is not automatically quantum entanglement or a QSL signal.
    import itertools
    encoded=np.array(list(itertools.product((0,1),repeat=3)))
    bits=np.stack((encoded,1-encoded),axis=-1).reshape(8,6)
    packed=bits@(1<<np.arange(5,-1,-1))
    weights=[np.full(8,1/8), (encoded[:,0]==encoded[:,2]).astype(float)/4]
    operators=[dense_kappa(6,((i,i+1),)) for i in (0,2,4)]
    ordinary=[];replica=[]
    for w in weights:
        rho=np.zeros((64,64));rho[packed,packed]=w
        ordinary.append(np.array([[np.trace(rho@a@b).real for b in operators] for a in operators]))
        replica.append(np.array([[np.trace(rho@a@rho@b).real/np.trace(rho@rho)
                                  for b in operators] for a in operators]))
    np.testing.assert_allclose(ordinary[0],ordinary[1],atol=1e-15,rtol=0)
    assert normalized_coordinates(replica[0])[0]==pytest.approx(1/3)
    assert normalized_coordinates(replica[1])[0]==pytest.approx(1)


def test_Renyi_matched_control_removes_cooling_and_equals_skew_information():
    # A theoretical control, NOT a newly measured36-site observable.
    rng=np.random.default_rng(9851)
    a=rng.normal(size=(16,16))+1j*rng.normal(size=(16,16))
    rho=a@a.conj().T;rho/=np.trace(rho)
    operators=[dense_kappa(4,edges) for edges in (((0,1),),((2,3),),((1,2),))]
    purity=np.trace(rho@rho).real
    m=np.array([[np.trace(rho@x@rho@y).real/purity for y in operators]for x in operators])
    c2=np.array([[np.trace(rho@rho@(x@y+y@x)).real/(2*purity)
                  for y in operators]for x in operators])
    commutators=[rho@x-x@rho for x in operators]
    independent=np.array([[np.trace(x.conj().T@y).real/(2*purity)
                           for y in commutators]for x in commutators])
    np.testing.assert_allclose(c2-m,independent,atol=2e-16,rtol=0)
    assert np.linalg.eigvalsh(c2-m)[0]>0


def test_SU2_invariant_separable_controls_also_have_equal_C_and_different_M():
    # Global SU2 twirling is a mixture of product rotations and therefore
    # preserves separability. It constructs a SYNTHETIC control; no actual
    # neural/DMRG RDM is repaired, replaced or relabelled as invariant.
    import itertools
    from su2_nqs.su2.triangle_blocks import triangle_block_tables
    encoded=np.array(list(itertools.product((0,1),repeat=3)))
    bits=np.stack((encoded,1-encoded),axis=-1).reshape(8,6)
    packed=bits@(1<<np.arange(5,-1,-1))
    configurations=(np.arange(64)[:,None]>>np.arange(5,-1,-1))&1
    magnetic=(2*configurations-1).sum(1)
    raising=np.zeros((64,64))
    for i in range(6):
        ids=np.flatnonzero(configurations[:,i]==0)
        raising[ids^(1<<(5-i)),ids]+=1
    weights=[np.full(8,1/8),(encoded[:,0]==encoded[:,2]).astype(float)/4]
    operators=[dense_kappa(6,((i,i+1),)) for i in (0,2,4)]
    ordinary=[];replica=[]
    for w in weights:
        rho=np.zeros((64,64));rho[packed,packed]=w
        averaged=np.zeros_like(rho)
        for q,u in triangle_block_tables()['six'].items():
            bases=[u*(magnetic==m)[:,None] for m in range(-q,q+1,2)]
            sigma=sum(v.T@rho@v for v in bases)/(q+1)
            averaged+=sum(v@sigma@v.T for v in bases)
        np.testing.assert_allclose(raising@averaged,averaged@raising,atol=3e-16,rtol=0)
        assert np.linalg.eigvalsh(averaged)[0]>-1e-15
        ordinary.append(np.array([[np.trace(averaged@a@b).real for b in operators]for a in operators]))
        replica.append(np.array([[np.trace(averaged@a@averaged@b).real/np.trace(averaged@averaged)
                                  for b in operators]for a in operators]))
    np.testing.assert_allclose(ordinary,np.tile(np.eye(3)/6,(2,1,1)),atol=3e-16,rtol=0)
    np.testing.assert_allclose(replica[0],49/432*np.eye(3),atol=3e-16,rtol=0)
    np.testing.assert_allclose(replica[1],7/256*np.array([[1,0,1],[0,4,0],[1,0,1]]),atol=3e-16,rtol=0)
    assert normalized_coordinates(replica[0])[0]==pytest.approx(1/3)
    assert normalized_coordinates(replica[1])[0]==pytest.approx(2/3)
