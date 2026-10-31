"""Physical covariance with support leakage and noncanonical complex parents."""
import numpy as np
import pytest

from su2_nqs.benchmarks.mps_rotation_covariance import (
    finite_mps_prefix_rotation_covariance, gram_leading_spin_content, gram_rotation_covariance,
)


def _raising(n):
    operator=np.zeros((2**n,2**n),dtype=complex)
    for packed in range(2**n):
        for site in range(n):
            mask=1<<(n-1-site)
            if packed & mask==0:
                operator[packed | mask,packed]+=1
    return operator


def test_Gram_covariance_includes_action_outside_Schmidt_support():
    rng=np.random.default_rng(20261014)
    charges=np.array([-1,1,1])
    physical=np.array([2*x.bit_count()-3 for x in range(8)])
    a=np.zeros((8,3),dtype=complex)
    b=np.zeros((8,3),dtype=complex)
    for i,m in enumerate(charges):
        select=physical==m
        a[select,i]=rng.normal(size=select.sum())+1j*rng.normal(size=select.sum())
        b[physical==-m,i]=rng.normal(size=select.sum())+1j*rng.normal(size=select.sum())
    plus=_raising(3)
    t=plus @ plus.conj().T+plus.conj().T @ plus
    report=gram_rotation_covariance(a.conj().T @ a,(b.conj().T @ b).T,
        a.conj().T @ plus @ a,a.conj().T @ t @ a,charges,n_region=3)
    wavefunction=a @ b.T
    rho=wavefunction @ wavefunction.conj().T
    rho/=np.trace(rho)
    commutator=rho @ plus-plus @ rho
    np.testing.assert_allclose(report['raising_commutator_hilbert_schmidt_squared'],
        np.linalg.norm(commutator)**2,atol=8e-15,rtol=0)
    assert report['operator_outside_schmidt_support_included']
    assert report['raising_commutator_hilbert_schmidt_squared']>0.5


def _small_mps(filtered=False,gauged=False):
    pytest.importorskip('tenpy')
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite
    rng=np.random.default_rng(206)
    n=8
    state=rng.normal(size=256)+1j*rng.normal(size=256)
    labels=np.array([2*x.bit_count()-n for x in range(256)])
    state[labels!=0]=0
    state/=np.linalg.norm(state)
    site=SpinHalfSite(conserve='Sz')
    tensor=npc.Array.from_ndarray(state.reshape([2]*n),[site.leg]*n,qtotal=[0],labels=[f'p{i}' for i in range(n)])
    psi=MPS.from_full([site]*n,tensor,bc='finite',unit_cell_width=n)
    if filtered:
        for position,filter_values in [(1,[.2,1.4]),(6,[2.,.1])]:
            values=np.asarray(filter_values)
            psi._B[position]=psi._B[position].scale_axis(values,'p')
            state*=values[(np.arange(256)>>(7-position))&1]
    if gauged:
        psi.gauge_total_charge(np.array([[2],[1],[-3],[0],[5],[-4],[1],[3]]))
    state/=np.linalg.norm(state)
    return psi,state


@pytest.mark.parametrize('filtered,gauged',[(False,False),(True,False),(False,True),(True,True)])
def test_actual_prefix_covariance_matches_dense_and_preserves_parent(filtered,gauged):
    psi,state=_small_mps(filtered,gauged)
    before=[b.to_ndarray().copy() for b in psi._B]
    before_s=[s.copy() for s in psi._S]
    for cut in (3,4,5):
        matrix=state.reshape(2**cut,-1)
        rho=matrix @ matrix.conj().T
        plus=_raising(cut)
        commutator=rho @ plus-plus @ rho
        report=finite_mps_prefix_rotation_covariance(psi,cut)
        np.testing.assert_allclose(report['raising_commutator_hilbert_schmidt_squared'],
                                  np.linalg.norm(commutator)**2,atol=8e-15,rtol=0)
        np.testing.assert_allclose(report['purity'],np.trace(rho @ rho).real,atol=8e-15,rtol=0)
        assert report['physical_lowering_adjoint_relative_error']<4e-15
        if cut==4:
            physical_m=np.array([2*x.bit_count()-cut for x in range(2**cut)])
            sz=np.diag(physical_m/2)
            casimir=sz @ sz+.5*(plus @ plus.conj().T+plus.conj().T @ plus)
            for m,sector in report['leading_spin_content']['sectors'].items():
                indices=np.flatnonzero(physical_m==int(m))
                weights,vectors=np.linalg.eigh(rho[np.ix_(indices,indices)])
                weights,vectors=weights[::-1],vectors[:,::-1]
                c=casimir[np.ix_(indices,indices)]
                expectations=np.einsum('ij,ij->j',vectors.conj(),c @ vectors).real
                for observed,value,expectation in zip(sector['leading_levels'],weights,expectations):
                    np.testing.assert_allclose(observed['eigenvalue'],value,atol=8e-15,rtol=0)
                    # An inverse Gram and a tiny Schmidt weight amplify
                    # eigenvector roundoff; the physical weighted moment
                    # remains a much better-conditioned independent oracle.
                    np.testing.assert_allclose(observed['physical_casimir_expectation'],expectation,atol=1e-10,rtol=0)
                    np.testing.assert_allclose(observed['eigenvalue']*observed['physical_casimir_expectation'],
                                              value*expectation,atol=8e-15,rtol=0)
    for old,b in zip(before,psi._B):
        np.testing.assert_array_equal(old,b.to_ndarray())
    for old,s in zip(before_s,psi._S):
        np.testing.assert_array_equal(old,s)


@pytest.mark.parametrize('kind',['three_crossing_dimers','ordered_singlet'])
def test_strict_singlet_controls_have_true_rotation_covariance(kind):
    pytest.importorskip('tenpy')
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite
    from su2_nqs.benchmarks.spin_rotation_calibration import lieb_mattis_singlet,singlet_covering_state
    state=(singlet_covering_state(6,[(0,3),(1,4),(2,5)])
           if kind=='three_crossing_dimers' else lieb_mattis_singlet(6,[0,1,2]))
    site=SpinHalfSite(conserve='Sz')
    tensor=npc.Array.from_ndarray(state.reshape([2]*6),[site.leg]*6,qtotal=[0],labels=[f'p{i}' for i in range(6)])
    psi=MPS.from_full([site]*6,tensor,bc='finite',unit_cell_width=6)
    report=finite_mps_prefix_rotation_covariance(psi,3)
    assert abs(report['raising_commutator_hilbert_schmidt_squared'])<3e-15
    assert abs(report['purity_normalized_commutator_squared'])<8e-15


def test_explicit_Gram_support_cutoff_cannot_silently_drop_physical_weight():
    with pytest.raises(ValueError,match='omitted excessive'):
        gram_leading_spin_content(np.diag([1.,1e-20]),np.diag([0.,1e20]),
                                 np.eye(2),np.zeros(2,dtype=int),n_region=2)


def test_bounded_rotation_worker_seals_parent_and_compares_dense_Casimir(tmp_path):
    psi,state=_small_mps(True,True)
    import hashlib
    import json
    import h5py
    import tenpy
    from tenpy.tools import hdf5_io
    from projects.clebsch_tree_nqs.workflows.mps_rotation_covariance_probe import run_probe
    checkpoint=tmp_path/'checkpoint.h5'
    with h5py.File(checkpoint,'w') as handle:
        hdf5_io.save_to_hdf5(handle,dict(psi=psi,tenpy_version=tenpy.__version__))
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,
        n_sites=8,chi_upper_bound=16,cut=4,threads=1,wall_seconds=30,memory_gib=2)))
    result=run_probe(config,tmp_path/'output')
    assert result['status']=='complete'
    worker=result['readout']
    assert worker['parent_fingerprint_before']==worker['parent_fingerprint_after']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    matrix=state.reshape(16,16)
    rho=matrix @ matrix.conj().T
    raising=_raising(4)
    expected=np.linalg.norm(rho @ raising-raising @ rho)**2
    np.testing.assert_allclose(worker['readout']['raising_commutator_hilbert_schmidt_squared'],
                              expected,atol=8e-15,rtol=0)
    assert worker['readout']['leading_spin_content']['support_cutoff_explicit']
    with pytest.raises(FileExistsError):
        run_probe(config,tmp_path/'output')


@pytest.mark.parametrize('n_region,labels,charges', [
    (True, [0], (0,)), (2.0, [0], (0,)), (2, [1], (0,)),
    (2, [4], (0,)), (2, [0], (0.0,)), (2, [0], (True,)),
])
def test_leading_spin_content_rejects_invalid_doubled_labels(n_region, labels, charges):
    from su2_nqs.benchmarks.mps_rotation_covariance import gram_leading_spin_content
    with pytest.raises(ValueError):
        gram_leading_spin_content(np.eye(1), np.eye(1), np.zeros((1, 1)),
            np.asarray(labels), n_region=n_region, measured_charges=charges)


@pytest.mark.parametrize('filtered,gauged', [(False, False), (True, True)])
def test_spatial_Schmidt_spin_profiles_match_dense_physical_vectors(filtered, gauged):
    psi, state = _small_mps(filtered, gauged)
    groups = {'left_pair': (0, 1), 'right_pair': (2, 3)}
    result = finite_mps_prefix_rotation_covariance(psi, 4, spin_groups=groups,
        leading_levels=8, measured_charges=(0, 2, 4))
    matrix = state.reshape(16, 16)
    rho = matrix @ matrix.conj().T
    bits = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1)
    charges = 2*bits.sum(axis=1)-4
    diagonal = {name: (bits[:, sites]-.5).sum(axis=1) for name, sites in groups.items()}
    for m, sector in result['leading_spin_content']['sectors'].items():
        ids = np.flatnonzero(charges == int(m))
        weights, vectors = np.linalg.eigh(rho[np.ix_(ids, ids)])
        vectors = vectors[:, ::-1]
        for name, diag in diagonal.items():
            expected = np.sum(abs(vectors)**2 * diag[ids, None], axis=0)
            observed = [x['physical_observable_expectations'][name] for x in sector['leading_levels']]
            np.testing.assert_allclose(observed, expected[:len(observed)], atol=1e-10, rtol=0)
        for level in sector['leading_levels']:
            assert sum(level['physical_observable_expectations'].values()) == pytest.approx(int(m)/2, abs=1e-10)
            assert level['observable_imaginary_max'] < 1e-12
    assert result['groups_partition_prefix']


def test_invalid_spatial_groups_are_rejected_without_contraction():
    psi, _ = _small_mps()
    for groups in ({'a': (0, 1), 'b': (1, 2)}, {'a': (True,)},
                   {'a': (4,)}, {'': (0,)}, {'a': ()}):
        with pytest.raises(ValueError):
            finite_mps_prefix_rotation_covariance(psi, 4, spin_groups=groups)


def test_fixed_spin_polarization_interpretation_matches_independent_CG_states():
    from su2_nqs.su2.clebsch_gordan import clebsch_gordan
    from su2_nqs.su2.quantum_numbers import coupling_range
    from su2_nqs.benchmarks.mps_rotation_covariance import fixed_spin_casimir_difference
    for q_left in range(7):
        for q_right in range(7):
            for q in coupling_range(q_left, q_right):
                if q == 0:
                    continue
                for m in (q, -q):
                    left = 0.
                    for ml in range(-q_left, q_left+1, 2):
                        mr = m-ml
                        if abs(mr) <= q_right and (q_right-mr)%2 == 0:
                            left += ml/2*clebsch_gordan(q_left,ml,q_right,mr,q,m)**2
                    expected = (q_left*(q_left+2)-q_right*(q_right+2))/4
                    assert fixed_spin_casimir_difference(q,m,left) == pytest.approx(expected, abs=3e-13)
    for q,m in ((True,1),(2,0),(2,1),(1,2),(2.0,2)):
        with pytest.raises(ValueError):
            fixed_spin_casimir_difference(q,m,.5)
