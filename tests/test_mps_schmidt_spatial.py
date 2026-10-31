"""Spatial labels use actual physical vectors, including noncanonical MPS."""
import itertools
import numpy as np
import pytest

from su2_nqs.benchmarks.mps_schmidt_spatial import strip_spatial_content


def test_strip_translation_and_mirror_preserve_actual_triangular_J1_J2_graph():
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    mirror=tuple(6*r+(r-c)%6 for r in range(6) for c in range(6))
    translation=tuple(6*r+(c-1)%6 for r in range(6) for c in range(6))
    for j2 in (0.,.125):
        bonds={(a,b):w for a,b,w in triangular_j1j2_bonds(6,6,j2=j2,periodic=True)}
        for mapping in (mirror,translation):
            transformed={tuple(sorted((mapping[a],mapping[b]))):w for (a,b),w in bonds.items()}
            assert transformed==bonds
            for cut in (12,18):assert {mapping[i] for i in range(cut)}==set(range(cut))


def test_spatial_projectors_resolve_complex_vectors_and_real_K_pair():
    n=6;d=2**n;columns=3
    rows=list(itertools.product((0,1),repeat=n));lookup={x:i for i,x in enumerate(rows)}
    t=(2,0,1,5,3,4);r=(0,2,1,4,3,5)
    permutation=lambda gather:np.eye(d)[[lookup[tuple(x[j] for j in gather)] for x in rows]]
    T,R=permutation(t),permutation(r)
    np.testing.assert_array_equal(R@T@R,T.T)
    rng=np.random.default_rng(2627)
    v=np.linalg.qr(rng.normal(size=(d,4))+1j*rng.normal(size=(d,4)))[0]
    result=strip_spatial_content(v,columns)
    expected=np.stack([np.diag(v.conj().T @ sum(np.exp(-2j*np.pi*k*j/3)*np.linalg.matrix_power(T,j)
        for j in range(3))/3 @ v).real for k in range(3)],axis=1)
    np.testing.assert_allclose(result['momentum_weights'],expected,atol=3e-16,rtol=0)
    P=sum(np.linalg.matrix_power(T,j) for j in range(3))/3
    np.testing.assert_allclose(result['gamma_parallel_odd_weights'],np.diag(v.conj().T@P@(np.eye(d)-R)/2@v).real,atol=2e-16,rtol=0)
    k=sum(np.exp(-2j*np.pi*j/3)*np.linalg.matrix_power(T,j) for j in range(3))/3@rng.normal(size=d)
    real=np.linalg.qr(np.column_stack((k.real,k.imag)))[0]
    weights=strip_spatial_content(real,3)['momentum_weights']
    np.testing.assert_allclose(weights,np.array([[0,.5,.5]]*2),atol=4e-16,rtol=0)
    with pytest.raises(ValueError,match='orthonormal'):strip_spatial_content(2*v,3)


@pytest.mark.parametrize('filtered,gauged',[(False,False),(True,True)])
def test_actual_MPS_vector_reconstruction_and_spatial_readout_match_dense(filtered,gauged):
    from test_mps_rotation_covariance import _small_mps
    from su2_nqs.benchmarks.mps_rotation_covariance import finite_mps_prefix_rotation_covariance
    psi,state=_small_mps(filtered,gauged)
    before=[a.to_ndarray().copy() for a in psi._B]
    r=finite_mps_prefix_rotation_covariance(psi,4,leading_levels=4,
        measured_charges=(0,2),spatial_strip_columns=2)
    rho=state.reshape(16,16)@state.reshape(16,16).conj().T
    m=np.array([2*x.bit_count()-4 for x in range(16)])
    for charge in (0,2):
        ids=np.flatnonzero(m==charge);_,v=np.linalg.eigh(rho[np.ix_(ids,ids)])
        count=min(4,len(ids));full=np.zeros((16,count),complex);full[ids]=v[:,::-1][:,:count]
        expected=strip_spatial_content(full,2)
        actual=r['leading_spatial_content'][str(charge)]
        for name in ('momentum_weights','gamma_parallel_even_weights','gamma_parallel_odd_weights'):
            np.testing.assert_allclose(actual[name],expected[name],atol=2e-11,rtol=0)
        assert actual['physical_orthonormality_error']<2e-11
    for old,new in zip(before,psi._B):np.testing.assert_array_equal(old,new.to_ndarray())


def test_extended_spatial_vectors_and_total_ladder_match_dense():
    from test_mps_rotation_covariance import _small_mps, _raising
    from su2_nqs.benchmarks.mps_rotation_covariance import finite_mps_prefix_rotation_covariance
    psi,state=_small_mps(True,True)
    r=finite_mps_prefix_rotation_covariance(psi,4,leading_levels=16,
        measured_charges=(0,2,4),spatial_strip_columns=2,spatial_levels=16,
        total_spin_ladder=True)
    rho=state.reshape(16,16)@state.reshape(16,16).conj().T
    m=np.array([2*x.bit_count()-4 for x in range(16)])
    plus=_raising(4);vectors={}
    for charge in (0,2,4):
        ids=np.flatnonzero(m==charge);_,v=np.linalg.eigh(rho[np.ix_(ids,ids)])
        full=np.zeros((16,len(ids)),complex);full[ids]=v[:,::-1]
        vectors[charge]=full
        assert len(r['leading_spatial_content'][str(charge)]['momentum_weights'])==len(ids)
    for initial in (0,2):
        block=r['leading_spin_content']['raising_transitions'][f'{initial}_to_{initial+2}']['site_matrices']['total_Splus']
        actual=np.array(block['real'])+1j*np.array(block['imag'])
        expected=vectors[initial+2].conj().T@plus@vectors[initial]
        # Arbitrary eigenvector phases must not be compared as amplitudes.
        np.testing.assert_allclose(abs(actual)**2,abs(expected)**2,atol=2e-10,rtol=0)
