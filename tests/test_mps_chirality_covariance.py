"""Physical C/M products retain leakage outside finite Schmidt support."""
import numpy as np
import pytest

from su2_nqs.benchmarks.mps_chirality_covariance import finite_mps_chirality_covariances
from su2_nqs.benchmarks.schmidt_covariance import dense_schmidt_covariances


@pytest.mark.parametrize('filtered,gauged',[(False,False),(True,True)])
@pytest.mark.parametrize('replica_method',['legacy_six_probe','three_direct_charge_zero_Grams'])
@pytest.mark.parametrize('narrow_support',[False,True])
def test_three_complex_chirality_matrices_match_dense_and_preserve_parent(filtered,gauged,replica_method,narrow_support):
    pytest.importorskip('tenpy')
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinHalfSite
    n=8;rng=np.random.default_rng(166001)
    state=rng.normal(size=2**n)+1j*rng.normal(size=2**n)
    packed=np.arange(2**n);state[np.array([k.bit_count() for k in packed])!=4]=0
    if narrow_support:
        state[:]=0
        state[(3<<4)|12]=np.sqrt(.5)
        state[(5<<4)|10]=1j*np.sqrt(.3)
        state[(10<<4)|5]=np.exp(.3j)*np.sqrt(.2)
    state/=np.linalg.norm(state)
    site=SpinHalfSite(conserve='Sz')
    tensor=npc.Array.from_ndarray(state.reshape([2]*n),[site.leg]*n,qtotal=[0],labels=[f'p{i}'for i in range(n)])
    psi=MPS.from_full([site]*n,tensor,bc='finite',unit_cell_width=n)
    if filtered:
        for i,v in ((1,[.2,1.4]),(6,[2.,.1])):
            psi._B[i]=psi._B[i].scale_axis(np.asarray(v),'p')
            state*=np.asarray(v)[(packed>>(n-1-i))&1]
    if gauged:psi.gauge_total_charge(np.array([[2],[1],[-3],[0],[5],[-4],[1],[3]]))
    state/=np.linalg.norm(state)
    before=[b.to_ndarray().copy()for b in psi._B];before_s=[s.copy()for s in psi._S]
    cut=4;edges=(((0,1),(1,2),(2,0)),((1,2),),((2,3),(3,1)))
    cache=[]
    result=finite_mps_chirality_covariances(psi,cut,edges,replica_method=replica_method,include_C2=True,
        progress=lambda value:cache.append(value['readout_cache']))
    sx=np.array([[0,1],[1,0]],complex)/2;sy=np.array([[0,1j],[-1j,0]],complex)/2
    def at(i,op):
        a=np.array([[1.]])
        for k in range(cut):a=np.kron(a,op if k==i else np.eye(2))
        return a
    ops=np.array([sum(at(a,sx)@at(b,sy)-at(a,sy)@at(b,sx)for a,b in group)for group in edges])
    rho=state.reshape(2**cut,-1)@state.reshape(2**cut,-1).conj().T
    oracle=dense_schmidt_covariances(rho,ops)
    np.testing.assert_allclose(result['ordinary_matrix'],oracle['ordinary'],atol=4e-14,rtol=0)
    np.testing.assert_allclose(result['replica_matrix'],oracle['inter_schmidt'],atol=4e-14,rtol=0)
    np.testing.assert_allclose(result['one_point'],oracle['one_point'].real,atol=4e-14,rtol=0)
    np.testing.assert_allclose(result['purity'],oracle['purity'],atol=4e-14,rtol=0)
    np.testing.assert_allclose(result['renyi_matched_matrix'],oracle['renyi_matched'],atol=4e-14,rtol=0)
    np.testing.assert_allclose(result['commutator_metric'],oracle['commutator_metric'],atol=4e-14,rtol=0)
    assert result['C2_new_MPO_queries']==0
    if narrow_support:
        assert max(psi.chi)<=3  # Far smaller than physical-prefix dimension16.
        assert np.trace(result['commutator_metric'])>.01
    # A resumed SAME registration can reuse complete physical scalars without
    # repeating the replica Gram or any MPO expectation. It still verifies
    # the actual parent norm and retains the full product matrix.
    replay=finite_mps_chirality_covariances(psi,cut,edges,replica_method=replica_method,cache=cache[-1],include_C2=True)
    np.testing.assert_array_equal(replay['ordinary_matrix'],result['ordinary_matrix'])
    np.testing.assert_array_equal(replay['replica_matrix'],result['replica_matrix'])
    np.testing.assert_array_equal(replay['renyi_matched_matrix'],result['renyi_matched_matrix'])
    assert replay['mpo_maximum_bond_dimensions']==[]
    for b,a in zip(psi._B,before):np.testing.assert_array_equal(b.to_ndarray(),a)
    for s,a in zip(psi._S,before_s):np.testing.assert_array_equal(s,a)
