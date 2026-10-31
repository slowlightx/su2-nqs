"""Independent basis and physical-state oracles for optional TeNPy readout."""
import numpy as np
import pytest
from su2_nqs.benchmarks.tensor_network_readout import physical_spin_block_readout,tenpy_region_density,tenpy_graph_model
from su2_nqs.benchmarks.tensor_network_observables import regional_physical_observables,tenpy_spin_correlations


def test_spin_weights_of_broken_symmetry_state_do_not_create_singlet_entropy():
    up=np.array([0.,1.]); v=np.kron(up,up); rho=np.outer(v,v)
    r=physical_spin_block_readout(rho)
    assert r['p_q']['2']==pytest.approx(1.)
    assert r['spin_squared_mean']==pytest.approx(2.)
    assert not r['strict_su2_eligible'] and 'invariant_readout' not in r
    singlet=np.array([0.,-1.,1.,0.])/np.sqrt(2)
    r=physical_spin_block_readout(np.outer(singlet,singlet))
    assert r['strict_su2_eligible'] and r['p_q']['0']==pytest.approx(1.)
    with pytest.raises(ValueError):
        physical_spin_block_readout(np.eye(4))


def test_tenpy_mpo_matches_independent_bit_hamiltonian_and_mps_readout():
    pytest.importorskip('tenpy')
    import tenpy.linalg.np_conserved as npc
    from tenpy.networks.mps import MPS
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from su2_nqs.benchmarks.su2_entanglement_spectrum import reduced_density_matrix
    n=6; bonds=triangular_j1j2_bonds(2,3,j2=.125,periodic=True)
    model=tenpy_graph_model(n,bonds,conserve=None)
    sites=model.lat.mps_sites()
    # Exact columns from contracting the MPO, not an eigensolve or DMRG.
    tensors=[model.H_MPO.get_W(i).to_ndarray() for i in range(n)]
    operator=tensors[0][model.H_MPO.get_IdL(0)]
    for tensor in tensors[1:]:
        operator=np.einsum('aij,abkl->bikjl',operator,tensor).reshape(tensor.shape[1],operator.shape[1]*2,operator.shape[2]*2)
    operator=operator[model.H_MPO.get_IdR(n-1)]
    # Physical axis may be up/down for conserve=None: convert using Sz.
    perm=np.argsort(np.diag(sites[0].Sz.to_ndarray()).real)
    basis=np.array([[(x>>(n-1-i))&1 for i in range(n)] for x in range(2**n)])
    native=np.array([sum(int(perm[b])<<(n-1-i) for i,b in enumerate(row)) for row in basis])
    operator=operator[np.ix_(native,native)]
    expected=np.zeros_like(operator)
    for old,bits in enumerate(basis):
        for i,j,w in bonds:
            expected[old,old]+=w*(bits[i]-.5)*(bits[j]-.5)
            if bits[i]!=bits[j]:
                expected[old^((1<<(n-1-i))|(1<<(n-1-j))),old]+=w/2
    np.testing.assert_allclose(operator,expected,atol=3e-14)
    rng=np.random.default_rng(36); v=rng.normal(size=64)+1j*rng.normal(size=64); v/=np.linalg.norm(v)
    native_vector=np.zeros_like(v); native_vector[native]=v
    labels=[f'p{i}' for i in range(n)]
    theta=npc.Array.from_ndarray_trivial(native_vector.reshape([2]*n),labels=labels)
    psi=MPS.from_full(sites,theta,cutoff=1e-15)
    for region in [(0,2,4),(4,0,2),(1,2,3,4)]:
        np.testing.assert_allclose(tenpy_region_density(psi,region),reduced_density_matrix(v,basis,region),atol=2e-14)
    assert model.H_MPO.expectation_value(psi).real==pytest.approx(np.vdot(v,expected@v).real,abs=3e-14)
    assert model.H_MPO.variance(psi)==pytest.approx(np.linalg.norm(expected@v)**2-abs(np.vdot(v,expected@v))**2,abs=5e-14)
    info,c=tenpy_spin_correlations(psi,2,3)
    exact_spin=3*n/4
    for i,j in __import__('itertools').combinations(range(n),2):
        op=np.diag((basis[:,i]-.5)*(basis[:,j]-.5)).astype(complex)
        for old,row in enumerate(basis):
            if row[i]!=row[j]:
                op[old^((1<<(n-1-i))|(1<<(n-1-j))),old]=.5
        assert c[i,j]==pytest.approx(np.vdot(v,op@v).real,abs=3e-14)
        exact_spin+=2*np.vdot(v,op@v).real
    assert info['spin_squared']==pytest.approx(exact_spin,abs=4e-14)


def test_physical_joint_weights_do_not_need_a_spurious_symmetry_repair():
    v=np.eye(16)[15]; rho=np.outer(v,v)
    result=regional_physical_observables(rho,cyclic=True)
    np.testing.assert_allclose(result['cyclic']['p_q_k']['4'],[1,0,0,0],atol=1e-15)
    assert not result['cyclic']['symmetry_spectrum_eligible']
    assert result['spin_squared_pair_closure_error']<2e-14
    for item in result['spin_pairs']:
        assert item['value']==pytest.approx(.25)
    for item in result['disjoint_connected_dimers']:
        assert abs(item['value'])<1e-14
    from su2_nqs.benchmarks.regional_su2_readout import su2_cyclic_entanglement_readout
    exact=su2_cyclic_entanglement_readout(np.eye(16)/16,(1,2,3,0),order=4)
    measured=regional_physical_observables(np.eye(16)/16,cyclic=True)['cyclic']
    for q,parts in exact['sectors'].items():
        np.testing.assert_allclose(measured['p_q_k'][q],[parts[str(k)]['p'] for k in range(4)],atol=3e-15)
    assert measured['probability_closure_error']<1e-14


def test_tenpy_fixed_actual_sweep_budget_uses_strict_stop_convention():
    pytest.importorskip('tenpy')
    from tenpy.algorithms.dmrg import TwoSiteDMRGEngine
    from tenpy.networks.mps import MPS
    model=tenpy_graph_model(4,[(0,1,1.),(1,2,1.),(2,3,1.)])
    psi=MPS.from_product_state(model.lat.mps_sites(),['up','down','up','down'],bc='finite')
    # TeNPy 1.1 tests sweeps > max_sweeps; one actual sweep needs max=0.
    engine=TwoSiteDMRGEngine(psi,model,dict(N_sweeps_check=1,min_sweeps=0,max_sweeps=0,
        mixer=False,trunc_params=dict(chi_max=4),max_trunc_err=None))
    energy,_=engine.run()
    assert engine.sweeps==1
    assert float(energy)==pytest.approx(float(model.H_MPO.expectation_value(psi).real),abs=3e-14)


def test_global_symmetrized_dimers_keep_overlap_order_and_match_dense_oracle():
    pytest.importorskip('tenpy')
    import tenpy.linalg.np_conserved as npc
    from tenpy.networks.mps import MPS
    from su2_nqs.benchmarks.tensor_network_observables import tenpy_dimer_correlations
    model=tenpy_graph_model(4,[(0,1,1.)],conserve=None)
    sites=model.lat.mps_sites();rng=np.random.default_rng(2026101071)
    v=rng.normal(size=16)+1j*rng.normal(size=16);v/=np.linalg.norm(v)
    bits=(np.arange(16)[:,None]>>np.arange(3,-1,-1))&1
    perm=np.argsort(np.diag(sites[0].Sz.to_ndarray()).real)
    native=perm[bits]@2**np.arange(3,-1,-1)
    native_vector=np.empty_like(v);native_vector[native]=v
    theta=npc.Array.from_ndarray_trivial(native_vector.reshape([2]*4),labels=[f'p{i}' for i in range(4)])
    psi=MPS.from_full(sites,theta,cutoff=1e-15)
    bonds=[(0,1),(1,2),(2,3),(3,0)]
    means,connected=tenpy_dimer_correlations(psi,bonds)
    operators=[]
    for i,j in bonds:
        op=np.diag((bits[:,i]-.5)*(bits[:,j]-.5)).astype(complex)
        for old,row in enumerate(bits):
            if row[i]!=row[j]:op[old^((1<<(3-i))|(1<<(3-j))),old]=.5
        operators.append(op)
    exact=np.array([np.vdot(v,op@v) for op in operators])
    np.testing.assert_allclose(means,exact,atol=4e-14,rtol=0)
    for a,A in enumerate(operators):
        for b,B in enumerate(operators):
            expected=np.vdot(v,((A@B+B@A)/2)@v)-exact[a]*exact[b]
            assert connected[a,b]==pytest.approx(expected,abs=4e-14)


def test_full_environments_measure_stale_canonical_forms_without_parent_repair():
    """A local nonunitary filter deliberately invalidates stored Schmidt data."""
    pytest.importorskip('tenpy')
    import tenpy.linalg.np_conserved as npc
    from tenpy.networks.mps import MPS
    from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix
    from su2_nqs.benchmarks.tensor_network_observables import tenpy_dimer_correlations
    model=tenpy_graph_model(4,[(0,1,1.)],conserve=None)
    sites=model.lat.mps_sites(); rng=np.random.default_rng(2026101078)
    v=rng.normal(size=16)+1j*rng.normal(size=16);v/=np.linalg.norm(v)
    bits=(np.arange(16)[:,None]>>np.arange(3,-1,-1))&1
    perm=np.argsort(np.diag(sites[0].Sz.to_ndarray()).real)
    native=perm[bits]@2**np.arange(3,-1,-1)
    nv=np.empty_like(v);nv[native]=v
    theta=npc.Array.from_ndarray_trivial(nv.reshape([2]*4),labels=[f'p{i}' for i in range(4)])
    psi=MPS.from_full(sites,theta,cutoff=1e-15)
    factor=np.array([.4,1.7])
    op=npc.Array.from_ndarray_trivial(np.diag(factor),labels=['p','p*'])
    changed=npc.tensordot(op,psi.get_B(1,form='B'),axes=['p*','p'])
    changed.itranspose(['vL','p','vR']);psi.set_B(1,changed,form='B')
    # Expected physical vector, normalized only for the independent oracle.
    expected=v*factor[perm[bits[:,1]]];expected/=np.linalg.norm(expected)
    before=[b.to_ndarray().copy() for b in psi._B]
    info,c=tenpy_spin_correlations(psi,2,2)
    for i in range(4):
        for j in range(i+1,4):
            assert c[i,j]==pytest.approx(np.vdot(expected,spin_dot_matrix(4,i,j)@expected).real,abs=5e-14)
    bonds=[(0,1),(1,2),(2,3)]
    means,connected=tenpy_dimer_correlations(psi,bonds)
    operators=[spin_dot_matrix(4,*b) for b in bonds]
    exact=np.array([np.vdot(expected,a@expected).real for a in operators])
    np.testing.assert_allclose(means,exact,atol=5e-14,rtol=0)
    for a,A in enumerate(operators):
        for b,B in enumerate(operators):
            assert connected[a,b]==pytest.approx(np.vdot(expected,((A@B+B@A)/2)@expected).real-exact[a]*exact[b],abs=5e-14)
    for a,b in zip(before,psi._B):np.testing.assert_array_equal(a,b.to_ndarray())
    assert info['diagonal_spin_error']<5e-14
