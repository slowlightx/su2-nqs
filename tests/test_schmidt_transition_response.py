"""Independent physical oracles for two-copy and Schmidt spin responses."""
import numpy as np
import pytest
import json
from pathlib import Path
from su2_nqs.benchmarks.mps_rotation_covariance import gram_leading_spin_content
from su2_nqs.benchmarks.replica_spin_response import (
    swap_response_records, summarize_swap_response, explicit_half_rdm_cost,
    vector_chirality_z_connections, operator_action_over_sample,
    hermitian_replica_numerator,
)


@pytest.mark.parametrize('fixed_m', [False, True])
def test_node_safe_vector_chirality_replica_against_dense_complex_oracle(fixed_m):
    rng = np.random.default_rng(20261016)
    bits = (np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1
    psi = rng.normal(size=16)+1j*rng.normal(size=16)
    if fixed_m:
        psi[bits.sum(1) != 2] = 0
    psi[6] = 0  # accidental carrier node, not only a forbidden charge
    psi /= np.linalg.norm(psi)
    # Independent Kronecker oracle. Basis order is down, up.
    sp = np.array([[0, 0], [1, 0]], complex)
    sm = sp.T
    k = .5j*(np.kron(sp, sm)-np.kron(sm, sp))
    target, source, coefficient = vector_chirality_z_connections(bits, ((0, 1),))
    dense = np.zeros((16, 16), complex)
    target_id = target @ (1 << np.arange(3, -1, -1))
    dense[source, target_id] = coefficient
    np.testing.assert_array_equal(dense, np.kron(k, np.eye(4)))
    # Opposite orientations cancel; repeated bonds must not duplicate targets.
    t, s, c = vector_chirality_z_connections(bits, ((0, 1), (1, 0)))
    assert len(t) == len(s) == len(c) == 0
    pairs = np.array([(a, b) for a in np.flatnonzero(psi)
                      for b in np.flatnonzero(psi)])
    a, b = pairs.T
    u, v = (b & 12) | (a & 3), (a & 12) | (b & 3)
    actions = []
    with np.errstate(divide='ignore'):
        for rows, old in ((u, a), (v, b)):
            t, s, c = vector_chirality_z_connections(bits[rows], ((0, 1),))
            ti = t @ (1 << np.arange(3, -1, -1))
            actions.append(operator_action_over_sample(np.log(psi[ti]), s, c, np.log(psi[old])))
    records = hermitian_replica_numerator(np.column_stack(actions))
    probability = abs(psi[a]*psi[b])**2
    rho = psi.reshape(4, 4) @ psi.reshape(4, 4).conj().T
    expected = np.trace(rho @ k @ rho @ k)
    np.testing.assert_allclose(probability @ records, expected, atol=2e-16, rtol=0)
    # Incorrect use of a conjugate is detected by the complex oracle.
    assert abs(probability @ (actions[0]*actions[1].conj())-expected) > 1e-4
    assert np.any((psi[u] == 0) & (abs(actions[0]) > 0))


def test_preregistered_cut_control_preserves_operator_not_only_site_count():
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    root=Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/configs/remote/chirality_cut_20261015'
    first=None
    for name in ('tri_j0_750','tri_j0_1000','tri_j0125_100','tri_j0125_200'):
        spec=json.loads((root/(name+'.json')).read_text())
        if first is None:first=spec
        assert spec['regions']==first['regions'] and spec['chirality_channels']==first['chirality_channels']
        loop=spec['chirality_channels']['prefix18']['row1_kappa_z']
        for cut,sites in spec['regions'].items():
            assert spec['chirality_channels'][cut]['row1_kappa_z']==loop
            assert all(a in sites and b in sites for a,b in loop)
    # The bump really changes geometry: it is not merely another translation
    # or rotation of the exact-A1 parent, which would give no new cut test.
    boundary=[]
    for cut in ('prefix18','bump18'):
        sites=set(first['regions'][cut])
        boundary.append(sum((a in sites)!=(b in sites) for a,b,_ in triangular_j1j2_bonds(6,6,j2=0.,periodic=True)))
    assert len(first['regions']['prefix18'])==len(first['regions']['bump18'])==18
    assert boundary[0] != boundary[1]


def test_uniform_vector_chirality_and_single_spin_have_different_mirror_selection_rules():
    # Exact algebra, not a new variational state: on a row-complete strip,
    # column-translation-zero single-spin probes are mirror even, whereas
    # oriented winding-bond chirality is mirror odd. Darkness CAN therefore
    # be symmetry-enforced, and is not by itself a QSL-specific mechanism.
    rows,columns=2,3;n=rows*columns;d=2**n
    bits=(np.arange(d)[:,None] >> np.arange(n-1,-1,-1)) & 1
    codes=1 << np.arange(n-1,-1,-1)
    grid=np.arange(n).reshape(rows,columns)
    def permutation(gather):
        return np.eye(d)[bits[:,gather]@codes]
    translations=sum(permutation(np.roll(grid,k,axis=1).ravel()) for k in range(columns))/columns
    mirror=permutation(np.array([r*columns+(r-c)%columns for r in range(rows) for c in range(columns)]))
    even=(translations+mirror@translations)/2
    odd=(translations-mirror@translations)/2
    for i in range(n):
        sp=np.zeros((d,d));ids=np.flatnonzero(bits[:,i]==0)
        sp[ids+(1<<(n-1-i)),ids]=1
        np.testing.assert_allclose(odd@sp@even,0,atol=1e-16,rtol=0)
    bonds=tuple((r*columns+c,r*columns+(c+1)%columns) for r in range(rows) for c in range(columns))
    target,source,coefficient=vector_chirality_z_connections(bits,bonds)
    kappa=np.zeros((d,d),complex);kappa[source,target@codes]=coefficient
    np.testing.assert_allclose(mirror@kappa@mirror.T,-kappa,atol=1e-16,rtol=0)
    assert np.linalg.norm(odd@kappa@even)>1


def test_charge_block_acceleration_matches_dense_complex_trace_without_canonical_gauge():
    from su2_nqs.benchmarks.mps_rotation_covariance import _charge_zero_replica_trace
    rng=np.random.default_rng(582)
    m=np.array([-2]*3+[0]*5+[2]*4)
    r=np.zeros((12,12),complex);g=np.zeros_like(r)
    for charge in np.unique(m):
        ids=np.flatnonzero(m == charge);n=len(ids)
        a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        r[np.ix_(ids,ids)]=a @ a.conj().T
        g[np.ix_(ids,ids)]=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    expected=np.trace(r @ g.conj().T @ r @ g)
    np.testing.assert_allclose(_charge_zero_replica_trace(r,g,m),expected,atol=1e-10,rtol=1e-14)
    g[0,-1]=1
    with pytest.raises(ValueError,match='forbidden'): _charge_zero_replica_trace(r,g,m)


def test_complex_noncanonical_transition_form_factors_against_physical_vectors():
    rng = np.random.default_rng(20261015)
    n = 4
    m = np.array([2*i.bit_count()-n for i in range(16)])
    a = np.zeros((16,16),complex)
    for charge in np.unique(m):
        ids = np.flatnonzero(m == charge)
        a[np.ix_(ids,ids)] = rng.normal(size=(len(ids),len(ids)))+1j*rng.normal(size=(len(ids),len(ids)))
    b = a.conj().T.copy()
    r = (b.conj().T @ b).T
    l = a.conj().T @ a
    sites = []
    for i in range(n):
        op = np.zeros((16,16),complex)
        for bits in range(16):
            mask = 1 << (3-i)
            if not bits & mask: op[bits | mask,bits] = 1
        sites.append(op)
    plus = sum(sites)
    t = a.conj().T @ (plus @ plus.conj().T+plus.conj().T @ plus) @ a
    result = gram_leading_spin_content(l,r,t,m,n_region=n,levels_per_charge=16,
        raising_transition_grams={str(i): a.conj().T @ op @ a for i,op in enumerate(sites)})
    rho = a @ r @ a.conj().T
    rho /= np.trace(rho)
    ii, jj = np.flatnonzero(m == 0), np.flatnonzero(m == 2)
    _, vi = np.linalg.eigh(rho[np.ix_(ii,ii)])
    _, vf = np.linalg.eigh(rho[np.ix_(jj,jj)])
    phase = np.exp(2j*np.pi*np.arange(n)/3)
    physical = vf[:,::-1].conj().T @ sum(p*o for p,o in zip(phase,sites))[np.ix_(jj,ii)] @ vi[:,::-1]
    matrices = result['raising_transitions']['0_to_2']['site_matrices']
    measured = sum(phase[i]*(np.array(matrices[str(i)]['real'])+1j*np.array(matrices[str(i)]['imag'])) for i in range(n))
    np.testing.assert_allclose(abs(measured)**2, abs(physical)**2, atol=2e-12, rtol=0)


@pytest.mark.parametrize('fixed_m', [False,True])
def test_replica_identity_exact_complex_state_including_forbidden_swaps(fixed_m):
    rng = np.random.default_rng(704)
    psi = rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    if fixed_m:
        mask = np.array([[a.bit_count()+b.bit_count() == 2 for b in range(4)] for a in range(4)])
        psi[~mask] = 0
    psi /= np.linalg.norm(psi)
    rho = psi @ psi.conj().T
    fourier = np.array([-.5-.5j, -.5+.5j, .5-.5j, .5+.5j])
    old, new, values, probabilities = [],[],[],[]
    for a,b in zip(*np.nonzero(psi)):
        for c,d in zip(*np.nonzero(psi)):
            old.append(np.log([psi[a,b],psi[c,d]]))
            with np.errstate(divide='ignore'): new.append(np.log([psi[c,b],psi[a,d]]))
            values.append([[fourier[a]],[fourier[c]]])
            probabilities.append(abs(psi[a,b]*psi[c,d])**2)
    w, numer = swap_response_records(old,new,values)
    np.testing.assert_allclose(np.dot(probabilities,w),np.trace(rho @ rho),atol=3e-16)
    expected = np.sum(abs(rho)**2*fourier.conj()[:,None]*fourier[None,:])
    np.testing.assert_allclose(np.dot(probabilities,numer[:,0]),expected,atol=3e-16)
    if fixed_m: assert np.count_nonzero(w == 0) > 0


def test_cluster_ratio_preserves_covariance_and_reports_exact_cost():
    w = np.arange(1,33)/30
    result = summarize_swap_response(w, w[:,None]*np.array([[2.,3.]]), np.repeat(np.arange(8),4))
    np.testing.assert_allclose(result['response'],[2.,3.])
    assert max(result['response_chain_pair_SE']) < 1e-14
    cost = explicit_half_rdm_cost()
    assert cost['dense_complex128_GiB'] == 1024
    assert cost['full_fixed_magnetization_amplitude_count'] == 9075135300
    assert cost['strict_SU2_multiplicity_entries'] == 477638700
    assert cost['strict_SU2_multiplicity_complex128_GiB'] < 7.2


def test_relative_replica_symmetry_average_keeps_exact_complex_expectation():
    rng=np.random.default_rng(673)
    bits=(np.arange(16)[:,None] >> np.arange(3,-1,-1)) & 1
    psi=rng.normal(size=16)+1j*rng.normal(size=16)
    psi[bits.sum(1) != 2]=0
    transforms=[]
    for shift in range(4):
        y=np.roll(bits,shift,axis=1)
        for flip in (False,True):
            t=1-y if flip else y
            transforms.append(t @ (1 << np.arange(3,-1,-1)))
    psi=sum(psi[t] for t in transforms)/len(transforms)
    psi/=np.linalg.norm(psi)
    original=[];swapped=[];values=[];probabilities=[]
    for a in np.flatnonzero(abs(psi)>0):
        for b in np.flatnonzero(abs(psi)>0):
            for t in transforms:
                z=t[b];u=(a&12)|(z&3);v=(z&12)|(a&3)
                # Region A is the low two bits; x' takes y_A, y' takes x_A.
                original.append(np.log([psi[a],psi[b]]))
                with np.errstate(divide='ignore'):swapped.append(np.log([psi[u],psi[v]]))
                o=lambda p: ((p&1)-.5)+1j*(((p>>1)&1)-.5)
                values.append([[o(a)],[o(z)]])
                probabilities.append(abs(psi[a]*psi[b])**2/len(transforms))
    w,n=swap_response_records(original,swapped,values)
    rho=psi.reshape(4,4).conj().T @ psi.reshape(4,4)
    o=np.array([-.5-.5j,.5-.5j,-.5+.5j,.5+.5j])
    np.testing.assert_allclose(np.dot(probabilities,w),np.trace(rho @ rho),atol=4e-16)
    np.testing.assert_allclose(np.dot(probabilities,n[:,0]),np.sum(abs(rho)**2*o.conj()[:,None]*o[None,:]),atol=4e-16)


@pytest.mark.parametrize('filtered,gauged', [(False,False),(True,True)])
def test_tenpy_site_transition_and_full_two_copy_response(filtered,gauged):
    from test_mps_rotation_covariance import _small_mps
    from su2_nqs.benchmarks.mps_rotation_covariance import finite_mps_prefix_rotation_covariance
    psi,state = _small_mps(filtered,gauged)
    weights = np.exp(2j*np.pi*np.arange(4)/3)
    report = finite_mps_prefix_rotation_covariance(psi,4,transition_sites=tuple(range(4)),
        chirality_groups={'kappa':((0,1),(3,2),(0,3))},
        chirality_z_response=True,
        fourier_spin_weights={'test':tuple(weights)})
    rho = state.reshape(16,16) @ state.reshape(16,16).conj().T
    bits = (np.arange(16)[:,None] >> np.arange(3,-1,-1)) & 1
    diagonal = (bits-.5) @ weights
    expected = np.sum(abs(rho)**2*diagonal.conj()[:,None]*diagonal[None,:])/np.trace(rho @ rho)
    np.testing.assert_allclose(report['two_copy_spin_response']['modes']['test']['purity_normalized'],
                               expected,atol=5e-14,rtol=0)
    assert '0_to_2' in report['leading_spin_content']['raising_transitions']
    operators=[];sz_ops=[]
    for i in range(4):
        op=np.zeros((16,16),complex)
        for b in range(16):
            bit=1<<(3-i)
            if not b & bit:op[b|bit,b]=1
        operators.append(op);sz_ops.append(np.diag(bits[:,i]-.5))
    chirality=sum(1j*(sz_ops[a]@operators[b]-operators[a]@sz_ops[b])
                  for a,b in ((0,1),(3,2),(0,3)))
    kappa_z=sum(.5j*(operators[a]@operators[b].conj().T-operators[a].conj().T@operators[b])
                   for a,b in ((0,1),(3,2),(0,3)))
    expected_z=np.trace(rho @ kappa_z @ rho @ kappa_z)/np.trace(rho @ rho)
    np.testing.assert_allclose(report['two_copy_chirality_z_response']['channels']['kappa']['purity_normalized'],
        expected_z,atol=2e-14,rtol=0)
    charges=2*bits.sum(axis=1)-4;ii=np.flatnonzero(charges == 0);jj=np.flatnonzero(charges == 2)
    wi,vi=np.linalg.eigh(rho[np.ix_(ii,ii)]);wf,vf=np.linalg.eigh(rho[np.ix_(jj,jj)])
    expected=vf[:,::-1].conj().T @ chirality[np.ix_(jj,ii)] @ vi[:,::-1]
    value=report['leading_spin_content']['raising_transitions']['0_to_2']['site_matrices']['kappa']
    measured=np.array(value['real'])+1j*np.array(value['imag'])
    # Inverse-Gram conditioning amplifies tiny-weight eigenvector errors.
    # Check the leading vectors directly AND every physical weighted matrix
    # element at a much stricter absolute tolerance; never drop tail levels.
    np.testing.assert_allclose(abs(measured[:3,:3])**2,abs(expected[:3,:3])**2,atol=1e-12,rtol=0)
    physical_weights=wf[::-1][:len(measured),None]*wi[::-1][None,:measured.shape[1]]
    np.testing.assert_allclose(physical_weights*abs(measured)**2,
        physical_weights*abs(expected[:len(measured),:measured.shape[1]])**2,atol=8e-15,rtol=0)
    # Squared chirality alone cannot detect an erroneous overall sign.
    # Its complex interference with an independently defined s_0^+ removes
    # Schmidt eigenvector phases and fixes the oriented cross-product sign.
    spin_value=report['leading_spin_content']['raising_transitions']['0_to_2']['site_matrices']['0']
    measured_spin=np.array(spin_value['real'])+1j*np.array(spin_value['imag'])
    expected_spin=vf[:,::-1].conj().T @ operators[0][np.ix_(jj,ii)] @ vi[:,::-1]
    np.testing.assert_allclose(physical_weights*measured*measured_spin.conj(),
        physical_weights*expected[:len(measured),:measured.shape[1]]*expected_spin[:len(measured),:measured.shape[1]].conj(),
        atol=8e-15,rtol=0)
