import numpy as np
import pytest

from fnqs.exact import TFIMReference
from fnqs.hamiltonians import IsingCouplings, all_bits, connectivity, dense_hamiltonian, ground_state
from fnqs.observables import correlations_from_state, fidelity_metric, quantum_geometric_tensor, renyi2_from_state
from fnqs.scaling import collapse_fit, fit_c, fit_eta, phase0_gate
from fnqs.subspace import project_subspace, select_snapshots


def kron_operator(n, sites, op):
    result = np.ones((1, 1))
    for site in range(n):
        result = np.kron(result, op if site in sites else np.eye(2))
    return result


@pytest.mark.parametrize("g", [IsingCouplings(h=0.8), IsingCouplings(j2=0.2,h=1.2), IsingCouplings(jx=0.1,h=0.9)])
def test_super_hamiltonian_against_independent_pauli_kronecker(g):
    n = 4
    x, z = np.array([[0, 1], [1, 0]]), np.diag([-1, 1])
    expected = np.zeros((1 << n, 1 << n))
    for i in range(n):
        expected -= g.j1 * kron_operator(n, {i, (i+1)%n}, z)
        expected -= g.j2 * kron_operator(n, {i, (i+2)%n}, z)
        expected -= g.jx * kron_operator(n, {i, (i+1)%n}, x)
        expected -= g.h * kron_operator(n, {i}, x)
    actual = dense_hamiltonian(n, g)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    flip = kron_operator(n, set(range(n)), x)
    np.testing.assert_allclose(actual @ flip, flip @ actual, atol=1e-14)
    for bits in all_bits(n):
        new, _ = connectivity(bits, g)
        assert len(np.unique(new, axis=0)) == len(new)


@pytest.mark.parametrize("n", [4, 6, 8, 10])
@pytest.mark.parametrize("h", [0.7, 1.0, 1.3])
def test_jordan_wigner_energy_against_ed(n,h):
    energy, state, residual = ground_state(n, IsingCouplings(h=h))
    assert residual < 4e-11
    np.testing.assert_allclose(energy, TFIMReference(n,h).energy(), rtol=0, atol=4e-11)
    np.testing.assert_allclose(np.linalg.norm(state), 1, atol=2e-14)


@pytest.mark.parametrize("n", [4, 6, 8])
@pytest.mark.parametrize("h", [0.7, 1.0, 1.3])
def test_majorana_observables_against_computational_state(n,h):
    _, state, _ = ground_state(n, IsingCouplings(h=h))
    reference = TFIMReference(n,h)
    correlations = correlations_from_state(state,n)
    for r in range(n):
        np.testing.assert_allclose(reference.correlation(r), correlations[r], atol=4e-11, rtol=0)
    for ell in range(1,n):
        np.testing.assert_allclose(reference.entropy(ell,2),renyi2_from_state(state,n,ell),atol=4e-11,rtol=0)


def test_fidelity_qgt_against_spectral_perturbation_and_critical_identity():
    n, h = 6, 0.95
    matrix = dense_hamiltonian(n,IsingCouplings(h=h))
    values, vectors = np.linalg.eigh(matrix)
    derivative = -sum(kron_operator(n,{i},np.array([[0,1],[1,0]])) for i in range(n))
    overlaps = vectors[:,1:].T @ derivative @ vectors[:,0]
    metric = np.sum(abs(overlaps)**2/(values[1:]-values[0])**2)
    np.testing.assert_allclose(metric,TFIMReference(n,h).fidelity_susceptibility(),atol=2e-12)
    delta = 1e-4
    _, plus, _ = ground_state(n,IsingCouplings(h=h+delta))
    _, minus, _ = ground_state(n,IsingCouplings(h=h-delta))
    np.testing.assert_allclose(fidelity_metric(vectors[:,0],plus,minus,delta),metric,atol=2e-7)
    for n in [4,16,24,32,48,64]:
        np.testing.assert_allclose(TFIMReference(n,1).fidelity_susceptibility(),n*(n-1)/32,atol=3e-12)


def test_qgt_gauge_invariance_hermiticity_and_psd():
    rng = np.random.default_rng(10)
    scores = rng.normal(size=(20,4))+1j*rng.normal(size=(20,4))
    weights = rng.random(20)
    qgt = quantum_geometric_tensor(scores,weights)
    np.testing.assert_allclose(qgt,quantum_geometric_tensor(scores+np.array([2,3j,-4,5+1j]),weights),atol=3e-14)
    np.testing.assert_allclose(qgt,qgt.conj().T,atol=2e-14)
    assert np.linalg.eigvalsh(qgt).min()>=-1e-14


def test_reference_scaling_and_entropy_coefficients_are_free_fits():
    reference = TFIMReference(64,1)
    correlations = np.array([reference.correlation(r) for r in range(64)])
    assert abs(fit_eta(64,correlations,8).value-.25)<.005
    ells=np.arange(16,33)
    for order in [1,2]:
        entropy=np.array([reference.entropy(int(ell),order) for ell in ells])
        assert abs(fit_c(64,ells,entropy,order).value-.5)<.03
    sizes,fields,chi=[],[],[]
    for n in [16,24,32,48,64]:
        for h in np.linspace(1-.8/n,1+.8/n,21):
            sizes.append(n);fields.append(h);chi.append(TFIMReference(n,float(h)).fidelity_susceptibility())
    fit=collapse_fit(sizes,fields,chi)
    assert fit["success"] and not fit["at_bound"]
    assert abs(fit["hc"]-1)<.02 and abs(fit["nu"]-1)<.05
    assert not phase0_gate(1,1,.25,True,[4,6,8],1)["passed"]
    assert phase0_gate(1,1,.25,True,[16,24,32],3)["passed"]


def test_nfs_snapshot_rank_and_reusable_projection():
    states=np.stack([ground_state(4,IsingCouplings(h=float(h)))[1] for h in np.linspace(.6,1.4,9)])
    selected=select_snapshots(states,.01)
    assert selected[0]==0 and selected[-1]==8
    matrix=dense_hamiltonian(4,IsingCouplings())
    projected=project_subspace(np.concatenate([states,states[:1]]),matrix)
    w=projected["whitening"]
    np.testing.assert_allclose(w.conj().T@projected["overlap"]@w,np.eye(projected["rank"]),atol=2e-8)
    np.testing.assert_allclose(projected["energies"][0],np.linalg.eigvalsh(matrix)[0],atol=1e-8)


@pytest.mark.parametrize("n",[2,3,5,-4,4.5,True])
def test_invalid_sizes_are_rejected(n):
    with pytest.raises(ValueError): TFIMReference(n,1)


def test_reference_memory_and_degeneracy_guards():
    with pytest.raises(ValueError): TFIMReference(4,0)
    with pytest.raises(ValueError): ground_state(16,IsingCouplings())
    with pytest.raises(ValueError): dense_hamiltonian(12,IsingCouplings())
    with pytest.raises(ValueError): IsingCouplings(h=float("nan"))
