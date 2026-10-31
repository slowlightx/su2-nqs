"""Independent physical-H oracles for the overlapping-carrier cost probe.

These tests use Pauli Kronecker products, not the connectivity routine under
test, and never diagonalize H. Complex amplitudes, nodes, repeated bonds and
parameter derivatives are retained. N=12 only checks the two-filter/H-query
composition on a few rows; it is not a variational optimization benchmark.
"""

from functools import reduce

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse

from su2_nqs.vmc.physical_h_queries import (
    compile_physical_h_queries, gather_physical_h_action,
)


def _bits(n_sites):
    return ((np.arange(1 << n_sites)[:, None] >>
             np.arange(n_sites - 1, -1, -1)) & 1).astype(np.int8)


def _indices(bits):
    return np.asarray(bits) @ (1 << np.arange(bits.shape[1] - 1, -1, -1))


def _pauli_hamiltonian(n_sites, bonds):
    """Independent sparse sum of SxSx+SySy+SzSz; bit zero means down."""
    spin = (
        np.array([[0., .5], [.5, 0.]], dtype=complex),
        np.array([[0., .5j], [-.5j, 0.]], dtype=complex),
        np.diag([-.5, .5]).astype(complex),
    )
    identity = sparse.eye(2, format="csr", dtype=complex)
    h = sparse.csr_matrix((1 << n_sites, 1 << n_sites), dtype=complex)
    for bond in bonds:
        first, second = bond[:2]
        coupling = bond[2] if len(bond) == 3 else 1.
        for operator in spin:
            factors = [sparse.csr_matrix(operator) if i in (first, second)
                       else identity for i in range(n_sites)]
            h += coupling * reduce(lambda a, b: sparse.kron(a, b, format="csr"), factors)
    return h


@pytest.mark.parametrize("n_sites", [2, 4, 6])
def test_complete_complex_action_agrees_with_independent_pauli_hamiltonian(n_sites):
    basis = _bits(n_sites)
    bonds = [(i, i + 1, (-1.)**i * (1. + .17 * i)) for i in range(n_sites - 1)]
    if n_sites > 2:
        bonds += [(0, n_sites - 1, .125), (1, n_sites - 1, -.31)]
    h = _pauli_hamiltonian(n_sites, bonds)
    rng = np.random.default_rng(6101200 + n_sites)
    carrier = rng.normal(size=len(basis)) + 1j * rng.normal(size=len(basis))
    plan = compile_physical_h_queries(basis, bonds)
    psi, hpsi = gather_physical_h_action(plan, carrier[_indices(plan.configurations)])
    error = np.max(abs(np.asarray(hpsi) - h @ carrier))
    np.testing.assert_array_equal(psi, carrier)
    np.testing.assert_allclose(hpsi, h @ carrier, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(np.vdot(carrier, hpsi).imag, 0., atol=3e-13)
    assert plan.diagnostics["full_H_centers"] == len(basis)
    print(f"N={n_sites} independent Pauli maximum H-action discrepancy: {error:.3e}")


def test_duplicate_reversed_bonds_and_duplicate_centers_are_exactly_coalesced():
    basis = _bits(4)
    bonds = ((0, 1), (1, 0, .25), (0, 1, -.5),
             (2, 3, .4), (3, 2, -.4), (0, 3, -.75))
    equivalent = ((0, 1, .75), (0, 3, -.75))
    selected = np.array([10, 0, 5, 10, 15, 3, 5])
    samples = basis[selected]
    before = samples.copy()
    plan = compile_physical_h_queries(samples, bonds)
    clean_plan = compile_physical_h_queries(samples, equivalent)
    rng = np.random.default_rng(6101207)
    carrier = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi, hpsi = gather_physical_h_action(plan, carrier[_indices(plan.configurations)])
    _, clean_action = gather_physical_h_action(clean_plan, carrier[_indices(clean_plan.configurations)])
    np.testing.assert_array_equal(samples, before)
    np.testing.assert_array_equal(psi, carrier[selected])
    np.testing.assert_allclose(hpsi, (_pauli_hamiltonian(4, bonds) @ carrier)[selected], atol=3e-15)
    np.testing.assert_allclose(hpsi, clean_action, atol=3e-15)
    assert len(np.unique(plan.configurations, axis=0)) == len(plan.configurations)
    assert plan.center_indices[0] == plan.center_indices[3]
    assert plan.center_indices[2] == plan.center_indices[6]
    assert plan.diagnostics["duplicate_requests_removed"] > 0
    for row, mask in zip(plan.connected_indices, plan.valid):
        assert len(set(row[mask])) == int(mask.sum())
    for array in (plan.configurations, plan.centers, plan.center_indices,
                  plan.connected_indices, plan.elements, plan.valid):
        assert not array.flags.writeable


def test_exact_nodes_keep_hpsi_and_the_missing_second_moment_contribution():
    basis = _bits(4)
    bonds = ((0, 1, 1.),)
    carrier = np.zeros(16, dtype=complex)
    carrier[9] = 1. + 2.j  # 1001; exchanging sites 0/1 reaches the node 0101.
    plan = compile_physical_h_queries(basis, bonds)
    psi, hpsi = map(np.asarray, gather_physical_h_action(plan, carrier[_indices(plan.configurations)]))
    assert psi[5] == 0.
    np.testing.assert_allclose(hpsi[5], .5 * carrier[9], atol=0., rtol=0.)
    assert np.isfinite(psi).all() and np.isfinite(hpsi).all()
    expected = _pauli_hamiltonian(4, bonds) @ carrier
    np.testing.assert_allclose(hpsi, expected, atol=0., rtol=0.)
    np.testing.assert_allclose(np.vdot(hpsi, hpsi), np.vdot(expected, expected), atol=0., rtol=0.)
    # Discarding psi==0 rows would lose a strictly positive piece of <H^2>.
    assert np.linalg.norm(hpsi[psi == 0.])**2 > 1.


@pytest.mark.parametrize("bonds", [(), ((0, 1, 0.),), ((0, 1, 1.), (1, 0, -1.))])
def test_empty_zero_and_cancelled_hamiltonians_preserve_all_centers(bonds):
    samples = _bits(4)[[0, 5, 15, 5]]
    plan = compile_physical_h_queries(samples, bonds)
    values = np.arange(len(plan.configurations)) + 1.j
    psi, hpsi = gather_physical_h_action(plan, values)
    np.testing.assert_array_equal(psi, values[plan.center_indices])
    np.testing.assert_array_equal(hpsi, np.zeros(len(samples), dtype=complex))


def test_hpsi_and_ratios_have_live_complex_amplitude_jvps():
    basis = _bits(6)
    bonds = ((0, 1), (1, 3, .2), (3, 5, -.4), (2, 4, .7), (5, 0, .13))
    selected = np.array([1, 12, 21, 42, 51, 62, 21])
    plan = compile_physical_h_queries(basis[selected], bonds)
    h = jnp.asarray(_pauli_hamiltonian(6, bonds).toarray())
    rng = np.random.default_rng(6101211)
    feature = jnp.asarray(rng.normal(scale=.2, size=(64, 5)))
    phase = jnp.asarray(rng.normal(scale=.15, size=(64, 5)))
    point = jnp.asarray(rng.normal(scale=.1, size=5))
    direction = jnp.asarray(rng.normal(size=5))
    query_indices = jnp.asarray(_indices(plan.configurations))

    def amplitudes(p):
        return jnp.exp(feature @ p + 1j * (phase @ p))

    def queried(p):
        return gather_physical_h_action(plan, amplitudes(p)[query_indices])

    def reference(p):
        value = amplitudes(p)
        return value[selected], (h @ value)[selected]

    (psi, action), (dpsi, daction) = jax.jit(lambda p, d: jax.jvp(queried, (p,), (d,)))(point, direction)
    expected, expected_tangent = jax.jvp(reference, (point,), (direction,))
    for observed, exact in zip((psi, action, dpsi, daction), (*expected, *expected_tangent)):
        np.testing.assert_allclose(observed, exact, atol=5e-14, rtol=5e-14)
    assert np.linalg.norm(daction) > 1e-4
    # Ratios are formed only here, using this deliberately nowhere-zero fixture.
    ratio = lambda p: queried(p)[1] / queried(p)[0]
    _, tangent = jax.jvp(ratio, (point,), (direction,))
    eps = 1e-6
    finite = (ratio(point + eps * direction) - ratio(point - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(tangent, finite, atol=5e-10, rtol=5e-9)


def test_overlapping_filter_plus_full_h_replays_pauli_rows_not_only_carrier_rows():
    from su2_nqs.benchmarks.overlapping_carrier_queries import (
        compile_overlapping_carrier_queries, overlapping_carrier_amplitudes,
    )
    from su2_nqs.models.conditional_carrier_resonance import conditional_filter_matrix

    basis = _bits(12)
    # Inter-cell edges change individual cell magnetizations. Restricting H to
    # the center's original gate inputs would silently omit these contributions.
    bonds = ((0, 1), (3, 4, .7), (7, 8, -.2), (10, 11), (11, 0, .125))
    selected = np.array([1365, 2730, 1450])
    h_plan = compile_physical_h_queries(basis[selected], bonds)
    q_plan = compile_overlapping_carrier_queries(h_plan.configurations, (tuple(range(12)),))
    rng = np.random.default_rng(6101212)
    carrier = rng.normal(size=4096) + 1j * rng.normal(size=4096)
    carrier /= np.linalg.norm(carrier)
    first, second = rng.normal(scale=.09, size=(2, 60))
    k1, k2 = [np.asarray(conditional_filter_matrix(p)).reshape(16, 16, 16, 16)
              for p in (first, second)]
    tensor = carrier.reshape(16, 16, 16)
    one = np.einsum("bade,edc->abc", k1, tensor)
    two = np.einsum("cbde,aed->abc", k2, tensor)
    composed = np.einsum("cbde,aed->abc", k2, one).reshape(-1)
    additive = (one + two - tensor).reshape(-1)
    h = _pauli_hamiltonian(12, bonds)
    cached_carrier = carrier[_indices(q_plan.configurations)]
    for mode, exact in (("cascade", composed), ("no_cascade", additive)):
        values = overlapping_carrier_amplitudes(first, second, q_plan, cached_carrier, mode=mode)
        psi, hpsi = gather_physical_h_action(h_plan, values)
        np.testing.assert_allclose(psi, exact[selected], atol=2e-14, rtol=2e-13)
        error = np.max(abs(np.asarray(hpsi) - (h @ exact)[selected]))
        np.testing.assert_allclose(hpsi, (h @ exact)[selected], atol=2e-14, rtol=2e-13)
        print(f"N=12 {mode} independent full-H composition discrepancy: {error:.3e}")
    assert h_plan.diagnostics["full_H_centers"] == len(selected)
    assert len(h_plan.configurations) > len(selected)


@pytest.mark.parametrize("bad", [np.zeros((0, 4), dtype=int), np.zeros((3, 1), dtype=int),
                                  np.zeros(4, dtype=int), np.zeros((2, 4), dtype=float),
                                  np.array([[0, 1, 2, 0]])])
def test_invalid_physical_centers_fail_closed(bad):
    with pytest.raises(ValueError):
        compile_physical_h_queries(bad, ((0, 1),))


@pytest.mark.parametrize("bonds", [((0,),), ((0, 1, 1., 2.),), ((True, 1),),
                                    ((0., 1),), ((0, 0),), ((-1, 1),),
                                    ((0, 4),), ((0, 1, np.nan),), ((0, 1, np.inf),)])
def test_invalid_bonds_fail_closed(bonds):
    with pytest.raises(ValueError):
        compile_physical_h_queries(_bits(4)[:2], bonds)


def test_missing_or_extra_query_amplitudes_fail_closed():
    plan = compile_physical_h_queries(_bits(4)[[5, 10]], ((0, 1),))
    for shape in ((len(plan.configurations) - 1,), (len(plan.configurations), 1),
                  (len(plan.configurations) + 1,)):
        with pytest.raises(ValueError):
            gather_physical_h_action(plan, jnp.zeros(shape))
