"""Independent finite-state oracles for IID proposals and physical moments."""

from collections import Counter
from itertools import combinations, product

import jax
import numpy as np
import pytest

from peps_vmc.diagnostics import finite_FS, paired_moments
from peps_vmc.model import SquareGraphSU2PEPS
from peps_vmc.proposals import CoveringBridgeProposal, torus_coverings
from peps_vmc.symmetry import intertwiner_labels, rvb_coefficients
from su2_nqs.benchmarks.exact_diagonalization import heisenberg_sz_matrix
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds

jax.config.update("jax_enable_x64", True)


def four_site_proposal():
    coverings = np.array([[[0, 1], [2, 3]], [[0, 2], [1, 3]]], dtype=np.int32)
    bonds = np.asarray(tuple(combinations(range(4), 2)), dtype=np.int32)
    probabilities = np.array([0.4, 0.05, 0.1, 0.2, 0.05, 0.2])
    return CoveringBridgeProposal(coverings, bonds, probabilities, epsilon=0.25)


def explicit_base_law(proposal):
    """Construct the complete finite generative law, without density callbacks."""
    n = proposal.n_sites
    basis = tuple(bits for bits in product((0, 1), repeat=n) if sum(bits) == n // 2)
    law = {row: proposal.epsilon / len(basis) for row in basis}
    for covering in proposal.coverings:
        for values in product((0, 1), repeat=n // 2):
            row = np.zeros(n, dtype=np.int32)
            for (first, second), value in zip(covering, values):
                row[first], row[second] = value, 1 - value
            law[tuple(row)] += (1 - proposal.epsilon) / len(proposal.coverings) / 2 ** (n // 2)
    return basis, law


def test_four_site_exact_density_normalization_and_null_swaps():
    proposal = four_site_proposal()
    basis, base = explicit_base_law(proposal)
    expected = {row: 0.5 * value for row, value in base.items()}
    null_mass = 0.0
    for row, mass in base.items():
        for (first, second), probability in zip(proposal.bonds, proposal.exchange_probabilities):
            changed = list(row)
            changed[first], changed[second] = changed[second], changed[first]
            expected[tuple(changed)] += 0.5 * probability * mass
            if row[first] == row[second]:
                null_mass += 0.5 * probability * mass
    rows = np.asarray(basis, dtype=np.int32)
    np.testing.assert_allclose(proposal.base_density(rows), [base[row] for row in basis], atol=2e-16, rtol=0)
    np.testing.assert_allclose(proposal.density(rows), [expected[row] for row in basis], atol=2e-16, rtol=0)
    assert sum(base.values()) == pytest.approx(1.0, abs=2e-16)
    assert sum(expected.values()) == pytest.approx(1.0, abs=2e-16)
    assert 0 < null_mass < 0.5
    assert np.min(proposal.density(rows)) >= proposal.epsilon / 6 - 2e-16


def test_four_site_iid_sampling_frequencies_and_rng_restore():
    proposal = four_site_proposal()
    basis, _ = explicit_base_law(proposal)
    rng = np.random.default_rng(2026103003)
    state = rng.bit_generator.state
    draws, density = proposal.draw(30_000, rng)
    counts = Counter(map(tuple, draws))
    empirical = np.array([counts[row] / len(draws) for row in basis])
    expected = proposal.density(np.asarray(basis, dtype=np.int32))
    np.testing.assert_allclose(empirical, expected, atol=9e-3, rtol=0)
    np.testing.assert_array_equal(draws.sum(1), np.full(len(draws), 2))
    np.testing.assert_array_equal(density, proposal.density(draws))
    restored = np.random.default_rng()
    restored.bit_generator.state = state
    repeated, repeated_density = proposal.draw(30_000, restored)
    np.testing.assert_array_equal(repeated, draws)
    np.testing.assert_array_equal(repeated_density, density)


@pytest.mark.parametrize("length", [4, 6])
def test_torus_coverings_are_complete_actual_triangular_nearest_neighbors(length):
    coverings = torus_coverings(length, length, count=18, seed=2026103004)
    n = length**2
    nearest = {(first, second) for first, second, _ in triangular_j1j2_bonds(length, length, periodic=True)}
    assert coverings.shape == (18, n // 2, 2)
    for covering in coverings:
        np.testing.assert_array_equal(np.sort(covering.reshape(-1)), np.arange(n))
        assert all(tuple(pair) in nearest for pair in covering)
    np.testing.assert_array_equal(coverings, torus_coverings(length, length, count=18, seed=2026103004))


@pytest.mark.parametrize("shared", [False, True])
def test_reduced_model_keeps_all_channels_and_declared_parameter_count(shared):
    model = SquareGraphSU2PEPS(4, 4, shared=shared)
    weights = model.initial_coefficients(2026103005)
    assert weights.shape == ((12,) if shared else (16, 12))
    assert np.isfinite(weights).all() and np.all(weights != 0)
    assert model.metadata()["real_parameter_count"] == (12 if shared else 192)
    assert not model.metadata()["triangular_point_group_A1_enforced"]
    tensor = np.asarray(model.tensors(weights))
    assert tensor.shape == (16, 2, 3, 3, 3, 3)
    assert np.isfinite(tensor).all()
    with pytest.raises(ValueError, match="coefficient shape"):
        model.tensors(np.zeros((16, 11)))


def test_actual_decoder_bond_absorption_closes_to_physical_singlet():
    # Four decoded tensors, with two parallel edges between adjacent sites,
    # form a small complete closed-loop oracle for the even-to-odd metric.
    model = SquareGraphSU2PEPS(4, 4)
    weights = np.random.default_rng(2026103006).normal(size=model.parameter_shape)
    tensors = np.asarray(model.tensors(weights))
    a, b, c, d = tensors[[0, 1, 5, 4]]  # even, odd, even, odd physical sites.
    psi = np.einsum(a, [0, 4, 5, 10, 11], b, [1, 4, 5, 6, 7],
                    c, [2, 6, 7, 8, 9], d, [3, 8, 9, 10, 11],
                    [0, 1, 2, 3], optimize="greedy").reshape(-1)
    assert np.linalg.norm(psi) > 1e-8
    psi /= np.linalg.norm(psi)
    local_generators = (np.array([[0, .5], [.5, 0]]),
                        np.array([[0, .5j], [-.5j, 0]]), np.diag([-.5, .5]))
    for local in local_generators:
        total = np.zeros((16, 16), dtype=complex)
        for site in range(4):
            term = np.ones((1, 1), dtype=complex)
            for other in range(4):
                term = np.kron(term, local if other == site else np.eye(2))
            total += term
        np.testing.assert_allclose(total @ psi, 0, atol=3e-14, rtol=0)


def test_three_leg_activation_on_both_sublattices_avoids_exact_dead_channels():
    model = SquareGraphSU2PEPS(4, 4)
    base = np.broadcast_to(rvb_coefficients(4), model.parameter_shape).copy()
    extra = [i for i, label in enumerate(intertwiner_labels(4)) if sum(label.virtual_q) == 3]

    def wavefunction(weights):
        a, b, c, d = np.asarray(model.tensors(weights))[[0, 1, 5, 4]]
        return np.einsum(a, [0, 4, 5, 10, 11], b, [1, 4, 5, 6, 7],
                         c, [2, 6, 7, 8, 9], d, [3, 8, 9, 10, 11],
                         [0, 1, 2, 3], optimize="greedy").reshape(-1)

    original = wavefunction(base)
    for channel in extra:
        one_vertex = base.copy()
        one_vertex[0, channel] += .1
        # A three-occupied-leg vertex on only one sublattice cannot close:
        # both halves of every occupied virtual edge must be occupied.
        np.testing.assert_allclose(wavefunction(one_vertex), original, atol=2e-14, rtol=0)
    activated = base.copy()
    activated[:, extra] += .08 * np.random.default_rng(2026103007).normal(size=(16, len(extra)))
    background = wavefunction(activated)
    norms = []
    for channel in extra:
        one_vertex = activated.copy()
        one_vertex[0, channel] += .1
        norms.append(np.linalg.norm((wavefunction(one_vertex) - background) / .1))
    assert max(norms) > 1e-4 * np.linalg.norm(background)


def test_node_safe_moments_and_energy_derivative_match_complete_dense_quadrature():
    ham, _ = heisenberg_sz_matrix(4, [(0, 1, 1.0), (1, 2, .7), (2, 3, 1.1), (0, 3, .125)], n_up=2)
    parent = np.array([1.0, 0.0, .2, 0.0, -.3, .4])
    tangent = np.array([.1, .3, -.1, .2, .15, -.05])
    candidate = parent + .03 * tangent
    density = np.full(6, 1 / 6)
    products = np.stack([np.stack((p, ham @ p), axis=-1) for p in (parent, candidate)], axis=1)
    moments = paired_moments(products, density, n_sites=4, groups=6)
    for endpoint, vector in enumerate((parent, candidate)):
        norm = vector @ vector
        energy = vector @ ham @ vector / norm
        variance = vector @ (ham @ ham) @ vector / norm - energy**2
        np.testing.assert_allclose(moments["endpoint_values"][endpoint][:3], [energy / 4, 0, variance / 4], atol=3e-16)
    assert moments["target_node_records"] == [2, 0]
    assert moments["target_node_H2_contributions"][0] > 0
    parent_energy = parent @ ham @ parent / (parent @ parent)
    exact_derivative = 2 * tangent @ (ham @ parent - parent_energy * parent) / (parent @ parent) / 4
    eps = 1e-6
    endpoint_energies = []
    for sign in (-1, 1):
        vector = parent + sign * eps * tangent
        values = np.stack((products[:, 0], np.stack((vector, ham @ vector), axis=-1)), axis=1)
        endpoint_energies.append(paired_moments(values, density, n_sites=4, groups=6)["endpoint_values"][1][0])
    assert (endpoint_energies[1] - endpoint_energies[0]) / (2 * eps) == pytest.approx(exact_derivative, rel=2e-9, abs=2e-11)
    geometry = finite_FS(np.stack((parent, candidate), axis=1), density)
    expected_fidelity = (parent @ candidate)**2 / ((parent @ parent) * (candidate @ candidate))
    assert geometry["fidelity"] == pytest.approx(expected_fidelity, abs=3e-16)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0, -1.0])
def test_nonfinite_or_nonpositive_geometry_density_is_rejected(bad):
    density = np.ones(6)
    density[0] = bad
    with pytest.raises((ValueError, FloatingPointError)):
        finite_FS(np.ones((6, 2)), density)


def test_empty_covering_library_and_noninteger_bonds_are_rejected():
    proposal = four_site_proposal()
    with pytest.raises(ValueError):
        CoveringBridgeProposal(np.empty((0, 2, 2), dtype=np.int32), proposal.bonds,
                               proposal.exchange_probabilities)
    with pytest.raises(ValueError):
        CoveringBridgeProposal(proposal.coverings, proposal.bonds.astype(float),
                               proposal.exchange_probabilities)


def test_frozen_proposal_owns_read_only_copies_of_registered_arrays():
    prototype = four_site_proposal()
    coverings = prototype.coverings.copy()
    bonds = prototype.bonds.copy()
    probabilities = prototype.exchange_probabilities.copy()
    proposal = CoveringBridgeProposal(coverings, bonds, probabilities)
    rows = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.int32)
    registered = proposal.density(rows)
    coverings.fill(0)
    bonds.fill(0)
    probabilities.fill(0)
    np.testing.assert_array_equal(proposal.density(rows), registered)
    for frozen in (proposal.coverings, proposal.bonds, proposal.exchange_probabilities):
        with pytest.raises(ValueError, match="read-only"):
            frozen.flat[0] = 0
