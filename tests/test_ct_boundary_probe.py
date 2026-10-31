"""Independent physical-basis checks for the bounded boundary probe."""
import numpy as np
from scipy.linalg import expm

from su2_nqs.benchmarks.ct_boundary_probe import (
    exchange_commutator, exchange_gate, four_by_four_singlets,
    physical_swap, rank_one_coefficients, rank_one_energy_gradient,
    isometric_rank_one_coefficients,
    factors_to_isometric,
)
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix


def test_singlets_and_cut_change_are_complete_orthonormal_su2_spaces():
    first = four_by_four_singlets((0, 1, 4, 5))
    second = four_by_four_singlets((0, 1, 2, 3))
    s2, _ = spin_squared_matrix(8)
    np.testing.assert_allclose(first.T @ first, np.eye(14), atol=3e-14)
    np.testing.assert_allclose(s2 @ first, 0, atol=3e-14)
    f = second.T @ first
    np.testing.assert_allclose(f.T @ f, np.eye(14), atol=3e-14)


def test_boundary_gate_matches_full_physical_exponential_and_stays_singlet():
    basis = four_by_four_singlets((0, 1, 4, 5))
    p, r = physical_swap(8, 1, 2), physical_swap(8, 2, 6)
    physical = p @ r - r @ p
    generator = exchange_commutator(basis, (1, 2, 6))
    gate, derivative = exchange_gate(generator, .31)
    np.testing.assert_allclose(gate.T @ gate, np.eye(14), atol=3e-14)
    np.testing.assert_allclose(basis @ gate, expm(.31 * physical) @ basis, atol=3e-14)
    np.testing.assert_allclose(basis @ derivative, physical @ basis @ gate, atol=3e-14)


def test_within_block_gate_is_absorbed_by_complete_block_factors():
    basis = four_by_four_singlets((0, 1, 4, 5))
    c, _ = rank_one_coefficients(np.random.default_rng(15).normal(size=12))
    within, _ = exchange_gate(exchange_commutator(basis, (0, 1, 5)), .71)
    after = within @ c
    assert np.linalg.svd(after[:4].reshape(2, 2), compute_uv=False)[1] < 1e-13
    assert np.linalg.svd(after[4:13].reshape(3, 3), compute_uv=False)[1] < 1e-13


def test_energy_gradient_matches_physical_rayleigh_finite_difference():
    basis = four_by_four_singlets((0, 1, 4, 5))
    h = basis.T @ (.5 * physical_swap(8, 1, 2) - .25 * np.eye(256)) @ basis
    generator = exchange_commutator(basis, (1, 2, 6))
    x = np.random.default_rng(9).normal(size=13)
    _, gradient, state = rank_one_energy_gradient(x, h, generator=generator)
    finite = []
    for i in range(13):
        step = np.eye(13)[i] * 1e-6
        ep = rank_one_energy_gradient(x + step, h, generator=generator)[0]
        em = rank_one_energy_gradient(x - step, h, generator=generator)[0]
        finite.append((ep - em) / 2e-6)
    np.testing.assert_allclose(gradient, finite, atol=5e-10)
    np.testing.assert_allclose(state @ state, 1, atol=2e-14)


def test_isometric_coordinates_normalization_jacobian_and_rank():
    x = np.random.default_rng(43).normal(size=8)
    c, jac = isometric_rank_one_coefficients(x)
    finite = []
    for i in range(8):
        step = np.eye(8)[i] * 1e-6
        finite.append((isometric_rank_one_coefficients(x+step)[0]
                       - isometric_rank_one_coefficients(x-step)[0]) / 2e-6)
    np.testing.assert_allclose(jac, np.array(finite).T, atol=3e-10)
    np.testing.assert_allclose(c @ c, 1, atol=2e-14)
    np.testing.assert_allclose(c @ jac, 0, atol=2e-14)
    assert np.linalg.svd(c[:4].reshape(2, 2), compute_uv=False)[1] < 1e-13
    assert np.linalg.svd(c[4:13].reshape(3, 3), compute_uv=False)[1] < 1e-13


def test_isometric_conversion_preserves_every_normalized_amplitude_and_sign():
    for seed in (71, 103, 211, 307):
        factors = np.random.default_rng(seed).normal(size=12)
        before = rank_one_coefficients(factors)[0]
        after = isometric_rank_one_coefficients(factors_to_isometric(factors))[0]
        np.testing.assert_allclose(after, before/np.linalg.norm(before), atol=2e-14)
