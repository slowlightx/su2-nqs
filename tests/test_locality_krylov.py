import numpy as np
import pytest

from su2_nqs.benchmarks.locality_krylov import (
    local_block_krylov_ritz,
    adaptive_local_imaginary_time_step,
    importance_sampling_diagnostics,
    local_krylov_hierarchy,
    optimal_polynomial_filter,
)


def _block_problem(seed: int = 2026091460):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(12, 12))
    hamiltonian = 0.5 * (raw + raw.T)
    seeds = rng.normal(size=(12, 3))
    return hamiltonian, seeds


def test_block_krylov_converges_and_reports_direct_residuals() -> None:
    hamiltonian, seeds = _block_problem()
    result = local_block_krylov_ritz(
        lambda vectors: hamiltonian @ vectors,
        seeds,
        maximum_depth=4,
        target_roots=3,
    )
    exact = np.linalg.eigvalsh(hamiltonian)[:3]
    assert [row.subspace_dimension for row in result.depths] == [3, 6, 9, 12]
    assert [row.hamiltonian_applications for row in result.depths] == [1, 2, 3, 4]
    assert result.stopped_after_rank_saturation
    np.testing.assert_allclose(result.depths[-1].energies, exact, atol=2.0e-13)
    assert np.max(result.depths[-1].residual_norms) < 3.0e-13
    assert max(row.orthonormality_error for row in result.depths) < 2.0e-14
    assert max(
        row.projected_hamiltonian_antihermitian_error for row in result.depths
    ) < 2.0e-14
    assert all(
        np.all(
            (0.0 <= row.seed_subspace_weights)
            & (row.seed_subspace_weights <= 1.0 + 1e-13)
        )
        for row in result.depths
    )


def test_block_depth_trajectory_is_invariant_to_nonsingular_seed_mixing() -> None:
    hamiltonian, seeds = _block_problem(2026091461)
    mixing = np.asarray(
        [[1.0, 0.2, -0.1], [0.3, 1.2, 0.15], [-0.2, 0.1, 0.8]]
    )
    first = local_block_krylov_ritz(
        lambda vectors: hamiltonian @ vectors, seeds, maximum_depth=2
    )
    second = local_block_krylov_ritz(
        lambda vectors: hamiltonian @ vectors,
        seeds @ mixing,
        maximum_depth=2,
    )
    for left, right in zip(first.depths, second.depths, strict=True):
        np.testing.assert_allclose(left.energies, right.energies, atol=2.0e-13)
        np.testing.assert_allclose(
            left.residual_norms, right.residual_norms, atol=3.0e-13
        )
        np.testing.assert_allclose(
            left.seed_subspace_weights, right.seed_subspace_weights, atol=2.0e-13
        )


def test_block_krylov_does_not_promote_near_breakdown_roundoff() -> None:
    hamiltonian = np.diag(np.arange(1.0, 7.0))
    coupling = 1.0e-14
    for index in range(3):
        hamiltonian[index, index + 3] = coupling
        hamiltonian[index + 3, index] = coupling
    seeds = np.eye(6)[:, :3]
    result = local_block_krylov_ritz(
        lambda vectors: hamiltonian @ vectors,
        seeds,
        maximum_depth=2,
        target_roots=3,
        relative_rank_tolerance=1.0e-11,
    )
    assert len(result.depths) == 1
    assert result.stopped_after_rank_saturation


def test_block_krylov_rejects_nonhermitian_projected_action() -> None:
    _, seeds = _block_problem(2026091463)
    nonhermitian = np.triu(np.ones((12, 12)), 1)
    with pytest.raises(FloatingPointError, match="not Hermitian"):
        local_block_krylov_ritz(
            lambda vectors: nonhermitian @ vectors,
            seeds,
            maximum_depth=1,
        )


def test_block_krylov_tracks_partial_complex_frontier_ranks() -> None:
    hamiltonian = np.diag(np.arange(7.0)).astype(np.complex128)
    for first, second, coupling in (
        (0, 3, 0.7 + 0.2j),
        (1, 4, -0.4 + 0.3j),
        (3, 5, 0.5 - 0.1j),
        (4, 6, -0.6 - 0.2j),
    ):
        hamiltonian[first, second] = coupling
        hamiltonian[second, first] = np.conj(coupling)
    seeds = np.eye(7, dtype=np.complex128)[:, :3]
    result = local_block_krylov_ritz(
        lambda vectors: hamiltonian @ vectors,
        seeds,
        maximum_depth=3,
        target_roots=3,
    )
    assert [row.subspace_dimension for row in result.depths] == [3, 5, 7]
    assert result.stopped_after_rank_saturation
    np.testing.assert_allclose(
        result.depths[-1].energies,
        np.linalg.eigvalsh(hamiltonian)[:3],
        atol=2.0e-13,
    )
    assert np.max(result.depths[-1].residual_norms) < 3.0e-13


def test_block_rank_and_action_contracts_fail_closed() -> None:
    hamiltonian, seeds = _block_problem(2026091462)
    collapsed = np.column_stack((seeds[:, 0], seeds[:, 0], seeds[:, 1]))
    with pytest.raises(ValueError, match="rank"):
        local_block_krylov_ritz(
            lambda vectors: hamiltonian @ vectors,
            collapsed,
            maximum_depth=2,
        )
    with pytest.raises(ValueError, match="preserve"):
        local_block_krylov_ritz(
            lambda vectors: (hamiltonian @ vectors)[:, :2],
            seeds,
            maximum_depth=1,
        )
    with pytest.raises(ValueError, match="maximum_depth"):
        local_block_krylov_ritz(
            lambda vectors: hamiltonian @ vectors,
            seeds,
            maximum_depth=-1,
        )


def test_local_krylov_energy_is_monotone_and_reaches_small_ground_state():
    matrix = np.asarray([[2.0, -1.0], [-1.0, 0.0]])
    exact = np.linalg.eigvalsh(matrix)[0]
    rows = local_krylov_hierarchy(matrix, np.asarray([1.0, 0.0]), maximum_depth=2)
    assert [row["energy"] for row in rows] == sorted(
        (row["energy"] for row in rows), reverse=True
    )
    assert rows[-1]["energy"] == pytest.approx(exact, abs=1e-12)
    assert rows[-1]["variance"] == pytest.approx(0.0, abs=1e-12)


def test_local_krylov_validates_inputs():
    with pytest.raises(ValueError):
        local_krylov_hierarchy(np.eye(2), np.zeros(2), maximum_depth=1)


def test_polynomial_filter_matches_ritz_and_reports_constant_weight_ess():
    matrix = np.asarray([[2.0, -1.0], [-1.0, 0.0]])
    initial = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    filtered, coefficients = optimal_polynomial_filter(matrix, initial, depth=1)
    assert coefficients.shape == (2,)
    assert np.real(np.vdot(filtered, matrix @ filtered)) == pytest.approx(
        np.linalg.eigvalsh(matrix)[0], abs=1e-12
    )
    diagnostics = importance_sampling_diagnostics(initial, initial)
    assert diagnostics["effective_sample_size_fraction"] == pytest.approx(1.0)
    assert diagnostics["maximum_normalized_weight"] == pytest.approx(1.0)


def test_adaptive_local_step_lowers_energy_without_breaking_ess_floor():
    matrix = np.asarray([[2.0, -1.0], [-1.0, 0.0]])
    initial = np.asarray([1.0, 0.0])
    _, report = adaptive_local_imaginary_time_step(
        matrix, initial, minimum_ess_fraction=0.6
    )
    assert report["energy_after"] < report["energy_before"]
    assert report["effective_sample_size_fraction"] >= 0.6


def test_importance_diagnostics_include_extremely_rare_nonzero_support():
    initial = np.sqrt(np.asarray([1.0 - 1e-20, 1e-20]))
    target = np.sqrt(np.asarray([0.5, 0.5]))
    diagnostics = importance_sampling_diagnostics(initial, target)
    assert diagnostics["target_weight_outside_base_support"] == 0.0
    assert diagnostics["effective_sample_size_fraction"] < 1e-18
    truncated = importance_sampling_diagnostics(initial, target, zero_tolerance=1e-14)
    assert truncated["target_weight_outside_base_support"] == pytest.approx(0.5)


def test_adaptive_step_resolves_quasi_node_scale_below_linear_grid():
    matrix = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    initial = np.sqrt(np.asarray([1.0 - 1e-12, 1e-12]))
    _, report = adaptive_local_imaginary_time_step(
        matrix,
        initial,
        minimum_ess_fraction=0.5,
        maximum_tau=0.25,
        grid_size=129,
    )
    assert 0.0 < report["tau"] < 0.25 / 64.0
    assert report["energy_after"] < report["energy_before"]
    assert report["effective_sample_size_fraction"] >= 0.5
