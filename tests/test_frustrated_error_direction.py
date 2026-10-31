import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.benchmarks.exact_diagonalization import (
    heisenberg_sz_matrix,
    spin_squared_matrix,
)
from su2_nqs.benchmarks.frustrated_error_direction import (
    apply_full_heisenberg_graph,
    first_crossing_interpolation,
    four_spin_multiplicity_operators,
    full_computational_configurations,
    full_heisenberg_graph_action,
    natural_observable_scales,
    normalized_observable_error,
    normalized_response_direction,
    projected_summed_local_operator,
    rectangular_four_site_cells,
    response_parallel_error,
)
from su2_nqs.su2.fusion_path import enumerate_paths


def test_full_graph_action_matches_dense_computational_reference() -> None:
    n = 6
    bonds = ((0, 1, 0.7), (1, 4, -0.2), (2, 5, 1.1))
    action = full_heisenberg_graph_action(n, bonds)
    dense, basis = heisenberg_sz_matrix(n, bonds)
    assert basis == tuple(range(1 << n))
    rng = np.random.default_rng(12)
    state = rng.normal(size=1 << n) + 1.0j * rng.normal(size=1 << n)
    actual = np.asarray(apply_full_heisenberg_graph(action, jnp.asarray(state)))
    assert np.max(np.abs(actual - dense @ state)) < 2.0e-14
    assert np.array_equal(
        full_computational_configurations(3),
        np.asarray(
            [
                [(integer >> (2 - site)) & 1 for site in range(3)]
                for integer in range(8)
            ],
            dtype=np.int32,
        ),
    )


def test_four_spin_multiplicity_generators_are_exact_su2_invariants() -> None:
    operators = four_spin_multiplicity_operators()
    horizontal = operators["horizontal"]
    vertical = operators["vertical"]
    perpendicular = operators["orthogonal_vertical"]
    projector = operators["projector"]
    x_matrix = operators["x"]
    z_matrix = operators["z"]
    spin_squared, basis = spin_squared_matrix(4)
    assert basis == tuple(range(16))
    assert np.vdot(horizontal, vertical) == pytest.approx(0.5, abs=1.0e-14)
    assert abs(np.vdot(horizontal, perpendicular)) < 1.0e-14
    assert np.linalg.norm(perpendicular) == pytest.approx(1.0, abs=1.0e-14)
    for operator in (projector, x_matrix, z_matrix):
        assert np.max(np.abs(operator - operator.conj().T)) < 1.0e-14
        assert np.max(np.abs(operator @ spin_squared - spin_squared @ operator)) < 1.0e-13
    assert np.max(np.abs(x_matrix @ x_matrix - projector)) < 1.0e-14
    assert np.max(np.abs(z_matrix @ z_matrix - projector)) < 1.0e-14
    assert np.max(np.abs(x_matrix @ z_matrix + z_matrix @ x_matrix)) < 1.0e-14


def test_projected_local_operator_closes_in_global_singlet_sector() -> None:
    n = 6
    transform = coupled_basis_matrix(enumerate_paths(n, 0), 0, 0)
    cells = ((0, 1, 2, 3), (2, 3, 4, 5))
    for name in ("x", "z"):
        matrix, residual = projected_summed_local_operator(
            transform, four_spin_multiplicity_operators()[name], cells
        )
        assert residual < 2.0e-14
        assert np.max(np.abs(matrix - matrix.conj().T)) < 2.0e-14
    assert rectangular_four_site_cells(2, 3) == ((0, 1, 3, 4), (1, 2, 4, 5))


def test_error_response_and_first_crossing_are_deterministic() -> None:
    reference = {"a": 1.0, "b": 2.0}
    scales = {"a": 2.0, "b": 4.0}
    error = normalized_observable_error({"a": 1.2, "b": 1.6}, reference, scales)
    assert tuple(error["names"]) == ("a", "b")
    assert np.asarray(error["signed"]) == pytest.approx([0.1, -0.1])
    assert error["rms"] == pytest.approx(0.1)
    response = normalized_response_direction(
        {"a": 0.8, "b": 2.2},
        {"a": 1.2, "b": 1.8},
        scales,
        coupling_step=0.1,
    )
    assert np.linalg.norm(response) == pytest.approx(1.0)
    assert response_parallel_error(error["signed"], response) > 0.0
    trajectory = [
        {
            "step": 0,
            "conditional_singlet_fidelity": 0.8,
            "observable_error_rms": 0.3,
            "response_parallel_error": 0.2,
            "elapsed_seconds": 0.0,
        },
        {
            "step": 10,
            "conditional_singlet_fidelity": 0.9,
            "observable_error_rms": 0.1,
            "response_parallel_error": 0.05,
            "elapsed_seconds": 2.0,
        },
    ]
    crossing = first_crossing_interpolation(trajectory, 0.85)
    assert crossing is not None
    assert crossing["interpolation_fraction"] == pytest.approx(0.5)
    assert crossing["observable_error_rms"] == pytest.approx(0.2)
    assert first_crossing_interpolation(trajectory, 0.95) is None


def test_direction_averaged_structure_factor_uses_composite_spectral_width() -> None:
    diagonal = lambda values: np.diag(np.asarray(values, dtype=np.float64))
    square = {
        "neel": diagonal((0.0, 1.0, 3.0)),
        "stripe_x": diagonal((0.0, 2.0, 0.0)),
        "stripe_y": diagonal((0.0, 0.0, 4.0)),
        "b1g": diagonal((0.0, 1.0, 2.0)),
        "columnar_x": diagonal((0.0, 2.0, 1.0)),
        "columnar_y": diagonal((0.0, 3.0, 1.0)),
    }
    square_scales = natural_observable_scales("square", square, n_sites=4)
    assert square_scales["stripe_m2_mean"] == pytest.approx(2.0)
    assert square_scales["b1g_mean"] == pytest.approx(0.5)

    triangular = {
        "k": diagonal((0.0, 1.0, 3.0)),
        "m1": diagonal((0.0, 3.0, 0.0)),
        "m2": diagonal((0.0, 0.0, 3.0)),
        "m3": diagonal((0.0, 1.5, 0.0)),
        "nematic_1": diagonal((0.0, 2.0, 1.0)),
        "nematic_2": diagonal((0.0, 3.0, 1.0)),
        "chirality": diagonal((0.0, 1.0, 2.0)),
    }
    triangular_scales = natural_observable_scales(
        "triangular",
        triangular,
        n_sites=4,
        triangular_plaquette_count=2,
    )
    assert triangular_scales["stripe_m2_mean"] == pytest.approx(1.5)
    assert triangular_scales["chirality_mean"] == pytest.approx(1.0)
