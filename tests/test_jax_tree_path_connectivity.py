import jax
import numpy as np
import pytest

from su2_nqs.operators.jax_tree_path_connectivity import (
    compile_jax_tree_connectivity_plan,
    jax_coalesce_tree_connectivity,
    jax_tree_bond_connectivity,
)
from su2_nqs.operators.tree_path_connectivity import (
    tree_pair_heisenberg_connectivity,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout


jax.config.update("jax_enable_x64", True)


def _combined(labels, coefficients, valid):
    result = {}
    for row, value, retain in zip(labels, coefficients, valid, strict=True):
        if retain:
            key = tuple(int(entry) for entry in row)
            result[key] = result.get(key, 0.0) + float(value)
    return {key: value for key, value in result.items() if abs(value) > 1.0e-13}


def test_device_bond_columns_match_independent_tree_reference_exhaustively():
    layout = rectangular_four_spin_layout(2, 2, triangular_coordinates=True)
    physical_bonds = (
        (0, 1, 0.7),
        (1, 3, -0.2),
        (3, 2, 1.1),
        (2, 0, 0.4),
        (0, 3, -0.35),
    )
    model_bonds = layout.remap_bonds(physical_bonds)
    plan = compile_jax_tree_connectivity_plan(
        layout.tree, model_bonds, target_q=0
    )
    kernel = jax.jit(
        lambda labels, bond: jax_tree_bond_connectivity(
            labels, plan, bond, candidate_capacity=8
        )
    )
    for state in layout.tree.enumerate_states(0):
        center = np.asarray(state.labels, dtype=np.int32)
        for bond_index, (first, second, coupling) in enumerate(model_bonds):
            labels, coefficients, valid, overflow = jax.device_get(
                kernel(center, bond_index)
            )
            assert not bool(overflow)
            actual = _combined(labels, coefficients, valid)
            expected_states, expected_values = tree_pair_heisenberg_connectivity(
                layout.tree, state, first, second
            )
            expected = {
                candidate.labels: coupling * float(value)
                for candidate, value in zip(
                    expected_states, expected_values, strict=True
                )
                if abs(coupling * float(value)) > 1.0e-13
            }
            assert actual.keys() == expected.keys()
            np.testing.assert_allclose(
                [actual[key] for key in sorted(actual)],
                [expected[key] for key in sorted(expected)],
                atol=4.0e-13,
            )


def test_device_candidate_overflow_is_reported_instead_of_truncated_silently():
    layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=True)
    physical_bonds = ((0, 10, 1.0),)
    model_bonds = layout.remap_bonds(physical_bonds)
    plan = compile_jax_tree_connectivity_plan(
        layout.tree, model_bonds, target_q=0
    )
    state = layout.tree.enumerate_states(0)[-1]
    _, _, _, overflow = jax_tree_bond_connectivity(
        np.asarray(state.labels, dtype=np.int32),
        plan,
        0,
        candidate_capacity=1,
    )
    assert bool(np.asarray(overflow))


def test_device_coalescing_is_collision_free_and_reports_output_overflow():
    labels = np.asarray(
        [[1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0]], dtype=np.int32
    )
    coefficients = np.asarray([0.25, 2.0, -0.5, 99.0])
    valid = np.asarray([True, True, True, False])
    compact = jax_coalesce_tree_connectivity(
        labels, coefficients, valid, n_sites=4, output_capacity=2
    )
    rows, values, retain, count, overflow = jax.device_get(compact)
    assert int(count) == 2
    assert not bool(overflow)
    actual = _combined(rows, values, retain)
    assert actual == {(0, 1, 0): 2.0, (1, 0, 1): -0.25}
    *_, overflow = jax_coalesce_tree_connectivity(
        labels, coefficients, valid, n_sites=4, output_capacity=1
    )
    assert bool(np.asarray(overflow))


@pytest.mark.parametrize("tolerance", [-1.0, np.inf, np.nan])
def test_device_coalescing_rejects_invalid_coefficient_tolerance(tolerance):
    with pytest.raises(ValueError, match="finite and non-negative"):
        jax_coalesce_tree_connectivity(
            np.zeros((1, 3), dtype=np.int32),
            np.ones((1,), dtype=np.float64),
            np.ones((1,), dtype=bool),
            n_sites=4,
            output_capacity=1,
            coefficient_tolerance=tolerance,
        )
