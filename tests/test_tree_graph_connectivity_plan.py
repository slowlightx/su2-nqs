import numpy as np
import pytest

import su2_nqs.operators.tree_path_connectivity as connectivity_module
from su2_nqs.operators import (
    TreeHeisenbergGraphConnectivityPlan,
    compile_tree_heisenberg_graph_connectivity_plan,
)
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.operators.tree_path_connectivity import (
    tree_heisenberg_graph_connectivity,
)
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.tree_rotations import recoupling_matrix_sparse


@pytest.mark.parametrize(
    "tree,target_q",
    (
        (FusionTree.balanced(6), 0),
        (FusionTree.balanced(6), 2),
        (FusionTree.from_nested(((0, (1, 2)), (3, (4, 5)))), 0),
    ),
)
def test_compiled_graph_plan_is_strictly_equal_to_legacy_and_dense(
    tree, target_q
):
    bonds = (
        (0, 1, 1.0),
        (1, 4, -0.37),
        (0, 5, 0.8),
        (0, 5, -0.2),
        (2, 3),
        (4, 5, 1.13),
    )
    plan = compile_tree_heisenberg_graph_connectivity_plan(
        tree, (bond for bond in bonds), target_q=target_q
    )
    assert isinstance(plan, TreeHeisenbergGraphConnectivityPlan)
    assert plan.bond_count == len(bonds)

    left = FusionTree.left_associated(tree.n_leaves)
    recoupling, _ = recoupling_matrix_sparse(left, tree, target_q)
    expected = (
        recoupling
        @ heisenberg_graph_fusion_matrix(
            tree.n_leaves, bonds, target_q=target_q
        )
        @ recoupling.T
    ).tocsc()
    states = tree.enumerate_states(target_q)
    positions = {state.labels: index for index, state in enumerate(states)}
    for column, state in enumerate(states):
        legacy_states, legacy_values = tree_heisenberg_graph_connectivity(
            tree, state, bonds, target_q=target_q
        )
        planned_states, planned_values = plan.column(state)
        assert planned_states == legacy_states
        np.testing.assert_array_equal(planned_values, legacy_values)

        actual = np.zeros(len(states), dtype=np.float64)
        for connected, value in zip(
            planned_states, planned_values, strict=True
        ):
            actual[positions[connected.labels]] = value
        np.testing.assert_allclose(
            actual,
            expected.getcol(column).toarray().ravel(),
            atol=8.0e-13,
            rtol=0.0,
        )


def test_compiled_graph_plan_caches_structure_and_validates_once(
    monkeypatch,
):
    tree = FusionTree.balanced(8)
    bonds = tuple((site, (site + 3) % 8, 0.2 + site) for site in range(8))
    plan = compile_tree_heisenberg_graph_connectivity_plan(
        tree, bonds, target_q=0
    )
    repeated = compile_tree_heisenberg_graph_connectivity_plan(
        tree, bonds, target_q=0
    )
    assert repeated is plan

    calls = 0
    original_validation = connectivity_module._validated_flat_state

    def counted_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_validation(*args, **kwargs)

    def forbidden_lca(*_args, **_kwargs):
        raise AssertionError("a compiled column must not rebuild bond LCAs")

    monkeypatch.setattr(
        connectivity_module, "_validated_flat_state", counted_validation
    )
    monkeypatch.setattr(
        connectivity_module, "_lowest_common_ancestor", forbidden_lca
    )
    connected, values = plan.column(tree.enumerate_states(0)[0])
    assert calls == 1
    assert len(connected) == len(values) > 0


def test_compiled_graph_plan_rejects_invalid_inputs_and_sector():
    tree = FusionTree.balanced(4)
    with pytest.raises(TypeError, match="endpoints must be integers"):
        compile_tree_heisenberg_graph_connectivity_plan(
            tree, ((False, 1, 1.0),), target_q=0
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        compile_tree_heisenberg_graph_connectivity_plan(
            tree, ((0, 1, 1.0),), coefficient_tolerance=-1.0
        )

    plan = compile_tree_heisenberg_graph_connectivity_plan(
        tree, ((0, 3, 1.0),), target_q=0
    )
    with pytest.raises(ValueError, match="inadmissible"):
        plan.column((1, 2, 0))
    with pytest.raises(ValueError, match="requested total-spin sector"):
        plan.column(tree.enumerate_states(2)[0])
