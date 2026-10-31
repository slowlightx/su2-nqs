from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.models.spin_clebsch_tree import (
    SpinClebschTreeConfig,
    init_spin_clebsch_tree,
    spin_clebsch_tree_parameter_count,
)
from su2_nqs.models.spin_control_tree import (
    SpinFullTreeConfig,
    SpinU1TreeConfig,
    init_spin_full_tree,
    init_spin_u1_tree,
    spin_control_parameter_budget_bracket,
    spin_control_parameter_match,
    spin_control_tree_parameter_count,
    spin_full_tree_amplitude,
    spin_full_tree_amplitudes,
    spin_full_tree_layouts,
    spin_full_tree_parameter_count_from_config,
    spin_u1_tree_amplitude,
    spin_u1_tree_amplitudes,
    spin_u1_tree_layouts,
    spin_u1_tree_parameter_count_from_config,
)
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.fusion_tree import FusionNode, Leaf, TreeNode


jax.config.update("jax_enable_x64", True)


def _full_basis(n_sites: int) -> np.ndarray:
    return np.asarray(
        list(itertools.product((0, 1), repeat=n_sites)), dtype=np.int32
    )


def _balanced_nodes(nodes: tuple[TreeNode, ...]) -> TreeNode:
    if len(nodes) == 1:
        return nodes[0]
    middle = len(nodes) // 2
    return FusionNode(
        _balanced_nodes(nodes[:middle]), _balanced_nodes(nodes[middle:])
    )


def _triangular_eighteen_site_cell_tree() -> FusionTree:
    cells = tuple(
        FusionNode(FusionNode(Leaf(3 * cell), Leaf(3 * cell + 1)), Leaf(3 * cell + 2))
        for cell in range(6)
    )
    return FusionTree(_balanced_nodes(cells))


def test_u1_control_has_exact_fixed_magnetization_but_not_exact_total_spin() -> None:
    n_sites = 4
    basis = _full_basis(n_sites)
    config = SpinU1TreeConfig(
        FusionTree.balanced(n_sites),
        site_order=(2, 0, 3, 1),
        target_m2=0,
        feature_dim=3,
    )
    params = init_spin_u1_tree(jax.random.PRNGKey(81), config)
    amplitudes = np.asarray(
        spin_u1_tree_amplitudes(params, jnp.asarray(basis), config)
    )
    magnetizations = 2 * basis.sum(axis=1) - n_sites
    assert np.linalg.norm(amplitudes[magnetizations == 0]) > 1.0e-12
    np.testing.assert_array_equal(amplitudes[magnetizations != 0], 0.0)

    fixed_basis = basis[magnetizations == 0]
    integers = tuple(
        int(sum(bit << (n_sites - site - 1) for site, bit in enumerate(row)))
        for row in fixed_basis
    )
    spin_squared, reference_integers = spin_squared_matrix(n_sites, n_up=2)
    positions = {state: index for index, state in enumerate(reference_integers)}
    order = [positions[state] for state in integers]
    spin_squared = spin_squared[np.ix_(order, order)]
    fixed_amplitudes = amplitudes[magnetizations == 0]
    relative_residual = np.linalg.norm(spin_squared @ fixed_amplitudes) / np.linalg.norm(
        fixed_amplitudes
    )
    assert relative_residual > 1.0e-4


def test_u1_intermediate_cutoff_is_exact_and_keeps_the_requested_root() -> None:
    tree = FusionTree.balanced(4)
    config = SpinU1TreeConfig(
        tree,
        target_m2=0,
        feature_dim=2,
        maximum_intermediate_abs_m2=0,
    )
    params = init_spin_u1_tree(jax.random.PRNGKey(88), config)
    # Each two-site child has M2=0, so this configuration survives.
    assert abs(float(spin_u1_tree_amplitude(params, jnp.asarray((0, 1, 0, 1)), config))) > 0.0
    # Total M2 remains zero, but the left child has M2=-2 and is cut out.
    assert float(spin_u1_tree_amplitude(params, jnp.asarray((0, 0, 1, 1)), config)) == 0.0


def test_full_control_has_support_in_multiple_magnetization_sectors() -> None:
    n_sites = 4
    basis = _full_basis(n_sites)
    config = SpinFullTreeConfig(
        FusionTree.balanced(n_sites),
        site_order=(1, 3, 0, 2),
        feature_dim=3,
    )
    params = init_spin_full_tree(jax.random.PRNGKey(92), config)
    amplitudes = np.asarray(
        spin_full_tree_amplitudes(params, jnp.asarray(basis), config)
    )
    magnetizations = 2 * basis.sum(axis=1) - n_sites
    occupied_sectors = {
        int(m2)
        for m2 in np.unique(magnetizations)
        if np.linalg.norm(amplitudes[magnetizations == m2]) > 1.0e-12
    }
    assert occupied_sectors == {-4, -2, 0, 2, 4}


@pytest.mark.parametrize("control", ("u1", "full"))
def test_site_order_is_an_explicit_permutation_of_the_same_tree(
    control: str,
) -> None:
    tree = FusionTree.from_nested(((0, 1), (2, (3, 4))))
    order = (3, 0, 4, 1, 2)
    states = jnp.asarray(_full_basis(5))
    if control == "u1":
        permuted_config = SpinU1TreeConfig(
            tree, order, target_m2=1, feature_dim=2
        )
        identity_config = SpinU1TreeConfig(
            tree, tuple(range(5)), target_m2=1, feature_dim=2
        )
        params = init_spin_u1_tree(jax.random.PRNGKey(13), permuted_config)
        actual = spin_u1_tree_amplitudes(params, states, permuted_config)
        expected = spin_u1_tree_amplitudes(
            params, states[:, jnp.asarray(order)], identity_config
        )
    else:
        permuted_config = SpinFullTreeConfig(tree, order, feature_dim=2)
        identity_config = SpinFullTreeConfig(
            tree, tuple(range(5)), feature_dim=2
        )
        params = init_spin_full_tree(jax.random.PRNGKey(14), permuted_config)
        actual = spin_full_tree_amplitudes(params, states, permuted_config)
        expected = spin_full_tree_amplitudes(
            params, states[:, jnp.asarray(order)], identity_config
        )
    np.testing.assert_allclose(actual, expected, atol=1.0e-14, rtol=0.0)


def test_equal_child_sizes_share_merge_parameters_for_both_controls() -> None:
    tree = FusionTree.balanced(8)
    u1 = SpinU1TreeConfig(tree, feature_dim=2)
    full = SpinFullTreeConfig(tree, feature_dim=2)
    u1_pair_indices = {
        layout.parameter_index
        for layout in spin_u1_tree_layouts(u1)
        if (layout.left_size, layout.right_size) == (1, 1)
    }
    full_pair_indices = {
        layout.parameter_index
        for layout in spin_full_tree_layouts(full)
        if (layout.left_size, layout.right_size) == (1, 1)
    }
    assert len(u1_pair_indices) == 1
    assert len(full_pair_indices) == 1


def test_counts_and_parameter_budget_brackets_are_auditable() -> None:
    tree = FusionTree.balanced(8)
    order = tuple(range(8))
    su2_config = SpinClebschTreeConfig(
        tree,
        target_q=0,
        target_m2=0,
        feature_dim=2,
        maximum_intermediate_q=4,
    )
    su2_params = init_spin_clebsch_tree(jax.random.PRNGKey(1), su2_config)
    target = spin_clebsch_tree_parameter_count(su2_params)

    for control in ("u1", "full"):
        bracket = spin_control_parameter_budget_bracket(
            control=control,
            tree=tree,
            site_order=order,
            target_parameter_count=target,
            maximum_feature_dim=16,
        )
        assert bracket.lower is not None
        assert bracket.upper is not None
        assert bracket.lower.parameter_count <= target
        assert bracket.upper.parameter_count >= target
        if bracket.lower != bracket.upper:
            assert bracket.lower.feature_dim + 1 == bracket.upper.feature_dim

    u1_config = SpinU1TreeConfig(tree, order, feature_dim=2)
    full_config = SpinFullTreeConfig(tree, order, feature_dim=2)
    u1_params = init_spin_u1_tree(jax.random.PRNGKey(2), u1_config)
    full_params = init_spin_full_tree(jax.random.PRNGKey(3), full_config)
    assert spin_control_tree_parameter_count(u1_params) == (
        spin_u1_tree_parameter_count_from_config(u1_config)
    )
    assert spin_control_tree_parameter_count(full_params) == (
        spin_full_tree_parameter_count_from_config(full_config)
    )
    # Equal hidden width is a useful structural control, not an equal-P arm.
    assert (
        spin_control_tree_parameter_count(u1_params)
        != spin_control_tree_parameter_count(full_params)
    )


def test_n18_q4_has_a_preregisterable_five_percent_raw_parameter_triplet() -> None:
    tree = _triangular_eighteen_site_cell_tree()
    order = tuple(range(18))
    su2_config = SpinClebschTreeConfig(
        tree,
        target_q=0,
        target_m2=0,
        feature_dim=7,
        maximum_intermediate_q=4,
    )
    su2_count = spin_clebsch_tree_parameter_count(
        init_spin_clebsch_tree(jax.random.PRNGKey(101), su2_config)
    )
    u1_config = SpinU1TreeConfig(
        tree,
        order,
        target_m2=0,
        feature_dim=6,
        maximum_intermediate_abs_m2=4,
    )
    full_config = SpinFullTreeConfig(tree, order, feature_dim=13)
    assert su2_count == 7_028
    assert spin_u1_tree_parameter_count_from_config(u1_config) == 7_158
    assert spin_full_tree_parameter_count_from_config(full_config) == 6_994

    u1_match = spin_control_parameter_match(
        control="u1",
        tree=tree,
        site_order=order,
        target_parameter_count=su2_count,
        maximum_intermediate_abs_m2=4,
        maximum_feature_dim=16,
    )
    full_match = spin_control_parameter_match(
        control="full",
        tree=tree,
        site_order=order,
        target_parameter_count=su2_count,
        maximum_feature_dim=16,
    )
    assert u1_match.within_tolerance
    assert full_match.within_tolerance
    assert (u1_match.closest.feature_dim, u1_match.closest.parameter_count) == (
        6,
        7_158,
    )
    assert (full_match.closest.feature_dim, full_match.closest.parameter_count) == (
        13,
        6_994,
    )

    # The existing low-capacity q4 D=2 arm has P=178.  Integer widths leave
    # both controls much farther than 5%; it must remain a width/Pareto arm.
    low_su2_count = spin_clebsch_tree_parameter_count(
        init_spin_clebsch_tree(
            jax.random.PRNGKey(102),
            SpinClebschTreeConfig(
                tree,
                target_q=0,
                target_m2=0,
                feature_dim=2,
                maximum_intermediate_q=4,
            ),
        )
    )
    assert low_su2_count == 178
    low_u1 = spin_control_parameter_match(
        control="u1",
        tree=tree,
        site_order=order,
        target_parameter_count=low_su2_count,
        maximum_intermediate_abs_m2=4,
        maximum_feature_dim=16,
    )
    low_full = spin_control_parameter_match(
        control="full",
        tree=tree,
        site_order=order,
        target_parameter_count=low_su2_count,
        maximum_feature_dim=16,
    )
    assert not low_u1.within_tolerance
    assert not low_full.within_tolerance


@pytest.mark.parametrize("control", ("u1", "full"))
def test_control_amplitudes_are_jittable_and_differentiable(control: str) -> None:
    tree = FusionTree.balanced(4)
    state = jnp.asarray((0, 1, 0, 1), dtype=jnp.int32)
    if control == "u1":
        config = SpinU1TreeConfig(tree, target_m2=0, feature_dim=2)
        params = init_spin_u1_tree(jax.random.PRNGKey(21), config)
        amplitude = spin_u1_tree_amplitude
    else:
        config = SpinFullTreeConfig(tree, feature_dim=2)
        params = init_spin_full_tree(jax.random.PRNGKey(22), config)
        amplitude = spin_full_tree_amplitude

    compiled = jax.jit(lambda candidate, row: amplitude(candidate, row, config))
    value = compiled(params, state)
    gradients = jax.grad(lambda candidate: compiled(candidate, state) ** 2)(params)
    assert np.isfinite(float(value))
    assert all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(gradients)
    )


def test_invalid_configs_and_configurations_fail_closed() -> None:
    tree = FusionTree.balanced(4)
    with pytest.raises(ValueError, match="site_order"):
        SpinU1TreeConfig(tree, (0, 1, 1, 3))
    with pytest.raises(ValueError, match="unreachable"):
        SpinU1TreeConfig(tree, target_m2=1)
    with pytest.raises(ValueError, match="positive integer"):
        SpinFullTreeConfig(tree, feature_dim=0)
    with pytest.raises(ValueError, match="non-negative doubled integer"):
        SpinU1TreeConfig(tree, maximum_intermediate_abs_m2=-1)

    u1 = SpinU1TreeConfig(tree, target_m2=0, feature_dim=2)
    u1_params = init_spin_u1_tree(jax.random.PRNGKey(31), u1)
    full = SpinFullTreeConfig(tree, feature_dim=2)
    full_params = init_spin_full_tree(jax.random.PRNGKey(32), full)
    invalid = jnp.asarray((0, 1, 2, 0))
    assert float(spin_u1_tree_amplitude(u1_params, invalid, u1)) == 0.0
    assert float(spin_full_tree_amplitude(full_params, invalid, full)) == 0.0
