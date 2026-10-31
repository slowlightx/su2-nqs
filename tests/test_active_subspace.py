from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.frontier.triangle_patch_physical_init import (
    init_triangle_patch_cluster_singlet,
)
from su2_nqs.models.triangle_patch_clebsch_tree import (
    triangle_patch_6x6_canary_plan,
    triangle_patch_amplitudes,
    triangle_root_layout,
    triangle_shared_reduced_layout,
)
from su2_nqs.vmc.active_subspace import ActiveParameterSubspace
from su2_nqs.vmc.active_subspace import (
    triangle_patch_low_spin_upper_tree_mask,
    triangle_patch_low_spin_upper_tree_subspace,
)


jax.config.update("jax_enable_x64", True)


def test_active_subspace_reconstructs_only_selected_entries_and_is_differentiable() -> None:
    reference = {
        "a": jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        "b": (
            jnp.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=jnp.float64),
        ),
    }
    mask = {
        "a": np.asarray([False, True]),
        "b": (np.asarray([[True, False], [False, True]]),),
    }
    subspace = ActiveParameterSubspace.from_mask(reference, mask)
    assert subspace.dimension == 3
    assert subspace.total_parameter_count == 6
    np.testing.assert_allclose(subspace.pack(reference), [2.0, 3.0, 6.0])

    vector = jnp.asarray([-2.0, -3.0, -6.0], dtype=jnp.float64)
    rebuilt = jax.jit(subspace.inject)(vector)
    np.testing.assert_array_equal(rebuilt["a"], [1.0, -2.0])
    np.testing.assert_array_equal(rebuilt["b"][0], [[-3.0, 4.0], [5.0, -6.0]])
    np.testing.assert_array_equal(subspace.pack(rebuilt), vector)

    tangent = subspace.tangent(jnp.ones(3, dtype=jnp.float64))
    np.testing.assert_array_equal(tangent["a"], [0.0, 1.0])
    np.testing.assert_array_equal(tangent["b"][0], [[1.0, 0.0], [0.0, 1.0]])
    gradient = jax.grad(
        lambda active: sum(
            jnp.sum(leaf**2)
            for leaf in jax.tree_util.tree_leaves(subspace.inject(active))
        )
    )(vector)
    np.testing.assert_allclose(gradient, 2.0 * vector, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("geometry", "ratio"),
    (("square", 0.5), ("triangular", 0.125)),
)
def test_n36_low_spin_upper_tree_chart_freezes_local_core_and_has_46_coordinates(
    geometry: str,
    ratio: float,
) -> None:
    plan = triangle_patch_6x6_canary_plan(geometry)
    reference = init_triangle_patch_cluster_singlet(
        jax.random.PRNGKey(1201),
        plan,
        j2_over_j1=ratio,
        background_scale=0.05,
    )
    subspace = triangle_patch_low_spin_upper_tree_subspace(
        reference, plan, stage="complete"
    )
    # Five q_left,q_right,q_parent <= 2 blocks, each 2x2x2, plus two
    # 2x2 root blocks, minus the reduced-core and root-scale anchors.
    assert subspace.dimension == 5 * 8 + 2 * 4 - 2 == 46
    assert subspace.total_parameter_count == 1252

    initial = subspace.pack(reference)
    round_trip = subspace.inject(initial)
    assert jax.tree_util.tree_structure(round_trip) == jax.tree_util.tree_structure(
        reference
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(round_trip),
        jax.tree_util.tree_leaves(reference),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    displacement = jnp.linspace(-0.03, 0.04, 46, dtype=jnp.float64)
    changed = subspace.inject(initial + displacement)
    tangent = subspace.tangent(jnp.ones(46, dtype=jnp.float64))

    for name in ("patch_pair", "patch_gates", "reduced_gates"):
        for actual, expected in zip(changed[name], reference[name], strict=True):
            np.testing.assert_array_equal(actual, expected)
        for block in tangent[name]:
            np.testing.assert_array_equal(block, np.zeros(block.shape))

    reduced_positions = {
        layout.key: index
        for index, layout in enumerate(triangle_shared_reduced_layout(plan))
    }
    core = reduced_positions[(0, 0, 0)]
    np.testing.assert_array_equal(
        changed["reduced"][core][0, 0, 0],
        reference["reduced"][core][0, 0, 0],
    )
    assert float(tangent["reduced"][core][0, 0, 0]) == 0.0
    np.testing.assert_array_equal(
        changed["root"][0][0, 0], reference["root"][0][0, 0]
    )
    assert float(tangent["root"][0][0, 0]) == 0.0
    for index, layout in enumerate(triangle_shared_reduced_layout(plan)):
        if max(layout.key) > 2:
            np.testing.assert_array_equal(
                tangent["reduced"][index],
                np.zeros(tangent["reduced"][index].shape),
            )
    for index, layout in enumerate(triangle_root_layout(plan)):
        if layout.q > 2:
            np.testing.assert_array_equal(
                tangent["root"][index],
                np.zeros(tangent["root"][index].shape),
            )

    rng = np.random.default_rng(1202)
    configurations = []
    for _ in range(8):
        row = np.zeros(36, dtype=np.int32)
        row[rng.choice(36, 18, replace=False)] = 1
        configurations.append(row)
    states = jnp.asarray(configurations)
    values, directional = jax.jvp(
        lambda active: triangle_patch_amplitudes(
            subspace.inject(active), states, plan
        ),
        (initial,),
        (jnp.ones(46, dtype=jnp.float64),),
    )
    assert np.all(np.isfinite(np.asarray(values)))
    assert np.all(np.isfinite(np.asarray(directional)))
    assert np.linalg.norm(np.asarray(directional)) > 1.0e-12


@pytest.mark.parametrize(
    ("geometry", "ratio"),
    (("square", 0.5), ("triangular", 0.125)),
)
def test_n36_phase_a_selector_has_18_individually_visible_directions(
    geometry: str,
    ratio: float,
) -> None:
    plan = triangle_patch_6x6_canary_plan(geometry)
    reference = init_triangle_patch_cluster_singlet(
        jax.random.PRNGKey(1201),
        plan,
        j2_over_j1=ratio,
        background_scale=0.05,
    )
    first = triangle_patch_low_spin_upper_tree_subspace(
        reference, plan, stage="first_order"
    )
    complete_mask = triangle_patch_low_spin_upper_tree_mask(
        reference, plan, stage="complete"
    )
    first_mask = triangle_patch_low_spin_upper_tree_mask(
        reference, plan, stage="first_order"
    )
    assert first.dimension == 18
    for selected, allowed in zip(
        jax.tree_util.tree_leaves(first_mask),
        jax.tree_util.tree_leaves(complete_mask),
        strict=True,
    ):
        assert np.all(np.asarray(selected) <= np.asarray(allowed))

    # The structural selector is exact and sample free.  This independent
    # finite probe uses a frozen generic M=0 set only to catch implementation
    # mistakes in its graph traversal.
    rng = np.random.default_rng(22)
    configurations = []
    for _ in range(256):
        row = np.zeros(36, dtype=np.int32)
        row[rng.choice(36, 18, replace=False)] = 1
        configurations.append(row)
    states = jnp.asarray(configurations)
    initial = first.pack(reference)
    jacobian = np.asarray(
        jax.jacfwd(
            lambda active: triangle_patch_amplitudes(
                first.inject(active), states, plan
            )
        )(initial)
    )
    assert jacobian.shape == (256, 18)
    column_norms = np.linalg.norm(jacobian, axis=0)
    assert np.all(np.isfinite(column_norms))
    assert np.all(column_norms > 0.0)
    normalized = jacobian / column_norms[None, :]
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    assert singular_values[-1] > 2.0e-2


@pytest.mark.parametrize(
    ("geometry", "ratio"),
    (("square", 0.5), ("triangular", 0.125)),
)
def test_n36_q0_control_is_a_ten_dimensional_subset_of_q0_q2_phase_a(
    geometry: str,
    ratio: float,
) -> None:
    plan = triangle_patch_6x6_canary_plan(geometry)
    reference = init_triangle_patch_cluster_singlet(
        jax.random.PRNGKey(1201),
        plan,
        j2_over_j1=ratio,
        background_scale=0.05,
    )
    q0 = triangle_patch_low_spin_upper_tree_subspace(
        reference,
        plan,
        stage="first_order",
        maximum_active_q=0,
    )
    q0_q2 = triangle_patch_low_spin_upper_tree_subspace(
        reference,
        plan,
        stage="first_order",
        maximum_active_q=2,
    )
    assert q0.dimension == 10
    assert q0_q2.dimension == 18
    assert set(q0.active_flat_indices) < set(q0_q2.active_flat_indices)


def test_low_spin_selector_rejects_unregistered_active_cutoff() -> None:
    plan = triangle_patch_6x6_canary_plan("square")
    reference = init_triangle_patch_cluster_singlet(
        jax.random.PRNGKey(1201), plan, j2_over_j1=0.5
    )
    with pytest.raises(ValueError, match="doubled spin 0 or 2"):
        triangle_patch_low_spin_upper_tree_mask(
            reference, plan, maximum_active_q=4
        )


def test_active_subspace_fails_closed_on_ambiguous_masks_and_coordinates() -> None:
    reference = {"x": jnp.ones((2,), dtype=jnp.float64)}
    with pytest.raises(TypeError, match="boolean"):
        ActiveParameterSubspace.from_mask(reference, {"x": np.asarray([1, 0])})
    with pytest.raises(ValueError, match="at least one"):
        ActiveParameterSubspace.from_mask(
            reference, {"x": np.asarray([False, False])}
        )
    with pytest.raises(ValueError, match="wrong shape"):
        ActiveParameterSubspace.from_mask(
            reference, {"x": np.asarray([True, False, True])}
        )
    subspace = ActiveParameterSubspace.from_mask(
        reference, {"x": np.asarray([True, False])}
    )
    with pytest.raises(ValueError, match="shape"):
        subspace.inject(jnp.ones((2,), dtype=jnp.float64))
    with pytest.raises(TypeError, match="dtype"):
        subspace.inject(jnp.ones((1,), dtype=jnp.float32))
