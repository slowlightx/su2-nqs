from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import (
    fixed_m_spin_basis,
    physical_permutation_index_table,
    reynolds_project_state,
)
from su2_nqs.benchmarks.spin_sector import spin_squared_sz_sparse
from su2_nqs.lattices.geometry_tree import morton_order
from su2_nqs.lattices.reduced_fusion_catalogs import diamond_fusion_catalog
from su2_nqs.lattices.spatial_symmetry import rectangular_translation_group
from su2_nqs.models.persistent_route_pair_dag import (
    SpinPersistentRoutePairConfig,
    select_group_closed_persistent_pair_supports,
    spin_persistent_route_pair_dag_amplitudes,
    spin_route_pair_effective_kernels,
)
from su2_nqs.models.reduced_fusion_dag import (
    SpinReducedFusionDAGConfig,
    spin_reduced_fusion_dag_amplitudes,
)
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.models.strict_residual_route_dag import (
    SpinStrictResidualRouteDAGConfig,
    init_paired_spin_strict_residual_route_dag,
    spin_strict_residual_inner_params,
    spin_strict_residual_route_dag_amplitudes,
    spin_strict_residual_route_dag_parameter_count,
)


jax.config.update("jax_enable_x64", True)


def _joint_config(rows: int = 3, columns: int = 4):
    n_sites = rows * columns
    order = morton_order(rows, columns)
    tree = diamond_fusion_catalog(order, resonating=False)
    seed = diamond_fusion_catalog(order, resonating=True)
    baseline = SpinReducedFusionDAGConfig(
        tree,
        target_q=0,
        target_m2=0,
        feature_dim=1,
        maximum_intermediate_q=2,
        route_mode="single_route",
    )
    resonance = SpinPersistentRoutePairConfig(
        seed,
        select_group_closed_persistent_pair_supports(seed),
        target_q=0,
        target_m2=0,
        feature_dim=1,
        maximum_intermediate_q=2,
        pair_mode="equivariant_full_rank",
    )
    projector = PhysicalSpatialProjector(
        rectangular_translation_group(rows, columns)
    )
    assert resonance.catalog.group == (tuple(range(n_sites)),)
    return SpinStrictResidualRouteDAGConfig(
        baseline, resonance, projector
    )


def _efficient_vector(params, config, basis, indices):
    tree_params = {"merges": params["merges"], "readout": params["readout"]}
    active = spin_strict_residual_inner_params(params, config, active=True)
    reference = spin_strict_residual_inner_params(params, config, active=False)
    tree = spin_reduced_fusion_dag_amplitudes(
        tree_params, basis, config.baseline
    )
    active_vector = spin_persistent_route_pair_dag_amplitudes(
        active, basis, config.resonance
    )
    reference_vector = spin_persistent_route_pair_dag_amplitudes(
        reference, basis, config.resonance
    )
    return sum(
        (
            reynolds_project_state(tree, indices),
            reynolds_project_state(active_vector, indices),
            -reynolds_project_state(reference_vector, indices),
        )
    )


def test_zero_residual_is_exact_projected_tree_for_arbitrary_backbone() -> None:
    config = _joint_config()
    _, _, params = init_paired_spin_strict_residual_route_dag(
        jax.random.PRNGKey(19), config
    )
    # The cancellation must not rely on the random initializer or a special
    # readout.  Alter every shared backbone leaf before comparing.
    params = {
        **params,
        "merges": tuple(
            value + 0.07 * (index + 1)
            for index, value in enumerate(params["merges"])
        ),
        "readout": params["readout"] - 0.13,
    }
    basis_tuple = fixed_m_spin_basis(12, 0)
    basis = jnp.asarray(basis_tuple, dtype=jnp.int32)
    indices = physical_permutation_index_table(
        basis_tuple, config.projector.site_permutations
    )
    observed = _efficient_vector(params, config, basis, indices)
    tree_raw = spin_reduced_fusion_dag_amplitudes(
        {"merges": params["merges"], "readout": params["readout"]},
        basis,
        config.baseline,
    )
    expected = reynolds_project_state(tree_raw, indices)
    np.testing.assert_allclose(observed, expected, atol=2.0e-14, rtol=0.0)

    # The public per-configuration contraction uses the same global-h rule.
    direct = spin_strict_residual_route_dag_amplitudes(
        params, basis[:7], config
    )
    np.testing.assert_allclose(direct, expected[:7], atol=2.0e-14, rtol=0.0)

    # The active/reference DAG backbone derivatives cancel as strictly as
    # their values: only the projected-tree tangent may remain at delta=0.
    direction = jax.tree_util.tree_map(jnp.zeros_like, params)
    direction = {
        **direction,
        "merges": tuple(
            jax.random.normal(jax.random.PRNGKey(500 + index), value.shape)
            for index, value in enumerate(params["merges"])
        ),
        "readout": jax.random.normal(
            jax.random.PRNGKey(599), params["readout"].shape
        ),
    }
    _, observed_tangent = jax.jvp(
        lambda candidate: _efficient_vector(
            candidate, config, basis, indices
        ),
        (params,),
        (direction,),
    )

    def projected_tree(candidate):
        raw = spin_reduced_fusion_dag_amplitudes(
            {
                "merges": candidate["merges"],
                "readout": candidate["readout"],
            },
            basis,
            config.baseline,
        )
        return reynolds_project_state(raw, indices)

    _, expected_tangent = jax.jvp(
        projected_tree, (params,), (direction,)
    )
    np.testing.assert_allclose(
        observed_tangent, expected_tangent, atol=3.0e-14, rtol=2.0e-12
    )


def test_zero_residual_has_nonzero_first_order_route_tangent_and_is_singlet() -> None:
    config = _joint_config()
    _, _, params = init_paired_spin_strict_residual_route_dag(
        jax.random.PRNGKey(23), config
    )
    basis_tuple = fixed_m_spin_basis(12, 0)
    basis = jnp.asarray(basis_tuple, dtype=jnp.int32)
    indices = jnp.asarray(
        physical_permutation_index_table(
            basis_tuple, config.projector.site_permutations
        )
    )
    direction = jax.tree_util.tree_map(jnp.zeros_like, params)
    direction = {
        **direction,
        "pair_route_deltas": tuple(
            jax.random.normal(jax.random.PRNGKey(100 + index), value.shape)
            for index, value in enumerate(params["pair_route_deltas"])
        ),
    }
    value, tangent = jax.jvp(
        lambda candidate: _efficient_vector(
            candidate, config, basis, indices
        ),
        (params,),
        (direction,),
    )
    tangent_ratio = float(jnp.linalg.norm(tangent) / jnp.linalg.norm(value))
    assert tangent_ratio > 1.0e-3

    normalized = np.asarray(value / jnp.linalg.norm(value))
    spin_squared, spin_basis = spin_squared_sz_sparse(12, n_up=6)
    assert tuple(spin_basis) == basis_tuple
    assert np.linalg.norm(spin_squared @ normalized) < 2.0e-10


def test_factorized_and_joint_start_common_but_have_different_rank_capacity() -> None:
    joint_config = _joint_config(4, 4)
    factor_config, factor, joint = init_paired_spin_strict_residual_route_dag(
        jax.random.PRNGKey(29), joint_config
    )
    assert spin_strict_residual_route_dag_parameter_count(factor) == 55
    assert spin_strict_residual_route_dag_parameter_count(joint) == 55
    assert factor["merges"] is joint["merges"]
    assert factor["readout"] is joint["readout"]

    factor_inner = spin_strict_residual_inner_params(
        factor, factor_config, active=True
    )
    joint_inner = spin_strict_residual_inner_params(
        joint, joint_config, active=True
    )
    factor_kernels = spin_route_pair_effective_kernels(
        factor_inner, factor_config.resonance
    )
    joint_kernels = spin_route_pair_effective_kernels(
        joint_inner, joint_config.resonance
    )
    for factor_kernel, joint_kernel in zip(
        factor_kernels, joint_kernels, strict=True
    ):
        np.testing.assert_allclose(
            factor_kernel, joint_kernel, atol=2.0e-15, rtol=0.0
        )

    # A joint delta can leave the rank-one tangent manifold immediately.
    joint = {
        **joint,
        "pair_route_deltas": tuple(
            0.05
            * jax.random.normal(jax.random.PRNGKey(300 + index), value.shape)
            for index, value in enumerate(joint["pair_route_deltas"])
        ),
    }
    kernels = spin_route_pair_effective_kernels(
        spin_strict_residual_inner_params(joint, joint_config, active=True),
        joint_config.resonance,
    )
    assert any(
        np.linalg.matrix_rank(np.asarray(kernel[channel]), tol=1.0e-12) > 1
        for kernel in kernels
        for channel in range(kernel.shape[0])
    )


def test_configuration_rejects_non_equivariant_pair_mode() -> None:
    config = _joint_config()
    with np.testing.assert_raises(ValueError):
        SpinStrictResidualRouteDAGConfig(
            config.baseline,
            replace(config.resonance, pair_mode="full_rank"),
            config.projector,
        )
