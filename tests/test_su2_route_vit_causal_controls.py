from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.benchmarks.support_aware_spatial_cg import spatial_cg_route_spec
from su2_nqs.models.su2_route_vit import (
    SU2RouteViTConfig,
    init_su2_route_vit,
    su2_route_vit_parameter_count,
)
from su2_nqs.models.su2_route_vit_causal_controls import (
    su2_route_local_gate_amplitudes,
    su2_route_local_gate_tokens,
)
from su2_nqs.models.support_aware_spatial_cg import SupportAwareSpatialCGConfig


jax.config.update("jax_enable_x64", True)


def _config() -> SU2RouteViTConfig:
    spatial, _ledger = spatial_cg_route_spec(
        "square",
        2,
        4,
        feature_dim=3,
        maximum_intermediate_q=4,
        initialization_scale=0.4,
    )
    return SU2RouteViTConfig(
        spatial=spatial,
        embedding_dim=8,
        head_count=2,
        depth=1,
        mlp_ratio=2,
        modulation_rank=2,
        modulation_scale=0.2,
        initialization_scale=0.3,
    )


def test_local_gate_has_same_parameters_but_no_cross_route_read() -> None:
    config = _config()
    params = init_su2_route_vit(jax.random.PRNGKey(53), config)
    count = su2_route_vit_parameter_count(params)
    assert count > 0

    original = np.asarray(su2_route_local_gate_tokens(params, config))
    features = np.asarray(config.spatial.route_features)
    modified_features = features.copy()
    modified_features[1] += np.linspace(0.2, 0.7, config.feature_count)
    modified_spatial = SupportAwareSpatialCGConfig(
        backbone=config.spatial.backbone,
        route_names=config.spatial.route_names,
        route_orders=config.spatial.route_orders,
        route_features=tuple(tuple(row) for row in modified_features),
        feature_names=config.spatial.feature_names,
        reference_route=config.spatial.reference_route,
    )
    modified_config = SU2RouteViTConfig(
        spatial=modified_spatial,
        embedding_dim=config.embedding_dim,
        head_count=config.head_count,
        depth=config.depth,
        mlp_ratio=config.mlp_ratio,
        modulation_rank=config.modulation_rank,
        modulation_scale=config.modulation_scale,
        initialization_scale=config.initialization_scale,
    )
    modified = np.asarray(su2_route_local_gate_tokens(params, modified_config))
    np.testing.assert_allclose(modified[0], original[0], atol=2.0e-14, rtol=0.0)
    assert np.linalg.norm(modified[1] - original[1]) > 1.0e-6
    assert su2_route_vit_parameter_count(params) == count


def test_local_gate_state_and_parameter_tangent_remain_exact_singlets() -> None:
    config = _config()
    params = init_su2_route_vit(jax.random.PRNGKey(61), config)
    params = {
        **params,
        "transformer": {
            **params["transformer"],
            "coefficient_head": jnp.linspace(
                -0.13, 0.17, config.embedding_dim, dtype=jnp.float64
            ),
            "modulation_head": jnp.reshape(
                jnp.linspace(
                    -0.19,
                    0.23,
                    config.embedding_dim * config.modulation_rank,
                    dtype=jnp.float64,
                ),
                (config.embedding_dim, config.modulation_rank),
            ),
        },
    }
    basis_tuple = fixed_m_spin_basis(config.n_sites, 0)
    basis = jnp.asarray(basis_tuple, dtype=jnp.int32)
    matrix, integers = spin_squared_matrix(config.n_sites, n_up=config.n_sites // 2)
    positions = {value: index for index, value in enumerate(integers)}
    order = [
        positions[
            sum(bit << (config.n_sites - site - 1) for site, bit in enumerate(row))
        ]
        for row in basis_tuple
    ]
    spin_squared = matrix[np.ix_(order, order)]

    def state(candidate):
        return su2_route_local_gate_amplitudes(candidate, basis, config)

    leaves, structure = jax.tree_util.tree_flatten(params)
    tangent = jax.tree_util.tree_unflatten(
        structure,
        [
            jax.random.normal(
                jax.random.fold_in(jax.random.PRNGKey(67), index),
                leaf.shape,
                dtype=leaf.dtype,
            )
            for index, leaf in enumerate(leaves)
        ],
    )
    amplitudes, tangent_state = jax.jvp(state, (params,), (tangent,))
    for vector in (np.asarray(amplitudes), np.asarray(tangent_state)):
        norm = np.linalg.norm(vector)
        assert norm > 1.0e-13
        assert np.linalg.norm(spin_squared @ vector) / norm < 5.0e-11
