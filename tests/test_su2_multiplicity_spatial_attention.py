from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.benchmarks.support_aware_spatial_cg import spatial_cg_route_spec
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.multiplicity_spatial_features import (
    multiplicity_spatial_features,
    scramble_multiplicity_spatial_features,
)
from su2_nqs.models.su2_multiplicity_spatial_attention import (
    SU2MultiplicitySpatialAttentionConfig,
    init_su2_multiplicity_spatial_attention,
    su2_multiplicity_spatial_attention_amplitudes,
    su2_multiplicity_spatial_attention_parameter_count,
    su2_multiplicity_spatial_component_matrix,
)
from su2_nqs.models.su2_route_vit import (
    SU2RouteViTConfig,
    init_su2_route_vit,
    su2_route_vit_amplitudes,
)


jax.config.update("jax_enable_x64", True)


def _config():
    spatial, _ = spatial_cg_route_spec(
        "square",
        2,
        4,
        feature_dim=3,
        maximum_intermediate_q=4,
        initialization_scale=0.25,
    )
    base = SU2RouteViTConfig(
        spatial=spatial,
        embedding_dim=8,
        head_count=2,
        depth=1,
        modulation_rank=2,
        modulation_scale=0.12,
        initialization_scale=0.2,
    )
    features = multiplicity_spatial_features(
        spatial.backbone.tree,
        spatial.route_orders,
        square_j1j2_bonds(2, 4, j1=1.0, j2=0.0, periodic=True),
        square_j1j2_bonds(2, 4, j1=0.0, j2=1.0, periodic=True),
    )
    return SU2MultiplicitySpatialAttentionConfig(
        base=base,
        features=features,
        embedding_dim=8,
        head_count=2,
    )


def _spin_squared_in_basis(basis):
    n_sites = len(basis[0])
    matrix, integers = spin_squared_matrix(n_sites, n_up=sum(basis[0]))
    positions = {value: index for index, value in enumerate(integers)}
    order = [
        positions[sum(bit << (n_sites - site - 1) for site, bit in enumerate(row))]
        for row in basis
    ]
    return matrix[np.ix_(order, order)]


def test_multiplicity_spatial_features_have_route_node_and_pair_axes() -> None:
    config = _config()
    assert np.asarray(config.features.node).shape == (
        config.n_routes,
        config.n_nodes,
        6,
    )
    assert np.asarray(config.features.pair).shape == (
        config.n_routes,
        config.n_nodes,
        config.n_nodes,
        5,
    )
    scrambled = scramble_multiplicity_spatial_features(
        config.features, seed=91
    )
    original_node = np.asarray(config.features.node)
    scrambled_node = np.asarray(scrambled.node)
    np.testing.assert_allclose(
        np.sort(original_node.reshape(-1, 6), axis=0),
        np.sort(scrambled_node.reshape(-1, 6), axis=0),
    )
    assert not np.array_equal(original_node, scrambled_node)


def test_zero_spatial_attention_exactly_nests_route_vit() -> None:
    config = _config()
    frozen = init_su2_route_vit(jax.random.PRNGKey(2), config.base)
    params = init_su2_multiplicity_spatial_attention(
        jax.random.PRNGKey(3), frozen, config
    )
    basis = jnp.asarray(fixed_m_spin_basis(config.n_sites, 0), dtype=jnp.int32)
    actual = su2_multiplicity_spatial_attention_amplitudes(
        params, frozen, basis, config
    )
    expected = su2_route_vit_amplitudes(frozen, basis, config.base)
    np.testing.assert_allclose(actual, expected, atol=3.0e-14, rtol=0.0)
    assert su2_multiplicity_spatial_attention_parameter_count(params) > 100


def test_spatial_attention_state_components_and_head_tangent_are_singlets() -> None:
    config = _config()
    frozen = init_su2_route_vit(jax.random.PRNGKey(5), config.base)
    params = init_su2_multiplicity_spatial_attention(
        jax.random.PRNGKey(7), frozen, config
    )
    params = {
        **params,
        "control_head": 0.08
        * jax.random.normal(
            jax.random.PRNGKey(11), params["control_head"].shape,
            dtype=jnp.float64,
        ),
    }
    basis_tuple = fixed_m_spin_basis(config.n_sites, 0)
    basis = jnp.asarray(basis_tuple, dtype=jnp.int32)
    spin_squared = _spin_squared_in_basis(basis_tuple)

    def state(candidate):
        return su2_multiplicity_spatial_attention_amplitudes(
            candidate, frozen, basis, config
        )

    amplitudes = np.asarray(state(params))
    components = np.asarray(
        su2_multiplicity_spatial_component_matrix(
            params, frozen, basis, config
        )
    )
    tangent = jax.tree_util.tree_map(jnp.zeros_like, params)
    tangent["control_head"] = jnp.ones_like(params["control_head"])
    _, tangent_state = jax.jvp(state, (params,), (tangent,))
    for vector in (amplitudes, *components.T, np.asarray(tangent_state)):
        norm = float(np.linalg.norm(vector))
        # Deep products of the deliberately small random reduced maps can be
        # tiny before normalization; only an exactly zero state is vacuous.
        assert norm > 1.0e-24
        assert np.linalg.norm(spin_squared @ vector) / norm < 1.0e-10
