from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.benchmarks.support_aware_spatial_cg import spatial_cg_route_spec
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.route_vit_features import multiscale_bond_route_features
from su2_nqs.models.spin_clebsch_tree import spin_clebsch_tree_amplitudes
from su2_nqs.models.su2_route_vit import (
    SU2RouteViTConfig,
    init_su2_route_vit,
    su2_route_vit_amplitudes,
    su2_route_vit_component_matrix,
    su2_route_vit_controls,
    su2_route_vit_parameter_count,
)


jax.config.update("jax_enable_x64", True)


def _config(*, modulation_rank: int = 2) -> SU2RouteViTConfig:
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
        modulation_rank=modulation_rank,
        modulation_scale=0.2,
        initialization_scale=0.3,
    )


def _spin_squared_in_basis(basis: tuple[tuple[int, ...], ...]) -> np.ndarray:
    n_sites = len(basis[0])
    matrix, integers = spin_squared_matrix(n_sites, n_up=sum(basis[0]))
    positions = {value: index for index, value in enumerate(integers)}
    order = [
        positions[
            sum(bit << (n_sites - site - 1) for site, bit in enumerate(row))
        ]
        for row in basis
    ]
    return matrix[np.ix_(order, order)]


def test_route_vit_initial_state_is_exactly_the_reference_tree() -> None:
    config = _config()
    params = init_su2_route_vit(jax.random.PRNGKey(7), config)
    basis = jnp.asarray(fixed_m_spin_basis(config.n_sites, 0), dtype=jnp.int32)
    coefficients, controls = su2_route_vit_controls(params, config)
    expected = np.zeros(config.n_routes)
    expected[config.spatial.reference_route] = 1.0
    np.testing.assert_allclose(coefficients, expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(controls, 0.0, atol=0.0, rtol=0.0)

    state = np.asarray(su2_route_vit_amplitudes(params, basis, config))
    order = jnp.asarray(
        config.spatial.route_orders[config.spatial.reference_route],
        dtype=jnp.int32,
    )
    reference = np.asarray(
        spin_clebsch_tree_amplitudes(
            params["backbone"], basis[:, order], config.spatial.backbone
        )
    )
    np.testing.assert_allclose(state, reference, atol=2.0e-14, rtol=0.0)


def test_route_vit_state_components_and_parameter_tangent_are_singlets() -> None:
    config = _config()
    params = init_su2_route_vit(jax.random.PRNGKey(11), config)
    # Activate both coherent route mixing and reduced-tensor conditioning.
    params = {
        **params,
        "transformer": {
            **params["transformer"],
            "coefficient_head": jnp.linspace(
                -0.17, 0.19, config.embedding_dim, dtype=jnp.float64
            ),
            "modulation_head": jnp.reshape(
                jnp.linspace(
                    -0.23,
                    0.21,
                    config.embedding_dim * config.modulation_rank,
                    dtype=jnp.float64,
                ),
                (config.embedding_dim, config.modulation_rank),
            ),
        },
    }
    basis_tuple = fixed_m_spin_basis(config.n_sites, 0)
    basis = jnp.asarray(basis_tuple, dtype=jnp.int32)
    spin_squared = _spin_squared_in_basis(basis_tuple)

    def state(candidate):
        return su2_route_vit_amplitudes(candidate, basis, config)

    leaves, structure = jax.tree_util.tree_flatten(params)
    tangent = jax.tree_util.tree_unflatten(
        structure,
        [
            jax.random.normal(
                jax.random.fold_in(jax.random.PRNGKey(92), index),
                leaf.shape,
                dtype=leaf.dtype,
            )
            for index, leaf in enumerate(leaves)
        ],
    )
    amplitudes, tangent_state = jax.jvp(state, (params,), (tangent,))
    components = np.asarray(
        su2_route_vit_component_matrix(params, basis, config)
    )
    for vector in (
        np.asarray(amplitudes),
        np.asarray(tangent_state),
        *(components[:, route] for route in range(config.n_routes)),
    ):
        norm = np.linalg.norm(vector)
        assert norm > 1.0e-13
        assert np.linalg.norm(spin_squared @ vector) / norm < 5.0e-11
    assert su2_route_vit_parameter_count(params) > 0


def test_coefficient_only_control_has_no_reduced_modulation() -> None:
    config = _config(modulation_rank=0)
    params = init_su2_route_vit(jax.random.PRNGKey(5), config)
    _coefficients, controls = su2_route_vit_controls(params, config)
    assert controls.shape == (config.n_routes, 0)
    assert all(
        leaf.shape[0] == 0
        for leaf in jax.tree_util.tree_leaves(params["directions"])
    )




def test_multiscale_route_tokens_resolve_physical_fusion_scales() -> None:
    config = _config()
    orders = dict(
        zip(
            config.spatial.route_names,
            config.spatial.route_orders,
            strict=True,
        )
    )
    features = multiscale_bond_route_features(
        config.spatial.backbone.tree,
        orders,
        {
            "J1": square_j1j2_bonds(
                2, 4, j1=1.0, j2=0.0, periodic=True
            ),
            "J2": square_j1j2_bonds(
                2, 4, j1=0.0, j2=1.0, periodic=True
            ),
        },
    )
    values = np.asarray(features.standardized)
    assert values.shape[0] == config.n_routes
    assert values.shape[1] >= 2
    assert len(set(features.standardized)) > 1
    np.testing.assert_allclose(values.mean(axis=0), 0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        np.sqrt(np.mean(values**2, axis=0)), 1.0, atol=2.0e-15
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"embedding_dim": 7, "head_count": 2}, "divisible"),
        ({"modulation_rank": -1}, "nonnegative"),
        ({"depth": 0}, "positive"),
    ],
)
def test_route_vit_config_fails_closed(kwargs: dict[str, int], message: str) -> None:
    spatial = _config().spatial
    with pytest.raises((TypeError, ValueError), match=message):
        SU2RouteViTConfig(spatial=spatial, **kwargs)
