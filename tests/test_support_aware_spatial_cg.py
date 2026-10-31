from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.benchmarks.support_aware_spatial_cg import (
    support_aware_spatial_cg_oracle,
)
from su2_nqs.lattices.geometry_tree import morton_order
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_routes import (
    coupling_shell_route_features,
    rectangular_support_route_orders,
)
from su2_nqs.models.spin_clebsch_tree import (
    SpinClebschTreeConfig,
    spin_clebsch_tree_amplitudes,
)
from su2_nqs.models.support_aware_spatial_cg import (
    SupportAwareSpatialCGConfig,
    init_support_aware_spatial_cg,
    support_aware_spatial_cg_amplitudes,
    support_aware_spatial_cg_parameter_count,
    support_aware_spatial_cg_route_coefficients,
)
from su2_nqs.su2.fusion_tree import FusionTree


jax.config.update("jax_enable_x64", True)


def _config() -> SupportAwareSpatialCGConfig:
    tree = FusionTree.balanced(8)
    routes = rectangular_support_route_orders(
        "square", 2, 4, drop_duplicates=True
    )
    features = coupling_shell_route_features(
        tree,
        routes,
        square_j1j2_bonds(2, 4, j1=1.0, j2=0.5, periodic=False),
    )
    names = tuple(routes)
    return SupportAwareSpatialCGConfig(
        backbone=SpinClebschTreeConfig(
            tree,
            target_q=0,
            target_m2=0,
            feature_dim=3,
            maximum_intermediate_q=4,
        ),
        route_names=names,
        route_orders=tuple(routes.values()),
        route_features=features.standardized_localities,
        feature_names=features.labels,
        reference_route=names.index("plaquette"),
    )


def _spin_squared_in_basis(basis: tuple[tuple[int, ...], ...]) -> np.ndarray:
    n_sites = len(basis[0])
    matrix, integer_basis = spin_squared_matrix(n_sites, n_up=sum(basis[0]))
    positions = {value: index for index, value in enumerate(integer_basis)}
    order = [
        positions[
            sum(bit << (n_sites - site - 1) for site, bit in enumerate(row))
        ]
        for row in basis
    ]
    return matrix[np.ix_(order, order)]


def test_route_bank_is_complete_deduplicated_and_shell_resolved() -> None:
    with pytest.raises(RuntimeError, match="duplicate"):
        rectangular_support_route_orders("square", 2, 4)
    routes = rectangular_support_route_orders(
        "square", 2, 4, drop_duplicates=True
    )
    assert len(routes) == 7
    assert len(set(routes.values())) == 7
    assert routes["plaquette"] == morton_order(2, 4)
    features = coupling_shell_route_features(
        FusionTree.balanced(8),
        routes,
        square_j1j2_bonds(2, 4, j1=1.0, j2=0.5, periodic=False),
    )
    matrix = np.asarray(features.standardized_localities)
    assert matrix.shape == (7, 2)
    np.testing.assert_allclose(matrix.mean(axis=0), 0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        np.sqrt(np.mean(matrix**2, axis=0)), 1.0, atol=2.0e-15
    )


def test_zero_query_is_exactly_the_morton_single_tree_baseline() -> None:
    config = _config()
    params = init_support_aware_spatial_cg(jax.random.PRNGKey(12), config)
    basis = jnp.asarray(fixed_m_spin_basis(8, 0), dtype=jnp.int32)
    spatial = support_aware_spatial_cg_amplitudes(params, basis, config)
    reference_order = jnp.asarray(
        config.route_orders[config.reference_route], dtype=jnp.int32
    )
    baseline = spin_clebsch_tree_amplitudes(
        params["backbone"], basis[:, reference_order], config.backbone
    )
    np.testing.assert_allclose(spatial, baseline, atol=1.0e-14, rtol=0.0)
    coefficients = np.asarray(
        support_aware_spatial_cg_route_coefficients(params, config)
    )
    expected = np.zeros(config.n_routes)
    expected[config.reference_route] = 1.0
    np.testing.assert_allclose(coefficients, expected, atol=0.0, rtol=0.0)
    assert support_aware_spatial_cg_parameter_count(params) == (
        sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params["backbone"]))
        + config.n_support_features
    )


def test_state_and_random_parameter_tangent_are_exact_singlets() -> None:
    config = _config()
    params = init_support_aware_spatial_cg(jax.random.PRNGKey(31), config)
    params = {
        **params,
        "geometry_query": jnp.asarray((0.31, -0.27), dtype=jnp.float64),
    }
    basis_tuple = fixed_m_spin_basis(8, 0)
    basis = jnp.asarray(basis_tuple, dtype=jnp.int32)
    spin_squared = _spin_squared_in_basis(basis_tuple)

    def state(candidate):
        return support_aware_spatial_cg_amplitudes(candidate, basis, config)

    keys = iter(jax.random.split(jax.random.PRNGKey(94), 32))
    tangent = jax.tree_util.tree_map(
        lambda leaf: jax.random.normal(next(keys), leaf.shape, dtype=leaf.dtype),
        params,
    )
    amplitudes, tangent_state = jax.jvp(state, (params,), (tangent,))
    for vector in (np.asarray(amplitudes), np.asarray(tangent_state)):
        assert np.linalg.norm(vector) > 1.0e-13
        residual = spin_squared @ vector
        assert np.linalg.norm(residual) / np.linalg.norm(vector) < 3.0e-12


def test_invalid_route_support_fails_closed() -> None:
    config = _config()
    orders = list(config.route_orders)
    orders[0] = orders[0][:-1] + (orders[0][0],)
    with pytest.raises(ValueError, match="each physical site"):
        SupportAwareSpatialCGConfig(
            backbone=config.backbone,
            route_names=config.route_names,
            route_orders=tuple(orders),
            route_features=config.route_features,
            feature_names=config.feature_names,
            reference_route=config.reference_route,
        )


@pytest.mark.parametrize("lattice", ("square", "triangular"))
def test_n8_architecture_oracle_certifies_state_tangents_and_route_rank(
    lattice: str,
) -> None:
    report = support_aware_spatial_cg_oracle(lattice, tangent_count=2)
    assert report["passed"]
    assert report["nested_single_tree_max_abs_error"] < 1.0e-13
    assert report["spin_squared_residual_ratio"] < 3.0e-12
    assert report["maximum_tangent_spin_squared_residual_ratio"] < 3.0e-12
    assert (
        report["translation_projected_component_span"]["rank_at_1e_minus_10"]
        > 1
    )
