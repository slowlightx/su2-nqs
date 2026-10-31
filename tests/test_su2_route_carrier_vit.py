from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.benchmarks.support_aware_spatial_cg import spatial_cg_route_spec
from su2_nqs.models.su2_route_carrier_vit import (
    SU2RouteCarrierViTConfig,
    init_su2_route_carrier_head,
    su2_route_carrier_component_matrix,
    su2_route_carrier_head_parameter_count,
    su2_route_carrier_vit_amplitudes,
)
from su2_nqs.models.su2_route_vit import (
    SU2RouteViTConfig,
    init_su2_route_vit,
    su2_route_vit_amplitudes,
)


jax.config.update("jax_enable_x64", True)


def _base_config() -> SU2RouteViTConfig:
    spatial, _ = spatial_cg_route_spec(
        "square",
        2,
        4,
        feature_dim=3,
        maximum_intermediate_q=4,
        initialization_scale=0.3,
    )
    return SU2RouteViTConfig(
        spatial=spatial,
        embedding_dim=8,
        head_count=2,
        depth=1,
        modulation_rank=2,
        modulation_scale=0.15,
        initialization_scale=0.25,
    )


def _config(head_count: int = 3) -> SU2RouteCarrierViTConfig:
    base = _base_config()
    rng = np.random.default_rng(19)
    matrix = rng.normal(size=(base.n_routes * 2, head_count))
    q, _ = np.linalg.qr(matrix)
    templates = np.moveaxis(q[:, :head_count].reshape(base.n_routes, 2, head_count), 2, 0)
    return SU2RouteCarrierViTConfig(
        base=base,
        templates=tuple(
            tuple(tuple(map(float, row)) for row in item) for item in templates
        ),
        labels=tuple(f"carrier-{index}" for index in range(head_count)),
        carrier_scale=0.2,
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


def test_zero_carrier_is_exactly_the_frozen_route_vit() -> None:
    config = _config()
    frozen = init_su2_route_vit(jax.random.PRNGKey(2), config.base)
    head = init_su2_route_carrier_head(config)
    basis = jnp.asarray(fixed_m_spin_basis(config.n_sites, 0), dtype=jnp.int32)
    actual = su2_route_carrier_vit_amplitudes(head, frozen, basis, config)
    expected = su2_route_vit_amplitudes(frozen, basis, config.base)
    np.testing.assert_allclose(actual, expected, atol=2.0e-14, rtol=0.0)
    assert su2_route_carrier_head_parameter_count(head) == 3


def test_carrier_state_components_and_every_head_tangent_are_singlets() -> None:
    config = _config()
    frozen = init_su2_route_vit(jax.random.PRNGKey(5), config.base)
    head = {"coefficients": jnp.asarray((0.21, -0.17, 0.13))}
    basis_tuple = fixed_m_spin_basis(config.n_sites, 0)
    basis = jnp.asarray(basis_tuple, dtype=jnp.int32)
    spin_squared = _spin_squared_in_basis(basis_tuple)

    def state(candidate):
        return su2_route_carrier_vit_amplitudes(candidate, frozen, basis, config)

    amplitudes = np.asarray(state(head))
    components = np.asarray(
        su2_route_carrier_component_matrix(head, frozen, basis, config)
    )
    tangents = []
    for index in range(config.head_count):
        tangent = {"coefficients": jnp.eye(config.head_count)[index]}
        _, value = jax.jvp(state, (head,), (tangent,))
        tangents.append(np.asarray(value))
    for vector in (amplitudes, *components.T, *tangents):
        norm = float(np.linalg.norm(vector))
        assert norm > 1.0e-13
        assert np.linalg.norm(spin_squared @ vector) / norm < 8.0e-11


def test_carrier_config_fails_closed_on_rank_and_shape() -> None:
    base = _base_config()
    duplicate = tuple((0.0, 0.0) for _ in range(base.n_routes))
    with pytest.raises(ValueError, match="linearly independent"):
        SU2RouteCarrierViTConfig(
            base=base,
            templates=(duplicate, duplicate),
            labels=("a", "b"),
        )
    with pytest.raises(ValueError, match="shape"):
        SU2RouteCarrierViTConfig(
            base=base,
            templates=(((1.0, 0.0),),),
            labels=("a",),
        )

