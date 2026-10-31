from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.benchmarks.support_aware_spatial_cg import spatial_cg_route_spec
from su2_nqs.models.coherent_su2_carrier import (
    CoherentSU2CarrierConfig,
    coherent_su2_carrier_amplitudes,
    coherent_su2_carrier_head_parameter_count,
    init_coherent_su2_carrier_head,
)
from su2_nqs.models.su2_route_vit import (
    SU2RouteViTConfig,
    init_su2_route_vit,
    su2_route_vit_amplitudes,
)


jax.config.update("jax_enable_x64", True)


def _route_config() -> SU2RouteViTConfig:
    spatial, _ = spatial_cg_route_spec(
        "square",
        2,
        4,
        feature_dim=3,
        maximum_intermediate_q=4,
        initialization_scale=0.25,
    )
    return SU2RouteViTConfig(
        spatial=spatial,
        embedding_dim=8,
        head_count=2,
        depth=1,
        modulation_rank=2,
        modulation_scale=0.12,
        initialization_scale=0.2,
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


def test_zero_head_exactly_nests_the_fixed_carrier() -> None:
    config = CoherentSU2CarrierConfig(n_sites=8, residual_scale=3.7)
    head = init_coherent_su2_carrier_head(config)
    carrier = jnp.asarray([1.0, -2.0, 3.0])
    residual = jnp.asarray([-4.0, 5.0, 6.0])
    actual = coherent_su2_carrier_amplitudes(
        head, carrier, residual, config
    )
    np.testing.assert_allclose(actual, carrier, atol=0.0, rtol=0.0)
    assert coherent_su2_carrier_head_parameter_count(head) == 1


def test_coherent_state_and_head_and_residual_tangents_are_singlets() -> None:
    route_config = _route_config()
    first = init_su2_route_vit(jax.random.PRNGKey(5), route_config)
    second = init_su2_route_vit(jax.random.PRNGKey(7), route_config)
    basis_tuple = fixed_m_spin_basis(route_config.n_sites, 0)
    basis = jnp.asarray(basis_tuple, dtype=jnp.int32)
    spin_squared = _spin_squared_in_basis(basis_tuple)
    carrier = su2_route_vit_amplitudes(first, basis, route_config)
    mixture_config = CoherentSU2CarrierConfig(
        n_sites=route_config.n_sites,
        residual_scale=1.3,
    )
    head = {"coordinates": jnp.asarray([0.31], dtype=jnp.float64)}

    def state(local_head, local_residual):
        residual = su2_route_vit_amplitudes(
            local_residual, basis, route_config
        )
        return coherent_su2_carrier_amplitudes(
            local_head, carrier, residual, mixture_config
        )

    amplitudes = state(head, second)
    head_tangent = {"coordinates": jnp.ones((1,), dtype=jnp.float64)}
    zero_residual = jax.tree_util.tree_map(jnp.zeros_like, second)
    _, tangent_head = jax.jvp(
        state, (head, second), (head_tangent, zero_residual)
    )
    residual_tangent = jax.tree_util.tree_map(jnp.zeros_like, second)
    residual_tangent["transformer"]["coefficient_head"] = jnp.ones_like(
        second["transformer"]["coefficient_head"]
    )
    zero_head = jax.tree_util.tree_map(jnp.zeros_like, head)
    _, tangent_residual = jax.jvp(
        state, (head, second), (zero_head, residual_tangent)
    )
    for vector in (amplitudes, tangent_head, tangent_residual):
        concrete = np.asarray(vector)
        norm = float(np.linalg.norm(concrete))
        # Products of deliberately small random reduced maps may have a tiny
        # gauge scale before normalization; only an exactly zero tangent is
        # vacuous for this symmetry test.
        assert norm > 1.0e-24
        assert np.linalg.norm(spin_squared @ concrete) / norm < 1.0e-10


def test_coherent_carrier_fails_closed_on_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CoherentSU2CarrierConfig(n_sites=0)
    with pytest.raises(ValueError, match="positive and finite"):
        CoherentSU2CarrierConfig(n_sites=8, residual_scale=0.0)
    config = CoherentSU2CarrierConfig(n_sites=8, complex_mixing=True)
    head = init_coherent_su2_carrier_head(config)
    assert coherent_su2_carrier_head_parameter_count(head) == 2
    with pytest.raises(ValueError, match="shapes must match"):
        coherent_su2_carrier_amplitudes(
            head, jnp.ones((2,)), jnp.ones((3,)), config
        )
