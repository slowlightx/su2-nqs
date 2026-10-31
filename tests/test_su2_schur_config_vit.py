from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.models.strict_schur_spatial import (
    two_plaquette_plan,
    two_triangle_plan,
)
from su2_nqs.models.su2_schur_config_vit import (
    SU2SchurConfigViTConfig,
    init_su2_schur_config_vit,
    projected_su2_schur_config_vit_amplitudes,
    schur_configuration_labels,
    su2_schur_config_vit_amplitudes,
    su2_schur_config_vit_coefficients,
    su2_schur_config_vit_tokens,
)


jax.config.update("jax_enable_x64", True)


def _all_configurations(n_sites: int) -> np.ndarray:
    return np.asarray(
        [
            [
                (state >> (n_sites - site - 1)) & 1
                for site in range(n_sites)
            ]
            for state in range(1 << n_sites)
        ],
        dtype=np.int32,
    )


@pytest.mark.parametrize(
    ("plan", "expected"),
    ((two_triangle_plan(), 5), (two_plaquette_plan(), 14)),
)
def test_configuration_labels_are_complete_reduced_singlet_basis(plan, expected) -> None:
    labels = schur_configuration_labels(plan)
    assert labels.shape == (expected, 2, 2)
    assert len({tuple(row.ravel()) for row in labels}) == expected
    assert not labels.flags.writeable
    np.testing.assert_array_equal(labels[:, 0, 0], labels[:, 1, 0])


def test_attention_really_depends_on_multiplicity_configuration() -> None:
    config = SU2SchurConfigViTConfig(two_plaquette_plan(), embedding_dim=12, head_count=3)
    params = init_su2_schur_config_vit(jax.random.PRNGKey(7), config)
    tokens, attentions = su2_schur_config_vit_tokens(params, config)
    coefficients = su2_schur_config_vit_coefficients(params, config)
    assert tokens.shape == (14, 2, 12)
    assert attentions[0].shape == (14, 3, 2, 2)
    assert float(jnp.std(coefficients)) > 1.0e-5
    # First two configurations differ only in the right-cell multiplicity.
    assert float(jnp.linalg.norm(tokens[0] - tokens[1])) > 1.0e-5
    assert float(jnp.linalg.norm(attentions[0][0] - attentions[0][1])) > 1.0e-7


@pytest.mark.parametrize("plan", (two_triangle_plan(), two_plaquette_plan()))
def test_every_state_and_parameter_tangent_is_an_exact_singlet(plan) -> None:
    config = SU2SchurConfigViTConfig(plan, embedding_dim=8, head_count=2, depth=1)
    params = init_su2_schur_config_vit(jax.random.PRNGKey(plan.n_sites), config)
    basis = jnp.asarray(_all_configurations(plan.n_sites))
    state = np.asarray(su2_schur_config_vit_amplitudes(params, basis, config))
    spin_squared, integer_basis = spin_squared_matrix(plan.n_sites)
    np.testing.assert_array_equal(integer_basis, tuple(range(1 << plan.n_sites)))
    assert np.linalg.norm(spin_squared @ state) / np.linalg.norm(state) < 3.0e-14

    tangent = jax.tree_util.tree_map(jnp.ones_like, params)
    _, state_tangent = jax.jvp(
        lambda values: su2_schur_config_vit_amplitudes(values, basis, config),
        (params,),
        (tangent,),
    )
    state_tangent = np.asarray(state_tangent)
    assert np.linalg.norm(state_tangent) > 1.0e-8
    assert (
        np.linalg.norm(spin_squared @ state_tangent) / np.linalg.norm(state_tangent)
        < 4.0e-13
    )


def test_reynolds_projection_is_exact_a1_and_preserves_singlet() -> None:
    plan = two_plaquette_plan()
    config = SU2SchurConfigViTConfig(plan, embedding_dim=8, head_count=2, depth=1)
    params = init_su2_schur_config_vit(jax.random.PRNGKey(29), config)
    basis = jnp.asarray(_all_configurations(plan.n_sites))
    identity = tuple(range(plan.n_sites))
    exchange = tuple(range(4, 8)) + tuple(range(4))
    group = (identity, exchange)
    projected = np.asarray(
        projected_su2_schur_config_vit_amplitudes(params, basis, config, group)
    )
    exchanged = np.asarray(
        projected_su2_schur_config_vit_amplitudes(
            params, basis[:, jnp.asarray(exchange)], config, group
        )
    )
    np.testing.assert_allclose(projected, exchanged, atol=2.0e-14, rtol=0.0)
    spin_squared, _ = spin_squared_matrix(plan.n_sites)
    assert np.linalg.norm(spin_squared @ projected) / np.linalg.norm(projected) < 3.0e-14

