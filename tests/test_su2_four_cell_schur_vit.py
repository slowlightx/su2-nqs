from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.four_cell_schur import four_cell_schur_state_vector
from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.models.su2_four_cell_schur_vit import (
    exhaustive_four_cell_actions,
    init_su2_four_cell_schur_vit,
    sample_su2_four_cell_schur_vit,
    square_four_cell_config,
    su2_four_cell_schur_log_amplitudes,
    triangular_four_cell_config,
)
from su2_nqs.su2.local_schur_fusion import (
    four_cell_action_table,
    four_cell_autoregressive_masks,
    four_cell_schur_states,
)


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("kind", "dimension", "width"),
    (
        ("triangular_triangle", 132, 7),
        ("square_plaquette", 1430, 9),
    ),
)
def test_four_cell_counts_and_exact_reachability_masks(kind, dimension, width) -> None:
    states = four_cell_schur_states(kind)
    actions = four_cell_action_table(kind)
    masks = four_cell_autoregressive_masks(kind)
    assert len(states) == dimension
    assert actions.shape == (dimension, 5)
    assert masks.shape == (dimension, 5, width)
    assert np.all(np.take_along_axis(masks, actions[..., None], axis=-1))
    for decision in range(5):
        by_prefix = {}
        for row, mask in zip(actions, masks[:, decision], strict=True):
            prefix = tuple(row[:decision])
            if prefix in by_prefix:
                np.testing.assert_array_equal(mask, by_prefix[prefix])
            else:
                by_prefix[prefix] = mask


@pytest.mark.parametrize(
    "config",
    (
        triangular_four_cell_config(embedding_dim=8, head_count=2, depth=1),
        square_four_cell_config(embedding_dim=8, head_count=2, depth=1),
    ),
)
def test_exhaustive_autoregressive_wavefunction_is_normalized(config) -> None:
    params = init_su2_four_cell_schur_vit(
        jax.random.PRNGKey(config.n_sites), config
    )
    actions = exhaustive_four_cell_actions(config)
    logs = su2_four_cell_schur_log_amplitudes(params, actions, config)
    assert bool(jnp.all(jnp.isfinite(logs)))
    np.testing.assert_allclose(
        jnp.sum(jnp.exp(2.0 * logs.real)), 1.0, atol=3.0e-14, rtol=0.0
    )
    tangent = jax.tree_util.tree_map(jnp.ones_like, params)
    amplitudes, derivative = jax.jvp(
        lambda values: jnp.exp(
            su2_four_cell_schur_log_amplitudes(values, actions, config)
        ),
        (params,),
        (tangent,),
    )
    normalization_tangent = 2.0 * jnp.real(jnp.vdot(amplitudes, derivative))
    assert abs(float(normalization_tangent)) < 2.0e-12


@pytest.mark.parametrize(
    "config",
    (
        triangular_four_cell_config(embedding_dim=8, head_count=2, depth=1),
        square_four_cell_config(embedding_dim=8, head_count=2, depth=1),
    ),
)
def test_sampled_actions_are_valid_without_repair(config) -> None:
    params = init_su2_four_cell_schur_vit(jax.random.PRNGKey(31), config)
    samples = sample_su2_four_cell_schur_vit(
        params, jax.random.PRNGKey(32), 4096, config
    )
    logs = su2_four_cell_schur_log_amplitudes(params, samples, config)
    assert bool(jnp.all(jnp.isfinite(logs)))
    registered = {tuple(row) for row in np.asarray(exhaustive_four_cell_actions(config))}
    assert all(tuple(row) in registered for row in np.asarray(samples))


@pytest.mark.parametrize(
    "config",
    (
        triangular_four_cell_config(embedding_dim=8, head_count=2, depth=1),
        square_four_cell_config(embedding_dim=8, head_count=2, depth=1),
    ),
)
def test_state_and_parameter_tangent_are_exact_singlets(config) -> None:
    params = init_su2_four_cell_schur_vit(jax.random.PRNGKey(47), config)
    actions = exhaustive_four_cell_actions(config)
    tangent = jax.tree_util.tree_map(jnp.ones_like, params)
    coefficients, coefficient_tangent = jax.jvp(
        lambda values: jnp.exp(
            su2_four_cell_schur_log_amplitudes(values, actions, config)
        ),
        (params,),
        (tangent,),
    )
    state, basis = four_cell_schur_state_vector(
        np.asarray(coefficients), config.kind
    )
    state_tangent, tangent_basis = four_cell_schur_state_vector(
        np.asarray(coefficient_tangent), config.kind
    )
    assert tangent_basis == basis
    np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=2.0e-14, rtol=0.0)
    assert spin_raising_relative_norm(state, basis) < 2.0e-14
    assert spin_raising_relative_norm(state_tangent, basis) < 3.0e-13

