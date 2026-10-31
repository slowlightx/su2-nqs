from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.four_cell_schur import (
    four_cell_schur_sector_state_vector,
    four_cell_schur_state_vector,
)
from su2_nqs.benchmarks.four_cell_schur_frontier import (
    SQUARE_N16_SUPPORTS,
    TRIANGULAR_N12_SUPPORTS,
)
from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.models.su2_four_cell_sector_vit import (
    exhaustive_four_cell_sector_actions,
    init_su2_four_cell_sector_vit,
    sample_su2_four_cell_sector_vit,
    square_sector_config,
    su2_four_cell_sector_log_amplitudes,
    triangular_sector_config,
)
from su2_nqs.su2.fusion_path import count_paths
from su2_nqs.su2.local_schur_fusion import (
    four_cell_schur_sector_states,
    four_cell_sector_action_table,
    four_cell_sector_autoregressive_masks,
)


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("kind", "n_sites", "target_q", "dimension", "width"),
    (
        ("triangular_triangle", 12, 0, 132, 7),
        ("triangular_triangle", 12, 2, 297, 7),
        ("square_plaquette", 16, 0, 1430, 9),
        ("square_plaquette", 16, 2, 3432, 9),
    ),
)
def test_sector_counts_and_exact_prefix_masks(
    kind: str,
    n_sites: int,
    target_q: int,
    dimension: int,
    width: int,
) -> None:
    states = four_cell_schur_sector_states(kind, target_q)
    actions = four_cell_sector_action_table(kind, target_q)
    masks = four_cell_sector_autoregressive_masks(kind, target_q)
    assert len(states) == dimension == count_paths(n_sites, target_q)
    assert actions.shape == (dimension, 6)
    assert masks.shape == (dimension, 6, width)
    assert np.all(np.take_along_axis(masks, actions[..., None], axis=-1))


def test_sector_q0_expansion_agrees_with_singlet_reference() -> None:
    rng = np.random.default_rng(101)
    coefficients = rng.normal(size=132) + 1.0j * rng.normal(size=132)
    reference, basis = four_cell_schur_state_vector(
        coefficients,
        "triangular_triangle",
        supports=TRIANGULAR_N12_SUPPORTS,
    )
    candidate, candidate_basis = four_cell_schur_sector_state_vector(
        coefficients,
        "triangular_triangle",
        0,
        supports=TRIANGULAR_N12_SUPPORTS,
    )
    assert candidate_basis == basis
    np.testing.assert_allclose(candidate, reference, atol=2.0e-13, rtol=2.0e-13)


@pytest.mark.parametrize("target_q,sector_index", ((0, 0), (2, 3)))
def test_shared_sector_backbone_is_exhaustively_normalized(
    target_q: int,
    sector_index: int,
) -> None:
    config = triangular_sector_config(
        target_q,
        sector_index,
        embedding_dim=8,
        head_count=2,
        depth=1,
    )
    params = init_su2_four_cell_sector_vit(jax.random.PRNGKey(109), config)
    actions = exhaustive_four_cell_sector_actions(config)
    logs = su2_four_cell_sector_log_amplitudes(params, actions, config)
    np.testing.assert_allclose(
        np.sum(np.exp(2.0 * np.real(np.asarray(logs)))),
        1.0,
        atol=2.0e-13,
    )
    samples = sample_su2_four_cell_sector_vit(
        params, jax.random.PRNGKey(113), 128, config
    )
    sampled_logs = su2_four_cell_sector_log_amplitudes(params, samples, config)
    assert np.all(np.isfinite(np.real(np.asarray(sampled_logs))))


@pytest.mark.parametrize(
    ("config_fn", "kind", "supports"),
    (
        (triangular_sector_config, "triangular_triangle", TRIANGULAR_N12_SUPPORTS),
        (square_sector_config, "square_plaquette", SQUARE_N16_SUPPORTS),
    ),
)
def test_target_q2_state_and_parameter_tangent_are_highest_weight(
    config_fn,
    kind: str,
    supports,
) -> None:
    config = config_fn(2, 3, embedding_dim=8, head_count=2, depth=1)
    params = init_su2_four_cell_sector_vit(jax.random.PRNGKey(127), config)
    actions = exhaustive_four_cell_sector_actions(config)
    direction = jax.tree_util.tree_map(jnp.ones_like, params)
    coefficients, tangent = jax.jvp(
        lambda values: jnp.exp(
            su2_four_cell_sector_log_amplitudes(values, actions, config)
        ),
        (params,),
        (direction,),
    )
    state, basis = four_cell_schur_sector_state_vector(
        np.asarray(coefficients), kind, 2, supports=supports
    )
    tangent_state, tangent_basis = four_cell_schur_sector_state_vector(
        np.asarray(tangent), kind, 2, supports=supports
    )
    assert tangent_basis == basis
    assert spin_raising_relative_norm(state, basis) < 3.0e-13
    assert spin_raising_relative_norm(tangent_state, basis) < 3.0e-13
