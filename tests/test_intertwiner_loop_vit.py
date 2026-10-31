from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.four_cell_schur import four_cell_schur_basis_matrix
from su2_nqs.models.intertwiner_loop_vit import (
    IntertwinerLoopViTConfig,
    init_intertwiner_loop_vit,
    intertwiner_loop_canonical_coefficients,
    intertwiner_loop_frame_coefficients,
    intertwiner_loop_frame_contributions,
    intertwiner_loop_labels,
    intertwiner_loop_parameter_count,
    project_intertwiner_loop_coefficients,
)
from su2_nqs.su2.four_cell_recoupling import (
    four_cell_pairing_recoupling,
    pair_singlet_state_indices,
    validate_pairing_order,
)
from su2_nqs.su2.local_schur_fusion import four_cell_schur_states


jax.config.update("jax_enable_x64", True)


def _config(**kwargs) -> IntertwinerLoopViTConfig:
    return IntertwinerLoopViTConfig(
        "square_plaquette",
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
        **kwargs,
    )


def test_pairing_order_validation_is_fail_closed() -> None:
    assert validate_pairing_order((0, 2, 1, 3)) == (0, 2, 1, 3)
    with pytest.raises(TypeError, match="integers"):
        validate_pairing_order((0, 1, 2, False))
    with pytest.raises(ValueError, match="permute"):
        validate_pairing_order((0, 1, 1, 3))


def test_four_cell_recoupling_is_orthogonal_and_identity_frame_is_exact() -> None:
    dimension = len(four_cell_schur_states("square_plaquette"))
    identity = four_cell_pairing_recoupling(
        "square_plaquette", (0, 1, 2, 3)
    )
    vertical = four_cell_pairing_recoupling(
        "square_plaquette", (0, 2, 1, 3)
    )
    np.testing.assert_allclose(
        identity, np.eye(dimension), atol=7.0e-16, rtol=0.0
    )
    np.testing.assert_allclose(
        vertical.T @ vertical,
        np.eye(dimension),
        atol=1.3e-15,
        rtol=0.0,
    )


def test_recoupling_matches_independent_computational_basis_contraction() -> None:
    kind = "triangular_triangle"
    supports = tuple(tuple(range(3 * cell, 3 * cell + 3)) for cell in range(4))
    order = (0, 2, 1, 3)
    canonical, configurations = four_cell_schur_basis_matrix(
        kind, supports=supports
    )
    frame, checked = four_cell_schur_basis_matrix(
        kind, supports=tuple(supports[index] for index in order)
    )
    assert checked == configurations
    explicit = canonical.T @ frame
    recoupled = four_cell_pairing_recoupling(kind, order)
    np.testing.assert_allclose(explicit, recoupled, atol=6.0e-15, rtol=0.0)


def test_native_loop_support_and_multiframe_recoupling_create_intercell_q() -> None:
    config = _config()
    params = init_intertwiner_loop_vit(jax.random.PRNGKey(13), config)
    labels = intertwiner_loop_labels(config.kind)
    assert labels.shape == (196, 4, 2)
    selected = pair_singlet_state_indices(config.kind)
    states = four_cell_schur_states(config.kind)
    assert all(states[int(index)].pair_q == 0 for index in selected)
    frame_coefficients = intertwiner_loop_frame_coefficients(
        params, config, (0, 1, 2, 3)
    )
    assert frame_coefficients.shape == (196,)
    contributions = tuple(
        np.asarray(value)
        for value in intertwiner_loop_frame_contributions(params, config)
    )
    nonzero_q = np.asarray([state.pair_q > 0 for state in states])
    assert np.linalg.norm(contributions[0][nonzero_q]) < 2.0e-15
    assert np.linalg.norm(contributions[1][nonzero_q]) > 1.0e-4
    assert np.linalg.norm(contributions[2][nonzero_q]) > 1.0e-4
    coherent = np.asarray(intertwiner_loop_canonical_coefficients(params, config))
    assert np.linalg.norm(coherent[nonzero_q]) > 1.0e-4


def test_model_is_differentiable_reduced_only_and_sector_projection_is_exact() -> None:
    config = _config(embedding_dim=4, head_count=1, depth=1, mlp_ratio=1)
    params = init_intertwiner_loop_vit(jax.random.PRNGKey(21), config)
    coefficients = intertwiner_loop_canonical_coefficients(params, config)
    jacobian = jax.jacrev(
        lambda candidate: intertwiner_loop_canonical_coefficients(candidate, config)
    )(params)
    packed = np.concatenate(
        [
            np.asarray(leaf).reshape(config.coefficient_dimension, -1)
            for leaf in jax.tree_util.tree_leaves(jacobian)
        ],
        axis=1,
    )
    assert packed.shape[1] == intertwiner_loop_parameter_count(params)
    assert np.isfinite(packed).all()
    rng = np.random.default_rng(22)
    raw = rng.normal(size=(config.coefficient_dimension, 7))
    sector, _ = np.linalg.qr(raw)
    projected = project_intertwiner_loop_coefficients(
        coefficients, jnp.asarray(sector)
    )
    np.testing.assert_allclose(
        np.asarray(projected),
        sector.T @ np.asarray(coefficients),
        atol=2.0e-15,
        rtol=0.0,
    )
