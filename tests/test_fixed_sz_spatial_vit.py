from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.fixed_sz_spatial_vit import (
    fixed_sz_spatial_vit_oracle,
)
from su2_nqs.models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig,
    fixed_sz_spatial_vit_amplitudes,
    fixed_sz_spatial_vit_log_amplitude,
    fixed_sz_spatial_vit_log_amplitudes,
    fixed_sz_spatial_vit_parameter_count,
    fixed_sz_spatial_vit_spatial_weights,
    init_fixed_sz_spatial_vit,
)


jax.config.update("jax_enable_x64", True)


def _config(geometry: str = "square", **updates) -> FixedSzSpatialViTConfig:
    values = {
        "geometry": geometry,
        "rows": 4,
        "columns": 4,
        "patch_rows": 2,
        "patch_columns": 2,
        "embedding_dim": 8,
        "head_count": 2,
        "depth": 1,
        "mlp_ratio": 2,
        "target_m2": 0,
        "initialization_scale": 0.2,
        "initial_distance_decay": 1.0,
    }
    values.update(updates)
    return FixedSzSpatialViTConfig(**values)


def _configuration() -> np.ndarray:
    return np.asarray(
        [
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.int32,
    ).reshape(-1)


def test_config_rejects_incompatible_patch_and_head_shapes() -> None:
    with pytest.raises(ValueError, match="patch dimensions"):
        _config(rows=3)
    with pytest.raises(ValueError, match="divisible by head_count"):
        _config(embedding_dim=7)
    with pytest.raises(ValueError, match="target_m2"):
        _config(target_m2=1)


def test_model_masks_outside_fixed_magnetization_and_counts_parameters() -> None:
    config = _config()
    params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(4), config)
    valid = _configuration()
    invalid = valid.copy()
    invalid[2] = 1
    valid_log = fixed_sz_spatial_vit_log_amplitude(params, valid, config)
    invalid_log = fixed_sz_spatial_vit_log_amplitude(params, invalid, config)
    assert np.isfinite(np.asarray(valid_log))
    assert np.isneginf(np.real(np.asarray(invalid_log)))
    assert fixed_sz_spatial_vit_parameter_count(params) == 650


@pytest.mark.parametrize("geometry", ("square", "triangular"))
def test_patch_translation_covariance_is_exact_before_external_projector(
    geometry: str,
) -> None:
    config = _config(geometry)
    params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(7), config)
    grid = _configuration().reshape((4, 4))
    reference = fixed_sz_spatial_vit_log_amplitude(params, grid.reshape(-1), config)
    for axis in (0, 1):
        translated = np.roll(grid, shift=2, axis=axis).reshape(-1)
        actual = fixed_sz_spatial_vit_log_amplitude(params, translated, config)
        np.testing.assert_allclose(actual, reference, atol=2.0e-12, rtol=0.0)


def test_distance_envelope_is_a_nontrivial_same_backbone_ablation() -> None:
    spatial = _config(use_distance_envelope=True)
    flat = _config(use_distance_envelope=False)
    params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(9), spatial)
    spatial_weights = np.asarray(
        fixed_sz_spatial_vit_spatial_weights(params, spatial, layer_index=0)
    )
    flat_weights = np.asarray(
        fixed_sz_spatial_vit_spatial_weights(params, flat, layer_index=0)
    )
    assert np.max(np.abs(spatial_weights - flat_weights)) > 1.0e-3
    assert spatial_weights.shape == (2, 4, 4)


def test_batch_amplitudes_and_random_jvp_are_finite() -> None:
    config = _config()
    params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(12), config)
    first = _configuration()
    second = np.roll(first.reshape((4, 4)), 1, axis=1).reshape(-1)
    batch = jnp.asarray(np.stack((first, second)), dtype=jnp.int32)

    def logs(candidate):
        return fixed_sz_spatial_vit_log_amplitudes(candidate, batch, config)

    leaves, structure = jax.tree_util.tree_flatten(params)
    tangent = jax.tree_util.tree_unflatten(
        structure,
        [
            jax.random.normal(
                jax.random.fold_in(jax.random.PRNGKey(31), index),
                leaf.shape,
                dtype=leaf.dtype,
            )
            for index, leaf in enumerate(leaves)
        ],
    )
    values, tangent_values = jax.jvp(logs, (params,), (tangent,))
    amplitudes = fixed_sz_spatial_vit_amplitudes(params, batch, config)
    assert np.all(np.isfinite(np.asarray(values)))
    assert np.all(np.isfinite(np.asarray(tangent_values)))
    assert np.all(np.isfinite(np.asarray(amplitudes)))
    assert np.linalg.norm(np.asarray(amplitudes)) > 0.0


@pytest.mark.parametrize("geometry", ("square", "triangular"))
def test_exact_small_oracle_passes_without_claiming_su2(geometry: str) -> None:
    report = fixed_sz_spatial_vit_oracle(geometry)
    assert report["passed"]
    assert report["batch_single_max_abs_error"] < 1.0e-13
    assert report["coarse_patch_translation_max_abs_error"] < 2.0e-12
    assert report["zero_decay_flat_nested_max_abs_error"] < 2.0e-12
    assert report["translation_projection_closure_error"] < 2.0e-12
    assert 0.0 <= report["initial_singlet_weight"] <= 1.0
