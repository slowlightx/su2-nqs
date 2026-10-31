"""Exact-small oracle for the fixed-``Sz`` factored Spatial-ViT baseline."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .clebsch_tree_exact import (
    fixed_m_spin_basis,
    physical_permutation_index_table,
    reynolds_project_state,
)
from .spin_sector import spin_purity_metrics, spin_squared_sz_sparse
from ..lattices.spatial_symmetry import (
    rectangular_translation_group,
    triangular_hnf_translation_group,
)
from ..models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig,
    fixed_sz_spatial_vit_log_amplitude,
    fixed_sz_spatial_vit_log_amplitudes,
    fixed_sz_spatial_vit_parameter_count,
    init_fixed_sz_spatial_vit,
)


jax.config.update("jax_enable_x64", True)


def _patch_translate(
    configuration: np.ndarray,
    config: FixedSzSpatialViTConfig,
    *,
    axis: int,
) -> np.ndarray:
    grid = np.asarray(configuration).reshape((config.rows, config.columns))
    shift = config.patch_rows if axis == 0 else config.patch_columns
    return np.roll(grid, shift=shift, axis=axis).reshape(-1)


def fixed_sz_spatial_vit_oracle(
    geometry: str,
    *,
    seed: int = 20260917,
) -> dict[str, object]:
    """Check batch consistency, coarse translation symmetry, and ``Sz`` gate."""

    rows, columns = 2, 4
    config = FixedSzSpatialViTConfig(
        geometry=geometry,
        rows=rows,
        columns=columns,
        patch_rows=2,
        patch_columns=2,
        embedding_dim=6,
        head_count=2,
        depth=1,
        mlp_ratio=2,
        target_m2=0,
        initialization_scale=0.2,
        initial_distance_decay=1.0,
    )
    params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(seed), config)
    basis_tuple = fixed_m_spin_basis(config.n_sites, config.target_m2)
    basis = jnp.asarray(basis_tuple, dtype=jnp.int32)
    batched = np.asarray(
        fixed_sz_spatial_vit_log_amplitudes(params, basis, config)
    )
    singles = np.asarray(
        [
            fixed_sz_spatial_vit_log_amplitude(params, row, config)
            for row in basis
        ]
    )
    batch_error = float(np.max(np.abs(batched - singles)))

    translation_errors = []
    for row, expected in zip(np.asarray(basis), batched, strict=True):
        for axis in (0, 1):
            translated = _patch_translate(row, config, axis=axis)
            actual = complex(
                fixed_sz_spatial_vit_log_amplitude(params, translated, config)
            )
            translation_errors.append(abs(actual - expected))
    coarse_translation_error = float(max(translation_errors, default=0.0))

    flat_config = FixedSzSpatialViTConfig(
        **{
            **config.__dict__,
            "use_distance_envelope": False,
        }
    )
    zero_decay = jax.tree_util.tree_map(lambda value: value, params)
    zero_layers = []
    for layer in zero_decay["layers"]:
        zero_layers.append(
            {
                **layer,
                "raw_distance_decay": jnp.full_like(
                    layer["raw_distance_decay"], -40.0
                ),
            }
        )
    zero_decay = {**zero_decay, "layers": tuple(zero_layers)}
    nested_spatial = np.asarray(
        fixed_sz_spatial_vit_log_amplitudes(zero_decay, basis, config)
    )
    flat = np.asarray(
        fixed_sz_spatial_vit_log_amplitudes(zero_decay, basis, flat_config)
    )
    zero_decay_nested_error = float(np.max(np.abs(nested_spatial - flat)))

    if geometry == "square":
        group = rectangular_translation_group(rows, columns)
    elif geometry == "triangular":
        group = triangular_hnf_translation_group(((columns, 0), (0, rows)))
    else:
        raise ValueError("geometry must be 'square' or 'triangular'")
    indices = physical_permutation_index_table(basis_tuple, group)
    amplitudes = np.exp(batched - np.max(np.real(batched)))
    projected = np.asarray(reynolds_project_state(amplitudes, indices))
    projection_closure = float(
        np.linalg.norm(
            np.asarray(reynolds_project_state(projected, indices)) - projected
        )
        / np.linalg.norm(projected)
    )
    spin_squared, spin_basis = spin_squared_sz_sparse(
        config.n_sites, n_up=config.target_n_up
    )
    if tuple(spin_basis) != tuple(basis_tuple):
        raise RuntimeError("Spatial-ViT oracle fixed-Sz basis order changed")
    purity = spin_purity_metrics(
        projected,
        spin_squared,
        maximum_spin=config.n_sites // 2,
    )

    outside = np.asarray(basis_tuple[0], dtype=np.int32).copy()
    outside[int(np.flatnonzero(outside == 0)[0])] = 1
    outside_log = complex(
        fixed_sz_spatial_vit_log_amplitude(params, outside, config)
    )
    sector_mask_passed = bool(np.isneginf(outside_log.real))
    return {
        "geometry": geometry,
        "rows": rows,
        "columns": columns,
        "patch_shape": [config.patch_rows, config.patch_columns],
        "parameter_count": fixed_sz_spatial_vit_parameter_count(params),
        "batch_single_max_abs_error": batch_error,
        "coarse_patch_translation_max_abs_error": coarse_translation_error,
        "zero_decay_flat_nested_max_abs_error": zero_decay_nested_error,
        "translation_projection_closure_error": projection_closure,
        "sector_mask_passed": sector_mask_passed,
        "initial_spin_squared_expectation": purity.spin_squared_expectation,
        "initial_singlet_weight": purity.singlet_weight,
        "passed": bool(
            batch_error < 1.0e-13
            and coarse_translation_error < 2.0e-12
            and zero_decay_nested_error < 2.0e-12
            and projection_closure < 2.0e-12
            and sector_mask_passed
        ),
    }


__all__ = ["fixed_sz_spatial_vit_oracle"]
