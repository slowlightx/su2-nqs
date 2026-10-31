"""Local scaling diagnostics for exact-SU(2) spatial Schur Transformers.

The benchmark deliberately separates the neural forward/backward cost from
the exact Hamiltonian-column construction.  It is a resource-admission tool,
not an energy or accuracy benchmark.
"""

from __future__ import annotations

import math
from numbers import Integral
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.j1j2 import square_j1j2_bonds
from ..lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from ..models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
    many_cell_schur_spatial_distances,
    many_cell_schur_vit_parameter_count,
    sample_many_cell_schur_vit,
)
from ..models.sequential_cell_schur_vit import (
    SequentialCellSchurViTConfig,
    init_sequential_cell_schur_vit,
    sample_sequential_cell_schur_vit,
    sequential_cell_schur_vit_parameter_count,
)
from ..su2.many_cell_schur import (
    rectangular_four_spin_layout,
    rectangular_sequential_four_spin_layout,
    validate_layout_state,
)
from ..vmc.many_cell_tree import build_many_cell_tree_connectivity


def spin_half_singlet_dimension(n_sites: int) -> int:
    """Return the exact multiplicity of total-spin zero for even ``n_sites``."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be an integer")
    n = int(n_sites)
    if n < 2 or n % 2:
        raise ValueError("n_sites must be positive and even")
    half = n // 2
    return math.comb(n, half) - math.comb(n, half - 1)


def _bonds(
    geometry: str, linear_size: int, j2_over_j1: float
) -> tuple[tuple[int, int, float], ...]:
    if geometry == "triangular":
        return triangular_hnf_j1j2_bonds(
            ((linear_size, 0), (0, linear_size)),
            j1=1.0,
            j2=float(j2_over_j1),
        )
    if geometry == "square":
        return square_j1j2_bonds(
            linear_size,
            linear_size,
            j1=1.0,
            j2=float(j2_over_j1),
            periodic=True,
        )
    raise ValueError("geometry must be 'square' or 'triangular'")


def _minimum_timed(call, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = call()
        jax.block_until_ready(output)
        values.append(time.perf_counter() - started)
    return float(min(values))


def profile_schur_spatial_scaling(
    *,
    geometry: str,
    linear_size: int,
    j2_over_j1: float,
    sample_count: int,
    connectivity_sample_count: int,
    seed: int,
    embedding_dim: int = 32,
    head_count: int = 4,
    depth: int = 2,
    mlp_ratio: int = 2,
    timing_repeats: int = 3,
) -> dict[str, object]:
    """Profile one balanced production tree and a one-state sequential control."""

    for name, value in (
        ("linear_size", linear_size),
        ("sample_count", sample_count),
        ("connectivity_sample_count", connectivity_sample_count),
        ("timing_repeats", timing_repeats),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    size = int(linear_size)
    if size % 2:
        raise ValueError("linear_size must be even")
    if connectivity_sample_count > sample_count:
        raise ValueError("connectivity_sample_count cannot exceed sample_count")

    layout = rectangular_four_spin_layout(
        size, size, triangular_coordinates=geometry == "triangular"
    )
    config = ManyCellSchurViTConfig(
        layout,
        embedding_dim=int(embedding_dim),
        head_count=int(head_count),
        depth=int(depth),
        mlp_ratio=int(mlp_ratio),
        geometry=geometry,
        use_distance_envelope=True,
    )
    params = init_many_cell_schur_vit(jax.random.PRNGKey(int(seed)), config)
    sample_key = jax.random.PRNGKey(int(seed) + 1)
    started = time.perf_counter()
    samples = sample_many_cell_schur_vit(
        params, sample_key, int(sample_count), config
    )
    jax.block_until_ready(samples)
    sample_seconds = float(time.perf_counter() - started)
    concrete = np.asarray(samples, dtype=np.int32)
    for row in concrete:
        validate_layout_state(layout, row, target_q=0)

    forward = jax.jit(
        lambda values, labels: many_cell_schur_log_amplitudes(
            values, labels, config
        )
    )
    compile_started = time.perf_counter()
    compiled_logs = forward(params, samples)
    jax.block_until_ready(compiled_logs)
    forward_compile_seconds = float(time.perf_counter() - compile_started)
    forward_seconds = _minimum_timed(
        lambda: forward(params, samples), int(timing_repeats)
    )

    value_gradient = jax.jit(
        jax.value_and_grad(
            lambda values, labels: -jnp.mean(
                2.0
                * jnp.real(
                    many_cell_schur_log_amplitudes(values, labels, config)
                )
            )
        )
    )
    compile_started = time.perf_counter()
    value, gradient = value_gradient(params, samples)
    jax.block_until_ready(value)
    gradient_compile_seconds = float(time.perf_counter() - compile_started)
    gradient_seconds = _minimum_timed(
        lambda: value_gradient(params, samples), int(timing_repeats)
    )

    bonds = _bonds(geometry, size, j2_over_j1)
    connectivity = build_many_cell_tree_connectivity(
        concrete[: int(connectivity_sample_count)], config, bonds
    )

    sequential_layout = rectangular_sequential_four_spin_layout(
        size, size, triangular_coordinates=geometry == "triangular"
    )
    sequential_config = SequentialCellSchurViTConfig(
        sequential_layout,
        embedding_dim=int(embedding_dim),
        head_count=int(head_count),
        depth=int(depth),
        mlp_ratio=int(mlp_ratio),
        q_conditioned_attention=True,
        geometry=geometry,
        use_distance_envelope=True,
    )
    sequential_params = init_sequential_cell_schur_vit(
        jax.random.PRNGKey(int(seed)), sequential_config
    )
    sequential_sample = sample_sequential_cell_schur_vit(
        sequential_params,
        jax.random.PRNGKey(int(seed) + 2),
        1,
        sequential_config,
    )
    jax.block_until_ready(sequential_sample)
    sequential_connectivity = build_many_cell_tree_connectivity(
        np.asarray(sequential_sample), sequential_config, bonds
    )

    distances = many_cell_schur_spatial_distances(config)
    positive = distances[distances > 0.0]
    return {
        "geometry": geometry,
        "linear_size": size,
        "n_sites": size * size,
        "singlet_dimension": spin_half_singlet_dimension(size * size),
        "bond_count": len(bonds),
        "sample_count": int(sample_count),
        "all_samples_valid_singlets": True,
        "balanced_parameter_count": many_cell_schur_vit_parameter_count(params),
        "sequential_parameter_count": sequential_cell_schur_vit_parameter_count(
            sequential_params
        ),
        "minimum_positive_support_distance": float(np.min(positive)),
        "sampling_seconds": sample_seconds,
        "forward_compile_seconds": forward_compile_seconds,
        "forward_warm_seconds": forward_seconds,
        "forward_samples_per_second": float(sample_count / forward_seconds),
        "gradient_compile_seconds": gradient_compile_seconds,
        "gradient_warm_seconds": gradient_seconds,
        "gradient_samples_per_second": float(sample_count / gradient_seconds),
        "balanced_connectivity": {
            "sample_count": int(connectivity_sample_count),
            "seconds": connectivity.connectivity_seconds,
            "median_width": float(np.median(connectivity.widths)),
            "maximum_width": connectivity.maximum_width,
            "total_entries": connectivity.total_entries,
        },
        "sequential_one_state_connectivity": {
            "seconds": sequential_connectivity.connectivity_seconds,
            "width": sequential_connectivity.maximum_width,
        },
        "balanced_to_sequential_width_ratio": float(
            sequential_connectivity.maximum_width
            / max(connectivity.maximum_width, 1)
        ),
    }


__all__ = ["profile_schur_spatial_scaling", "spin_half_singlet_dimension"]
