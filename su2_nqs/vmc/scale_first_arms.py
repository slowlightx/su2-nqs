"""Matched runtime adapters for the scale-first spin three-arm campaign.

The adapters expose one common sampler/local-energy/log-amplitude boundary to
the resumable matched-work SR driver.  All Hamiltonian actions are batched JAX
kernels; only fail-closed diagnostics and wall-clock synchronization touch the
host between optimization steps.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..models.fixed_sz_autoregressive_spatial_vit import (
    FixedSzAutoregressiveSpatialViTConfig,
    fixed_sz_autoregressive_spatial_log_amplitude,
    fixed_sz_autoregressive_spatial_log_amplitudes,
    init_fixed_sz_autoregressive_spatial_vit,
    sample_fixed_sz_autoregressive_spatial_vit,
)
from ..models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    init_many_cell_schur_vit,
    many_cell_schur_log_amplitudes,
    sample_many_cell_schur_vit,
)
from ..su2.many_cell_schur import rectangular_four_spin_layout
from .many_cell_tree import (
    compile_many_cell_device_hamiltonian,
    many_cell_device_local_energies_jax,
)
from .matched_work_sr import SampledLocalEnergyBatch
from .spin_device import spin_device_hpsi


EXACT_SU2_ARM = "exact_su2_balanced_schur_spatial_vit"
FIXED_SZ_ARM = "fixed_sz_shared_autoregressive_spatial_vit"
SYMMETRY_UNAWARE_ARM = "symmetry_unaware_autoregressive_spatial_vit"
SCALE_FIRST_ARM_IDS = (EXACT_SU2_ARM, FIXED_SZ_ARM, SYMMETRY_UNAWARE_ARM)


@dataclass(frozen=True)
class ScaleFirstArmRuntime:
    """One representation behind the common matched-work training boundary."""

    arm_id: str
    params: object
    model_config: object
    sample_fn: Callable[[object, jax.Array, int], jax.Array]
    local_energy_fn: Callable[[object, jax.Array], SampledLocalEnergyBatch]
    log_amplitude_fn: Callable[[object, jax.Array], jax.Array]
    bond_count: int
    connected_capacity: int | None


def _physical_runtime(
    arm_id: str,
    *,
    geometry: str,
    rows: int,
    columns: int,
    bonds: tuple[tuple[int, int, float], ...],
    shared_model: dict[str, object],
    initialization_key: jax.Array,
    phase_convention: str,
) -> ScaleFirstArmRuntime:
    target_m2 = 0 if arm_id == FIXED_SZ_ARM else None
    config = FixedSzAutoregressiveSpatialViTConfig(
        geometry,
        rows,
        columns,
        target_m2=target_m2,
        phase_convention=phase_convention,
        **shared_model,
    )
    params = init_fixed_sz_autoregressive_spatial_vit(initialization_key, config)

    def amplitude(candidate, row, model):
        return jnp.exp(
            fixed_sz_autoregressive_spatial_log_amplitude(candidate, row, model)
        )

    def action(candidate, states):
        return spin_device_hpsi(
            candidate, states, bonds, config, amplitude_fn=amplitude
        )

    kernel = jax.jit(action)

    def local_energy_fn(candidate, states):
        started = time.perf_counter()
        centers, hpsi = kernel(candidate, states)
        jax.block_until_ready(hpsi)
        elapsed = float(time.perf_counter() - started)
        centers_host = np.asarray(centers)
        if not np.all(np.isfinite(centers_host)) or np.any(centers_host == 0.0):
            raise FloatingPointError("physical arm produced invalid center amplitudes")
        return SampledLocalEnergyBatch(
            local_energies=hpsi / centers,
            bond_count=len(bonds),
            evaluation_seconds=elapsed,
        )

    return ScaleFirstArmRuntime(
        arm_id=arm_id,
        params=params,
        model_config=config,
        sample_fn=lambda candidate, key, count: (
            sample_fixed_sz_autoregressive_spatial_vit(
                candidate, key, count, config
            )
        ),
        local_energy_fn=local_energy_fn,
        log_amplitude_fn=lambda candidate, states: (
            fixed_sz_autoregressive_spatial_log_amplitudes(
                candidate, states, config
            )
        ),
        bond_count=len(bonds),
        connected_capacity=None,
    )


def build_scale_first_arm_runtime(
    arm_id: str,
    *,
    geometry: str,
    rows: int,
    columns: int,
    bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
    shared_model: dict[str, object],
    initialization_key: jax.Array,
    su2_connected_capacity: int,
    su2_amplitude_tile_size: int | None = None,
    physical_phase_convention: str = "none",
) -> ScaleFirstArmRuntime:
    """Build one arm with shared initialization and accelerator-native H."""

    if arm_id not in SCALE_FIRST_ARM_IDS:
        raise ValueError(f"unknown scale-first arm: {arm_id!r}")
    normalized_rows = []
    for bond in bonds:
        if len(bond) == 2:
            first, second = bond
            coupling = 1.0
        elif len(bond) == 3:
            first, second, coupling = bond
        else:
            raise ValueError(f"invalid bond: {bond!r}")
        normalized_rows.append((int(first), int(second), float(coupling)))
    normalized_bonds = tuple(normalized_rows)
    if not normalized_bonds:
        raise ValueError("bonds must be nonempty")
    if arm_id != EXACT_SU2_ARM:
        return _physical_runtime(
            arm_id,
            geometry=geometry,
            rows=rows,
            columns=columns,
            bonds=normalized_bonds,
            shared_model=shared_model,
            initialization_key=initialization_key,
            phase_convention=physical_phase_convention,
        )

    layout = rectangular_four_spin_layout(
        rows, columns, triangular_coordinates=geometry == "triangular"
    )
    config = ManyCellSchurViTConfig(
        layout, target_q=0, geometry=geometry, **shared_model
    )
    params = init_many_cell_schur_vit(initialization_key, config)
    compiled = compile_many_cell_device_hamiltonian(
        config,
        normalized_bonds,
        connected_capacity=su2_connected_capacity,
        amplitude_tile_size=su2_amplitude_tile_size,
    )

    def local_energy_fn(candidate, labels):
        started = time.perf_counter()
        local, overflow, _ = many_cell_device_local_energies_jax(
            compiled, candidate, labels
        )
        local, overflow = jax.block_until_ready((local, overflow))
        elapsed = float(time.perf_counter() - started)
        overflow_host = np.asarray(overflow, dtype=bool)
        return SampledLocalEnergyBatch(
            local_energies=local,
            bond_count=len(normalized_bonds),
            evaluation_seconds=elapsed,
            overflow_count=int(np.count_nonzero(overflow_host)),
        )

    return ScaleFirstArmRuntime(
        arm_id=arm_id,
        params=params,
        model_config=config,
        sample_fn=lambda candidate, key, count: sample_many_cell_schur_vit(
            candidate, key, count, config
        ),
        local_energy_fn=local_energy_fn,
        log_amplitude_fn=lambda candidate, labels: (
            many_cell_schur_log_amplitudes(candidate, labels, config)
        ),
        bond_count=len(normalized_bonds),
        connected_capacity=int(su2_connected_capacity),
    )


__all__ = [
    "EXACT_SU2_ARM",
    "FIXED_SZ_ARM",
    "SCALE_FIRST_ARM_IDS",
    "SYMMETRY_UNAWARE_ARM",
    "ScaleFirstArmRuntime",
    "build_scale_first_arm_runtime",
]
