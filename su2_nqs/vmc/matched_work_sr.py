"""Reusable resumable SR driver with an explicit matched-work ledger.

The driver is representation agnostic.  A campaign supplies native sampling,
one checked full-H local-energy evaluation, and batched log amplitudes.  Each
accepted step therefore spends exactly one Hamiltonian action per sample;
matrix-free SR reuses the frozen local energies and differentiates only the
log amplitudes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .accelerator_sampled_force import sampled_vmc_log_value_and_grad
from .physical_sampled_sr import sampled_physical_sr_direction
from .resumable_checkpoint import (
    load_latest_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)


NativeSampler = Callable[[object, jax.Array, int], jax.Array]
LocalEnergyEvaluator = Callable[[object, jax.Array], "SampledLocalEnergyBatch"]
BatchLogAmplitude = Callable[[object, jax.Array], jax.Array]


@dataclass(frozen=True)
class SampledLocalEnergyBatch:
    """One complete-H evaluation and its auditable work counters."""

    local_energies: np.ndarray
    bond_count: int
    evaluation_seconds: float
    overflow_count: int = 0

    def __post_init__(self) -> None:
        values = np.asarray(self.local_energies)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError("local_energies must contain at least two samples")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("local_energies contain non-finite values")
        if (
            isinstance(self.bond_count, bool)
            or not isinstance(self.bond_count, Integral)
            or int(self.bond_count) < 1
        ):
            raise ValueError("bond_count must be a positive integer")
        if not math.isfinite(float(self.evaluation_seconds)) or self.evaluation_seconds < 0:
            raise ValueError("evaluation_seconds must be finite and non-negative")
        if (
            isinstance(self.overflow_count, bool)
            or not isinstance(self.overflow_count, Integral)
            or int(self.overflow_count) < 0
        ):
            raise ValueError("overflow_count must be a non-negative integer")


@dataclass(frozen=True)
class MatchedWorkSRControls:
    """Static optimizer and checkpoint controls included in run identity."""

    total_steps: int
    sample_count: int
    seed: int
    damping: float
    cg_tolerance: float
    cg_max_iterations: int
    step_size: float
    maximum_relative_parameter_step: float
    checkpoint_every_steps: int = 1

    def __post_init__(self) -> None:
        for name in (
            "total_steps",
            "sample_count",
            "cg_max_iterations",
            "checkpoint_every_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least two")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        for name in (
            "damping",
            "cg_tolerance",
            "step_size",
            "maximum_relative_parameter_step",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


def run_resumable_matched_work_sr(
    initial_params: object,
    *,
    sample_fn: NativeSampler,
    local_energy_fn: LocalEnergyEvaluator,
    log_amplitude_fn: BatchLogAmplitude,
    controls: MatchedWorkSRControls,
    checkpoint_directory: str | Path,
    run_identity: Mapping[str, object],
    maximum_steps_this_call: int | None = None,
) -> dict[str, object]:
    """Run or resume one arm while sealing samples, H work, and wall time."""

    if maximum_steps_this_call is not None and (
        isinstance(maximum_steps_this_call, bool)
        or not isinstance(maximum_steps_this_call, Integral)
        or maximum_steps_this_call < 1
    ):
        raise ValueError("maximum_steps_this_call must be a positive integer")
    identity = {
        **dict(run_identity),
        "driver": "resumable_matched_work_matrix_free_sr_v1",
        "controls": {
            name: getattr(controls, name)
            for name in controls.__dataclass_fields__
        },
    }
    root = Path(checkpoint_directory)
    if root.is_dir() and any(root.glob("step_*.pkl")):
        restored = load_latest_resumable_checkpoint(
            root, expected_run_identity=identity
        )
        params = restored["params"]
        completed = int(restored["completed_steps"])
        history = list(restored["history"])
        resumed = True
    else:
        params = initial_params
        completed = 0
        history: list[dict[str, object]] = []
        resumed = False
    if completed > controls.total_steps:
        raise ValueError("checkpoint is beyond the registered total_steps")
    remaining = controls.total_steps - completed
    execute = remaining
    if maximum_steps_this_call is not None:
        execute = min(execute, int(maximum_steps_this_call))
    cumulative_samples = int(history[-1]["cumulative_samples"]) if history else 0
    cumulative_bond_terms = (
        int(history[-1]["cumulative_full_h_bond_terms"]) if history else 0
    )
    cumulative_wall = float(history[-1]["cumulative_wall_seconds"]) if history else 0.0
    latest_checkpoint: Path | None = None

    for step in range(completed + 1, completed + execute + 1):
        wall_started = time.perf_counter()
        sample_started = time.perf_counter()
        key = jax.random.fold_in(
            jax.random.PRNGKey(int(controls.seed)), step - 1
        )
        samples = jnp.asarray(sample_fn(params, key, int(controls.sample_count)))
        jax.block_until_ready(samples)
        sampling_seconds = float(time.perf_counter() - sample_started)
        if samples.ndim < 1 or samples.shape[0] != controls.sample_count:
            raise ValueError("sample_fn returned the wrong sample count")

        local_batch = local_energy_fn(params, samples)
        if len(local_batch.local_energies) != controls.sample_count:
            raise ValueError("local_energy_fn returned the wrong sample count")
        if local_batch.overflow_count:
            raise RuntimeError("full-H connectivity overflowed; step rejected")
        force_started = time.perf_counter()
        force = sampled_vmc_log_value_and_grad(
            params,
            samples,
            local_batch.local_energies,
            log_amplitude_fn=log_amplitude_fn,
        )
        direction = sampled_physical_sr_direction(
            params,
            samples,
            energy_gradient=force.gradient,
            log_amplitude_fn=log_amplitude_fn,
            damping=controls.damping,
            cg_tolerance=controls.cg_tolerance,
            cg_max_iterations=controls.cg_max_iterations,
            step_size=controls.step_size,
            maximum_relative_parameter_step=(
                controls.maximum_relative_parameter_step
            ),
            raise_on_nonconvergence=True,
        )
        candidate = jax.tree_util.tree_map(
            lambda value, change: value - change,
            params,
            direction.direction,
        )
        for leaf in jax.tree_util.tree_leaves(candidate):
            jax.block_until_ready(leaf)
        force_sr_seconds = float(time.perf_counter() - force_started)
        params = candidate
        step_wall = float(time.perf_counter() - wall_started)
        cumulative_samples += controls.sample_count
        cumulative_bond_terms += controls.sample_count * local_batch.bond_count
        cumulative_wall += step_wall
        history.append(
            {
                "step": step,
                "energy_before_update": force.energy,
                "variance_before_update": force.variance,
                "energy_imaginary_abs": force.energy_imaginary_abs,
                "gradient_norm": force.gradient_norm,
                "sr_cg_iterations": direction.diagnostics.cg_iterations,
                "sr_qgt_matvec_count": direction.diagnostics.qgt_matvec_count,
                "sr_relative_linear_residual": (
                    direction.diagnostics.relative_linear_residual
                ),
                "sr_relative_parameter_step": (
                    direction.diagnostics.relative_parameter_step
                ),
                "samples": controls.sample_count,
                "full_h_evaluations": 1,
                "full_h_bond_terms": controls.sample_count
                * local_batch.bond_count,
                "sampling_seconds": sampling_seconds,
                "full_h_seconds": local_batch.evaluation_seconds,
                "force_sr_seconds": force_sr_seconds,
                "step_wall_seconds": step_wall,
                "cumulative_samples": cumulative_samples,
                "cumulative_full_h_bond_terms": cumulative_bond_terms,
                "cumulative_wall_seconds": cumulative_wall,
            }
        )
        if (
            step % controls.checkpoint_every_steps == 0
            or step == controls.total_steps
            or step == completed + execute
        ):
            latest_checkpoint = save_resumable_checkpoint_generation(
                root,
                params,
                optimizer_state={"kind": "stateless_matrix_free_sr_v1"},
                completed_steps=step,
                next_sample_index=cumulative_samples,
                run_identity=identity,
                history=history,
                metadata={
                    "cumulative_full_h_bond_terms": cumulative_bond_terms,
                    "cumulative_wall_seconds": cumulative_wall,
                },
            )

    return {
        "params": params,
        "history": history,
        "completed_steps": completed + execute,
        "target_steps": controls.total_steps,
        "resumed": resumed,
        "latest_checkpoint": (
            None if latest_checkpoint is None else str(latest_checkpoint)
        ),
        "cumulative_samples": cumulative_samples,
        "cumulative_full_h_bond_terms": cumulative_bond_terms,
        "cumulative_wall_seconds": cumulative_wall,
        "run_identity": identity,
    }


__all__ = [
    "BatchLogAmplitude",
    "LocalEnergyEvaluator",
    "MatchedWorkSRControls",
    "NativeSampler",
    "SampledLocalEnergyBatch",
    "run_resumable_matched_work_sr",
]
