"""Preemptible single-root compact-tree Adam training.

This module closes the infrastructure loop needed before a long GPU campaign:
samples use global indexed random streams, and each atomic checkpoint contains
the optimizer state plus the next unused sample index.  It is deliberately a
single-root trainer.  It does not implement the sampled multi-root Ritz/SR or
spatial-sector machinery required for the P9v2 D1 physics claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np

from ..models.compact_tree_gru import CompactTreeGRUConfig
from .adaptive_tree import (
    DEFAULT_TREE_CONNECTIVITY_BUCKETS,
    ExecutionMode,
    compact_tree_vmc_energy_gradient_auto,
    sample_compact_tree_data_parallel,
)
from .resumable_checkpoint import (
    load_latest_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)


@dataclass(frozen=True)
class ResumableTreeAdamResult:
    """State and audit ledger returned by a resumable training invocation."""

    params: object
    optimizer_state: object
    completed_steps: int
    next_sample_index: int
    history: tuple[dict[str, object], ...]
    run_identity: dict[str, object]
    resumed: bool
    stopped_early: bool


def pytree_sha256(tree: object) -> str:
    """Hash a pytree's structure, shapes, dtypes, and canonical host bytes."""

    leaves, structure = jax.tree_util.tree_flatten(tree)
    digest = hashlib.sha256(repr(structure).encode("utf-8"))
    for leaf in leaves:
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _device_labels(devices: Sequence[jax.Device]) -> list[str]:
    return [
        f"{device.platform}:{device.process_index}:{device.id}:{device.device_kind}"
        for device in devices
    ]


def _installed_version(distribution: str) -> str | None:
    """Return an installed distribution version without making it required."""

    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _resolved_identity(
    user_identity: Mapping[str, object],
    *,
    config: CompactTreeGRUConfig,
    bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
    total_steps: int,
    batch_size: int,
    learning_rate: float,
    sample_key: jax.Array,
    execution_mode: ExecutionMode,
    registered_widths: Sequence[int],
    microbatch_size: int,
    devices: Sequence[jax.Device],
    optax_version: str,
) -> dict[str, object]:
    """Add every trajectory-affecting trainer choice to a caller identity."""

    return {
        "schema": "resumable_compact_tree_adam_v1",
        "campaign": dict(user_identity),
        "model": {
            "tree": repr(config.tree.root),
            "n_sites": config.n,
            "target_q": config.target_q,
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
        },
        "hamiltonian_bonds": [list(map(float, bond)) for bond in bonds],
        "optimizer": {
            "name": "optax.adam",
            "version": optax_version,
            "learning_rate": float(learning_rate),
        },
        "schedule": {
            "total_steps": total_steps,
            "batch_size": batch_size,
            "sample_key": np.asarray(sample_key, dtype=np.uint32).tolist(),
            "indexed_sampling": True,
            "weighting": "uniform_from_current_Born_distribution",
        },
        "execution": {
            "mode": execution_mode,
            "registered_widths": list(map(int, registered_widths)),
            "microbatch_size_per_device": microbatch_size,
            "devices": _device_labels(devices),
            "jax_version": jax.__version__,
            "jaxlib_version": _installed_version("jaxlib"),
            "accelerator_plugin_versions": {
                distribution: _installed_version(distribution)
                for distribution in ("jax-cuda12-plugin", "jax-cuda12-pjrt")
            },
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "allocator_environment": {
                name: os.environ.get(name)
                for name in (
                    "XLA_PYTHON_CLIENT_PREALLOCATE",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION",
                    "XLA_PYTHON_CLIENT_ALLOCATOR",
                    "XLA_FLAGS",
                )
            },
        },
    }


def _finite_pytree(tree: object) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def run_resumable_compact_tree_adam(
    initial_params: Mapping[str, object],
    *,
    config: CompactTreeGRUConfig,
    bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
    total_steps: int,
    batch_size: int,
    learning_rate: float,
    sample_key: jax.Array,
    run_identity: Mapping[str, object],
    checkpoint_path: str | Path,
    resume: bool = False,
    checkpoint_every: int = 1,
    max_steps_this_invocation: int | None = None,
    execution_mode: ExecutionMode = "auto",
    registered_widths: Sequence[int] = DEFAULT_TREE_CONNECTIVITY_BUCKETS,
    microbatch_size: int = 8,
    devices: Sequence[jax.Device] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> ResumableTreeAdamResult:
    """Run or resume indexed-sample Adam and checkpoint complete updates.

    A checkpoint at step ``s`` owns samples ``[0, s * batch_size)``; the next
    step starts exactly at ``s * batch_size``.  ``max_steps_this_invocation``
    is intentionally excluded from the identity so a scheduler can split one
    registered trajectory across allocations without changing it.
    """

    if total_steps < 1 or batch_size < 1 or checkpoint_every < 1:
        raise ValueError("steps, batch_size, and checkpoint_every must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if max_steps_this_invocation is not None and max_steps_this_invocation < 1:
        raise ValueError("max_steps_this_invocation must be positive")
    selected_devices = tuple(jax.local_devices() if devices is None else devices)
    if not selected_devices:
        raise ValueError("at least one JAX device is required")
    try:
        import optax  # type: ignore
    except ImportError as error:
        raise RuntimeError("resumable tree Adam requires the train extra (optax)") from error

    identity = _resolved_identity(
        run_identity,
        config=config,
        bonds=bonds,
        total_steps=total_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        sample_key=sample_key,
        execution_mode=execution_mode,
        registered_widths=registered_widths,
        microbatch_size=microbatch_size,
        devices=selected_devices,
        optax_version=getattr(optax, "__version__", "unknown"),
    )
    optimizer = optax.adam(learning_rate)
    destination = Path(checkpoint_path)
    if resume:
        checkpoint = load_latest_resumable_checkpoint(
            destination, expected_run_identity=identity
        )
        params = checkpoint["params"]
        optimizer_state = checkpoint["optimizer_state"]
        completed_steps = int(checkpoint["completed_steps"])
        next_sample_index = int(checkpoint["next_sample_index"])
        history = [dict(row) for row in checkpoint["history"]]  # type: ignore[arg-type]
        if completed_steps > total_steps:
            raise ValueError("checkpoint is beyond the registered total_steps")
        if next_sample_index != completed_steps * batch_size:
            raise ValueError("checkpoint sample counter does not match its step")
    else:
        if destination.exists() and not destination.is_dir():
            raise FileExistsError(
                f"checkpoint series path is not a directory: {destination}"
            )
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(
                f"checkpoint series is not empty: {destination}; pass resume=True"
            )
        params = initial_params
        optimizer_state = optimizer.init(params)
        completed_steps = 0
        next_sample_index = 0
        history: list[dict[str, object]] = []
        save_resumable_checkpoint_generation(
            destination,
            params,
            optimizer_state=optimizer_state,
            completed_steps=0,
            next_sample_index=0,
            run_identity=identity,
            history=history,
            metadata={"status": "initialized"},
        )

    invocation_stop = total_steps
    if max_steps_this_invocation is not None:
        invocation_stop = min(
            invocation_stop, completed_steps + max_steps_this_invocation
        )
    for step in range(completed_steps + 1, invocation_stop + 1):
        if stop_requested is not None and stop_requested():
            break
        sampled = sample_compact_tree_data_parallel(
            params,
            sample_key,
            batch_size,
            config,
            global_index_start=next_sample_index,
            microbatch_size=microbatch_size,
            devices=selected_devices,
        )
        diagnostic = compact_tree_vmc_energy_gradient_auto(
            params,
            sampled.labels,
            config,
            bonds,
            weighting="uniform",
            execution_mode=execution_mode,
            registered_widths=registered_widths,
            microbatch_size=microbatch_size,
            devices=selected_devices,
        )
        if not (
            np.isfinite(diagnostic.energy)
            and np.isfinite(diagnostic.variance)
            and np.isfinite(diagnostic.gradient_norm)
            and _finite_pytree(diagnostic.gradient)
        ):
            raise FloatingPointError("non-finite D1 preliminary update rejected")
        before_sha256 = pytree_sha256(params)
        updates, optimizer_state = optimizer.update(
            diagnostic.gradient, optimizer_state, params
        )
        params = optax.apply_updates(params, updates)
        if not _finite_pytree(params):
            raise FloatingPointError("non-finite D1 preliminary parameters rejected")
        sample_start = next_sample_index
        next_sample_index += batch_size
        completed_steps = step
        execution = diagnostic.execution
        history.append(
            {
                "step": step,
                "sample_index_start": sample_start,
                "sample_index_stop": next_sample_index,
                "energy_before_update": diagnostic.energy,
                "variance_before_update": diagnostic.variance,
                "gradient_norm": diagnostic.gradient_norm,
                "params_before_sha256": before_sha256,
                "params_after_sha256": pytree_sha256(params),
                "sampling_seconds": sampled.compile_and_evaluate_seconds,
                "connectivity_seconds": diagnostic.connectivity_seconds,
                "first_pass_seconds": (
                    diagnostic.first_pass_compile_and_evaluate_seconds
                ),
                "second_pass_seconds": (
                    diagnostic.second_pass_compile_and_evaluate_seconds
                ),
                "backend": execution.backend,
                "device_count": execution.device_count,
                "selected_mode": execution.selected_mode,
                "chunk_count": execution.chunk_count,
                "dummy_sample_count": execution.dummy_sample_count,
                "active_bucket_widths": list(execution.active_bucket_widths),
                "maximum_resident_connectivity_slots": (
                    execution.maximum_resident_connectivity_slots
                ),
            }
        )
        should_stop = stop_requested is not None and stop_requested()
        if (
            step % checkpoint_every == 0
            or step == total_steps
            or step == invocation_stop
            or should_stop
        ):
            save_resumable_checkpoint_generation(
                destination,
                params,
                optimizer_state=optimizer_state,
                completed_steps=completed_steps,
                next_sample_index=next_sample_index,
                run_identity=identity,
                history=history,
                metadata={
                    "status": (
                        "complete" if completed_steps == total_steps else "partial"
                    ),
                    "params_sha256": pytree_sha256(params),
                },
            )
        if should_stop:
            break

    stopped_early = completed_steps < total_steps
    return ResumableTreeAdamResult(
        params=params,
        optimizer_state=optimizer_state,
        completed_steps=completed_steps,
        next_sample_index=next_sample_index,
        history=tuple(history),
        run_identity=identity,
        resumed=resume,
        stopped_early=stopped_early,
    )


__all__ = [
    "ResumableTreeAdamResult",
    "pytree_sha256",
    "run_resumable_compact_tree_adam",
]
