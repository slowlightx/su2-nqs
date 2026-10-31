"""Preemptible Adam gate for three unprojected compact-tree roots.

This module is deliberately narrower than the eventual D1 physics engine.  It
updates three independent ``q=0`` compact-tree states with the sampled
horizontal Ky--Fan force, using an equal-stratum shared Born-mixture proposal.
It has no spatial carrier or projector.  Its checkpoints and diagnostics are
therefore infrastructure evidence only, never ``Gamma.A1`` root ownership or
a spin-liquid result.

Every immutable checkpoint generation contains all three parameter pytrees,
three independent Adam states, and three independent next-unused indexed
sample counters.  A scalar ``next_sample_index`` is retained solely for the
generic checkpoint schema and equals the sum of the three root counters.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
import os
from pathlib import Path

import jax
import numpy as np

from ..models.compact_tree_gru import CompactTreeGRUConfig
from .adaptive_tree import DEFAULT_TREE_CONNECTIVITY_BUCKETS
from .resumable_checkpoint import (
    load_latest_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)
from .resumable_tree_adam import pytree_sha256
from .tree_multistate_gate import (
    D1A_TREE_GATE_PROPOSAL,
    D1A_TREE_GATE_SCIENTIFIC_SCOPE,
    D1_TREE_GATE_DEFAULT_EXACT_NORMALIZED_OVERLAP_DIAGONAL,
    compact_tree_three_root_ritz_force_gate,
)


@dataclass(frozen=True)
class ResumableThreeRootTreeAdamResult:
    """State and audit ledger returned by one resumable invocation."""

    params: tuple[object, object, object]
    optimizer_states: tuple[object, object, object]
    completed_steps: int
    next_root_sample_indices: tuple[int, int, int]
    next_sample_index: int
    history: tuple[dict[str, object], ...]
    run_identity: dict[str, object]
    resumed: bool
    stopped_early: bool
    recovered_checkpoint_failures: tuple[str, ...]


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _device_labels(devices: Sequence[jax.Device]) -> list[str]:
    return [
        f"{device.platform}:{device.process_index}:{device.id}:{device.device_kind}"
        for device in devices
    ]


def _finite_pytree(tree: object) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _validate_root_indices(value: object) -> tuple[int, int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(
            "checkpoint next_root_sample_indices must contain three integers"
        )
    result = tuple(value)
    if any(
        isinstance(index, bool)
        or not isinstance(index, (int, np.integer))
        or int(index) < 0
        for index in result
    ):
        raise ValueError(
            "checkpoint next_root_sample_indices must contain three "
            "nonnegative integers"
        )
    return tuple(int(index) for index in result)  # type: ignore[return-value]


def _resolved_identity(
    user_identity: Mapping[str, object],
    *,
    config: CompactTreeGRUConfig,
    bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
    total_steps: int,
    samples_per_root: int,
    learning_rate: float,
    sample_key: jax.Array,
    registered_widths: Sequence[int],
    sampling_microbatch_size: int,
    action_microbatch_size: int,
    relative_rank_tolerance: float,
    maximum_condition_number: float | None,
    exact_normalized_overlap_diagonal: bool,
    devices: Sequence[jax.Device],
    optax_version: str,
    initial_params: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Resolve every trajectory-affecting choice into a stable identity."""

    return {
        "schema": "resumable_three_root_compact_tree_adam_v1",
        "campaign": dict(user_identity),
        "scientific_scope": D1A_TREE_GATE_SCIENTIFIC_SCOPE,
        "model": {
            "root_count": 3,
            "independent_parameters": True,
            "tree": repr(config.tree.root),
            "n_sites": config.n,
            "target_q": config.target_q,
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "initial_root_params_sha256": [
                pytree_sha256(root) for root in initial_params
            ],
        },
        "hamiltonian_bonds": [list(map(float, bond)) for bond in bonds],
        "optimizer": {
            "name": "three_independent_optax.adam",
            "version": optax_version,
            "learning_rate": float(learning_rate),
        },
        "schedule": {
            "total_steps": int(total_steps),
            "samples_per_root_per_step": int(samples_per_root),
            "sample_key": np.asarray(sample_key, dtype=np.uint32).tolist(),
            "indexed_sampling": True,
            "proposal": D1A_TREE_GATE_PROPOSAL,
        },
        "sampled_ritz": {
            "relative_rank_tolerance": float(relative_rank_tolerance),
            "maximum_condition_number": (
                None
                if maximum_condition_number is None
                else float(maximum_condition_number)
            ),
            "rank_policy": "full_rank_three_or_fail_closed",
            "overlap_diagonal_policy": (
                "exact_normalized"
                if exact_normalized_overlap_diagonal
                else "sampled_legacy"
            ),
            "hamiltonian_policy": "explicit_Hermitization_with_raw_diagnostic",
            "force": (
                "frozen_proposal_C_Araw_horizontal_residual_VJP_"
                + (
                    "exact_normalized_overlap_diagonal"
                    if exact_normalized_overlap_diagonal
                    else "sampled_overlap_diagonal_legacy"
                )
            ),
        },
        "execution": {
            "registered_widths": list(map(int, registered_widths)),
            "sampling_microbatch_size": int(sampling_microbatch_size),
            "action_microbatch_size": int(action_microbatch_size),
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
                    "XLA_CLIENT_MEM_FRACTION",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION",
                    "XLA_PYTHON_CLIENT_ALLOCATOR",
                    "XLA_FLAGS",
                )
            },
        },
    }


def _checkpoint_metadata(
    *,
    status: str,
    params: tuple[object, object, object],
    optimizer_states: tuple[object, object, object],
    next_root_sample_indices: tuple[int, int, int],
) -> dict[str, object]:
    return {
        "status": status,
        "scientific_scope": D1A_TREE_GATE_SCIENTIFIC_SCOPE,
        "proposal": D1A_TREE_GATE_PROPOSAL,
        "next_root_sample_indices": list(next_root_sample_indices),
        "root_params_sha256": [pytree_sha256(root) for root in params],
        "root_optimizer_state_sha256": [
            pytree_sha256(state) for state in optimizer_states
        ],
        "joint_params_sha256": pytree_sha256(params),
        "joint_optimizer_state_sha256": pytree_sha256(optimizer_states),
    }


def run_resumable_three_root_tree_adam(
    initial_params: Sequence[Mapping[str, object]],
    *,
    config: CompactTreeGRUConfig,
    bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
    total_steps: int,
    samples_per_root: int,
    learning_rate: float,
    sample_key: jax.Array,
    run_identity: Mapping[str, object],
    checkpoint_path: str | Path,
    resume: bool = False,
    checkpoint_every: int = 1,
    max_steps_this_invocation: int | None = None,
    registered_widths: Sequence[int] = DEFAULT_TREE_CONNECTIVITY_BUCKETS,
    sampling_microbatch_size: int = 256,
    action_microbatch_size: int = 8,
    relative_rank_tolerance: float = 1.0e-10,
    maximum_condition_number: float | None = None,
    exact_normalized_overlap_diagonal: bool = (
        D1_TREE_GATE_DEFAULT_EXACT_NORMALIZED_OVERLAP_DIAGONAL
    ),
    devices: Sequence[jax.Device] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> ResumableThreeRootTreeAdamResult:
    """Run or resume the three-root D1a engineering trajectory.

    A completed step consumes exactly ``samples_per_root`` indexed draws from
    each root stream.  Parameters, all three optimizer states, all three next
    indices, and the complete deterministic/timing ledger are committed only
    after the entire finite three-root update is accepted.

    The overlap-diagonal policy is bound into the checkpoint identity.  The
    force-capable runner currently accepts only the legacy sampled policy;
    exact normalized diagonals remain available for standalone Ritz analysis
    but are rejected here until their force estimator passes phase-gauge tests.
    """

    roots = tuple(initial_params)
    if len(roots) != 3:
        raise ValueError("three independent initial parameter roots are required")
    if config.target_q != 0:
        raise ValueError("the D1a engineering runner is restricted to target_q=0")
    if total_steps < 1 or samples_per_root < 1 or checkpoint_every < 1:
        raise ValueError(
            "total_steps, samples_per_root, and checkpoint_every must be positive"
        )
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if max_steps_this_invocation is not None and max_steps_this_invocation < 1:
        raise ValueError("max_steps_this_invocation must be positive")
    if not isinstance(exact_normalized_overlap_diagonal, (bool, np.bool_)):
        raise TypeError("exact_normalized_overlap_diagonal must be boolean")
    if exact_normalized_overlap_diagonal:
        raise ValueError(
            "exact normalized overlap diagonals are Ritz-only and are not "
            "available in the resumable Ky-Fan force trajectory"
        )
    if not _finite_pytree(roots):
        raise FloatingPointError("initial three-root parameters must be finite")
    selected_devices = tuple(jax.local_devices() if devices is None else devices)
    if len(selected_devices) != 1:
        raise ValueError("the D1a runner requires exactly one JAX device")
    try:
        import optax  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "resumable three-root Adam requires the train extra (optax)"
        ) from error

    identity = _resolved_identity(
        run_identity,
        config=config,
        bonds=bonds,
        total_steps=total_steps,
        samples_per_root=samples_per_root,
        learning_rate=learning_rate,
        sample_key=sample_key,
        registered_widths=registered_widths,
        sampling_microbatch_size=sampling_microbatch_size,
        action_microbatch_size=action_microbatch_size,
        relative_rank_tolerance=relative_rank_tolerance,
        maximum_condition_number=maximum_condition_number,
        exact_normalized_overlap_diagonal=(
            exact_normalized_overlap_diagonal
        ),
        devices=selected_devices,
        optax_version=getattr(optax, "__version__", "unknown"),
        initial_params=roots,
    )
    optimizer = optax.adam(learning_rate)
    destination = Path(checkpoint_path)
    if resume:
        checkpoint = load_latest_resumable_checkpoint(
            destination, expected_run_identity=identity
        )
        checkpoint_failures = tuple(checkpoint["newer_generation_failures"])
        loaded_params = checkpoint["params"]
        loaded_states = checkpoint["optimizer_state"]
        if not isinstance(loaded_params, tuple) or len(loaded_params) != 3:
            raise ValueError("checkpoint does not contain three parameter roots")
        if not isinstance(loaded_states, tuple) or len(loaded_states) != 3:
            raise ValueError("checkpoint does not contain three optimizer states")
        params = loaded_params
        optimizer_states = loaded_states
        completed_steps = int(checkpoint["completed_steps"])
        next_sample_index = int(checkpoint["next_sample_index"])
        history = [dict(row) for row in checkpoint["history"]]  # type: ignore[arg-type]
        metadata = checkpoint["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint metadata must be an object")
        next_root_sample_indices = _validate_root_indices(
            metadata.get("next_root_sample_indices")
        )
        expected_root_index = completed_steps * samples_per_root
        if completed_steps > total_steps:
            raise ValueError("checkpoint is beyond the registered total_steps")
        if next_root_sample_indices != (expected_root_index,) * 3:
            raise ValueError("checkpoint root sample counters do not match its step")
        if next_sample_index != sum(next_root_sample_indices):
            raise ValueError("checkpoint aggregate sample counter is inconsistent")
        if not _finite_pytree((params, optimizer_states)):
            raise FloatingPointError("checkpoint contains non-finite state")
    else:
        if destination.exists() and not destination.is_dir():
            raise FileExistsError(
                f"checkpoint series path is not a directory: {destination}"
            )
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(
                f"checkpoint series is not empty: {destination}; pass resume=True"
            )
        params = roots
        optimizer_states = tuple(optimizer.init(root) for root in params)
        completed_steps = 0
        next_root_sample_indices = (0, 0, 0)
        next_sample_index = 0
        history: list[dict[str, object]] = []
        save_resumable_checkpoint_generation(
            destination,
            params,
            optimizer_state=optimizer_states,
            completed_steps=0,
            next_sample_index=0,
            run_identity=identity,
            history=history,
            metadata=_checkpoint_metadata(
                status="initialized",
                params=params,
                optimizer_states=optimizer_states,
                next_root_sample_indices=next_root_sample_indices,
            ),
        )
        checkpoint_failures: tuple[str, ...] = ()

    invocation_stop = total_steps
    if max_steps_this_invocation is not None:
        invocation_stop = min(
            invocation_stop, completed_steps + max_steps_this_invocation
        )
    for step in range(completed_steps + 1, invocation_stop + 1):
        if stop_requested is not None and stop_requested():
            break
        before_root_indices = next_root_sample_indices
        before_sha256 = pytree_sha256(params)
        gate = compact_tree_three_root_ritz_force_gate(
            params,
            sample_key,
            samples_per_root,
            config,
            bonds,
            starting_root_sample_indices=before_root_indices,
            registered_widths=registered_widths,
            sampling_microbatch_size=sampling_microbatch_size,
            action_microbatch_size=action_microbatch_size,
            relative_rank_tolerance=relative_rank_tolerance,
            maximum_condition_number=maximum_condition_number,
            exact_normalized_overlap_diagonal=(
                exact_normalized_overlap_diagonal
            ),
            devices=selected_devices,
        )
        if gate.scientific_scope != D1A_TREE_GATE_SCIENTIFIC_SCOPE:
            raise RuntimeError("three-root gate returned an unexpected scientific scope")
        if gate.proposal_kind != D1A_TREE_GATE_PROPOSAL:
            raise RuntimeError("three-root gate returned an unexpected proposal")
        gradients = tuple(gate.force.gradient)
        if len(gradients) != 3 or not _finite_pytree(gradients):
            raise FloatingPointError("non-finite or incomplete three-root force rejected")
        reference = gate.reference
        finite_diagnostics = (
            np.asarray(reference.ritz.energies),
            np.asarray(reference.ritz.overlap_eigenvalues),
            np.asarray(reference.physical_residuals.residual_norms),
            np.asarray(gate.root_ratio_kish_ess_fractions),
        )
        if not all(np.all(np.isfinite(value)) for value in finite_diagnostics):
            raise FloatingPointError("non-finite sampled Ritz diagnostic rejected")

        candidate_params = []
        candidate_states = []
        for root in range(3):
            updates, state = optimizer.update(
                gradients[root], optimizer_states[root], params[root]
            )
            candidate_params.append(optax.apply_updates(params[root], updates))
            candidate_states.append(state)
        updated_params = tuple(candidate_params)
        updated_states = tuple(candidate_states)
        if not _finite_pytree((updated_params, updated_states)):
            raise FloatingPointError("non-finite three-root Adam update rejected")

        params = updated_params  # type: ignore[assignment]
        optimizer_states = updated_states  # type: ignore[assignment]
        next_root_sample_indices = gate.samples.next_root_sample_indices
        next_sample_index = sum(next_root_sample_indices)
        completed_steps = step
        execution = gate.execution
        per_root_gradient_norms = [
            float(
                np.sqrt(
                    sum(
                        float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2))
                        for leaf in jax.tree_util.tree_leaves(gradient)
                    )
                )
            )
            for gradient in gradients
        ]
        history.append(
            {
                "step": step,
                "root_sample_index_start": list(before_root_indices),
                "root_sample_index_stop": list(next_root_sample_indices),
                "aggregate_sample_index_start": sum(before_root_indices),
                "aggregate_sample_index_stop": next_sample_index,
                "mean_ritz_energy_before_update": reference.mean_ritz_energy,
                "ritz_energies_before_update": reference.ritz.energies.tolist(),
                "physical_residual_norms_before_update": (
                    reference.physical_residuals.residual_norms.tolist()
                ),
                "overlap_eigenvalues": reference.ritz.overlap_eigenvalues.tolist(),
                "overlap_condition_number": (
                    reference.ritz.overlap_condition_number
                ),
                "overlap_orthonormality_error": (
                    reference.ritz.orthonormality_error
                ),
                "overlap_diagonal_policy": (
                    reference.overlap_diagonal_policy
                ),
                "hamiltonian_antihermitian_maximum_absolute_error": (
                    reference.moments.hamiltonian_hermiticity.maximum_absolute_error
                ),
                "hamiltonian_antihermitian_relative_frobenius_error": (
                    reference.moments.hamiltonian_hermiticity.relative_frobenius_error
                ),
                "maximum_projected_physical_residual": (
                    reference.physical_residuals.maximum_projected_residual
                ),
                "maximum_horizontality_error": (
                    reference.maximum_horizontality_error
                ),
                "gradient_norm": gate.force.gradient_norm,
                "root_gradient_norms": per_root_gradient_norms,
                "root_ratio_kish_ess_fractions": (
                    gate.root_ratio_kish_ess_fractions.tolist()
                ),
                "maximum_root_importance_ratio": (
                    gate.maximum_root_importance_ratio
                ),
                "params_before_sha256": before_sha256,
                "params_after_sha256": pytree_sha256(params),
                "root_params_after_sha256": [
                    pytree_sha256(root) for root in params
                ],
                "scientific_scope": gate.scientific_scope,
                "proposal": gate.proposal_kind,
                "sampling_seconds": execution.sampling_seconds,
                "connectivity_seconds": execution.connectivity_seconds,
                "action_seconds": execution.action_compile_and_evaluate_seconds,
                "force_seconds": execution.force_compile_and_evaluate_seconds,
                "device": execution.device_label,
                "active_bucket_widths": list(execution.active_bucket_widths),
                "total_connectivity_entries": execution.total_connectivity_entries,
                "allocated_connectivity_slots": execution.allocated_connectivity_slots,
                "maximum_resident_connectivity_slots": (
                    execution.maximum_resident_connectivity_slots
                ),
                "padded_state_amplitude_evaluations": (
                    execution.padded_state_amplitude_evaluations
                ),
                "host_connectivity_bytes": execution.host_connectivity_bytes,
                "maximum_static_device_input_bytes_excluding_xla_workspace": (
                    execution.maximum_static_device_input_bytes_excluding_xla_workspace
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
                optimizer_state=optimizer_states,
                completed_steps=completed_steps,
                next_sample_index=next_sample_index,
                run_identity=identity,
                history=history,
                metadata=_checkpoint_metadata(
                    status=(
                        "complete" if completed_steps == total_steps else "partial"
                    ),
                    params=params,
                    optimizer_states=optimizer_states,
                    next_root_sample_indices=next_root_sample_indices,
                ),
            )
        if should_stop:
            break

    return ResumableThreeRootTreeAdamResult(
        params=params,  # type: ignore[arg-type]
        optimizer_states=optimizer_states,  # type: ignore[arg-type]
        completed_steps=completed_steps,
        next_root_sample_indices=next_root_sample_indices,
        next_sample_index=next_sample_index,
        history=tuple(history),
        run_identity=identity,
        resumed=resume,
        stopped_early=completed_steps < total_steps,
        recovered_checkpoint_failures=tuple(checkpoint_failures),
    )


__all__ = [
    "ResumableThreeRootTreeAdamResult",
    "run_resumable_three_root_tree_adam",
]
