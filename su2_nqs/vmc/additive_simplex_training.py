"""Resumable common-proposal training for additive simplex states.

The immutable carrier and the trainable simplex residual are independently
normalized autoregressive states in the same exact SU(2) fusion basis.  Their
coherent sum is not normalized autoregressively, so every update uses the
known classical mixture of their Born laws and an explicit importance
correction.  A same-batch energy line search rejects updates that damage the
carrier on the configurations which generated the force.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    additive_log_amplitudes,
    many_cell_schur_log_amplitudes,
)
from ..models.many_cell_simplex_vit import (
    ManyCellSimplexViTConfig,
    many_cell_simplex_log_amplitudes,
)
from .common_sample_vmc import common_sample_vmc_value_and_grad
from .many_cell_simplex import (
    evaluate_many_cell_simplex_additive_local_energy,
    sample_many_cell_simplex_additive_proposal,
)
from .many_cell_tree import build_many_cell_tree_connectivity
from .resumable_checkpoint import (
    load_latest_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class AdditiveSimplexTrainingControls:
    """Static optimizer, proposal, and checkpoint contract."""

    total_steps: int
    sample_count: int
    seed: int
    learning_rate: float
    carrier_probability: float
    minimum_ess_fraction: float
    maximum_variance_increase: float = 0.0
    line_search_factors: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    checkpoint_every_steps: int = 1

    def __post_init__(self) -> None:
        for name in ("total_steps", "sample_count", "checkpoint_every_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least two")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not 0.0 < float(self.carrier_probability) < 1.0:
            raise ValueError("carrier_probability must lie strictly between zero and one")
        if not 0.0 < float(self.minimum_ess_fraction) <= 1.0:
            raise ValueError("minimum_ess_fraction must lie in (0,1]")
        if (
            not math.isfinite(float(self.maximum_variance_increase))
            or self.maximum_variance_increase < 0.0
        ):
            raise ValueError("maximum_variance_increase must be finite and nonnegative")
        if not self.line_search_factors or any(
            not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0
            for value in self.line_search_factors
        ):
            raise ValueError("line_search_factors must be finite values in (0,1]")
        if any(
            left <= right
            for left, right in zip(
                self.line_search_factors,
                self.line_search_factors[1:],
            )
        ):
            raise ValueError("line_search_factors must be strictly decreasing")


def _center_amplitudes(
    params: object,
    labels: jax.Array,
    carrier_params: object,
    carrier_config: ManyCellSchurViTConfig,
    residual_config: ManyCellSimplexViTConfig,
) -> jax.Array:
    carrier_logs = many_cell_schur_log_amplitudes(
        carrier_params, labels, carrier_config
    )
    residual_logs = many_cell_simplex_log_amplitudes(
        params["residual"], labels, residual_config
    )
    return jnp.exp(
        additive_log_amplitudes(
            carrier_logs, residual_logs, params["residual_scale"]
        )
    )


def additive_simplex_importance_metrics(
    centers: np.ndarray,
    actions: np.ndarray,
    log_proposal_probability: np.ndarray,
) -> dict[str, float]:
    amplitudes = np.asarray(centers, dtype=np.complex128)
    hpsi = np.asarray(actions, dtype=np.complex128)
    log_q = np.asarray(log_proposal_probability, dtype=np.float64)
    if amplitudes.ndim != 1 or hpsi.shape != amplitudes.shape or log_q.shape != amplitudes.shape:
        raise ValueError("importance arrays must be matching rank-one vectors")
    if np.any(~np.isfinite(amplitudes)) or np.any(np.abs(amplitudes) == 0.0):
        raise FloatingPointError("target amplitudes must be finite and nonzero")
    log_weight = 2.0 * np.log(np.abs(amplitudes)) - log_q
    log_weight -= np.max(log_weight)
    raw = np.exp(log_weight)
    weights = raw / np.sum(raw)
    local = hpsi / amplitudes
    energy = np.sum(weights * local)
    variance = float(np.sum(weights * np.abs(local - energy) ** 2).real)
    ess = float(1.0 / np.sum(weights**2))
    return {
        "energy": float(energy.real),
        "energy_imaginary_abs": float(abs(energy.imag)),
        "variance": variance,
        "ess": ess,
        "ess_fraction": float(ess / len(weights)),
        "maximum_normalized_weight": float(np.max(weights)),
    }


def _gradient_norm(gradient: object) -> float:
    return float(
        np.sqrt(
            sum(
                float(np.sum(np.abs(np.asarray(leaf)) ** 2))
                for leaf in jax.tree_util.tree_leaves(gradient)
            )
        )
    )


def run_resumable_additive_simplex_training(
    initial_residual_params: object,
    initial_residual_scale: float,
    *,
    carrier_params: object,
    carrier_config: ManyCellSchurViTConfig,
    residual_config: ManyCellSimplexViTConfig,
    physical_bonds: Sequence[tuple[int, int] | tuple[int, int, float]],
    controls: AdditiveSimplexTrainingControls,
    checkpoint_directory: str | Path,
    run_identity: Mapping[str, object],
    maximum_steps_this_call: int | None = None,
) -> dict[str, object]:
    """Train the full residual backbone/head with an exact nested carrier."""

    if not math.isfinite(float(initial_residual_scale)):
        raise ValueError("initial_residual_scale must be finite")
    if maximum_steps_this_call is not None and (
        isinstance(maximum_steps_this_call, bool)
        or not isinstance(maximum_steps_this_call, Integral)
        or maximum_steps_this_call < 1
    ):
        raise ValueError("maximum_steps_this_call must be a positive integer")
    identity = {
        **dict(run_identity),
        "driver": "additive_simplex_common_proposal_gradient_v1",
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
        params = {
            "residual": initial_residual_params,
            "residual_scale": jnp.asarray(
                float(initial_residual_scale), dtype=jnp.float64
            ),
        }
        completed = 0
        history: list[dict[str, object]] = []
        resumed = False
    if completed > controls.total_steps:
        raise ValueError("checkpoint is beyond the registered total_steps")
    remaining = controls.total_steps - completed
    execute = remaining if maximum_steps_this_call is None else min(
        remaining, int(maximum_steps_this_call)
    )
    cumulative_samples = int(history[-1]["cumulative_samples"]) if history else 0
    cumulative_h_calls = int(history[-1]["cumulative_full_h_calls"]) if history else 0
    cumulative_wall = float(history[-1]["cumulative_wall_seconds"]) if history else 0.0
    latest_checkpoint: Path | None = None

    for step in range(completed + 1, completed + execute + 1):
        started = time.perf_counter()
        key = jax.random.fold_in(jax.random.PRNGKey(int(controls.seed)), step - 1)
        proposal = sample_many_cell_simplex_additive_proposal(
            carrier_params,
            params["residual"],
            params["residual_scale"],
            key,
            int(controls.sample_count),
            carrier_config,
            residual_config,
            carrier_probability=float(controls.carrier_probability),
        )
        labels = np.asarray(proposal.labels)
        log_q = np.asarray(proposal.log_proposal_probability)
        workload = build_many_cell_tree_connectivity(
            labels, carrier_config, physical_bonds
        )

        def evaluate(candidate):
            return evaluate_many_cell_simplex_additive_local_energy(
                carrier_params,
                candidate["residual"],
                candidate["residual_scale"],
                workload,
                carrier_config,
                residual_config,
            )

        def hpsi(candidate, _labels):
            evaluated = evaluate(candidate)
            return (
                jnp.asarray(evaluated.center_amplitudes),
                jnp.asarray(evaluated.action_amplitudes),
            )

        reference = jnp.exp(0.5 * jnp.asarray(log_q, dtype=jnp.float64))
        gradient = common_sample_vmc_value_and_grad(
            params,
            jnp.asarray(labels),
            reference,
            hpsi_fn=hpsi,
            center_fn=lambda candidate, rows: _center_amplitudes(
                candidate,
                rows,
                carrier_params,
                carrier_config,
                residual_config,
            ),
        )
        # Re-evaluate once to retain the exact amplitude/action arrays used by
        # the line search; the first pass belongs to the gradient work ledger.
        current_evaluation = evaluate(params)
        current = additive_simplex_importance_metrics(
            current_evaluation.center_amplitudes,
            current_evaluation.action_amplitudes,
            log_q,
        )
        norm = _gradient_norm(gradient.gradient)
        base_scale = 0.0 if norm <= 1.0e-14 else float(controls.learning_rate) / norm
        accepted = False
        chosen_factor = 0.0
        candidate_metrics = current
        candidate_evaluations = 0
        if base_scale > 0.0:
            for factor in controls.line_search_factors:
                candidate = jax.tree_util.tree_map(
                    lambda value, direction: value
                    - (base_scale * float(factor)) * direction,
                    params,
                    gradient.gradient,
                )
                evaluated = evaluate(candidate)
                candidate_evaluations += 1
                metrics = additive_simplex_importance_metrics(
                    evaluated.center_amplitudes,
                    evaluated.action_amplitudes,
                    log_q,
                )
                if (
                    metrics["ess_fraction"] >= controls.minimum_ess_fraction
                    and metrics["energy"] < current["energy"]
                    and metrics["variance"]
                    <= current["variance"] + controls.maximum_variance_increase
                ):
                    params = candidate
                    candidate_metrics = metrics
                    chosen_factor = float(factor)
                    accepted = True
                    break
        cumulative_samples += int(controls.sample_count)
        full_h_calls = 2 + candidate_evaluations
        cumulative_h_calls += full_h_calls
        cumulative_wall += float(time.perf_counter() - started)
        history.append(
            {
                "step": int(step),
                "energy_before_update": current["energy"],
                "variance_before_update": current["variance"],
                "ess_fraction_before_update": current["ess_fraction"],
                "energy_after_update": candidate_metrics["energy"],
                "variance_after_update": candidate_metrics["variance"],
                "ess_fraction_after_update": candidate_metrics["ess_fraction"],
                "gradient_energy_crosscheck": float(gradient.energy),
                "gradient_norm": norm,
                "base_gradient_scale": base_scale,
                "accepted_line_search_factor": chosen_factor,
                "accepted": bool(accepted),
                "carrier_draw_count": int(np.count_nonzero(np.asarray(proposal.carrier_draw))),
                "maximum_connectivity_width": int(np.max(workload.widths)),
                "full_h_calls": int(full_h_calls),
                "full_h_bond_terms": int(
                    full_h_calls * controls.sample_count * len(workload.physical_bonds)
                ),
                "cumulative_samples": int(cumulative_samples),
                "cumulative_full_h_calls": int(cumulative_h_calls),
                "cumulative_wall_seconds": float(cumulative_wall),
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
                optimizer_state={"kind": "normalized_gradient_backtracking_v1"},
                completed_steps=int(step),
                next_sample_index=int(cumulative_samples),
                run_identity=identity,
                history=history,
                metadata={
                    "cumulative_full_h_calls": int(cumulative_h_calls),
                    "cumulative_wall_seconds": float(cumulative_wall),
                },
            )

    return {
        "params": params,
        "history": history,
        "completed_steps": int(completed + execute),
        "target_steps": int(controls.total_steps),
        "resumed": bool(resumed),
        "latest_checkpoint": None if latest_checkpoint is None else str(latest_checkpoint),
        "cumulative_samples": int(cumulative_samples),
        "cumulative_full_h_calls": int(cumulative_h_calls),
        "cumulative_wall_seconds": float(cumulative_wall),
        "run_identity": identity,
    }


__all__ = [
    "AdditiveSimplexTrainingControls",
    "additive_simplex_importance_metrics",
    "run_resumable_additive_simplex_training",
]
