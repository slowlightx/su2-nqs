"""Sampled local-energy boundary for the many-cell simplex ansatz."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..models.many_cell_simplex_vit import (
    ManyCellSimplexViTConfig,
    frozen_backbone_simplex_log_amplitudes,
    many_cell_simplex_log_amplitudes,
    sample_many_cell_simplex_vit,
)
from ..models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,
    additive_log_amplitudes,
    many_cell_schur_log_amplitudes,
    sample_many_cell_schur_vit,
)
from ..su2.wigner6j import f_move
from .many_cell_tree import ManyCellTreeConnectivity


@dataclass(frozen=True)
class ManyCellSimplexLocalEnergy:
    """Exact local F-move ratios on a sampled batch."""

    ratios: np.ndarray
    local_energies: np.ndarray
    amplitude_seconds: float


@dataclass(frozen=True)
class ManyCellSimplexAdditiveProposal:
    """Known mixture proposal for a carrier plus simplex residual.

    The proposal is a classical mixture of two independently normalized
    autoregressive Born laws.  The target remains their *coherent* additive
    sum, so the returned proposal probability must be retained for every
    importance-corrected estimator.
    """

    labels: jax.Array
    log_proposal_probability: jax.Array
    log_target_amplitude: jax.Array
    carrier_draw: jax.Array


@dataclass(frozen=True)
class ManyCellSimplexAdditiveEvaluation:
    """Exact sampled ``psi`` and ``H psi`` for an additive simplex state."""

    center_amplitudes: np.ndarray
    action_amplitudes: np.ndarray
    ratios: np.ndarray
    local_energies: np.ndarray
    amplitude_seconds: float


@partial(jax.jit, static_argnames=("config",))
def _simplex_local_energy_kernel(
    head,
    backbone,
    centers,
    connected,
    coefficients,
    mask,
    *,
    config: ManyCellSimplexViTConfig,
):
    center_logs = frozen_backbone_simplex_log_amplitudes(
        head, centers, config, backbone
    )
    flat = connected.reshape((-1, config.n - 1))
    connected_logs = frozen_backbone_simplex_log_amplitudes(
        head, flat, config, backbone
    ).reshape(connected.shape[:2])
    ratios = jnp.exp(connected_logs - center_logs[:, None])
    ratios = jnp.where(mask, ratios, 0.0 + 0.0j)
    return ratios, jnp.sum(coefficients * ratios, axis=1)


def evaluate_many_cell_simplex_local_energy(
    head: dict[str, jax.Array],
    backbone: dict[str, object],
    workload: ManyCellTreeConnectivity,
    config: ManyCellSimplexViTConfig,
) -> ManyCellSimplexLocalEnergy:
    """Evaluate exact ``H[new,old] psi(new)/psi(old)`` on a sampled batch."""

    started = time.perf_counter()
    ratios, local = _simplex_local_energy_kernel(
        head,
        backbone,
        jnp.asarray(workload.center_labels),
        jnp.asarray(workload.connected_labels),
        jnp.asarray(workload.coefficients),
        jnp.asarray(workload.mask),
        config=config,
    )
    ratios_host, local_host = jax.device_get((ratios, local))
    values = np.asarray(local_host)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("simplex local energies contain non-finite values")
    return ManyCellSimplexLocalEnergy(
        ratios=np.asarray(ratios_host),
        local_energies=values,
        amplitude_seconds=float(time.perf_counter() - started),
    )


def sample_many_cell_simplex_additive_proposal(
    carrier_params: dict[str, object],
    residual_params: dict[str, object],
    residual_scale: complex | float | jax.Array,
    key: jax.Array,
    sample_count: int,
    carrier_config: ManyCellSchurViTConfig,
    residual_config: ManyCellSimplexViTConfig,
    *,
    carrier_probability: float = 0.5,
) -> ManyCellSimplexAdditiveProposal:
    """Sample a support-safe proposal for ``carrier + scale * simplex``.

    ``carrier_config`` is deliberately separate from the residual geometry.
    Physical and spatially scrambled residuals can therefore share exactly
    the same immutable carrier rather than silently scrambling the anchor.
    """

    if carrier_config.layout.tree != residual_config.simplex_layout.schur.tree:
        raise ValueError("carrier and residual must use the same fusion tree")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    probability = float(carrier_probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("carrier_probability must lie strictly between zero and one")
    arm_key, carrier_key, residual_key = jax.random.split(key, 3)
    carrier_labels = sample_many_cell_schur_vit(
        carrier_params, carrier_key, sample_count, carrier_config
    )
    residual_labels = sample_many_cell_simplex_vit(
        residual_params, residual_key, sample_count, residual_config
    )
    carrier_draw = jax.random.bernoulli(
        arm_key, probability, (int(sample_count),)
    )
    labels = jnp.where(carrier_draw[:, None], carrier_labels, residual_labels)
    carrier_logs = many_cell_schur_log_amplitudes(
        carrier_params, labels, carrier_config
    )
    residual_logs = many_cell_simplex_log_amplitudes(
        residual_params, labels, residual_config
    )
    log_proposal = jnp.logaddexp(
        np.log(probability) + 2.0 * jnp.real(carrier_logs),
        np.log1p(-probability) + 2.0 * jnp.real(residual_logs),
    )
    target_logs = additive_log_amplitudes(
        carrier_logs, residual_logs, residual_scale
    )
    return ManyCellSimplexAdditiveProposal(
        labels=labels,
        log_proposal_probability=log_proposal,
        log_target_amplitude=target_logs,
        carrier_draw=carrier_draw,
    )


@partial(jax.jit, static_argnames=("carrier_config", "residual_config"))
def _simplex_additive_hpsi_kernel(
    carrier_params,
    residual_params,
    residual_scale,
    centers,
    connected,
    coefficients,
    mask,
    *,
    carrier_config: ManyCellSchurViTConfig,
    residual_config: ManyCellSimplexViTConfig,
):
    carrier_center_logs = many_cell_schur_log_amplitudes(
        carrier_params, centers, carrier_config
    )
    residual_center_logs = many_cell_simplex_log_amplitudes(
        residual_params, centers, residual_config
    )
    center_logs = additive_log_amplitudes(
        carrier_center_logs, residual_center_logs, residual_scale
    )
    flat = connected.reshape((-1, residual_config.n - 1))
    carrier_connected_logs = many_cell_schur_log_amplitudes(
        carrier_params, flat, carrier_config
    ).reshape(connected.shape[:2])
    residual_connected_logs = many_cell_simplex_log_amplitudes(
        residual_params, flat, residual_config
    ).reshape(connected.shape[:2])
    connected_logs = additive_log_amplitudes(
        carrier_connected_logs, residual_connected_logs, residual_scale
    )
    center_amplitudes = jnp.exp(center_logs)
    connected_amplitudes = jnp.exp(connected_logs)
    ratios = connected_amplitudes / center_amplitudes[:, None]
    ratios = jnp.where(mask, ratios, 0.0 + 0.0j)
    actions = jnp.sum(coefficients * connected_amplitudes * mask, axis=1)
    local = actions / center_amplitudes
    return center_amplitudes, actions, ratios, local


def evaluate_many_cell_simplex_additive_local_energy(
    carrier_params: dict[str, object],
    residual_params: dict[str, object],
    residual_scale: complex | float | jax.Array,
    workload: ManyCellTreeConnectivity,
    carrier_config: ManyCellSchurViTConfig,
    residual_config: ManyCellSimplexViTConfig,
) -> ManyCellSimplexAdditiveEvaluation:
    """Evaluate exact sampled amplitudes and H-ratios of the nested state."""

    if carrier_config.layout.tree != residual_config.simplex_layout.schur.tree:
        raise ValueError("carrier and residual must use the same fusion tree")
    started = time.perf_counter()
    values = _simplex_additive_hpsi_kernel(
        carrier_params,
        residual_params,
        jnp.asarray(residual_scale, dtype=jnp.complex128),
        jnp.asarray(workload.center_labels),
        jnp.asarray(workload.connected_labels),
        jnp.asarray(workload.coefficients),
        jnp.asarray(workload.mask),
        carrier_config=carrier_config,
        residual_config=residual_config,
    )
    centers, actions, ratios, local = (
        np.asarray(value) for value in jax.device_get(values)
    )
    if any(
        not np.all(np.isfinite(value))
        for value in (centers, actions, ratios[workload.mask], local)
    ):
        raise FloatingPointError("additive simplex evaluation is non-finite")
    return ManyCellSimplexAdditiveEvaluation(
        center_amplitudes=centers,
        action_amplitudes=actions,
        ratios=ratios,
        local_energies=local,
        amplitude_seconds=float(time.perf_counter() - started),
    )


def audit_sampled_local_f_moves(
    labels: np.ndarray,
    config: ManyCellSimplexViTConfig,
) -> dict[str, float | int]:
    """Audit every selected local F column without constructing a dense basis."""

    rows = np.asarray(labels, dtype=np.int32)
    if rows.ndim != 2 or rows.shape[1] != config.n - 1:
        raise ValueError(f"labels must have shape (batch,{config.n - 1})")
    layout = config.simplex_layout
    cell_roots = layout.schur.cell_root_indices
    maximum_norm_error = 0.0
    resonant = 0
    total = 0
    off_diagonal_weights = []
    for row in rows:
        for simplex, group in enumerate(layout.cell_groups):
            q0, q1, q2 = (
                int(row[cell_roots[group[index]]]) for index in range(3)
            )
            q12 = int(row[layout.pair_indices[simplex]])
            q_total = int(row[layout.root_indices[simplex]])
            matrix, left, _ = f_move(q0, q1, q2, q_total)
            column = matrix[:, left.index(q12)]
            probabilities = np.square(column)
            maximum_norm_error = max(
                maximum_norm_error, abs(float(np.sum(probabilities)) - 1.0)
            )
            off_diagonal_weights.append(1.0 - float(np.max(probabilities)))
            resonant += int(len(column) > 1)
            total += 1
    return {
        "sample_count": int(len(rows)),
        "f_move_column_count": int(total),
        "maximum_f_column_norm_error": float(maximum_norm_error),
        "multi_channel_fraction": float(resonant / total),
        "mean_off_diagonal_weight": float(np.mean(off_diagonal_weights)),
    }


__all__ = [
    "ManyCellSimplexAdditiveEvaluation",
    "ManyCellSimplexAdditiveProposal",
    "ManyCellSimplexLocalEnergy",
    "audit_sampled_local_f_moves",
    "evaluate_many_cell_simplex_additive_local_energy",
    "evaluate_many_cell_simplex_local_energy",
    "sample_many_cell_simplex_additive_proposal",
]
