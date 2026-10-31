"""Root-conditioned exact-mixture proposal repair for the N=36 carrier bank.

V1 showed that one shared autoregressive distribution did not cover the
physically distinct DSL, 120-degree, and stripe state weights.  This module
implements its preregistered repair: one full-support fixed-``Sz`` proposal
per carrier and an exactly evaluated equal mixture for diagnostics and later
multiple-importance use.  No Hamiltonian logic lives here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from su2_nqs.models.sz_gru import SzGRUConfig
from su2_nqs.vmc.induced_three_carrier_proposal import (
    CARRIER_LABELS,
    FourStreamCursors,
    FourStreamSeeds,
)
from su2_nqs.vmc.resumable_checkpoint import (
    load_latest_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)
from su2_nqs.vmc.sz_proposal_preflight import (
    full_support_sz_mixture_log_probabilities,
    initialize_proposal_adam,
    log_weight_diagnostics,
    normalized_log_importance_weights,
    proposal_adam_update,
    sample_indexed_full_support_sz_mixture,
    state_weight_audit,
    weighted_sz_negative_log_likelihood,
)


_MIXTURE_SELECTION_DOMAIN = 2**32 - 2


class IndexedRootMixtureSample(NamedTuple):
    """Cursor-addressable sample from the equal mixture of three proposals."""

    configurations: jax.Array
    log_probabilities: jax.Array
    proposal_components: jax.Array
    uniform_components: jax.Array
    sample_indices: jax.Array


def _validated_params_by_root(params_by_root: Mapping[str, object]) -> None:
    if tuple(params_by_root) != CARRIER_LABELS:
        raise ValueError("proposal parameter mapping must follow fixed carrier order")


def exact_root_mixture_log_probabilities(
    params_by_root: Mapping[str, object],
    configurations: object,
    config: SzGRUConfig,
    epsilon: float,
) -> jax.Array:
    """Evaluate ``log(sum_r q_r,epsilon / 3)`` for every state."""

    _validated_params_by_root(params_by_root)
    rows = jnp.stack(
        tuple(
            full_support_sz_mixture_log_probabilities(
                params_by_root[label], configurations, config, epsilon
            )
            for label in CARRIER_LABELS
        ),
        axis=1,
    )
    return jax.scipy.special.logsumexp(rows, axis=1) - jnp.log(
        float(len(CARRIER_LABELS))
    )


def sample_indexed_exact_root_mixture(
    params_by_root: Mapping[str, object],
    master_key: jax.Array,
    *,
    start_index: int,
    n_samples: int,
    config: SzGRUConfig,
    epsilon: float,
) -> IndexedRootMixtureSample:
    """Sample the exact mixture with identities independent of chunking."""

    _validated_params_by_root(params_by_root)
    candidates = tuple(
        sample_indexed_full_support_sz_mixture(
            params_by_root[label],
            jax.random.fold_in(master_key, root),
            start_index=start_index,
            n_samples=n_samples,
            config=config,
            epsilon=epsilon,
        )
        for root, label in enumerate(CARRIER_LABELS)
    )
    indices = candidates[0].sample_indices
    component_key = jax.random.fold_in(master_key, _MIXTURE_SELECTION_DOMAIN)
    component_keys = jax.vmap(lambda index: jax.random.fold_in(component_key, index))(
        indices
    )
    components = jax.vmap(
        lambda key: jax.random.randint(
            key, (), 0, len(CARRIER_LABELS), dtype=jnp.int32
        )
    )(component_keys)
    candidate_states = jnp.stack(
        tuple(sample.configurations for sample in candidates), axis=0
    )
    candidate_uniform = jnp.stack(
        tuple(sample.uniform_component for sample in candidates), axis=0
    )
    rows = jnp.arange(n_samples, dtype=jnp.int32)
    states = candidate_states[components, rows]
    uniform = candidate_uniform[components, rows]
    return IndexedRootMixtureSample(
        configurations=states,
        log_probabilities=exact_root_mixture_log_probabilities(
            params_by_root, states, config, epsilon
        ),
        proposal_components=components,
        uniform_components=uniform,
        sample_indices=indices,
    )


@dataclass(frozen=True)
class RootMixtureFitState:
    """Complete restart state for one proposal per registered carrier."""

    params_by_root: dict[str, object]
    optimizer_state_by_root: dict[str, object]
    completed_steps: int
    cursors: FourStreamCursors
    frozen_root_log_normalizers: np.ndarray
    calibration_summary: dict[str, object]
    history: tuple[dict[str, object], ...]


def initialize_root_mixture_calibration(
    params_by_root: Mapping[str, object],
    *,
    stream_seeds: FourStreamSeeds,
    config: SzGRUConfig,
    epsilon: float,
    calibration_base_sample_count: int,
    target_log_numerators: Callable[[jax.Array], np.ndarray],
    tail_fraction: float,
) -> RootMixtureFitState:
    """Freeze three root normalizers from an independent initial-mixture draw."""

    _validated_params_by_root(params_by_root)
    base = int(calibration_base_sample_count)
    if base < 2:
        raise ValueError("calibration M must be at least two")
    sample = sample_indexed_exact_root_mixture(
        params_by_root,
        stream_seeds.key("calibration"),
        start_index=0,
        n_samples=2 * base,
        config=config,
        epsilon=epsilon,
    )
    logs = np.asarray(target_log_numerators(sample.configurations), dtype=np.float64)
    log_q = np.asarray(sample.log_probabilities, dtype=np.float64)
    if logs.shape != (2 * base, len(CARRIER_LABELS)):
        raise ValueError("target callback must return all three carrier columns")
    prefix = [
        log_weight_diagnostics(
            logs[:base, root] - log_q[:base], tail_fraction=tail_fraction
        ).__dict__
        for root in range(len(CARRIER_LABELS))
    ]
    doubled = [
        log_weight_diagnostics(
            logs[:, root] - log_q, tail_fraction=tail_fraction
        ).__dict__
        for root in range(len(CARRIER_LABELS))
    ]
    frozen = np.asarray(
        [row["log_normalizer"] for row in doubled], dtype=np.float64
    )
    component_counts = np.bincount(
        np.asarray(sample.proposal_components), minlength=len(CARRIER_LABELS)
    )
    summary = {
        "design": "independent_initial_exact_mixture_nested_prefix_M_and_2M",
        "base_sample_count": base,
        "doubled_sample_count": 2 * base,
        "prefix_M_roots": prefix,
        "doubled_2M_roots": doubled,
        "frozen_root_log_normalizers": frozen.tolist(),
        "sample_indices_are_nested_prefix": bool(
            np.array_equal(np.asarray(sample.sample_indices), np.arange(2 * base))
        ),
        "proposal_component_counts": component_counts.tolist(),
        "uniform_component_fraction": float(
            np.mean(np.asarray(sample.uniform_components))
        ),
    }
    return RootMixtureFitState(
        params_by_root=dict(params_by_root),
        optimizer_state_by_root={
            label: initialize_proposal_adam(params_by_root[label])
            for label in CARRIER_LABELS
        },
        completed_steps=0,
        cursors=FourStreamCursors(calibration=2 * base),
        frozen_root_log_normalizers=frozen,
        calibration_summary=summary,
        history=(),
    )


def fit_root_mixture_step(
    state: RootMixtureFitState,
    *,
    stream_seeds: FourStreamSeeds,
    config: SzGRUConfig,
    epsilon: float,
    fit_batch_size_per_root: int,
    validation_batch_size: int,
    learning_rate: float,
    target_log_numerators: Callable[[jax.Array], np.ndarray],
    tail_fraction: float,
) -> RootMixtureFitState:
    """Update each carrier proposal once, then diagnose their exact mixture."""

    _validated_params_by_root(state.params_by_root)
    if state.cursors.calibration <= 0 or state.cursors.audit != 0:
        raise ValueError("fit requires frozen calibration and untouched audit")
    updated_params: dict[str, object] = {}
    updated_optimizers: dict[str, object] = {}
    root_history = []
    for root, label in enumerate(CARRIER_LABELS):
        key = jax.random.fold_in(stream_seeds.key("fit"), root)
        sample = sample_indexed_full_support_sz_mixture(
            state.params_by_root[label],
            key,
            start_index=state.cursors.fit,
            n_samples=int(fit_batch_size_per_root),
            config=config,
            epsilon=epsilon,
        )
        target = jnp.asarray(
            target_log_numerators(sample.configurations)[:, root],
            dtype=jnp.float64,
        )
        weights = normalized_log_importance_weights(
            target, sample.log_probabilities
        )
        loss_function = lambda candidate: weighted_sz_negative_log_likelihood(
            candidate, sample.configurations, weights, config
        )
        loss, gradients = jax.value_and_grad(loss_function)(
            state.params_by_root[label]
        )
        candidate, optimizer = proposal_adam_update(
            state.params_by_root[label],
            gradients,
            state.optimizer_state_by_root[label],
            learning_rate=learning_rate,
        )
        updated_params[label] = candidate
        updated_optimizers[label] = optimizer
        gradient_norm = float(
            np.sqrt(
                sum(
                    float(np.sum(np.abs(np.asarray(leaf)) ** 2))
                    for leaf in jax.tree_util.tree_leaves(gradients)
                )
            )
        )
        root_history.append(
            {
                "carrier": label,
                "weighted_negative_log_likelihood": float(loss),
                "gradient_l2_norm": gradient_norm,
                "fit_uniform_component_fraction": float(
                    np.mean(np.asarray(sample.uniform_component))
                ),
            }
        )

    validation = sample_indexed_exact_root_mixture(
        updated_params,
        stream_seeds.key("validation"),
        start_index=state.cursors.validation,
        n_samples=int(validation_batch_size),
        config=config,
        epsilon=epsilon,
    )
    validation_roots = np.asarray(
        target_log_numerators(validation.configurations), dtype=np.float64
    )
    validation_audit = state_weight_audit(
        validation_roots,
        np.asarray(validation.log_probabilities, dtype=np.float64),
        root_log_normalizers=state.frozen_root_log_normalizers,
        tail_fraction=tail_fraction,
    )
    component_counts = np.bincount(
        np.asarray(validation.proposal_components), minlength=len(CARRIER_LABELS)
    )
    row = {
        "step": state.completed_steps + 1,
        "fit_sample_index_start_per_root": state.cursors.fit,
        "fit_sample_index_stop_per_root": (
            state.cursors.fit + int(fit_batch_size_per_root)
        ),
        "validation_sample_index_start": state.cursors.validation,
        "validation_sample_index_stop": (
            state.cursors.validation + int(validation_batch_size)
        ),
        "root_updates": root_history,
        "validation_state_weight_audit": validation_audit,
        "validation_proposal_component_counts": component_counts.tolist(),
        "validation_uniform_component_fraction": float(
            np.mean(np.asarray(validation.uniform_components))
        ),
        "calibration_stream_cursor": state.cursors.calibration,
        "audit_stream_cursor": 0,
    }
    return RootMixtureFitState(
        params_by_root=updated_params,
        optimizer_state_by_root=updated_optimizers,
        completed_steps=state.completed_steps + 1,
        cursors=FourStreamCursors(
            calibration=state.cursors.calibration,
            fit=state.cursors.fit + int(fit_batch_size_per_root),
            validation=state.cursors.validation + int(validation_batch_size),
            audit=0,
        ),
        frozen_root_log_normalizers=np.asarray(
            state.frozen_root_log_normalizers, dtype=np.float64
        ),
        calibration_summary=dict(state.calibration_summary),
        history=(*state.history, row),
    )


def _validate_identity(
    run_identity: Mapping[str, object], stream_seeds: FourStreamSeeds
) -> None:
    required = {
        "carrier_bank",
        "proposal",
        "source_sha256",
        "config_sha256",
        "stream_seeds",
    }
    if not required.issubset(run_identity):
        raise ValueError("root-mixture run identity is incomplete")
    if run_identity["stream_seeds"] != stream_seeds.as_dict():
        raise ValueError("root-mixture checkpoint stream seeds changed")


def save_root_mixture_checkpoint_generation(
    directory: str | Path,
    state: RootMixtureFitState,
    *,
    run_identity: Mapping[str, object],
    stream_seeds: FourStreamSeeds,
) -> Path:
    """Commit one atomic generation containing all three proposals."""

    _validate_identity(run_identity, stream_seeds)
    _validated_params_by_root(state.params_by_root)
    if len(state.history) != state.completed_steps or state.cursors.audit != 0:
        raise ValueError("invalid root-mixture checkpoint counters")
    return save_resumable_checkpoint_generation(
        directory,
        state.params_by_root,
        optimizer_state=state.optimizer_state_by_root,
        completed_steps=int(state.completed_steps),
        next_sample_index=int(state.cursors.fit),
        run_identity=run_identity,
        history=state.history,
        metadata={
            "kind": "n36_induced_root_conditioned_exact_mixture_v2",
            "stream_cursors": state.cursors.as_dict(),
            "stream_seeds": stream_seeds.as_dict(),
            "carrier_labels": list(CARRIER_LABELS),
            "frozen_root_log_normalizers": np.asarray(
                state.frozen_root_log_normalizers
            ).tolist(),
            "calibration_summary": state.calibration_summary,
            "audit_stream_used_during_fit": False,
        },
    )


def load_latest_root_mixture_checkpoint(
    directory: str | Path,
    *,
    run_identity: Mapping[str, object],
    stream_seeds: FourStreamSeeds,
) -> RootMixtureFitState:
    """Load and validate the latest complete three-proposal generation."""

    _validate_identity(run_identity, stream_seeds)
    payload = load_latest_resumable_checkpoint(
        directory, expected_run_identity=run_identity
    )
    metadata = payload.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("kind")
        != "n36_induced_root_conditioned_exact_mixture_v2"
        or metadata.get("stream_seeds") != stream_seeds.as_dict()
        or metadata.get("carrier_labels") != list(CARRIER_LABELS)
        or metadata.get("audit_stream_used_during_fit") is not False
        or not isinstance(metadata.get("stream_cursors"), Mapping)
        or not isinstance(metadata.get("calibration_summary"), Mapping)
    ):
        raise ValueError("invalid root-mixture checkpoint metadata")
    cursors = FourStreamCursors(**dict(metadata["stream_cursors"]))
    if cursors.fit != payload["next_sample_index"] or cursors.audit != 0:
        raise ValueError("root-mixture checkpoint cursors are inconsistent")
    params = dict(payload["params"])
    optimizers = dict(payload["optimizer_state"])
    _validated_params_by_root(params)
    if tuple(optimizers) != CARRIER_LABELS:
        raise ValueError("root-mixture optimizer mapping changed order")
    normalizers = np.asarray(
        metadata["frozen_root_log_normalizers"], dtype=np.float64
    )
    if normalizers.shape != (len(CARRIER_LABELS),) or np.any(~np.isfinite(normalizers)):
        raise ValueError("root-mixture normalizers are invalid")
    return RootMixtureFitState(
        params_by_root=params,
        optimizer_state_by_root=optimizers,
        completed_steps=int(payload["completed_steps"]),
        cursors=cursors,
        frozen_root_log_normalizers=normalizers,
        calibration_summary=dict(metadata["calibration_summary"]),
        history=tuple(dict(row) for row in payload["history"]),
    )


__all__ = [
    "IndexedRootMixtureSample",
    "RootMixtureFitState",
    "exact_root_mixture_log_probabilities",
    "fit_root_mixture_step",
    "initialize_root_mixture_calibration",
    "load_latest_root_mixture_checkpoint",
    "sample_indexed_exact_root_mixture",
    "save_root_mixture_checkpoint_generation",
]
