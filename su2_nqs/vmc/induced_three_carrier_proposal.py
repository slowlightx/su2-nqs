"""Full-support proposal fitting for the physical N=36 three-carrier bank.

This module is intentionally narrower than a VMC driver.  It evaluates the
fixed ``DSL / 120 / stripe`` carrier weights, fits an exactly normalized
fixed-``Sz`` proposal to their calibration-normalized equal mixture, and
persists every complete step.  It never constructs a Hamiltonian and never
uses energy to select a proposal.

The proposal itself is

``q_eps = (1-eps) q_GRU + eps / binom(36, 18)``.

Thus every fixed-magnetization configuration has positive probability even
when one or more physical carriers have a node.  The carrier amplitudes are
not floored or repaired.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.triangular_induced_parton import (
    PiZeroFluxOrbit,
    compressed_collective_tower_a1,
    projected_pi_zero_flux_a1_amplitudes,
)
from su2_nqs.models.sz_gru import SzGRUConfig
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


CARRIER_LABELS: tuple[str, str, str] = (
    "DSL_pi0_GammaA1",
    "collective_120_GammaA1",
    "collective_stripe_GammaA1",
)


@dataclass(frozen=True)
class FourStreamSeeds:
    """Independent calibration, fit, validation, and audit keys."""

    calibration: int
    fit: int
    validation: int
    audit: int

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= int(value) < 2**32
            for value in values
        ):
            raise ValueError("all proposal stream seeds must be uint32 integers")
        if len(set(int(value) for value in values)) != 4:
            raise ValueError(
                "calibration, fit, validation, and audit seeds must be distinct"
            )

    def key(self, stream: str) -> jax.Array:
        if stream not in ("calibration", "fit", "validation", "audit"):
            raise ValueError("unknown proposal stream")
        return jax.random.key(int(getattr(self, stream)))

    def as_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class FourStreamCursors:
    """Next unused global index in each proposal stream."""

    calibration: int = 0
    fit: int = 0
    validation: int = 0
    audit: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 0 <= int(value) < 2**32
            ):
                raise ValueError(f"{name} cursor must be a uint32 integer")

    def as_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}


class N36ThreeCarrierBatch(NamedTuple):
    """Amplitudes and node/cancellation diagnostics in fixed carrier order."""

    amplitudes: np.ndarray
    log_state_numerators: np.ndarray
    exact_nodes: np.ndarray
    cancellation_ratios: np.ndarray


def _validated_n36_sz0(configurations: object) -> np.ndarray:
    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] != 36:
        raise ValueError("configurations must have shape (nonempty batch, 36)")
    if np.any((states != 0) & (states != 1)):
        raise ValueError("configurations must contain only bits zero and one")
    if np.any(np.sum(states, axis=1) != 18):
        raise ValueError("every N=36 carrier configuration must have N_up=18")
    return states


def evaluate_n36_three_carrier_batch(
    configurations: object,
    *,
    orbit: PiZeroFluxOrbit | None = None,
    microbatch_size: int = 64,
) -> N36ThreeCarrierBatch:
    """Evaluate only the registered physical ``DSL / 120 / stripe`` bank.

    Exact zeros remain ``-inf`` in ``log_state_numerators``.  Cancellation
    ratios are observational diagnostics and are never used to modify an
    amplitude or importance weight.
    """

    states = _validated_n36_sz0(configurations)
    dsl = projected_pi_zero_flux_a1_amplitudes(
        states, orbit=orbit, microbatch_size=microbatch_size
    )
    tower_120 = compressed_collective_tower_a1(states, family="120")
    stripe = compressed_collective_tower_a1(states, family="stripe")
    amplitudes = np.stack(
        (
            np.asarray(dsl.amplitudes, dtype=np.float64),
            np.asarray(tower_120.amplitudes, dtype=np.float64),
            np.asarray(stripe.amplitudes, dtype=np.float64),
        ),
        axis=1,
    )
    cancellation = np.stack(
        (
            np.asarray(dsl.cancellation_ratio, dtype=np.float64),
            np.asarray(tower_120.cancellation_ratio, dtype=np.float64),
            np.asarray(stripe.cancellation_ratio, dtype=np.float64),
        ),
        axis=1,
    )
    exact_nodes = amplitudes == 0.0
    with np.errstate(divide="ignore", invalid="raise"):
        log_numerators = np.where(
            exact_nodes, -np.inf, 2.0 * np.log(np.abs(amplitudes))
        )
    return N36ThreeCarrierBatch(
        amplitudes=amplitudes,
        log_state_numerators=np.asarray(log_numerators, dtype=np.float64),
        exact_nodes=np.asarray(exact_nodes, dtype=bool),
        cancellation_ratios=np.asarray(cancellation, dtype=np.float64),
    )


def carrier_support_diagnostics(
    batch: N36ThreeCarrierBatch,
    *,
    near_cancellation_threshold: float,
) -> dict[str, object]:
    """Return finite, exact-node, common-node, and cancellation fractions."""

    threshold = float(near_cancellation_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("near_cancellation_threshold must be finite and positive")
    amplitudes = np.asarray(batch.amplitudes)
    nodes = np.asarray(batch.exact_nodes)
    cancellation = np.asarray(batch.cancellation_ratios)
    if (
        amplitudes.ndim != 2
        or amplitudes.shape[1] != len(CARRIER_LABELS)
        or nodes.shape != amplitudes.shape
        or cancellation.shape != amplitudes.shape
    ):
        raise ValueError("carrier batch has an invalid shape")
    nonfinite = ~np.isfinite(amplitudes) | ~np.isfinite(cancellation)
    near = (cancellation <= threshold) & ~nodes
    return {
        "sample_count": int(amplitudes.shape[0]),
        "labels": list(CARRIER_LABELS),
        "nonfinite_amplitude_or_cancellation_fraction": float(
            np.mean(np.any(nonfinite, axis=1))
        ),
        "each_carrier_exact_node_fraction": np.mean(nodes, axis=0).tolist(),
        "common_node_fraction": float(np.mean(np.all(nodes, axis=1))),
        "union_support_fraction": float(np.mean(~np.all(nodes, axis=1))),
        "each_carrier_near_cancellation_fraction": np.mean(near, axis=0).tolist(),
        "minimum_nonzero_cancellation_ratio": [
            (
                float(np.min(cancellation[:, root][cancellation[:, root] > 0.0]))
                if np.any(cancellation[:, root] > 0.0)
                else 0.0
            )
            for root in range(len(CARRIER_LABELS))
        ],
    }


@dataclass(frozen=True)
class InducedProposalFitState:
    """All state required to reproduce an interrupted proposal trajectory."""

    params: object
    optimizer_state: object
    completed_steps: int
    cursors: FourStreamCursors
    frozen_root_log_normalizers: np.ndarray
    calibration_summary: dict[str, object]
    history: tuple[dict[str, object], ...]


def initialize_calibrated_fit_state(
    params: object,
    *,
    stream_seeds: FourStreamSeeds,
    config: SzGRUConfig,
    epsilon: float,
    calibration_base_sample_count: int,
    target_log_numerators: Callable[[jax.Array], np.ndarray],
    tail_fraction: float,
) -> InducedProposalFitState:
    """Consume calibration once and freeze root normalizers before fitting."""

    base = int(calibration_base_sample_count)
    if base < 2:
        raise ValueError("calibration M must be at least two")
    sample = sample_indexed_full_support_sz_mixture(
        params,
        stream_seeds.key("calibration"),
        start_index=0,
        n_samples=2 * base,
        config=config,
        epsilon=epsilon,
    )
    logs = np.asarray(target_log_numerators(sample.configurations), dtype=np.float64)
    log_q = np.asarray(sample.log_probabilities, dtype=np.float64)
    if logs.shape != (2 * base, len(CARRIER_LABELS)):
        raise ValueError("target callback must return one column per physical carrier")
    prefix_rows = [
        asdict(
            log_weight_diagnostics(
                logs[:base, root] - log_q[:base], tail_fraction=tail_fraction
            )
        )
        for root in range(logs.shape[1])
    ]
    full_rows = [
        asdict(
            log_weight_diagnostics(
                logs[:, root] - log_q, tail_fraction=tail_fraction
            )
        )
        for root in range(logs.shape[1])
    ]
    frozen = np.asarray(
        [row["log_normalizer"] for row in full_rows], dtype=np.float64
    )
    summary = {
        "design": "independent_nested_prefix_M_and_2M_before_fit",
        "base_sample_count": base,
        "doubled_sample_count": 2 * base,
        "prefix_M_roots": prefix_rows,
        "doubled_2M_roots": full_rows,
        "frozen_root_log_normalizers": frozen.tolist(),
        "sample_indices_are_nested_prefix": bool(
            np.array_equal(np.asarray(sample.sample_indices), np.arange(2 * base))
        ),
        "uniform_component_fraction": float(
            np.mean(np.asarray(sample.uniform_component))
        ),
    }
    return InducedProposalFitState(
        params=params,
        optimizer_state=initialize_proposal_adam(params),
        completed_steps=0,
        cursors=FourStreamCursors(calibration=2 * base),
        frozen_root_log_normalizers=frozen,
        calibration_summary=summary,
        history=(),
    )


def fit_induced_proposal_step(
    state: InducedProposalFitState,
    *,
    stream_seeds: FourStreamSeeds,
    config: SzGRUConfig,
    epsilon: float,
    fit_batch_size: int,
    validation_batch_size: int,
    learning_rate: float,
    target_log_numerators: Callable[[jax.Array], np.ndarray],
    tail_fraction: float,
) -> InducedProposalFitState:
    """Fit once on the fixed calibration-normalized target and then validate."""

    if state.cursors.calibration <= 0 or state.cursors.audit != 0:
        raise ValueError("fit requires frozen calibration and untouched audit streams")
    fit_sample = sample_indexed_full_support_sz_mixture(
        state.params,
        stream_seeds.key("fit"),
        start_index=state.cursors.fit,
        n_samples=int(fit_batch_size),
        config=config,
        epsilon=epsilon,
    )
    root_logs = jnp.asarray(
        target_log_numerators(fit_sample.configurations), dtype=jnp.float64
    )
    expected = (int(fit_batch_size), len(CARRIER_LABELS))
    if root_logs.shape != expected:
        raise ValueError(f"target callback must return shape {expected}")
    normalized = root_logs - jnp.asarray(
        state.frozen_root_log_normalizers, dtype=jnp.float64
    )[None, :]
    log_target = jax.scipy.special.logsumexp(normalized, axis=1) - jnp.log(
        len(CARRIER_LABELS)
    )
    weights = normalized_log_importance_weights(
        log_target, fit_sample.log_probabilities
    )
    loss_function = lambda candidate: weighted_sz_negative_log_likelihood(
        candidate, fit_sample.configurations, weights, config
    )
    loss, gradients = jax.value_and_grad(loss_function)(state.params)
    updated_params, updated_optimizer = proposal_adam_update(
        state.params,
        gradients,
        state.optimizer_state,
        learning_rate=learning_rate,
    )

    validation_sample = sample_indexed_full_support_sz_mixture(
        updated_params,
        stream_seeds.key("validation"),
        start_index=state.cursors.validation,
        n_samples=int(validation_batch_size),
        config=config,
        epsilon=epsilon,
    )
    validation_roots = np.asarray(
        target_log_numerators(validation_sample.configurations), dtype=np.float64
    )
    validation_q = np.asarray(
        validation_sample.log_probabilities, dtype=np.float64
    )
    validation = state_weight_audit(
        validation_roots,
        validation_q,
        root_log_normalizers=state.frozen_root_log_normalizers,
        tail_fraction=tail_fraction,
    )
    gradient_norm = float(
        np.sqrt(
            sum(
                float(np.sum(np.abs(np.asarray(leaf)) ** 2))
                for leaf in jax.tree_util.tree_leaves(gradients)
            )
        )
    )
    row = {
        "step": state.completed_steps + 1,
        "fit_sample_index_start": state.cursors.fit,
        "fit_sample_index_stop": state.cursors.fit + int(fit_batch_size),
        "validation_sample_index_start": state.cursors.validation,
        "validation_sample_index_stop": (
            state.cursors.validation + int(validation_batch_size)
        ),
        "weighted_negative_log_likelihood": float(loss),
        "gradient_l2_norm": gradient_norm,
        "fit_uniform_component_fraction": float(
            np.mean(np.asarray(fit_sample.uniform_component))
        ),
        "validation_uniform_component_fraction": float(
            np.mean(np.asarray(validation_sample.uniform_component))
        ),
        "validation_state_weight_audit": validation,
        "calibration_stream_cursor": state.cursors.calibration,
        "audit_stream_cursor": 0,
    }
    return InducedProposalFitState(
        params=updated_params,
        optimizer_state=updated_optimizer,
        completed_steps=state.completed_steps + 1,
        cursors=FourStreamCursors(
            calibration=state.cursors.calibration,
            fit=state.cursors.fit + int(fit_batch_size),
            validation=state.cursors.validation + int(validation_batch_size),
            audit=0,
        ),
        frozen_root_log_normalizers=np.asarray(
            state.frozen_root_log_normalizers, dtype=np.float64
        ),
        calibration_summary=dict(state.calibration_summary),
        history=(*state.history, row),
    )


_IDENTITY_FIELDS = {
    "carrier_bank",
    "proposal",
    "source_sha256",
    "config_sha256",
    "stream_seeds",
}


def _validate_checkpoint_identity(
    run_identity: Mapping[str, object], stream_seeds: FourStreamSeeds
) -> None:
    missing = _IDENTITY_FIELDS - set(run_identity)
    if missing:
        raise ValueError(f"proposal run identity is missing fields: {sorted(missing)}")
    if run_identity["stream_seeds"] != stream_seeds.as_dict():
        raise ValueError("checkpoint stream seeds do not match the active streams")
    sources = run_identity["source_sha256"]
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("source_sha256 must be a nonempty mapping")


def save_induced_proposal_checkpoint_generation(
    directory: str | Path,
    state: InducedProposalFitState,
    *,
    run_identity: Mapping[str, object],
    stream_seeds: FourStreamSeeds,
) -> Path:
    """Commit one atomic, immutable calibrated-proposal generation."""

    _validate_checkpoint_identity(run_identity, stream_seeds)
    if len(state.history) != state.completed_steps:
        raise ValueError("proposal history length must equal completed steps")
    if state.cursors.calibration <= 0 or state.cursors.audit != 0:
        raise ValueError("calibration must be frozen and audit untouched")
    return save_resumable_checkpoint_generation(
        directory,
        state.params,
        optimizer_state=state.optimizer_state,
        completed_steps=int(state.completed_steps),
        next_sample_index=int(state.cursors.fit),
        run_identity=run_identity,
        history=state.history,
        metadata={
            "kind": "n36_induced_three_carrier_proposal_fit_v1",
            "stream_cursors": state.cursors.as_dict(),
            "stream_seeds": stream_seeds.as_dict(),
            "carrier_labels": list(CARRIER_LABELS),
            "frozen_root_log_normalizers": np.asarray(
                state.frozen_root_log_normalizers, dtype=np.float64
            ).tolist(),
            "calibration_summary": state.calibration_summary,
            "audit_stream_used_during_fit": False,
        },
    )


def load_latest_induced_proposal_checkpoint(
    directory: str | Path,
    *,
    run_identity: Mapping[str, object],
    stream_seeds: FourStreamSeeds,
) -> InducedProposalFitState:
    """Load the newest valid generation and restore all four stream cursors."""

    _validate_checkpoint_identity(run_identity, stream_seeds)
    payload = load_latest_resumable_checkpoint(
        directory, expected_run_identity=run_identity
    )
    metadata = payload.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("kind")
        != "n36_induced_three_carrier_proposal_fit_v1"
        or metadata.get("stream_seeds") != stream_seeds.as_dict()
        or metadata.get("carrier_labels") != list(CARRIER_LABELS)
        or metadata.get("audit_stream_used_during_fit") is not False
        or not isinstance(metadata.get("stream_cursors"), Mapping)
        or not isinstance(metadata.get("calibration_summary"), Mapping)
    ):
        raise ValueError("invalid induced-proposal checkpoint metadata")
    cursors = FourStreamCursors(**dict(metadata["stream_cursors"]))
    if (
        cursors.fit != payload["next_sample_index"]
        or cursors.calibration <= 0
        or cursors.audit != 0
    ):
        raise ValueError("induced-proposal checkpoint cursors are inconsistent")
    normalizers = np.asarray(
        metadata.get("frozen_root_log_normalizers"), dtype=np.float64
    )
    if normalizers.shape != (len(CARRIER_LABELS),) or np.any(~np.isfinite(normalizers)):
        raise ValueError("checkpoint root normalizers are invalid")
    return InducedProposalFitState(
        params=payload["params"],
        optimizer_state=payload["optimizer_state"],
        completed_steps=int(payload["completed_steps"]),
        cursors=cursors,
        frozen_root_log_normalizers=normalizers,
        calibration_summary=dict(metadata["calibration_summary"]),
        history=tuple(dict(row) for row in payload["history"]),
    )


def proposal_log_probabilities(
    params: object,
    configurations: object,
    config: SzGRUConfig,
    epsilon: float,
) -> np.ndarray:
    """Host convenience wrapper used by support and report audits."""

    return np.asarray(
        full_support_sz_mixture_log_probabilities(
            params, configurations, config, epsilon
        ),
        dtype=np.float64,
    )


__all__ = [
    "CARRIER_LABELS",
    "FourStreamCursors",
    "FourStreamSeeds",
    "InducedProposalFitState",
    "N36ThreeCarrierBatch",
    "carrier_support_diagnostics",
    "evaluate_n36_three_carrier_batch",
    "fit_induced_proposal_step",
    "initialize_calibrated_fit_state",
    "load_latest_induced_proposal_checkpoint",
    "proposal_log_probabilities",
    "save_induced_proposal_checkpoint_generation",
]
