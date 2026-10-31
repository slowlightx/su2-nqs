"""Full-support learned ``Sz`` proposal preflight for strict carriers.

This module is deliberately independent of any carrier or Hamiltonian.  It
combines the exactly normalized :class:`~su2_nqs.models.sz_gru.SzGRU` Born
distribution with a constrained-uniform floor,

``q_eps(x) = (1-eps) q_GRU(x) + eps / binom(N, n_up)``.

The floor applies only inside the fixed-magnetization sector.  Proposal
probabilities are evaluated with ``logaddexp`` and samples are indexed by a
global cursor, so changing a microbatch boundary or resuming a job does not
change the sample stream.  Host-side state-weight diagnostics also remain in
log space.  They are proposal-coverage diagnostics, not energy or physics
estimators.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from math import ceil
from numbers import Integral
from pathlib import Path
from typing import Any, NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from ..models.sz_gru import SzGRUConfig, sample_sz, sz_log_psi
from .constrained_uniform import (
    sample_uniform_sz_configs,
    uniform_sz_log_probability,
)
from .resumable_checkpoint import (
    load_latest_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)


@dataclass(frozen=True)
class ProposalStreamSeeds:
    """Independent deterministic seeds for fit, validation, and final audit."""

    fit: int
    validation: int
    audit: int

    def __post_init__(self) -> None:
        values = (self.fit, self.validation, self.audit)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= int(value) < 2**32
            for value in values
        ):
            raise ValueError("proposal stream seeds must be uint32 integers")
        if len(set(int(value) for value in values)) != 3:
            raise ValueError("fit, validation, and audit seeds must be distinct")

    def key(self, stream: str) -> jax.Array:
        if stream not in ("fit", "validation", "audit"):
            raise ValueError("stream must be fit, validation, or audit")
        return jax.random.key(int(getattr(self, stream)))

    def as_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class ProposalStreamCursors:
    """Next unused global sample index in each independent stream."""

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

    def advanced(self, stream: str, count: int) -> "ProposalStreamCursors":
        if stream not in ("fit", "validation", "audit"):
            raise ValueError("stream must be fit, validation, or audit")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("cursor advance must be a non-negative integer")
        values = self.as_dict()
        values[stream] += int(count)
        return ProposalStreamCursors(**values)

    def as_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}


class IndexedSzMixtureSample(NamedTuple):
    """One cursor-addressable IID sample batch from ``q_eps``."""

    configurations: jax.Array
    log_probabilities: jax.Array
    uniform_component: jax.Array
    sample_indices: jax.Array


def _validate_epsilon(epsilon: float) -> float:
    value = float(epsilon)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("uniform-floor epsilon must lie strictly between zero and one")
    return value


def _valid_fixed_sz(configurations: jax.Array, config: SzGRUConfig) -> jax.Array:
    binary = jnp.all((configurations == 0) | (configurations == 1), axis=-1)
    return binary & (jnp.sum(configurations, axis=-1) == config.n_up)


@partial(jax.jit, static_argnames=("config",))
def sz_gru_log_probabilities(
    params: Mapping[str, object],
    configurations: jax.Array,
    config: SzGRUConfig,
) -> jax.Array:
    """Evaluate the exactly normalized GRU Born log probability in batch."""

    states = jnp.asarray(configurations, dtype=jnp.int32)
    if states.ndim != 2 or states.shape[1] != config.n:
        raise ValueError("configurations must have shape (batch, config.n)")
    return 2.0 * jnp.real(
        jax.vmap(lambda state: sz_log_psi(params, state, config))(states)
    )


def full_support_sz_mixture_log_probabilities(
    params: Mapping[str, object],
    configurations: jax.Array | np.ndarray | Sequence[Sequence[int]],
    config: SzGRUConfig,
    epsilon: float,
) -> jax.Array:
    """Evaluate ``log q_eps`` exactly, returning ``-inf`` outside the sector."""

    floor = _validate_epsilon(epsilon)
    states = jnp.asarray(configurations, dtype=jnp.int32)
    learned = sz_gru_log_probabilities(params, states, config)
    uniform = uniform_sz_log_probability(config.n, int(config.n_up))
    mixture = jnp.logaddexp(
        jnp.log1p(-floor) + learned,
        jnp.log(floor) + uniform,
    )
    return jnp.where(_valid_fixed_sz(states, config), mixture, -jnp.inf)


@partial(jax.jit, static_argnames=("n_samples", "config"))
def _sample_indexed_kernel(
    params: Mapping[str, object],
    master_key: jax.Array,
    start_index: jax.Array,
    epsilon: jax.Array,
    *,
    n_samples: int,
    config: SzGRUConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    indices = start_index + jnp.arange(n_samples, dtype=jnp.uint32)

    def draw_one(index: jax.Array) -> tuple[jax.Array, jax.Array]:
        indexed_key = jax.random.fold_in(master_key, index)
        component_key, learned_key, uniform_key = jax.random.split(indexed_key, 3)
        learned = sample_sz(params, learned_key, 1, config)[0]
        uniform = sample_uniform_sz_configs(
            uniform_key, 1, config.n, int(config.n_up)
        )[0]
        take_uniform = jax.random.bernoulli(component_key, epsilon)
        return jnp.where(take_uniform, uniform, learned), take_uniform

    states, components = jax.vmap(draw_one)(indices)
    return states, components, indices


def sample_indexed_full_support_sz_mixture(
    params: Mapping[str, object],
    master_key: jax.Array,
    *,
    start_index: int,
    n_samples: int,
    config: SzGRUConfig,
    epsilon: float,
) -> IndexedSzMixtureSample:
    """Draw IID ``q_eps`` samples whose identity is fixed by global indices."""

    floor = _validate_epsilon(epsilon)
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, Integral)
        or start_index < 0
        or isinstance(n_samples, bool)
        or not isinstance(n_samples, Integral)
        or n_samples < 1
        or int(start_index) + int(n_samples) > 2**32
    ):
        raise ValueError("sample index interval must be a nonempty uint32 interval")
    states, components, indices = _sample_indexed_kernel(
        params,
        master_key,
        jnp.asarray(int(start_index), dtype=jnp.uint32),
        jnp.asarray(floor, dtype=jnp.float64),
        n_samples=int(n_samples),
        config=config,
    )
    log_probabilities = full_support_sz_mixture_log_probabilities(
        params, states, config, floor
    )
    return IndexedSzMixtureSample(
        configurations=states,
        log_probabilities=log_probabilities,
        uniform_component=components,
        sample_indices=indices,
    )


@dataclass(frozen=True)
class LogWeightDiagnostics:
    """Stable host diagnostics for one unnormalised state-weight target."""

    sample_count: int
    positive_weight_count: int
    zero_weight_fraction: float
    log_normalizer: float
    log_normalizer_standard_error: float
    effective_sample_size: float
    effective_sample_size_fraction: float
    maximum_ratio_to_mean: float
    largest_normalized_weight: float
    tail_fraction: float
    top_tail_mass: float


def log_weight_diagnostics(
    log_importance_weights: np.ndarray | Sequence[float],
    *,
    tail_fraction: float = 0.01,
) -> LogWeightDiagnostics:
    """Compute ESS, maximum ratio, and upper-tail mass without exponent overflow."""

    values = np.asarray(log_importance_weights, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("log importance weights must be a vector of length at least two")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("log importance weights may contain only finite values or -inf")
    tail = float(tail_fraction)
    if not np.isfinite(tail) or not 0.0 < tail <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    finite = np.isfinite(values)
    if not np.any(finite):
        raise FloatingPointError("all target state weights are zero")
    maximum = float(np.max(values[finite]))
    scaled = np.zeros_like(values)
    scaled[finite] = np.exp(values[finite] - maximum)
    total = float(np.sum(scaled, dtype=np.float64))
    squared_total = float(np.sum(scaled * scaled, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0 or squared_total <= 0.0:
        raise FloatingPointError("state-weight accumulation failed")
    sample_count = len(values)
    mean = total / sample_count
    variance = float(np.var(scaled, ddof=1, dtype=np.float64))
    log_standard_error = np.sqrt(variance / sample_count) / mean
    effective = total * total / squared_total
    tail_count = max(1, int(ceil(tail * sample_count)))
    ordered = np.partition(scaled, sample_count - tail_count)
    tail_mass = float(np.sum(ordered[-tail_count:], dtype=np.float64) / total)
    return LogWeightDiagnostics(
        sample_count=sample_count,
        positive_weight_count=int(np.count_nonzero(finite)),
        zero_weight_fraction=float(1.0 - np.count_nonzero(finite) / sample_count),
        log_normalizer=float(maximum + np.log(mean)),
        log_normalizer_standard_error=float(log_standard_error),
        effective_sample_size=float(effective),
        effective_sample_size_fraction=float(effective / sample_count),
        maximum_ratio_to_mean=float(sample_count * np.max(scaled) / total),
        largest_normalized_weight=float(np.max(scaled) / total),
        tail_fraction=tail,
        top_tail_mass=tail_mass,
    )


def state_weight_audit(
    log_state_numerators: np.ndarray,
    log_proposal_probabilities: np.ndarray,
    *,
    root_log_normalizers: np.ndarray,
    tail_fraction: float = 0.01,
) -> dict[str, object]:
    """Audit per-root and equal-normalized-bank state weights.

    ``root_log_normalizers`` must be frozen from a non-audit stream.  This
    prevents arbitrary AGP column scales from changing the equal-root target
    and prevents the held-out audit from calibrating its own estimand.
    """

    numerator = np.asarray(log_state_numerators, dtype=np.float64)
    proposal = np.asarray(log_proposal_probabilities, dtype=np.float64)
    normalizers = np.asarray(root_log_normalizers, dtype=np.float64)
    if numerator.ndim != 2 or numerator.shape[0] < 2 or numerator.shape[1] < 1:
        raise ValueError("log_state_numerators must have shape (sample, root)")
    if proposal.shape != (numerator.shape[0],) or np.any(~np.isfinite(proposal)):
        raise ValueError("sample log proposal probabilities must be finite")
    if normalizers.shape != (numerator.shape[1],) or np.any(~np.isfinite(normalizers)):
        raise ValueError("one finite frozen log normalizer is required per root")
    if np.any(np.isnan(numerator)) or np.any(np.isposinf(numerator)):
        raise ValueError("state log numerators may contain only finite values or -inf")
    root_rows = [
        log_weight_diagnostics(
            numerator[:, root] - proposal, tail_fraction=tail_fraction
        )
        for root in range(numerator.shape[1])
    ]
    shifted = numerator - normalizers[None, :] - np.log(numerator.shape[1])
    maximum = np.max(shifted, axis=1)
    finite = np.isfinite(maximum)
    aggregate = np.full(numerator.shape[0], -np.inf, dtype=np.float64)
    if np.any(finite):
        aggregate[finite] = maximum[finite] + np.log(
            np.sum(np.exp(shifted[finite] - maximum[finite, None]), axis=1)
        )
    aggregate_row = log_weight_diagnostics(
        aggregate - proposal, tail_fraction=tail_fraction
    )
    return {
        "sample_count": int(numerator.shape[0]),
        "root_count": int(numerator.shape[1]),
        "frozen_root_log_normalizers": normalizers.tolist(),
        "roots": [asdict(row) for row in root_rows],
        "equal_normalized_bank": asdict(aggregate_row),
    }


def m_doubling_state_weight_audit(
    log_state_numerators: np.ndarray,
    log_proposal_probabilities: np.ndarray,
    *,
    base_sample_count: int,
    root_log_normalizers: np.ndarray,
    tail_fraction: float,
    minimum_ess_fraction: float,
    maximum_ratio_to_mean: float,
    maximum_top_tail_mass: float,
    maximum_absolute_log_normalizer_shift: float,
    standard_error_multiplier: float,
) -> dict[str, object]:
    """Compare nested ``M`` and ``2M`` audit prefixes using frozen gates."""

    base = int(base_sample_count)
    numerators = np.asarray(log_state_numerators, dtype=np.float64)
    proposal = np.asarray(log_proposal_probabilities, dtype=np.float64)
    if base < 2 or numerators.shape[0] != 2 * base or proposal.shape != (2 * base,):
        raise ValueError("the audit batch must contain exactly the nested 2M prefix")
    for name, value in (
        ("minimum_ess_fraction", minimum_ess_fraction),
        ("maximum_ratio_to_mean", maximum_ratio_to_mean),
        ("maximum_top_tail_mass", maximum_top_tail_mass),
        ("maximum_absolute_log_normalizer_shift", maximum_absolute_log_normalizer_shift),
        ("standard_error_multiplier", standard_error_multiplier),
    ):
        if not np.isfinite(value) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not 0.0 < minimum_ess_fraction <= 1.0:
        raise ValueError("minimum_ess_fraction must lie in (0, 1]")
    if not 0.0 < maximum_top_tail_mass <= 1.0:
        raise ValueError("maximum_top_tail_mass must lie in (0, 1]")

    prefix = state_weight_audit(
        numerators[:base],
        proposal[:base],
        root_log_normalizers=root_log_normalizers,
        tail_fraction=tail_fraction,
    )
    doubled = state_weight_audit(
        numerators,
        proposal,
        root_log_normalizers=root_log_normalizers,
        tail_fraction=tail_fraction,
    )
    names = ["equal_normalized_bank"] + [
        f"root_{index}" for index in range(numerators.shape[1])
    ]
    prefix_rows = [prefix["equal_normalized_bank"], *prefix["roots"]]
    doubled_rows = [doubled["equal_normalized_bank"], *doubled["roots"]]
    stability = []
    for name, first, second in zip(names, prefix_rows, doubled_rows, strict=True):
        shift = abs(second["log_normalizer"] - first["log_normalizer"])
        uncertainty = float(standard_error_multiplier) * np.hypot(
            first["log_normalizer_standard_error"],
            second["log_normalizer_standard_error"],
        )
        bound = float(maximum_absolute_log_normalizer_shift) + uncertainty
        stability.append(
            {
                "target": name,
                "absolute_log_normalizer_shift": float(shift),
                "registered_bound_including_standard_error": float(bound),
                "pass": bool(shift <= bound),
            }
        )
    aggregate = doubled["equal_normalized_bank"]
    root_ess = min(row["effective_sample_size_fraction"] for row in doubled["roots"])
    gates = {
        "aggregate_ess": aggregate["effective_sample_size_fraction"]
        >= float(minimum_ess_fraction),
        "minimum_root_ess": root_ess >= float(minimum_ess_fraction),
        "aggregate_maximum_ratio": aggregate["maximum_ratio_to_mean"]
        <= float(maximum_ratio_to_mean),
        "aggregate_top_tail_mass": aggregate["top_tail_mass"]
        <= float(maximum_top_tail_mass),
        "all_log_normalizers_stable": all(row["pass"] for row in stability),
    }
    return {
        "design": "nested_prefix_M_and_2M_on_audit_stream_only",
        "base_sample_count": base,
        "doubled_sample_count": 2 * base,
        "prefix_M": prefix,
        "doubled_2M": doubled,
        "log_normalizer_stability": stability,
        "gates": gates,
        "pass": bool(all(gates.values())),
    }


def normalized_log_importance_weights(
    log_target_numerators: jax.Array,
    log_proposal_probabilities: jax.Array,
) -> jax.Array:
    """Return stable self-normalized weights with zero mass at target nodes."""

    target = jnp.asarray(log_target_numerators, dtype=jnp.float64)
    proposal = jnp.asarray(log_proposal_probabilities, dtype=jnp.float64)
    if target.ndim != 1 or proposal.shape != target.shape:
        raise ValueError("target and proposal logs must be matching vectors")
    log_weights = target - proposal
    maximum = jnp.max(jnp.where(jnp.isfinite(log_weights), log_weights, -jnp.inf))
    scaled = jnp.where(jnp.isfinite(log_weights), jnp.exp(log_weights - maximum), 0.0)
    total = jnp.sum(scaled)
    return scaled / total


def weighted_sz_negative_log_likelihood(
    params: Mapping[str, object],
    configurations: jax.Array,
    normalized_target_weights: jax.Array,
    config: SzGRUConfig,
) -> jax.Array:
    """Forward-KL cross entropy for a frozen self-normalized target batch."""

    states = jnp.asarray(configurations, dtype=jnp.int32)
    weights = jnp.asarray(normalized_target_weights, dtype=jnp.float64)
    if weights.shape != (states.shape[0],):
        raise ValueError("one target weight is required per configuration")
    return -jnp.sum(
        jax.lax.stop_gradient(weights)
        * sz_gru_log_probabilities(params, states, config)
    )


def initialize_proposal_adam(params: object) -> dict[str, object]:
    """Initialize a dependency-free Adam state suitable for checkpoints."""

    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"step": 0, "first": zeros, "second": zeros}


def proposal_adam_update(
    params: object,
    gradients: object,
    optimizer_state: Mapping[str, object],
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
) -> tuple[object, dict[str, object]]:
    """Apply one deterministic Adam update to proposal parameters."""

    rate = float(learning_rate)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    step = int(optimizer_state["step"]) + 1
    first = jax.tree_util.tree_map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad,
        optimizer_state["first"],
        gradients,
    )
    second = jax.tree_util.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.abs(grad) ** 2,
        optimizer_state["second"],
        gradients,
    )
    first_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta1**step), first)
    second_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta2**step), second)
    updated = jax.tree_util.tree_map(
        lambda value, mean, variance: value
        - rate * mean / (jnp.sqrt(variance) + epsilon),
        params,
        first_hat,
        second_hat,
    )
    return updated, {"step": step, "first": first, "second": second}


@dataclass(frozen=True)
class ProposalFitState:
    """Complete step-level state for deterministic proposal fitting."""

    params: object
    optimizer_state: object
    completed_steps: int
    cursors: ProposalStreamCursors
    history: tuple[dict[str, object], ...]


_RUN_IDENTITY_FIELDS = {
    "graph",
    "group",
    "bank",
    "proposal",
    "source_sha256",
    "stream_seeds",
}


def validate_proposal_run_identity(
    run_identity: Mapping[str, object], stream_seeds: ProposalStreamSeeds
) -> None:
    """Require every scientific and executable identity component."""

    if not _RUN_IDENTITY_FIELDS.issubset(run_identity):
        missing = sorted(_RUN_IDENTITY_FIELDS - set(run_identity))
        raise ValueError(f"proposal run identity is missing fields: {missing}")
    if run_identity["stream_seeds"] != stream_seeds.as_dict():
        raise ValueError("run identity stream seeds do not match the active streams")
    sources = run_identity["source_sha256"]
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("source_sha256 must be a nonempty mapping")
    hexadecimal = set("0123456789abcdef")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or not set(value).issubset(hexadecimal)
        for value in sources.values()
    ):
        raise ValueError("every source digest must be a lowercase SHA256")
    for field in ("graph", "group", "bank", "proposal"):
        if not isinstance(run_identity[field], Mapping) or not run_identity[field]:
            raise ValueError(f"run identity {field} must be a nonempty mapping")


def save_proposal_fit_checkpoint_generation(
    directory: str | Path,
    state: ProposalFitState,
    *,
    run_identity: Mapping[str, object],
    stream_seeds: ProposalStreamSeeds,
) -> Path:
    """Commit one immutable proposal-fit checkpoint generation."""

    validate_proposal_run_identity(run_identity, stream_seeds)
    if len(state.history) != state.completed_steps:
        raise ValueError("proposal history length must equal completed steps")
    if state.cursors.audit != 0:
        raise ValueError("the held-out audit stream must remain unused during fitting")
    return save_resumable_checkpoint_generation(
        directory,
        state.params,
        optimizer_state=state.optimizer_state,
        completed_steps=int(state.completed_steps),
        next_sample_index=int(state.cursors.fit),
        run_identity=run_identity,
        history=state.history,
        metadata={
            "kind": "full_support_sz_proposal_fit_v1",
            "stream_cursors": state.cursors.as_dict(),
            "stream_seeds": stream_seeds.as_dict(),
            "audit_stream_used_during_fit": False,
        },
    )


def load_latest_proposal_fit_checkpoint(
    directory: str | Path,
    *,
    run_identity: Mapping[str, object],
    stream_seeds: ProposalStreamSeeds,
) -> ProposalFitState:
    """Load a verified generation and restore all three sample cursors."""

    validate_proposal_run_identity(run_identity, stream_seeds)
    payload = load_latest_resumable_checkpoint(
        directory, expected_run_identity=run_identity
    )
    metadata = payload.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("kind") != "full_support_sz_proposal_fit_v1"
        or metadata.get("stream_seeds") != stream_seeds.as_dict()
        or metadata.get("audit_stream_used_during_fit") is not False
        or not isinstance(metadata.get("stream_cursors"), Mapping)
    ):
        raise ValueError("invalid proposal-fit checkpoint metadata")
    cursors = ProposalStreamCursors(**dict(metadata["stream_cursors"]))
    if cursors.fit != payload["next_sample_index"] or cursors.audit != 0:
        raise ValueError("proposal checkpoint sample cursors are inconsistent")
    return ProposalFitState(
        params=payload["params"],
        optimizer_state=payload["optimizer_state"],
        completed_steps=int(payload["completed_steps"]),
        cursors=cursors,
        history=tuple(dict(row) for row in payload["history"]),
    )


def fit_step_from_log_target(
    state: ProposalFitState,
    *,
    stream_seeds: ProposalStreamSeeds,
    config: SzGRUConfig,
    epsilon: float,
    fit_batch_size: int,
    validation_batch_size: int,
    learning_rate: float,
    target_log_numerators: Callable[[jax.Array], jax.Array],
    tail_fraction: float = 0.01,
) -> ProposalFitState:
    """Run one fit step and independent validation observation.

    The callback returns ``log |psi_r(x)|^2`` with shape ``(sample, root)``.
    The old proposal and batch-local root normalizers are frozen before the
    weighted likelihood update.  Validation is diagnostic only.  The audit
    key is never touched by this function.
    """

    fit_sample = sample_indexed_full_support_sz_mixture(
        state.params,
        stream_seeds.key("fit"),
        start_index=state.cursors.fit,
        n_samples=fit_batch_size,
        config=config,
        epsilon=epsilon,
    )
    root_logs = jnp.asarray(target_log_numerators(fit_sample.configurations))
    if root_logs.ndim != 2 or root_logs.shape[0] != fit_batch_size:
        raise ValueError("target callback must return shape (sample, root)")
    host_roots = np.asarray(root_logs, dtype=np.float64)
    host_q = np.asarray(fit_sample.log_probabilities, dtype=np.float64)
    root_normalizers = np.asarray(
        [
            log_weight_diagnostics(host_roots[:, root] - host_q).log_normalizer
            for root in range(host_roots.shape[1])
        ]
    )
    normalized_roots = root_logs - jnp.asarray(root_normalizers)[None, :]
    log_target = jax.scipy.special.logsumexp(normalized_roots, axis=1) - jnp.log(
        root_logs.shape[1]
    )
    weights = normalized_log_importance_weights(
        log_target, fit_sample.log_probabilities
    )
    loss_function = lambda candidate: weighted_sz_negative_log_likelihood(
        candidate,
        fit_sample.configurations,
        weights,
        config,
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
        n_samples=validation_batch_size,
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
        root_log_normalizers=root_normalizers,
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
        "fit_sample_index_stop": state.cursors.fit + fit_batch_size,
        "validation_sample_index_start": state.cursors.validation,
        "validation_sample_index_stop": (
            state.cursors.validation + validation_batch_size
        ),
        "weighted_negative_log_likelihood": float(loss),
        "gradient_l2_norm": gradient_norm,
        "fit_uniform_component_fraction": float(
            np.mean(np.asarray(fit_sample.uniform_component))
        ),
        "validation_uniform_component_fraction": float(
            np.mean(np.asarray(validation_sample.uniform_component))
        ),
        "fit_root_log_normalizers": root_normalizers.tolist(),
        "validation_state_weight_audit": validation,
        "audit_stream_cursor": 0,
    }
    return ProposalFitState(
        params=updated_params,
        optimizer_state=updated_optimizer,
        completed_steps=state.completed_steps + 1,
        cursors=ProposalStreamCursors(
            fit=state.cursors.fit + fit_batch_size,
            validation=state.cursors.validation + validation_batch_size,
            audit=0,
        ),
        history=(*state.history, row),
    )


__all__ = [
    "IndexedSzMixtureSample",
    "LogWeightDiagnostics",
    "ProposalFitState",
    "ProposalStreamCursors",
    "ProposalStreamSeeds",
    "fit_step_from_log_target",
    "full_support_sz_mixture_log_probabilities",
    "initialize_proposal_adam",
    "load_latest_proposal_fit_checkpoint",
    "log_weight_diagnostics",
    "m_doubling_state_weight_audit",
    "normalized_log_importance_weights",
    "proposal_adam_update",
    "sample_indexed_full_support_sz_mixture",
    "save_proposal_fit_checkpoint_generation",
    "state_weight_audit",
    "sz_gru_log_probabilities",
    "validate_proposal_run_identity",
    "weighted_sz_negative_log_likelihood",
]
