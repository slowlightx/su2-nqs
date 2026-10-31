"""Fixed-shape sparse-connectivity VMC for compilation-stable local tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
import time
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


StandardErrorMethod = Literal["delta", "batch", "max"]


@dataclass(frozen=True)
class ProposalSampledEnergyStatistics:
    """Diagnostics for a self-normalized proposal-sampled energy.

    ``standard_error`` is selected by ``standard_error_method``.  The
    delta-method estimate is always available; ``batch_standard_error`` is
    populated only when the estimator was constructed with ``batch_count``.
    Effective sample size is the usual Kish ESS of the normalized importance
    weights and is therefore bounded by ``sample_count``.
    """

    energy: float
    effective_sample_size: float
    effective_sample_size_fraction: float
    delta_method_standard_error: float
    batch_standard_error: float | None
    standard_error: float
    standard_error_method: str
    sample_count: int
    batch_count: int | None


@dataclass(frozen=True)
class PairedProposalSampledEnergyDifference:
    """Paired candidate-minus-reference held-out energy comparison.

    The uncertainty uses paired influence values on the same proposal sample,
    retaining the usually substantial candidate/reference covariance.  The
    one-sided normal confidence bound is suitable for the gate
    ``upper_confidence_bound < 0``.
    """

    candidate: ProposalSampledEnergyStatistics
    reference: ProposalSampledEnergyStatistics
    energy_difference: float
    delta_method_standard_error: float
    batch_standard_error: float | None
    standard_error: float
    standard_error_method: str
    confidence_level: float
    upper_confidence_bound: float

    @property
    def statistically_significant_improvement(self) -> bool:
        """Whether the one-sided upper confidence bound is below zero."""

        return self.upper_confidence_bound < 0.0


def padded_connectivity(matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad CSC columns while preserving ``H[new, old]`` connectivity."""

    csc = matrix.tocsc()
    counts = np.diff(csc.indptr)
    width = int(np.max(counts, initial=1))
    neighbors = np.zeros((csc.shape[1], width), dtype=np.int32)
    elements = np.zeros((csc.shape[1], width), dtype=np.complex128)
    valid = np.zeros((csc.shape[1], width), dtype=bool)
    for column in range(csc.shape[1]):
        start, stop = csc.indptr[column : column + 2]
        count = stop - start
        neighbors[column, :count] = csc.indices[start:stop]
        elements[column, :count] = csc.data[start:stop]
        valid[column, :count] = True
    return neighbors, elements, valid


def padded_sparse_importance_sr_step_diagnostics(
    params: object,
    *,
    matrix,
    basis: Sequence[Sequence[int]],
    samples: jax.Array,
    importance_weights: jax.Array,
    fisher_samples: jax.Array | None = None,
    fisher_importance_weights: jax.Array | None = None,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    learning_rate: float,
    damping: float,
    cg_tolerance: float = 1.0e-6,
    cg_max_iterations: int = 100,
    maximum_predicted_feature_step: float | None = None,
    maximum_actual_feature_step: float | None = None,
    feature_backtracking_steps: int = 8,
    fisher_probe: jax.Array | np.ndarray | None = None,
) -> tuple[object, dict[str, Any]]:
    """Audit one importance-SR step on a fixed concrete sample batch.

    The gradient, damped Fisher product, CG direction, and nonlinear
    centered-log trust decision use the same formulas as
    :func:`train_padded_sparse_importance_sr`.  Arrays are returned in the
    flattened real parameter gauge so two probability-accumulation schemes
    can be compared before their trajectories amplify roundoff differences.
    By default the gradient and Fisher use the same weighted sample. Supplying
    both ``fisher_samples`` and ``fisher_importance_weights`` audits a split
    estimator, for example connected children for the residual gradient and
    their Born-distributed parents for the state-geometry metric.
    The function is diagnostic only and never mutates ``params``.
    """

    if learning_rate <= 0.0 or damping <= 0.0:
        raise ValueError("learning_rate and damping must be positive")
    if cg_tolerance <= 0.0 or cg_max_iterations < 1:
        raise ValueError("CG controls must be positive")
    if maximum_predicted_feature_step is not None and (
        not np.isfinite(maximum_predicted_feature_step)
        or maximum_predicted_feature_step <= 0.0
    ):
        raise ValueError("maximum_predicted_feature_step must be finite and positive")
    if maximum_actual_feature_step is not None and (
        not np.isfinite(maximum_actual_feature_step)
        or maximum_actual_feature_step <= 0.0
    ):
        raise ValueError("maximum_actual_feature_step must be finite and positive")
    if feature_backtracking_steps < 1:
        raise ValueError("feature_backtracking_steps must be positive")

    basis_tuple = tuple(tuple(int(x) for x in state) for state in basis)
    if matrix.shape != (len(basis_tuple), len(basis_tuple)):
        raise ValueError("matrix and basis dimensions differ")
    positions = {state: index for index, state in enumerate(basis_tuple)}
    if len(positions) != len(basis_tuple):
        raise ValueError("basis states must be unique")
    concrete_samples = np.asarray(samples, dtype=np.int32)
    concrete_importance = np.asarray(importance_weights, dtype=np.float64)
    if concrete_samples.ndim != 2 or concrete_samples.shape[0] < 1:
        raise ValueError("samples must be a nonempty two-dimensional array")
    if concrete_importance.shape != (len(concrete_samples),):
        raise ValueError("importance_weights must match samples")
    if np.any(~np.isfinite(concrete_importance)) or np.any(
        concrete_importance <= 0.0
    ):
        raise ValueError("importance_weights must be finite and positive")
    try:
        sample_indices = np.asarray(
            [positions[tuple(row)] for row in concrete_samples], dtype=np.int32
        )
    except KeyError as error:
        raise ValueError("samples contain a state outside the basis") from error
    if (fisher_samples is None) != (fisher_importance_weights is None):
        raise ValueError(
            "fisher_samples and fisher_importance_weights must be supplied together"
        )
    if fisher_samples is None:
        concrete_fisher_samples = concrete_samples
        concrete_fisher_importance = concrete_importance
    else:
        concrete_fisher_samples = np.asarray(fisher_samples, dtype=np.int32)
        concrete_fisher_importance = np.asarray(
            fisher_importance_weights, dtype=np.float64
        )
        if (
            concrete_fisher_samples.ndim != 2
            or concrete_fisher_samples.shape[0] < 1
        ):
            raise ValueError(
                "fisher_samples must be a nonempty two-dimensional array"
            )
        if concrete_fisher_importance.shape != (len(concrete_fisher_samples),):
            raise ValueError("fisher_importance_weights must match fisher_samples")
        if np.any(~np.isfinite(concrete_fisher_importance)) or np.any(
            concrete_fisher_importance <= 0.0
        ):
            raise ValueError(
                "fisher_importance_weights must be finite and positive"
            )
        try:
            _ = [positions[tuple(row)] for row in concrete_fisher_samples]
        except KeyError as error:
            raise ValueError(
                "fisher_samples contain a state outside the basis"
            ) from error

    basis_array = jnp.asarray(basis_tuple, dtype=jnp.int32)
    neighbors, elements, valid = padded_connectivity(matrix)
    neighbors_array = jnp.asarray(neighbors)
    elements_array = jnp.asarray(elements)
    valid_array = jnp.asarray(valid)
    flat, unravel = ravel_pytree(params)
    if fisher_probe is None:
        probe = jnp.sin(jnp.arange(len(flat), dtype=flat.dtype) + 1.0)
        probe /= jnp.linalg.norm(probe)
    else:
        probe = jnp.asarray(fisher_probe, dtype=flat.dtype)
        if probe.shape != flat.shape:
            raise ValueError("fisher_probe must match the flattened parameters")
        probe_norm = float(jnp.linalg.norm(probe))
        if not np.isfinite(probe_norm) or probe_norm == 0.0:
            raise ValueError("fisher_probe must have finite nonzero norm")
        probe /= probe_norm

    def sampled_observables(values, concrete, indices):
        candidate = unravel(values)
        logs = jax.vmap(lambda state: log_psi_fn(candidate, state))(concrete)
        connected_states = basis_array[neighbors_array[indices]]
        connected_logs = jax.vmap(
            jax.vmap(lambda state: log_psi_fn(candidate, state))
        )(connected_states)
        contributions = jnp.where(
            valid_array[indices],
            elements_array[indices] * jnp.exp(connected_logs - logs[:, None]),
            0.0,
        )
        return logs, jnp.sum(contributions, axis=1)

    @jax.jit
    def audit(
        values,
        concrete,
        indices,
        raw_importance,
        fisher_concrete,
        raw_fisher_importance,
        probe_vector,
    ):
        weights = jax.lax.stop_gradient(raw_importance)
        weights = weights / jnp.sum(weights)
        fisher_weights = jax.lax.stop_gradient(raw_fisher_importance)
        fisher_weights = fisher_weights / jnp.sum(fisher_weights)
        logs, local = sampled_observables(values, concrete, indices)
        mean = jnp.sum(weights * local)
        centered_local = jax.lax.stop_gradient(local - mean)

        def objective(candidate_values):
            candidate_logs, _ = sampled_observables(
                candidate_values, concrete, indices
            )
            surrogate = 2.0 * jnp.real(
                jnp.sum(weights * jnp.conj(candidate_logs) * centered_local)
            )
            return (
                jax.lax.stop_gradient(jnp.real(mean))
                + surrogate
                - jax.lax.stop_gradient(surrogate)
            )

        def centered_features(candidate_values):
            candidate = unravel(candidate_values)
            candidate_logs = jax.vmap(
                lambda state: log_psi_fn(candidate, state)
            )(fisher_concrete)
            centered = candidate_logs - jnp.sum(
                fisher_weights * candidate_logs
            )
            scaled = jnp.sqrt(2.0 * fisher_weights) * centered
            return jnp.concatenate((jnp.real(scaled), jnp.imag(scaled)))

        current_features, pullback = jax.vjp(centered_features, values)

        def fisher_product(vector):
            _, tangent = jax.jvp(centered_features, (values,), (vector,))
            return pullback(tangent)[0] + damping * vector

        gradient = jax.grad(objective)(values)
        direction, _ = jax.scipy.sparse.linalg.cg(
            fisher_product,
            gradient,
            tol=cg_tolerance,
            maxiter=cg_max_iterations,
        )
        residual = fisher_product(direction) - gradient
        relative_residual = jnp.linalg.norm(residual) / jnp.maximum(
            jnp.linalg.norm(gradient), jnp.finfo(values.dtype).tiny
        )
        undamped_direction = fisher_product(direction) - damping * direction
        fisher_direction_norm = jnp.sqrt(
            jnp.maximum(jnp.vdot(direction, undamped_direction).real, 0.0)
        )
        predicted_unclipped = learning_rate * fisher_direction_norm
        predicted_scale = (
            jnp.asarray(1.0, dtype=values.dtype)
            if maximum_predicted_feature_step is None
            else jnp.minimum(
                1.0,
                maximum_predicted_feature_step
                / jnp.maximum(predicted_unclipped, jnp.finfo(values.dtype).tiny),
            )
        )
        if maximum_actual_feature_step is None:
            step_scale = predicted_scale
            candidate_values = values - learning_rate * step_scale * direction
            actual_feature_step = jnp.linalg.norm(
                centered_features(candidate_values) - current_features
            )
            backtracking_count = jnp.asarray(0, dtype=jnp.int32)
            step_found = jnp.asarray(True)
        else:
            candidate_scales = predicted_scale * 0.5 ** jnp.arange(
                feature_backtracking_steps, dtype=values.dtype
            )
            candidates = values[None, :] - (
                learning_rate * candidate_scales[:, None] * direction[None, :]
            )
            candidate_feature_steps = jax.vmap(
                lambda candidate: jnp.linalg.norm(
                    centered_features(candidate) - current_features
                )
            )(candidates)
            valid_candidates = (
                jnp.isfinite(candidate_feature_steps)
                & (candidate_feature_steps <= maximum_actual_feature_step)
            )
            step_found = jnp.any(valid_candidates)
            selected = jnp.argmax(valid_candidates)
            step_scale = jnp.where(
                step_found, candidate_scales[selected], 0.0
            )
            actual_feature_step = jnp.where(
                step_found, candidate_feature_steps[selected], 0.0
            )
            backtracking_count = jnp.where(
                step_found, selected, feature_backtracking_steps
            )
            candidate_values = values - learning_rate * step_scale * direction
        return (
            candidate_values,
            gradient,
            fisher_product(probe_vector),
            direction,
            jnp.real(mean),
            jnp.real(jnp.sum(weights * jnp.abs(local - mean) ** 2)),
            relative_residual,
            predicted_unclipped,
            step_scale,
            actual_feature_step,
            backtracking_count,
            step_found,
        )

    outputs = audit(
        flat,
        jnp.asarray(concrete_samples),
        jnp.asarray(sample_indices),
        jnp.asarray(concrete_importance),
        jnp.asarray(concrete_fisher_samples),
        jnp.asarray(concrete_fisher_importance),
        probe,
    )
    candidate_flat, gradient, probe_product, direction = outputs[:4]
    diagnostics = {
        "gradient": np.asarray(gradient, dtype=np.float64),
        "fisher_probe_product": np.asarray(probe_product, dtype=np.float64),
        "natural_direction": np.asarray(direction, dtype=np.float64),
        "sampled_energy": float(outputs[4]),
        "sampled_variance": float(outputs[5]),
        "cg_relative_residual": float(outputs[6]),
        "predicted_feature_step_unclipped": float(outputs[7]),
        "step_scale": float(outputs[8]),
        "actual_feature_step": float(outputs[9]),
        "feature_backtracking_count": int(outputs[10]),
        "trust_region_step_found": bool(outputs[11]),
        "gradient_sample_count": int(len(concrete_samples)),
        "fisher_sample_count": int(len(concrete_fisher_samples)),
    }
    return unravel(candidate_flat), diagnostics


def make_importance_sampled_energy(
    matrix,
    basis: Sequence[Sequence[int]],
    samples: jax.Array,
    base_logs: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
) -> Callable[[object], float]:
    """Build a fixed held-out energy estimator using base-distribution samples."""

    return make_proposal_sampled_energy(
        matrix,
        basis,
        samples,
        2.0 * jnp.real(base_logs),
        log_psi_fn,
    )


def make_proposal_sampled_energy(
    matrix,
    basis: Sequence[Sequence[int]],
    samples: jax.Array,
    proposal_log_probabilities: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
) -> Callable[[object], float]:
    """Build a held-out energy estimator for a known sampling proposal.

    ``proposal_log_probabilities[k]`` is ``log r(samples[k])``.  The candidate
    NQS is exactly normalized, so the self-normalized weights are proportional
    to ``|psi_candidate|**2 / r``.  The ordinary base-Born estimator above is
    the special case ``r=|psi_base|**2``.
    """

    observations = _make_proposal_sampled_energy_observations(
        matrix,
        basis,
        samples,
        proposal_log_probabilities,
        log_psi_fn,
    )

    @jax.jit
    def estimate(params):
        local_real, weights = observations(params)
        return jnp.sum(weights * local_real)

    return lambda params: float(estimate(params))


def make_chunked_proposal_sampled_energy(
    matrix,
    basis: Sequence[Sequence[int]],
    samples: jax.Array,
    proposal_log_probabilities: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    *,
    chunk_size: int = 256,
) -> Callable[[object], float]:
    """Build the proposal-sampled energy with a fixed-shape chunk kernel.

    This is mathematically the same self-normalized ratio estimator as
    :func:`make_proposal_sampled_energy`.  The difference is operational: one
    JIT-compiled kernel receives sample indices and proposal log probabilities
    as dynamic arrays of shape ``(chunk_size,)``.  Consequently XLA never has
    to materialize a ``sample_count x maximum_connectivity`` connected-state
    tensor.  A final short chunk is padded and masked on the host, while the
    global softmax normalization is performed only over genuine samples.
    """

    observations = _make_chunked_proposal_sampled_energy_observations(
        matrix,
        basis,
        samples,
        proposal_log_probabilities,
        log_psi_fn,
        chunk_size=chunk_size,
    )

    def estimate(params) -> float:
        local_real, normalized_weights = observations(params)
        return float(np.sum(normalized_weights * local_real))

    return estimate


def make_proposal_sampled_energy_statistics(
    matrix,
    basis: Sequence[Sequence[int]],
    samples: jax.Array,
    proposal_log_probabilities: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    *,
    batch_count: int | None = None,
    standard_error_method: StandardErrorMethod = "max",
) -> Callable[[object], ProposalSampledEnergyStatistics]:
    """Build a paper-quality held-out energy diagnostic.

    The returned callable evaluates a self-normalized importance-sampling
    estimate using ``Re(E_local)``.  Its delta-method standard error uses the
    ratio-estimator influence function.  If ``batch_count`` is supplied,
    contiguous, equally sized batch means of that influence function provide
    a second error estimate that can reveal within-stream correlation.

    ``standard_error_method='max'`` (the default) reports the larger of the
    delta and batch estimates.  Use ``'delta'`` for genuinely iid samples or
    ``'batch'`` to require the batching diagnostic explicitly.
    """

    sample_count = len(samples)
    _validate_statistical_options(
        sample_count,
        batch_count=batch_count,
        standard_error_method=standard_error_method,
    )
    observations = jax.jit(
        _make_proposal_sampled_energy_observations(
            matrix,
            basis,
            samples,
            proposal_log_probabilities,
            log_psi_fn,
        )
    )

    def estimate(params) -> ProposalSampledEnergyStatistics:
        local_real, normalized_weights = observations(params)
        summary, _ = _summarize_energy_observations(
            np.asarray(local_real, dtype=np.float64),
            np.asarray(normalized_weights, dtype=np.float64),
            batch_count=batch_count,
            standard_error_method=standard_error_method,
        )
        return summary

    return estimate


def make_paired_proposal_sampled_energy_difference(
    matrix,
    basis: Sequence[Sequence[int]],
    samples: jax.Array,
    proposal_log_probabilities: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    *,
    batch_count: int | None = None,
    standard_error_method: StandardErrorMethod = "max",
    confidence_level: float = 0.95,
) -> Callable[[object, object], PairedProposalSampledEnergyDifference]:
    """Build a paired held-out candidate-minus-reference energy test.

    Both states are evaluated on exactly the same proposal samples.  The
    paired standard error is computed from the *difference* of their
    ratio-estimator influence values, rather than treating the two energy
    estimates as independent.  A candidate passes the default one-sided 95%
    improvement gate exactly when
    ``result.statistically_significant_improvement`` is true.
    """

    sample_count = len(samples)
    _validate_statistical_options(
        sample_count,
        batch_count=batch_count,
        standard_error_method=standard_error_method,
    )
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0.5 and 1")
    observations = jax.jit(
        _make_proposal_sampled_energy_observations(
            matrix,
            basis,
            samples,
            proposal_log_probabilities,
            log_psi_fn,
        )
    )
    normal_quantile = NormalDist().inv_cdf(confidence_level)

    def compare(
        candidate_params, reference_params
    ) -> PairedProposalSampledEnergyDifference:
        candidate_local, candidate_weights = observations(candidate_params)
        reference_local, reference_weights = observations(reference_params)
        candidate, candidate_influence = _summarize_energy_observations(
            np.asarray(candidate_local, dtype=np.float64),
            np.asarray(candidate_weights, dtype=np.float64),
            batch_count=batch_count,
            standard_error_method=standard_error_method,
        )
        reference, reference_influence = _summarize_energy_observations(
            np.asarray(reference_local, dtype=np.float64),
            np.asarray(reference_weights, dtype=np.float64),
            batch_count=batch_count,
            standard_error_method=standard_error_method,
        )
        paired_influence = candidate_influence - reference_influence
        delta_standard_error = _delta_method_standard_error(paired_influence)
        batch_standard_error = _batch_standard_error(
            paired_influence, batch_count=batch_count
        )
        standard_error, selected_method = _select_standard_error(
            delta_standard_error,
            batch_standard_error,
            standard_error_method,
        )
        difference = candidate.energy - reference.energy
        return PairedProposalSampledEnergyDifference(
            candidate=candidate,
            reference=reference,
            energy_difference=difference,
            delta_method_standard_error=delta_standard_error,
            batch_standard_error=batch_standard_error,
            standard_error=standard_error,
            standard_error_method=selected_method,
            confidence_level=float(confidence_level),
            upper_confidence_bound=difference + normal_quantile * standard_error,
        )

    return compare


def make_chunked_paired_proposal_sampled_energy_difference(
    matrix,
    basis: Sequence[Sequence[int]],
    samples: jax.Array,
    proposal_log_probabilities: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    *,
    chunk_size: int = 256,
    batch_count: int | None = None,
    standard_error_method: StandardErrorMethod = "max",
    confidence_level: float = 0.95,
) -> Callable[[object, object], PairedProposalSampledEnergyDifference]:
    """Build the paired energy gate using one reusable fixed-shape kernel.

    Chunking changes only how local energies and unnormalized log weights are
    evaluated.  Global normalization, ratio-estimator influence values,
    contiguous batch ordering, max-SE selection, and the one-sided confidence
    bound are delegated to the same routines as the monolithic implementation.
    """

    sample_count = len(samples)
    _validate_statistical_options(
        sample_count,
        batch_count=batch_count,
        standard_error_method=standard_error_method,
    )
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0.5 and 1")
    observations = _make_chunked_proposal_sampled_energy_observations(
        matrix,
        basis,
        samples,
        proposal_log_probabilities,
        log_psi_fn,
        chunk_size=chunk_size,
    )
    normal_quantile = NormalDist().inv_cdf(confidence_level)

    def compare(
        candidate_params, reference_params
    ) -> PairedProposalSampledEnergyDifference:
        candidate_local, candidate_weights = observations(candidate_params)
        reference_local, reference_weights = observations(reference_params)
        candidate, candidate_influence = _summarize_energy_observations(
            candidate_local,
            candidate_weights,
            batch_count=batch_count,
            standard_error_method=standard_error_method,
        )
        reference, reference_influence = _summarize_energy_observations(
            reference_local,
            reference_weights,
            batch_count=batch_count,
            standard_error_method=standard_error_method,
        )
        paired_influence = candidate_influence - reference_influence
        delta_standard_error = _delta_method_standard_error(paired_influence)
        batch_standard_error = _batch_standard_error(
            paired_influence, batch_count=batch_count
        )
        standard_error, selected_method = _select_standard_error(
            delta_standard_error,
            batch_standard_error,
            standard_error_method,
        )
        difference = candidate.energy - reference.energy
        return PairedProposalSampledEnergyDifference(
            candidate=candidate,
            reference=reference,
            energy_difference=difference,
            delta_method_standard_error=delta_standard_error,
            batch_standard_error=batch_standard_error,
            standard_error=standard_error,
            standard_error_method=selected_method,
            confidence_level=float(confidence_level),
            upper_confidence_bound=difference + normal_quantile * standard_error,
        )

    return compare


def _make_chunked_proposal_sampled_energy_observations(
    matrix,
    basis: Sequence[Sequence[int]],
    samples: jax.Array,
    proposal_log_probabilities: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    *,
    chunk_size: int,
) -> Callable[[object], tuple[np.ndarray, np.ndarray]]:
    """Return host-collected observations from one fixed-shape JIT kernel."""

    if (
        isinstance(chunk_size, (bool, np.bool_))
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    basis_tuple = tuple(tuple(int(x) for x in state) for state in basis)
    positions = {state: index for index, state in enumerate(basis_tuple)}
    concrete_samples = np.asarray(samples)
    sample_count = len(concrete_samples)
    if sample_count < 1:
        raise ValueError("chunked energy estimates require at least one sample")
    concrete_proposal_logs = np.asarray(proposal_log_probabilities)
    if concrete_proposal_logs.shape != (sample_count,):
        raise ValueError("proposal log probabilities must have one value per sample")
    if not np.all(np.isfinite(concrete_proposal_logs)):
        raise ValueError("sampled proposal log probabilities must be finite")
    try:
        concrete_indices = np.asarray(
            [positions[tuple(row)] for row in concrete_samples], dtype=np.int32
        )
    except KeyError as error:
        raise ValueError(
            "every proposal sample must belong to the supplied basis"
        ) from error

    padded_count = ((sample_count + chunk_size - 1) // chunk_size) * chunk_size
    padded_indices = np.zeros(padded_count, dtype=np.int32)
    padded_proposal_logs = np.zeros(padded_count, dtype=concrete_proposal_logs.dtype)
    padded_indices[:sample_count] = concrete_indices
    padded_proposal_logs[:sample_count] = concrete_proposal_logs
    index_chunks = padded_indices.reshape(-1, chunk_size)
    proposal_chunks = padded_proposal_logs.reshape(-1, chunk_size)

    basis_array = jnp.asarray(basis_tuple, dtype=jnp.int32)
    neighbors, elements, valid = padded_connectivity(matrix)
    neighbors_array = jnp.asarray(neighbors)
    elements_array = jnp.asarray(elements)
    valid_array = jnp.asarray(valid)

    @jax.jit
    def chunk_observations(params, indices, proposal_logs):
        sample_chunk = basis_array[indices]
        logs = jax.vmap(lambda state: log_psi_fn(params, state))(sample_chunk)
        connected_states = basis_array[neighbors_array[indices]]
        connected_logs = jax.vmap(
            jax.vmap(lambda state: log_psi_fn(params, state))
        )(connected_states)
        local = jnp.sum(
            jnp.where(
                valid_array[indices],
                elements_array[indices]
                * jnp.exp(connected_logs - logs[:, None]),
                0.0,
            ),
            axis=1,
        )
        log_weights = 2.0 * jnp.real(logs) - proposal_logs
        return jnp.real(local), log_weights

    def observations(params) -> tuple[np.ndarray, np.ndarray]:
        local_chunks = []
        log_weight_chunks = []
        for indices, proposal_logs in zip(index_chunks, proposal_chunks, strict=True):
            local_real, log_weights = chunk_observations(
                params, jnp.asarray(indices), jnp.asarray(proposal_logs)
            )
            local_chunks.append(np.asarray(local_real, dtype=np.float64))
            log_weight_chunks.append(np.asarray(log_weights, dtype=np.float64))
        local_real = np.concatenate(local_chunks)[:sample_count]
        log_weights = np.concatenate(log_weight_chunks)[:sample_count]
        if not np.all(np.isfinite(local_real)) or not np.all(np.isfinite(log_weights)):
            raise FloatingPointError("non-finite local energy or importance weight")
        shifted = log_weights - float(np.max(log_weights))
        unnormalized_weights = np.exp(shifted)
        weight_sum = float(np.sum(unnormalized_weights))
        if not weight_sum > 0.0 or not np.isfinite(weight_sum):
            raise FloatingPointError("importance weights have zero or non-finite mass")
        return local_real, unnormalized_weights / weight_sum

    return observations


def _make_proposal_sampled_energy_observations(
    matrix,
    basis: Sequence[Sequence[int]],
    samples: jax.Array,
    proposal_log_probabilities: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
) -> Callable[[object], tuple[jax.Array, jax.Array]]:
    """Return real local energies and normalized proposal weights."""

    basis_tuple = tuple(tuple(int(x) for x in state) for state in basis)
    positions = {state: index for index, state in enumerate(basis_tuple)}
    sample_array = jnp.asarray(samples)
    proposal_log_probabilities = jnp.asarray(proposal_log_probabilities)
    if proposal_log_probabilities.shape != (len(sample_array),):
        raise ValueError("proposal log probabilities must have one value per sample")
    concrete_proposal_logs = np.asarray(proposal_log_probabilities)
    if not np.all(np.isfinite(concrete_proposal_logs)):
        raise ValueError("sampled proposal log probabilities must be finite")
    concrete = np.asarray(sample_array)
    try:
        indices = jnp.asarray(
            [positions[tuple(row)] for row in concrete], dtype=jnp.int32
        )
    except KeyError as error:
        raise ValueError(
            "every proposal sample must belong to the supplied basis"
        ) from error
    basis_array = jnp.asarray(basis_tuple, dtype=jnp.int32)
    neighbors, elements, valid = padded_connectivity(matrix)
    neighbors = jnp.asarray(neighbors)
    elements = jnp.asarray(elements)
    valid = jnp.asarray(valid)

    def observations(params):
        logs = jax.vmap(lambda state: log_psi_fn(params, state))(sample_array)
        connected_states = basis_array[neighbors[indices]]
        connected_logs = jax.vmap(
            jax.vmap(lambda state: log_psi_fn(params, state))
        )(connected_states)
        local = jnp.sum(
            jnp.where(
                valid[indices],
                elements[indices] * jnp.exp(connected_logs - logs[:, None]),
                0.0,
            ),
            axis=1,
        )
        log_weights = 2.0 * jnp.real(logs) - proposal_log_probabilities
        normalized_weights = jax.nn.softmax(log_weights)
        return jnp.real(local), normalized_weights

    return observations


def _validate_statistical_options(
    sample_count: int,
    *,
    batch_count: int | None,
    standard_error_method: StandardErrorMethod,
) -> None:
    if sample_count < 2:
        raise ValueError("statistical energy estimates require at least two samples")
    if standard_error_method not in {"delta", "batch", "max"}:
        raise ValueError("standard_error_method must be 'delta', 'batch', or 'max'")
    if batch_count is None:
        if standard_error_method == "batch":
            raise ValueError("standard_error_method='batch' requires batch_count")
        return
    if batch_count < 2 or batch_count > sample_count:
        raise ValueError("batch_count must lie between 2 and the sample count")
    if sample_count % batch_count:
        raise ValueError("batch_count must divide the sample count into equal batches")


def _summarize_energy_observations(
    local_real: np.ndarray,
    normalized_weights: np.ndarray,
    *,
    batch_count: int | None,
    standard_error_method: StandardErrorMethod,
) -> tuple[ProposalSampledEnergyStatistics, np.ndarray]:
    if local_real.ndim != 1 or normalized_weights.shape != local_real.shape:
        raise ValueError("local energies and weights must be matching one-dimensional arrays")
    if not np.all(np.isfinite(local_real)) or not np.all(np.isfinite(normalized_weights)):
        raise FloatingPointError("non-finite local energy or importance weight")
    weight_sum = float(np.sum(normalized_weights))
    if not weight_sum > 0.0:
        raise FloatingPointError("importance weights have zero total mass")
    weights = normalized_weights / weight_sum
    sample_count = len(local_real)
    energy = float(np.sum(weights * local_real))
    # For a ratio estimator E[W X] / E[W], the centered influence values are
    # W_i (X_i - E_hat) / mean(W).  Softmax-normalized weights give the
    # equivalent, scale-free expression below.
    influence = sample_count * weights * (local_real - energy)
    delta_standard_error = _delta_method_standard_error(influence)
    batch_standard_error = _batch_standard_error(
        influence, batch_count=batch_count
    )
    standard_error, selected_method = _select_standard_error(
        delta_standard_error,
        batch_standard_error,
        standard_error_method,
    )
    effective_sample_size = float(1.0 / np.sum(weights**2))
    return (
        ProposalSampledEnergyStatistics(
            energy=energy,
            effective_sample_size=effective_sample_size,
            effective_sample_size_fraction=effective_sample_size / sample_count,
            delta_method_standard_error=delta_standard_error,
            batch_standard_error=batch_standard_error,
            standard_error=standard_error,
            standard_error_method=selected_method,
            sample_count=sample_count,
            batch_count=batch_count,
        ),
        influence,
    )


def _delta_method_standard_error(influence: np.ndarray) -> float:
    return float(np.std(influence, ddof=1) / np.sqrt(len(influence)))


def _batch_standard_error(
    influence: np.ndarray, *, batch_count: int | None
) -> float | None:
    if batch_count is None:
        return None
    batch_means = np.mean(influence.reshape(batch_count, -1), axis=1)
    return float(np.std(batch_means, ddof=1) / np.sqrt(batch_count))


def _select_standard_error(
    delta_standard_error: float,
    batch_standard_error: float | None,
    method: StandardErrorMethod,
) -> tuple[float, str]:
    if method == "delta":
        return delta_standard_error, "delta"
    if method == "batch":
        assert batch_standard_error is not None
        return batch_standard_error, "batch"
    if batch_standard_error is None or delta_standard_error >= batch_standard_error:
        return delta_standard_error, "delta"
    return batch_standard_error, "batch"


def train_padded_sparse_adam(
    params: object,
    key: jax.Array,
    *,
    matrix,
    basis: Sequence[Sequence[int]],
    sample_fn: Callable[[object, jax.Array, int], jax.Array],
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[object, list[dict[str, float]]]:
    """Sampled VMC Adam with a fixed padded connected-state tensor."""

    import optax

    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    basis_tuple = tuple(tuple(int(x) for x in state) for state in basis)
    positions = {state: index for index, state in enumerate(basis_tuple)}
    basis_array = jnp.asarray(basis_tuple, dtype=jnp.int32)
    neighbors, elements, valid = padded_connectivity(matrix)
    neighbors = jnp.asarray(neighbors)
    elements = jnp.asarray(elements)
    valid = jnp.asarray(valid)
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(params)

    def objective(candidate, samples, indices):
        logs = jax.vmap(lambda state: log_psi_fn(candidate, state))(samples)
        local_neighbors = neighbors[indices]
        connected_states = basis_array[local_neighbors]
        connected_logs = jax.vmap(
            jax.vmap(lambda state: log_psi_fn(candidate, state))
        )(connected_states)
        contributions = jnp.where(
            valid[indices],
            elements[indices] * jnp.exp(connected_logs - logs[:, None]),
            0.0,
        )
        local_energies = jnp.sum(contributions, axis=1)
        mean = jnp.mean(local_energies)
        centered = jax.lax.stop_gradient(local_energies - mean)
        surrogate = 2.0 * jnp.real(jnp.mean(jnp.conj(logs) * centered))
        loss = jax.lax.stop_gradient(jnp.real(mean)) + surrogate - jax.lax.stop_gradient(surrogate)
        variance = jnp.mean(jnp.abs(local_energies - mean) ** 2)
        return loss, variance

    value_gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))
    history = []
    for step in range(1, steps + 1):
        key, subkey = jax.random.split(key)
        samples = sample_fn(params, subkey, batch_size)
        concrete = np.asarray(samples)
        indices = jnp.asarray([positions[tuple(row)] for row in concrete], dtype=jnp.int32)
        (energy, variance), gradients = value_gradient(params, samples, indices)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        history.append({"step": float(step), "energy": float(energy), "variance": float(variance)})
    return params, history


def train_padded_sparse_sr(
    params: object,
    key: jax.Array,
    *,
    matrix,
    basis: Sequence[Sequence[int]],
    sample_fn: Callable[[object, jax.Array, int], jax.Array],
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    steps: int,
    batch_size: int,
    learning_rate: float,
    damping: float,
    cg_tolerance: float = 1.0e-6,
    cg_max_iterations: int = 100,
    observer: Callable[[object, int], Mapping[str, float]] | None = None,
    observe_every: int = 1,
    maximum_cg_relative_residual: float | None = None,
    sample_observer: Callable[
        [object, np.ndarray, int], Mapping[str, Any]
    ]
    | None = None,
) -> tuple[object, list[dict[str, Any]]]:
    """Sampled energy SR with fixed-shape padded sparse connectivity.

    The Born samples enter both the unbiased VMC energy gradient and the
    matrix-free quantum Fisher product.  Padding fixes the compiled tensor
    shapes across updates; no parameter-square Fisher matrix is formed.
    ``matrix`` follows ``H[new, old]`` and ``basis`` must use the same column
    order.
    """

    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if learning_rate <= 0.0 or damping <= 0.0:
        raise ValueError("learning_rate and damping must be positive")
    if cg_tolerance <= 0.0 or cg_max_iterations < 1:
        raise ValueError("CG controls must be positive")
    if observe_every < 1:
        raise ValueError("observe_every must be positive")
    if maximum_cg_relative_residual is not None and (
        not np.isfinite(maximum_cg_relative_residual)
        or maximum_cg_relative_residual <= 0.0
    ):
        raise ValueError("maximum_cg_relative_residual must be finite and positive")
    basis_tuple = tuple(tuple(int(x) for x in state) for state in basis)
    if matrix.shape != (len(basis_tuple), len(basis_tuple)):
        raise ValueError("matrix and basis dimensions differ")
    positions = {state: index for index, state in enumerate(basis_tuple)}
    if len(positions) != len(basis_tuple):
        raise ValueError("basis states must be unique")
    basis_array = jnp.asarray(basis_tuple, dtype=jnp.int32)
    neighbors, elements, valid = padded_connectivity(matrix)
    neighbors_array = jnp.asarray(neighbors)
    elements_array = jnp.asarray(elements)
    valid_array = jnp.asarray(valid)
    flat, unravel = ravel_pytree(params)

    def sampled_observables(values, samples, indices):
        candidate = unravel(values)
        logs = jax.vmap(lambda state: log_psi_fn(candidate, state))(samples)
        connected_states = basis_array[neighbors_array[indices]]
        connected_logs = jax.vmap(
            jax.vmap(lambda state: log_psi_fn(candidate, state))
        )(connected_states)
        contributions = jnp.where(
            valid_array[indices],
            elements_array[indices]
            * jnp.exp(connected_logs - logs[:, None]),
            0.0,
        )
        local = jnp.sum(contributions, axis=1)
        return logs, local

    @jax.jit
    def update(values, samples, indices):
        logs, local = sampled_observables(values, samples, indices)
        mean = jnp.mean(local)
        centered_local = jax.lax.stop_gradient(local - mean)

        def objective(candidate_values):
            candidate_logs, _ = sampled_observables(
                candidate_values, samples, indices
            )
            surrogate = 2.0 * jnp.real(
                jnp.mean(jnp.conj(candidate_logs) * centered_local)
            )
            return (
                jax.lax.stop_gradient(jnp.real(mean))
                + surrogate
                - jax.lax.stop_gradient(surrogate)
            )

        def centered_features(candidate_values):
            candidate = unravel(candidate_values)
            candidate_logs = jax.vmap(
                lambda state: log_psi_fn(candidate, state)
            )(samples)
            centered = candidate_logs - jnp.mean(candidate_logs)
            scale = jnp.sqrt(2.0 / batch_size)
            return jnp.concatenate(
                (scale * jnp.real(centered), scale * jnp.imag(centered))
            )

        _, pullback = jax.vjp(centered_features, values)

        def fisher_product(vector):
            _, tangent = jax.jvp(
                centered_features, (values,), (vector,)
            )
            return pullback(tangent)[0] + damping * vector

        gradient = jax.grad(objective)(values)
        direction, _ = jax.scipy.sparse.linalg.cg(
            fisher_product,
            gradient,
            tol=cg_tolerance,
            maxiter=cg_max_iterations,
        )
        residual = fisher_product(direction) - gradient
        relative_residual = jnp.linalg.norm(residual) / jnp.maximum(
            jnp.linalg.norm(gradient), jnp.finfo(values.dtype).tiny
        )
        variance = jnp.mean(jnp.abs(local - mean) ** 2)
        return (
            values - learning_rate * direction,
            jnp.real(mean),
            jnp.real(variance),
            jnp.linalg.norm(gradient),
            jnp.linalg.norm(direction),
            relative_residual,
        )

    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        step_started = time.perf_counter()
        key, subkey = jax.random.split(key)
        samples = sample_fn(unravel(flat), subkey, batch_size)
        concrete = np.asarray(samples)
        sample_seconds = time.perf_counter() - step_started
        sample_observation = None
        sample_observation_seconds = None
        if sample_observer is not None:
            sample_observation_started = time.perf_counter()
            sample_observation = dict(
                sample_observer(unravel(flat), concrete, step)
            )
            sample_observation_seconds = (
                time.perf_counter() - sample_observation_started
            )
        try:
            indices = jnp.asarray(
                [positions[tuple(row)] for row in concrete],
                dtype=jnp.int32,
            )
        except KeyError as error:
            raise ValueError("sampler returned a state outside the basis") from error
        update_started = time.perf_counter()
        (
            candidate_flat,
            energy,
            variance,
            gradient_norm,
            direction_norm,
            relative_residual,
        ) = update(flat, samples, indices)
        energy_value = float(energy)
        variance_value = float(variance)
        gradient_norm_value = float(gradient_norm)
        direction_norm_value = float(direction_norm)
        relative_residual_value = float(relative_residual)
        update_seconds = time.perf_counter() - update_started
        update_accepted = (
            maximum_cg_relative_residual is None
            or relative_residual_value <= maximum_cg_relative_residual
        )
        if update_accepted:
            flat = candidate_flat
        item: dict[str, Any] = {
            "step": float(step),
            "energy": energy_value,
            "variance": variance_value,
            "gradient_norm": gradient_norm_value,
            "natural_direction_norm": direction_norm_value,
            "cg_relative_residual": relative_residual_value,
            "update_accepted": update_accepted,
            "unique_sample_fraction": float(
                len(np.unique(concrete, axis=0)) / batch_size
            ),
            "sample_seconds": sample_seconds,
            "sr_update_seconds": update_seconds,
        }
        if sample_observation is not None:
            item["sample_observation"] = sample_observation
            item["sample_observation_seconds"] = sample_observation_seconds
        if observer is not None and (
            step % observe_every == 0 or step == steps or not update_accepted
        ):
            observation_started = time.perf_counter()
            item["observation"] = {
                str(name): float(value)
                for name, value in observer(unravel(flat), step).items()
            }
            item["observation_seconds"] = (
                time.perf_counter() - observation_started
            )
        item["step_seconds"] = time.perf_counter() - step_started
        if not update_accepted:
            item["termination_reason"] = "cg_relative_residual_gate"
        history.append(item)
        if not update_accepted:
            break
    return unravel(flat), history


def train_padded_sparse_importance_sr(
    params: object,
    key: jax.Array,
    *,
    matrix,
    basis: Sequence[Sequence[int]],
    proposal_sample_fn: Callable[
        [object, jax.Array, int],
        tuple[jax.Array, jax.Array, Mapping[str, Any]],
    ],
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    steps: int,
    batch_size: int,
    learning_rate: float,
    damping: float,
    cg_tolerance: float = 1.0e-6,
    cg_max_iterations: int = 100,
    observer: Callable[[object, int], Mapping[str, float]] | None = None,
    observe_every: int = 1,
    maximum_cg_relative_residual: float | None = None,
    maximum_predicted_feature_step: float | None = None,
    maximum_actual_feature_step: float | None = None,
    feature_backtracking_steps: int = 8,
    pre_update_observer: Callable[
        [object, np.ndarray, np.ndarray, int], Mapping[str, Any]
    ]
    | None = None,
) -> tuple[object, list[dict[str, Any]]]:
    """Sampled energy SR for a known non-Born proposal.

    ``proposal_sample_fn`` returns samples, positive target-over-proposal
    weights ``p/q``, and diagnostic metadata.  Both the phase-preserving VMC
    gradient and the quantum Fisher product use the same self-normalized
    importance weights.  The proposal and its weights are held fixed during
    each update; differentiating through their construction would change the
    intended importance estimator.

    This routine does not prescribe how ``q`` is evaluated.  Enumerable
    benchmarks may audit it on the full support, whereas a scalable caller
    may evaluate ``q`` from reverse local Hamiltonian connections.
    """

    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if learning_rate <= 0.0 or damping <= 0.0:
        raise ValueError("learning_rate and damping must be positive")
    if cg_tolerance <= 0.0 or cg_max_iterations < 1:
        raise ValueError("CG controls must be positive")
    if observe_every < 1:
        raise ValueError("observe_every must be positive")
    if maximum_cg_relative_residual is not None and (
        not np.isfinite(maximum_cg_relative_residual)
        or maximum_cg_relative_residual <= 0.0
    ):
        raise ValueError("maximum_cg_relative_residual must be finite and positive")
    if maximum_predicted_feature_step is not None and (
        not np.isfinite(maximum_predicted_feature_step)
        or maximum_predicted_feature_step <= 0.0
    ):
        raise ValueError("maximum_predicted_feature_step must be finite and positive")
    if maximum_actual_feature_step is not None and (
        not np.isfinite(maximum_actual_feature_step)
        or maximum_actual_feature_step <= 0.0
    ):
        raise ValueError("maximum_actual_feature_step must be finite and positive")
    if feature_backtracking_steps < 1:
        raise ValueError("feature_backtracking_steps must be positive")

    basis_tuple = tuple(tuple(int(x) for x in state) for state in basis)
    if matrix.shape != (len(basis_tuple), len(basis_tuple)):
        raise ValueError("matrix and basis dimensions differ")
    positions = {state: index for index, state in enumerate(basis_tuple)}
    if len(positions) != len(basis_tuple):
        raise ValueError("basis states must be unique")
    basis_array = jnp.asarray(basis_tuple, dtype=jnp.int32)
    neighbors, elements, valid = padded_connectivity(matrix)
    neighbors_array = jnp.asarray(neighbors)
    elements_array = jnp.asarray(elements)
    valid_array = jnp.asarray(valid)
    flat, unravel = ravel_pytree(params)

    def sampled_observables(values, samples, indices):
        candidate = unravel(values)
        logs = jax.vmap(lambda state: log_psi_fn(candidate, state))(samples)
        connected_states = basis_array[neighbors_array[indices]]
        connected_logs = jax.vmap(
            jax.vmap(lambda state: log_psi_fn(candidate, state))
        )(connected_states)
        contributions = jnp.where(
            valid_array[indices],
            elements_array[indices]
            * jnp.exp(connected_logs - logs[:, None]),
            0.0,
        )
        return logs, jnp.sum(contributions, axis=1)

    @jax.jit
    def update(values, samples, indices, raw_importance_weights):
        importance = jax.lax.stop_gradient(raw_importance_weights)
        weights = importance / jnp.sum(importance)
        logs, local = sampled_observables(values, samples, indices)
        mean = jnp.sum(weights * local)
        centered_local = jax.lax.stop_gradient(local - mean)

        def objective(candidate_values):
            candidate_logs, _ = sampled_observables(
                candidate_values, samples, indices
            )
            surrogate = 2.0 * jnp.real(
                jnp.sum(weights * jnp.conj(candidate_logs) * centered_local)
            )
            return (
                jax.lax.stop_gradient(jnp.real(mean))
                + surrogate
                - jax.lax.stop_gradient(surrogate)
            )

        def centered_features(candidate_values):
            candidate = unravel(candidate_values)
            candidate_logs = jax.vmap(
                lambda state: log_psi_fn(candidate, state)
            )(samples)
            centered = candidate_logs - jnp.sum(weights * candidate_logs)
            scaled = jnp.sqrt(2.0 * weights) * centered
            return jnp.concatenate((jnp.real(scaled), jnp.imag(scaled)))

        current_features, pullback = jax.vjp(centered_features, values)

        def fisher_product(vector):
            _, tangent = jax.jvp(centered_features, (values,), (vector,))
            return pullback(tangent)[0] + damping * vector

        gradient = jax.grad(objective)(values)
        direction, _ = jax.scipy.sparse.linalg.cg(
            fisher_product,
            gradient,
            tol=cg_tolerance,
            maxiter=cg_max_iterations,
        )
        residual = fisher_product(direction) - gradient
        relative_residual = jnp.linalg.norm(residual) / jnp.maximum(
            jnp.linalg.norm(gradient), jnp.finfo(values.dtype).tiny
        )
        undamped_fisher_direction = fisher_product(direction) - damping * direction
        fisher_direction_norm = jnp.sqrt(
            jnp.maximum(
                jnp.vdot(direction, undamped_fisher_direction).real,
                0.0,
            )
        )
        predicted_feature_step_unclipped = learning_rate * fisher_direction_norm
        predicted_step_scale = (
            jnp.asarray(1.0, dtype=values.dtype)
            if maximum_predicted_feature_step is None
            else jnp.minimum(
                1.0,
                maximum_predicted_feature_step
                / jnp.maximum(
                    predicted_feature_step_unclipped,
                    jnp.finfo(values.dtype).tiny,
                ),
            )
        )
        if maximum_actual_feature_step is None:
            step_scale = predicted_step_scale
            candidate_values = values - learning_rate * step_scale * direction
            actual_feature_step = jnp.linalg.norm(
                centered_features(candidate_values) - current_features
            )
            feature_backtracking_count = jnp.asarray(0, dtype=jnp.int32)
            trust_region_step_found = jnp.asarray(True)
        else:
            candidate_scales = predicted_step_scale * 0.5 ** jnp.arange(
                feature_backtracking_steps, dtype=values.dtype
            )
            candidate_values = values[None, :] - (
                learning_rate * candidate_scales[:, None] * direction[None, :]
            )
            candidate_feature_steps = jax.vmap(
                lambda candidate: jnp.linalg.norm(
                    centered_features(candidate) - current_features
                )
            )(candidate_values)
            valid_candidate = (
                jnp.isfinite(candidate_feature_steps)
                & (candidate_feature_steps <= maximum_actual_feature_step)
            )
            trust_region_step_found = jnp.any(valid_candidate)
            selected_index = jnp.argmax(valid_candidate)
            step_scale = jnp.where(
                trust_region_step_found, candidate_scales[selected_index], 0.0
            )
            actual_feature_step = jnp.where(
                trust_region_step_found, candidate_feature_steps[selected_index], 0.0
            )
            feature_backtracking_count = jnp.where(
                trust_region_step_found,
                selected_index,
                feature_backtracking_steps,
            )
        variance = jnp.sum(weights * jnp.abs(local - mean) ** 2)
        effective_sample_size = 1.0 / jnp.sum(weights**2)
        return (
            values - learning_rate * step_scale * direction,
            jnp.real(mean),
            jnp.real(variance),
            jnp.linalg.norm(gradient),
            jnp.linalg.norm(direction),
            relative_residual,
            fisher_direction_norm,
            predicted_feature_step_unclipped,
            step_scale,
            actual_feature_step,
            feature_backtracking_count,
            trust_region_step_found,
            effective_sample_size,
            jnp.max(importance),
        )

    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        step_started = time.perf_counter()
        key, subkey = jax.random.split(key)
        samples, importance_weights, proposal_observation = proposal_sample_fn(
            unravel(flat), subkey, batch_size
        )
        concrete = np.asarray(samples)
        concrete_weights = np.asarray(importance_weights, dtype=np.float64)
        if concrete.ndim != 2 or concrete.shape[0] != batch_size:
            raise ValueError("proposal sampler returned an invalid sample array")
        if concrete_weights.shape != (batch_size,):
            raise ValueError("proposal sampler returned an invalid weight array")
        if np.any(~np.isfinite(concrete_weights)) or np.any(
            concrete_weights <= 0.0
        ):
            raise ValueError("importance weights must be finite and positive")
        sample_seconds = time.perf_counter() - step_started
        pre_update_observation = None
        pre_update_observation_seconds = None
        if pre_update_observer is not None:
            observation_started = time.perf_counter()
            pre_update_observation = dict(
                pre_update_observer(
                    unravel(flat), concrete, concrete_weights, step
                )
            )
            pre_update_observation_seconds = (
                time.perf_counter() - observation_started
            )
        try:
            indices = jnp.asarray(
                [positions[tuple(row)] for row in concrete], dtype=jnp.int32
            )
        except KeyError as error:
            raise ValueError("proposal sampler returned a state outside the basis") from error

        update_started = time.perf_counter()
        (
            candidate_flat,
            energy,
            variance,
            gradient_norm,
            direction_norm,
            relative_residual,
            fisher_direction_norm,
            predicted_feature_step_unclipped,
            step_scale,
            actual_feature_step,
            feature_backtracking_count,
            trust_region_step_found,
            effective_sample_size,
            maximum_importance_weight,
        ) = update(
            flat,
            jnp.asarray(samples),
            indices,
            jnp.asarray(concrete_weights),
        )
        relative_residual_value = float(relative_residual)
        update_accepted = bool(trust_region_step_found) and (
            maximum_cg_relative_residual is None
            or relative_residual_value <= maximum_cg_relative_residual
        )
        if update_accepted:
            flat = candidate_flat
        item: dict[str, Any] = {
            "step": float(step),
            "energy": float(energy),
            "variance": float(variance),
            "gradient_norm": float(gradient_norm),
            "natural_direction_norm": float(direction_norm),
            "cg_relative_residual": relative_residual_value,
            "fisher_direction_norm": float(fisher_direction_norm),
            "predicted_feature_step_unclipped": float(
                predicted_feature_step_unclipped
            ),
            "predicted_feature_step": float(
                step_scale * predicted_feature_step_unclipped
            ),
            "step_scale": float(step_scale),
            "actual_feature_step": float(actual_feature_step),
            "feature_backtracking_count": int(feature_backtracking_count),
            "trust_region_step_found": bool(trust_region_step_found),
            "update_accepted": update_accepted,
            "unique_sample_fraction": float(
                len(np.unique(concrete, axis=0)) / batch_size
            ),
            "importance_effective_sample_size": float(effective_sample_size),
            "importance_effective_sample_size_fraction": float(
                effective_sample_size / batch_size
            ),
            "maximum_importance_weight": float(maximum_importance_weight),
            "proposal_observation": dict(proposal_observation),
            "sample_seconds": sample_seconds,
            "sr_update_seconds": time.perf_counter() - update_started,
        }
        if pre_update_observation is not None:
            item["pre_update_observation"] = pre_update_observation
            item["pre_update_observation_seconds"] = pre_update_observation_seconds
        if observer is not None and (
            step % observe_every == 0 or step == steps or not update_accepted
        ):
            observation_started = time.perf_counter()
            item["observation"] = {
                str(name): float(value)
                for name, value in observer(unravel(flat), step).items()
            }
            item["observation_seconds"] = (
                time.perf_counter() - observation_started
            )
        item["step_seconds"] = time.perf_counter() - step_started
        if not update_accepted:
            item["termination_reason"] = (
                "actual_feature_trust_region"
                if not bool(trust_region_step_found)
                else "cg_relative_residual_gate"
            )
        history.append(item)
        if not update_accepted:
            break
    return unravel(flat), history
