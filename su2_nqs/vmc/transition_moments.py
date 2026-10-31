"""Mixture-sampled complex transition moments and joint block bootstrap.

For amplitudes ``Phi[x, a] = psi_a(x)`` and operator actions
``A[x, o, a] = (O_o psi_a)(x)``, this module estimates

``M[o, b, a] = <psi_b|O_o|psi_a>``

from samples drawn from a positive shared proposal ``rho``.  One observation
is ``conj(Phi[x,b]) * A[x,o,a] / rho[x]``.  This orientation matches the
transition-RDM convention ``rho_ab = Tr_env |psi_a><psi_b|``: its moments are
stored in matrix entry ``M[o,b,a]``.

The routines deliberately distinguish *design weights* from importance
ratios.  Uniform IID draws use weights ``1/M``.  If component ``s`` of a
stratified mixture has mass ``pi_s`` and exactly ``M_s`` draws, every draw in
that component has the fixed weight ``pi_s/M_s``.  Supplied weights must
already sum to one and are never randomly self-normalized.  This is essential
for unbiased stratified transition estimates.

The block bootstrap resamples complete design-balanced blocks and retains the
full covariance of the real and imaginary moment components.  A nonlinear
callback may reconstruct RDMs and QDDM diagnostics inside every replicate;
propagating separate observable error bars through a final trace norm is not
equivalent near a local-indistinguishability null.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EffectiveSampleSizeDiagnostics:
    """Kish ESS for nonnegative proposal-ratio magnitudes.

    ``effective_samples`` has the shape of all axes after the observation
    axis.  ``fractions`` divides it by the design ESS ``1/sum(w_i**2)`` and
    therefore lies in ``[0,1]`` up to roundoff.
    """

    effective_samples: np.ndarray
    fractions: np.ndarray
    design_effective_sample_size: float


@dataclass(frozen=True)
class TransitionMomentEstimate:
    """A common-sample estimate and its per-observation diagnostics."""

    moments: np.ndarray
    observations: np.ndarray
    observation_weights: np.ndarray
    proposal_probabilities: np.ndarray
    root_probability_ratios: np.ndarray
    root_ratio_ess: EffectiveSampleSizeDiagnostics
    transition_magnitude_ess: EffectiveSampleSizeDiagnostics
    sample_count: int
    operator_count: int
    state_count: int


@dataclass(frozen=True)
class NonlinearBootstrapSummary:
    """Joint-bootstrap propagation of a real nonlinear output vector."""

    names: tuple[str, ...]
    point_values: np.ndarray
    replicate_values: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    percentile_interval: np.ndarray


@dataclass(frozen=True)
class JointBlockBootstrap:
    """Full complex-moment covariance from resampled observation blocks.

    Complex arrays are flattened in C order with every real component first,
    followed by every imaginary component.  ``percentile_interval[:,0]`` and
    ``[:,1]`` are the componentwise lower and upper bounds in that ordering.
    """

    point_estimate: np.ndarray
    replicate_estimates: np.ndarray
    flattened_point: np.ndarray
    flattened_replicates: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    percentile_interval: np.ndarray
    block_labels: np.ndarray
    block_weights: np.ndarray
    block_count: int
    bootstrap_resamples: int
    nonlinear: NonlinearBootstrapSummary | None


def _design_weights(
    sample_count: int, observation_weights: np.ndarray | None
) -> np.ndarray:
    if observation_weights is None:
        return np.full(sample_count, 1.0 / sample_count, dtype=np.float64)
    weights = np.asarray(observation_weights, dtype=np.float64)
    if weights.shape != (sample_count,):
        raise ValueError("observation_weights must have one entry per sample")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("observation_weights must be finite and nonnegative")
    total = float(np.sum(weights, dtype=np.float64))
    if not np.isclose(total, 1.0, rtol=5.0e-13, atol=5.0e-13):
        raise ValueError(
            "observation_weights are fixed design weights and must sum to one; "
            "they are not self-normalized"
        )
    if not np.any(weights > 0.0):
        raise ValueError("at least one observation weight must be positive")
    return weights


def stratified_observation_weights(
    stratum_indices: np.ndarray,
    mixture_weights: Sequence[float],
) -> np.ndarray:
    """Return fixed ``pi_s/M_s`` weights for a stratified mixture sample.

    ``stratum_indices[i]`` records which component generated observation
    ``i``.  Every component with positive mixture mass must have at least one
    draw.  The returned weights sum to one by construction and must be passed
    unchanged to :func:`estimate_transition_moments`.
    """

    strata = np.asarray(stratum_indices)
    if strata.ndim != 1 or len(strata) < 1:
        raise ValueError("stratum_indices must be a nonempty vector")
    if not np.issubdtype(strata.dtype, np.integer):
        if np.any(~np.isfinite(strata)) or np.any(strata != np.floor(strata)):
            raise ValueError("stratum_indices must contain integers")
    strata = np.asarray(strata, dtype=np.int64)
    mixture = np.asarray(tuple(mixture_weights), dtype=np.float64)
    if mixture.ndim != 1 or len(mixture) < 1:
        raise ValueError("mixture_weights must be a nonempty vector")
    if np.any(~np.isfinite(mixture)) or np.any(mixture < 0.0):
        raise ValueError("mixture_weights must be finite and nonnegative")
    if not np.isclose(float(np.sum(mixture)), 1.0, rtol=5.0e-13, atol=5.0e-13):
        raise ValueError("mixture_weights must sum to one")
    if np.any(strata < 0) or np.any(strata >= len(mixture)):
        raise ValueError("a stratum index lies outside mixture_weights")
    counts = np.bincount(strata, minlength=len(mixture))
    if np.any((mixture > 0.0) & (counts == 0)):
        raise ValueError("every positive-mass mixture component needs a draw")
    weights = np.zeros(len(strata), dtype=np.float64)
    for index, mass in enumerate(mixture):
        if counts[index] > 0:
            weights[strata == index] = mass / float(counts[index])
    # Componentwise assignment makes the mathematical sum exactly one up to
    # deterministic floating summation.  Do not divide by its observed sum.
    if not np.isclose(float(np.sum(weights)), 1.0, rtol=5.0e-13, atol=5.0e-13):
        raise FloatingPointError("stratified design weights failed normalization")
    return weights


def _magnitude_ess(
    nonnegative_values: np.ndarray, weights: np.ndarray
) -> EffectiveSampleSizeDiagnostics:
    values = np.asarray(nonnegative_values, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != len(weights):
        raise ValueError("ESS values must start with the observation axis")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("ESS values must be finite and nonnegative")
    reshape = (len(weights),) + (1,) * (values.ndim - 1)
    weighted = values * weights.reshape(reshape)
    numerator = np.sum(weighted, axis=0) ** 2
    denominator = np.sum(weighted**2, axis=0)
    ess = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )
    design = float(1.0 / np.sum(weights**2))
    fractions = np.minimum(1.0, np.maximum(0.0, ess / design))
    return EffectiveSampleSizeDiagnostics(
        effective_samples=np.asarray(ess, dtype=np.float64),
        fractions=np.asarray(fractions, dtype=np.float64),
        design_effective_sample_size=design,
    )


def estimate_transition_moments(
    amplitudes: np.ndarray,
    operator_actions: np.ndarray,
    proposal_probabilities: np.ndarray,
    *,
    observation_weights: np.ndarray | None = None,
) -> TransitionMomentEstimate:
    """Estimate all ``<psi_b|O_o|psi_a>`` on one shared proposal sample.

    Args:
        amplitudes: Complex ``(sample, state)`` amplitudes at sampled
            configurations.
        operator_actions: Complex ``(sample, operator, state)`` rows of
            ``O_o|psi_a>`` at the same configurations.  Include the state
            amplitudes themselves as an action to estimate the identity/Gram.
        proposal_probabilities: The actual positive shared proposal ``rho`` at
            each sampled configuration.
        observation_weights: Fixed sampling-design weights.  Omit only for IID
            proposal draws.  Stratified component draws require
            ``pi_s/M_s`` weights, conveniently constructed by
            :func:`stratified_observation_weights`.

    Returns:
        Matrices are ordered ``(operator, bra_state, ket_state)``.  No
        Hermitization or normalization is applied to the raw unbiased
        estimate.
    """

    states = np.asarray(amplitudes, dtype=np.complex128)
    actions = np.asarray(operator_actions, dtype=np.complex128)
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("amplitudes must be a nonempty sample-by-state matrix")
    if actions.ndim != 3 or actions.shape[0] != states.shape[0] or actions.shape[2] != states.shape[1]:
        raise ValueError(
            "operator_actions must have shape (sample, operator, state)"
        )
    if actions.shape[1] < 1:
        raise ValueError("at least one operator action is required")
    if np.any(~np.isfinite(states)) or np.any(~np.isfinite(actions)):
        raise ValueError("amplitudes and operator_actions must be finite")
    proposal = np.asarray(proposal_probabilities, dtype=np.float64)
    if proposal.shape != (states.shape[0],):
        raise ValueError("proposal_probabilities must have one entry per sample")
    if np.any(~np.isfinite(proposal)) or np.any(proposal <= 0.0):
        raise ValueError("proposal_probabilities must be finite and positive")
    weights = _design_weights(states.shape[0], observation_weights)

    observations = (
        states.conj()[:, None, :, None]
        * actions[:, :, None, :]
        / proposal[:, None, None, None]
    )
    if np.any(~np.isfinite(observations)):
        raise FloatingPointError("proposal scaling produced non-finite moments")
    moments = np.einsum("x,xoba->oba", weights, observations, optimize=True)
    root_ratios = np.abs(states) ** 2 / proposal[:, None]
    return TransitionMomentEstimate(
        moments=np.asarray(moments, dtype=np.complex128),
        observations=np.asarray(observations, dtype=np.complex128),
        observation_weights=np.asarray(weights, dtype=np.float64),
        proposal_probabilities=proposal,
        root_probability_ratios=np.asarray(root_ratios, dtype=np.float64),
        root_ratio_ess=_magnitude_ess(root_ratios, weights),
        transition_magnitude_ess=_magnitude_ess(np.abs(observations), weights),
        sample_count=int(states.shape[0]),
        operator_count=int(actions.shape[1]),
        state_count=int(states.shape[1]),
    )


def hermitize_transition_moments(moments: np.ndarray) -> np.ndarray:
    """Hermitize every operator matrix without changing axis orientation.

    This variance-reducing projection is valid only when every supplied
    operator is physically Hermitian.  The raw estimate should still be kept
    for anti-Hermiticity diagnostics and coverage tests.
    """

    values = np.asarray(moments, dtype=np.complex128)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("moments must have shape (operator, state, state)")
    if np.any(~np.isfinite(values)):
        raise ValueError("moments must be finite")
    return np.asarray(
        0.5 * (values + values.conj().transpose(0, 2, 1)),
        dtype=np.complex128,
    )


def flatten_complex_moments(values: np.ndarray) -> np.ndarray:
    """Flatten complex moments as ``[all real, all imaginary]`` in C order."""

    array = np.asarray(values, dtype=np.complex128)
    if np.any(~np.isfinite(array)):
        raise ValueError("complex moments must be finite")
    flat = array.reshape(-1)
    return np.concatenate((flat.real, flat.imag)).astype(np.float64, copy=False)


def unflatten_complex_moments(values: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Invert :func:`flatten_complex_moments` for a registered array shape."""

    flat = np.asarray(values, dtype=np.float64)
    concrete_shape = tuple(int(value) for value in shape)
    if any(value < 1 for value in concrete_shape):
        raise ValueError("every target shape extent must be positive")
    size = int(np.prod(concrete_shape, dtype=np.int64))
    if flat.shape != (2 * size,) or np.any(~np.isfinite(flat)):
        raise ValueError("flattened values do not match twice the target size")
    return np.asarray(
        (flat[:size] + 1.0j * flat[size:]).reshape(concrete_shape),
        dtype=np.complex128,
    )


def _real_nonlinear_vector(
    callback: Callable[[np.ndarray], np.ndarray],
    moments: np.ndarray,
) -> np.ndarray:
    values = np.asarray(callback(moments), dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or np.any(~np.isfinite(values)):
        raise ValueError("nonlinear_transform must return a finite real vector")
    return values


def joint_block_bootstrap(
    estimate: TransitionMomentEstimate,
    block_ids: np.ndarray,
    *,
    bootstrap_resamples: int,
    seed: int,
    confidence_level: float = 0.95,
    require_equal_block_weight: bool = True,
    nonlinear_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    nonlinear_names: Sequence[str] | None = None,
) -> JointBlockBootstrap:
    """Resample complete blocks and retain the full joint covariance.

    Blocks should be independent time-series batches and, for a stratified
    mixture, contain the same registered allocation from every component.
    The default fails closed unless all block design masses are equal.  If an
    externally justified unequal-block design is used, setting
    ``require_equal_block_weight=False`` invokes a probability-proportional-
    to-design-mass Horvitz bootstrap; weights are still never normalized by a
    random replicate sum.
    """

    labels_raw = np.asarray(block_ids)
    if labels_raw.shape != (estimate.sample_count,):
        raise ValueError("block_ids must have one entry per observation")
    if not np.issubdtype(labels_raw.dtype, np.integer):
        if np.any(~np.isfinite(labels_raw)) or np.any(labels_raw != np.floor(labels_raw)):
            raise ValueError("block_ids must contain integers")
    labels_raw = np.asarray(labels_raw, dtype=np.int64)
    labels, inverse = np.unique(labels_raw, return_inverse=True)
    block_count = int(len(labels))
    if block_count < 2:
        raise ValueError("at least two bootstrap blocks are required")
    resamples = int(bootstrap_resamples)
    if resamples < 2:
        raise ValueError("bootstrap_resamples must be at least two")
    level = float(confidence_level)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    weights = estimate.observation_weights
    block_weights = np.bincount(
        inverse, weights=weights, minlength=block_count
    ).astype(np.float64)
    if np.any(block_weights <= 0.0):
        raise ValueError("every bootstrap block must have positive design weight")
    equal_mass = 1.0 / block_count
    mass_error = float(np.max(np.abs(block_weights - equal_mass)))
    if require_equal_block_weight and mass_error > 5.0e-12:
        raise ValueError(
            "bootstrap blocks must have equal design weight; balance every "
            "mixture stratum within every block"
        )

    flat_observations = estimate.observations.reshape(estimate.sample_count, -1)
    weighted_rows = flat_observations * weights[:, None]
    block_sums = np.zeros((block_count, flat_observations.shape[1]), dtype=np.complex128)
    np.add.at(block_sums, inverse, weighted_rows)
    original = np.sum(block_sums, axis=0).reshape(estimate.moments.shape)
    if not np.allclose(original, estimate.moments, rtol=2.0e-13, atol=2.0e-13):
        raise FloatingPointError("block aggregation changed the point estimate")

    probabilities = (
        np.full(block_count, equal_mass, dtype=np.float64)
        if require_equal_block_weight
        else block_weights / float(np.sum(block_weights))
    )
    rng = np.random.default_rng(int(seed))
    counts = rng.multinomial(block_count, probabilities, size=resamples)
    scaled_blocks = block_sums / (block_count * probabilities[:, None])
    replicate_flat_complex = counts @ scaled_blocks
    replicates = replicate_flat_complex.reshape((resamples,) + estimate.moments.shape)
    flat_replicates = np.stack(
        [flatten_complex_moments(value) for value in replicates], axis=0
    )
    flattened_point = flatten_complex_moments(estimate.moments)
    covariance = np.atleast_2d(
        np.cov(flat_replicates, rowvar=False, ddof=1)
    ).astype(np.float64)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    alpha = 0.5 * (1.0 - level)
    interval = np.quantile(flat_replicates, (alpha, 1.0 - alpha), axis=0).T

    nonlinear = None
    if nonlinear_transform is not None:
        point_values = _real_nonlinear_vector(
            nonlinear_transform, estimate.moments
        )
        nonlinear_replicates = np.stack(
            [
                _real_nonlinear_vector(nonlinear_transform, value)
                for value in replicates
            ],
            axis=0,
        )
        if nonlinear_replicates.shape[1:] != point_values.shape:
            raise ValueError("nonlinear_transform changed output shape")
        if nonlinear_names is None:
            names = tuple(f"value_{index}" for index in range(len(point_values)))
        else:
            names = tuple(str(value) for value in nonlinear_names)
            if len(names) != len(point_values) or len(set(names)) != len(names):
                raise ValueError("nonlinear_names must uniquely name every output")
        nonlinear_covariance = np.atleast_2d(
            np.cov(nonlinear_replicates, rowvar=False, ddof=1)
        ).astype(np.float64)
        nonlinear_standard_errors = np.sqrt(
            np.maximum(np.diag(nonlinear_covariance), 0.0)
        )
        nonlinear_interval = np.quantile(
            nonlinear_replicates, (alpha, 1.0 - alpha), axis=0
        ).T
        nonlinear = NonlinearBootstrapSummary(
            names=names,
            point_values=point_values,
            replicate_values=nonlinear_replicates,
            covariance=nonlinear_covariance,
            standard_errors=nonlinear_standard_errors,
            percentile_interval=np.asarray(nonlinear_interval, dtype=np.float64),
        )

    return JointBlockBootstrap(
        point_estimate=np.asarray(estimate.moments, dtype=np.complex128),
        replicate_estimates=np.asarray(replicates, dtype=np.complex128),
        flattened_point=flattened_point,
        flattened_replicates=np.asarray(flat_replicates, dtype=np.float64),
        covariance=covariance,
        standard_errors=np.asarray(standard_errors, dtype=np.float64),
        percentile_interval=np.asarray(interval, dtype=np.float64),
        block_labels=np.asarray(labels, dtype=np.int64),
        block_weights=block_weights,
        block_count=block_count,
        bootstrap_resamples=resamples,
        nonlinear=nonlinear,
    )


__all__ = [
    "EffectiveSampleSizeDiagnostics",
    "JointBlockBootstrap",
    "NonlinearBootstrapSummary",
    "TransitionMomentEstimate",
    "estimate_transition_moments",
    "flatten_complex_moments",
    "hermitize_transition_moments",
    "joint_block_bootstrap",
    "stratified_observation_weights",
    "unflatten_complex_moments",
]
