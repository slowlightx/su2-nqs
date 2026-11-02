"""Exact-small algebra for mixture-sampled multistate VMC.

This module separates the statistical identities needed by a future scalable
multi-root engine from any neural model, optimizer, or accelerator backend.
For sampled configurations ``x`` let

``Phi[x, a] = psi_a(x)`` and ``B[x, a] = (H psi_a)(x)``.

If the configurations are drawn from a positive proposal ``rho``, define the
scaled rows ``V = Phi / sqrt(rho)`` and ``T = B / sqrt(rho)``.  Their proposal
expectations give

``S = <V^dagger V>``, ``H = <V^dagger T>``, and ``K = <T^dagger T>``.

The routines below implement only these finite-sample matrices, a strict
full-rank generalized Ritz solve, and direct residual norms.  In particular,
they do not differentiate noisy Ritz eigenvectors and they do not silently
truncate a collapsed represented span.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SAMPLED_OVERLAP_DIAGONAL = "sampled"
EXACT_OVERLAP_DIAGONAL = "exact"


class NumericalSubspaceRankError(ValueError):
    """Raised when a requested sampled state span is not numerically full rank."""


@dataclass(frozen=True)
class HermiticityDiagnostics:
    """Deviation of a raw sampled matrix from Hermiticity."""

    maximum_absolute_error: float
    relative_frobenius_error: float


@dataclass(frozen=True)
class SampledSubspaceMoments:
    """Finite-sample overlap, Hamiltonian, and Hamiltonian-action Gram matrices.

    ``hamiltonian_raw`` is retained because a Monte Carlo estimate of
    ``<psi_a|H|psi_b>`` is generally not Hermitian sample by sample even when
    the physical Hamiltonian is.  ``hamiltonian`` is its explicitly
    Hermitized counterpart used by the generalized Ritz solve.  The overlap
    and action Gram matrices are also Hermitized after recording their raw
    roundoff diagnostics.  If an exact overlap diagonal is supplied,
    ``sampled_overlap_diagonal`` and the diagnostics still describe the
    unmodified sample; ``overlap`` alone contains the control variate.
    """

    overlap: np.ndarray
    hamiltonian_raw: np.ndarray
    hamiltonian: np.ndarray
    action_gram: np.ndarray
    overlap_hermiticity: HermiticityDiagnostics
    hamiltonian_hermiticity: HermiticityDiagnostics
    action_gram_hermiticity: HermiticityDiagnostics
    overlap_diagonal_policy: str
    sampled_overlap_diagonal: np.ndarray
    exact_overlap_diagonal: np.ndarray | None
    sample_count: int
    state_count: int


@dataclass(frozen=True)
class FailClosedGeneralizedRitz:
    """Full-rank generalized Ritz roots and their sampled normalization."""

    energies: np.ndarray
    coefficients: np.ndarray
    overlap_eigenvalues: np.ndarray
    overlap_condition_number: float
    orthonormality_error: float
    effective_hamiltonian_hermiticity: HermiticityDiagnostics

    @property
    def rank(self) -> int:
        """Number of represented roots; truncation is never performed."""

        return int(len(self.energies))


@dataclass(frozen=True)
class SampledPhysicalResiduals:
    """Direct proposal estimates of ``||(H-E_n)|psi_n>||``."""

    residual_norms: np.ndarray
    residual_norms_squared: np.ndarray
    residual_gram: np.ndarray
    spectral_intervals: np.ndarray
    projected_residual_matrix: np.ndarray
    maximum_projected_residual: float
    residual_gram_minimum_eigenvalue: float


def _normalized_observation_weights(
    sample_count: int, observation_weights: np.ndarray | None
) -> np.ndarray:
    if observation_weights is None:
        return np.full(sample_count, 1.0 / sample_count, dtype=np.float64)
    weights = np.asarray(observation_weights, dtype=np.float64)
    if weights.shape != (sample_count,):
        raise ValueError("observation_weights must have one entry per sample")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("observation_weights must be finite and nonnegative")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("observation_weights must have positive total weight")
    return weights / total


def _scaled_observations(
    amplitudes: np.ndarray,
    hamiltonian_actions: np.ndarray,
    proposal_probabilities: np.ndarray,
    observation_weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(amplitudes, dtype=np.complex128)
    actions = np.asarray(hamiltonian_actions, dtype=np.complex128)
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("amplitudes must be a nonempty sample-by-state matrix")
    if actions.shape != states.shape:
        raise ValueError("hamiltonian_actions must match amplitudes")
    if np.any(~np.isfinite(states)) or np.any(~np.isfinite(actions)):
        raise ValueError("amplitudes and hamiltonian_actions must be finite")
    proposal = np.asarray(proposal_probabilities, dtype=np.float64)
    if proposal.shape != (states.shape[0],):
        raise ValueError("proposal_probabilities must have one entry per sample")
    if np.any(~np.isfinite(proposal)) or np.any(proposal <= 0.0):
        raise ValueError("proposal_probabilities must be finite and positive")
    weights = _normalized_observation_weights(
        states.shape[0], observation_weights
    )
    inverse_sqrt = 1.0 / np.sqrt(proposal)
    scaled_states = states * inverse_sqrt[:, None]
    scaled_actions = actions * inverse_sqrt[:, None]
    if np.any(~np.isfinite(scaled_states)) or np.any(~np.isfinite(scaled_actions)):
        raise FloatingPointError("proposal scaling produced a non-finite value")
    return scaled_states, scaled_actions, weights


def _weighted_gram(
    left: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    return np.einsum(
        "x,xa,xb->ab", weights, left.conj(), right, optimize=True
    )


def _hermiticity_diagnostics(matrix: np.ndarray) -> HermiticityDiagnostics:
    difference = matrix - matrix.conj().T
    maximum = float(np.max(np.abs(difference), initial=0.0))
    denominator = max(
        float(np.linalg.norm(matrix, ord="fro")),
        float(np.finfo(np.float64).tiny),
    )
    return HermiticityDiagnostics(
        maximum_absolute_error=maximum,
        relative_frobenius_error=float(
            np.linalg.norm(difference, ord="fro") / denominator
        ),
    )


def _hermitize(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * (matrix + matrix.conj().T), dtype=np.complex128)


def sampled_subspace_moments(
    amplitudes: np.ndarray,
    hamiltonian_actions: np.ndarray,
    proposal_probabilities: np.ndarray,
    *,
    observation_weights: np.ndarray | None = None,
    exact_overlap_diagonal: np.ndarray | None = None,
) -> SampledSubspaceMoments:
    """Estimate ``S``, ``H``, and ``K`` from a common proposal sample.

    Ordinary independent samples drawn from the supplied proposal use the
    default uniform observation weights.  For exhaustive quadrature, pass the
    normalized proposal probabilities themselves.  For stratified sampling
    from ``rho = sum_i pi_i p_i``, a sample from stratum ``i`` carries
    observation weight ``pi_i / M_i``; the routine normalizes the supplied
    weights to protect against harmless summation roundoff.

    The proposal need not be the Born mixture, but it must be the actual
    positive probability at each observed configuration.  This permits a
    future Hamiltonian-connected mixture without changing the algebra.

    By default every entry of ``S`` is sampled, preserving the historical
    estimator.  Passing ``exact_overlap_diagonal`` is an explicit opt-in
    control variate: only the diagonal of ``S`` is replaced by the supplied
    known squared norms.  Normalized autoregressive roots should pass a vector
    of ones.  Off-diagonal overlaps, all ``H``/``K`` entries, and the raw
    Hamiltonian anti-Hermiticity diagnostic remain untouched.
    """

    scaled_states, scaled_actions, weights = _scaled_observations(
        amplitudes,
        hamiltonian_actions,
        proposal_probabilities,
        observation_weights,
    )
    overlap_raw = _weighted_gram(scaled_states, scaled_states, weights)
    hamiltonian_raw = _weighted_gram(scaled_states, scaled_actions, weights)
    action_gram_raw = _weighted_gram(scaled_actions, scaled_actions, weights)
    overlap_diagnostics = _hermiticity_diagnostics(overlap_raw)
    hamiltonian_diagnostics = _hermiticity_diagnostics(hamiltonian_raw)
    action_diagnostics = _hermiticity_diagnostics(action_gram_raw)
    overlap = _hermitize(overlap_raw)
    sampled_diagonal = np.asarray(np.real(np.diag(overlap)), dtype=np.float64)
    if exact_overlap_diagonal is None:
        diagonal_policy = SAMPLED_OVERLAP_DIAGONAL
        exact_diagonal = None
    else:
        exact_diagonal = np.array(
            exact_overlap_diagonal, dtype=np.float64, copy=True
        )
        if (
            exact_diagonal.shape != (scaled_states.shape[1],)
            or np.any(~np.isfinite(exact_diagonal))
            or np.any(exact_diagonal <= 0.0)
        ):
            raise ValueError(
                "exact_overlap_diagonal must be finite and positive per state"
            )
        overlap = np.array(overlap, dtype=np.complex128, copy=True)
        overlap[np.diag_indices_from(overlap)] = exact_diagonal
        diagonal_policy = EXACT_OVERLAP_DIAGONAL
    return SampledSubspaceMoments(
        overlap=np.asarray(overlap, dtype=np.complex128),
        hamiltonian_raw=np.asarray(hamiltonian_raw, dtype=np.complex128),
        hamiltonian=_hermitize(hamiltonian_raw),
        action_gram=_hermitize(action_gram_raw),
        overlap_hermiticity=overlap_diagnostics,
        hamiltonian_hermiticity=hamiltonian_diagnostics,
        action_gram_hermiticity=action_diagnostics,
        overlap_diagonal_policy=diagonal_policy,
        sampled_overlap_diagonal=np.asarray(sampled_diagonal, dtype=np.float64),
        exact_overlap_diagonal=(
            None
            if exact_diagonal is None
            else np.array(exact_diagonal, dtype=np.float64, copy=True)
        ),
        sample_count=int(scaled_states.shape[0]),
        state_count=int(scaled_states.shape[1]),
    )


def fail_closed_generalized_ritz(
    moments: SampledSubspaceMoments,
    *,
    relative_rank_tolerance: float = 1.0e-10,
    maximum_condition_number: float | None = None,
    maximum_orthonormality_error: float = 1.0e-8,
) -> FailClosedGeneralizedRitz:
    """Solve ``H c = E S c`` only when the full sampled span is reliable.

    Every represented column is required.  A deficient ``R``-column span
    raises :class:`NumericalSubspaceRankError`; no eigenvector is discarded
    and no lower-rank physics result is returned.
    """

    if (
        not np.isfinite(relative_rank_tolerance)
        or not 0.0 < relative_rank_tolerance < 1.0
    ):
        raise ValueError("relative_rank_tolerance must lie strictly between zero and one")
    if maximum_condition_number is not None and (
        not np.isfinite(maximum_condition_number)
        or maximum_condition_number < 1.0
    ):
        raise ValueError("maximum_condition_number must be finite and at least one")
    if (
        not np.isfinite(maximum_orthonormality_error)
        or maximum_orthonormality_error <= 0.0
    ):
        raise ValueError("maximum_orthonormality_error must be finite and positive")

    overlap = np.asarray(moments.overlap, dtype=np.complex128)
    hamiltonian = np.asarray(moments.hamiltonian, dtype=np.complex128)
    expected = (moments.state_count, moments.state_count)
    if overlap.shape != expected or hamiltonian.shape != expected:
        raise ValueError("sampled moment matrices do not match state_count")
    if np.any(~np.isfinite(overlap)) or np.any(~np.isfinite(hamiltonian)):
        raise FloatingPointError("sampled Ritz matrices must be finite")

    try:
        overlap_values, overlap_vectors = np.linalg.eigh(_hermitize(overlap))
    except np.linalg.LinAlgError as error:
        raise NumericalSubspaceRankError(
            "sampled overlap eigendecomposition failed"
        ) from error
    largest = float(overlap_values[-1])
    threshold = relative_rank_tolerance * largest
    numerical_rank = int(np.count_nonzero(overlap_values > threshold))
    if largest <= 0.0 or numerical_rank != moments.state_count:
        raise NumericalSubspaceRankError(
            "sampled overlap is not full rank: "
            f"rank={numerical_rank}/{moments.state_count}, "
            f"lambda_min={float(overlap_values[0]):.6e}, "
            f"lambda_max={largest:.6e}"
        )
    condition = largest / float(overlap_values[0])
    if maximum_condition_number is not None and condition > maximum_condition_number:
        raise NumericalSubspaceRankError(
            "sampled overlap exceeds the registered condition-number gate: "
            f"{condition:.6e} > {maximum_condition_number:.6e}"
        )

    whitening = overlap_vectors / np.sqrt(overlap_values)[None, :]
    effective_raw = whitening.conj().T @ hamiltonian @ whitening
    effective_diagnostics = _hermiticity_diagnostics(effective_raw)
    effective = _hermitize(effective_raw)
    try:
        energies, ritz_vectors = np.linalg.eigh(effective)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError(
            "sampled effective Hamiltonian eigendecomposition failed"
        ) from error
    coefficients = whitening @ ritz_vectors
    normalization = coefficients.conj().T @ overlap @ coefficients
    orthonormality_error = float(
        np.max(
            np.abs(normalization - np.eye(moments.state_count)), initial=0.0
        )
    )
    if (
        not np.all(np.isfinite(energies))
        or not np.all(np.isfinite(coefficients))
        or orthonormality_error > maximum_orthonormality_error
    ):
        raise FloatingPointError(
            "generalized Ritz normalization failed: "
            f"error={orthonormality_error:.6e}"
        )
    return FailClosedGeneralizedRitz(
        energies=np.asarray(energies, dtype=np.float64),
        coefficients=np.asarray(coefficients, dtype=np.complex128),
        overlap_eigenvalues=np.asarray(overlap_values, dtype=np.float64),
        overlap_condition_number=float(condition),
        orthonormality_error=orthonormality_error,
        effective_hamiltonian_hermiticity=effective_diagnostics,
    )


def sampled_physical_residuals(
    amplitudes: np.ndarray,
    hamiltonian_actions: np.ndarray,
    proposal_probabilities: np.ndarray,
    ritz: FailClosedGeneralizedRitz,
    *,
    observation_weights: np.ndarray | None = None,
) -> SampledPhysicalResiduals:
    """Estimate Ritz residuals directly from sampled ``(H-E)|psi>`` rows.

    Direct residual rows retain positivity and avoid subtracting two noisy
    extensive quantities such as ``c^dagger K c - E**2``.  The returned
    spectral intervals are the elementary residual intervals ``[E-r,E+r]``;
    assigning an interval to a particular exact root still requires a
    separate spectral-isolation argument.
    """

    scaled_states, scaled_actions, weights = _scaled_observations(
        amplitudes,
        hamiltonian_actions,
        proposal_probabilities,
        observation_weights,
    )
    state_count = scaled_states.shape[1]
    if ritz.coefficients.shape != (state_count, state_count):
        raise ValueError("Ritz coefficients do not match the sampled states")
    if ritz.energies.shape != (state_count,):
        raise ValueError("Ritz energies do not match the sampled states")
    ritz_states = scaled_states @ ritz.coefficients
    ritz_actions = scaled_actions @ ritz.coefficients
    residual_rows = ritz_actions - ritz_states * ritz.energies[None, :]
    residual_gram_raw = _weighted_gram(residual_rows, residual_rows, weights)
    residual_gram = _hermitize(residual_gram_raw)
    squared = np.real(np.diag(residual_gram))
    scale = max(float(np.max(np.abs(residual_gram), initial=0.0)), 1.0)
    negative_tolerance = 256.0 * np.finfo(np.float64).eps * scale
    if np.any(squared < -negative_tolerance):
        raise FloatingPointError("a directly sampled residual norm is negative")
    squared = np.maximum(squared, 0.0)
    norms = np.sqrt(squared)
    projected = _weighted_gram(ritz_states, residual_rows, weights)
    minimum_residual_eigenvalue = float(
        np.min(np.linalg.eigvalsh(residual_gram))
    )
    return SampledPhysicalResiduals(
        residual_norms=np.asarray(norms, dtype=np.float64),
        residual_norms_squared=np.asarray(squared, dtype=np.float64),
        residual_gram=np.asarray(residual_gram, dtype=np.complex128),
        spectral_intervals=np.stack(
            (ritz.energies - norms, ritz.energies + norms), axis=1
        ),
        projected_residual_matrix=np.asarray(projected, dtype=np.complex128),
        maximum_projected_residual=float(
            np.max(np.abs(projected), initial=0.0)
        ),
        residual_gram_minimum_eigenvalue=minimum_residual_eigenvalue,
    )


__all__ = [
    "EXACT_OVERLAP_DIAGONAL",
    "FailClosedGeneralizedRitz",
    "HermiticityDiagnostics",
    "NumericalSubspaceRankError",
    "SAMPLED_OVERLAP_DIAGONAL",
    "SampledPhysicalResiduals",
    "SampledSubspaceMoments",
    "fail_closed_generalized_ritz",
    "sampled_physical_residuals",
    "sampled_subspace_moments",
]
