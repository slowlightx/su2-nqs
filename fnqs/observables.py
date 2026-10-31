"""Independent wavefunction observables and covariance QGT estimators."""

from __future__ import annotations

import numpy as np

from .basis import all_bits, validate_size


def normalized_state(state: np.ndarray) -> np.ndarray:
    """Validate and normalize a nonzero finite complex state."""
    state = np.asarray(state, dtype=np.complex128)
    norm = np.linalg.norm(state)
    if state.ndim != 1 or not np.isfinite(norm) or norm == 0:
        raise ValueError("state must be a finite nonzero vector")
    return state / norm


def correlations_from_state(state: np.ndarray, n: int) -> np.ndarray:
    """Translation-averaged Pauli ZZ correlations for all separations."""
    state = normalized_state(state)
    bits = all_bits(n)
    if len(state) != len(bits):
        raise ValueError("state dimension does not match n")
    z = 2 * bits - 1
    p = np.abs(state) ** 2
    return np.asarray([p @ np.mean(z * np.roll(z, -r, axis=1), axis=1) for r in range(n)])


def renyi2_from_state(state: np.ndarray, n: int, ell: int) -> float:
    """Exact bipartite S2 via Schmidt singular values, small systems only."""
    validate_size(n)
    if not isinstance(ell, (int, np.integer)) or not 1 <= ell < n or n > 14:
        raise ValueError("requires 1<=ell<n<=14")
    state = normalized_state(state)
    if len(state) != 1 << n:
        raise ValueError("state dimension does not match n")
    singular = np.linalg.svd(state.reshape(1 << ell, 1 << (n - ell)), compute_uv=False)
    return -float(np.log(np.sum(singular ** 4)))


def quantum_geometric_tensor(scores: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Covariance <O_i* O_j>-<O_i*> <O_j> of coupling log derivatives.

    Uniform weights give a Born-sampled (population, 1/M) estimate. Exhaustive
    calculations must pass Born probabilities. This is a coupling metric, not
    the parameter-space SR matrix. Constant complex gauge scores cancel.
    """
    scores = np.asarray(scores, dtype=np.complex128)
    if scores.ndim != 2 or len(scores) < 1 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be a finite nonempty (samples,couplings) array")
    weights = np.ones(len(scores)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(scores),) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative with positive sum")
    p = weights / weights.sum()
    centered = scores - p @ scores
    return centered.conj().T @ (p[:, None] * centered)


def fidelity_metric(state: np.ndarray, plus: np.ndarray, minus: np.ndarray, delta: float) -> float:
    """Central fidelity estimate of a diagonal metric at fixed neural parameters."""
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    state, plus, minus = map(normalized_state, (state, plus, minus))
    return float((2 - abs(np.vdot(state, plus)) ** 2 - abs(np.vdot(state, minus)) ** 2) / (2 * delta ** 2))


def sampled_correlations(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C(r) and independent-sample standard errors for autoregressive draws."""
    bits = np.asarray(bits)
    if bits.ndim != 2 or len(bits) < 2 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("at least two binary configurations are required")
    validate_size(bits.shape[1])
    z = 2 * bits.astype(np.int32) - 1
    records = np.stack([np.mean(z * np.roll(z, -r, axis=1), axis=1)
                        for r in range(bits.shape[1])], axis=1)
    return records.mean(axis=0), records.std(axis=0, ddof=1) / np.sqrt(len(bits))
