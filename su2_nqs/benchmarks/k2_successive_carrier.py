"""Exact K2 successive-exchange oracle for spin-half Hamiltonians.

For ``H = sum_b J_b h_b`` with ``h_b = S_i.S_j``, the second action is split
into repeated-bond, connected distinct-bond, and disconnected distinct-bond
histories.  The repeated sector is reduced analytically with

``h_b**2 = 3 I / 16 - h_b / 2``.

This is an exact finite-vector reference for designing a scalable causal K2
carrier.  It does not enumerate magnetic states, but it does act on a full
fusion-space vector and therefore is not itself a production NQS recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class K2SuccessiveAction:
    """Exact components of one ``H**2`` action in a fusion basis."""

    repeated: np.ndarray
    connected: np.ndarray
    disconnected: np.ndarray
    total: np.ndarray
    direct: np.ndarray
    bond_count: int
    repeated_history_count: int
    connected_history_count: int
    disconnected_history_count: int

    @property
    def relative_closure_error(self) -> float:
        """Relative error of the component sum against direct ``H(H psi)``."""

        denominator = max(np.linalg.norm(self.direct), np.finfo(float).tiny)
        return float(np.linalg.norm(self.total - self.direct) / denominator)


@dataclass(frozen=True)
class K2LogCumulant:
    """Second imaginary-time derivative of ``log(psi)`` by bond locality.

    The stored vectors are ``psi * kappa_2`` rather than the pointwise
    cumulants themselves.  This representation has the ordinary Hilbert-space
    norm and transforms covariantly under phase choices of the fusion basis.
    For ``psi(tau) = exp(-tau H) psi``, their sum obeys

    ``kappa_2 = (H**2 psi) / psi - ((H psi) / psi)**2``.

    A disconnected component that vanishes is generated completely by the
    product of first-order local responses and needs no independent K2 carrier.
    """

    repeated: np.ndarray
    connected: np.ndarray
    disconnected: np.ndarray
    total: np.ndarray
    direct: np.ndarray
    action: K2SuccessiveAction
    minimum_amplitude: float

    @property
    def relative_closure_error(self) -> float:
        """Relative error of the locality sum against the direct cumulant."""

        denominator = max(np.linalg.norm(self.direct), np.finfo(float).tiny)
        return float(np.linalg.norm(self.total - self.direct) / denominator)


def decompose_k2_successive_action(
    initial: np.ndarray,
    local_exchange_operators: Sequence[object],
    bond_endpoints: Sequence[tuple[int, int]],
    couplings: Sequence[Real],
) -> K2SuccessiveAction:
    """Decompose ``H**2 initial`` by physical bond-line-graph locality.

    ``bond_endpoints[b]`` contains physical site labels, independent of the
    fusion-tree leaf order used to construct the corresponding operator.
    Ordered histories are retained because overlapping exchanges need not
    commute.  All arrays are complex128 exact-reference vectors.
    """

    vector = np.asarray(initial, dtype=np.complex128)
    if vector.ndim != 1 or len(vector) < 1:
        raise ValueError("initial must be a nonempty vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("initial must be finite")
    operators = tuple(local_exchange_operators)
    endpoints = tuple((int(first), int(second)) for first, second in bond_endpoints)
    weights = np.asarray(couplings, dtype=np.float64)
    bond_count = len(operators)
    if bond_count < 1:
        raise ValueError("at least one local exchange operator is required")
    if len(endpoints) != bond_count or weights.shape != (bond_count,):
        raise ValueError("operators, endpoints, and couplings must have equal length")
    if not np.all(np.isfinite(weights)):
        raise ValueError("couplings must be finite")
    for index, (operator, endpoint) in enumerate(zip(operators, endpoints, strict=True)):
        if getattr(operator, "shape", None) != (len(vector), len(vector)):
            raise ValueError(f"local operator {index} has the wrong shape")
        if endpoint[0] == endpoint[1] or min(endpoint) < 0:
            raise ValueError("each bond must join two distinct non-negative sites")

    first_actions = tuple(
        np.asarray(operator @ vector, dtype=np.complex128)
        for operator in operators
    )
    repeated = np.zeros_like(vector)
    connected = np.zeros_like(vector)
    disconnected = np.zeros_like(vector)
    connected_count = 0
    disconnected_count = 0
    endpoint_sets = tuple(set(pair) for pair in endpoints)
    for first in range(bond_count):
        repeated += weights[first] ** 2 * (
            (3.0 / 16.0) * vector - 0.5 * first_actions[first]
        )
        for second in range(bond_count):
            if second == first:
                continue
            action = np.asarray(
                operators[second] @ first_actions[first], dtype=np.complex128
            )
            weighted = weights[second] * weights[first] * action
            if endpoint_sets[first] & endpoint_sets[second]:
                connected += weighted
                connected_count += 1
            else:
                disconnected += weighted
                disconnected_count += 1

    h_initial = sum(
        (weight * action for weight, action in zip(weights, first_actions, strict=True)),
        np.zeros_like(vector),
    )
    direct = sum(
        (
            weight * np.asarray(operator @ h_initial, dtype=np.complex128)
            for weight, operator in zip(weights, operators, strict=True)
        ),
        np.zeros_like(vector),
    )
    total = repeated + connected + disconnected
    return K2SuccessiveAction(
        repeated=repeated,
        connected=connected,
        disconnected=disconnected,
        total=total,
        direct=direct,
        bond_count=bond_count,
        repeated_history_count=bond_count,
        connected_history_count=connected_count,
        disconnected_history_count=disconnected_count,
    )


def decompose_k2_log_cumulant(
    initial: np.ndarray,
    local_exchange_operators: Sequence[object],
    bond_endpoints: Sequence[tuple[int, int]],
    couplings: Sequence[Real],
    *,
    amplitude_tolerance: Real = 0.0,
) -> K2LogCumulant:
    """Decompose the exact second log-wavefunction response by bond locality.

    No fit parameters enter this oracle.  Ordered bond pairs are assigned to
    repeated, connected, or disconnected sectors exactly as in
    :func:`decompose_k2_successive_action`.  For each pair ``(b, c)``, the
    factorized first-order response

    ``(h_b psi) * (h_c psi) / psi``

    is subtracted from ``h_c h_b psi``.  Exact or near nodes are rejected
    because a pointwise logarithmic derivative is not defined there.
    """

    tolerance = float(amplitude_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("amplitude_tolerance must be finite and non-negative")
    action = decompose_k2_successive_action(
        initial, local_exchange_operators, bond_endpoints, couplings
    )
    vector = np.asarray(initial, dtype=np.complex128)
    minimum_amplitude = float(np.min(np.abs(vector)))
    if minimum_amplitude <= tolerance:
        raise ValueError(
            "log cumulant requires every initial amplitude to exceed "
            "amplitude_tolerance"
        )

    operators = tuple(local_exchange_operators)
    endpoints = tuple((int(first), int(second)) for first, second in bond_endpoints)
    weights = np.asarray(couplings, dtype=np.float64)
    first_actions = tuple(
        np.asarray(operator @ vector, dtype=np.complex128)
        for operator in operators
    )
    products = {
        "repeated": np.zeros_like(vector),
        "connected": np.zeros_like(vector),
        "disconnected": np.zeros_like(vector),
    }
    endpoint_sets = tuple(set(pair) for pair in endpoints)
    for first in range(len(operators)):
        for second in range(len(operators)):
            if first == second:
                name = "repeated"
            elif endpoint_sets[first] & endpoint_sets[second]:
                name = "connected"
            else:
                name = "disconnected"
            products[name] += (
                weights[second]
                * weights[first]
                * first_actions[first]
                * first_actions[second]
                / vector
            )

    repeated = action.repeated - products["repeated"]
    connected = action.connected - products["connected"]
    disconnected = action.disconnected - products["disconnected"]
    total = repeated + connected + disconnected
    h_initial = sum(
        (
            weight * first_action
            for weight, first_action in zip(weights, first_actions, strict=True)
        ),
        np.zeros_like(vector),
    )
    direct = action.direct - h_initial * h_initial / vector
    return K2LogCumulant(
        repeated=repeated,
        connected=connected,
        disconnected=disconnected,
        total=total,
        direct=direct,
        action=action,
        minimum_amplitude=minimum_amplitude,
    )


def born_centered_vector(initial: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Remove the scalar log-amplitude gauge in the Born inner product."""

    vector = np.asarray(initial, dtype=np.complex128)
    candidate = np.asarray(value, dtype=np.complex128)
    if vector.ndim != 1 or candidate.shape != vector.shape:
        raise ValueError("initial and value must be equal-length vectors")
    norm_squared = float(np.vdot(vector, vector).real)
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("initial must have positive finite norm")
    mean = np.vdot(vector, candidate) / norm_squared
    return candidate - mean * vector


def log_cumulant_diagnostics(
    initial: np.ndarray, cumulant: K2LogCumulant
) -> dict[str, object]:
    """Measure linked-sector weights without changing model dimension.

    Component fractions are Gram-matrix entries divided by the squared norm of
    the centered total cumulant, so interference is explicit and diagonal
    fractions need not sum to one.  Suppression ratios compare each cumulant
    sector with the corresponding raw ``H**2`` action sector.
    """

    vector = np.asarray(initial, dtype=np.complex128)
    if vector.shape != cumulant.total.shape:
        raise ValueError("initial and cumulant shapes disagree")
    names = ("repeated", "connected", "disconnected")
    centered = {
        name: born_centered_vector(vector, np.asarray(getattr(cumulant, name)))
        for name in names
    }
    centered_action = {
        name: born_centered_vector(
            vector, np.asarray(getattr(cumulant.action, name))
        )
        for name in names
    }
    total = born_centered_vector(vector, cumulant.total)
    action_total = born_centered_vector(vector, cumulant.action.total)
    denominator = max(float(np.vdot(total, total).real), np.finfo(float).tiny)
    gram = {
        first: {
            second: float(
                np.vdot(centered[first], centered[second]).real / denominator
            )
            for second in names
        }
        for first in names
    }

    def suppression_ratio(name: str) -> float:
        return float(
            np.linalg.norm(centered[name])
            / max(np.linalg.norm(centered_action[name]), np.finfo(float).tiny)
        )

    reconstructed = sum(centered.values(), np.zeros_like(total))
    return {
        "centered_cumulant_norm": float(np.linalg.norm(total)),
        "centered_h2_action_norm": float(np.linalg.norm(action_total)),
        "total_cumulant_to_h2_action_norm_ratio": float(
            np.linalg.norm(total)
            / max(np.linalg.norm(action_total), np.finfo(float).tiny)
        ),
        "component_norm_squared_fractions": {
            name: gram[name][name] for name in names
        },
        "component_gram_fractions": gram,
        "connected_disconnected_interference_fraction": 2.0
        * gram["connected"]["disconnected"],
        "component_cumulant_to_action_norm_ratios": {
            name: suppression_ratio(name) for name in names
        },
        "centered_reconstruction_relative_error": float(
            np.linalg.norm(reconstructed - total)
            / max(np.linalg.norm(total), np.finfo(float).tiny)
        ),
    }


def k1_complement_decomposition(
    initial: np.ndarray,
    h_initial: np.ndarray,
    action: K2SuccessiveAction,
    *,
    resolved_first_actions: Sequence[np.ndarray] = (),
) -> dict[str, object]:
    """Measure K2 components after removing a declared K1 space.

    With no ``resolved_first_actions`` the removed space is the collapsed
    ``span{psi, H psi}``.  Supplying all ``h_b psi`` actions gives resolved K1,
    in whose complement the repeated-bond K2 sector must vanish exactly.
    Fractions use the squared norm of the total K1-orthogonal K2 residual as
    denominator; off-diagonal Gram entries expose interference.
    """

    seed = np.asarray(initial, dtype=np.complex128)
    first = np.asarray(h_initial, dtype=np.complex128)
    if seed.shape != action.total.shape or first.shape != seed.shape:
        raise ValueError("K1 vectors and K2 action shapes disagree")
    resolved = tuple(np.asarray(value, dtype=np.complex128) for value in resolved_first_actions)
    if any(value.shape != seed.shape for value in resolved):
        raise ValueError("resolved K1 action shapes disagree")
    candidates = np.column_stack((seed, first, *resolved))
    left, singular, _ = np.linalg.svd(candidates, full_matrices=False)
    threshold = np.finfo(float).eps * max(float(singular[0]), 1.0) * 16.0
    rank = int(np.count_nonzero(singular > threshold))
    basis = left[:, :rank]

    def remove_k1(value: np.ndarray) -> np.ndarray:
        return value - basis @ (basis.conj().T @ value)

    names = ("repeated", "connected", "disconnected")
    residuals = {
        name: remove_k1(np.asarray(getattr(action, name))) for name in names
    }
    total = remove_k1(action.total)
    denominator = max(float(np.vdot(total, total).real), np.finfo(float).tiny)
    gram = {
        first_name: {
            second_name: float(
                np.vdot(residuals[first_name], residuals[second_name]).real
                / denominator
            )
            for second_name in names
        }
        for first_name in names
    }
    reconstructed = sum(residuals.values(), np.zeros_like(total))
    return {
        "k1_rank": rank,
        "k1_space": "resolved" if resolved else "collapsed",
        "k2_complement_norm": float(np.sqrt(denominator)),
        "component_norm_squared_fractions": {
            name: gram[name][name] for name in names
        },
        "component_gram_fractions": gram,
        "connected_disconnected_interference_fraction": 2.0
        * gram["connected"]["disconnected"],
        "repeated_complement_relative_norm": float(
            np.linalg.norm(residuals["repeated"])
            / max(np.linalg.norm(action.repeated), np.finfo(float).tiny)
        ),
        "complement_reconstruction_relative_error": float(
            np.linalg.norm(reconstructed - total)
            / max(np.linalg.norm(total), np.finfo(float).tiny)
        ),
    }


__all__ = [
    "K2LogCumulant",
    "K2SuccessiveAction",
    "born_centered_vector",
    "decompose_k2_log_cumulant",
    "decompose_k2_successive_action",
    "k1_complement_decomposition",
    "log_cumulant_diagnostics",
]
