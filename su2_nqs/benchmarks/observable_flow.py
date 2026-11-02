"""Exact observable-resolved imaginary-time and local-action flows.

These routines are intentionally model agnostic: a sparse Hermitian matrix,
an initial vector, and observable matrices are sufficient.  They provide the
small-system reference against which a sampled SU(2)-NQS flow can later be
compared without changing observable definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Iterator, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ImaginaryTimePoint:
    """One normalized state on an exact imaginary-time trajectory."""

    tau: float
    state: np.ndarray
    energy: float
    ground_fidelity: float


@dataclass(frozen=True)
class RitzPoint:
    """Ground Ritz vector in ``span{psi,H psi,...,H**depth psi}``."""

    depth: int
    state: np.ndarray
    energy: float


def normalized_state(state: np.ndarray, *, tolerance: float = 1.0e-14) -> np.ndarray:
    """Return a complex128 unit vector and reject zero/nonfinite input."""

    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or np.any(~np.isfinite(vector)):
        raise ValueError("state must be a finite one-dimensional vector")
    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        raise ValueError("state norm is zero")
    return vector / norm


def _dense_hermitian(matrix) -> np.ndarray:
    dense = np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix)
    dense = np.asarray(dense, dtype=np.complex128)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError("hamiltonian must be square")
    defect = float(np.linalg.norm(dense - dense.conj().T))
    if defect > 1.0e-10:
        raise ValueError(f"hamiltonian is not Hermitian: defect={defect:.3e}")
    return 0.5 * (dense + dense.conj().T)


def exact_imaginary_time_flow(
    hamiltonian,
    initial: np.ndarray,
    times: Sequence[float],
) -> tuple[np.ndarray, tuple[ImaginaryTimePoint, ...]]:
    """Return the exact normalized ``exp(-tau H)`` trajectory.

    The propagator is shifted by the ground energy before exponentiation; the
    normalized state is unchanged and large-``tau`` underflow is avoided.
    """

    values = tuple(float(value) for value in times)
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("times must be a nonempty finite nonnegative sequence")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("times must be strictly increasing")
    dense = _dense_hermitian(hamiltonian)
    vector = normalized_state(initial)
    if dense.shape[0] != len(vector):
        raise ValueError("hamiltonian and initial state dimensions disagree")
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    ground = normalized_state(eigenvectors[:, 0])
    coefficients = eigenvectors.conj().T @ vector
    points: list[ImaginaryTimePoint] = []
    for tau in values:
        weights = np.exp(-tau * (eigenvalues - eigenvalues[0]))
        state = normalized_state(eigenvectors @ (weights * coefficients))
        energy = float(np.real(np.vdot(state, dense @ state)))
        fidelity = float(abs(np.vdot(ground, state)) ** 2)
        points.append(ImaginaryTimePoint(tau, state, energy, fidelity))
    return ground, tuple(points)


def full_reorthogonalized_ritz_flow(
    hamiltonian,
    initial: np.ndarray,
    maximum_depth: int,
    *,
    breakdown_tolerance: float = 1.0e-12,
) -> Iterator[RitzPoint]:
    """Yield nested, twice-reorthogonalized polynomial-Krylov Ritz states."""

    if isinstance(maximum_depth, bool) or not isinstance(maximum_depth, int):
        raise TypeError("maximum_depth must be an integer")
    if maximum_depth < 0:
        raise ValueError("maximum_depth must be nonnegative")
    vector = normalized_state(initial)
    if hamiltonian.shape != (len(vector), len(vector)):
        raise ValueError("hamiltonian and initial state dimensions disagree")
    basis_vectors = [vector]
    for depth in range(maximum_depth + 1):
        basis = np.column_stack(basis_vectors)
        projected = basis.conj().T @ (hamiltonian @ basis)
        projected = 0.5 * (projected + projected.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(projected)
        state = normalized_state(basis @ eigenvectors[:, 0])
        yield RitzPoint(depth, state, float(eigenvalues[0]))
        if depth == maximum_depth:
            break
        candidate = np.asarray(hamiltonian @ basis_vectors[-1], dtype=np.complex128)
        for _ in range(2):
            for previous in basis_vectors:
                candidate -= previous * np.vdot(previous, candidate)
        norm = float(np.linalg.norm(candidate))
        if norm <= breakdown_tolerance:
            break
        basis_vectors.append(candidate / norm)


def expectation(state: np.ndarray, operator) -> complex:
    """Return ``<operator>`` in a normalized state."""

    vector = normalized_state(state)
    if operator.shape != (len(vector), len(vector)):
        raise ValueError("operator and state dimensions disagree")
    return complex(np.vdot(vector, operator @ vector))


def scalar_moments(
    state: np.ndarray, operator, *, normalization: Real
) -> dict[str, float]:
    """Return scalar mean, raw second moment, and connected second moment.

    ``normalization`` divides the mean, while its square divides both second
    moments.  The definition works for Hermitian real or complex operators.
    """

    scale = float(normalization)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization must be finite and positive")
    vector = normalized_state(state)
    if operator.shape != (len(vector), len(vector)):
        raise ValueError("operator and state dimensions disagree")
    applied = np.asarray(operator @ vector, dtype=np.complex128)
    mean = complex(np.vdot(vector, applied))
    raw_second = float(np.real(np.vdot(applied, applied))) / scale**2
    mean_squared = float(abs(mean) ** 2) / scale**2
    return {
        "mean_real": float(mean.real) / scale,
        "mean_imaginary": float(mean.imag) / scale,
        "raw_second": raw_second,
        "mean_squared": mean_squared,
        "connected_second": max(raw_second - mean_squared, 0.0),
    }


def observable_expectations(
    state: np.ndarray, operators: Mapping[str, object]
) -> dict[str, float]:
    """Evaluate a mapping of Hermitian operators as real expectations."""

    result: dict[str, float] = {}
    for name, operator in operators.items():
        value = expectation(state, operator)
        if abs(value.imag) > 1.0e-10:
            raise ValueError(f"observable {name!r} has a complex expectation")
        result[str(name)] = float(value.real)
    return result


def lattice_observable_panel(
    lattice: str,
    state: np.ndarray,
    hamiltonian,
    h2,
    operators: Mapping[str, object],
    *,
    n_sites: int,
    triangular_plaquette_count: int | None = None,
) -> dict[str, float]:
    """Evaluate the common square/triangular physical-accuracy panel."""

    if lattice not in ("square", "triangular"):
        raise ValueError("lattice must be square or triangular")
    if n_sites < 2:
        raise ValueError("n_sites must be at least two")
    vector = normalized_state(state)
    energy = expectation(vector, hamiltonian)
    centered = hamiltonian @ vector - energy * vector
    panel = {
        "energy_per_site": float(energy.real) / n_sites,
        "energy_variance_per_site_squared": float(
            np.vdot(centered, centered).real
        )
        / n_sites**2,
        "h2_connected": scalar_moments(
            vector, h2, normalization=n_sites
        )["connected_second"],
    }
    if lattice == "square":
        required = {
            "neel",
            "stripe_x",
            "stripe_y",
            "b1g",
            "columnar_x",
            "columnar_y",
        }
        if not required <= set(operators):
            raise ValueError("square observable panel is incomplete")
        b1g = scalar_moments(vector, operators["b1g"], normalization=n_sites)
        dimer_x = scalar_moments(
            vector, operators["columnar_x"], normalization=n_sites
        )
        dimer_y = scalar_moments(
            vector, operators["columnar_y"], normalization=n_sites
        )
        panel.update(
            {
                "neel_m2": float(expectation(vector, operators["neel"]).real),
                "stripe_m2_mean": float(
                    (
                        expectation(vector, operators["stripe_x"]).real
                        + expectation(vector, operators["stripe_y"]).real
                    )
                    / 2.0
                ),
                "b1g_mean": b1g["mean_real"],
                "b1g_connected": b1g["connected_second"],
                "columnar_dimer_connected": dimer_x["connected_second"]
                + dimer_y["connected_second"],
                "columnar_dimer_raw": dimer_x["raw_second"]
                + dimer_y["raw_second"],
            }
        )
        return panel

    required = {
        "k",
        "m1",
        "m2",
        "m3",
        "nematic_1",
        "nematic_2",
        "chirality",
    }
    if not required <= set(operators):
        raise ValueError("triangular observable panel is incomplete")
    if triangular_plaquette_count is None or triangular_plaquette_count < 1:
        raise ValueError("triangular_plaquette_count must be positive")
    nematic_1 = scalar_moments(
        vector, operators["nematic_1"], normalization=n_sites
    )
    nematic_2 = scalar_moments(
        vector, operators["nematic_2"], normalization=n_sites
    )
    chirality = scalar_moments(
        vector,
        operators["chirality"],
        normalization=triangular_plaquette_count,
    )
    panel.update(
        {
            "k_m2": float(expectation(vector, operators["k"]).real),
            "stripe_m2_mean": float(
                sum(
                    expectation(vector, operators[name]).real
                    for name in ("m1", "m2", "m3")
                )
                / 3.0
            ),
            "bond_nematic_connected": nematic_1["connected_second"]
            + nematic_2["connected_second"],
            "chirality_mean": chirality["mean_real"],
            "chirality_connected": chirality["connected_second"],
        }
    )
    return panel


__all__ = [
    "ImaginaryTimePoint",
    "RitzPoint",
    "exact_imaginary_time_flow",
    "expectation",
    "full_reorthogonalized_ritz_flow",
    "lattice_observable_panel",
    "normalized_state",
    "observable_expectations",
    "scalar_moments",
]
