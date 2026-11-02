"""Metrics used to compare fusion-space and fixed-Sz wavefunctions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WavefunctionMetrics:
    energy: float
    variance: float
    spin_squared: float
    parameter_count: int | None = None


def normalized_expectation(vector: np.ndarray, operator: np.ndarray) -> complex:
    norm = np.vdot(vector, vector)
    if abs(norm) == 0:
        raise ValueError("wavefunction has zero norm")
    return np.vdot(vector, operator @ vector) / norm


def wavefunction_metrics(
    vector: np.ndarray,
    hamiltonian: np.ndarray,
    spin_squared: np.ndarray,
    *,
    parameter_count: int | None = None,
) -> WavefunctionMetrics:
    """Compute energy, variance, and spin contamination exactly."""

    energy = normalized_expectation(vector, hamiltonian)
    energy_squared = normalized_expectation(vector, hamiltonian @ hamiltonian)
    variance = float(np.real(energy_squared - energy * energy))
    s2 = normalized_expectation(vector, spin_squared)
    return WavefunctionMetrics(float(np.real(energy)), variance, float(np.real(s2)), parameter_count)

