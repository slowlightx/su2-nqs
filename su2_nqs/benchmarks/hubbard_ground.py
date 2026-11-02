"""Exact physical observables for Hubbard states in fusion-basis order.

Fock expansion is used only as a small-system observable oracle.  Hamiltonian
construction remains in the reduced charge/spin fusion space.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np

from ..su2.hubbard_fusion import HubbardFusionState
from .hubbard_coupled_basis import hubbard_coupled_state
from .hubbard_fock import enumerate_fock_basis


def hubbard_fock_vector(
    states: Sequence[HubbardFusionState], coefficients: np.ndarray
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Expand fusion coefficients into the fixed-charge Fock basis."""

    if not states:
        raise ValueError("states must be nonempty")
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    if coefficients.shape != (len(states),):
        raise ValueError("coefficient count must match the fusion basis")
    n_sites = states[0].n_sites
    n_electrons = states[0].n_electrons
    if any(
        state.n_sites != n_sites or state.n_electrons != n_electrons
        for state in states
    ):
        raise ValueError("all states must belong to one charge sector")
    basis = enumerate_fock_basis(n_sites, n_electrons)
    positions = {mask: index for index, mask in enumerate(basis)}
    vector = np.zeros(len(basis), dtype=np.complex128)
    for coefficient, state in zip(coefficients, states, strict=True):
        if abs(coefficient) <= 1.0e-16:
            continue
        for mask, amplitude in hubbard_coupled_state(state, total_m=0).items():
            vector[positions[mask]] += coefficient * amplitude
    return vector, basis


def hubbard_ground_correlations(
    fock_vector: np.ndarray,
    fock_basis: Sequence[int],
    n_sites: int,
    n_electrons: int,
) -> dict[str, float]:
    """Evaluate singlet spin and charge correlations in a Fock oracle."""

    fock_vector = np.asarray(fock_vector, dtype=np.complex128)
    if fock_vector.shape != (len(fock_basis),):
        raise ValueError("Fock vector and basis sizes differ")
    probabilities = np.abs(fock_vector) ** 2
    density = n_electrons / n_sites
    nearest_spin = 0.0
    nearest_charge = 0.0
    staggered_spin = 0.0
    staggered_charge = 0.0
    phases = (-1.0) ** np.arange(n_sites)
    for probability, mask in zip(probabilities, fock_basis, strict=True):
        occupations = np.asarray(
            [
                ((mask >> (2 * site)) & 1)
                + ((mask >> (2 * site + 1)) & 1)
                for site in range(n_sites)
            ],
            dtype=np.float64,
        )
        spin_z = np.asarray(
            [
                0.5
                * (
                    ((mask >> (2 * site)) & 1)
                    - ((mask >> (2 * site + 1)) & 1)
                )
                for site in range(n_sites)
            ],
            dtype=np.float64,
        )
        charge = occupations - density
        nearest_spin += probability * 3.0 * float(
            np.mean(spin_z[:-1] * spin_z[1:])
        )
        nearest_charge += probability * float(
            np.mean(charge[:-1] * charge[1:])
        )
        staggered_spin += probability * 3.0 / n_sites * float(
            (phases @ spin_z) ** 2
        )
        staggered_charge += probability / n_sites * float(
            (phases @ charge) ** 2
        )
    return {
        "fock_normalization": float(np.sum(probabilities)),
        "nearest_neighbor_spin_correlation": float(nearest_spin),
        "nearest_neighbor_charge_correlation": float(nearest_charge),
        "staggered_spin_structure_factor": float(staggered_spin),
        "staggered_charge_structure_factor": float(staggered_charge),
    }


def hubbard_ground_state_metrics(
    matrix: Any,
    states: Sequence[HubbardFusionState],
    energy: float,
    vector: np.ndarray,
    residual: float,
    eigensolver_seconds: float,
) -> dict[str, object]:
    """Return exact small-system ground-state observables and costs.

    ``matrix`` must provide SciPy sparse ``tolil``/``tocsc`` operations.  Its
    rows and columns, ``states``, and ``vector`` must share deterministic
    fusion-basis order.
    """

    if not states:
        raise ValueError("states must be nonempty")
    vector = np.asarray(vector, dtype=np.complex128)
    if vector.shape != (len(states),):
        raise ValueError("state vector must match the fusion basis")
    probabilities = np.abs(vector) ** 2
    n_sites = states[0].n_sites
    n_electrons = states[0].n_electrons
    doublon_counts = np.asarray(
        [state.occupations.count(2) for state in states], dtype=np.float64
    )
    minimum_doublons = int(np.min(doublon_counts))
    path_maximum = np.asarray(
        [max(state.path) for state in states], dtype=np.float64
    )
    path_area = np.asarray(
        [sum(state.path) for state in states], dtype=np.float64
    )
    off_diagonal = matrix.copy().tolil()
    off_diagonal.setdiag(0.0)
    off_diagonal = off_diagonal.tocsr()
    off_diagonal.eliminate_zeros()
    degrees = np.diff(off_diagonal.tocsc().indptr)
    positive = probabilities > 0.0
    entropy = -float(
        np.sum(probabilities[positive] * np.log(probabilities[positive]))
    )
    fock_vector, fock_basis = hubbard_fock_vector(states, vector)
    correlations = hubbard_ground_correlations(
        fock_vector, fock_basis, n_sites, n_electrons
    )
    expected_doublons = float(probabilities @ doublon_counts)
    return {
        "energy": float(energy),
        "energy_per_site": float(energy) / n_sites,
        "eigen_residual": float(residual),
        "eigensolver_seconds": float(eigensolver_seconds),
        "normalization": float(np.sum(probabilities)),
        "double_occupancy_per_site": expected_doublons / n_sites,
        "local_moment_per_site": (n_electrons - 2.0 * expected_doublons)
        / n_sites,
        "minimum_doublon_sector_mass": float(
            np.sum(probabilities[doublon_counts == minimum_doublons])
        ),
        "fusion_probability_participation": math.exp(entropy),
        "mean_maximum_path_q": float(probabilities @ path_maximum),
        "mean_path_q_area": float(probabilities @ path_area),
        "mean_hopping_degree": float(np.mean(degrees)),
        "probability_weighted_hopping_degree": float(probabilities @ degrees),
        "maximum_hopping_degree": int(np.max(degrees, initial=0)),
        **correlations,
    }


def hubbard_variational_state_metrics(
    matrix: Any,
    states: Sequence[HubbardFusionState],
    vector: np.ndarray,
    exact_ground: np.ndarray,
    exact_energy: float,
) -> dict[str, float]:
    """Compare one variational fusion vector with an exact ground state.

    Both vectors use the deterministic order in ``states``.  The variational
    vector is normalized here so callers cannot accidentally turn amplitude
    scale into an observable error.
    """

    vector = np.asarray(vector, dtype=np.complex128)
    exact_ground = np.asarray(exact_ground, dtype=np.complex128)
    expected_shape = (len(states),)
    if vector.shape != expected_shape or exact_ground.shape != expected_shape:
        raise ValueError("variational and exact vectors must match the fusion basis")
    norm = np.linalg.norm(vector)
    exact_norm = np.linalg.norm(exact_ground)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("variational vector must have finite nonzero norm")
    if not np.isfinite(exact_norm) or exact_norm == 0.0:
        raise ValueError("exact vector must have finite nonzero norm")
    vector = vector / norm
    exact_ground = exact_ground / exact_norm
    energy = float(np.real(np.vdot(vector, matrix @ vector)))
    residual = float(np.linalg.norm(matrix @ vector - energy * vector))
    physical = hubbard_ground_state_metrics(
        matrix, states, energy, vector, residual, 0.0
    )
    return {
        "energy": energy,
        "energy_error": energy - exact_energy,
        "energy_error_per_site": (energy - exact_energy) / states[0].n_sites,
        "variance_per_site": residual**2 / states[0].n_sites,
        "ground_state_fidelity": float(
            abs(np.vdot(exact_ground, vector)) ** 2
        ),
        "state_residual": residual,
        "normalization": float(np.vdot(vector, vector).real),
        "double_occupancy_per_site": float(
            physical["double_occupancy_per_site"]
        ),
        "local_moment_per_site": float(physical["local_moment_per_site"]),
        "staggered_spin_structure_factor": float(
            physical["staggered_spin_structure_factor"]
        ),
        "staggered_charge_structure_factor": float(
            physical["staggered_charge_structure_factor"]
        ),
        "fusion_probability_participation": float(
            physical["fusion_probability_participation"]
        ),
        "probability_weighted_hopping_degree": float(
            physical["probability_weighted_hopping_degree"]
        ),
    }


def hubbard_observable_errors(
    metrics: dict[str, float], exact: dict[str, float]
) -> dict[str, float]:
    """Signed errors for the registered Hubbard physical observables."""

    names = (
        "double_occupancy_per_site",
        "local_moment_per_site",
        "staggered_spin_structure_factor",
        "staggered_charge_structure_factor",
        "fusion_probability_participation",
        "probability_weighted_hopping_degree",
    )
    return {name: metrics[name] - exact[name] for name in names}


__all__ = [
    "hubbard_fock_vector",
    "hubbard_ground_correlations",
    "hubbard_ground_state_metrics",
    "hubbard_observable_errors",
    "hubbard_variational_state_metrics",
]
