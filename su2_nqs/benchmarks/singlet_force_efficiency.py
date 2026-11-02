"""Exact-Born work--variance benchmark for singlet force estimators.

This benchmark asks when the diagonal ``3 Hz`` estimator enabled by an exact
singlet ansatz repays its larger sampling variance by avoiding connected-state
network evaluations.  It uses only ``N=8`` exact-small states and i.i.d. Born
sampling.  The reported work count is an amplitude-evaluation proxy, not a
wall-clock or GPU claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Iterable

import numpy as np

from ..operators.spin_config import SpinBond, heisenberg_spin_connectivity
from .exact_diagonalization import heisenberg_sz_matrix
from .singlet_diagonal_estimator import heisenberg_z_diagonal, singlet_isometry_in_sz


@dataclass(frozen=True)
class SingletForceEfficiencyRow:
    """One state-quality and sample-count force comparison."""

    ground_state_overlap: float
    energy: float
    energy_error: float
    full_local_energy_variance: float
    diagonal_local_energy_variance: float
    sample_count: int
    replicates: int
    tangent_count: int
    full_force_asymptotic_mse_coefficient: float
    diagonal_force_asymptotic_mse_coefficient: float
    diagonal_over_full_asymptotic_mse: float | None
    diagonal_over_full_asymptotic_work_adjusted_mse: float | None
    full_force_mse: float
    diagonal_force_mse: float
    diagonal_over_full_force_mse: float | None
    full_amplitude_work_per_sample: float
    diagonal_amplitude_work_per_sample: float
    diagonal_over_full_work_adjusted_mse: float | None


def _bits(state: int, n_sites: int) -> tuple[int, ...]:
    return tuple((int(state) >> (n_sites - site - 1)) & 1 for site in range(n_sites))


def _checked_probabilities(state: np.ndarray) -> np.ndarray:
    probabilities = np.abs(state) ** 2
    norm = float(np.sum(probabilities))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("state must have finite positive norm")
    result = probabilities / norm
    return result / np.sum(result)


def _centered_force(
    probabilities: np.ndarray,
    scores: np.ndarray,
    local: np.ndarray,
) -> np.ndarray:
    energy = np.sum(probabilities * local)
    return 2.0 * np.real(
        np.sum(
            probabilities[:, None]
            * np.conj(scores)
            * (local - energy)[:, None],
            axis=0,
        )
    )


def _sampled_force(
    indices: np.ndarray,
    scores: np.ndarray,
    local: np.ndarray,
) -> np.ndarray:
    count = len(indices)
    centered = local[indices] - np.mean(local[indices])
    # The ordinary sample covariance uses divisor M and has expectation
    # (1-1/M) times the population covariance.  Debias it so the MSE measures
    # variance rather than mixing a known finite-M bias into the comparison.
    return (
        2.0
        * count
        / (count - 1)
        * np.real(np.mean(np.conj(scores[indices]) * centered[:, None], axis=0))
    )


def _asymptotic_force_mse_coefficient(
    probabilities: np.ndarray,
    scores: np.ndarray,
    local: np.ndarray,
) -> float:
    """Return ``lim_M M E[||g_M-g||^2/K]`` for sample covariance."""

    mean_score = np.sum(probabilities[:, None] * scores, axis=0)
    energy = np.sum(probabilities * local)
    influence = 2.0 * np.real(
        np.conj(scores - mean_score[None, :])
        * (local - energy)[:, None]
    )
    force = np.sum(probabilities[:, None] * influence, axis=0)
    variances = np.sum(
        probabilities[:, None] * (influence - force[None, :]) ** 2,
        axis=0,
    )
    return float(np.mean(variances))


def singlet_force_efficiency_benchmark(
    bonds: Iterable[SpinBond],
    *,
    overlaps: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99, 1.0),
    sample_counts: tuple[int, ...] = (64, 256, 1024),
    replicates: int = 256,
    tangent_count: int = 8,
    seed: int = 20260916,
) -> dict[str, object]:
    """Compare full-``H`` and diagonal-force MSE on one ``N=8`` graph."""

    if isinstance(replicates, bool) or not isinstance(replicates, Integral) or replicates < 2:
        raise ValueError("replicates must be an integer at least two")
    if isinstance(tangent_count, bool) or not isinstance(tangent_count, Integral) or tangent_count < 1:
        raise ValueError("tangent_count must be a positive integer")
    if any(count < 2 for count in sample_counts):
        raise ValueError("every sample count must be at least two")
    if any(not 0.0 <= value <= 1.0 for value in overlaps):
        raise ValueError("overlaps must lie in [0,1]")
    checked_bonds = tuple(bonds)
    n_sites = 8
    hamiltonian, basis = heisenberg_sz_matrix(
        n_sites, checked_bonds, n_up=n_sites // 2
    )
    singlets, singlet_basis = singlet_isometry_in_sz(n_sites)
    if basis != singlet_basis:
        raise RuntimeError("Hamiltonian and singlet basis orders differ")
    reduced = singlets.conj().T @ hamiltonian @ singlets
    eigenvalues, eigenvectors = np.linalg.eigh(reduced)
    ground_coefficients = eigenvectors[:, 0]
    ground_state = singlets @ ground_coefficients
    ground_energy = float(np.real(eigenvalues[0]))
    diagonal = 3.0 * heisenberg_z_diagonal(n_sites, checked_bonds, basis=basis)

    rng = np.random.default_rng(int(seed))
    random_coefficients = rng.normal(size=singlets.shape[1]).astype(np.complex128)
    random_coefficients -= ground_coefficients * np.vdot(
        ground_coefficients, random_coefficients
    )
    random_coefficients /= np.linalg.norm(random_coefficients)
    tangent_coefficients = rng.normal(size=(singlets.shape[1], int(tangent_count)))
    tangent_coefficients, _ = np.linalg.qr(tangent_coefficients)
    tangents = singlets @ tangent_coefficients[:, : int(tangent_count)]

    rows: list[SingletForceEfficiencyRow] = []
    exact_force_maximum_difference = 0.0
    for overlap in overlaps:
        coefficients = (
            np.sqrt(float(overlap)) * ground_coefficients
            + np.sqrt(1.0 - float(overlap)) * random_coefficients
        )
        state = singlets @ coefficients
        probabilities = _checked_probabilities(state)
        action = hamiltonian @ state
        positive = probabilities > 1.0e-28
        local_full = np.zeros_like(state, dtype=np.complex128)
        local_full[positive] = action[positive] / state[positive]
        local_diagonal = diagonal.astype(np.complex128)
        scores = np.zeros((len(state), int(tangent_count)), dtype=np.complex128)
        scores[positive] = tangents[positive] / state[positive, None]
        exact_full = _centered_force(probabilities, scores, local_full)
        exact_diagonal = _centered_force(probabilities, scores, local_diagonal)
        force_difference = float(np.max(np.abs(exact_full - exact_diagonal)))
        exact_force_maximum_difference = max(
            exact_force_maximum_difference, force_difference
        )
        if force_difference > 3.0e-12:
            raise RuntimeError("full and diagonal exact singlet forces disagree")
        energy = float(np.real(np.vdot(state, action) / np.vdot(state, state)))
        full_variance = float(
            np.sum(probabilities * np.abs(local_full - energy) ** 2)
        )
        diagonal_energy = float(np.sum(probabilities * local_diagonal).real)
        diagonal_variance = float(
            np.sum(probabilities * np.abs(local_diagonal - diagonal_energy) ** 2)
        )
        full_noise = _asymptotic_force_mse_coefficient(
            probabilities, scores, local_full
        )
        diagonal_noise = _asymptotic_force_mse_coefficient(
            probabilities, scores, local_diagonal
        )
        connected_counts = np.asarray(
            [
                len(heisenberg_spin_connectivity(_bits(value, n_sites), checked_bonds)[0])
                for value in basis
            ],
            dtype=np.float64,
        )
        # Full center-only VMC: one Hpsi connected-amplitude pass plus one
        # differentiated center pass. Diagonal VMC: one stopped center pass
        # plus one differentiated center pass.
        full_work = float(np.sum(probabilities * connected_counts) + 1.0)
        diagonal_work = 2.0
        asymptotic_ratio = (
            None if full_noise <= 1.0e-26 else diagonal_noise / full_noise
        )
        asymptotic_adjusted = (
            None
            if asymptotic_ratio is None
            else asymptotic_ratio * diagonal_work / full_work
        )
        for sample_count in sample_counts:
            full_errors = []
            diagonal_errors = []
            for _ in range(int(replicates)):
                indices = rng.choice(
                    len(probabilities),
                    size=int(sample_count),
                    replace=True,
                    p=probabilities,
                )
                full_estimate = _sampled_force(indices, scores, local_full)
                diagonal_estimate = _sampled_force(indices, scores, local_diagonal)
                full_errors.append(np.mean((full_estimate - exact_full) ** 2))
                diagonal_errors.append(
                    np.mean((diagonal_estimate - exact_diagonal) ** 2)
                )
            full_mse = float(np.mean(full_errors))
            diagonal_mse = float(np.mean(diagonal_errors))
            # At an eigenstate the full estimator is exactly zero-variance;
            # floating-point eigensolver residuals must not be presented as a
            # finite, astronomically large ratio.
            ratio = None if full_mse <= 1.0e-28 else diagonal_mse / full_mse
            adjusted = (
                None
                if ratio is None
                else ratio * diagonal_work / full_work
            )
            rows.append(
                SingletForceEfficiencyRow(
                    ground_state_overlap=float(overlap),
                    energy=energy,
                    energy_error=energy - ground_energy,
                    full_local_energy_variance=full_variance,
                    diagonal_local_energy_variance=diagonal_variance,
                    sample_count=int(sample_count),
                    replicates=int(replicates),
                    tangent_count=int(tangent_count),
                    full_force_asymptotic_mse_coefficient=full_noise,
                    diagonal_force_asymptotic_mse_coefficient=diagonal_noise,
                    diagonal_over_full_asymptotic_mse=asymptotic_ratio,
                    diagonal_over_full_asymptotic_work_adjusted_mse=asymptotic_adjusted,
                    full_force_mse=full_mse,
                    diagonal_force_mse=diagonal_mse,
                    diagonal_over_full_force_mse=ratio,
                    full_amplitude_work_per_sample=full_work,
                    diagonal_amplitude_work_per_sample=diagonal_work,
                    diagonal_over_full_work_adjusted_mse=adjusted,
                )
            )
    return {
        "n_sites": n_sites,
        "sz_dimension": len(basis),
        "singlet_dimension": singlets.shape[1],
        "ground_energy": ground_energy,
        "exact_force_maximum_difference": exact_force_maximum_difference,
        "work_metric": "projected_amplitude_evaluation_proxy",
        "rows": [asdict(row) for row in rows],
    }


__all__ = ["SingletForceEfficiencyRow", "singlet_force_efficiency_benchmark"]
