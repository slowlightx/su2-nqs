"""SU(2)-resolved bipartite entanglement diagnostics for fixed-``M`` states.

The routines in this module are exact-small diagnostics, not a scalable
entropy estimator.  They embed a fixed-magnetization wavefunction into a
bipartite amplitude matrix, trace out the complement, and resolve the reduced
density matrix by the doubled subsystem spin ``q = 2 S_A``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _checked_state_and_basis(
    state: np.ndarray,
    basis: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(state, dtype=np.complex128)
    configurations = np.asarray(basis, dtype=np.int8)
    if values.ndim != 1:
        raise ValueError("state must be a one-dimensional coefficient vector")
    if configurations.ndim != 2 or configurations.shape[0] != values.size:
        raise ValueError("basis must have shape (state.size, n_sites)")
    if configurations.shape[1] < 2:
        raise ValueError("at least two sites are required")
    if np.any((configurations != 0) & (configurations != 1)):
        raise ValueError("basis entries must be computational bits 0 or 1")
    if len({tuple(row) for row in configurations.tolist()}) != values.size:
        raise ValueError("basis configurations must be unique")
    norm = float(np.vdot(values, values).real)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("state must have finite nonzero norm")
    return values / np.sqrt(norm), configurations


def _checked_subsystem(subsystem: Sequence[int], n_sites: int) -> tuple[int, ...]:
    sites = tuple(int(site) for site in subsystem)
    if not sites or len(sites) == n_sites:
        raise ValueError("subsystem must be nonempty and proper")
    if len(set(sites)) != len(sites):
        raise ValueError("subsystem sites must be unique")
    if any(site < 0 or site >= n_sites for site in sites):
        raise ValueError("subsystem site lies outside the lattice")
    return sites


def bipartite_amplitude_matrix(
    state: np.ndarray,
    basis: Sequence[Sequence[int]],
    subsystem: Sequence[int],
) -> np.ndarray:
    """Return ``Psi[a,b]`` in site-order-preserving computational bases.

    Within each subsystem, the first listed site is the most significant bit.
    Complement sites retain their original increasing lattice order.
    """

    values, configurations = _checked_state_and_basis(state, basis)
    sites_a = _checked_subsystem(subsystem, configurations.shape[1])
    set_a = set(sites_a)
    sites_b = tuple(site for site in range(configurations.shape[1]) if site not in set_a)
    powers_a = 2 ** np.arange(len(sites_a) - 1, -1, -1, dtype=np.int64)
    powers_b = 2 ** np.arange(len(sites_b) - 1, -1, -1, dtype=np.int64)
    rows = configurations[:, sites_a] @ powers_a
    columns = configurations[:, sites_b] @ powers_b
    matrix = np.zeros((2 ** len(sites_a), 2 ** len(sites_b)), dtype=np.complex128)
    matrix[rows, columns] = values
    return matrix


def reduced_density_matrix(
    state: np.ndarray,
    basis: Sequence[Sequence[int]],
    subsystem: Sequence[int],
) -> np.ndarray:
    """Trace out the complement of ``subsystem`` from a normalized state."""

    amplitudes = bipartite_amplitude_matrix(state, basis, subsystem)
    rho = amplitudes @ amplitudes.conj().T
    return 0.5 * (rho + rho.conj().T)


def subsystem_total_spin_squared(n_sites: int) -> np.ndarray:
    """Return dense ``S_A^2`` for ``n_sites`` spin-1/2 degrees of freedom."""

    if n_sites < 1:
        raise ValueError("n_sites must be positive")
    dimension = 2**n_sites
    operator = np.eye(dimension, dtype=np.float64) * (0.75 * n_sites)
    powers = 2 ** np.arange(n_sites - 1, -1, -1, dtype=np.int64)
    bits = ((np.arange(dimension)[:, None] // powers[None, :]) % 2).astype(np.int8)
    for first in range(n_sites):
        for second in range(first + 1, n_sites):
            same = bits[:, first] == bits[:, second]
            operator[np.arange(dimension), np.arange(dimension)] += np.where(
                same, 0.5, -0.5
            )
            different_rows = np.flatnonzero(~same)
            flipped = bits[different_rows].copy()
            flipped[:, [first, second]] = flipped[:, [second, first]]
            flipped_indices = flipped @ powers
            operator[flipped_indices, different_rows] += 1.0
    return operator


def _entropy(eigenvalues: np.ndarray) -> tuple[float, float, int]:
    positive = np.asarray(eigenvalues)[np.asarray(eigenvalues) > 1.0e-14]
    von_neumann = float(-np.sum(positive * np.log(positive)))
    purity = float(np.sum(np.asarray(eigenvalues) ** 2))
    renyi_2 = float(-np.log(max(purity, np.finfo(float).tiny)))
    return von_neumann, renyi_2, int(positive.size)


def su2_resolved_entanglement(
    state: np.ndarray,
    basis: Sequence[Sequence[int]],
    subsystem: Sequence[int],
    *,
    eigenvalue_cutoff: float = 1.0e-14,
) -> dict[str, object]:
    """Resolve a subsystem density matrix into doubled-spin ``q`` sectors.

    For a global singlet, ``rho_A`` commutes with ``S_A^2`` and each returned
    sector spectrum consists of the ordinary density-matrix eigenvalues in
    that spin sector, including magnetic degeneracy.
    """

    if eigenvalue_cutoff <= 0.0:
        raise ValueError("eigenvalue_cutoff must be positive")
    rho = reduced_density_matrix(state, basis, subsystem)
    spin_squared = subsystem_total_spin_squared(len(tuple(subsystem)))
    spin_eigenvalues, spin_eigenvectors = np.linalg.eigh(spin_squared)
    q_labels = np.rint(np.sqrt(np.maximum(0.0, 1.0 + 4.0 * spin_eigenvalues)) - 1.0).astype(int)
    reconstructed = 0.25 * q_labels * (q_labels + 2)
    if float(np.max(np.abs(reconstructed - spin_eigenvalues))) > 2.0e-10:
        raise RuntimeError("failed to assign integer doubled-spin labels")

    density_eigenvalues = np.linalg.eigvalsh(rho)
    density_eigenvalues = np.maximum(density_eigenvalues, 0.0)
    von_neumann, renyi_2, rank = _entropy(density_eigenvalues)
    commutator = rho @ spin_squared - spin_squared @ rho
    sectors: dict[str, object] = {}
    for q in sorted(set(q_labels.tolist())):
        vectors = spin_eigenvectors[:, q_labels == q]
        block = vectors.conj().T @ rho @ vectors
        block = 0.5 * (block + block.conj().T)
        spectrum = np.maximum(np.linalg.eigvalsh(block), 0.0)
        positive = spectrum[spectrum > eigenvalue_cutoff]
        weight = float(np.sum(spectrum))
        sector_entropy = float(-np.sum(positive * np.log(positive)))
        sectors[str(q)] = {
            "q": int(q),
            "spin": 0.5 * float(q),
            "hilbert_dimension": int(vectors.shape[1]),
            "weight": weight,
            "entropy_contribution": sector_entropy,
            "nonzero_rank": int(positive.size),
            "spectrum": spectrum[::-1].tolist(),
        }
    return {
        "subsystem": [int(site) for site in subsystem],
        "subsystem_size": len(tuple(subsystem)),
        "von_neumann_entropy": von_neumann,
        "renyi_2_entropy": renyi_2,
        "purity": float(np.trace(rho @ rho).real),
        "nonzero_rank": rank,
        "commutator_frobenius": float(np.linalg.norm(commutator)),
        "sectors": sectors,
        "rho": rho,
    }


def singlet_entropy_components(report: dict[str, object]) -> dict[str, float]:
    """Separate spin-weight, magnetic and multiplicity entropy (natural logs).

    For an SU(2)-invariant density matrix, Schur's lemma gives
    ``rho = direct_sum_q p_q I_(q+1)/(q+1) tensor sigma_q``. Check invariance
    under S+ explicitly: commuting with S^2 alone would be insufficient.
    This is an exact-small diagnostic, not an entropy estimator for VMC.
    """
    rho = np.asarray(report["rho"])
    n_sites = int(report["subsystem_size"])
    raising = np.zeros_like(rho)
    for packed in range(2**n_sites):
        for site in range(n_sites):
            mask = 1 << site
            if not packed & mask:
                raising[packed | mask, packed] += 1.0
    residual = float(np.linalg.norm(rho @ raising - raising @ rho))
    if residual > 1e-9:
        raise ValueError("entropy decomposition requires an SU(2)-invariant rho")
    classical = magnetic = multiplicity = 0.0
    for sector in report["sectors"].values():
        q, weight = int(sector["q"]), float(sector["weight"])
        if weight > 1e-14:
            classical -= weight * np.log(weight)
            magnetic += weight * np.log(q + 1)
            multiplicity += float(sector["entropy_contribution"]) + weight * np.log(weight) - weight * np.log(q + 1)
    return {
        "spin_weight_entropy": float(classical),
        "magnetic_entropy": float(magnetic),
        "multiplicity_entropy": float(multiplicity),
        "sum_error": float(abs(classical + magnetic + multiplicity - float(report["von_neumann_entropy"]))),
        "raising_commutator_norm": residual,
    }


def compare_entanglement_reports(
    candidate: dict[str, object],
    reference: dict[str, object],
) -> dict[str, float]:
    """Compare two reports for the same cut by trace and sector distances."""

    candidate_rho = np.asarray(candidate["rho"], dtype=np.complex128)
    reference_rho = np.asarray(reference["rho"], dtype=np.complex128)
    if candidate_rho.shape != reference_rho.shape:
        raise ValueError("reports must describe the same subsystem dimension")
    delta = 0.5 * ((candidate_rho - reference_rho) + (candidate_rho - reference_rho).conj().T)
    trace_distance = 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(delta))))
    candidate_sectors = candidate["sectors"]
    reference_sectors = reference["sectors"]
    if not isinstance(candidate_sectors, dict) or not isinstance(reference_sectors, dict):
        raise ValueError("reports contain invalid sector payloads")
    labels = set(candidate_sectors) | set(reference_sectors)
    sector_l1 = sum(
        abs(
            float(candidate_sectors.get(q, {"weight": 0.0})["weight"])
            - float(reference_sectors.get(q, {"weight": 0.0})["weight"])
        )
        for q in labels
    )
    spectrum_l1 = 0.0
    for q in labels:
        candidate_spectrum = np.asarray(
            candidate_sectors.get(q, {"spectrum": []})["spectrum"], dtype=float
        )
        reference_spectrum = np.asarray(
            reference_sectors.get(q, {"spectrum": []})["spectrum"], dtype=float
        )
        size = max(candidate_spectrum.size, reference_spectrum.size)
        candidate_padded = np.pad(
            candidate_spectrum, (0, size - candidate_spectrum.size)
        )
        reference_padded = np.pad(
            reference_spectrum, (0, size - reference_spectrum.size)
        )
        spectrum_l1 += float(np.sum(np.abs(candidate_padded - reference_padded)))
    return {
        "trace_distance": trace_distance,
        "sector_weight_l1": float(sector_l1),
        "sector_weight_total_variation": 0.5 * float(sector_l1),
        "resolved_spectrum_total_variation": 0.5 * float(spectrum_l1),
        "von_neumann_error": abs(
            float(candidate["von_neumann_entropy"])
            - float(reference["von_neumann_entropy"])
        ),
        "renyi_2_error": abs(
            float(candidate["renyi_2_entropy"])
            - float(reference["renyi_2_entropy"])
        ),
    }


def json_ready_entanglement_report(report: dict[str, object]) -> dict[str, object]:
    """Drop the dense reduced density matrix from a report for serialization."""

    return {key: value for key, value in report.items() if key != "rho"}


__all__ = [
    "bipartite_amplitude_matrix",
    "compare_entanglement_reports",
    "json_ready_entanglement_report",
    "reduced_density_matrix",
    "su2_resolved_entanglement",
    "subsystem_total_spin_squared",
]
