"""Exact-small time-reversal-invariant projected paired-Z2 carriers.

The mean-field state contains both hopping and spin-singlet pairing.  Its
number- and Gutzwiller-projected amplitude is the determinant of the BdG pair
kernel on the up/down sites.  A caller must still apply the physical lattice
projector; no spatial symmetry is inferred from the gauge-fixed mean field.

The continuous IGG diagnostic solves the infinitesimal local SU(2) gauge
constraints on every nonzero link.  Nullity zero is the finite-graph
signature used here for a Z2, rather than U(1), ansatz.  This diagnostic does
not identify topological order after projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np

from ..lattices.triangular_hnf import (
    J1_POSITIVE,
    J2_POSITIVE,
    IntMatrix,
    hnf_sites,
    reduce_hnf,
    validate_hnf,
)


Geometry = Literal[
    "triangular_n12", "triangular_n16", "triangular_n36", "square_n16"
]


@dataclass(frozen=True)
class PairedZ2Kernel:
    """One fixed BdG pair kernel and its mean-field diagnostics."""

    label: str
    pairing: np.ndarray
    quasiparticle_gap: float
    u_condition_number: float
    pairing_symmetry_residual: float
    continuous_igg_nullity: int
    smallest_nonzero_igg_singular_value: float


def _checked_real_symmetric(matrix: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 2:
        raise ValueError(f"{name} must be a square matrix of size at least two")
    if np.any(~np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    scale = max(float(np.max(np.abs(value))), 1.0)
    if float(np.max(np.abs(value - value.T))) > 2.0e-13 * scale:
        raise ValueError(f"{name} must be real symmetric")
    return value


def continuous_igg_diagnostics(
    hopping: np.ndarray,
    pair_field: np.ndarray,
    *,
    link_tolerance: float = 1.0e-12,
    singular_tolerance: float = 1.0e-10,
) -> dict[str, object]:
    r"""Solve ``X_i u_ij - u_ij X_j = 0`` for local SU(2) generators.

    The real link convention is ``u_ij = chi_ij tau_3 + Delta_ij tau_1`` and
    ``X_i = i x_i^a tau_a``.  A zero-dimensional continuous solution space
    leaves only the discrete center ``{+I,-I}``, hence the label Z2.
    """

    hopping = _checked_real_symmetric(hopping, "hopping")
    pair_field = _checked_real_symmetric(pair_field, "pair_field")
    if pair_field.shape != hopping.shape:
        raise ValueError("hopping and pair_field must have the same shape")
    if not np.isfinite(link_tolerance) or link_tolerance <= 0.0:
        raise ValueError("link_tolerance must be positive and finite")
    if not np.isfinite(singular_tolerance) or singular_tolerance <= 0.0:
        raise ValueError("singular_tolerance must be positive and finite")

    tau = (
        np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.complex128),
        np.asarray(((0.0, -1.0j), (1.0j, 0.0)), dtype=np.complex128),
        np.asarray(((1.0, 0.0), (0.0, -1.0)), dtype=np.complex128),
    )
    n_sites = hopping.shape[0]
    links = [
        (first, second)
        for first in range(n_sites)
        for second in range(first + 1, n_sites)
        if max(abs(hopping[first, second]), abs(pair_field[first, second]))
        > link_tolerance
    ]
    if not links:
        raise ValueError("at least one nonzero mean-field link is required")
    columns = 3 * n_sites
    constraint = np.zeros((8 * len(links), columns), dtype=np.float64)
    for link_index, (first, second) in enumerate(links):
        link = (
            hopping[first, second] * tau[2]
            + pair_field[first, second] * tau[0]
        )
        for component, generator in enumerate(tau):
            left = 1.0j * generator @ link
            right = -link @ (1.0j * generator)
            row = 8 * link_index
            constraint[row : row + 4, 3 * first + component] = left.real.ravel()
            constraint[row + 4 : row + 8, 3 * first + component] = left.imag.ravel()
            constraint[row : row + 4, 3 * second + component] = right.real.ravel()
            constraint[row + 4 : row + 8, 3 * second + component] = right.imag.ravel()
    singular_values = np.linalg.svd(constraint, compute_uv=False)
    scale = max(float(singular_values[0]), 1.0)
    rank = int(np.sum(singular_values > singular_tolerance * scale))
    nullity = columns - rank
    nonzero = singular_values[singular_values > singular_tolerance * scale]
    return {
        "continuous_nullity": int(nullity),
        "constraint_rank": rank,
        "variable_count": columns,
        "link_count": len(links),
        "smallest_nonzero_singular_value": (
            float(nonzero[-1]) if len(nonzero) else 0.0
        ),
        "largest_singular_value": float(singular_values[0]),
    }


def singlet_bdg_pair_kernel(
    hopping: np.ndarray,
    pair_field: np.ndarray,
    *,
    chemical_potential: float = 0.0,
    gap_tolerance: float = 1.0e-10,
    condition_limit: float = 1.0e12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the real symmetric number-projected pair kernel ``V U^-1``."""

    hopping = _checked_real_symmetric(hopping, "hopping")
    pair_field = _checked_real_symmetric(pair_field, "pair_field")
    if pair_field.shape != hopping.shape:
        raise ValueError("hopping and pair_field must have the same shape")
    if not np.isfinite(chemical_potential):
        raise ValueError("chemical_potential must be finite")
    if not np.isfinite(gap_tolerance) or gap_tolerance <= 0.0:
        raise ValueError("gap_tolerance must be positive and finite")
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("condition_limit must be finite and greater than one")

    n_sites = hopping.shape[0]
    normal = hopping - float(chemical_potential) * np.eye(n_sites)
    bdg = np.block([[normal, pair_field], [pair_field, -normal]])
    eigenvalues, eigenvectors = np.linalg.eigh(bdg)
    gap = float(np.min(np.abs(eigenvalues)))
    if gap <= gap_tolerance:
        raise ValueError(
            f"BdG spectrum is gapless: gap={gap:.6e} <= {gap_tolerance:.6e}"
        )
    positive = np.flatnonzero(eigenvalues > 0.0)
    if len(positive) != n_sites:
        raise RuntimeError("a gapped particle-hole BdG spectrum must have N positive modes")
    modes = eigenvectors[:, positive]
    u = modes[:n_sites]
    v = modes[n_sites:]
    condition = float(np.linalg.cond(u))
    if not np.isfinite(condition) or condition >= condition_limit:
        raise ValueError(
            f"BdG U block is ill-conditioned: cond={condition:.6e}"
        )
    raw = np.linalg.solve(u.T, v.T).T
    scale = max(float(np.max(np.abs(raw))), np.finfo(np.float64).tiny)
    residual = float(np.max(np.abs(raw - raw.T)) / scale)
    if residual > 2.0e-9:
        raise RuntimeError(
            f"BdG singlet pair kernel is not symmetric: residual={residual:.6e}"
        )
    pairing = 0.5 * (raw + raw.T)
    # Onsite pairs are removed by exact Gutzwiller projection and never enter
    # an up/down determinant with disjoint site sets.
    np.fill_diagonal(pairing, 0.0)
    pairing /= max(float(np.linalg.norm(pairing)), np.finfo(np.float64).tiny)
    return pairing, {
        "quasiparticle_gap": gap,
        "u_condition_number": condition,
        "pairing_symmetry_residual": residual,
    }


def _fermi_midpoint(hopping: np.ndarray) -> float:
    values = np.linalg.eigvalsh(hopping)
    middle = len(values) // 2
    return float(0.5 * (values[middle - 1] + values[middle]))


def _add_symmetric_edge(matrix: np.ndarray, first: int, second: int, value: float) -> None:
    if first == second:
        raise ValueError("mean-field bonds may not alias to onsite terms")
    matrix[first, second] += float(value)
    matrix[second, first] += float(value)


def _triangular_mean_fields(
    supercell: IntMatrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = validate_hnf(supercell)
    sites = hnf_sites(matrix)
    positions = {site: index for index, site in enumerate(sites)}
    n_sites = len(sites)
    hopping = np.zeros((n_sites, n_sites), dtype=np.float64)
    j2_pair = np.zeros_like(hopping)
    e2_pair = np.zeros_like(hopping)
    e2_weights = (2.0, -1.0, -1.0)
    for source in sites:
        first = positions[source]
        for direction, displacement in enumerate(J1_POSITIVE):
            target = reduce_hnf(
                (source[0] + displacement[0], source[1] + displacement[1]),
                matrix,
            )
            gauge = 1.0 if direction == 0 else (-1.0) ** source[0]
            _add_symmetric_edge(hopping, first, positions[target], gauge)
            _add_symmetric_edge(
                e2_pair, first, positions[target], e2_weights[direction]
            )
        for displacement in J2_POSITIVE:
            target = reduce_hnf(
                (source[0] + displacement[0], source[1] + displacement[1]),
                matrix,
            )
            _add_symmetric_edge(j2_pair, first, positions[target], 1.0)
    return hopping, j2_pair, e2_pair


def _square_mean_fields(
    linear_size: int,
    *,
    antiperiodic_x: bool,
    antiperiodic_y: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        isinstance(linear_size, bool)
        or not isinstance(linear_size, Integral)
        or int(linear_size) < 4
        or int(linear_size) % 2
    ):
        raise ValueError("linear_size must be an even integer at least four")
    linear_size = int(linear_size)
    n_sites = linear_size**2
    hopping = np.zeros((n_sites, n_sites), dtype=np.float64)
    diagonal_pair = np.zeros_like(hopping)
    d_pair = np.zeros_like(hopping)

    def site(x: int, y: int) -> int:
        return (y % linear_size) * linear_size + (x % linear_size)

    def wrap_phase(x: int, y: int, dx: int, dy: int) -> float:
        phase = 1.0
        if antiperiodic_x and (x + dx < 0 or x + dx >= linear_size):
            phase *= -1.0
        if antiperiodic_y and (y + dy < 0 or y + dy >= linear_size):
            phase *= -1.0
        return phase

    for y in range(linear_size):
        for x in range(linear_size):
            first = site(x, y)
            x_phase = wrap_phase(x, y, 1, 0)
            y_phase = wrap_phase(x, y, 0, 1)
            _add_symmetric_edge(hopping, first, site(x + 1, y), x_phase)
            _add_symmetric_edge(
                hopping, first, site(x, y + 1), ((-1.0) ** x) * y_phase
            )
            _add_symmetric_edge(d_pair, first, site(x + 1, y), x_phase)
            _add_symmetric_edge(d_pair, first, site(x, y + 1), -y_phase)
            _add_symmetric_edge(
                diagonal_pair,
                first,
                site(x + 1, y + 1),
                wrap_phase(x, y, 1, 1),
            )
            _add_symmetric_edge(
                diagonal_pair,
                first,
                site(x + 1, y - 1),
                wrap_phase(x, y, 1, -1),
            )
    return hopping, diagonal_pair, d_pair


def _build_kernel(
    label: str,
    hopping: np.ndarray,
    pair_field: np.ndarray,
) -> PairedZ2Kernel:
    igg = continuous_igg_diagnostics(hopping, pair_field)
    if igg["continuous_nullity"] != 0:
        raise RuntimeError(
            f"{label} retains a continuous IGG of dimension "
            f"{igg['continuous_nullity']}"
        )
    pairing, bdg = singlet_bdg_pair_kernel(
        hopping,
        pair_field,
        chemical_potential=_fermi_midpoint(hopping),
    )
    return PairedZ2Kernel(
        label=label,
        pairing=pairing,
        quasiparticle_gap=bdg["quasiparticle_gap"],
        u_condition_number=bdg["u_condition_number"],
        pairing_symmetry_residual=bdg["pairing_symmetry_residual"],
        continuous_igg_nullity=int(igg["continuous_nullity"]),
        smallest_nonzero_igg_singular_value=float(
            igg["smallest_nonzero_singular_value"]
        ),
    )


def paired_z2_kernel_bank(geometry: Geometry) -> tuple[PairedZ2Kernel, ...]:
    """Return a fixed three-member, Hamiltonian-independent paired bank."""

    if geometry in {"triangular_n12", "triangular_n16", "triangular_n36"}:
        supercell = {
            "triangular_n12": ((6, 4), (0, 2)),
            "triangular_n16": ((4, 0), (0, 4)),
            "triangular_n36": ((6, 0), (0, 6)),
        }[geometry]
        hopping, j2_pair, e2_pair = _triangular_mean_fields(supercell)
        specifications = (
            ("z2_j2_delta_035", 0.35 * j2_pair),
            ("z2_j2_delta_070", 0.70 * j2_pair),
            ("z2_j2_e2_mixed", 0.50 * j2_pair + 0.20 * e2_pair),
        )
    elif geometry == "square_n16":
        specifications = []
        for label, boundary_x, boundary_y, amplitude in (
            ("z2_AP_delta_035", True, False, 0.35),
            ("z2_AP_delta_050", True, False, 0.50),
            ("z2_AA_delta_050", True, True, 0.50),
        ):
            hopping, diagonal_pair, d_pair = _square_mean_fields(
                4,
                antiperiodic_x=boundary_x,
                antiperiodic_y=boundary_y,
            )
            specifications.append(
                (label, hopping, amplitude * (diagonal_pair + 0.20 * d_pair))
            )
    else:
        raise ValueError(f"unknown paired-Z2 geometry: {geometry!r}")
    if geometry in {"triangular_n12", "triangular_n16", "triangular_n36"}:
        return tuple(
            _build_kernel(label, hopping, pair_field)
            for label, pair_field in specifications
        )
    return tuple(
        _build_kernel(label, candidate_hopping, pair_field)
        for label, candidate_hopping, pair_field in specifications
    )


__all__ = [
    "Geometry",
    "PairedZ2Kernel",
    "continuous_igg_diagnostics",
    "paired_z2_kernel_bank",
    "singlet_bdg_pair_kernel",
]
