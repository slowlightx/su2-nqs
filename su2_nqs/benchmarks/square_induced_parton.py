"""Compact exact-``A1`` projected Fermi sea on an even square torus.

The half-filled nearest-neighbour hopping problem is open shell for periodic
and doubly antiperiodic boundaries on the even tori used here.  The two
mixed-boundary sectors ``AP`` and ``PA`` are closed shell and form a point-
group orbit.  Their parity-adjusted superposition is the trivial square-
lattice irrep.  A quarter turn exchanges the two determinants with the
projected-fermion phase ``(-1)**(L/2)``.

The compact two-determinant formula is checked against the full translation
by D4 Reynolds projector in exact-small tests.  It is a carrier construction,
not a claim about the phase of the square-lattice J1-J2 model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from su2_nqs.benchmarks.projected_singlet_agp import square_gamma_a1_group
from su2_nqs.benchmarks.triangular_induced_parton import (
    InducedCarrierBatch,
    closed_shell_pairing_from_hopping,
)
from su2_nqs.models.projected_singlet_agp import (
    ProjectedAGPBatch,
    ProjectedSingletAGPConfig,
    projected_singlet_agp_amplitudes,
)


Boundary = Literal["P", "A"]
BOUNDARY_ORBIT: tuple[tuple[Boundary, Boundary], ...] = (
    ("A", "P"),
    ("P", "A"),
)


def square_a1_coefficients(linear_size: int) -> np.ndarray:
    """Return normalized AP/PA coefficients in the physical spin basis."""

    if linear_size < 2 or linear_size % 2:
        raise ValueError("linear_size must be an even integer at least two")
    return np.asarray((1.0, (-1.0) ** (linear_size // 2)), dtype=np.float64) / 2.0


@dataclass(frozen=True)
class SquareFermiSeaOrbit:
    """Pair kernels for the two mixed-boundary closed shells."""

    linear_size: int
    pairings: np.ndarray
    one_body_gaps: np.ndarray
    excluded_pp_gap: float
    excluded_aa_gap: float


def _validate_boundary(boundary: str) -> Boundary:
    if boundary not in {"P", "A"}:
        raise ValueError("boundary must be 'P' or 'A'")
    return boundary  # type: ignore[return-value]


def _site_index(x: int, y: int, linear_size: int) -> int:
    return (y % linear_size) * linear_size + (x % linear_size)


def square_nearest_neighbor_hopping(
    linear_size: int,
    *,
    boundary_x: Boundary = "P",
    boundary_y: Boundary = "P",
) -> np.ndarray:
    """Return the real nearest-neighbour square-lattice hopping matrix."""

    if linear_size < 2 or linear_size % 2:
        raise ValueError("linear_size must be an even integer at least two")
    boundary_x = _validate_boundary(boundary_x)
    boundary_y = _validate_boundary(boundary_y)
    n_sites = linear_size**2
    hopping = np.zeros((n_sites, n_sites), dtype=np.float64)
    for y in range(linear_size):
        for x in range(linear_size):
            source = _site_index(x, y, linear_size)
            for dx, dy in ((1, 0), (0, 1)):
                target = _site_index(x + dx, y + dy, linear_size)
                phase = 1.0
                if boundary_x == "A" and x + dx >= linear_size:
                    phase *= -1.0
                if boundary_y == "A" and y + dy >= linear_size:
                    phase *= -1.0
                hopping[source, target] = phase
                hopping[target, source] = phase
    return hopping


def build_square_fermi_sea_orbit(
    linear_size: int, *, gap_tolerance: float = 1.0e-10
) -> SquareFermiSeaOrbit:
    """Build and validate the two mixed-boundary closed shells."""

    pairings = []
    gaps = []
    for boundary_x, boundary_y in BOUNDARY_ORBIT:
        packed, gap = closed_shell_pairing_from_hopping(
            square_nearest_neighbor_hopping(
                linear_size,
                boundary_x=boundary_x,
                boundary_y=boundary_y,
            ),
            gap_tolerance=gap_tolerance,
            require_closed_shell=True,
        )
        pairings.append(packed)
        gaps.append(gap)
    excluded = []
    for boundary_x, boundary_y in (("P", "P"), ("A", "A")):
        _, gap = closed_shell_pairing_from_hopping(
            square_nearest_neighbor_hopping(
                linear_size,
                boundary_x=boundary_x,
                boundary_y=boundary_y,
            ),
            gap_tolerance=gap_tolerance,
            require_closed_shell=False,
        )
        excluded.append(gap)
    if any(gap > gap_tolerance for gap in excluded):
        raise RuntimeError("a registered excluded square boundary became closed shell")
    return SquareFermiSeaOrbit(
        linear_size=int(linear_size),
        pairings=np.stack(pairings),
        one_body_gaps=np.asarray(gaps),
        excluded_pp_gap=float(excluded[0]),
        excluded_aa_gap=float(excluded[1]),
    )


def _group_tuple(group: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(site) for site in image) for image in np.asarray(group))


def projected_square_fermi_sea_a1_amplitudes(
    configurations: np.ndarray,
    *,
    orbit: SquareFermiSeaOrbit,
    microbatch_size: int = 64,
) -> InducedCarrierBatch:
    """Evaluate the compact two-determinant exact-singlet ``A1`` carrier."""

    states = np.asarray(configurations, dtype=np.int8)
    n_sites = orbit.linear_size**2
    if states.ndim != 2 or states.shape[1] != n_sites:
        raise ValueError(f"configurations must have shape (batch, {n_sites})")
    config = ProjectedSingletAGPConfig(
        n_sites=n_sites, group_images=(tuple(range(n_sites)),)
    )
    components = []
    for pairing in orbit.pairings:
        components.append(
            np.asarray(
                projected_singlet_agp_amplitudes(
                    {"pairing_real": pairing},
                    states,
                    config,
                    microbatch_size=microbatch_size,
                ).amplitudes
            )
        )
    component_array = np.stack(components, axis=1)
    weighted = component_array * square_a1_coefficients(orbit.linear_size)[None, :]
    amplitudes = np.sum(weighted, axis=1)
    denominator = np.sum(np.abs(weighted), axis=1)
    cancellation = np.divide(
        np.abs(amplitudes),
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0.0,
    )
    return InducedCarrierBatch(amplitudes, component_array, cancellation)


def full_reynolds_square_fermi_sea_reference(
    configurations: np.ndarray,
    *,
    orbit: SquareFermiSeaOrbit,
    boundary_index: int = 0,
    microbatch_size: int = 64,
) -> ProjectedAGPBatch:
    """Evaluate the full translation-by-D4 Reynolds projection oracle."""

    if boundary_index not in (0, 1):
        raise ValueError("boundary_index must be zero or one")
    states = np.asarray(configurations, dtype=np.int8)
    n_sites = orbit.linear_size**2
    if states.ndim != 2 or states.shape[1] != n_sites:
        raise ValueError(f"configurations must have shape (batch, {n_sites})")
    config = ProjectedSingletAGPConfig(
        n_sites=n_sites,
        group_images=square_gamma_a1_group(orbit.linear_size),
    )
    return projected_singlet_agp_amplitudes(
        {"pairing_real": orbit.pairings[boundary_index]},
        states,
        config,
        microbatch_size=microbatch_size,
    )


__all__ = [
    "BOUNDARY_ORBIT",
    "SquareFermiSeaOrbit",
    "build_square_fermi_sea_orbit",
    "full_reynolds_square_fermi_sea_reference",
    "projected_square_fermi_sea_a1_amplitudes",
    "square_a1_coefficients",
    "square_nearest_neighbor_hopping",
]
