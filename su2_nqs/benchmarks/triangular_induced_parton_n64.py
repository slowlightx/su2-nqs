"""Certified three-boundary projected pi/0-flux carrier on triangular 8x8.

For linear size eight the periodic-periodic one-body sector is open shell,
whereas ``AP``, ``PA``, and ``AA`` are gapped and close on a three-dimensional
space-group orbit.  Translations act trivially; C6 cyclically permutes the
three components and the mirror swaps ``AP`` with ``AA``.  The unique A1
combination is therefore their equal-weight mean.

This module is a size-transfer bridge, not a large-system energy result.  Its
compressed carrier is independently checked against the full 768-image
Reynolds projector in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from ..models.projected_singlet_agp import (
    ProjectedSingletAGPConfig,
    projected_singlet_agp_amplitudes,
)
from .triangular_induced_parton import (
    InducedCarrierBatch,
    closed_shell_pairing_from_hopping,
    triangular_pi_zero_flux_hopping,
)


LINEAR_SIZE = 8
N_SITES = 64
BOUNDARY_ORBIT = (("A", "P"), ("P", "A"), ("A", "A"))
A1_COEFFICIENTS = np.full((3,), 1.0 / 3.0, dtype=np.float64)
TX_REPRESENTATION = np.eye(3, dtype=np.float64)
TY_REPRESENTATION = np.eye(3, dtype=np.float64)
C6_REPRESENTATION = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
MIRROR_REPRESENTATION = np.asarray(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PiZeroFluxN64Orbit:
    """Three gapped boundary pair kernels and the excluded open shell."""

    pairings: np.ndarray
    one_body_gaps: np.ndarray
    excluded_pp_gap: float


def build_n64_pi_zero_flux_orbit(
    *, gap_tolerance: float = 1.0e-10
) -> PiZeroFluxN64Orbit:
    """Build the gapped ``AP/PA/AA`` orbit and audit excluded ``PP``."""

    pairings = []
    gaps = []
    for boundary_x, boundary_y in BOUNDARY_ORBIT:
        hopping = triangular_pi_zero_flux_hopping(
            LINEAR_SIZE,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
        )
        pairing, gap = closed_shell_pairing_from_hopping(
            hopping,
            gap_tolerance=gap_tolerance,
            require_closed_shell=True,
        )
        pairings.append(pairing)
        gaps.append(gap)
    pp = triangular_pi_zero_flux_hopping(
        LINEAR_SIZE, boundary_x="P", boundary_y="P"
    )
    _, pp_gap = closed_shell_pairing_from_hopping(
        pp,
        gap_tolerance=gap_tolerance,
        require_closed_shell=False,
    )
    if pp_gap > gap_tolerance:
        raise RuntimeError("PP unexpectedly became a closed-shell sector")
    return PiZeroFluxN64Orbit(
        pairings=np.stack(pairings),
        one_body_gaps=np.asarray(gaps),
        excluded_pp_gap=pp_gap,
    )


def _identity_config() -> ProjectedSingletAGPConfig:
    return ProjectedSingletAGPConfig(
        n_sites=N_SITES,
        group_images=(tuple(range(N_SITES)),),
    )


def projected_n64_pi_zero_flux_a1_amplitudes(
    configurations: np.ndarray,
    *,
    orbit: PiZeroFluxN64Orbit | None = None,
    microbatch_size: int = 64,
) -> InducedCarrierBatch:
    """Evaluate the compressed exact ``S=0, Gamma-A1`` N64 carrier."""

    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[1] != N_SITES:
        raise ValueError("configurations must have shape (batch, 64)")
    if orbit is None:
        orbit = build_n64_pi_zero_flux_orbit()
    components = []
    config = _identity_config()
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
    weighted = component_array * A1_COEFFICIENTS[None, :]
    amplitudes = np.sum(weighted, axis=1)
    denominator = np.sum(np.abs(weighted), axis=1)
    cancellation = np.divide(
        np.abs(amplitudes),
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0.0,
    )
    return InducedCarrierBatch(
        amplitudes=np.asarray(amplitudes),
        components=component_array,
        cancellation_ratio=np.asarray(cancellation),
    )


def full_reynolds_n64_pi_zero_flux_reference(
    configurations: np.ndarray,
    *,
    boundary_index: int = 0,
    orbit: PiZeroFluxN64Orbit | None = None,
    microbatch_size: int = 64,
):
    """Apply the full 768-image A1 projector as an independent oracle."""

    if boundary_index not in range(3):
        raise ValueError("boundary_index must be 0, 1, or 2")
    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[1] != N_SITES:
        raise ValueError("configurations must have shape (batch, 64)")
    if orbit is None:
        orbit = build_n64_pi_zero_flux_orbit()
    group = triangular_hnf_gamma_a1_group(
        ((LINEAR_SIZE, 0), (0, LINEAR_SIZE))
    )
    config = ProjectedSingletAGPConfig(
        n_sites=N_SITES,
        group_images=tuple(tuple(int(site) for site in image) for image in group),
    )
    return projected_singlet_agp_amplitudes(
        {"pairing_real": orbit.pairings[boundary_index]},
        states,
        config,
        microbatch_size=microbatch_size,
    )


__all__ = [
    "A1_COEFFICIENTS",
    "BOUNDARY_ORBIT",
    "C6_REPRESENTATION",
    "MIRROR_REPRESENTATION",
    "PiZeroFluxN64Orbit",
    "TX_REPRESENTATION",
    "TY_REPRESENTATION",
    "build_n64_pi_zero_flux_orbit",
    "full_reynolds_n64_pi_zero_flux_reference",
    "projected_n64_pi_zero_flux_a1_amplitudes",
]
