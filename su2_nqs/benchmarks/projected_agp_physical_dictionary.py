"""Prospective local pairing seeds for triangular projected-AGP studies.

The dictionary encodes distinct *initial physical hypotheses* before any
N36 optimization: a real K-star/three-sublattice texture, a real J1 sign
pattern with pi loop flux, and the three M-star stripe textures.  These are
initialization features for a symmetry-restored AGP, not phase labels.  In
particular the pi-flux pattern is only DSL-inspired; a projected-AGP pairing
matrix is not by itself a certified spinon mean-field Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .projected_agp_local_pairing import packed_pair_index
from ..lattices.triangular_hnf import (
    J1_POSITIVE,
    IntMatrix,
    hnf_sites,
    reduce_hnf,
    validate_hnf,
)


@dataclass(frozen=True)
class PhysicalPairingSeed:
    """One packed real short-range pairing seed plus conservative metadata."""

    name: str
    packed_pairing: np.ndarray
    family: str
    interpretation: str


def _j1_edge_table(
    supercell: IntMatrix,
) -> tuple[
    tuple[tuple[int, int], ...],
    dict[tuple[int, int], int],
    list[tuple[tuple[int, int], int, int]],
]:
    matrix = validate_hnf(supercell)
    sites = hnf_sites(matrix)
    positions = {site: index for index, site in enumerate(sites)}
    packed = packed_pair_index(len(sites))
    table: list[tuple[tuple[int, int], int, int]] = []
    seen: set[int] = set()
    for direction, displacement in enumerate(J1_POSITIVE):
        for source in sites:
            target = reduce_hnf(
                (source[0] + displacement[0], source[1] + displacement[1]),
                matrix,
            )
            edge = tuple(sorted((positions[source], positions[target])))
            if edge[0] == edge[1] or packed[edge] in seen:
                raise ValueError("the torus aliases a positive J1 edge")
            seen.add(packed[edge])
            table.append((source, direction, packed[edge]))
    return sites, positions, table


def triangular_physical_pairing_seed_dictionary(
    supercell: IntMatrix,
) -> dict[str, PhysicalPairingSeed]:
    """Return N-independent-locality seeds spanning K, flux, and M stories.

    The K-star seeds require exact three-sublattice compatibility and the M
    seeds require even HNF entries.  The intended first complete geometry is
    the 6x6 full-D6 torus; N16 deliberately fails the K compatibility check.
    """

    matrix = validate_hnf(supercell)
    sites, _, table = _j1_edge_table(matrix)
    horizontal, tilt = matrix[0]
    vertical = matrix[1][1]
    if any(
        (column[0] + column[1]) % 3
        for column in ((horizontal, 0), (tilt, vertical))
    ):
        raise ValueError("the torus does not contain the triangular K star")
    if any(value % 2 for row in matrix for value in row):
        raise ValueError("the torus does not contain all three triangular M points")
    width = len(sites) * (len(sites) - 1) // 2

    def make(name: str, family: str, interpretation: str, value_function):
        packed = np.zeros(width, dtype=np.float64)
        for source, direction, index in table:
            packed[index] = float(value_function(source, direction))
        norm = float(np.linalg.norm(packed))
        if norm <= 0.0:
            raise RuntimeError(f"physical seed {name} vanished")
        return PhysicalPairingSeed(name, packed / norm, family, interpretation)

    seeds = [
        make(
            "three_sublattice_120_cos",
            "K_star_120",
            "Real +K/-K three-sublattice local-pairing texture; not a magnetic-order claim.",
            lambda source, direction: math.cos(
                2.0 * math.pi * ((source[0] + source[1]) % 3) / 3.0
            ),
        ),
        make(
            "three_sublattice_120_sin",
            "K_star_120",
            "Quadrature partner completing the real K-star initialization family.",
            lambda source, direction: math.sin(
                2.0 * math.pi * ((source[0] + source[1]) % 3) / 3.0
            ),
        ),
        make(
            "uniform_pi_flux_DSL_inspired",
            "pi_flux_DSL_inspired",
            "Real J1 sign texture with pi product around every elementary triangle; not a certified DSL mean field.",
            lambda source, direction: (1.0, 1.0, -1.0)[direction],
        ),
        make(
            "stripe_M1",
            "M_star_stripe",
            "M=(1/2,0) real local-pairing texture.",
            lambda source, direction: -1.0 if source[0] % 2 else 1.0,
        ),
        make(
            "stripe_M2",
            "M_star_stripe",
            "M=(0,1/2) real local-pairing texture.",
            lambda source, direction: -1.0 if source[1] % 2 else 1.0,
        ),
        make(
            "stripe_M3",
            "M_star_stripe",
            "M=(1/2,1/2) real local-pairing texture.",
            lambda source, direction: -1.0
            if (source[0] + source[1]) % 2
            else 1.0,
        ),
    ]
    return {seed.name: seed for seed in seeds}


def elementary_triangle_fluxes(
    packed_pairing: np.ndarray, supercell: IntMatrix
) -> np.ndarray:
    """Return signs of the two elementary J1 triangle orientations per site."""

    matrix = validate_hnf(supercell)
    sites = hnf_sites(matrix)
    positions = {site: index for index, site in enumerate(sites)}
    packed_index = packed_pair_index(len(sites))
    values = np.asarray(packed_pairing, dtype=np.float64)
    if values.shape != (len(packed_index),):
        raise ValueError("packed_pairing has the wrong width")

    def edge(first, second) -> float:
        pair = tuple(sorted((positions[first], positions[second])))
        return float(values[packed_index[pair]])

    fluxes = []
    for source in sites:
        a = reduce_hnf((source[0] + 1, source[1]), matrix)
        b = reduce_hnf((source[0], source[1] + 1), matrix)
        c = reduce_hnf((source[0] + 1, source[1] + 1), matrix)
        fluxes.append(np.sign(edge(source, a) * edge(a, c) * edge(c, source)))
        fluxes.append(np.sign(edge(source, b) * edge(b, c) * edge(c, source)))
    return np.asarray(fluxes, dtype=np.float64)


__all__ = [
    "PhysicalPairingSeed",
    "elementary_triangle_fluxes",
    "triangular_physical_pairing_seed_dictionary",
]
