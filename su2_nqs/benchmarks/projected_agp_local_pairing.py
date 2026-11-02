"""Pure exact-small maps for physically local projected-AGP pairings.

These maps restrict the packed symmetric pairing used by the strict AGP
carrier.  They contain no Hamiltonian or target-state information.  Their
purpose is to expose whether an apparent capacity result comes from the
quadratic unrestricted parameter count rather than a local physical ansatz.

The two families here are deliberately only N16 mechanism screens.  A future
N36 carrier must separately represent 120-degree/three-sublattice, flux/DSL,
and stripe initial families; these maps do not establish that capability.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from ..lattices.triangular_hnf import (
    J1_POSITIVE,
    J2_POSITIVE,
    IntMatrix,
    hnf_sites,
    reduce_hnf,
    validate_hnf,
)


LocalPairingFamily = Literal[
    "translation_coherent_J1_J2_orientation",
    "J1_two_by_two_texture",
]


def square_local_pairing_parameter_map(linear_size: int) -> np.ndarray:
    """Return the eight-parameter square-J1 two-by-two texture map.

    Each of the two positive nearest-neighbour orientations carries one
    coefficient for every ``(row mod 2, column mod 2)`` source class.  The
    parameter count stays eight for every compatible even torus.
    """

    if (
        isinstance(linear_size, bool)
        or not isinstance(linear_size, int)
        or linear_size < 4
        or linear_size % 2
    ):
        raise ValueError("linear_size must be an even integer at least four")
    n_sites = linear_size**2
    packed = packed_pair_index(n_sites)
    result = np.zeros((8, len(packed)), dtype=np.float64)
    seen: set[int] = set()
    for row in range(linear_size):
        for column in range(linear_size):
            source = row * linear_size + column
            parity = (column % 2) + 2 * (row % 2)
            targets = (
                row * linear_size + (column + 1) % linear_size,
                ((row + 1) % linear_size) * linear_size + column,
            )
            for direction, target in enumerate(targets):
                edge = tuple(sorted((source, target)))
                index = packed[edge]
                if index in seen:
                    raise ValueError("the square torus aliases a positive J1 edge")
                seen.add(index)
                result[4 * direction + parity, index] = 1.0
    return result


def packed_pair_index(n_sites: int) -> dict[tuple[int, int], int]:
    """Map each strict-upper-triangle site pair to packed position."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, int) or n_sites < 2:
        raise ValueError("n_sites must be an integer at least two")
    first, second = np.triu_indices(n_sites, k=1)
    return {
        (int(left), int(right)): index
        for index, (left, right) in enumerate(zip(first, second, strict=True))
    }


def triangular_local_pairing_parameter_map(
    supercell: IntMatrix,
    family: LocalPairingFamily,
) -> np.ndarray:
    """Return ``local_parameter -> packed_pairing`` for one short-range family.

    ``translation_coherent_J1_J2_orientation`` has six N-independent real
    parameters, one for each positive orientation in the J1 and J2 shells.
    ``J1_two_by_two_texture`` has twelve N-independent parameters, one for
    each J1 orientation and source-coordinate parity class.  The latter
    requires both torus periods to be even so the texture closes exactly.
    """

    matrix = validate_hnf(supercell)
    sites = hnf_sites(matrix)
    positions = {site: index for index, site in enumerate(sites)}
    packed = packed_pair_index(len(sites))

    def edge_index(source: tuple[int, int], displacement: tuple[int, int]) -> int:
        target = reduce_hnf(
            (source[0] + displacement[0], source[1] + displacement[1]), matrix
        )
        first, second = sorted((positions[source], positions[target]))
        if first == second:
            raise ValueError("a requested local displacement aliases to a self-edge")
        return packed[(first, second)]

    if family == "translation_coherent_J1_J2_orientation":
        result = np.zeros((6, len(packed)), dtype=np.float64)
        seen: set[int] = set()
        for shell, displacements in enumerate((J1_POSITIVE, J2_POSITIVE)):
            for direction, displacement in enumerate(displacements):
                parameter = 3 * shell + direction
                for source in sites:
                    index = edge_index(source, displacement)
                    if index in seen:
                        raise ValueError(
                            "the torus aliases local shell/orientation edges"
                        )
                    seen.add(index)
                    result[parameter, index] = 1.0
        return result

    if family == "J1_two_by_two_texture":
        horizontal, tilt = matrix[0]
        vertical = matrix[1][1]
        if horizontal % 2 or vertical % 2 or tilt % 2:
            raise ValueError("the two-by-two texture does not close on this HNF torus")
        result = np.zeros((12, len(packed)), dtype=np.float64)
        seen = set()
        for direction, displacement in enumerate(J1_POSITIVE):
            for source in sites:
                parity = (source[0] % 2) + 2 * (source[1] % 2)
                parameter = 4 * direction + parity
                index = edge_index(source, displacement)
                if index in seen:
                    raise ValueError("the torus aliases J1 texture edges")
                seen.add(index)
                result[parameter, index] = 1.0
        return result

    raise ValueError(f"unknown local pairing family: {family!r}")


def expand_packed_pairing(
    local_parameters: Sequence[float], parameter_map: np.ndarray
) -> np.ndarray:
    """Expand local parameters into one real symmetric zero-diagonal matrix."""

    mapping = np.asarray(parameter_map, dtype=np.float64)
    values = np.asarray(local_parameters, dtype=np.float64)
    if mapping.ndim != 2 or values.shape != (mapping.shape[0],):
        raise ValueError("local parameters and parameter map have incompatible shapes")
    pair_count = mapping.shape[1]
    discriminant = 1 + 8 * pair_count
    n_sites = int((1 + np.sqrt(discriminant)) / 2)
    if n_sites * (n_sites - 1) // 2 != pair_count:
        raise ValueError("parameter-map width is not a packed strict triangle")
    packed = values @ mapping
    first, second = np.triu_indices(n_sites, k=1)
    pairing = np.zeros((n_sites, n_sites), dtype=np.float64)
    pairing[first, second] = packed
    pairing[second, first] = packed
    return pairing


def permute_packed_pairing_parameter_map(
    parameter_map: np.ndarray, site_permutation: Sequence[int]
) -> np.ndarray:
    """Relabel every supported edge while preserving map degree and row norms.

    A fixed non-automorphism of the physical Hamiltonian graph gives a useful
    graph-scrambled locality control: it has the identical parameter count,
    per-parameter support, support-graph degree sequence, and determinant
    algebra as the physical map, but its edges are not local in the target
    lattice geometry.
    """

    mapping = np.asarray(parameter_map, dtype=np.float64)
    if mapping.ndim != 2:
        raise ValueError("parameter_map must be a matrix")
    permutation = tuple(int(value) for value in site_permutation)
    if tuple(sorted(permutation)) != tuple(range(len(permutation))):
        raise ValueError("site_permutation must contain every site exactly once")
    reference = packed_pair_index(len(permutation))
    if mapping.shape[1] != len(reference):
        raise ValueError("parameter-map width and site permutation disagree")
    result = np.zeros_like(mapping)
    for edge, old_index in reference.items():
        new_edge = tuple(sorted((permutation[edge[0]], permutation[edge[1]])))
        result[:, reference[new_edge]] = mapping[:, old_index]
    return result


__all__ = [
    "LocalPairingFamily",
    "expand_packed_pairing",
    "packed_pair_index",
    "permute_packed_pairing_parameter_map",
    "square_local_pairing_parameter_map",
    "triangular_local_pairing_parameter_map",
]
