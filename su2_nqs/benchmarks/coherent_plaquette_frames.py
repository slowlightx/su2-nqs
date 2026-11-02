"""Exact-small coherent plaquette-frame carriers and geometry scrambles.

The physical carrier starts from one tensor product of four-spin singlets and
adds every translated/rotated image with the ``Gamma-A1`` character.  At
finite size this is exactly a Reynolds projection.  Keeping the construction
in terms of frames is nevertheless useful: it exposes which Hamiltonian bonds
are internal in at least one frame and supplies a clean control obtained by a
single non-geometric site permutation of the whole coherent frame bank.

Every column is an exact total-spin singlet before the spatial projection.
The learned coordinate is only the shared local ``A1/B1`` mixing angle; no
magnetic index is represented or optimized.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from ..lattices.spatial_symmetry import SitePermutation, validate_site_permutation
from .clebsch_tree_exact import physical_permutation_index_table
from .local_carrier_product_subspace import square_a1_b1_product_columns


Supports = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CoherentPlaquetteFrameColumns:
    """Shared-angle columns for one coherent plaquette-frame bank."""

    columns: np.ndarray
    labels: tuple[tuple[int, ...], ...]
    group_size: int
    distinct_partition_count: int
    primary_partition: Supports


def _canonical_partition(cells: Sequence[Sequence[int]]) -> Supports:
    return tuple(
        sorted(tuple(sorted(int(site) for site in cell)) for cell in cells)
    )


def distinct_partition_orbit(
    cells: Sequence[Sequence[int]],
    group: Sequence[Sequence[int]],
) -> tuple[Supports, ...]:
    """Return distinct unordered cell partitions in one spatial-group orbit."""

    primary = tuple(tuple(int(site) for site in cell) for cell in cells)
    if not primary or any(len(cell) != 4 for cell in primary):
        raise ValueError("coherent plaquette frames require four-site cells")
    n_sites = sum(len(cell) for cell in primary)
    if tuple(sorted(site for cell in primary for site in cell)) != tuple(
        range(n_sites)
    ):
        raise ValueError("primary cells must partition every site exactly once")
    images = tuple(
        validate_site_permutation(image, n_sites=n_sites) for image in group
    )
    if not images:
        raise ValueError("frame group must be nonempty")
    result: list[Supports] = []
    seen: set[Supports] = set()
    for image in images:
        mapped = tuple(
            tuple(image[site] for site in cell) for cell in primary
        )
        signature = _canonical_partition(mapped)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(mapped)
    return tuple(result)


def coherent_plaquette_frame_columns(
    basis: Sequence[Sequence[int]],
    cells: Sequence[Sequence[int]],
    group: Sequence[Sequence[int]],
) -> CoherentPlaquetteFrameColumns:
    """Return the explicit equal-character frame sum in the physical basis.

    The returned columns are the full space-group average of the primary
    product columns.  This is the exact coherent sum over translated/rotated
    frames, including the correct multiplicity from the stabilizer of the
    primary partition.
    """

    primary = tuple(tuple(int(site) for site in cell) for cell in cells)
    columns, labels = square_a1_b1_product_columns(basis, primary)
    indices = physical_permutation_index_table(basis, group)
    projected = np.mean(columns[indices], axis=0)
    if np.max(np.abs(projected.imag), initial=0.0) <= 2.0e-14:
        projected = np.asarray(projected.real, dtype=np.float64)
    projected.setflags(write=False)
    return CoherentPlaquetteFrameColumns(
        columns=projected,
        labels=labels,
        group_size=int(indices.shape[0]),
        distinct_partition_count=len(distinct_partition_orbit(primary, group)),
        primary_partition=primary,
    )


def scrambled_coherent_frame_columns(
    physical: CoherentPlaquetteFrameColumns,
    basis: Sequence[Sequence[int]],
    group: Sequence[Sequence[int]],
    site_permutation: Sequence[int],
) -> np.ndarray:
    """Apply one non-geometric site scramble and project back to ``Gamma-A1``.

    The same permutation acts on every frame and local channel, so the control
    preserves the pre-projection column Gram matrix, parameter count, frame
    count, and exact total-spin singlet structure.  Only the registration of
    the frame bank with the Hamiltonian geometry is destroyed.
    """

    n_sites = len(basis[0])
    scramble = validate_site_permutation(site_permutation, n_sites=n_sites)
    scramble_indices = physical_permutation_index_table(basis, (scramble,))[0]
    group_indices = physical_permutation_index_table(basis, group)
    moved = physical.columns[scramble_indices]
    result = np.mean(moved[group_indices], axis=0)
    if np.max(np.abs(result.imag), initial=0.0) <= 2.0e-14:
        result = np.asarray(result.real, dtype=np.float64)
    result.setflags(write=False)
    return result


def nongeometric_site_permutation(
    *,
    n_sites: int,
    group: Sequence[Sequence[int]],
    seed: int,
) -> SitePermutation:
    """Draw a deterministic site permutation outside the supplied group."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 2:
        raise ValueError("n_sites must be an integer of at least two")
    group_set = {
        validate_site_permutation(image, n_sites=int(n_sites)) for image in group
    }
    rng = np.random.default_rng(int(seed))
    for _ in range(10_000):
        candidate = tuple(int(value) for value in rng.permutation(int(n_sites)))
        if candidate not in group_set:
            return validate_site_permutation(candidate, n_sites=int(n_sites))
    raise RuntimeError("failed to draw a non-geometric site permutation")


def internal_bond_coverage(
    partitions: Sequence[Sequence[Sequence[int]]],
    bonds: Sequence[tuple[int, int, float]],
) -> dict[str, float | int]:
    """Count nonzero bonds internal to at least one supplied cell partition."""

    relevant = tuple((int(left), int(right)) for left, right, weight in bonds if weight)
    if not relevant:
        raise ValueError("at least one nonzero bond is required")
    covered: set[tuple[int, int]] = set()
    for partition in partitions:
        for cell in partition:
            sites = set(int(site) for site in cell)
            for left, right in relevant:
                if left in sites and right in sites:
                    covered.add(tuple(sorted((left, right))))
    unique = {tuple(sorted(bond)) for bond in relevant}
    return {
        "bond_count": len(unique),
        "covered_bond_count": len(covered & unique),
        "coverage_fraction": len(covered & unique) / len(unique),
    }


__all__ = [
    "CoherentPlaquetteFrameColumns",
    "coherent_plaquette_frame_columns",
    "distinct_partition_orbit",
    "internal_bond_coverage",
    "nongeometric_site_permutation",
    "scrambled_coherent_frame_columns",
]
