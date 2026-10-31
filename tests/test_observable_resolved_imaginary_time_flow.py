"""Small deterministic gates for the L5 lattice flow."""

from __future__ import annotations

from su2_nqs.benchmarks.lattice_observables import (
    triangular_nearest_neighbor_orientation_bonds,
)


def test_triangular_orientation_families_partition_j1_bonds() -> None:
    families = triangular_nearest_neighbor_orientation_bonds(3, 4)
    assert tuple(len(family) for family in families) == (8, 9, 6)
    edges = [tuple(sorted((first, second))) for family in families for first, second, _ in family]
    assert len(edges) == len(set(edges)) == 23
