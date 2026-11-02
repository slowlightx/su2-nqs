"""Explicit torus-tree reconnection and exact physical-bond bookkeeping.

This opt-in geometry helper never changes a production decoder default. It
reuses the ordered nine-site leaves and the physical 72|72 partition of an
existing L=12 cover. New 18/36/72 intertwiners must be constructed separately:
changing a finite-width tree is NOT a state-preserving tensor permutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Sequence

from .graph_hierarchy import _canonical_edges
from .triangular_j1j2 import triangular_j1j2_bonds
from .triangular_schur_hierarchy import TriangularSchurHierarchy

Bond = tuple[int, int, float]
Merge = tuple[int, int, int]


@dataclass(frozen=True)
class MergeBondLedger:
    """Every physical pair assigned once, with child-local site coordinates.

    ``leaf_bonds[i]`` is supplied to the existing physical leaf projector.
    ``merge_bonds[k]`` is supplied unchanged to ``solve_sector_merge`` or
    ``project_sector_maps`` for merge k. Complete doubled-q multiplets and
    retained widths are selected by those operators, not by this geometry.
    """
    supports: tuple[tuple[int, ...], ...]
    physical_bonds: tuple[Bond, ...]
    leaf_bonds: tuple[tuple[Bond, ...], ...]
    merge_bonds: tuple[tuple[Bond, ...], ...]


def _supports(leaves: Sequence[Sequence[int]], merges: Sequence[Merge]) -> tuple[tuple[int, ...], ...]:
    if not leaves or len(merges) != len(leaves)-1:
        raise ValueError("one complete binary hierarchy over nonempty leaves required")
    out = []
    for sites in leaves:
        if not sites or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in sites):
            raise ValueError("leaf supports require nonnegative integer physical sites")
        out.append(tuple(int(i) for i in sites))
    flat = [i for row in out for i in row]
    if sorted(flat) != list(range(len(flat))):
        raise ValueError("ordered leaves must partition physical sites 0..N-1 exactly")
    used = set()
    for row in merges:
        if len(row) != 3 or any(isinstance(i, bool) or not isinstance(i, Integral) for i in row):
            raise ValueError("merge IDs must be integer triples")
        left, right, parent = map(int, row)
        if (parent != len(out) or min(left, right) < 0 or max(left, right) >= parent
                or left == right or left in used or right in used):
            raise ValueError("merge IDs violate single-use topological tree ordering")
        if set(out[left]) & set(out[right]):
            raise ValueError("merge child supports overlap")
        used.update((left, right)); out.append(out[left]+out[right])
    if sorted(out[-1]) != list(range(len(flat))):
        raise ValueError("root does not cover the physical system")
    return tuple(out)


def build_merge_bond_ledger(leaf_supports: Sequence[Sequence[int]], binary_merges: Sequence[Merge],
                            bonds: Sequence[Bond]) -> MergeBondLedger:
    """Route actual H bonds once to their physical lowest common ancestor.

    Repeated undirected pairs are summed algebraically before routing. The
    physical local-site order follows each ordered support, even when a new
    torus pairing changes it. No energy or wavefunction is evaluated.
    """
    supports = _supports(leaf_supports, binary_merges)
    edges = _canonical_edges(len(supports[-1]), bonds)
    leaf_count = len(leaf_supports)
    positions = [{site: j for j, site in enumerate(row)} for row in supports]
    leaf_rows: list[list[Bond]] = [[] for _ in leaf_supports]
    merge_rows: list[list[Bond]] = [[] for _ in binary_merges]
    for first, second, strength in edges:
        assigned = 0
        for i, position in enumerate(positions[:leaf_count]):
            if first in position and second in position:
                leaf_rows[i].append((position[first], position[second], strength)); assigned += 1
        for k, (left, right, _) in enumerate(binary_merges):
            lp, rp = positions[left], positions[right]
            if first in lp and second in rp:
                merge_rows[k].append((lp[first], rp[second], strength)); assigned += 1
            elif second in lp and first in rp:
                merge_rows[k].append((lp[second], rp[first], strength)); assigned += 1
        if assigned != 1:
            raise ArithmeticError("physical bond was omitted or assigned more than once")
    return MergeBondLedger(supports, edges, tuple(tuple(x) for x in leaf_rows),
                           tuple(tuple(x) for x in merge_rows))


def reconnect_triangular_hierarchy(base: TriangularSchurHierarchy, *,
        pairs18: Sequence[Sequence[int]], groups36: Sequence[Sequence[int]],
        require_J1_connected: bool = True) -> TriangularSchurHierarchy:
    """Opt-in 18/36 reconnection inside the SAME L=12 physical root halves.

    Only sixteen ordered nine-site leaf objects are reused exactly. Pair and
    group labels are canonicalized lexicographically; node numbering remains
    0..15 leaves, 16..23 pairs, 24..27 groups, 28/29 halves, 30 root. This
    returns geometry, NOT recoupled tensors or a preserved finite-width state.
    """
    if not isinstance(base, TriangularSchurHierarchy) or base.length != 12 or len(base.blocks) != 16:
        raise ValueError("this bounded reconnection requires the explicit L=12 sixteen-leaf cover")
    if not isinstance(require_J1_connected, bool):
        raise ValueError("connectivity flag must be boolean")
    original = _supports(tuple(block.sites for block in base.blocks), base.binary_merges)
    root_left, root_right, _ = base.binary_merges[-1]
    physical_halves = (set(original[root_left]), set(original[root_right]))
    halves = tuple(frozenset(i for i, block in enumerate(base.blocks) if set(block.sites) <= region)
                   for region in physical_halves)
    if tuple(map(len, halves)) != (8, 8) or halves[0] & halves[1]:
        raise ValueError("base does not have two eight-leaf physical root halves")
    normalized = []
    for rows, size, count in ((pairs18, 2, 8), (groups36, 4, 4)):
        checked = []
        for row in rows:
            if (len(row) != size or any(isinstance(i, bool) or not isinstance(i, Integral)
                                       or not 0 <= i < 16 for i in row)):
                raise ValueError("pair/group rows require distinct valid leaf IDs")
            value = tuple(sorted(map(int, row)))
            if len(set(value)) != size:
                raise ValueError("pair/group rows repeat a leaf")
            checked.append(value)
        if len(checked) != count or sorted(i for row in checked for i in row) != list(range(16)):
            raise ValueError("pairs and groups must each partition all sixteen leaves")
        if any(not any(set(row) <= half for half in halves) for row in checked):
            raise ValueError("reconnection cannot cross the frozen physical 72|72 partition")
        normalized.append(tuple(sorted(checked)))
    pairs, groups = normalized
    groups = tuple(g for half in halves for g in groups if set(g) <= half)
    merges = [(a, b, 16+k) for k, (a, b) in enumerate(pairs)]
    for k, group in enumerate(groups):
        children = [16+i for i, pair in enumerate(pairs) if set(pair) <= set(group)]
        if len(children) != 2 or set(x for p in pairs if set(p) <= set(group) for x in p) != set(group):
            raise ValueError("each 36-site group must contain exactly two whole 18-site pairs")
        merges.append((children[0], children[1], 24+k))
    merges.extend(((24, 25, 28), (26, 27, 29), (28, 29, 30)))
    candidate = TriangularSchurHierarchy(base.length, base.blocks, tuple(merges))
    supports = _supports(tuple(block.sites for block in candidate.blocks), candidate.binary_merges)
    if any(set(supports[28+i]) != physical_halves[i] for i in (0, 1)):
        raise ArithmeticError("new hierarchy changed the physical root partition")
    if require_J1_connected:
        edges = triangular_j1j2_bonds(12, 12, j2=0., periodic=True)
        adjacent = [set() for _ in range(144)]
        for first, second, _ in edges:
            adjacent[first].add(second); adjacent[second].add(first)
        for region in supports[16:28]:
            unseen = set(region); queue = [unseen.pop()]
            for site in queue:
                found = unseen & adjacent[site]; unseen.difference_update(found); queue.extend(found)
            if unseen:
                raise ValueError("new 18/36 region is disconnected in the physical J1 graph")
    return candidate


def weighted_layer_cuts(hierarchy: TriangularSchurHierarchy, bonds: Sequence[Bond], *,
                         region_sizes: Sequence[int] = (18, 36)) -> Mapping[int, float]:
    """Geometry proxy: sum |J| cuts of the selected virtual-edge regions.

    This is not an energy percentage, entanglement estimate, or variational
    comparison. The unchanged root is excluded unless its size is requested.
    """
    ledger = build_merge_bond_ledger(tuple(block.sites for block in hierarchy.blocks),
                                    hierarchy.binary_merges, bonds)
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in region_sizes):
        raise ValueError("positive integer region sizes required")
    return {int(n): sum(abs(g) for region in ledger.supports[:-1] if len(region) == n
                       for a, b, g in ledger.physical_bonds if (a in region) != (b in region))
            for n in region_sizes}
