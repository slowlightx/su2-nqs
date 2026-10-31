"""Deterministic polynomial-time hierarchies from weighted Hamiltonian graphs.

This module is deliberately state blind.  Its inputs are a number of sites
and weighted, undirected Hamiltonian edges; it has no wavefunction, energy,
optimizer, or filesystem dependency.  The construction has two stages:

1. coarsen a dominant-bond triangular or square-cycle motif cover when a
   greedy full cover exists, otherwise fall back to individual vertices;
2. recursively bisect the contracted graph, using a bounded sequence of
   balanced pair swaps to lower the weighted cut at every node.

The bounded refinement makes the construction polynomial rather than a
search over cell permutations.  All numerical and label ties have explicit
lexicographic rules.  The greedy initialization also uses canonical integer
site labels as an external lattice/HNF frame.  In particular, a unique
labelled hierarchy cannot be equivariant under every automorphism of a
vertex-transitive graph; the returned audit states the relabelling boundary
instead of claiming arbitrary covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from math import fsum, isfinite
from numbers import Integral, Real
from typing import Literal, Sequence

from ..su2.fusion_tree import FusionNode, FusionTree, Leaf, TreeNode
from .j1j2 import square_j1j2_bonds
from .triangular_hnf import IntMatrix, triangular_hnf_j1j2_bonds
from .triangular_j1j2 import triangular_j1j2_bonds


WeightedEdge = tuple[int, int, float]
MotifKind = Literal["triangle", "square_cycle", "vertices"]

RELABEL_COVARIANCE_BOUNDARY = (
    "bond-row-order invariant; canonical integer site labels seed the greedy "
    "coarsening and bisection and resolve graph-symmetric ties, so arbitrary "
    "site-relabelling covariance is not guaranteed; lattice/HNF canonical labels "
    "are an explicit external frame"
)

EDGE_AGGREGATION_CONTRACT = (
    "repeated undirected edge rows are summed algebraically before the "
    "absolute coupling is used for boundary scoring"
)

PERIODIC_ALIAS_LIMITATION = (
    "the hierarchy consumes an effective pair graph and cannot reconstruct "
    "periodic-image or interaction-shell provenance after the lattice builder "
    "has aggregated it; production geometries must audit unintended aliases"
)


@dataclass(frozen=True)
class HierarchyWorkAudit:
    """Deterministic operation counts for the graph construction."""

    motif_candidates: int
    motif_rank_evaluations: int
    motif_feasibility_checks: int
    bisection_cut_evaluations: int
    bisection_edge_accumulations: int
    bisection_swap_candidates: int
    accepted_balanced_swaps: int
    label_tie_breaks: int


@dataclass(frozen=True)
class GraphHierarchy:
    """A fusion tree, physical leaf assignment, and state-blind audit."""

    tree: FusionTree
    site_order: tuple[int, ...]
    cells: tuple[tuple[int, ...], ...]
    requested_motif: MotifKind
    realized_coarsening: MotifKind
    hierarchy_boundary_score: float
    graph_sha256: str
    work: HierarchyWorkAudit
    construction: str = "dominant_motif_bounded_recursive_bisection_v1"
    canonical_label_frame_used: bool = True
    relabel_covariance_boundary: str = RELABEL_COVARIANCE_BOUNDARY
    edge_aggregation_contract: str = EDGE_AGGREGATION_CONTRACT
    periodic_image_provenance_available: bool = False
    periodic_alias_limitation: str = PERIODIC_ALIAS_LIMITATION


@dataclass
class _Work:
    motif_candidates: int = 0
    motif_rank_evaluations: int = 0
    motif_feasibility_checks: int = 0
    bisection_cut_evaluations: int = 0
    bisection_edge_accumulations: int = 0
    bisection_swap_candidates: int = 0
    accepted_balanced_swaps: int = 0
    label_tie_breaks: int = 0

    def freeze(self) -> HierarchyWorkAudit:
        return HierarchyWorkAudit(
            motif_candidates=self.motif_candidates,
            motif_rank_evaluations=self.motif_rank_evaluations,
            motif_feasibility_checks=self.motif_feasibility_checks,
            bisection_cut_evaluations=self.bisection_cut_evaluations,
            bisection_edge_accumulations=self.bisection_edge_accumulations,
            bisection_swap_candidates=self.bisection_swap_candidates,
            accepted_balanced_swaps=self.accepted_balanced_swaps,
            label_tie_breaks=self.label_tie_breaks,
        )


def _validate_n_sites(n_sites: int) -> int:
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be an integer")
    checked = int(n_sites)
    if checked < 2:
        raise ValueError("a graph hierarchy requires at least two sites")
    return checked


def _canonical_edges(
    n_sites: int, edges: Sequence[Sequence[Real]],
) -> tuple[WeightedEdge, ...]:
    """Combine duplicate undirected edges independent of input row order."""

    grouped: dict[tuple[int, int], list[float]] = {}
    for row in edges:
        if len(row) != 3:
            raise ValueError("each edge must contain first, second, and coupling")
        first_raw, second_raw, coupling_raw = row
        if (
            isinstance(first_raw, bool)
            or isinstance(second_raw, bool)
            or not isinstance(first_raw, Integral)
            or not isinstance(second_raw, Integral)
        ):
            raise TypeError("edge endpoints must be integers")
        first, second = int(first_raw), int(second_raw)
        if not 0 <= first < n_sites or not 0 <= second < n_sites:
            raise ValueError("edge endpoint lies outside the graph")
        if first == second:
            raise ValueError("self edges are not supported")
        coupling = float(coupling_raw)
        if not isfinite(coupling):
            raise ValueError("edge couplings must be finite")
        pair = tuple(sorted((first, second)))
        grouped.setdefault(pair, []).append(coupling)
    canonical = []
    for (first, second), values in sorted(grouped.items()):
        # Sorting plus fsum makes duplicate-edge reduction independent of the
        # row order, including mixed-sign diagnostic graphs.
        coupling = fsum(sorted(values))
        if coupling != 0.0:
            canonical.append((first, second, coupling))
    if not canonical:
        raise ValueError("the Hamiltonian graph must contain a nonzero edge")
    return tuple(canonical)


def _graph_hash(n_sites: int, edges: Sequence[WeightedEdge]) -> str:
    payload = {"n_sites": n_sites, "edges": [list(row) for row in edges]}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _edge_weights(edges: Sequence[WeightedEdge]) -> dict[tuple[int, int], float]:
    return {(first, second): abs(coupling) for first, second, coupling in edges}


def _dominant_adjacency(
    n_sites: int, weights: dict[tuple[int, int], float], *, tolerance: float,
) -> tuple[set[int], ...]:
    maximum = max(weights.values())
    threshold = maximum - tolerance * max(1.0, maximum)
    adjacency = [set() for _ in range(n_sites)]
    for (first, second), weight in weights.items():
        if weight >= threshold:
            adjacency[first].add(second)
            adjacency[second].add(first)
    return tuple(adjacency)


def _triangle_candidates(adjacency: Sequence[set[int]]) -> tuple[tuple[int, ...], ...]:
    result = []
    for first in range(len(adjacency)):
        for second in sorted(site for site in adjacency[first] if site > first):
            result.extend(
                (first, second, third)
                for third in sorted(adjacency[first].intersection(adjacency[second]))
                if third > second
            )
    return tuple(result)


def _square_cycle_candidates(
    adjacency: Sequence[set[int]],
) -> tuple[tuple[int, ...], ...]:
    """Enumerate chordless four-cycles without an ``N!`` ordering search."""

    found: set[tuple[int, ...]] = set()
    n_sites = len(adjacency)
    for first in range(n_sites):
        for opposite in range(first + 1, n_sites):
            common = sorted(adjacency[first].intersection(adjacency[opposite]))
            for second, fourth in combinations(common, 2):
                cell = tuple(sorted((first, second, opposite, fourth)))
                if len(set(cell)) != 4:
                    continue
                edge_count = sum(
                    right in adjacency[left]
                    for left, right in combinations(cell, 2)
                )
                if edge_count == 4:
                    found.add(cell)
    return tuple(sorted(found))


def _support_boundary(
    support: Sequence[int], weights: dict[tuple[int, int], float],
) -> float:
    inside = set(support)
    return fsum(
        weight
        for (first, second), weight in weights.items()
        if (first in inside) != (second in inside)
    )


def _support_internal_weight(
    support: Sequence[int], weights: dict[tuple[int, int], float],
) -> float:
    inside = set(support)
    return fsum(
        weight
        for (first, second), weight in weights.items()
        if first in inside and second in inside
    )


def _greedy_full_motif_cover(
    n_sites: int,
    candidates: Sequence[tuple[int, ...]],
    weights: dict[tuple[int, int], float],
    work: _Work,
    *,
    tolerance: float,
) -> tuple[tuple[int, ...], ...] | None:
    """Return a deterministic cover or ``None`` so the caller can fall back."""

    work.motif_candidates += len(candidates)
    if not candidates:
        return None
    motif_size = len(candidates[0])
    if n_sites % motif_size:
        return None
    candidate_sets = tuple((candidate, frozenset(candidate)) for candidate in candidates)
    by_site = {
        site: tuple(row for row in candidate_sets if site in row[1])
        for site in range(n_sites)
    }
    remaining = frozenset(range(n_sites))
    selected: list[tuple[int, ...]] = []
    while remaining:
        anchor = min(remaining)
        available = [
            row for row in by_site[anchor] if row[1].issubset(remaining)
        ]
        if not available:
            return None
        ranked = []
        for candidate, sites in available:
            residual = remaining.difference(sites)
            dead = 0
            for site in residual:
                coverable = False
                for _, other_sites in by_site[site]:
                    work.motif_feasibility_checks += 1
                    if other_sites.issubset(residual):
                        coverable = True
                        break
                dead += not coverable
            ranked.append(
                (
                    dead,
                    _support_boundary(candidate, weights),
                    -_support_internal_weight(candidate, weights),
                    candidate,
                )
            )
            work.motif_rank_evaluations += 1
        best_prefix = min(row[:3] for row in ranked)
        tied = [
            row
            for row in ranked
            if row[0] == best_prefix[0]
            and abs(row[1] - best_prefix[1]) <= tolerance
            and abs(row[2] - best_prefix[2]) <= tolerance
        ]
        if len(tied) > 1:
            work.label_tie_breaks += 1
        chosen = min(tied, key=lambda row: row[3])[3]
        selected.append(chosen)
        remaining = remaining.difference(chosen)
    return tuple(selected)


def _cell_graph(
    cells: Sequence[tuple[int, ...]],
    weights: dict[tuple[int, int], float],
) -> tuple[tuple[float, ...], ...]:
    cell_of = {site: cell for cell, sites in enumerate(cells) for site in sites}
    if len(cell_of) != sum(map(len, cells)):
        raise RuntimeError("coarsening cells overlap")
    matrix = [[0.0 for _ in cells] for _ in cells]
    for (first, second), weight in weights.items():
        left, right = cell_of[first], cell_of[second]
        if left != right:
            matrix[left][right] += weight
            matrix[right][left] += weight
    return tuple(tuple(row) for row in matrix)


def _cut_weight(
    left: set[int], indices: Sequence[int], matrix: Sequence[Sequence[float]], work: _Work,
) -> float:
    work.bisection_cut_evaluations += 1
    right = set(indices).difference(left)
    work.bisection_edge_accumulations += len(left) * len(right)
    return fsum(matrix[first][second] for first in left for second in right)


def _group_signature(indices: Sequence[int], cells: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    return tuple(sorted(site for index in indices for site in cells[index]))


def _recursive_bisection_order(
    indices: Sequence[int],
    cells: Sequence[tuple[int, ...]],
    matrix: Sequence[Sequence[float]],
    work: _Work,
    *,
    tolerance: float,
) -> tuple[int, ...]:
    ordered = tuple(sorted(indices, key=lambda index: cells[index]))
    size = len(ordered)
    if size <= 1:
        return ordered
    left_size = size // 2
    left = set(ordered[:left_size])
    current = _cut_weight(left, ordered, matrix, work)

    # A hard polynomial cap is part of the algorithmic contract.  In
    # practice the regular-lattice canaries accept only a handful of swaps.
    for _ in range(size * size):
        right = set(ordered).difference(left)
        candidates = []
        for departing in sorted(left, key=lambda index: cells[index]):
            for arriving in sorted(right, key=lambda index: cells[index]):
                candidate_left = left.difference((departing,)).union((arriving,))
                score = _cut_weight(candidate_left, ordered, matrix, work)
                signature = _group_signature(tuple(candidate_left), cells)
                candidates.append((score, signature, candidate_left))
                work.bisection_swap_candidates += 1
        if not candidates:
            break
        best_score = min(row[0] for row in candidates)
        tied = [row for row in candidates if abs(row[0] - best_score) <= tolerance]
        if len(tied) > 1:
            work.label_tie_breaks += 1
        selected_score, _, selected_left = min(tied, key=lambda row: row[1])
        if selected_score >= current - tolerance:
            break
        left = set(selected_left)
        current = selected_score
        work.accepted_balanced_swaps += 1

    right = set(ordered).difference(left)
    left_indices = tuple(sorted(left, key=lambda index: cells[index]))
    right_indices = tuple(sorted(right, key=lambda index: cells[index]))
    # The balanced tree uses floor(size/2) leaves on the left.  Swapping
    # unequal groups would silently change which subproblem receives that
    # shape, so orientation canonicalization is legal only for equal sizes.
    if len(left_indices) == len(right_indices) and _group_signature(
        left_indices, cells
    ) > _group_signature(right_indices, cells):
        left_indices, right_indices = right_indices, left_indices
    return _recursive_bisection_order(
        left_indices, cells, matrix, work, tolerance=tolerance
    ) + _recursive_bisection_order(
        right_indices, cells, matrix, work, tolerance=tolerance
    )


def _local_cell_order(
    cell: tuple[int, ...], kind: MotifKind, weights: dict[tuple[int, int], float]
) -> tuple[int, ...]:
    if kind == "vertices":
        return cell
    if kind == "triangle":
        pairs = []
        for first, second in combinations(cell, 2):
            pair = tuple(sorted((first, second)))
            pairs.append((-weights.get(pair, 0.0), pair))
        _, pair = min(pairs)
        remaining = next(site for site in cell if site not in pair)
        return pair + (remaining,)
    if kind == "square_cycle":
        first, second, third, fourth = cell
        matchings = (
            ((first, second), (third, fourth)),
            ((first, third), (second, fourth)),
            ((first, fourth), (second, third)),
        )
        ranked = []
        for raw_left, raw_right in matchings:
            left, right = tuple(sorted(raw_left)), tuple(sorted(raw_right))
            if right < left:
                left, right = right, left
            score = weights.get(left, 0.0) + weights.get(right, 0.0)
            ranked.append((-score, left, right))
        _, left, right = min(ranked)
        return left + right
    raise ValueError(f"unknown motif kind {kind!r}")


def _balanced_nodes(nodes: Sequence[TreeNode]) -> TreeNode:
    if len(nodes) == 1:
        return nodes[0]
    middle = len(nodes) // 2
    return FusionNode(_balanced_nodes(nodes[:middle]), _balanced_nodes(nodes[middle:]))


def _tree_and_order(
    cells: Sequence[tuple[int, ...]], kind: MotifKind, weights: dict[tuple[int, int], float]
) -> tuple[FusionTree, tuple[int, ...], tuple[tuple[int, ...], ...]]:
    site_order: list[int] = []
    ordered_cells = []
    nodes: list[TreeNode] = []
    for raw_cell in cells:
        cell = _local_cell_order(raw_cell, kind, weights)
        start = len(site_order)
        site_order.extend(cell)
        ordered_cells.append(cell)
        if kind == "triangle":
            nodes.append(
                FusionNode(FusionNode(Leaf(start), Leaf(start + 1)), Leaf(start + 2))
            )
        elif kind == "square_cycle":
            nodes.append(
                FusionNode(
                    FusionNode(Leaf(start), Leaf(start + 1)),
                    FusionNode(Leaf(start + 2), Leaf(start + 3)),
                )
            )
        else:
            nodes.append(Leaf(start))
    tree = FusionTree(_balanced_nodes(nodes))
    return tree, tuple(site_order), tuple(ordered_cells)


def _physical_support(node: TreeNode, site_order: Sequence[int]) -> tuple[int, ...]:
    if isinstance(node, Leaf):
        return (site_order[node.index],)
    return _physical_support(node.left, site_order) + _physical_support(
        node.right, site_order
    )


def hierarchy_boundary_score(
    tree: FusionTree, site_order: Sequence[int], edges: Sequence[Sequence[Real]]
) -> float:
    """Sum weighted cut boundaries over non-root internal tree nodes.

    After repeated undirected rows have been algebraically aggregated into
    canonical Hamiltonian edges, this is exactly the discovery-runner
    convention: each non-root internal support contributes the absolute
    coupling of every crossing effective edge, with neither a factor of one
    half nor a per-site normalization.  On raw mixed-sign duplicate rows the
    discovery helper instead sums row-wise absolute values; callers must not
    conflate that different diagnostic convention with an effective graph.
    """

    if tuple(sorted(site_order)) != tuple(range(tree.n_leaves)):
        raise ValueError("site_order must be a permutation of all sites")
    canonical = _canonical_edges(tree.n_leaves, edges)
    weights = _edge_weights(canonical)
    return fsum(
        _support_boundary(_physical_support(node, site_order), weights)
        for node in tree.internal_nodes[:-1]
    )


def hamiltonian_graph_hierarchy(
    n_sites: int,
    edges: Sequence[Sequence[Real]],
    *,
    motif: MotifKind = "vertices",
    tolerance: float = 1.0e-12,
) -> GraphHierarchy:
    """Construct a deterministic hierarchy using only a weighted graph.

    ``motif='triangle'`` and ``motif='square_cycle'`` inspect the subgraph of
    maximum-absolute-coupling edges.  If the deterministic greedy motif pass
    does not cover every vertex, the method safely falls back to vertex-level
    recursive bisection; it never launches an exact-cover or permutation
    search.
    """

    checked_n = _validate_n_sites(n_sites)
    if motif not in ("triangle", "square_cycle", "vertices"):
        raise ValueError(f"unknown motif kind {motif!r}")
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    canonical = _canonical_edges(checked_n, edges)
    weights = _edge_weights(canonical)
    work = _Work()

    cells: tuple[tuple[int, ...], ...] | None = None
    realized: MotifKind = "vertices"
    if motif != "vertices":
        adjacency = _dominant_adjacency(checked_n, weights, tolerance=tolerance)
        candidates = (
            _triangle_candidates(adjacency)
            if motif == "triangle"
            else _square_cycle_candidates(adjacency)
        )
        cells = _greedy_full_motif_cover(
            checked_n, candidates, weights, work, tolerance=tolerance
        )
        if cells is not None:
            realized = motif
    if cells is None:
        cells = tuple((site,) for site in range(checked_n))

    canonical_cells = tuple(sorted(tuple(sorted(cell)) for cell in cells))
    matrix = _cell_graph(canonical_cells, weights)
    cell_order = _recursive_bisection_order(
        tuple(range(len(canonical_cells))),
        canonical_cells,
        matrix,
        work,
        tolerance=tolerance,
    )
    ordered_cells = tuple(canonical_cells[index] for index in cell_order)
    tree, site_order, displayed_cells = _tree_and_order(
        ordered_cells, realized, weights
    )
    if tuple(sorted(site_order)) != tuple(range(checked_n)):
        raise RuntimeError("constructed hierarchy lost or duplicated a site")
    score = hierarchy_boundary_score(tree, site_order, canonical)
    return GraphHierarchy(
        tree=tree,
        site_order=site_order,
        cells=displayed_cells,
        requested_motif=motif,
        realized_coarsening=realized,
        hierarchy_boundary_score=score,
        graph_sha256=_graph_hash(checked_n, canonical),
        work=work.freeze(),
    )


def square_pbc_graph_hierarchy(
    rows: int, columns: int, *, j1: float = 1.0, j2: float = 0.0
) -> GraphHierarchy:
    """Construct a square-torus hierarchy from its effective Hamiltonian graph.

    The lattice builder has already aggregated periodic images by site pair;
    inspect the returned alias-limitation audit before using unusually narrow
    tori, where image or shell provenance can affect the dominant motif.
    """

    bonds = square_j1j2_bonds(rows, columns, j1=j1, j2=j2, periodic=True)
    return hamiltonian_graph_hierarchy(
        rows * columns, bonds, motif="square_cycle"
    )


def triangular_pbc_graph_hierarchy(
    rows: int, columns: int, *, j1: float = 1.0, j2: float = 0.0
) -> GraphHierarchy:
    """Construct a triangular-torus hierarchy from its effective graph.

    The lattice builder has already aggregated periodic images by site pair;
    inspect the returned alias-limitation audit before using unusually narrow
    tori, where image or shell provenance can affect the dominant motif.
    """

    bonds = triangular_j1j2_bonds(rows, columns, j1=j1, j2=j2, periodic=True)
    return hamiltonian_graph_hierarchy(rows * columns, bonds, motif="triangle")


def triangular_hnf_graph_hierarchy(
    supercell: IntMatrix, *, j1: float = 1.0, j2: float = 0.0
) -> GraphHierarchy:
    """Construct a general-HNF hierarchy from its effective graph.

    The HNF builder has already aggregated periodic images by site pair;
    inspect the returned alias-limitation audit before using unusually narrow
    tori, where image or shell provenance can affect the dominant motif.
    """

    bonds = triangular_hnf_j1j2_bonds(supercell, j1=j1, j2=j2)
    n_sites = int(supercell[0][0]) * int(supercell[1][1])
    return hamiltonian_graph_hierarchy(n_sites, bonds, motif="triangle")


__all__ = [
    "GraphHierarchy",
    "EDGE_AGGREGATION_CONTRACT",
    "HierarchyWorkAudit",
    "MotifKind",
    "PERIODIC_ALIAS_LIMITATION",
    "RELABEL_COVARIANCE_BOUNDARY",
    "WeightedEdge",
    "hamiltonian_graph_hierarchy",
    "hierarchy_boundary_score",
    "square_pbc_graph_hierarchy",
    "triangular_hnf_graph_hierarchy",
    "triangular_pbc_graph_hierarchy",
]
