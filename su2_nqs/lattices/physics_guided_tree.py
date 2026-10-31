"""State-blind physical diagnostics for spatial SU(2) fusion trees.

The functions in this module only consume a lattice, exchange bonds, point-
group permutations, and a fusion-tree structure.  In particular, no API
accepts a wavefunction, an exact eigenvector, a neural checkpoint, or an
energy.  They therefore define coordinates that can be frozen before any
variational calculation is inspected.

Every non-root tree node labels one *tree edge* and hence one edge-indexed
bipartition.  Complementary root-child cuts are deliberately retained: they
are two different tree edges.  With that convention the exact identity is

    sum_tree_edges sum_b |J_b| 1[b crosses edge]
      = sum_b |J_b| d_T(i_b, j_b).

There is no factor of one half.  This cut-L1 sum is consequently a different
representation of bond distance, not an independent objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import comb, exp, log, log2
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..su2.fusion_tree import FusionNode, FusionTree, Leaf, TreeNode


@dataclass(frozen=True, order=True)
class PhysicalBond:
    """One undirected exchange bond in row-major physical site labels."""

    shell: str
    first: int
    second: int
    coupling: float = 1.0

    def __post_init__(self) -> None:
        if not self.shell:
            raise ValueError("bond shell must be non-empty")
        if (
            isinstance(self.first, bool)
            or isinstance(self.second, bool)
            or not isinstance(self.first, int)
            or not isinstance(self.second, int)
            or self.first < 0
            or self.second < 0
            or self.first == self.second
        ):
            raise ValueError("bond endpoints must be distinct non-negative integers")
        if not np.isfinite(self.coupling):
            raise ValueError("bond coupling must be finite")


@dataclass(frozen=True)
class NodeBlock:
    """Real-space block carried by one tree node."""

    path: tuple[int, ...]
    depth: int
    is_leaf: bool
    leaf_positions: tuple[int, ...]
    physical_sites: tuple[int, ...]


@dataclass(frozen=True)
class ClosureMetric:
    """Minimal tree closure of two or more physical sites."""

    physical_sites: tuple[int, ...]
    lca_path: tuple[int, ...]
    lca_depth: int
    closure_block_size: int
    steiner_edge_count: int
    pairwise_path_sum: int
    tree_path_diameter: int


def _validated_order(tree: FusionTree, physical_order: Sequence[int]) -> tuple[int, ...]:
    order = tuple(int(site) for site in physical_order)
    if len(order) != tree.n_leaves or sorted(order) != list(range(tree.n_leaves)):
        raise ValueError("physical_order must be a permutation of all physical sites")
    return order


def tree_node_blocks(
    tree: FusionTree, physical_order: Sequence[int]
) -> tuple[NodeBlock, ...]:
    """Return every node's physical block in deterministic preorder."""

    order = _validated_order(tree, physical_order)
    result: list[NodeBlock] = []

    def visit(node: TreeNode, path: tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(node, Leaf):
            positions = (node.index,)
        else:
            positions = visit(node.left, path + (0,)) + visit(
                node.right, path + (1,)
            )
        result.append(
            NodeBlock(
                path=path,
                depth=len(path),
                is_leaf=isinstance(node, Leaf),
                leaf_positions=positions,
                physical_sites=tuple(sorted(order[position] for position in positions)),
            )
        )
        return positions

    visit(tree.root, ())
    return tuple(sorted(result, key=lambda item: (item.path, item.is_leaf)))


def _leaf_paths(tree: FusionTree) -> dict[int, tuple[int, ...]]:
    result: dict[int, tuple[int, ...]] = {}

    def visit(node: TreeNode, path: tuple[int, ...]) -> None:
        if isinstance(node, Leaf):
            result[node.index] = path
            return
        visit(node.left, path + (0,))
        visit(node.right, path + (1,))

    visit(tree.root, ())
    return result


def _common_prefix(paths: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    if not paths:
        raise ValueError("at least one path is required")
    length = min(map(len, paths))
    end = 0
    while end < length and len({path[end] for path in paths}) == 1:
        end += 1
    return paths[0][:end]


def closure_metric(
    tree: FusionTree,
    physical_order: Sequence[int],
    physical_sites: Iterable[int],
) -> ClosureMetric:
    """Return LCA-block and minimal Steiner-subtree metrics for physical sites."""

    order = _validated_order(tree, physical_order)
    sites = tuple(sorted(set(int(site) for site in physical_sites)))
    if len(sites) < 2:
        raise ValueError("closure metrics require at least two distinct sites")
    if sites[0] < 0 or sites[-1] >= tree.n_leaves:
        raise ValueError("physical site lies outside the tree")
    physical_to_leaf = {site: leaf for leaf, site in enumerate(order)}
    paths_by_leaf = _leaf_paths(tree)
    paths = tuple(paths_by_leaf[physical_to_leaf[site]] for site in sites)
    lca = _common_prefix(paths)
    block_lookup = {block.path: block for block in tree_node_blocks(tree, order)}

    # A tree edge is identified by the path of its child endpoint.  The union
    # of the leaf-to-LCA paths is therefore the exact minimal Steiner subtree.
    steiner_edges = {
        path[:depth]
        for path in paths
        for depth in range(len(lca) + 1, len(path) + 1)
    }
    pairwise = []
    for first in range(len(paths)):
        for second in range(first + 1, len(paths)):
            common = len(_common_prefix((paths[first], paths[second])))
            pairwise.append(len(paths[first]) + len(paths[second]) - 2 * common)
    return ClosureMetric(
        physical_sites=sites,
        lca_path=lca,
        lca_depth=len(lca),
        closure_block_size=len(block_lookup[lca].physical_sites),
        steiner_edge_count=len(steiner_edges),
        pairwise_path_sum=sum(pairwise),
        tree_path_diameter=max(pairwise),
    )


def square_obc_exchange_catalog(rows: int, columns: int) -> tuple[PhysicalBond, ...]:
    """Return unweighted J1 then J2 bonds of an open square lattice."""

    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    result: list[PhysicalBond] = []
    for row in range(rows):
        for column in range(columns):
            site = row * columns + column
            if column + 1 < columns:
                result.append(PhysicalBond("J1", site, site + 1))
            if row + 1 < rows:
                result.append(PhysicalBond("J1", site, site + columns))
    for row in range(rows - 1):
        for column in range(columns - 1):
            upper_left = row * columns + column
            result.append(PhysicalBond("J2", upper_left, upper_left + columns + 1))
            result.append(PhysicalBond("J2", upper_left + 1, upper_left + columns))
    return tuple(
        sorted(result, key=lambda bond: (bond.shell, bond.first, bond.second))
    )


def square_d4_permutations(linear_size: int) -> tuple[tuple[int, ...], ...]:
    """Return the eight active D4 maps in row-major physical labels."""

    if isinstance(linear_size, bool) or not isinstance(linear_size, int):
        raise TypeError("linear_size must be an integer")
    if linear_size < 2:
        raise ValueError("linear_size must be at least two")

    def site(x: int, y: int) -> int:
        return y * linear_size + x

    maps: list[tuple[int, ...]] = []
    for reflected in (False, True):
        for rotations in range(4):
            image = []
            for old_y in range(linear_size):
                for old_x in range(linear_size):
                    x, y = old_x, old_y
                    if reflected:
                        x = linear_size - 1 - x
                    for _ in range(rotations):
                        x, y = linear_size - 1 - y, x
                    image.append(site(x, y))
            maps.append(tuple(image))
    if len(set(maps)) != 8:
        raise RuntimeError("D4 construction did not produce eight distinct maps")
    return tuple(maps)


def _bond_lookup(bonds: Sequence[PhysicalBond]) -> dict[tuple[str, int, int], int]:
    result: dict[tuple[str, int, int], int] = {}
    for index, bond in enumerate(bonds):
        first, second = sorted((bond.first, bond.second))
        key = (bond.shell, first, second)
        if key in result:
            raise ValueError("bond catalogue contains a duplicate shell/pair")
        result[key] = index
    return result


def bond_orbits(
    bonds: Sequence[PhysicalBond], permutations: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    """Partition bonds into point-group orbits without multiplicity weighting."""

    lookup = _bond_lookup(bonds)
    remaining = set(range(len(bonds)))
    result: list[tuple[int, ...]] = []
    while remaining:
        representative = min(remaining)
        bond = bonds[representative]
        members = set()
        for permutation in permutations:
            endpoints = sorted((permutation[bond.first], permutation[bond.second]))
            key = (bond.shell, endpoints[0], endpoints[1])
            if key not in lookup:
                raise ValueError("point-group image is absent from the bond catalogue")
            members.add(lookup[key])
        orbit = tuple(sorted(members))
        remaining.difference_update(orbit)
        result.append(orbit)
    if sorted(index for orbit in result for index in orbit) != list(range(len(bonds))):
        raise RuntimeError("bond orbits do not partition the catalogue")
    return tuple(result)


def connected_exchange_paths(
    bonds: Sequence[PhysicalBond],
) -> tuple[tuple[int, int], ...]:
    """Return ordered distinct bond pairs sharing at least one physical site."""

    return tuple(
        (first, second)
        for first, bond_first in enumerate(bonds)
        for second, bond_second in enumerate(bonds)
        if first != second
        and {bond_first.first, bond_first.second}
        & {bond_second.first, bond_second.second}
    )


def connected_path_orbits(
    bonds: Sequence[PhysicalBond],
    paths: Sequence[tuple[int, int]],
    permutations: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Partition ordered connected K2 paths under simultaneous point-group action."""

    lookup = _bond_lookup(bonds)
    valid = set(paths)
    remaining = set(paths)
    result: list[tuple[tuple[int, int], ...]] = []
    while remaining:
        representative = min(remaining)
        members = set()
        for permutation in permutations:
            transformed = []
            for bond_index in representative:
                bond = bonds[bond_index]
                endpoints = sorted((permutation[bond.first], permutation[bond.second]))
                transformed.append(lookup[(bond.shell, endpoints[0], endpoints[1])])
            pair = (transformed[0], transformed[1])
            if pair not in valid:
                raise RuntimeError("point-group image left the connected-path catalogue")
            members.add(pair)
        orbit = tuple(sorted(members))
        remaining.difference_update(orbit)
        result.append(orbit)
    if sorted(pair for orbit in result for pair in orbit) != sorted(paths):
        raise RuntimeError("connected-path orbits do not partition the catalogue")
    return tuple(result)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return float(sum((value - mean) ** 2 for value in values) / len(values))


def orbit_equal_exchange_scores(
    tree: FusionTree,
    physical_order: Sequence[int],
    bonds: Sequence[PhysicalBond],
    permutations: Sequence[Sequence[int]],
) -> dict[str, object]:
    """Return D4-orbit-equal K1 and connected-K2 locality scores.

    Every orbit contributes once, irrespective of its number of members.  K1
    is resolved by J1/J2 shell and K2 by ordered shell sequence.  The scores
    are structural and contain no Hamiltonian ratio or state-dependent weight.
    """

    first_orbits = bond_orbits(bonds, permutations)
    paths = connected_exchange_paths(bonds)
    second_orbits = connected_path_orbits(bonds, paths, permutations)
    k1_rows = []
    for orbit in first_orbits:
        metrics = [
            closure_metric(
                tree, physical_order, (bonds[index].first, bonds[index].second)
            )
            for index in orbit
        ]
        k1_rows.append(
            {
                "shell": bonds[orbit[0]].shell,
                "size": len(orbit),
                "closure_block_size_mean": _mean(
                    [metric.closure_block_size for metric in metrics]
                ),
                "steiner_edge_count_mean": _mean(
                    [metric.steiner_edge_count for metric in metrics]
                ),
                "lca_depth_mean": _mean([metric.lca_depth for metric in metrics]),
                "lca_depth_variance": _variance(
                    [metric.lca_depth for metric in metrics]
                ),
            }
        )
    k2_rows = []
    for orbit in second_orbits:
        metrics = []
        for first, second in orbit:
            sites = {
                bonds[first].first,
                bonds[first].second,
                bonds[second].first,
                bonds[second].second,
            }
            metrics.append(closure_metric(tree, physical_order, sites))
        k2_rows.append(
            {
                "shell_sequence": [
                    bonds[orbit[0][0]].shell,
                    bonds[orbit[0][1]].shell,
                ],
                "size": len(orbit),
                "closure_block_size_mean": _mean(
                    [metric.closure_block_size for metric in metrics]
                ),
                "steiner_edge_count_mean": _mean(
                    [metric.steiner_edge_count for metric in metrics]
                ),
                "tree_path_diameter_mean": _mean(
                    [metric.tree_path_diameter for metric in metrics]
                ),
                "lca_depth_mean": _mean([metric.lca_depth for metric in metrics]),
                "lca_depth_variance": _variance(
                    [metric.lca_depth for metric in metrics]
                ),
            }
        )

    def grouped(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, object]:
        groups: dict[str, list[Mapping[str, object]]] = {}
        for row in rows:
            label_value = row[key]
            label = (
                "->".join(str(value) for value in label_value)
                if isinstance(label_value, list)
                else str(label_value)
            )
            groups.setdefault(label, []).append(row)
        result = {}
        for label, members in groups.items():
            result[label] = {
                "orbit_count": len(members),
                "closure_block_size_mean": _mean(
                    [float(row["closure_block_size_mean"]) for row in members]
                ),
                "steiner_edge_count_mean": _mean(
                    [float(row["steiner_edge_count_mean"]) for row in members]
                ),
                "lca_depth_variance_mean": _mean(
                    [float(row["lca_depth_variance"]) for row in members]
                ),
                "lca_depth_variance_peak": max(
                    float(row["lca_depth_variance"]) for row in members
                ),
            }
        return result

    all_rows = [*k1_rows, *k2_rows]
    return {
        "bond_orbit_count": len(first_orbits),
        "connected_path_count": len(paths),
        "connected_path_orbit_count": len(second_orbits),
        "K1": {"by_shell": grouped(k1_rows, "shell"), "orbits": k1_rows},
        "K2_connected": {
            "by_shell_sequence": grouped(k2_rows, "shell_sequence"),
            "orbits": k2_rows,
        },
        "aggregate": {
            "k1_orbit_equal_closure_block_size": _mean(
                [float(row["closure_block_size_mean"]) for row in k1_rows]
            ),
            "k1_orbit_equal_steiner_edges": _mean(
                [float(row["steiner_edge_count_mean"]) for row in k1_rows]
            ),
            "k2_orbit_equal_closure_block_size": _mean(
                [float(row["closure_block_size_mean"]) for row in k2_rows]
            ),
            "k2_orbit_equal_steiner_edges": _mean(
                [float(row["steiner_edge_count_mean"]) for row in k2_rows]
            ),
            "orbit_closure_depth_variance_mean": _mean(
                [float(row["lca_depth_variance"]) for row in all_rows]
            ),
            "orbit_closure_depth_variance_peak": max(
                (float(row["lca_depth_variance"]) for row in all_rows), default=0.0
            ),
        },
    }


def edge_cut_catalog(
    tree: FusionTree,
    physical_order: Sequence[int],
    bonds: Sequence[PhysicalBond],
) -> tuple[dict[str, object], ...]:
    """Return every edge-indexed real-space cut, including leaf edges."""

    blocks = tree_node_blocks(tree, physical_order)
    n_sites = tree.n_leaves
    result = []
    for block in blocks:
        if not block.path:
            continue
        inside = set(block.physical_sites)
        crossing = [
            index
            for index, bond in enumerate(bonds)
            if (bond.first in inside) != (bond.second in inside)
        ]
        shell_l1 = {
            shell: float(
                sum(abs(bonds[index].coupling) for index in crossing if bonds[index].shell == shell)
            )
            for shell in sorted({bond.shell for bond in bonds})
        }
        result.append(
            {
                "path": list(block.path),
                "depth": block.depth,
                "is_leaf": block.is_leaf,
                "leaf_positions": list(block.leaf_positions),
                "physical_sites": list(block.physical_sites),
                "complement_sites": sorted(set(range(n_sites)) - inside),
                "crossing_bond_indices": crossing,
                "crossing_l1_by_shell": shell_l1,
                "crossing_l1_total": float(sum(shell_l1.values())),
            }
        )
    return tuple(result)


def cut_l1_distance_identity(
    tree: FusionTree,
    physical_order: Sequence[int],
    bonds: Sequence[PhysicalBond],
) -> dict[str, float]:
    """Audit the exact cut-L1/bond-distance identity with no half factor."""

    cuts = edge_cut_catalog(tree, physical_order, bonds)
    cut_sum = float(sum(float(cut["crossing_l1_total"]) for cut in cuts))
    pair_sum = 0.0
    for bond in bonds:
        metric = closure_metric(tree, physical_order, (bond.first, bond.second))
        pair_sum += abs(bond.coupling) * metric.steiner_edge_count
    return {
        "edge_indexed_cut_l1_sum": cut_sum,
        "weighted_tree_bond_distance": float(pair_sum),
        "absolute_error": abs(cut_sum - pair_sum),
        "multiplicity_factor": 1.0,
    }


def _effective_rank(singular_values: np.ndarray) -> float:
    weights = np.square(np.asarray(singular_values, dtype=np.float64))
    total = float(weights.sum())
    if total == 0.0:
        return 0.0
    probabilities = weights[weights > 0.0] / total
    return float(exp(-sum(float(p) * log(float(p)) for p in probabilities)))


def crossing_coupling_svd(
    tree: FusionTree,
    physical_order: Sequence[int],
    bonds: Sequence[PhysicalBond],
    *,
    relative_tolerance: float = 1.0e-12,
) -> dict[str, object]:
    """Resolve boundary-coupling channel ranks independently for each shell.

    The matrix for a block has inside sites as rows, outside sites as columns,
    and bare exchange couplings as entries.  Statistics use internal non-root
    fusion blocks; leaf cuts, whose rank is trivially at most one, are listed
    by :func:`edge_cut_catalog` but excluded from the summary.
    """

    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    order = _validated_order(tree, physical_order)
    shells = sorted({bond.shell for bond in bonds})
    rows = []
    for block in tree_node_blocks(tree, order):
        if not block.path or block.is_leaf:
            continue
        inside = list(block.physical_sites)
        outside = sorted(set(range(tree.n_leaves)) - set(inside))
        records: dict[str, object] = {}
        for shell in shells:
            matrix = np.zeros((len(inside), len(outside)), dtype=np.float64)
            inside_position = {site: index for index, site in enumerate(inside)}
            outside_position = {site: index for index, site in enumerate(outside)}
            for bond in bonds:
                if bond.shell != shell:
                    continue
                if bond.first in inside_position and bond.second in outside_position:
                    matrix[inside_position[bond.first], outside_position[bond.second]] += bond.coupling
                elif bond.second in inside_position and bond.first in outside_position:
                    matrix[inside_position[bond.second], outside_position[bond.first]] += bond.coupling
            singular = np.linalg.svd(matrix, compute_uv=False)
            threshold = relative_tolerance * singular[0] if len(singular) else 0.0
            rank = int(np.sum(singular > threshold)) if len(singular) and singular[0] > 0 else 0
            records[shell] = {
                "rank": rank,
                "effective_rank": _effective_rank(singular),
                "frobenius_norm": float(np.linalg.norm(singular)),
                "singular_values": [float(value) for value in singular if value > threshold],
            }
        rows.append(
            {
                "path": list(block.path),
                "depth": block.depth,
                "block_size": len(inside),
                "by_shell": records,
            }
        )
    summary = {}
    for shell in shells:
        active = [row["by_shell"][shell] for row in rows if row["by_shell"][shell]["rank"] > 0]
        summary[shell] = {
            "active_cut_count": len(active),
            "rank_mean": _mean([float(record["rank"]) for record in active]),
            "rank_peak": max((int(record["rank"]) for record in active), default=0),
            "effective_rank_mean": _mean(
                [float(record["effective_rank"]) for record in active]
            ),
            "effective_rank_peak": max(
                (float(record["effective_rank"]) for record in active), default=0.0
            ),
        }
    return {"relative_tolerance": relative_tolerance, "by_shell": summary, "cuts": rows}


def _canonical_bipartition(block: Iterable[int], n_sites: int) -> tuple[int, ...]:
    first = tuple(sorted(set(block)))
    second = tuple(sorted(set(range(n_sites)) - set(first)))
    return min(first, second, key=lambda sites: (len(sites), sites))


def _partition_distance(first: Sequence[int], second: Sequence[int], n_sites: int) -> int:
    a, b = set(first), set(second)
    return min(len(a ^ b), len(a ^ (set(range(n_sites)) - b)))


def d4_bipartition_defect(
    tree: FusionTree,
    physical_order: Sequence[int],
    permutations: Sequence[Sequence[int]],
) -> dict[str, float | int]:
    """Measure failure of nontrivial fusion cuts to close under D4."""

    n_sites = tree.n_leaves
    blocks = [
        block
        for block in tree_node_blocks(tree, physical_order)
        if block.path and not block.is_leaf and len(block.physical_sites) < n_sites
    ]
    partitions = {
        _canonical_bipartition(block.physical_sites, n_sites) for block in blocks
    }
    mismatches = []
    missing = 0
    for partition in sorted(partitions, key=lambda value: (len(value), value)):
        for permutation in permutations:
            transformed = _canonical_bipartition(
                (permutation[site] for site in partition), n_sites
            )
            if transformed not in partitions:
                missing += 1
            mismatches.append(
                min(
                    _partition_distance(transformed, candidate, n_sites)
                    for candidate in partitions
                )
            )
    comparisons = len(mismatches)
    return {
        "distinct_nontrivial_bipartitions": len(partitions),
        "group_comparisons": comparisons,
        "exact_missing_fraction": float(missing / comparisons) if comparisons else 0.0,
        "mean_site_mismatch_fraction": _mean(mismatches) / n_sites,
        "peak_site_mismatch_fraction": (max(mismatches) / n_sites if mismatches else 0.0),
    }


def spin_half_irrep_multiplicity(n_spins: int, q: int) -> int:
    """Multiplicity of doubled total spin ``q`` in ``(1/2)**n_spins``."""

    if isinstance(n_spins, bool) or not isinstance(n_spins, int) or n_spins < 0:
        raise ValueError("n_spins must be a non-negative integer")
    if isinstance(q, bool) or not isinstance(q, int) or q < 0:
        raise ValueError("q must be a non-negative doubled integer")
    if q > n_spins or (n_spins - q) % 2:
        return 0
    lower = (n_spins - q) // 2
    return comb(n_spins, lower) - (comb(n_spins, lower - 1) if lower else 0)


def spin_half_mixed_q_prior(n_spins: int) -> dict[int, float]:
    """Total-q law of ``n_spins`` maximally mixed spin-half boundary legs.

    ``p_n(q) = mult(n,q) (q+1) / 2**n`` in doubled-spin convention.
    """

    if isinstance(n_spins, bool) or not isinstance(n_spins, int) or n_spins < 0:
        raise ValueError("n_spins must be a non-negative integer")
    return {
        q: spin_half_irrep_multiplicity(n_spins, q) * (q + 1) / (2**n_spins)
        for q in range(n_spins % 2, n_spins + 1, 2)
    }


def q_casimir(q: int) -> float:
    """Return ``j(j+1)=q(q+2)/4`` for a doubled spin label."""

    if isinstance(q, bool) or not isinstance(q, int) or q < 0:
        raise ValueError("q must be a non-negative doubled integer")
    return q * (q + 2) / 4.0


def enumerate_short_range_dimer_coverings(
    n_sites: int, bonds: Sequence[PhysicalBond]
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Enumerate perfect matchings of the supplied short-range bond graph."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, int) or n_sites < 0:
        raise ValueError("n_sites must be a non-negative integer")
    if n_sites % 2:
        return ()
    adjacency = {site: set() for site in range(n_sites)}
    for bond in bonds:
        if bond.first >= n_sites or bond.second >= n_sites:
            raise ValueError("dimer bond endpoint lies outside the lattice")
        adjacency[bond.first].add(bond.second)
        adjacency[bond.second].add(bond.first)
    result: list[tuple[tuple[int, int], ...]] = []

    def solve(remaining: frozenset[int], chosen: tuple[tuple[int, int], ...]) -> None:
        if not remaining:
            result.append(tuple(sorted(chosen)))
            return
        first = min(remaining)
        for second in sorted(adjacency[first] & remaining):
            solve(
                remaining - {first, second},
                chosen + ((min(first, second), max(first, second)),),
            )

    solve(frozenset(range(n_sites)), ())
    return tuple(sorted(set(result)))


def dimer_crossing_count(
    block: Iterable[int], covering: Sequence[tuple[int, int]]
) -> int:
    """Count singlets with exactly one endpoint in a real-space block."""

    inside = set(int(site) for site in block)
    return sum((first in inside) != (second in inside) for first, second in covering)


def short_range_rvb_q_prior(
    block: Iterable[int],
    coverings: Sequence[Sequence[tuple[int, int]]],
    *,
    weights: Sequence[float] | None = None,
) -> dict[int, float]:
    """Mixture q prior for a classical short-range valence-bond ensemble.

    For a covering ``D``, internal singlets carry q=0 while each of its
    ``n_cross(D,B)`` cut singlets contributes one maximally mixed spin-half
    boundary leg.  Thus

    ``p_B(q)=sum_D w_D p_{n_cross(D,B)}(q) / sum_D w_D``.

    This is a structural RVB prior, not the reduced density matrix of an
    equal-amplitude *coherent* RVB wavefunction (whose coverings overlap).
    """

    if not coverings:
        raise ValueError("at least one dimer covering is required")
    values = (
        np.ones(len(coverings), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if values.shape != (len(coverings),) or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("RVB weights must be a finite non-negative vector")
    normalization = float(values.sum())
    if normalization <= 0.0:
        raise ValueError("RVB weights must have positive total")
    prior: dict[int, float] = {}
    for weight, covering in zip(values, coverings, strict=True):
        conditional = spin_half_mixed_q_prior(dimer_crossing_count(block, covering))
        for q, probability in conditional.items():
            prior[q] = prior.get(q, 0.0) + float(weight) * probability / normalization
    return dict(sorted(prior.items()))


def rvb_tree_prior_metrics(
    tree: FusionTree,
    physical_order: Sequence[int],
    coverings: Sequence[Sequence[tuple[int, int]]],
) -> dict[str, object]:
    """Evaluate short-range-dimer q priors on every internal fusion block."""

    rows = []
    for block in tree_node_blocks(tree, physical_order):
        if block.is_leaf:
            continue
        counts = [dimer_crossing_count(block.physical_sites, covering) for covering in coverings]
        prior = short_range_rvb_q_prior(block.physical_sites, coverings)
        entropy = -sum(p * log2(p) for p in prior.values() if p > 0.0)
        casimir = sum(probability * q_casimir(q) for q, probability in prior.items())
        expected_crossings = _mean(counts)
        rows.append(
            {
                "path": list(block.path),
                "depth": block.depth,
                "block_size": len(block.physical_sites),
                "physical_sites": list(block.physical_sites),
                "crossing_count_mean": expected_crossings,
                "crossing_count_values": sorted(set(counts)),
                "q_prior": {str(q): probability for q, probability in prior.items()},
                "q_entropy_bits": entropy,
                "normalization_error": abs(sum(prior.values()) - 1.0),
                "casimir": casimir,
                "casimir_identity_error": abs(casimir - 0.75 * expected_crossings),
            }
        )
    non_root = [row for row in rows if row["path"]]
    entropy_sum = sum(float(row["q_entropy_bits"]) for row in non_root)
    return {
        "covering_count": len(coverings),
        "formula": "p_B(q)=sum_D w_D mult(n_D,q)(q+1)/2^n_D / sum_D w_D",
        "casimir_identity": "E[S_B^2]=E[q(q+2)/4]=3 E[n_cross]/4",
        "maximum_normalization_error": max(
            (float(row["normalization_error"]) for row in rows), default=0.0
        ),
        "maximum_casimir_identity_error": max(
            (float(row["casimir_identity_error"]) for row in rows), default=0.0
        ),
        "q_entropy_mean_bits": _mean(
            [float(row["q_entropy_bits"]) for row in non_root]
        ),
        "q_entropy_peak_bits": max(
            (float(row["q_entropy_bits"]) for row in non_root), default=0.0
        ),
        "q_entropy_weighted_depth": (
            sum((int(row["depth"]) + 1) * float(row["q_entropy_bits"]) for row in non_root)
            / entropy_sum
            if entropy_sum > 0.0
            else 0.0
        ),
        "nodes": rows,
    }


def tree_nested(node: TreeNode) -> int | list[object]:
    """Return a JSON-compatible ordered tree description."""

    if isinstance(node, Leaf):
        return node.index
    assert isinstance(node, FusionNode)
    return [tree_nested(node.left), tree_nested(node.right)]


def tree_coordinate_fingerprint(
    tree: FusionTree,
    physical_order: Sequence[int],
    *,
    selection_contract: str,
) -> str:
    """Hash a frozen tree coordinate and its state-blind selection contract."""

    payload = {
        "tree": tree_nested(tree.root),
        "physical_order": list(_validated_order(tree, physical_order)),
        "selection_contract": str(selection_contract),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()
