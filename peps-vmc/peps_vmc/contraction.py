"""Exact single-layer PEPS contraction with reusable, differentiable paths.

The virtual graph is deliberately independent of Hamiltonian connectivity.
For example, the inexpensive four-leg square torus graph can parameterize a
state of the triangular-lattice Hamiltonian.  This does not impose triangular
point-group symmetry.  There is no boundary-MPS truncation in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe

Array = Any


@dataclass(frozen=True)
class ExactContractionPlan:
    """A fixed contraction path; each site's virtual axes follow ``legs``.

    ``edges`` are undirected virtual edges.  Their order determines the virtual
    leg order at each site.  Local tensors have shape ``(2, D, ..., D)`` with
    physical bit zero denoting doubled magnetic label ``m=-1``.
    """

    n_sites: int
    edges: tuple[tuple[int, int], ...]
    dimension: int
    legs: tuple[tuple[int, ...], ...]
    equation: str
    path: tuple[tuple[int, ...], ...]
    estimated_flops: int
    largest_intermediate: int

    def amplitude(self, tensors: Sequence[Array] | Array, spins: Array) -> Array:
        """Contract one physical configuration exactly (pure JAX)."""
        selected = [tensors[i][spins[i]] for i in range(self.n_sites)]
        return oe.contract(
            self.equation, *selected, optimize=list(self.path), backend="jax"
        )

    def numpy_amplitude(
        self, tensors: Sequence[np.ndarray] | np.ndarray, spins: np.ndarray
    ) -> complex:
        """Independent NumPy backend using the same mathematical network."""
        return complex(
            oe.contract(
                self.equation,
                *(tensors[i][int(spins[i])] for i in range(self.n_sites)),
                optimize=list(self.path),
                backend="numpy",
            )
        )

    def metadata(self) -> dict[str, Any]:
        """Return path costs, not a measured wall-time or peak-memory claim."""
        return {
            "method": "exact_single_layer_fixed_path",
            "n_sites": self.n_sites,
            "virtual_edges": [list(e) for e in self.edges],
            "dimension": self.dimension,
            "leg_edge_ids": [list(x) for x in self.legs],
            "path": [list(x) for x in self.path],
            "estimated_flops_per_amplitude": self.estimated_flops,
            "largest_intermediate_elements": self.largest_intermediate,
            "largest_complex128_intermediate_bytes": 16 * self.largest_intermediate,
            "approximate_contraction": False,
        }


def make_exact_plan(
    n_sites: int,
    edges: Sequence[tuple[int, int]],
    dimension: int,
    *,
    optimize: str = "greedy",
    max_intermediate_elements: int = 2_000_000,
    leg_order: Sequence[Sequence[int]] | None = None,
) -> ExactContractionPlan:
    """Plan an exact network without allocating its dense intermediates.

    The explicit size guard prevents inadvertently executing a native six-leg
    ``6x6`` torus contraction.  Callers can plan that graph with a larger guard
    to inspect its prohibitive cost, but should not allocate the tensors.
    """
    if n_sites < 2 or dimension < 1 or max_intermediate_elements < 1:
        raise ValueError("positive dimension/size guard and at least two sites required")
    normalized: list[tuple[int, int]] = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("each virtual edge has exactly two endpoints")
        a, b = int(edge[0]), int(edge[1])
        if a == b or not (0 <= a < n_sites and 0 <= b < n_sites):
            raise ValueError("virtual edge has invalid endpoints")
        normalized.append((min(a, b), max(a, b)))
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate virtual edges are not supported")
    legs: list[list[int]] = [[] for _ in range(n_sites)]
    for label, (a, b) in enumerate(normalized):
        legs[a].append(label)
        legs[b].append(label)
    if any(not row for row in legs):
        raise ValueError("isolated physical sites are not supported")
    if leg_order is not None:
        ordered = [list(map(int, row)) for row in leg_order]
        if len(ordered) != n_sites or any(sorted(row) != sorted(expected)
                                         for row, expected in zip(ordered, legs)):
            raise ValueError("leg_order must reorder each site's incident edge IDs")
        legs = ordered
    equation = ",".join("".join(oe.get_symbol(i) for i in row) for row in legs) + "->"
    shapes = [(dimension,) * len(row) for row in legs]
    path, info = oe.contract_path(equation, *shapes, shapes=True, optimize=optimize)
    largest = int(info.largest_intermediate)
    if largest > max_intermediate_elements:
        raise MemoryError(
            f"exact contraction needs {largest} intermediate elements, "
            f"above guard {max_intermediate_elements}"
        )
    return ExactContractionPlan(
        n_sites,
        tuple(normalized),
        int(dimension),
        tuple(tuple(row) for row in legs),
        equation,
        tuple(tuple(step) for step in path),
        int(info.opt_cost),
        largest,
    )


def square_torus_edges(length_x: int, length_y: int) -> tuple[tuple[int, int], ...]:
    """Four-leg virtual torus, site ``y * length_x + x``.

    Length-two cycles would require parallel bonds, so use lengths at least
    three; smaller exact references can instead provide their edges directly.
    """
    if length_x < 3 or length_y < 3:
        raise ValueError("simple square torus requires both lengths >= 3")
    edges: list[tuple[int, int]] = []
    for y in range(length_y):
        for x in range(length_x):
            a = y * length_x + x
            for b in (y * length_x + (x + 1) % length_x,
                      ((y + 1) % length_y) * length_x + x):
                edges.append((min(a, b), max(a, b)))
    return tuple(edges)


def make_square_torus_plan(
    rows: int,
    cols: int,
    dimension: int,
    *,
    optimize: str = "greedy",
    max_intermediate_elements: int = 2_000_000,
) -> ExactContractionPlan:
    """Plan a torus with physical axes ``(s, left, right, up, down)``.

    Site index is ``y * cols + x``.  Bond metrics must already be absorbed
    into the local right/down legs; this contraction joins axes by deltas.
    """
    edges = square_torus_edges(cols, rows)
    lookup = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    leg_order = []
    for y in range(rows):
        for x in range(cols):
            site = y * cols + x
            neighbors = [y * cols + (x - 1) % cols,
                         y * cols + (x + 1) % cols,
                         ((y - 1) % rows) * cols + x,
                         ((y + 1) % rows) * cols + x]
            leg_order.append([lookup[tuple(sorted((site, j)))] for j in neighbors])
    return make_exact_plan(
        rows * cols, edges, dimension, optimize=optimize,
        max_intermediate_elements=max_intermediate_elements, leg_order=leg_order,
    )


def make_amplitude_batch(
    plan: ExactContractionPlan, *, chunk_size: int = 1
) -> Callable[[Sequence[Array] | Array, Array], Array]:
    """JAX batch amplitude with bounded live intermediates.

    ``lax.map`` limits memory to ``chunk_size`` configurations.  A caller may
    JIT this function together with the reduced-tensor decoder.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    def amplitudes(tensors: Sequence[Array] | Array, configurations: Array) -> Array:
        return jax.lax.map(
            lambda row: plan.amplitude(tensors, row), configurations,
            batch_size=chunk_size,
        )

    return amplitudes


@dataclass(frozen=True)
class _ReuseNode:
    left: int
    right: int
    equation: str
    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]


@dataclass(frozen=True)
class SwapQueryPlan:
    """Exact exchange-query DAG reusing unaffected contraction branches.

    Query zero is the parent.  Query ``k+1`` swaps ``swaps[k]``.  Distinct
    *structural* substitutions are evaluated once per intermediate.  This
    cache is valid for arbitrary parameters and configurations; no stale
    numerical environment or state-dependent approximation is stored.
    """

    contraction: ExactContractionPlan
    swaps: tuple[tuple[int, int], ...]
    leaf_sources: tuple[tuple[int, ...], ...]
    nodes: tuple[_ReuseNode, ...]
    root_indices: tuple[int, ...]
    naive_flops: int
    reused_flops: int
    largest_batched_intermediate: int

    def amplitudes(self, tensors: Sequence[Array] | Array, spins: Array) -> Array:
        """Return parent and all swaps without changing the physical state."""
        values = [
            tensors[i][spins[jnp.asarray(sources, dtype=jnp.int32)]]
            for i, sources in enumerate(self.leaf_sources)
        ]
        for node in self.nodes:
            left = values[node.left][jnp.asarray(node.left_indices, dtype=jnp.int32)]
            right = values[node.right][jnp.asarray(node.right_indices, dtype=jnp.int32)]
            values.append(jnp.einsum(node.equation, left, right, optimize=False))
        return values[-1][jnp.asarray(self.root_indices, dtype=jnp.int32)]

    def metadata(self) -> dict[str, Any]:
        return {
            "query_count": len(self.swaps) + 1,
            "naive_estimated_flops": self.naive_flops,
            "reused_estimated_flops": self.reused_flops,
            "estimated_flop_reduction": self.naive_flops / self.reused_flops,
            "largest_batched_intermediate_elements": self.largest_batched_intermediate,
            "reuse_is_exact": True,
        }


def make_swap_query_plan(
    contraction: ExactContractionPlan,
    swaps: Sequence[tuple[int, int]],
    *,
    max_intermediate_elements: int = 100_000_000,
) -> SwapQueryPlan:
    """Reuse a binary contraction tree for a fixed Hamiltonian's exchanges."""
    n = contraction.n_sites
    permutations = [tuple(range(n))]
    checked_swaps = []
    for a, b in swaps:
        a, b = int(a), int(b)
        if a == b or not (0 <= a < n and 0 <= b < n):
            raise ValueError("swap has invalid endpoints")
        perm = list(range(n))
        perm[a], perm[b] = b, a
        permutations.append(tuple(perm))
        checked_swaps.append((a, b))
    sites = [(i,) for i in range(n)]
    labels = list(contraction.legs)
    # Signatures are source-site tuples in each node's fixed site order.
    signatures = [tuple(dict.fromkeys((p[i],) for p in permutations)) for i in range(n)]
    leaf_sources = tuple(tuple(x[0] for x in row) for row in signatures)
    active = list(range(n))
    batch_symbol = oe.get_symbol(len(contraction.edges))
    nodes = []
    total_flops = 0
    peak = max(len(s) * contraction.dimension ** len(l) for s, l in zip(signatures, labels))
    for step in contraction.path:
        if len(step) != 2:
            raise ValueError("exchange reuse currently needs a binary contraction path")
        pos_left, pos_right = step
        left, right = active[pos_left], active[pos_right]
        all_sites = sites[left] + sites[right]
        out_labels = tuple(x for x in labels[left] + labels[right]
                           if (labels[left] + labels[right]).count(x) == 1)
        out_signatures = tuple(dict.fromkeys(tuple(p[i] for i in all_sites) for p in permutations))
        left_lookup = {s: i for i, s in enumerate(signatures[left])}
        right_lookup = {s: i for i, s in enumerate(signatures[right])}
        nl = len(sites[left])
        li = tuple(left_lookup[s[:nl]] for s in out_signatures)
        ri = tuple(right_lookup[s[nl:]] for s in out_signatures)
        sub_left = "".join(oe.get_symbol(x) for x in labels[left])
        sub_right = "".join(oe.get_symbol(x) for x in labels[right])
        sub_out = "".join(oe.get_symbol(x) for x in out_labels)
        equation = f"{batch_symbol}{sub_left},{batch_symbol}{sub_right}->{batch_symbol}{sub_out}"
        nodes.append(_ReuseNode(left, right, equation, li, ri))
        unique_labels = set(labels[left]) | set(labels[right])
        total_flops += 2 * len(out_signatures) * contraction.dimension ** len(unique_labels)
        peak = max(peak, len(out_signatures) * contraction.dimension ** len(out_labels))
        sites.append(all_sites)
        labels.append(out_labels)
        signatures.append(out_signatures)
        for index in sorted(step, reverse=True):
            active.pop(index)
        active.append(len(sites) - 1)
    if len(active) != 1 or labels[-1]:
        raise ValueError("contraction path does not finish in a scalar")
    if peak > max_intermediate_elements:
        raise MemoryError(f"exchange reuse needs {peak} intermediate elements above guard")
    root_lookup = {s: i for i, s in enumerate(signatures[-1])}
    root_indices = tuple(root_lookup[tuple(p[i] for i in sites[-1])] for p in permutations)
    return SwapQueryPlan(
        contraction, tuple(checked_swaps), leaf_sources, tuple(nodes), root_indices,
        (len(checked_swaps) + 1) * contraction.estimated_flops, total_flops, peak,
    )


@dataclass(frozen=True)
class _ActionNode:
    left: int
    right: int
    base_equation: str
    flip_equation: str
    cross_equation: str
    flip_left: tuple[int, ...]
    flip_right: tuple[int, ...]
    cross_left: tuple[int, ...]
    cross_right: tuple[int, ...]
    cross_bonds: tuple[int, ...]
    left_has_internal: bool
    right_has_internal: bool


@dataclass(frozen=True)
class HeisenbergActionPlan:
    """Exact ``(psi, H psi)`` with one-site boundary environments.

    Each exchange bond is inserted exactly once, at its endpoints' lowest
    common contraction-tree ancestor.  A subtree retains only single-spin
    flips whose sites have a Hamiltonian neighbor outside that subtree.
    This is operator propagation through a *contraction* tree, not a change
    from PEPS to a tree wavefunction or an approximation to the Hamiltonian.
    """

    contraction: ExactContractionPlan
    bonds: tuple[tuple[int, int], ...]
    couplings: tuple[float, ...]
    nodes: tuple[_ActionNode, ...]
    leaf_has_boundary: tuple[bool, ...]
    estimated_flops: int
    largest_boundary_intermediate: int

    def products(self, tensors: Sequence[Array] | Array, spins: Array) -> Array:
        """Return ``jnp.array([psi, Hpsi])``; no division or node clipping."""
        values = []
        internal = []
        for i, needs_flip in enumerate(self.leaf_has_boundary):
            base = tensors[i][spins[i]]
            values.append(jnp.stack([base, tensors[i][1 - spins[i]]]) if needs_flip
                          else base[None])
            internal.append(jnp.zeros_like(base))
        bonds = jnp.asarray(self.bonds, dtype=jnp.int32)
        couplings = jnp.asarray(self.couplings, dtype=jnp.float64)
        different = spins[bonds[:, 0]] != spins[bonds[:, 1]]
        for node in self.nodes:
            left, right = values[node.left], values[node.right]
            base = jnp.einsum(node.base_equation, left[0], right[0], optimize=False)
            h = jnp.zeros_like(base)
            if node.left_has_internal:
                h = h + jnp.einsum(node.base_equation, internal[node.left], right[0],
                                   optimize=False)
            if node.right_has_internal:
                h = h + jnp.einsum(node.base_equation, left[0], internal[node.right],
                                   optimize=False)
            if node.cross_bonds:
                ids = jnp.asarray(node.cross_bonds, dtype=jnp.int32)
                a = left[jnp.asarray(node.cross_left, dtype=jnp.int32)]
                b = right[jnp.asarray(node.cross_right, dtype=jnp.int32)]
                weights = .5 * couplings[ids] * different[ids]
                a = a * weights.reshape((-1,) + (1,) * (a.ndim - 1))
                h = h + jnp.einsum(node.cross_equation, a, b, optimize=False)
            if node.flip_left:
                flips = jnp.einsum(
                    node.flip_equation,
                    left[jnp.asarray(node.flip_left, dtype=jnp.int32)],
                    right[jnp.asarray(node.flip_right, dtype=jnp.int32)],
                    optimize=False,
                )
                values.append(jnp.concatenate((base[None], flips), axis=0))
            else:
                values.append(base[None])
            internal.append(h)
        psi = values[-1][0]
        diagonal = .25 * jnp.sum(couplings * (1 - 2 * different.astype(jnp.int32)))
        return jnp.stack((psi, internal[-1] + diagonal * psi))

    def metadata(self) -> dict[str, Any]:
        return {
            "method": "exact_single_layer_H_boundary_operator_propagation",
            "hamiltonian_bonds": len(self.bonds),
            "estimated_flops": self.estimated_flops,
            "largest_boundary_intermediate_elements": self.largest_boundary_intermediate,
            "each_exchange_inserted_once": True,
            "approximate_contraction": False,
        }


def make_heisenberg_action(
    contraction: ExactContractionPlan,
    bonds: Sequence[tuple[int, int]],
    couplings: Sequence[float],
    *,
    max_intermediate_elements: int = 20_000_000,
) -> HeisenbergActionPlan:
    """Compile exact full-H action for ``sum J_ij S_i dot S_j``.

    Uses bit zero for doubled ``m=-1``.  Duplicate bonds are combined first;
    non-nearest Hamiltonian bonds need not be virtual PEPS edges.
    """
    if len(bonds) != len(couplings) or not bonds:
        raise ValueError("nonempty bonds and couplings must have matching lengths")
    combined: dict[tuple[int, int], float] = {}
    for (a, b), coupling in zip(bonds, couplings):
        a, b, coupling = int(a), int(b), float(coupling)
        if a == b or not (0 <= a < contraction.n_sites and 0 <= b < contraction.n_sites):
            raise ValueError("Hamiltonian bond has invalid endpoints")
        if not np.isfinite(coupling):
            raise ValueError("Hamiltonian coupling must be finite")
        key = (min(a, b), max(a, b))
        combined[key] = combined.get(key, 0.0) + coupling
    combined = {k: v for k, v in combined.items() if v != 0}
    if not combined:
        raise ValueError("Hamiltonian has no nonzero bonds")
    checked_bonds, checked_couplings = tuple(combined), tuple(combined.values())
    n = contraction.n_sites
    neighbors = [set() for _ in range(n)]
    for a, b in checked_bonds:
        neighbors[a].add(b)
        neighbors[b].add(a)
    sites = [frozenset([i]) for i in range(n)]
    labels = list(contraction.legs)
    boundaries = [(i,) if neighbors[i] else () for i in range(n)]
    has_internal = [False] * n
    active = list(range(n))
    batch = oe.get_symbol(len(contraction.edges))
    nodes = []
    total_flops = 0
    peak = max((1 + len(b)) * contraction.dimension ** len(l)
               for b, l in zip(boundaries, labels))
    for step in contraction.path:
        if len(step) != 2:
            raise ValueError("Hamiltonian propagation needs a binary contraction path")
        left, right = active[step[0]], active[step[1]]
        all_sites = sites[left] | sites[right]
        out_labels = tuple(x for x in labels[left] + labels[right]
                           if (labels[left] + labels[right]).count(x) == 1)
        boundary = tuple(sorted(i for i in all_sites if neighbors[i] - all_sites))
        left_lookup = {i: k + 1 for k, i in enumerate(boundaries[left])}
        right_lookup = {i: k + 1 for k, i in enumerate(boundaries[right])}
        flip_left = tuple(left_lookup[i] if i in sites[left] else 0 for i in boundary)
        flip_right = tuple(right_lookup[i] if i in sites[right] else 0 for i in boundary)
        cross_ids, cross_left, cross_right = [], [], []
        for k, (a, b) in enumerate(checked_bonds):
            if a in sites[left] and b in sites[right]:
                cross_ids.append(k)
                cross_left.append(left_lookup[a])
                cross_right.append(right_lookup[b])
            elif b in sites[left] and a in sites[right]:
                cross_ids.append(k)
                cross_left.append(left_lookup[b])
                cross_right.append(right_lookup[a])
        sl = "".join(oe.get_symbol(x) for x in labels[left])
        sr = "".join(oe.get_symbol(x) for x in labels[right])
        so = "".join(oe.get_symbol(x) for x in out_labels)
        nodes.append(_ActionNode(
            left, right, f"{sl},{sr}->{so}", f"{batch}{sl},{batch}{sr}->{batch}{so}",
            f"{batch}{sl},{batch}{sr}->{so}", flip_left, flip_right,
            tuple(cross_left), tuple(cross_right), tuple(cross_ids),
            has_internal[left], has_internal[right],
        ))
        multiplicity = (1 + len(boundary) + len(cross_ids)
                        + int(has_internal[left]) + int(has_internal[right]))
        total_flops += (2 * multiplicity
                        * contraction.dimension ** len(set(labels[left]) | set(labels[right])))
        peak = max(peak, (1 + len(boundary)) * contraction.dimension ** len(out_labels))
        sites.append(all_sites)
        labels.append(out_labels)
        boundaries.append(boundary)
        has_internal.append(has_internal[left] or has_internal[right] or bool(cross_ids))
        for pos in sorted(step, reverse=True):
            active.pop(pos)
        active.append(len(sites) - 1)
    if len(active) != 1 or labels[-1]:
        raise ValueError("contraction path does not finish in a scalar")
    if peak > max_intermediate_elements:
        raise MemoryError(f"Hamiltonian boundary contraction needs {peak} elements above guard")
    if sorted(k for node in nodes for k in node.cross_bonds) != list(range(len(checked_bonds))):
        raise AssertionError("each Hamiltonian bond must have exactly one insertion")
    return HeisenbergActionPlan(
        contraction, checked_bonds, checked_couplings, tuple(nodes),
        tuple(bool(neighbors[i]) for i in range(n)), total_flops, peak,
    )
