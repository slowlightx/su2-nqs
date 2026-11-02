"""Many-cell complete Schur layouts built from physical fusion trees.

Each four-spin cell uses the balanced tree ``((0,1),(2,3))``.  Its root
label is the local doubled spin ``q`` and the two pair labels resolve the
local multiplicity ``a``.  A second binary tree fuses the cell roots, so all
remaining internal labels are inter-cell doubled spins.  No magnetic index is
stored or exposed to a learned model.

The tree leaves are kept in a cell-major *model* order.  ``site_order`` maps a
model leaf to the original lattice site; graph bonds must therefore be passed
through :meth:`ManyCellSchurLayout.remap_bonds` before using tree-native
Hamiltonian connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral, Real
from typing import Iterable, Sequence

import numpy as np

from .fusion_tree import FusionTree, FusionTreeState
from .tree_rotations import leaf_signature


def _balanced_nested(items: Sequence[object]) -> object:
    if not items:
        raise ValueError("a fusion tree requires at least one item")
    if len(items) == 1:
        return items[0]
    middle = len(items) // 2
    return (_balanced_nested(items[:middle]), _balanced_nested(items[middle:]))


def _left_nested(items: Sequence[object]) -> object:
    """Associate an ordered list from the left without changing leaf order."""

    if not items:
        raise ValueError("a fusion tree requires at least one item")
    node = items[0]
    for item in items[1:]:
        node = (node, item)
    return node


def four_spin_multiplicity_channels() -> dict[int, tuple[tuple[int, int], ...]]:
    """Balanced-cell multiplicity channels in deterministic tree order."""

    return {
        0: ((0, 0), (2, 2)),
        2: ((0, 2), (2, 0), (2, 2)),
        4: ((2, 2),),
    }


@dataclass(frozen=True)
class ManyCellSchurLabels:
    """Decoded reduced labels for one or more complete tree states."""

    cell_q: np.ndarray
    cell_a: np.ndarray
    intercell_q: np.ndarray


@dataclass(frozen=True)
class ManyCellSchurLayout:
    """A cell-major physical fusion tree and its lattice bookkeeping."""

    tree: FusionTree
    site_order: tuple[int, ...]
    physical_cells: tuple[tuple[int, int, int, int], ...]
    site_coordinates: tuple[tuple[float, float], ...]
    cell_root_indices: tuple[int, ...]
    cell_pair_indices: tuple[tuple[int, int], ...]
    intercell_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        n = self.tree.n_leaves
        if len(self.site_order) != n or tuple(sorted(self.site_order)) != tuple(range(n)):
            raise ValueError("site_order must be a permutation of physical sites")
        if len(self.site_coordinates) != n:
            raise ValueError("one coordinate is required per model leaf")
        coordinates = np.asarray(self.site_coordinates, dtype=np.float64)
        if coordinates.shape != (n, 2) or np.any(~np.isfinite(coordinates)):
            raise ValueError("site coordinates must be finite two-vectors")
        if len(self.physical_cells) * 4 != n:
            raise ValueError("the layout must be tiled by complete four-spin cells")
        flattened = tuple(site for cell in self.physical_cells for site in cell)
        if flattened != self.site_order:
            raise ValueError("physical cells must flatten to the model site order")
        node_count = n - 1
        if len(self.cell_root_indices) != len(self.physical_cells):
            raise ValueError("one cell root index is required per cell")
        if len(self.cell_pair_indices) != len(self.physical_cells):
            raise ValueError("two pair indices are required per cell")
        covered = set(self.intercell_indices)
        for root, pair in zip(
            self.cell_root_indices, self.cell_pair_indices, strict=True
        ):
            covered.add(root)
            covered.update(pair)
        if covered != set(range(node_count)):
            raise ValueError("cell and inter-cell indices must partition tree nodes")

    @property
    def n_sites(self) -> int:
        return self.tree.n_leaves

    @property
    def n_cells(self) -> int:
        return len(self.physical_cells)

    @property
    def model_cells(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            tuple(range(4 * cell, 4 * cell + 4))
            for cell in range(self.n_cells)
        )

    @property
    def physical_to_model(self) -> tuple[int, ...]:
        inverse = [0] * self.n_sites
        for model, physical in enumerate(self.site_order):
            inverse[physical] = model
        return tuple(inverse)

    def remap_bonds(
        self,
        bonds: Iterable[tuple[int, int] | tuple[int, int, float]],
    ) -> tuple[tuple[int, int, float], ...]:
        """Map physical-site graph bonds into the cell-major leaf order."""

        inverse = self.physical_to_model
        result = []
        for bond in bonds:
            if len(bond) == 2:
                first, second = bond
                coupling = 1.0
            elif len(bond) == 3:
                first, second, coupling = bond
            else:
                raise ValueError(f"invalid bond: {bond!r}")
            if any(
                isinstance(site, bool) or not isinstance(site, Integral)
                for site in (first, second)
            ):
                raise TypeError("bond endpoints must be integers")
            if isinstance(coupling, bool) or not isinstance(coupling, Real):
                raise TypeError("bond coupling must be real")
            first, second = int(first), int(second)
            if first == second or not 0 <= first < self.n_sites or not 0 <= second < self.n_sites:
                raise ValueError("bond endpoints must be distinct physical sites")
            value = float(coupling)
            if not np.isfinite(value):
                raise ValueError("bond coupling must be finite")
            result.append((inverse[first], inverse[second], value))
        return tuple(result)

    def decode(self, states: Sequence[Sequence[int]] | np.ndarray) -> ManyCellSchurLabels:
        """Decode complete postorder labels into local ``(q,a)`` and cell-tree q."""

        labels = np.asarray(states, dtype=np.int32)
        squeeze = labels.ndim == 1
        if squeeze:
            labels = labels[None, :]
        if labels.ndim != 2 or labels.shape[1] != self.n_sites - 1:
            raise ValueError(
                f"states must have shape (...,{self.n_sites - 1})"
            )
        cell_q = labels[:, np.asarray(self.cell_root_indices)]
        cell_a = np.empty_like(cell_q)
        channels = four_spin_multiplicity_channels()
        for cell, (left_index, right_index) in enumerate(self.cell_pair_indices):
            for row in range(len(labels)):
                q = int(cell_q[row, cell])
                pair = (int(labels[row, left_index]), int(labels[row, right_index]))
                try:
                    cell_a[row, cell] = channels[q].index(pair)
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        f"invalid four-spin multiplicity labels q={q}, pair={pair}"
                    ) from error
        intercell = labels[:, np.asarray(self.intercell_indices)]
        if squeeze:
            return ManyCellSchurLabels(cell_q[0], cell_a[0], intercell[0])
        return ManyCellSchurLabels(cell_q, cell_a, intercell)

    def decision_coordinates(self) -> tuple[tuple[float, float], ...]:
        """Return root-first support centroids for autoregressive attention."""

        coordinates = np.asarray(self.site_coordinates, dtype=np.float64)
        centers = []
        root_first = []

        def visit(node) -> None:
            from .fusion_tree import Leaf

            if isinstance(node, Leaf):
                return
            root_first.append(node)
            visit(node.left)
            visit(node.right)

        visit(self.tree.root)
        for node in root_first:
            sites = np.asarray(leaf_signature(node), dtype=np.int64)
            centers.append(tuple(np.mean(coordinates[sites], axis=0)))
        values = np.asarray(centers, dtype=np.float64)
        span = np.maximum(np.ptp(values, axis=0), 1.0)
        values = (values - np.mean(values, axis=0, keepdims=True)) / span
        return tuple(tuple(float(value) for value in row) for row in values)


@dataclass(frozen=True)
class ManyCellSimplexLayout:
    """Nine complete Schur cells grouped into three three-cell intertwiners.

    ``cell_groups`` use the model-cell ordering.  Each group is associated as
    ``((cell_0, cell_1), cell_2)``; ``pair_indices`` and ``root_indices`` are
    postorder labels of the corresponding virtual bond and scalar simplex
    root.  The wrapper contains no magnetic indices.
    """

    schur: ManyCellSchurLayout
    cell_groups: tuple[tuple[int, int, int], ...]
    pair_indices: tuple[int, ...]
    root_indices: tuple[int, ...]
    lattice_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if len(self.cell_groups) < 1:
            raise ValueError("at least one three-cell simplex is required")
        flattened = tuple(cell for group in self.cell_groups for cell in group)
        if flattened != tuple(range(self.schur.n_cells)):
            raise ValueError("simplex groups must partition model cells in order")
        if any(len(group) != 3 for group in self.cell_groups):
            raise ValueError("every simplex must contain exactly three cells")
        if len(self.pair_indices) != len(self.cell_groups):
            raise ValueError("one virtual-pair label is required per simplex")
        if len(self.root_indices) != len(self.cell_groups):
            raise ValueError("one root label is required per simplex")
        node_count = self.schur.n_sites - 1
        if any(not 0 <= index < node_count for index in self.pair_indices):
            raise ValueError("simplex pair index lies outside the fusion tree")
        if any(not 0 <= index < node_count for index in self.root_indices):
            raise ValueError("simplex root index lies outside the fusion tree")

    @property
    def n_sites(self) -> int:
        return self.schur.n_sites

    @property
    def n_simplexes(self) -> int:
        return len(self.cell_groups)


def _rectangular_cell_data(
    lx: int, ly: int, triangular_coordinates: bool
) -> tuple[
    tuple[tuple[int, int, int, int], ...],
    tuple[int, ...],
    tuple[tuple[float, float], ...],
    tuple[object, ...],
]:
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (lx, ly)):
        raise TypeError("lx and ly must be integers")
    lx, ly = int(lx), int(ly)
    if lx < 2 or ly < 2 or lx % 2 or ly % 2:
        raise ValueError("lx and ly must be positive even dimensions")
    cells: list[tuple[int, int, int, int]] = []
    for block_y in range(0, ly, 2):
        for block_x in range(0, lx, 2):
            cells.append(
                (
                    block_y * lx + block_x,
                    block_y * lx + block_x + 1,
                    (block_y + 1) * lx + block_x,
                    (block_y + 1) * lx + block_x + 1,
                )
            )
    site_order = tuple(site for cell in cells for site in cell)
    cell_nested = []
    for cell in range(len(cells)):
        start = 4 * cell
        cell_nested.append(((start, start + 1), (start + 2, start + 3)))
    coordinates = []
    for physical in site_order:
        x = physical % lx
        y = physical // lx
        if triangular_coordinates:
            coordinates.append((x + 0.5 * y, 0.5 * np.sqrt(3.0) * y))
        else:
            coordinates.append((float(x), float(y)))
    return tuple(cells), site_order, tuple(coordinates), tuple(cell_nested)


def _layout_from_cell_tree(
    cells: tuple[tuple[int, int, int, int], ...],
    site_order: tuple[int, ...],
    coordinates: tuple[tuple[float, float], ...],
    nested: object,
) -> ManyCellSchurLayout:
    tree = FusionTree.from_nested(nested)  # type: ignore[arg-type]

    nodes = tree.internal_nodes
    signatures = [leaf_signature(node) for node in nodes]
    cell_roots = []
    cell_pairs = []
    cell_node_indices: set[int] = set()
    for cell in range(len(cells)):
        model_sites = tuple(range(4 * cell, 4 * cell + 4))
        left_pair = model_sites[:2]
        right_pair = model_sites[2:]
        try:
            left_index = signatures.index(left_pair)
            right_index = signatures.index(right_pair)
            root_index = signatures.index(model_sites)
        except ValueError as error:
            raise RuntimeError("constructed tree lost a four-spin cell subtree") from error
        cell_pairs.append((left_index, right_index))
        cell_roots.append(root_index)
        cell_node_indices.update((left_index, right_index, root_index))
    intercell = tuple(
        index for index in range(len(nodes)) if index not in cell_node_indices
    )
    return ManyCellSchurLayout(
        tree=tree,
        site_order=site_order,
        physical_cells=cells,
        site_coordinates=coordinates,
        cell_root_indices=tuple(cell_roots),
        cell_pair_indices=tuple(cell_pairs),
        intercell_indices=intercell,
    )


@lru_cache(maxsize=None)
def rectangular_four_spin_layout(
    lx: int,
    ly: int,
    *,
    triangular_coordinates: bool = True,
) -> ManyCellSchurLayout:
    """Tile an even ``lx x ly`` torus; fuse cell roots in a balanced tree."""

    cells, site_order, coordinates, cell_nested = _rectangular_cell_data(
        lx, ly, triangular_coordinates
    )
    return _layout_from_cell_tree(
        cells,
        site_order,
        coordinates,
        _balanced_nested(cell_nested),
    )


@lru_cache(maxsize=None)
def rectangular_sequential_four_spin_layout(
    lx: int,
    ly: int,
    *,
    triangular_coordinates: bool = True,
) -> ManyCellSchurLayout:
    """Tile an even torus and fuse complete cell irreps from the left.

    The local four-spin Schur transform is identical to
    :func:`rectangular_four_spin_layout`.  Only the inter-cell association is
    changed.  This exposes the cumulative irrep ``Q_i`` after cell ``i`` and
    permits one exact autoregressive decision ``(q_i,a_i,Q_i)`` per cell.
    """

    cells, site_order, coordinates, cell_nested = _rectangular_cell_data(
        lx, ly, triangular_coordinates
    )
    return _layout_from_cell_tree(
        cells,
        site_order,
        coordinates,
        _left_nested(cell_nested),
    )


@lru_cache(maxsize=None)
def triangular_six_by_six_simplex_layout() -> ManyCellSimplexLayout:
    """Return the local-intertwiner layout used by the triangular ``6x6`` gate.

    The nine ``2x2`` cells form a periodic ``3x3`` coarse lattice.  The
    disjoint upward simplexes are ``(0,1,3)``, ``(4,5,7)``, and ``(8,6,2)``
    in physical coarse-cell coordinates.  Cells are reordered only in the
    model tree; :meth:`ManyCellSchurLayout.remap_bonds` retains the physical
    Hamiltonian geometry.
    """

    cells, _, _, _ = _rectangular_cell_data(6, 6, True)
    physical_groups = ((0, 1, 3), (4, 5, 7), (8, 6, 2))
    cell_order = tuple(cell for group in physical_groups for cell in group)
    reordered_cells = tuple(cells[index] for index in cell_order)
    site_order = tuple(site for cell in reordered_cells for site in cell)
    coordinates = []
    for physical in site_order:
        x = physical % 6
        y = physical // 6
        coordinates.append((x + 0.5 * y, 0.5 * np.sqrt(3.0) * y))

    cell_nested = []
    for cell in range(9):
        start = 4 * cell
        cell_nested.append(((start, start + 1), (start + 2, start + 3)))
    simplex_nested = tuple(
        ((cell_nested[3 * group], cell_nested[3 * group + 1]), cell_nested[3 * group + 2])
        for group in range(3)
    )
    layout = _layout_from_cell_tree(
        reordered_cells,
        site_order,
        tuple(coordinates),
        _balanced_nested(simplex_nested),
    )
    signatures = [leaf_signature(node) for node in layout.tree.internal_nodes]
    pair_indices = []
    root_indices = []
    for group in range(3):
        start = 12 * group
        pair_indices.append(signatures.index(tuple(range(start, start + 8))))
        root_indices.append(signatures.index(tuple(range(start, start + 12))))
    return ManyCellSimplexLayout(
        schur=layout,
        cell_groups=((0, 1, 2), (3, 4, 5), (6, 7, 8)),
        pair_indices=tuple(pair_indices),
        root_indices=tuple(root_indices),
        lattice_shape=(6, 6),
    )


def scramble_simplex_coordinates(
    layout: ManyCellSimplexLayout,
    permutation: Sequence[int],
) -> ManyCellSimplexLayout:
    """Return an SU(2)-identical control with permuted cell-coordinate blocks."""

    order = tuple(int(value) for value in permutation)
    if tuple(sorted(order)) != tuple(range(layout.schur.n_cells)):
        raise ValueError("permutation must contain every model cell exactly once")
    chunks = tuple(
        layout.schur.site_coordinates[4 * cell : 4 * cell + 4]
        for cell in range(layout.schur.n_cells)
    )
    scrambled = ManyCellSchurLayout(
        tree=layout.schur.tree,
        site_order=layout.schur.site_order,
        physical_cells=layout.schur.physical_cells,
        site_coordinates=tuple(
            coordinate for cell in order for coordinate in chunks[cell]
        ),
        cell_root_indices=layout.schur.cell_root_indices,
        cell_pair_indices=layout.schur.cell_pair_indices,
        intercell_indices=layout.schur.intercell_indices,
    )
    return ManyCellSimplexLayout(
        schur=scrambled,
        cell_groups=layout.cell_groups,
        pair_indices=layout.pair_indices,
        root_indices=layout.root_indices,
        lattice_shape=layout.lattice_shape,
    )


def validate_layout_state(
    layout: ManyCellSchurLayout,
    labels: Sequence[int],
    *,
    target_q: int = 0,
) -> FusionTreeState:
    """Validate a complete doubled-label assignment at an API boundary."""

    state = FusionTreeState(tuple(int(value) for value in labels))
    mapping = state.label_map(layout.tree)
    from .fusion_tree import node_q
    from .quantum_numbers import coupling_range, validate_q

    for node in layout.tree.internal_nodes:
        q = mapping[node]
        validate_q(q, name="state label")
        if q not in coupling_range(node_q(node.left, mapping), node_q(node.right, mapping)):
            raise ValueError("state contains an inadmissible doubled-spin coupling")
    if state.total_q != int(target_q):
        raise ValueError("state does not belong to the requested total-spin sector")
    return state


__all__ = [
    "ManyCellSchurLabels",
    "ManyCellSchurLayout",
    "ManyCellSimplexLayout",
    "four_spin_multiplicity_channels",
    "rectangular_four_spin_layout",
    "rectangular_sequential_four_spin_layout",
    "scramble_simplex_coordinates",
    "triangular_six_by_six_simplex_layout",
    "validate_layout_state",
]
