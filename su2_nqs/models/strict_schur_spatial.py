"""Strict local-Schur coefficient model and geometry-only reduced generator.

This module is the exact-small boundary for a future SU(2)-spatial network.
Configuration-dependent carriers are never passed through an activation or
multiplied with an overlapping support.  A cell is transformed by the complete
local Schur table, and two disjoint cells are coupled with fixed
Condon--Shortley Clebsch--Gordan coefficients.  Trainable arrays act only on
reduced multiplicity indices.

The optional ``StaticSchurGenerator`` is deliberately separated from the
physical contraction.  Its GELU network sees only immutable geometry/channel
descriptors and produces reduced coefficients; it never sees a spin
configuration.  Consequently its nonlinearities cannot repeat a physical
input leg.  This is a validation-scale coefficient model, not yet a scalable
four-patch SU(2)-ViT.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.fusion_catalog import (
    DisjointMergeRoute,
    FusionSupport,
)
from ..su2.clebsch_gordan import clebsch_gordan
from .spin_clebsch_tree import spin_cg_tensor
from ..su2.local_schur import (
    CellKind,
    cell_schur_spec,
    cell_schur_tensor,
)
from ..su2.quantum_numbers import magnetic_numbers
from ..su2.quantum_numbers import coupling_range


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SchurCell:
    """One complete local Schur carrier with an ordered physical support."""

    support: tuple[int, ...]
    kind: CellKind

    def __post_init__(self) -> None:
        spec = cell_schur_spec(self.kind)
        raw = tuple(self.support)
        if len(raw) != spec.cell_size:
            raise ValueError(
                f"{self.kind} requires exactly {spec.cell_size} physical sites"
            )
        if any(isinstance(site, bool) or not isinstance(site, Integral) for site in raw):
            raise TypeError("cell support entries must be integers")
        checked = tuple(int(site) for site in raw)
        if any(site < 0 for site in checked):
            raise ValueError("cell support entries must be nonnegative")
        if len(set(checked)) != len(checked):
            raise ValueError("a Schur cell cannot repeat a physical site")
        object.__setattr__(self, "support", checked)

    @property
    def size(self) -> int:
        return len(self.support)


@dataclass(frozen=True)
class StrictSchurSpatialPlan:
    """One- or two-cell complete-singlet exact-small contraction plan.

    The two-cell case is a single ledger merge between disjoint complete
    Schur cells.  It spans the full singlet spaces used by the permanent
    oracle: two triangles for ``N=6`` and two plaquettes for ``N=8``.
    ``cell_coordinates`` are metadata for the static reduced generator only.
    """

    cells: tuple[SchurCell, ...]
    cell_coordinates: tuple[tuple[float, float], ...]
    target_q: int = 0
    target_m2: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("target_q", self.target_q),
            ("target_m2", self.target_m2),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer doubled quantum number")
        cells = tuple(self.cells)
        if any(not isinstance(cell, SchurCell) for cell in cells):
            raise TypeError("cells must contain SchurCell instances")
        object.__setattr__(self, "cells", cells)
        if len(cells) not in (1, 2):
            raise ValueError("the exact-small plan supports one or two Schur cells")
        if len(self.cell_coordinates) != len(cells):
            raise ValueError("one coordinate is required per Schur cell")
        coordinates = np.asarray(self.cell_coordinates, dtype=np.float64)
        if coordinates.shape != (len(cells), 2) or np.any(~np.isfinite(coordinates)):
            raise ValueError("cell coordinates must be finite two-vectors")
        object.__setattr__(
            self,
            "cell_coordinates",
            tuple(tuple(float(value) for value in row) for row in coordinates),
        )
        flattened = tuple(site for cell in cells for site in cell.support)
        if tuple(sorted(flattened)) != tuple(range(len(flattened))):
            raise ValueError("Schur cells must cover physical sites 0..N-1 exactly once")
        if self.target_q != 0 or self.target_m2 != 0:
            raise ValueError("the first strict Schur-spatial boundary targets q=m2=0")
        if len(cells) == 2:
            if cells[0].kind != cells[1].kind:
                raise ValueError("the complete two-cell oracle requires equal cell kinds")
            if np.linalg.norm(coordinates[1] - coordinates[0]) <= 0.0:
                raise ValueError("two Schur cells require distinct coordinates")
            # Constructor validation is the support-ledger proof obligation:
            # children are disjoint and cover the all-site parent exactly.
            DisjointMergeRoute(
                FusionSupport(tuple(range(self.n_sites))),
                FusionSupport(self.cells[0].support),
                FusionSupport(self.cells[1].support),
            )

    @property
    def n_sites(self) -> int:
        return sum(cell.size for cell in self.cells)

    @property
    def active_q(self) -> tuple[int, ...]:
        if len(self.cells) == 1:
            return (0,)
        left = set(cell_schur_spec(self.cells[0].kind).q_values)
        right = set(cell_schur_spec(self.cells[1].kind).q_values)
        return tuple(sorted(left & right))

    @property
    def reduced_shapes(self) -> tuple[tuple[int, ...], ...]:
        if len(self.cells) == 1:
            multiplicity = cell_schur_spec(self.cells[0].kind).multiplicity(0)
            return ((multiplicity,),)
        left = cell_schur_spec(self.cells[0].kind)
        right = cell_schur_spec(self.cells[1].kind)
        return tuple(
            (left.multiplicity(q), right.multiplicity(q)) for q in self.active_q
        )

    @property
    def reduced_parameter_count(self) -> int:
        return int(sum(np.prod(shape) for shape in self.reduced_shapes))

    @property
    def ledger_merge(self) -> DisjointMergeRoute | None:
        if len(self.cells) == 1:
            return None
        return DisjointMergeRoute(
            FusionSupport(tuple(range(self.n_sites))),
            FusionSupport(self.cells[0].support),
            FusionSupport(self.cells[1].support),
        )


@dataclass(frozen=True)
class ReducedSchurMergeBlock:
    """One reduced channel in a strict disjoint-support Schur merge."""

    q_left: int
    q_right: int
    q_total: int
    output_width: int
    left_width: int
    right_width: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.output_width, self.left_width, self.right_width)


@dataclass(frozen=True)
class FourPlaquetteSchurPlan:
    """Strict ``4x4`` bridge built from four complete ``2x2`` plaquettes.

    The registered topology merges the two plaquettes in each lattice row,
    then couples the two eight-site carriers to a singlet.  Every nonlinear
    or attention-like operation remains outside this physical contraction.
    """

    cells: tuple[SchurCell, ...]
    cell_coordinates: tuple[tuple[float, float], ...]
    pairings: tuple[tuple[int, int], tuple[int, int]] = ((0, 1), (2, 3))
    reduced_width: int = 4
    target_q: int = 0
    target_m2: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("reduced_width", self.reduced_width),
            ("target_q", self.target_q),
            ("target_m2", self.target_m2),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if self.reduced_width < 1:
            raise ValueError("reduced_width must be positive")
        if self.target_q != 0 or self.target_m2 != 0:
            raise ValueError("the four-plaquette bridge targets q=m2=0")
        cells = tuple(self.cells)
        if any(not isinstance(cell, SchurCell) for cell in cells):
            raise TypeError("cells must contain SchurCell instances")
        object.__setattr__(self, "cells", cells)
        if len(cells) != 4 or any(
            cell.kind != "square_plaquette" for cell in cells
        ):
            raise ValueError("four complete square-plaquette cells are required")
        flattened = tuple(site for cell in cells for site in cell.support)
        if tuple(sorted(flattened)) != tuple(range(16)):
            raise ValueError("four plaquettes must cover physical sites 0..15 once")
        coordinates = np.asarray(self.cell_coordinates, dtype=np.float64)
        if coordinates.shape != (4, 2) or np.any(~np.isfinite(coordinates)):
            raise ValueError("four finite two-dimensional cell coordinates are required")
        if len({tuple(row) for row in coordinates}) != 4:
            raise ValueError("plaquette coordinates must be distinct")
        object.__setattr__(
            self,
            "cell_coordinates",
            tuple(tuple(float(value) for value in row) for row in coordinates),
        )
        raw_pairings = tuple(tuple(pair) for pair in self.pairings)
        if len(raw_pairings) != 2 or any(len(pair) != 2 for pair in raw_pairings):
            raise ValueError("exactly two two-cell pairings are required")
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for pair in raw_pairings
            for index in pair
        ):
            raise TypeError("pairing entries must be integer cell indices")
        pairings = tuple(tuple(int(index) for index in pair) for pair in raw_pairings)
        object.__setattr__(self, "pairings", pairings)
        flattened_pairs = tuple(index for pair in pairings for index in pair)
        if tuple(sorted(flattened_pairs)) != (0, 1, 2, 3):
            raise ValueError("pairings must partition the four cell indices")
        if any(left == right for left, right in self.pairings):
            raise ValueError("a level-one merge cannot repeat a cell")
        # Constructing every route is the fail-closed support-ledger proof.
        _ = self.level1_routes
        _ = self.root_route

    @property
    def n_sites(self) -> int:
        return 16

    @property
    def local_q(self) -> tuple[int, ...]:
        return cell_schur_spec("square_plaquette").q_values

    @property
    def intermediate_multiplicities(self) -> tuple[tuple[int, int], ...]:
        spec = cell_schur_spec("square_plaquette")
        counts: dict[int, int] = {}
        for q_left in self.local_q:
            for q_right in self.local_q:
                for q_total in coupling_range(q_left, q_right):
                    counts[q_total] = counts.get(q_total, 0) + (
                        spec.multiplicity(q_left) * spec.multiplicity(q_right)
                    )
        return tuple(sorted(counts.items()))

    @property
    def intermediate_widths(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (q, min(self.reduced_width, multiplicity))
            for q, multiplicity in self.intermediate_multiplicities
        )

    @property
    def level1_layout(self) -> tuple[ReducedSchurMergeBlock, ...]:
        spec = cell_schur_spec("square_plaquette")
        widths = dict(self.intermediate_widths)
        return tuple(
            ReducedSchurMergeBlock(
                q_left,
                q_right,
                q_total,
                widths[q_total],
                spec.multiplicity(q_left),
                spec.multiplicity(q_right),
            )
            for q_left in self.local_q
            for q_right in self.local_q
            for q_total in coupling_range(q_left, q_right)
        )

    @property
    def root_shapes(self) -> tuple[tuple[int, int], ...]:
        return tuple((width, width) for _q, width in self.intermediate_widths)

    @property
    def level1_routes(self) -> tuple[DisjointMergeRoute, ...]:
        routes = []
        for left_index, right_index in self.pairings:
            left = FusionSupport(self.cells[left_index].support)
            right = FusionSupport(self.cells[right_index].support)
            parent = FusionSupport(left.sites + right.sites)
            routes.append(DisjointMergeRoute(parent, left, right))
        return tuple(routes)

    @property
    def root_route(self) -> DisjointMergeRoute:
        left, right = (route.parent for route in self.level1_routes)
        return DisjointMergeRoute(
            FusionSupport(tuple(range(self.n_sites))), left, right
        )

    @property
    def reduced_parameter_count(self) -> int:
        level1 = sum(int(np.prod(block.shape)) for block in self.level1_layout)
        root = sum(int(np.prod(shape)) for shape in self.root_shapes)
        return 2 * level1 + root

    @property
    def internal_gauge_dimension(self) -> int:
        """Dimension of the two independent intermediate feature gauges."""

        return 2 * sum(width * width for _q, width in self.intermediate_widths)

    @property
    def generic_effective_tangent_dimension(self) -> int:
        """Expected generic physical tangent rank after quotienting gauges."""

        return self.reduced_parameter_count - self.internal_gauge_dimension


def square_plaquette_plan() -> StrictSchurSpatialPlan:
    """Return the complete two-dimensional singlet space of one plaquette."""

    return StrictSchurSpatialPlan(
        (SchurCell((0, 1, 2, 3), "square_plaquette"),),
        ((0.0, 0.0),),
    )


def two_triangle_plan() -> StrictSchurSpatialPlan:
    """Return two disjoint triangle carriers spanning the ``N=6`` singlets."""

    return StrictSchurSpatialPlan(
        (
            SchurCell((0, 1, 2), "triangular_triangle"),
            SchurCell((3, 4, 5), "triangular_triangle"),
        ),
        ((0.0, 0.0), (1.0, 0.0)),
    )


def two_plaquette_plan() -> StrictSchurSpatialPlan:
    """Return two disjoint plaquette carriers spanning the ``N=8`` singlets."""

    return StrictSchurSpatialPlan(
        (
            SchurCell((0, 1, 2, 3), "square_plaquette"),
            SchurCell((4, 5, 6, 7), "square_plaquette"),
        ),
        ((0.0, 0.0), (1.0, 0.0)),
    )


def four_plaquette_d4_plan() -> FourPlaquetteSchurPlan:
    """Return the registered square ``4x4`` four-plaquette bridge."""

    return FourPlaquetteSchurPlan(
        cells=(
            SchurCell((0, 1, 4, 5), "square_plaquette"),
            SchurCell((2, 3, 6, 7), "square_plaquette"),
            SchurCell((8, 9, 12, 13), "square_plaquette"),
            SchurCell((10, 11, 14, 15), "square_plaquette"),
        ),
        cell_coordinates=((0.5, 0.5), (2.5, 0.5), (0.5, 2.5), (2.5, 2.5)),
    )


def init_strict_schur_reduced(
    key: jax.Array,
    plan: StrictSchurSpatialPlan,
    *,
    scale: float = 0.2,
) -> tuple[jax.Array, ...]:
    """Initialize one real reduced coefficient block per active ``q``."""

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    keys = jax.random.split(key, len(plan.reduced_shapes))
    return tuple(
        (float(scale) / np.sqrt(max(int(np.prod(shape)), 1)))
        * jax.random.normal(subkey, shape, dtype=jnp.float64)
        for subkey, shape in zip(keys, plan.reduced_shapes, strict=True)
    )


def init_four_plaquette_reduced(
    key: jax.Array,
    plan: FourPlaquetteSchurPlan,
    *,
    scale: float = 0.2,
) -> dict[str, object]:
    """Initialize the 619-real-parameter direct ``D=4`` bridge."""

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    count = 2 * len(plan.level1_layout) + len(plan.root_shapes)
    keys = iter(jax.random.split(key, count))
    level1 = []
    for _route in plan.level1_routes:
        blocks = []
        for layout in plan.level1_layout:
            fan_in = max(layout.left_width * layout.right_width, 1)
            blocks.append(
                (float(scale) / np.sqrt(fan_in))
                * jax.random.normal(next(keys), layout.shape, dtype=jnp.float64)
            )
        level1.append(tuple(blocks))
    root = tuple(
        (float(scale) / np.sqrt(max(int(np.prod(shape)), 1)))
        * jax.random.normal(next(keys), shape, dtype=jnp.float64)
        for shape in plan.root_shapes
    )
    params = {"level1": tuple(level1), "root": root}
    if strict_schur_parameter_count(params) != plan.reduced_parameter_count:
        raise RuntimeError("four-plaquette reduced parameter packing changed")
    return params


def _validate_four_plaquette_params(
    params: dict[str, object], plan: FourPlaquetteSchurPlan
) -> tuple[tuple[tuple[jax.Array, ...], ...], tuple[jax.Array, ...]]:
    if not isinstance(params, dict) or set(params) != {"level1", "root"}:
        raise ValueError("four-plaquette params require level1 and root")
    raw_level1 = params["level1"]
    if not isinstance(raw_level1, (tuple, list)) or len(raw_level1) != 2:
        raise ValueError("two unshared level-one merge parameter sets are required")
    checked_level1 = []
    expected_level1 = tuple(block.shape for block in plan.level1_layout)
    for route_blocks in raw_level1:
        if not isinstance(route_blocks, (tuple, list)) or len(route_blocks) != len(
            expected_level1
        ):
            raise ValueError("level-one reduced blocks do not match the plan")
        checked = tuple(jnp.asarray(block) for block in route_blocks)
        if tuple(block.shape for block in checked) != expected_level1:
            raise ValueError("a level-one reduced block has the wrong shape")
        checked_level1.append(checked)
    raw_root = params["root"]
    if not isinstance(raw_root, (tuple, list)) or len(raw_root) != len(
        plan.root_shapes
    ):
        raise ValueError("root reduced blocks do not match the plan")
    checked_root = tuple(jnp.asarray(block) for block in raw_root)
    if tuple(block.shape for block in checked_root) != plan.root_shapes:
        raise ValueError("a root reduced block has the wrong shape")
    return tuple(checked_level1), checked_root


def _validate_blocks(
    blocks: tuple[jax.Array, ...] | list[jax.Array],
    plan: StrictSchurSpatialPlan,
) -> tuple[jax.Array, ...]:
    if not isinstance(blocks, (tuple, list)) or len(blocks) != len(
        plan.reduced_shapes
    ):
        raise ValueError("reduced blocks do not match the plan")
    checked = tuple(jnp.asarray(block) for block in blocks)
    for index, (block, shape) in enumerate(
        zip(checked, plan.reduced_shapes, strict=True)
    ):
        if block.shape != shape:
            raise ValueError(
                f"reduced block {index} has shape {block.shape}, expected {shape}"
            )
    return checked


def _cell_carrier(spinors: jax.Array, cell: SchurCell) -> jax.Array:
    """Return all ``(q,m,a)`` coefficients for one product-spinor cell."""

    spec = cell_schur_spec(cell.kind)
    local = spinors[jnp.asarray(cell.support, dtype=jnp.int32)]
    states = np.asarray(
        [
            [
                (state >> (spec.cell_size - site - 1)) & 1
                for site in range(spec.cell_size)
            ]
            for state in range(1 << spec.cell_size)
        ],
        dtype=np.int32,
    )
    product = jnp.ones((1 << spec.cell_size,), dtype=spinors.dtype)
    for site in range(spec.cell_size):
        product = product * local[site, jnp.asarray(states[:, site])]
    table = jnp.asarray(cell_schur_tensor(cell.kind), dtype=spinors.dtype)
    return jnp.einsum("s,sqma->qma", product, table)


def strict_schur_amplitude_from_spinors(
    blocks: tuple[jax.Array, ...] | list[jax.Array],
    spinors: jax.Array,
    plan: StrictSchurSpatialPlan,
) -> jax.Array:
    """Evaluate the multilinear coefficient polynomial on physical spinors."""

    checked = _validate_blocks(blocks, plan)
    raw_values = jnp.asarray(spinors)
    values = jnp.asarray(
        raw_values,
        dtype=jnp.result_type(raw_values.dtype, jnp.float64),
    )
    if values.shape != (plan.n_sites, 2):
        raise ValueError(f"expected spinor shape ({plan.n_sites}, 2)")
    carriers = tuple(_cell_carrier(values, cell) for cell in plan.cells)
    if len(plan.cells) == 1:
        spec = cell_schur_spec(plan.cells[0].kind)
        multiplicity = spec.multiplicity(0)
        return jnp.dot(
            carriers[0][0, spec.cell_size, :multiplicity], checked[0]
        )

    left_spec = cell_schur_spec(plan.cells[0].kind)
    right_spec = cell_schur_spec(plan.cells[1].kind)
    result = jnp.asarray(0.0, dtype=jnp.result_type(values, *checked))
    for block, q in zip(checked, plan.active_q, strict=True):
        left_multiplicity = left_spec.multiplicity(q)
        right_multiplicity = right_spec.multiplicity(q)
        for m_left in magnetic_numbers(q):
            m_right = -m_left
            coefficient = clebsch_gordan(q, m_left, q, m_right, 0, 0)
            left = carriers[0][
                q, m_left + left_spec.cell_size, :left_multiplicity
            ]
            right = carriers[1][
                q, m_right + right_spec.cell_size, :right_multiplicity
            ]
            result = result + coefficient * jnp.einsum(
                "a,ab,b->", left, block, right
            )
    return result


def strict_schur_amplitude(
    blocks: tuple[jax.Array, ...] | list[jax.Array],
    spins: jax.Array,
    plan: StrictSchurSpatialPlan,
) -> jax.Array:
    """Evaluate one computational-basis coefficient, ``0=down, 1=up``."""

    bits = jnp.asarray(spins)
    if bits.shape != (plan.n_sites,):
        raise ValueError(f"expected spin shape ({plan.n_sites},)")
    if bits.dtype == jnp.bool_ or not jnp.issubdtype(bits.dtype, jnp.integer):
        raise TypeError("computational-basis spins must have an integer dtype")
    bits = bits.astype(jnp.int32)
    valid = jnp.all((bits == 0) | (bits == 1))
    safe = jnp.clip(bits, 0, 1)
    spinors = jnp.eye(2, dtype=jnp.float64)[safe]
    value = strict_schur_amplitude_from_spinors(blocks, spinors, plan)
    return jnp.where(valid, value, jnp.asarray(0.0, dtype=value.dtype))


def strict_schur_amplitudes(
    blocks: tuple[jax.Array, ...] | list[jax.Array],
    spins: jax.Array,
    plan: StrictSchurSpatialPlan,
) -> jax.Array:
    """Vectorize strict Schur coefficients over physical configurations."""

    array = jnp.asarray(spins)
    if array.ndim != 2 or array.shape[1] != plan.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {plan.n_sites})")
    if array.dtype == jnp.bool_ or not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError("computational-basis spins must have an integer dtype")
    array = array.astype(jnp.int32)
    return jax.vmap(lambda row: strict_schur_amplitude(blocks, row, plan))(array)


def _merge_four_spin_carriers(
    left: jax.Array,
    right: jax.Array,
    blocks: tuple[jax.Array, ...],
    plan: FourPlaquetteSchurPlan,
) -> jax.Array:
    """Merge two disjoint complete plaquette carriers into eight sites."""

    width = int(plan.reduced_width)
    dtype = jnp.result_type(left.dtype, right.dtype, *(block.dtype for block in blocks))
    result = jnp.zeros((9, 17, width), dtype=dtype)
    cg = jnp.asarray(spin_cg_tensor(4, 4), dtype=dtype)
    magnetic_sum_indices = (
        jnp.arange(9, dtype=jnp.int32)[:, None]
        + jnp.arange(9, dtype=jnp.int32)[None, :]
    ).reshape(-1)
    for layout, block in zip(plan.level1_layout, blocks, strict=True):
        left_values = left[
            layout.q_left, :, : layout.left_width
        ]
        right_values = right[
            layout.q_right, :, : layout.right_width
        ]
        coefficients = cg[
            layout.q_left, :, layout.q_right, :, layout.q_total
        ]
        contributions = jnp.einsum(
            "ma,nb,cab,mn->mnc",
            left_values,
            right_values,
            block,
            coefficients,
        ).reshape((81, layout.output_width))
        by_m = jnp.zeros((17, layout.output_width), dtype=dtype).at[
            magnetic_sum_indices
        ].add(contributions)
        result = result.at[
            layout.q_total, :, : layout.output_width
        ].add(by_m)
    return result


def four_plaquette_amplitude_from_spinors(
    params: dict[str, object],
    spinors: jax.Array,
    plan: FourPlaquetteSchurPlan,
) -> jax.Array:
    """Evaluate the strict four-plaquette coefficient polynomial."""

    level1, root = _validate_four_plaquette_params(params, plan)
    raw_values = jnp.asarray(spinors)
    values = jnp.asarray(
        raw_values,
        dtype=jnp.result_type(raw_values.dtype, jnp.float64),
    )
    if values.shape != (plan.n_sites, 2):
        raise ValueError(f"expected spinor shape ({plan.n_sites}, 2)")
    local = tuple(_cell_carrier(values, cell) for cell in plan.cells)
    merged = tuple(
        _merge_four_spin_carriers(
            local[left_index],
            local[right_index],
            route_blocks,
            plan,
        )
        for (left_index, right_index), route_blocks in zip(
            plan.pairings, level1, strict=True
        )
    )
    dtype = jnp.result_type(merged[0].dtype, merged[1].dtype, *(x.dtype for x in root))
    cg = jnp.asarray(spin_cg_tensor(8, 8), dtype=dtype)
    result = jnp.asarray(0.0, dtype=dtype)
    for (q, q_width), block in zip(
        plan.intermediate_widths, root, strict=True
    ):
        result = result + jnp.einsum(
            "ma,nb,ab,mn->",
            merged[0][q, :, :q_width],
            merged[1][q, :, :q_width],
            block,
            cg[q, :, q, :, 0],
        )
    return result


def four_plaquette_amplitude(
    params: dict[str, object],
    spins: jax.Array,
    plan: FourPlaquetteSchurPlan,
) -> jax.Array:
    """Evaluate one square-``4x4`` computational-basis coefficient."""

    bits = jnp.asarray(spins)
    if bits.shape != (plan.n_sites,):
        raise ValueError(f"expected spin shape ({plan.n_sites},)")
    if bits.dtype == jnp.bool_ or not jnp.issubdtype(bits.dtype, jnp.integer):
        raise TypeError("computational-basis spins must have an integer dtype")
    bits = bits.astype(jnp.int32)
    valid = jnp.all((bits == 0) | (bits == 1))
    safe = jnp.clip(bits, 0, 1)
    spinors = jnp.eye(2, dtype=jnp.float64)[safe]
    value = four_plaquette_amplitude_from_spinors(params, spinors, plan)
    return jnp.where(valid, value, jnp.asarray(0.0, dtype=value.dtype))


def four_plaquette_amplitudes(
    params: dict[str, object],
    spins: jax.Array,
    plan: FourPlaquetteSchurPlan,
) -> jax.Array:
    """Vectorize the strict four-plaquette coefficient over configurations."""

    array = jnp.asarray(spins)
    if array.ndim != 2 or array.shape[1] != plan.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {plan.n_sites})")
    if array.dtype == jnp.bool_ or not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError("computational-basis spins must have an integer dtype")
    array = array.astype(jnp.int32)
    return jax.vmap(
        lambda row: four_plaquette_amplitude(params, row, plan)
    )(array)


def strict_schur_spatial_descriptors(
    plan: StrictSchurSpatialPlan,
) -> np.ndarray:
    """Return immutable per-reduced-entry geometry/channel descriptors.

    Columns are bias, normalized ``q``, centered left/right multiplicity
    labels, normalized displacement components, and cell separation.  These
    values contain no spin/configuration data.
    """

    if len(plan.cells) == 1:
        displacement = np.zeros(2, dtype=np.float64)
    else:
        displacement = np.asarray(plan.cell_coordinates[1]) - np.asarray(
            plan.cell_coordinates[0]
        )
    distance = float(np.linalg.norm(displacement))
    scale = max(distance, 1.0)
    rows = []
    for q, shape in zip(plan.active_q, plan.reduced_shapes, strict=True):
        if len(shape) == 1:
            for left in range(shape[0]):
                left_coordinate = 0.0 if shape[0] == 1 else 2.0 * left / (shape[0] - 1) - 1.0
                rows.append((1.0, 0.0, left_coordinate, 0.0, 0.0, 0.0, 0.0))
            continue
        for left in range(shape[0]):
            for right in range(shape[1]):
                left_coordinate = 0.0 if shape[0] == 1 else 2.0 * left / (shape[0] - 1) - 1.0
                right_coordinate = 0.0 if shape[1] == 1 else 2.0 * right / (shape[1] - 1) - 1.0
                rows.append(
                    (
                        1.0,
                        q / max(plan.n_sites, 1),
                        left_coordinate,
                        right_coordinate,
                        displacement[0] / scale,
                        displacement[1] / scale,
                        distance,
                    )
                )
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (plan.reduced_parameter_count, 7):
        raise RuntimeError("static Schur descriptor packing changed")
    result.setflags(write=False)
    return result


def init_static_schur_generator(
    key: jax.Array,
    plan: StrictSchurSpatialPlan,
    *,
    hidden_dim: int = 4,
    scale: float = 0.2,
) -> dict[str, object]:
    """Initialize a geometry-only reduced-tensor generator at its direct nest."""

    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, Integral) or hidden_dim < 1:
        raise ValueError("hidden_dim must be a positive integer")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    direct_key, input_key, output_key = jax.random.split(key, 3)
    direct = init_strict_schur_reduced(direct_key, plan, scale=scale)
    descriptor_dim = strict_schur_spatial_descriptors(plan).shape[1]
    return {
        "direct": direct,
        "spatial": {
            "input": (scale / np.sqrt(descriptor_dim))
            * jax.random.normal(
                input_key, (descriptor_dim, int(hidden_dim)), dtype=jnp.float64
            ),
            "bias": jnp.zeros((int(hidden_dim),), dtype=jnp.float64),
            "output": (scale / np.sqrt(int(hidden_dim)))
            * jax.random.normal(output_key, (int(hidden_dim),), dtype=jnp.float64),
            "output_bias": jnp.asarray(0.0, dtype=jnp.float64),
            "residual_scale": jnp.asarray(0.0, dtype=jnp.float64),
        },
    }


def static_schur_reduced_blocks(
    params: dict[str, object],
    plan: StrictSchurSpatialPlan,
) -> tuple[jax.Array, ...]:
    """Generate reduced blocks from geometry only, with an exact direct skip."""

    if not isinstance(params, dict) or set(params) != {"direct", "spatial"}:
        raise ValueError("generator params require direct and spatial entries")
    direct = _validate_blocks(params["direct"], plan)
    spatial = params["spatial"]
    if not isinstance(spatial, dict) or set(spatial) != {
        "input",
        "bias",
        "output",
        "output_bias",
        "residual_scale",
    }:
        raise ValueError("invalid static spatial generator parameters")
    descriptors = jnp.asarray(strict_schur_spatial_descriptors(plan))
    hidden = jax.nn.gelu(descriptors @ spatial["input"] + spatial["bias"])
    residual = hidden @ spatial["output"] + spatial["output_bias"]
    direct_flat = jnp.concatenate([jnp.ravel(block) for block in direct])
    generated = direct_flat + spatial["residual_scale"] * residual
    blocks = []
    start = 0
    for shape in plan.reduced_shapes:
        stop = start + int(np.prod(shape))
        blocks.append(generated[start:stop].reshape(shape))
        start = stop
    return tuple(blocks)


def strict_schur_spatial_amplitudes(
    params: dict[str, object],
    spins: jax.Array,
    plan: StrictSchurSpatialPlan,
) -> jax.Array:
    """Evaluate a batch using geometry-generated strict reduced tensors."""

    return strict_schur_amplitudes(
        static_schur_reduced_blocks(params, plan), spins, plan
    )


def strict_schur_parameter_count(params: object) -> int:
    """Count real trainable degrees in reduced blocks or a generator pytree."""

    return int(
        sum(
            np.prod(np.asarray(leaf).shape)
            * (2 if np.issubdtype(np.asarray(leaf).dtype, np.complexfloating) else 1)
            for leaf in jax.tree_util.tree_leaves(params)
        )
    )


__all__ = [
    "FourPlaquetteSchurPlan",
    "ReducedSchurMergeBlock",
    "SchurCell",
    "StrictSchurSpatialPlan",
    "four_plaquette_amplitude",
    "four_plaquette_amplitude_from_spinors",
    "four_plaquette_amplitudes",
    "four_plaquette_d4_plan",
    "init_four_plaquette_reduced",
    "init_static_schur_generator",
    "init_strict_schur_reduced",
    "square_plaquette_plan",
    "static_schur_reduced_blocks",
    "strict_schur_amplitude",
    "strict_schur_amplitude_from_spinors",
    "strict_schur_amplitudes",
    "strict_schur_parameter_count",
    "strict_schur_spatial_amplitudes",
    "strict_schur_spatial_descriptors",
    "two_plaquette_plan",
    "two_triangle_plan",
]
