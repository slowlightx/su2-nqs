"""Strict shared Clebsch tree built from complete three-spin Schur patches.

This module is the first scalable continuation of the exact-small strict
Schur coefficient oracle.  A physical configuration is divided into
disjoint three-site patches.  Every patch retains the complete decomposition

``(V_1) ** tensor 3 = 2 V_1 + V_3``

in doubled-spin notation.  Fixed Condon--Shortley Clebsch--Gordan tensors
then join only disjoint physical supports.  Trainable arrays act exclusively
on reduced multiplicity axes.  Consequently an intermediate-spin cutoff and
a finite reduced width change variational expressivity but never break exact
global SU(2) or physical-leg multi-affinity.

The registered ``6x6`` construction uses twelve compact triangles for both
the square ``J1-J2`` and triangular geometries.  Pairs first form six-site
blocks; all later non-root reduced maps share one tensor bank across position
and scale.  The final root selects ``q=m=0``.  Geometry is metadata here: a
future static spatial generator may produce the reduced banks, but no API in
this module lets a nonlinear network consume a spin configuration.

All angular momenta and magnetic projections are doubled integers.  Bits use
the repository convention ``0=down``, ``1=up``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from typing import Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.fusion_catalog import DisjointMergeRoute, FusionSupport
from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.local_schur import cell_schur_spec, cell_schur_tensor
from ..su2.quantum_numbers import coupling_range, magnetic_numbers, validate_q


jax.config.update("jax_enable_x64", True)


Geometry = Literal["square", "triangular"]


@dataclass(frozen=True)
class TriangleSchurPatch:
    """One ordered, complete three-spin Schur carrier."""

    sites: tuple[int, int, int]

    def __post_init__(self) -> None:
        raw = tuple(self.sites)
        if len(raw) != 3:
            raise ValueError("a triangle Schur patch requires exactly three sites")
        if any(isinstance(site, bool) or not isinstance(site, Integral) for site in raw):
            raise TypeError("triangle-patch site labels must be integers")
        checked = tuple(int(site) for site in raw)
        if any(site < 0 for site in checked):
            raise ValueError("triangle-patch site labels must be nonnegative")
        if len(set(checked)) != 3:
            raise ValueError("a triangle Schur patch cannot repeat a physical site")
        object.__setattr__(self, "sites", checked)

    @property
    def support(self) -> FusionSupport:
        """Return the canonical support while retaining ``sites`` orientation."""

        return FusionSupport(self.sites)


@dataclass(frozen=True, order=True)
class TriangleReducedMergeBlock:
    """One packed reduced tensor block ``q_left x q_right -> q_parent``."""

    q_left: int
    q_right: int
    q_parent: int
    left_width: int
    right_width: int
    output_width: int

    @property
    def key(self) -> tuple[int, int, int]:
        return self.q_left, self.q_right, self.q_parent

    @property
    def shape(self) -> tuple[int, int, int]:
        """Use the fixed ``(output, left, right)`` reduced-index order."""

        return self.output_width, self.left_width, self.right_width


@dataclass(frozen=True, order=True)
class TriangleRootBlock:
    """One reduced root block coupling equal child irreps to a singlet."""

    q: int
    left_width: int
    right_width: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.left_width, self.right_width


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _retained_q(size: int, maximum_q: int) -> tuple[int, ...]:
    upper = min(int(size), int(maximum_q))
    return tuple(range(size % 2, upper + 1, 2))


@dataclass(frozen=True)
class TrianglePatchClebschTreePlan:
    """Typed support ledger and shared reduced-width contract.

    ``routes`` are in topological order and form a strict binary tree.  Every
    non-root merge must join either two patch leaves or two composite
    supports.  Patch-pair maps share one bank; every later non-root
    composite merge shares a second bank, including across physical scale.
    The root has its own singlet blocks.
    """

    geometry: Geometry
    rows: int
    columns: int
    patches: tuple[TriangleSchurPatch, ...]
    routes: tuple[DisjointMergeRoute, ...]
    feature_dim: int = 4
    maximum_intermediate_q: int = 4
    target_q: int = 0
    target_m2: int = 0

    def __post_init__(self) -> None:
        if self.geometry not in {"square", "triangular"}:
            raise ValueError("geometry must be 'square' or 'triangular'")
        rows = _positive_integer("rows", self.rows)
        columns = _positive_integer("columns", self.columns)
        feature_dim = _positive_integer("feature_dim", self.feature_dim)
        if (
            isinstance(self.maximum_intermediate_q, bool)
            or not isinstance(self.maximum_intermediate_q, Integral)
        ):
            raise TypeError("maximum_intermediate_q must be a doubled integer")
        maximum_q = validate_q(
            int(self.maximum_intermediate_q), name="maximum_intermediate_q"
        )
        for name, value in (("target_q", self.target_q), ("target_m2", self.target_m2)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer doubled quantum number")
        if int(self.target_q) != 0 or int(self.target_m2) != 0:
            raise ValueError("the shared triangle-patch tree currently targets q=m2=0")

        patches = tuple(self.patches)
        if len(patches) < 2 or any(
            not isinstance(patch, TriangleSchurPatch) for patch in patches
        ):
            raise TypeError("patches must contain at least two TriangleSchurPatch values")
        n_sites = rows * columns
        flattened = tuple(site for patch in patches for site in patch.sites)
        if tuple(sorted(flattened)) != tuple(range(n_sites)):
            raise ValueError(
                "triangle patches must cover physical sites 0..N-1 exactly once"
            )
        patch_supports = tuple(patch.support for patch in patches)
        if len(set(patch_supports)) != len(patch_supports):
            raise ValueError("triangle-patch supports must be duplicate free")

        routes = tuple(self.routes)
        if len(routes) != len(patches) - 1 or any(
            not isinstance(route, DisjointMergeRoute) for route in routes
        ):
            raise TypeError(
                "routes must contain exactly patch_count-1 DisjointMergeRoute values"
            )
        all_sites = FusionSupport(tuple(range(n_sites)))
        live = set(patch_supports)
        leaf_set = set(patch_supports)
        for index, route in enumerate(routes):
            if route.parent.sites[-1] >= n_sites:
                raise ValueError("a merge route contains a site outside the lattice")
            if route.left not in live or route.right not in live:
                raise ValueError(
                    "merge routes must consume two live disjoint supports in topological order"
                )
            if route.parent in live:
                raise ValueError("a merge parent support must be created exactly once")
            left_is_patch = route.left in leaf_set
            right_is_patch = route.right in leaf_set
            if index < len(routes) - 1 and left_is_patch != right_is_patch:
                raise ValueError(
                    "a non-root merge cannot mix one patch and one composite support"
                )
            live.remove(route.left)
            live.remove(route.right)
            live.add(route.parent)
        if live != {all_sites} or routes[-1].parent != all_sites:
            raise ValueError("the support ledger must terminate at the all-site root")

        # Compile eagerly.  This fails closed if a cutoff deletes every
        # channel needed by any registered non-root or root merge.
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "patches", patches)
        object.__setattr__(self, "routes", routes)
        object.__setattr__(self, "feature_dim", feature_dim)
        object.__setattr__(self, "maximum_intermediate_q", maximum_q)
        object.__setattr__(self, "target_q", 0)
        object.__setattr__(self, "target_m2", 0)
        if any(
            not _channels_for_route(self, route)
            for route in self.non_root_routes
        ):
            raise ValueError("the intermediate-spin cutoff removes a required merge")
        if not triangle_root_layout(self):
            raise ValueError("the intermediate-spin cutoff removes the singlet root")

    @property
    def n_sites(self) -> int:
        return self.rows * self.columns

    @property
    def root_route(self) -> DisjointMergeRoute:
        return self.routes[-1]

    @property
    def non_root_routes(self) -> tuple[DisjointMergeRoute, ...]:
        return self.routes[:-1]

    @property
    def patch_supports(self) -> tuple[FusionSupport, ...]:
        return tuple(patch.support for patch in self.patches)

    @property
    def patch_pair_shapes(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(layout.shape for layout in triangle_patch_pair_layout(self))

    @property
    def reduced_shapes(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(layout.shape for layout in triangle_shared_reduced_layout(self))

    @property
    def root_shapes(self) -> tuple[tuple[int, int], ...]:
        return tuple(layout.shape for layout in triangle_root_layout(self))

    @property
    def patch_pair_routes(self) -> tuple[DisjointMergeRoute, ...]:
        """Non-root patch merges, each carrying a small static route gate."""

        leaves = set(self.patch_supports)
        return tuple(
            route
            for route in self.non_root_routes
            if route.left in leaves and route.right in leaves
        )

    @property
    def shared_reduced_routes(self) -> tuple[DisjointMergeRoute, ...]:
        """Composite non-root merges using the scale-shared tensor bank."""

        leaves = set(self.patch_supports)
        return tuple(
            route
            for route in self.non_root_routes
            if route.left not in leaves and route.right not in leaves
        )

    @property
    def patch_gate_shapes(self) -> tuple[tuple[int], ...]:
        return tuple(
            (len(_channels_for_route(self, route)),)
            for route in self.patch_pair_routes
        )

    @property
    def reduced_gate_shapes(self) -> tuple[tuple[int], ...]:
        return tuple(
            (len(_channels_for_route(self, route)),)
            for route in self.shared_reduced_routes
        )

    @property
    def parameter_count(self) -> int:
        return int(
            sum(np.prod(shape) for shape in self.patch_pair_shapes)
            + sum(np.prod(shape) for shape in self.reduced_shapes)
            + sum(np.prod(shape) for shape in self.root_shapes)
            + sum(np.prod(shape) for shape in self.patch_gate_shapes)
            + sum(np.prod(shape) for shape in self.reduced_gate_shapes)
        )


def _is_patch_support(
    plan: TrianglePatchClebschTreePlan, support: FusionSupport
) -> bool:
    return support in set(plan.patch_supports)


def _support_q(
    plan: TrianglePatchClebschTreePlan, support: FusionSupport
) -> tuple[int, ...]:
    if _is_patch_support(plan, support):
        return cell_schur_spec("triangular_triangle").q_values
    return _retained_q(len(support), plan.maximum_intermediate_q)


def _support_width(
    plan: TrianglePatchClebschTreePlan, support: FusionSupport, q: int
) -> int:
    if _is_patch_support(plan, support):
        return cell_schur_spec("triangular_triangle").multiplicity(q)
    return plan.feature_dim


def _channels_for_route(
    plan: TrianglePatchClebschTreePlan,
    route: DisjointMergeRoute,
) -> tuple[TriangleReducedMergeBlock, ...]:
    output_q = set(_retained_q(len(route.parent), plan.maximum_intermediate_q))
    blocks = []
    for q_left in _support_q(plan, route.left):
        for q_right in _support_q(plan, route.right):
            for q_parent in coupling_range(q_left, q_right):
                if q_parent not in output_q:
                    continue
                blocks.append(
                    TriangleReducedMergeBlock(
                        q_left,
                        q_right,
                        q_parent,
                        _support_width(plan, route.left, q_left),
                        _support_width(plan, route.right, q_right),
                        plan.feature_dim,
                    )
                )
    return tuple(blocks)


def triangle_route_layout(
    plan: TrianglePatchClebschTreePlan,
    route: DisjointMergeRoute,
) -> tuple[TriangleReducedMergeBlock, ...]:
    """Return the deterministic reduced-block layout for one registered route.

    Physical initializers sometimes need to normalize reduced states route by
    route.  Exposing this read-only layout avoids duplicating the support and
    doubled-spin cutoff rules outside the strict model.  The route must be an
    actual member of ``plan.routes`` so a caller cannot silently compile a
    tensor contract that the forward pass never uses.
    """

    if route not in plan.routes:
        raise ValueError("route is not registered in the triangle-patch plan")
    return _channels_for_route(plan, route)


@lru_cache(maxsize=None)
def triangle_patch_pair_layout(
    plan: TrianglePatchClebschTreePlan,
) -> tuple[TriangleReducedMergeBlock, ...]:
    """Return the shared complete-patch pair tensor bank."""

    leaf_set = set(plan.patch_supports)
    candidates = [
        route
        for route in plan.non_root_routes
        if route.left in leaf_set and route.right in leaf_set
    ]
    if not candidates:
        return ()
    layouts = {_block.key: _block for route in candidates for _block in _channels_for_route(plan, route)}
    return tuple(layouts[key] for key in sorted(layouts))


@lru_cache(maxsize=None)
def triangle_shared_reduced_layout(
    plan: TrianglePatchClebschTreePlan,
) -> tuple[TriangleReducedMergeBlock, ...]:
    """Return the one bank shared by every composite non-root merge."""

    leaf_set = set(plan.patch_supports)
    candidates = [
        route
        for route in plan.non_root_routes
        if route.left not in leaf_set and route.right not in leaf_set
    ]
    if not candidates:
        return ()
    layouts = {_block.key: _block for route in candidates for _block in _channels_for_route(plan, route)}
    return tuple(layouts[key] for key in sorted(layouts))


@lru_cache(maxsize=None)
def triangle_root_layout(
    plan: TrianglePatchClebschTreePlan,
) -> tuple[TriangleRootBlock, ...]:
    """Return root blocks ``q tensor q -> 0`` in deterministic order."""

    route = plan.root_route
    common = sorted(set(_support_q(plan, route.left)) & set(_support_q(plan, route.right)))
    return tuple(
        TriangleRootBlock(
            q,
            _support_width(plan, route.left, q),
            _support_width(plan, route.right, q),
        )
        for q in common
    )


def triangle_patch_grid_plan(
    geometry: Geometry,
    rows: int,
    columns: int,
    *,
    feature_dim: int = 4,
    maximum_intermediate_q: int = 4,
) -> TrianglePatchClebschTreePlan:
    """Tile an even-by-three-divisible torus with compact disjoint triangles.

    Each ``2x3`` block has top row ``a,b,c`` and bottom row ``d,e,f``.  The
    square geometry uses ``(a,b,d)`` and ``(c,e,f)``, two right triangles
    carrying two nearest-neighbour and one diagonal edge.  The repository's
    triangular coordinates use 120-degree primitive vectors, whose third
    nearest-neighbour direction is ``(1,1)``.  Its disjoint elementary-
    triangle cover is therefore ``(a,d,e)`` and ``(b,c,f)``.  Selecting the
    support by geometry is essential: reusing the square split would put a
    second-neighbour edge inside every nominal triangular patch.
    """

    checked_rows = _positive_integer("rows", rows)
    checked_columns = _positive_integer("columns", columns)
    if checked_rows % 2 or checked_columns % 3:
        raise ValueError("triangle patches require rows divisible by 2 and columns by 3")

    patches: list[TriangleSchurPatch] = []
    block_supports: list[FusionSupport] = []
    routes: list[DisjointMergeRoute] = []
    for row in range(0, checked_rows, 2):
        for column in range(0, checked_columns, 3):
            a = row * checked_columns + column
            b = a + 1
            c = a + 2
            d = (row + 1) * checked_columns + column
            e = d + 1
            f = d + 2
            if geometry == "square":
                left_patch = TriangleSchurPatch((a, b, d))
                right_patch = TriangleSchurPatch((c, e, f))
            else:
                left_patch = TriangleSchurPatch((a, d, e))
                right_patch = TriangleSchurPatch((b, c, f))
            patches.extend((left_patch, right_patch))
            parent = FusionSupport(left_patch.sites + right_patch.sites)
            routes.append(
                DisjointMergeRoute(parent, left_patch.support, right_patch.support)
            )
            block_supports.append(parent)

    # For one 2x3 block the patch-pair route is itself the root.  Otherwise
    # all patch pairs are non-root and the compact blocks are reduced in a
    # deterministic balanced hierarchy.  An unpaired support is carried to
    # the next level without introducing a fake identity map.
    current = block_supports
    while len(current) > 1:
        following: list[FusionSupport] = []
        for index in range(0, len(current), 2):
            if index + 1 == len(current):
                following.append(current[index])
                continue
            left, right = current[index : index + 2]
            parent = FusionSupport(left.sites + right.sites)
            routes.append(DisjointMergeRoute(parent, left, right))
            following.append(parent)
        current = following

    return TrianglePatchClebschTreePlan(
        geometry=geometry,
        rows=checked_rows,
        columns=checked_columns,
        patches=tuple(patches),
        routes=tuple(routes),
        feature_dim=feature_dim,
        maximum_intermediate_q=maximum_intermediate_q,
    )


def triangle_patch_6x6_plan(
    geometry: Geometry,
    *,
    feature_dim: int = 4,
    maximum_intermediate_q: int = 4,
) -> TrianglePatchClebschTreePlan:
    """Return the common square/triangular twelve-patch ``6x6`` plan."""

    return triangle_patch_grid_plan(
        geometry,
        6,
        6,
        feature_dim=feature_dim,
        maximum_intermediate_q=maximum_intermediate_q,
    )


def triangle_patch_6x6_canary_plan(
    geometry: Geometry,
) -> TrianglePatchClebschTreePlan:
    """Return the first sampling-safe local canary, ``D=2, qmax=10``.

    Smaller cutoffs remain useful algebra and throughput stresses, but they
    delete all channels compatible with a material fraction of generic
    fixed-``M=0`` configurations.  A deterministic 2048-state support gate
    selects ``qmax=10`` as the smallest registered zero-free candidate; this
    is a finite diagnostic, not a proof that the variational state has no
    physical nodes.
    """

    return triangle_patch_6x6_plan(
        geometry, feature_dim=2, maximum_intermediate_q=10
    )


def _bank_shapes(
    plan: TrianglePatchClebschTreePlan,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    return {
        "patch_pair": plan.patch_pair_shapes,
        "reduced": plan.reduced_shapes,
        "root": plan.root_shapes,
        "patch_gates": plan.patch_gate_shapes,
        "reduced_gates": plan.reduced_gate_shapes,
    }


def init_triangle_patch_clebsch_tree(
    key: jax.Array,
    plan: TrianglePatchClebschTreePlan,
    *,
    scale: float = 0.2,
    route_gate_scale: float = 0.05,
) -> dict[str, tuple[jax.Array, ...]]:
    """Initialize shared reduced tensors and configuration-independent gates.

    Each merge occurrence has one scalar per active angular-momentum triple.
    Small random deviations from one break accidental exchange constraints of
    otherwise identical subtrees.  The gates act only on reduced tensors and
    therefore preserve exact SU(2) and multi-affinity.
    """

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    if not np.isfinite(route_gate_scale) or route_gate_scale < 0.0:
        raise ValueError("route_gate_scale must be finite and nonnegative")
    shapes = _bank_shapes(plan)
    count = sum(len(group) for group in shapes.values())
    keys = iter(jax.random.split(key, count))
    params: dict[str, tuple[jax.Array, ...]] = {}
    for name in ("patch_pair", "reduced", "root"):
        blocks = []
        for shape in shapes[name]:
            fan_in = max(
                int(np.prod(shape if name == "root" else shape[1:])), 1
            )
            blocks.append(
                (float(scale) / np.sqrt(fan_in))
                * jax.random.normal(next(keys), shape, dtype=jnp.float64)
            )
        params[name] = tuple(blocks)
    for name in ("patch_gates", "reduced_gates"):
        params[name] = tuple(
            jnp.ones(shape, dtype=jnp.float64)
            + float(route_gate_scale)
            * jax.random.normal(next(keys), shape, dtype=jnp.float64)
            for shape in shapes[name]
        )
    return params


def _validate_params(
    params: Mapping[str, object],
    plan: TrianglePatchClebschTreePlan,
) -> dict[str, tuple[jax.Array, ...]]:
    if not isinstance(params, Mapping) or set(params) != {
        "patch_pair",
        "reduced",
        "root",
        "patch_gates",
        "reduced_gates",
    }:
        raise ValueError(
            "parameters require patch_pair, reduced, root, and route-gate banks"
        )
    checked: dict[str, tuple[jax.Array, ...]] = {}
    for name, shapes in _bank_shapes(plan).items():
        raw = params[name]
        if not isinstance(raw, (tuple, list)) or len(raw) != len(shapes):
            raise ValueError(f"{name} reduced blocks do not match the plan")
        blocks = tuple(jnp.asarray(block) for block in raw)
        if tuple(block.shape for block in blocks) != shapes:
            raise ValueError(f"a {name} reduced block has the wrong shape")
        if any(not jnp.issubdtype(block.dtype, jnp.inexact) for block in blocks):
            raise TypeError("reduced tensors must have real or complex dtype")
        checked[name] = blocks
    return checked


@lru_cache(maxsize=None)
def _triangle_basis_bits() -> np.ndarray:
    states = np.asarray(
        [
            [(state >> (2 - site)) & 1 for site in range(3)]
            for state in range(8)
        ],
        dtype=np.int32,
    )
    states.setflags(write=False)
    return states


def _triangle_carrier(
    spinors: jax.Array,
    patch: TriangleSchurPatch,
) -> dict[int, jax.Array]:
    local = spinors[jnp.asarray(patch.sites, dtype=jnp.int32)]
    states = jnp.asarray(_triangle_basis_bits(), dtype=jnp.int32)
    product = jnp.ones((8,), dtype=spinors.dtype)
    for site in range(3):
        product = product * local[site, states[:, site]]
    table = jnp.asarray(
        cell_schur_tensor("triangular_triangle"), dtype=spinors.dtype
    )
    dense = jnp.einsum("s,sqma->qma", product, table)
    spec = cell_schur_spec("triangular_triangle")
    return {
        q: dense[
            q,
            jnp.asarray(
                [m2 + spec.cell_size for m2 in magnetic_numbers(q)],
                dtype=jnp.int32,
            ),
            : spec.multiplicity(q),
        ]
        for q in spec.q_values
    }


@lru_cache(maxsize=None)
def _compact_cg_tensor(q_left: int, q_right: int, q_parent: int) -> np.ndarray:
    """Return ``C[parent_m,left_m,right_m]`` without support-size padding."""

    left_m = magnetic_numbers(q_left)
    right_m = magnetic_numbers(q_right)
    parent_m = magnetic_numbers(q_parent)
    positions = {m2: index for index, m2 in enumerate(parent_m)}
    result = np.zeros(
        (len(parent_m), len(left_m), len(right_m)), dtype=np.float64
    )
    for left_index, m_left in enumerate(left_m):
        for right_index, m_right in enumerate(right_m):
            total = m_left + m_right
            if total not in positions:
                continue
            result[
                positions[total], left_index, right_index
            ] = clebsch_gordan(
                q_left,
                m_left,
                q_right,
                m_right,
                q_parent,
                total,
            )
    result.setflags(write=False)
    return result


def _merge_carriers(
    left: dict[int, jax.Array],
    right: dict[int, jax.Array],
    route: DisjointMergeRoute,
    blocks: tuple[jax.Array, ...],
    layouts: tuple[TriangleReducedMergeBlock, ...],
    plan: TrianglePatchClebschTreePlan,
) -> dict[int, jax.Array]:
    if len(blocks) != len(layouts):
        raise RuntimeError("reduced bank and layout differ")
    dtypes = [value.dtype for value in left.values()]
    dtypes.extend(value.dtype for value in right.values())
    dtypes.extend(block.dtype for block in blocks)
    dtype = jnp.result_type(*dtypes)
    retained = _retained_q(len(route.parent), plan.maximum_intermediate_q)
    result = {
        q: jnp.zeros((q + 1, plan.feature_dim), dtype=dtype) for q in retained
    }
    for layout, block in zip(layouts, blocks, strict=True):
        if (
            layout.q_left not in left
            or layout.q_right not in right
            or layout.q_parent not in result
        ):
            continue
        cg = jnp.asarray(
            _compact_cg_tensor(
                layout.q_left, layout.q_right, layout.q_parent
            ),
            dtype=dtype,
        )
        contribution = jnp.einsum(
            "la,rb,cab,plr->pc",
            left[layout.q_left],
            right[layout.q_right],
            block,
            cg,
        )
        result[layout.q_parent] = result[layout.q_parent] + contribution
    return result


def triangle_patch_amplitude_from_spinors(
    params: Mapping[str, object],
    spinors: jax.Array,
    plan: TrianglePatchClebschTreePlan,
) -> jax.Array:
    """Evaluate the exact-singlet multilinear polynomial on physical spinors."""

    checked = _validate_params(params, plan)
    raw = jnp.asarray(spinors)
    values = jnp.asarray(raw, dtype=jnp.result_type(raw.dtype, jnp.float64))
    if values.shape != (plan.n_sites, 2):
        raise ValueError(f"expected spinor shape ({plan.n_sites}, 2)")

    carriers: dict[FusionSupport, dict[int, jax.Array]] = {
        patch.support: _triangle_carrier(values, patch) for patch in plan.patches
    }
    patch_set = set(plan.patch_supports)
    patch_layout = triangle_patch_pair_layout(plan)
    reduced_layout = triangle_shared_reduced_layout(plan)
    patch_blocks = {
        layout.key: block
        for layout, block in zip(
            patch_layout, checked["patch_pair"], strict=True
        )
    }
    reduced_blocks = {
        layout.key: block
        for layout, block in zip(
            reduced_layout, checked["reduced"], strict=True
        )
    }
    patch_route_index = 0
    reduced_route_index = 0
    for route in plan.non_root_routes:
        is_patch_pair = route.left in patch_set and route.right in patch_set
        layouts = _channels_for_route(plan, route)
        if is_patch_pair:
            gate = checked["patch_gates"][patch_route_index]
            source = patch_blocks
            patch_route_index += 1
        else:
            gate = checked["reduced_gates"][reduced_route_index]
            source = reduced_blocks
            reduced_route_index += 1
        blocks = tuple(
            source[layout.key] * gate[index]
            for index, layout in enumerate(layouts)
        )
        carriers[route.parent] = _merge_carriers(
            carriers[route.left],
            carriers[route.right],
            route,
            blocks,
            layouts,
            plan,
        )

    root = plan.root_route
    left = carriers[root.left]
    right = carriers[root.right]
    result_dtype = jnp.result_type(
        *(value.dtype for value in left.values()),
        *(value.dtype for value in right.values()),
        *(block.dtype for block in checked["root"]),
    )
    result = jnp.asarray(0.0, dtype=result_dtype)
    for layout, block in zip(
        triangle_root_layout(plan), checked["root"], strict=True
    ):
        cg = jnp.asarray(
            _compact_cg_tensor(layout.q, layout.q, 0)[0],
            dtype=result_dtype,
        )
        result = result + jnp.einsum(
            "la,rb,ab,lr->",
            left[layout.q],
            right[layout.q],
            block,
            cg,
        )
    return result


def triangle_patch_amplitude(
    params: Mapping[str, object],
    spins: jax.Array,
    plan: TrianglePatchClebschTreePlan,
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
    value = triangle_patch_amplitude_from_spinors(params, spinors, plan)
    in_sector = jnp.sum(safe) == plan.n_sites // 2
    return jnp.where(
        valid & in_sector, value, jnp.asarray(0.0, dtype=value.dtype)
    )


def triangle_patch_amplitudes(
    params: Mapping[str, object],
    spins: jax.Array,
    plan: TrianglePatchClebschTreePlan,
) -> jax.Array:
    """Vectorize :func:`triangle_patch_amplitude` over configurations."""

    array = jnp.asarray(spins)
    if array.ndim != 2 or array.shape[1] != plan.n_sites:
        raise ValueError(f"expected spin batch shape (batch, {plan.n_sites})")
    if array.dtype == jnp.bool_ or not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError("computational-basis spins must have an integer dtype")
    array = array.astype(jnp.int32)
    return jax.vmap(lambda row: triangle_patch_amplitude(params, row, plan))(array)


def triangle_patch_parameter_count(params: Mapping[str, object]) -> int:
    """Count real trainable scalars, counting a complex entry twice."""

    return int(
        sum(
            np.prod(np.asarray(leaf).shape)
            * (
                2
                if np.issubdtype(np.asarray(leaf).dtype, np.complexfloating)
                else 1
            )
            for leaf in jax.tree_util.tree_leaves(params)
        )
    )


__all__ = [
    "TrianglePatchClebschTreePlan",
    "TriangleReducedMergeBlock",
    "TriangleRootBlock",
    "TriangleSchurPatch",
    "init_triangle_patch_clebsch_tree",
    "triangle_patch_6x6_plan",
    "triangle_patch_6x6_canary_plan",
    "triangle_patch_amplitude",
    "triangle_patch_amplitude_from_spinors",
    "triangle_patch_amplitudes",
    "triangle_patch_grid_plan",
    "triangle_patch_pair_layout",
    "triangle_patch_parameter_count",
    "triangle_route_layout",
    "triangle_root_layout",
    "triangle_shared_reduced_layout",
]
