"""Physics-led initialization for the strict triangle-patch Clebsch tree.

The model itself contains no Hamiltonian logic.  This frontier module composes
its exact local Schur carriers with a six-site exact reference calculation.
For every ``2 x 3`` block, feature channel zero is the normalized ground
singlet of the open cluster at the requested ``J2/J1``.  The upper tree then
propagates only ``q=0, channel=0``, producing a product of six-site singlets.

Feature channel one is a separately normalized, exactly orthogonal generic
singlet tree before spatial projection.  The unprojected root therefore
represents the transparent state

``psi = psi_cluster + background_scale * psi_background``.

Both branches use only reduced multiplicity tensors.  Their sum consequently
retains exact global SU(2) and physical-leg multi-affinity.  In particular,
no configuration-dependent Marshall sign is attached after the fact.

This is a finite-torus, short-range-RVB *causal initialization arm*: the
translation projector can resonate its cluster coverings before training.
It is not the same variational family as a fully symmetric ``Z2`` RVB iPEPS,
and neither topological order nor an iPEPS bond dimension is implied by the
construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Integral
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..benchmarks.exact_diagonalization import heisenberg_sz_matrix
from ..lattices.j1j2 import square_j1j2_bonds
from ..lattices.spatial_symmetry import (
    rectangular_translation_group,
    triangular_hnf_translation_group,
)
from ..lattices.triangular_j1j2 import triangular_j1j2_bonds
from ..models.triangle_patch_clebsch_tree import (
    TrianglePatchClebschTreePlan,
    triangle_patch_pair_layout,
    triangle_root_layout,
    triangle_route_layout,
    triangle_shared_reduced_layout,
)
from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.local_schur import cell_schur_spec, cell_schur_tensor
from ..su2.quantum_numbers import magnetic_numbers


jax.config.update("jax_enable_x64", True)


Geometry = Literal["square", "triangular"]


@dataclass(frozen=True)
class TrianglePatchClusterReference:
    """Exact open-``2 x 3`` singlet expressed in the two-patch Schur basis."""

    geometry: Geometry
    j2_over_j1: float
    bonds: tuple[tuple[int, int, float], ...]
    energy: float
    singlet_gap: float
    coefficients: np.ndarray
    computational_state: np.ndarray
    basis_orthonormality_error: float
    eigen_residual: float


def _checked_geometry(geometry: str) -> Geometry:
    if geometry not in {"square", "triangular"}:
        raise ValueError("geometry must be 'square' or 'triangular'")
    return geometry


def _checked_finite(name: str, value: float, *, nonnegative: bool) -> float:
    checked = float(value)
    if not math.isfinite(checked) or (nonnegative and checked < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return checked


def _patch_local_sites(geometry: Geometry) -> tuple[tuple[int, int, int], ...]:
    # Canonical local row-major order is (a,b,c,d,e,f).
    if geometry == "square":
        return ((0, 1, 3), (2, 4, 5))
    return ((0, 3, 4), (1, 2, 5))


def _bits_to_index(bits: np.ndarray) -> np.ndarray:
    return (
        4 * bits[:, 0].astype(np.int64)
        + 2 * bits[:, 1].astype(np.int64)
        + bits[:, 2].astype(np.int64)
    )


@lru_cache(maxsize=None)
def _cluster_singlet_basis(geometry: Geometry) -> np.ndarray:
    """Return five orthonormal six-spin singlets in patch-layout order."""

    geometry = _checked_geometry(geometry)
    states = np.arange(64, dtype=np.int64)
    bits = np.asarray(
        [[(state >> (5 - site)) & 1 for site in range(6)] for state in states],
        dtype=np.int32,
    )
    left_sites, right_sites = _patch_local_sites(geometry)
    left_index = _bits_to_index(bits[:, left_sites])
    right_index = _bits_to_index(bits[:, right_sites])
    table = cell_schur_tensor("triangular_triangle")
    spec = cell_schur_spec("triangular_triangle")
    columns: list[np.ndarray] = []
    for q in spec.q_values:
        multiplicity = spec.multiplicity(q)
        for left_a in range(multiplicity):
            for right_a in range(multiplicity):
                column = np.zeros(64, dtype=np.float64)
                for left_m in magnetic_numbers(q):
                    right_m = -left_m
                    column += (
                        table[left_index, q, left_m + 3, left_a]
                        * table[right_index, q, right_m + 3, right_a]
                        * clebsch_gordan(q, left_m, q, right_m, 0, 0)
                    )
                columns.append(column)
    basis = np.column_stack(columns)
    if basis.shape != (64, 5):
        raise RuntimeError("the two complete triangle carriers must span five singlets")
    gram_error = float(np.linalg.norm(basis.T @ basis - np.eye(5)))
    if gram_error > 2.0e-12:
        raise RuntimeError("the local two-triangle singlet basis is not orthonormal")
    basis.setflags(write=False)
    return basis


@lru_cache(maxsize=None)
def _cluster_reference_cached(
    geometry: Geometry,
    j2_over_j1: float,
) -> TrianglePatchClusterReference:
    geometry = _checked_geometry(geometry)
    ratio = _checked_finite("j2_over_j1", j2_over_j1, nonnegative=True)
    bonds = (
        square_j1j2_bonds(2, 3, j1=1.0, j2=ratio, periodic=False)
        if geometry == "square"
        else triangular_j1j2_bonds(2, 3, j1=1.0, j2=ratio, periodic=False)
    )
    hamiltonian, integer_basis = heisenberg_sz_matrix(6, bonds)
    if integer_basis != tuple(range(64)):
        raise RuntimeError("the local Hamiltonian changed computational-basis order")
    singlets = _cluster_singlet_basis(geometry)
    reduced = singlets.T @ hamiltonian @ singlets
    reduced = 0.5 * (reduced + reduced.T)
    values, vectors = np.linalg.eigh(reduced)
    coefficients = np.asarray(vectors[:, 0], dtype=np.float64)
    pivot = int(np.argmax(np.abs(coefficients)))
    if coefficients[pivot] < 0.0:
        coefficients *= -1.0
    state = np.asarray(singlets @ coefficients, dtype=np.float64)
    residual = float(np.linalg.norm(hamiltonian @ state - values[0] * state))
    gram_error = float(np.linalg.norm(singlets.T @ singlets - np.eye(5)))
    if residual > 3.0e-12 or abs(np.linalg.norm(state) - 1.0) > 2.0e-13:
        raise RuntimeError("failed to construct the normalized cluster eigenstate")
    coefficients.setflags(write=False)
    state.setflags(write=False)
    return TrianglePatchClusterReference(
        geometry=geometry,
        j2_over_j1=ratio,
        bonds=tuple(bonds),
        energy=float(values[0]),
        singlet_gap=float(values[1] - values[0]),
        coefficients=coefficients,
        computational_state=state,
        basis_orthonormality_error=gram_error,
        eigen_residual=residual,
    )


def triangle_patch_cluster_reference(
    geometry: Geometry,
    j2_over_j1: float,
) -> TrianglePatchClusterReference:
    """Return a defensive copy of the deterministic six-site reference."""

    reference = _cluster_reference_cached(
        _checked_geometry(geometry),
        _checked_finite("j2_over_j1", j2_over_j1, nonnegative=True),
    )
    coefficients = np.array(reference.coefficients, copy=True)
    state = np.array(reference.computational_state, copy=True)
    coefficients.setflags(write=False)
    state.setflags(write=False)
    return TrianglePatchClusterReference(
        geometry=reference.geometry,
        j2_over_j1=reference.j2_over_j1,
        bonds=reference.bonds,
        energy=reference.energy,
        singlet_gap=reference.singlet_gap,
        coefficients=coefficients,
        computational_state=state,
        basis_orthonormality_error=reference.basis_orthonormality_error,
        eigen_residual=reference.eigen_residual,
    )


def triangle_patch_cluster_six_site_probabilities(
    geometry: Geometry,
    j2_over_j1: float,
) -> np.ndarray:
    """Return the canonical local cluster Born probabilities on 64 bitstrings."""

    reference = _cluster_reference_cached(
        _checked_geometry(geometry),
        _checked_finite("j2_over_j1", j2_over_j1, nonnegative=True),
    )
    probabilities = np.square(reference.computational_state)
    probabilities[np.abs(reference.computational_state) <= 2.0e-15] = 0.0
    probabilities /= np.sum(probabilities)
    probabilities.setflags(write=False)
    return probabilities


def _cluster_sites(plan: TrianglePatchClebschTreePlan) -> tuple[tuple[int, ...], ...]:
    result = []
    for row in range(0, plan.rows, 2):
        for column in range(0, plan.columns, 3):
            top = row * plan.columns + column
            bottom = (row + 1) * plan.columns + column
            result.append(
                (top, top + 1, top + 2, bottom, bottom + 1, bottom + 2)
            )
    if len(result) * 6 != plan.n_sites:
        raise RuntimeError("the plan is not covered by canonical 2x3 clusters")
    return tuple(result)


def sample_triangle_patch_cluster_product_chains(
    key: jax.Array,
    plan: TrianglePatchClebschTreePlan,
    *,
    j2_over_j1: float,
    n_chains: int,
    translate: bool = True,
) -> jax.Array:
    """Draw independent high-weight starts from the cluster-product branch.

    Every canonical ``2 x 3`` block is sampled independently from the exact
    local ground-state Born distribution.  A uniformly random lattice
    translation is then applied by default.  Translation keeps the fixed
    ``M=0`` sector and places each chain on the support of the coherently
    projected cluster orbit, without claiming that these draws already follow
    the full projected Born distribution.
    """

    if not isinstance(plan, TrianglePatchClebschTreePlan):
        raise TypeError("plan must be a TrianglePatchClebschTreePlan")
    if isinstance(n_chains, bool) or not isinstance(n_chains, Integral):
        raise TypeError("n_chains must be an integer")
    if int(n_chains) < 1:
        raise ValueError("n_chains must be positive")
    if not isinstance(translate, bool):
        raise TypeError("translate must be boolean")
    count = int(n_chains)
    probabilities = jnp.asarray(
        triangle_patch_cluster_six_site_probabilities(
            plan.geometry, j2_over_j1
        ),
        dtype=jnp.float64,
    )
    sample_key, translation_key = jax.random.split(key)
    cluster_sites = _cluster_sites(plan)
    local_states = jax.random.categorical(
        sample_key,
        jnp.log(probabilities),
        shape=(count, len(cluster_sites)),
    )
    local_bits = (
        (local_states[..., None] >> jnp.arange(5, -1, -1, dtype=jnp.int32))
        & 1
    ).astype(jnp.int32)
    states = jnp.zeros((count, plan.n_sites), dtype=jnp.int32)
    for block, sites in enumerate(cluster_sites):
        states = states.at[:, jnp.asarray(sites, dtype=jnp.int32)].set(
            local_bits[:, block]
        )
    if not translate:
        return states
    group = (
        rectangular_translation_group(plan.rows, plan.columns)
        if plan.geometry == "square"
        else triangular_hnf_translation_group(
            ((plan.rows, 0), (0, plan.columns))
        )
    )
    selected = jax.random.randint(translation_key, (count,), 0, len(group))
    images = jnp.asarray(group, dtype=jnp.int32)[selected]
    return jnp.take_along_axis(states, images, axis=1)


def _stable_random(
    key: jax.Array,
    index: int,
    shape: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(
        jax.random.normal(
            jax.random.fold_in(key, index), shape, dtype=jnp.float64
        )
    )
    # Exact or near zeros make subsequent route normalization needlessly ill
    # conditioned.  This transformation keeps random signs but bounds their
    # magnitude away from zero.
    signs = np.where(values < 0.0, -1.0, 1.0)
    return signs * (0.5 + np.abs(values))


def init_triangle_patch_cluster_singlet(
    key: jax.Array,
    plan: TrianglePatchClebschTreePlan,
    *,
    j2_over_j1: float,
    background_scale: float = 0.05,
    route_gate_scale: float = 0.05,
) -> dict[str, tuple[jax.Array, ...]]:
    """Initialize a cluster-singlet branch plus an orthogonal support floor.

    The current contract is deliberately narrow: ``feature_dim`` must be two
    and the plan must contain non-root patch-pair routes.  Channel zero is the
    exact open-``2 x 3`` ground singlet on every block.  Channel one retains
    one normalized generic state for every available block spin and is
    coupled through route-normalized random reduced coefficients.  No channel
    mixing occurs before the root, making ``background_scale`` the exact
    *pre-projection* norm ratio of the orthogonal background and
    cluster-product branches.  A spatial projector can make the projected
    branches non-orthogonal, so the ratio must not be carried through it.
    """

    if not isinstance(plan, TrianglePatchClebschTreePlan):
        raise TypeError("plan must be a TrianglePatchClebschTreePlan")
    if plan.feature_dim != 2:
        raise ValueError("cluster-singlet initialization currently requires feature_dim=2")
    if not plan.patch_pair_routes:
        raise ValueError("cluster-singlet initialization requires non-root patch-pair routes")
    ratio = _checked_finite("j2_over_j1", j2_over_j1, nonnegative=True)
    eta = _checked_finite("background_scale", background_scale, nonnegative=True)
    gate_noise = _checked_finite(
        "route_gate_scale", route_gate_scale, nonnegative=True
    )
    reference = _cluster_reference_cached(plan.geometry, ratio)

    random_index = 0

    def random(shape: tuple[int, ...]) -> np.ndarray:
        nonlocal random_index
        result = _stable_random(key, random_index, shape)
        random_index += 1
        return result

    patch_layout = triangle_patch_pair_layout(plan)
    patch_arrays = [np.zeros(layout.shape, dtype=np.float64) for layout in patch_layout]
    patch_positions = {layout.key: index for index, layout in enumerate(patch_layout)}

    # The local reference column order is (q=1,a,b), followed by the unique
    # (q=3) column, exactly matching the flattened q0 patch blocks.
    offset = 0
    for block_key in ((1, 1, 0), (3, 3, 0)):
        position = patch_positions.get(block_key)
        if position is None:
            raise ValueError("the plan removes a q channel needed by the cluster singlet")
        block = patch_arrays[position]
        size = block.shape[1] * block.shape[2]
        block[0] = reference.coefficients[offset : offset + size].reshape(
            block.shape[1:]
        )
        offset += size
    if offset != reference.coefficients.size:
        raise RuntimeError("cluster coefficients and patch layout disagree")

    # Channel one is normalized jointly across all child-spin blocks for each
    # parent q.  At q=0 it is explicitly orthogonal to the physical cluster
    # ground state, which makes the two global branches exactly orthogonal.
    parent_q_values = sorted({layout.q_parent for layout in patch_layout})
    for q_parent in parent_q_values:
        positions = [
            index
            for index, layout in enumerate(patch_layout)
            if layout.q_parent == q_parent
        ]
        vector = np.concatenate(
            [random(patch_arrays[index].shape[1:]).reshape(-1) for index in positions]
        )
        if q_parent == 0:
            vector -= reference.coefficients * np.dot(
                reference.coefficients, vector
            )
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-12:
            raise RuntimeError("failed to construct an independent background channel")
        vector /= norm
        offset = 0
        for position in positions:
            block = patch_arrays[position]
            size = block.shape[1] * block.shape[2]
            block[1] = vector[offset : offset + size].reshape(block.shape[1:])
            offset += size

    patch_gates = tuple(
        jnp.ones(shape, dtype=jnp.float64) for shape in plan.patch_gate_shapes
    )

    reduced_layout = triangle_shared_reduced_layout(plan)
    reduced_arrays = [np.zeros(layout.shape, dtype=np.float64) for layout in reduced_layout]
    reduced_positions = {layout.key: index for index, layout in enumerate(reduced_layout)}
    generic_weights: dict[tuple[int, int, int], float] = {}
    for layout, block in zip(reduced_layout, reduced_arrays, strict=True):
        weight = float(random((1,))[0])
        generic_weights[layout.key] = weight
        block[1, 1, 1] = weight
    physical_position = reduced_positions.get((0, 0, 0))
    if reduced_layout and physical_position is None:
        raise ValueError("the plan removes the q=0 propagation channel")
    if physical_position is not None:
        reduced_arrays[physical_position][0, 0, 0] = 1.0

    physical_scalars = {
        route.parent: 1.0 for route in plan.patch_pair_routes
    }
    reduced_gates: list[jax.Array] = []
    for route_index, route in enumerate(plan.shared_reduced_routes):
        layouts = triangle_route_layout(plan, route)
        route_modulation = np.exp(
            gate_noise * random((len(layouts),))
        )
        gate = np.zeros((len(layouts),), dtype=np.float64)
        for q_parent in sorted({layout.q_parent for layout in layouts}):
            indices = [
                index
                for index, layout in enumerate(layouts)
                if layout.q_parent == q_parent
            ]
            norm = math.sqrt(
                sum(
                    (
                        generic_weights[layouts[index].key]
                        * route_modulation[index]
                    )
                    ** 2
                    for index in indices
                )
            )
            if not norm > 0.0:
                raise RuntimeError("a generic reduced route has zero norm")
            gate[indices] = route_modulation[indices] / norm
        reduced_gates.append(jnp.asarray(gate))
        zero_index = next(
            (
                index
                for index, layout in enumerate(layouts)
                if layout.key == (0, 0, 0)
            ),
            None,
        )
        if zero_index is None:
            raise RuntimeError("a composite route lost its physical q=0 channel")
        physical_scalars[route.parent] = (
            physical_scalars[route.left]
            * physical_scalars[route.right]
            * float(gate[zero_index])
        )

    root_layout = triangle_root_layout(plan)
    root_arrays = [np.zeros(layout.shape, dtype=np.float64) for layout in root_layout]
    root_positions = {layout.q: index for index, layout in enumerate(root_layout)}
    if 0 not in root_positions:
        raise ValueError("the root does not retain a singlet channel")
    root_route = plan.root_route
    physical_denominator = (
        physical_scalars[root_route.left] * physical_scalars[root_route.right]
    )
    if abs(physical_denominator) <= 1.0e-14:
        raise RuntimeError("physical cluster branch became numerically singular")
    root_arrays[root_positions[0]][0, 0] = 1.0 / physical_denominator

    root_generic = random((len(root_layout),))
    root_generic /= np.linalg.norm(root_generic)
    for coefficient, block in zip(root_generic, root_arrays, strict=True):
        block[1, 1] = eta * coefficient

    return {
        "patch_pair": tuple(jnp.asarray(block) for block in patch_arrays),
        "reduced": tuple(jnp.asarray(block) for block in reduced_arrays),
        "root": tuple(jnp.asarray(block) for block in root_arrays),
        "patch_gates": patch_gates,
        "reduced_gates": tuple(reduced_gates),
    }


__all__ = [
    "TrianglePatchClusterReference",
    "init_triangle_patch_cluster_singlet",
    "sample_triangle_patch_cluster_product_chains",
    "triangle_patch_cluster_reference",
    "triangle_patch_cluster_six_site_probabilities",
]
