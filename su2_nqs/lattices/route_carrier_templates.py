"""Geometry-only carrier templates for triangular exact-SU(2) route banks.

The templates in this module never inspect a Hamiltonian expectation value or
a trained wavefunction.  They summarize how well each fusion-tree route keeps
the three bond orientations local.  Quadratic contractions of the uniform
``A1`` and nematic ``E2`` orientation components then provide three directions
per irrep channel for the ``J1-J1``, ``J1-J2``, and ``J2-J2`` shell pairs.

The ``E2`` directions are response carriers: the two E2 components are
contracted to a scalar, ``E2 x E2 -> A1``.  They therefore probe nematic
fluctuations without explicitly selecting a nematic orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np

from .geometry_tree import tree_leaf_distance
from .triangular_e2 import E2_ORIENTATION_FRAME, J2_E2_ORIENTATION_FRAME
from .triangular_hnf import (
    IntMatrix,
    IntVector,
    J1_POSITIVE,
    J2_POSITIVE,
    hnf_sites,
    reduce_hnf,
    validate_hnf,
)
from ..su2.fusion_tree import FusionTree


@dataclass(frozen=True)
class TriangularRouteCarrierTemplates:
    """Pre-registered physical and matched-generic route-channel templates."""

    route_names: tuple[str, ...]
    physical_labels: tuple[str, ...]
    a1: tuple[tuple[tuple[float, float], ...], ...]
    e2: tuple[tuple[tuple[float, float], ...], ...]
    generic: tuple[tuple[tuple[float, float], ...], ...]
    orientation_localities_j1: tuple[tuple[float, float, float], ...]
    orientation_localities_j2: tuple[tuple[float, float, float], ...]


def _oriented_bond_families(
    supercell: IntMatrix,
    displacements: Sequence[IntVector],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    matrix = validate_hnf(supercell)
    sites = hnf_sites(matrix)
    positions = {site: index for index, site in enumerate(sites)}
    rows = []
    for displacement in displacements:
        family = []
        for first, site in enumerate(sites):
            target = reduce_hnf(
                (site[0] + int(displacement[0]), site[1] + int(displacement[1])),
                matrix,
            )
            second = positions[target]
            if first == second:
                raise ValueError("a carrier bond aliases a site to itself")
            family.append((first, second))
        rows.append(tuple(family))
    return tuple(rows)


def _route_orientation_localities(
    tree: FusionTree,
    route_orders: Sequence[Sequence[int]],
    families: Sequence[Sequence[tuple[int, int]]],
) -> np.ndarray:
    rows = []
    n_sites = tree.n_leaves
    for order in route_orders:
        checked = tuple(int(site) for site in order)
        if tuple(sorted(checked)) != tuple(range(n_sites)):
            raise ValueError("every carrier route must be a complete site order")
        inverse = {site: leaf for leaf, site in enumerate(checked)}
        rows.append(
            [
                float(
                    np.mean(
                        [
                            1.0
                            / tree_leaf_distance(
                                tree, inverse[int(first)], inverse[int(second)]
                            )
                            for first, second in family
                        ]
                    )
                )
                for family in families
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _ordered_orthonormal_columns(
    values: np.ndarray,
    *,
    against: np.ndarray | None = None,
    tolerance: float = 1.0e-11,
) -> np.ndarray:
    """Center and Gram-Schmidt columns without data-dependent reordering."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < matrix.shape[1] + 1:
        raise ValueError("carrier descriptor matrix has insufficient route rank")
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    basis: list[np.ndarray] = []
    if against is not None:
        external = np.asarray(against, dtype=np.float64)
        if external.ndim != 2 or external.shape[0] != matrix.shape[0]:
            raise ValueError("against basis has incompatible shape")
        basis.extend(external[:, index] for index in range(external.shape[1]))
    answer = []
    for column in range(matrix.shape[1]):
        vector = matrix[:, column].copy()
        for direction in basis:
            vector -= np.dot(direction, vector) * direction
        norm = float(np.linalg.norm(vector))
        if norm <= tolerance:
            raise ValueError("pre-registered carrier descriptors lost rank")
        vector /= norm
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            vector = -vector
        basis.append(vector)
        answer.append(vector)
    return np.column_stack(answer)


def _quadratic_descriptors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.ndim == 1:
        return np.column_stack((first * first, first * second, second * second))
    if first.ndim == 2 and first.shape[1] == 2:
        return np.column_stack(
            (
                np.einsum("ri,ri->r", first, first),
                np.einsum("ri,ri->r", first, second),
                np.einsum("ri,ri->r", second, second),
            )
        )
    raise ValueError("quadratic carrier inputs have the wrong shape")


def _route_channel_templates(route_vectors: np.ndarray) -> np.ndarray:
    # The first channel drives coherent route resonance; the second changes
    # the first-pair log singlet/triplet reduced-weight ratio.  Equal norm in
    # the two channels prevents either mechanism from receiving hidden scale.
    return np.stack((route_vectors, route_vectors), axis=-1) / np.sqrt(2.0)


def _generic_complement(
    physical: np.ndarray,
    *,
    count: int,
    seed: int,
    tolerance: float = 1.0e-11,
) -> np.ndarray:
    flattened = np.asarray(physical, dtype=np.float64).reshape(
        physical.shape[0], -1
    ).T
    q_physical, _ = np.linalg.qr(flattened)
    rng = np.random.default_rng(int(seed))
    columns: list[np.ndarray] = []
    for _attempt in range(10_000):
        vector = rng.normal(size=flattened.shape[0])
        vector -= q_physical @ (q_physical.T @ vector)
        for direction in columns:
            vector -= np.dot(direction, vector) * direction
        norm = float(np.linalg.norm(vector))
        if norm <= tolerance:
            continue
        vector /= norm
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            vector = -vector
        columns.append(vector)
        if len(columns) == count:
            return np.stack(columns, axis=0).reshape(
                (count, physical.shape[1], physical.shape[2])
            )
    raise RuntimeError("could not construct the matched generic carrier bank")


def triangular_route_carrier_templates(
    tree: FusionTree,
    route_orders: Mapping[str, Sequence[int]],
    supercell: IntMatrix,
    *,
    generic_seed: int = 0,
) -> TriangularRouteCarrierTemplates:
    """Return three ``A1``, three E2-response, and six generic templates.

    The generic templates are seeded geometry-blind directions orthogonal to
    the entire six-dimensional physical span.  Prefixes of length three and
    six therefore provide parameter-count-matched controls without reusing a
    physical carrier direction accidentally.
    """

    if not isinstance(route_orders, Mapping) or not route_orders:
        raise ValueError("route_orders must be a nonempty ordered mapping")
    matrix = validate_hnf(supercell)
    if len(hnf_sites(matrix)) != tree.n_leaves:
        raise ValueError("supercell volume and fusion tree size disagree")
    names = tuple(str(name) for name in route_orders)
    orders = tuple(route_orders.values())
    j1 = _route_orientation_localities(
        tree, orders, _oriented_bond_families(matrix, J1_POSITIVE)
    )
    j2 = _route_orientation_localities(
        tree, orders, _oriented_bond_families(matrix, J2_POSITIVE)
    )
    uniform = np.ones((3,), dtype=np.float64) / np.sqrt(3.0)
    a1_j1 = j1 @ uniform
    a1_j2 = j2 @ uniform
    e2_j1 = j1 @ E2_ORIENTATION_FRAME.T
    e2_j2 = j2 @ J2_E2_ORIENTATION_FRAME.T
    a1_routes = _ordered_orthonormal_columns(
        _quadratic_descriptors(a1_j1, a1_j2)
    )
    e2_routes = _ordered_orthonormal_columns(
        _quadratic_descriptors(e2_j1, e2_j2), against=a1_routes
    )
    a1 = np.moveaxis(_route_channel_templates(a1_routes), 1, 0)
    e2 = np.moveaxis(_route_channel_templates(e2_routes), 1, 0)
    physical = np.concatenate((a1, e2), axis=0)
    generic = _generic_complement(
        physical, count=physical.shape[0], seed=generic_seed
    )
    gram = physical.reshape(physical.shape[0], -1) @ physical.reshape(
        physical.shape[0], -1
    ).T
    if not np.allclose(gram, np.eye(6), atol=2.0e-12, rtol=0.0):
        raise RuntimeError("physical carrier templates are not orthonormal")
    return TriangularRouteCarrierTemplates(
        route_names=names,
        physical_labels=(
            "A1:J1-J1",
            "A1:J1-J2",
            "A1:J2-J2",
            "E2xE2->A1:J1-J1",
            "E2xE2->A1:J1-J2",
            "E2xE2->A1:J2-J2",
        ),
        a1=tuple(tuple(tuple(map(float, row)) for row in item) for item in a1),
        e2=tuple(tuple(tuple(map(float, row)) for row in item) for item in e2),
        generic=tuple(
            tuple(tuple(map(float, row)) for row in item) for item in generic
        ),
        orientation_localities_j1=tuple(tuple(map(float, row)) for row in j1),
        orientation_localities_j2=tuple(tuple(map(float, row)) for row in j2),
    )


__all__ = [
    "TriangularRouteCarrierTemplates",
    "triangular_route_carrier_templates",
]
