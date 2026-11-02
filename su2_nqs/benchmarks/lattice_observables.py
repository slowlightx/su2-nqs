"""Exact small-system spin observables for square and triangular benchmarks.

The computational-basis routines are independent references for the reduced
SU(2) compilers.  Bits follow the project convention: zero is spin down, one
is spin up, and site zero is the most significant bit.  Spherical components
use ``T_0=S_z``, ``T_+1=-S_+/sqrt(2)``, and ``T_-1=S_-/sqrt(2)``; component
labels are doubled integers ``-2, 0, 2``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from ..operators.permutation import heisenberg_graph_fusion_matrix
from ..operators.sz_graph_sparse import heisenberg_sz_graph_sparse
from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.fusion_path import FusionPath, enumerate_paths
from ..su2.reduced_tensor_product import (
    ReducedTensorTable,
    compile_weighted_spin_vector_product,
)
from .coupled_basis import coupled_basis_matrix


SPHERICAL_COMPONENTS: tuple[int, int, int] = (-2, 0, 2)


@dataclass(frozen=True)
class SectorRank2Operator:
    """Reduced ``q_op=4`` matrix between fixed total-spin sectors."""

    n_sites: int
    q_new: int
    q_old: int
    new_paths: tuple[FusionPath, ...]
    old_paths: tuple[FusionPath, ...]
    matrix: np.ndarray
    table: ReducedTensorTable


def _validate_n_site(n_sites: int, site: int) -> tuple[int, int]:
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be an integer")
    if isinstance(site, bool) or not isinstance(site, Integral):
        raise TypeError("site must be an integer")
    n_sites, site = int(n_sites), int(site)
    if n_sites < 1 or not 0 <= site < n_sites:
        raise ValueError("site lies outside the spin system")
    return n_sites, site


def computational_spherical_spin_component(
    n_sites: int, site: int, component_q: int
) -> np.ndarray:
    """Return one spin-half spherical component in the full bit basis."""

    n_sites, site = _validate_n_site(n_sites, site)
    if component_q not in SPHERICAL_COMPONENTS:
        raise ValueError("component_q must be -2, 0, or 2")
    if component_q == 0:
        local = np.diag((-0.5, 0.5)).astype(np.complex128)
    elif component_q == 2:
        local = np.asarray([[0.0, 0.0], [-1.0 / math.sqrt(2.0), 0.0]], dtype=np.complex128)
    else:
        local = np.asarray([[0.0, 1.0 / math.sqrt(2.0)], [0.0, 0.0]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    result = np.asarray([[1.0]], dtype=np.complex128)
    for index in range(n_sites):
        result = np.kron(result, local if index == site else identity)
    return result


def weighted_spherical_spin_component(
    endpoint_weights: Sequence[complex], component_q: int
) -> np.ndarray:
    """Return ``sum_i c_i T_i(component_q)`` in the full bit basis."""

    weights = np.asarray(tuple(endpoint_weights), dtype=np.complex128)
    if weights.ndim != 1 or len(weights) < 1 or np.any(~np.isfinite(weights)):
        raise ValueError("endpoint_weights must be a finite nonempty sequence")
    result = np.zeros((2 ** len(weights), 2 ** len(weights)), dtype=np.complex128)
    for site, weight in enumerate(weights):
        result += weight * computational_spherical_spin_component(
            len(weights), site, component_q
        )
    return result


def computational_rank2_component(
    left_endpoint_weights: Sequence[complex],
    right_endpoint_weights: Sequence[complex],
    component_q: int,
) -> np.ndarray:
    """Return one component of ``[S(left) x S(right)]^(q_op=4)``."""

    left = tuple(left_endpoint_weights)
    right = tuple(right_endpoint_weights)
    if not left or len(left) != len(right):
        raise ValueError("endpoint-weight sequences must have equal positive length")
    if component_q not in range(-4, 5, 2):
        raise ValueError("component_q must be a projection of q_op=4")
    result = np.zeros((2 ** len(left), 2 ** len(left)), dtype=np.complex128)
    for mu_q in SPHERICAL_COMPONENTS:
        nu_q = component_q - mu_q
        if nu_q not in SPHERICAL_COMPONENTS:
            continue
        coefficient = clebsch_gordan(2, mu_q, 2, nu_q, 4, component_q)
        if coefficient:
            result += coefficient * (
                weighted_spherical_spin_component(left, mu_q)
                @ weighted_spherical_spin_component(right, nu_q)
            )
    return result


def reduced_tensor_sector_matrix(
    table: ReducedTensorTable, *, q_new: int, q_old: int
) -> tuple[tuple[FusionPath, ...], tuple[FusionPath, ...], np.ndarray]:
    """Extract one fixed-spin block of a reduced tensor table."""

    if not isinstance(q_new, Integral) or not isinstance(q_old, Integral):
        raise TypeError("q_new and q_old must be doubled integers")
    q_new, q_old = int(q_new), int(q_old)
    new_paths = tuple(path for path in table.paths if path[-1] == q_new)
    old_paths = tuple(path for path in table.paths if path[-1] == q_old)
    if not new_paths or not old_paths:
        raise ValueError("a requested spin sector is unreachable")
    new_position = {path: index for index, path in enumerate(new_paths)}
    old_position = {path: index for index, path in enumerate(old_paths)}
    matrix = np.zeros((len(new_paths), len(old_paths)), dtype=np.complex128)
    for (new, old), value in table.entry_dict().items():
        row = new_position.get(new)
        column = old_position.get(old)
        if row is not None and column is not None:
            matrix[row, column] += value
    return new_paths, old_paths, matrix


def compile_sector_rank2_composite(
    left_endpoint_weights: Sequence[complex],
    right_endpoint_weights: Sequence[complex],
    *,
    q_new: int = 4,
    q_old: int = 0,
) -> SectorRank2Operator:
    """Compile an ordered rank-two spin composite in multiplicity space."""

    table = compile_weighted_spin_vector_product(
        left_endpoint_weights, right_endpoint_weights, 4
    )
    new_paths, old_paths, matrix = reduced_tensor_sector_matrix(
        table, q_new=q_new, q_old=q_old
    )
    return SectorRank2Operator(
        n_sites=table.n_sites,
        q_new=int(q_new),
        q_old=int(q_old),
        new_paths=new_paths,
        old_paths=old_paths,
        matrix=matrix,
        table=table,
    )


def scalar_chirality_computational_matrix(
    n_sites: int, first: int, middle: int, last: int
) -> np.ndarray:
    """Return ``S_first . (S_middle x S_last)`` from a bond commutator."""

    sites = (first, middle, last)
    if len(set(sites)) != 3:
        raise ValueError("a chirality triangle requires three distinct sites")
    for site in sites:
        _validate_n_site(n_sites, site)

    def cartesian(site: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        plus = -math.sqrt(2.0) * computational_spherical_spin_component(
            n_sites, site, 2
        )
        minus = math.sqrt(2.0) * computational_spherical_spin_component(
            n_sites, site, -2
        )
        x = 0.5 * (plus + minus)
        y = (plus - minus) / (2.0j)
        z = computational_spherical_spin_component(n_sites, site, 0)
        return x, y, z

    spins = tuple(cartesian(site) for site in sites)
    first_spin, middle_spin, last_spin = spins
    return (
        first_spin[0] @ (middle_spin[1] @ last_spin[2] - middle_spin[2] @ last_spin[1])
        + first_spin[1] @ (middle_spin[2] @ last_spin[0] - middle_spin[0] @ last_spin[2])
        + first_spin[2] @ (middle_spin[0] @ last_spin[1] - middle_spin[1] @ last_spin[0])
    )


def scalar_chirality_fusion_matrix(
    n_sites: int,
    first: int,
    middle: int,
    last: int,
    *,
    target_q: int = 0,
):
    """Compile chirality as ``1j * [h_first,middle, h_middle,last]``."""

    sites = (first, middle, last)
    if len(set(sites)) != 3:
        raise ValueError("a chirality triangle requires three distinct sites")
    for site in sites:
        _validate_n_site(n_sites, site)
    first_bond = heisenberg_graph_fusion_matrix(
        n_sites, ((first, middle, 1.0),), target_q=target_q
    )
    second_bond = heisenberg_graph_fusion_matrix(
        n_sites, ((middle, last, 1.0),), target_q=target_q
    )
    result = 1.0j * (first_bond @ second_bond - second_bond @ first_bond)
    result.eliminate_zeros()
    return result.tocsr()


def oriented_triangular_plaquettes(
    rows: int, columns: int
) -> tuple[tuple[int, int, int], ...]:
    """Return every elementary OBC triangle in counterclockwise order.

    Primitive row and column vectors are ``(1,0)`` and
    ``(-1/2,sqrt(3)/2)``, respectively, matching the 120-degree metric used
    by :mod:`su2_nqs.lattices.triangular_j1j2`.
    """

    if rows < 2 or columns < 2:
        return ()

    def point(site: int) -> np.ndarray:
        row, column = divmod(site, columns)
        return np.asarray(
            [row - 0.5 * column, math.sqrt(3.0) * column / 2.0],
            dtype=np.float64,
        )

    def ccw(triangle: tuple[int, int, int]) -> tuple[int, int, int]:
        a, b, c = triangle
        first_edge = point(b) - point(a)
        second_edge = point(c) - point(a)
        cross = float(
            first_edge[0] * second_edge[1]
            - first_edge[1] * second_edge[0]
        )
        if abs(cross) < 1.0e-14:
            raise RuntimeError("a triangular plaquette became collinear")
        return triangle if cross > 0.0 else (a, c, b)

    answer = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = row * columns + column
            b = (row + 1) * columns + column
            c = row * columns + column + 1
            d = (row + 1) * columns + column + 1
            answer.extend((ccw((a, b, d)), ccw((a, d, c))))
    return tuple(answer)


def uniform_triangular_chirality_fusion_matrix(
    rows: int,
    columns: int,
    physical_to_leaf: Mapping[int, int] | None = None,
    *,
    target_q: int = 0,
):
    """Return the sum of consistently oriented elementary chiralities."""

    try:
        from scipy import sparse
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("triangular chirality requires scipy") from error
    n_sites = rows * columns
    if physical_to_leaf is None:
        mapping = {site: site for site in range(n_sites)}
    else:
        mapping = {int(key): int(value) for key, value in physical_to_leaf.items()}
        if set(mapping) != set(range(n_sites)) or set(mapping.values()) != set(
            range(n_sites)
        ):
            raise ValueError("physical_to_leaf must be a site permutation")
    dimension = len(enumerate_paths(n_sites, target_q))
    result = sparse.csr_matrix((dimension, dimension), dtype=np.complex128)
    for first, middle, last in oriented_triangular_plaquettes(rows, columns):
        result = result + scalar_chirality_fusion_matrix(
            n_sites,
            mapping[first],
            mapping[middle],
            mapping[last],
            target_q=target_q,
        )
    result.eliminate_zeros()
    return result.tocsr()


def spin_structure_factor_operator(
    coordinates: Sequence[tuple[float, float]],
    wavevector: tuple[float, float],
    physical_to_leaf: Mapping[int, int],
    *,
    target_q: int = 0,
    normalize_by_n_squared: bool = True,
):
    """Return ``sum_ij exp(i q.(r_i-r_j)) S_i.S_j`` in one spin sector."""

    try:
        from scipy import sparse
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("structure factors require scipy") from error
    points = tuple((float(x), float(y)) for x, y in coordinates)
    n_sites = len(points)
    mapping = {int(key): int(value) for key, value in physical_to_leaf.items()}
    if set(mapping) != set(range(n_sites)) or set(mapping.values()) != set(
        range(n_sites)
    ):
        raise ValueError("physical_to_leaf must be a site permutation")
    qx, qy = map(float, wavevector)
    if not all(math.isfinite(value) for value in (qx, qy)):
        raise ValueError("wavevector must be finite")
    scale = float(n_sites**2 if normalize_by_n_squared else n_sites)
    bonds = []
    for first in range(n_sites):
        for second in range(first + 1, n_sites):
            dx = points[first][0] - points[second][0]
            dy = points[first][1] - points[second][1]
            coefficient = 2.0 * math.cos(qx * dx + qy * dy) / scale
            if abs(coefficient) > 1.0e-15:
                bonds.append((mapping[first], mapping[second], coefficient))
    pair = heisenberg_graph_fusion_matrix(
        n_sites, bonds, target_q=target_q
    )
    onsite = sparse.identity(pair.shape[0], format="csr", dtype=np.float64)
    onsite *= 0.75 * n_sites / scale
    return (pair + onsite).tocsr()


def spin_structure_factor_sz_operator(
    coordinates: Sequence[tuple[float, float]],
    wavevector: tuple[float, float],
    physical_to_leaf: Mapping[int, int],
    *,
    n_up: int | None = None,
    normalize_by_n_squared: bool = True,
):
    """Return a structure factor in deterministic fixed-``Sz`` basis order."""

    try:
        from scipy import sparse
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("structure factors require scipy") from error
    points = tuple((float(x), float(y)) for x, y in coordinates)
    n_sites = len(points)
    mapping = _validated_physical_mapping(n_sites, physical_to_leaf)
    qx, qy = map(float, wavevector)
    if not all(math.isfinite(value) for value in (qx, qy)):
        raise ValueError("wavevector must be finite")
    scale = float(n_sites**2 if normalize_by_n_squared else n_sites)
    bonds = []
    for first in range(n_sites):
        for second in range(first + 1, n_sites):
            dx = points[first][0] - points[second][0]
            dy = points[first][1] - points[second][1]
            coefficient = 2.0 * math.cos(qx * dx + qy * dy) / scale
            if abs(coefficient) > 1.0e-15:
                bonds.append(
                    (mapping[first], mapping[second], coefficient)
                )
    pair, basis = heisenberg_sz_graph_sparse(n_sites, bonds, n_up=n_up)
    onsite = sparse.identity(len(basis), format="csr", dtype=np.float64)
    onsite *= 0.75 * n_sites / scale
    return (pair + onsite).tocsr(), basis


def square_coordinates(rows: int, columns: int) -> tuple[tuple[float, float], ...]:
    """Return row-major Cartesian square coordinates ``(column,row)``."""

    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    return tuple((float(column), float(row)) for row in range(rows) for column in range(columns))


def triangular_phase_coordinates(
    rows: int, columns: int
) -> tuple[tuple[float, float], ...]:
    """Return reciprocal-phase coordinates ``(row,column)``.

    Wavevectors are then specified as ``2*pi*(h,k)``.  In particular the
    registered momenta are ``K=(1/3,1/3)`` and the three M points
    ``(1/2,0)``, ``(0,1/2)``, and ``(1/2,1/2)``.
    """

    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    return tuple((float(row), float(column)) for row in range(rows) for column in range(columns))


def triangular_nearest_neighbor_orientation_bonds(
    rows: int, columns: int
) -> tuple[tuple[tuple[int, int, float], ...], ...]:
    """Return the three disjoint OBC nearest-neighbour orientation families."""

    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    families: list[tuple[tuple[int, int, float], ...]] = []
    for dr, dc in ((1, 0), (0, 1), (1, 1)):
        bonds = []
        for row in range(rows):
            for column in range(columns):
                shifted_row, shifted_column = row + dr, column + dc
                if shifted_row < rows and shifted_column < columns:
                    bonds.append(
                        (
                            row * columns + column,
                            shifted_row * columns + shifted_column,
                            1.0,
                        )
                    )
        families.append(tuple(bonds))
    return tuple(families)


def _validated_physical_mapping(
    n_sites: int, physical_to_leaf: Mapping[int, int] | None
) -> dict[int, int]:
    if physical_to_leaf is None:
        return {site: site for site in range(n_sites)}
    mapping = {int(key): int(value) for key, value in physical_to_leaf.items()}
    if set(mapping) != set(range(n_sites)) or set(mapping.values()) != set(
        range(n_sites)
    ):
        raise ValueError("physical_to_leaf must be a site permutation")
    return mapping


def weighted_bond_fusion_matrix(
    n_sites: int,
    physical_bonds: Sequence[tuple[int, int, float]],
    physical_to_leaf: Mapping[int, int] | None = None,
    *,
    target_q: int = 0,
):
    """Compile a physically labelled weighted bond sum in one fusion sector."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be an integer")
    n_sites = int(n_sites)
    if n_sites < 2:
        raise ValueError("n_sites must be at least two")
    mapping = _validated_physical_mapping(n_sites, physical_to_leaf)
    bonds = []
    for first, second, coefficient in physical_bonds:
        first = _validate_n_site(n_sites, first)[1]
        second = _validate_n_site(n_sites, second)[1]
        if first == second:
            raise ValueError("a physical bond must join distinct sites")
        value = float(coefficient)
        if not math.isfinite(value):
            raise ValueError("bond coefficients must be finite")
        bonds.append((mapping[first], mapping[second], value))
    return heisenberg_graph_fusion_matrix(
        n_sites, tuple(bonds), target_q=target_q
    ).tocsr()


def weighted_bond_sz_matrix(
    n_sites: int,
    physical_bonds: Sequence[tuple[int, int, float]],
    physical_to_leaf: Mapping[int, int] | None = None,
    *,
    n_up: int | None = None,
):
    """Compile a physically labelled weighted bond sum at fixed ``Sz``."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be an integer")
    n_sites = int(n_sites)
    if n_sites < 2:
        raise ValueError("n_sites must be at least two")
    mapping = _validated_physical_mapping(n_sites, physical_to_leaf)
    bonds = []
    for first, second, coefficient in physical_bonds:
        first = _validate_n_site(n_sites, first)[1]
        second = _validate_n_site(n_sites, second)[1]
        if first == second:
            raise ValueError("a physical bond must join distinct sites")
        value = float(coefficient)
        if not math.isfinite(value):
            raise ValueError("bond coefficients must be finite")
        bonds.append((mapping[first], mapping[second], value))
    return heisenberg_sz_graph_sparse(n_sites, tuple(bonds), n_up=n_up)


def scalar_chirality_sz_matrix(
    n_sites: int,
    first: int,
    middle: int,
    last: int,
    *,
    n_up: int | None = None,
):
    """Compile one oriented chirality from fixed-``Sz`` bond commutators."""

    sites = (first, middle, last)
    if len(set(sites)) != 3:
        raise ValueError("a chirality triangle requires three distinct sites")
    for site in sites:
        _validate_n_site(n_sites, site)
    first_bond, basis = heisenberg_sz_graph_sparse(
        n_sites, ((first, middle, 1.0),), n_up=n_up
    )
    second_bond, second_basis = heisenberg_sz_graph_sparse(
        n_sites, ((middle, last, 1.0),), n_up=n_up
    )
    if basis != second_basis:
        raise RuntimeError("fixed-Sz basis order changed between bond operators")
    result = 1.0j * (
        first_bond @ second_bond - second_bond @ first_bond
    )
    result.eliminate_zeros()
    return result.tocsr(), basis


def uniform_triangular_chirality_sz_matrix(
    rows: int,
    columns: int,
    physical_to_leaf: Mapping[int, int] | None = None,
    *,
    n_up: int | None = None,
):
    """Return uniform oriented chirality in a fixed-``Sz`` sector."""

    try:
        from scipy import sparse
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("triangular chirality requires scipy") from error
    n_sites = rows * columns
    mapping = _validated_physical_mapping(n_sites, physical_to_leaf)
    basis = None
    result = None
    for first, middle, last in oriented_triangular_plaquettes(rows, columns):
        term, term_basis = scalar_chirality_sz_matrix(
            n_sites,
            mapping[first],
            mapping[middle],
            mapping[last],
            n_up=n_up,
        )
        if basis is None:
            basis = term_basis
            result = sparse.csr_matrix(term.shape, dtype=np.complex128)
        elif basis != term_basis:
            raise RuntimeError("fixed-Sz basis order changed between chiralities")
        result = result + term
    if basis is None or result is None:
        _, basis = heisenberg_sz_graph_sparse(n_sites, (), n_up=n_up)
        result = sparse.csr_matrix((len(basis), len(basis)), dtype=np.complex128)
    result.eliminate_zeros()
    return result.tocsr(), basis


def _square_bond_families(
    rows: int, columns: int
) -> dict[str, tuple[tuple[int, int, float], ...]]:
    """Return the physically labelled square bond-channel definitions."""

    horizontal: list[tuple[int, int, float]] = []
    vertical: list[tuple[int, int, float]] = []
    columnar_x: list[tuple[int, int, float]] = []
    columnar_y: list[tuple[int, int, float]] = []
    for row in range(rows):
        for column in range(columns - 1):
            first = row * columns + column
            horizontal.append((first, first + 1, 1.0))
            columnar_x.append((first, first + 1, (-1.0) ** column))
    for row in range(rows - 1):
        for column in range(columns):
            first = row * columns + column
            vertical.append((first, first + columns, 1.0))
            columnar_y.append((first, first + columns, (-1.0) ** row))
    return {
        "b1g": tuple(
            horizontal
            + [(first, second, -value) for first, second, value in vertical]
        ),
        "columnar_x": tuple(columnar_x),
        "columnar_y": tuple(columnar_y),
    }


def square_obc_channel_operators(
    rows: int,
    columns: int,
    physical_to_leaf: Mapping[int, int] | None = None,
    *,
    target_q: int = 0,
) -> dict[str, object]:
    """Return square magnetic, B1g, and columnar-dimer channel operators."""

    n_sites = rows * columns
    mapping = _validated_physical_mapping(n_sites, physical_to_leaf)
    coordinates = square_coordinates(rows, columns)
    bonds = _square_bond_families(rows, columns)
    return {
        "neel": spin_structure_factor_operator(
            coordinates, (math.pi, math.pi), mapping, target_q=target_q
        ),
        "stripe_x": spin_structure_factor_operator(
            coordinates, (math.pi, 0.0), mapping, target_q=target_q
        ),
        "stripe_y": spin_structure_factor_operator(
            coordinates, (0.0, math.pi), mapping, target_q=target_q
        ),
        "b1g": weighted_bond_fusion_matrix(
            n_sites, bonds["b1g"], mapping, target_q=target_q
        ),
        "columnar_x": weighted_bond_fusion_matrix(
            n_sites, bonds["columnar_x"], mapping, target_q=target_q
        ),
        "columnar_y": weighted_bond_fusion_matrix(
            n_sites, bonds["columnar_y"], mapping, target_q=target_q
        ),
    }


def square_obc_channel_operators_sz(
    rows: int,
    columns: int,
    physical_to_leaf: Mapping[int, int] | None = None,
    *,
    n_up: int | None = None,
) -> tuple[dict[str, object], tuple[tuple[int, ...], ...]]:
    """Return the square channel panel in a fixed-``Sz`` sector."""

    n_sites = rows * columns
    mapping = _validated_physical_mapping(n_sites, physical_to_leaf)
    coordinates = square_coordinates(rows, columns)
    bonds = _square_bond_families(rows, columns)
    neel, basis = spin_structure_factor_sz_operator(
        coordinates, (math.pi, math.pi), mapping, n_up=n_up
    )
    stripe_x, stripe_x_basis = spin_structure_factor_sz_operator(
        coordinates, (math.pi, 0.0), mapping, n_up=n_up
    )
    stripe_y, stripe_y_basis = spin_structure_factor_sz_operator(
        coordinates, (0.0, math.pi), mapping, n_up=n_up
    )
    b1g, b1g_basis = weighted_bond_sz_matrix(
        n_sites, bonds["b1g"], mapping, n_up=n_up
    )
    columnar_x, columnar_x_basis = weighted_bond_sz_matrix(
        n_sites, bonds["columnar_x"], mapping, n_up=n_up
    )
    columnar_y, columnar_y_basis = weighted_bond_sz_matrix(
        n_sites, bonds["columnar_y"], mapping, n_up=n_up
    )
    if any(
        candidate != basis
        for candidate in (
            stripe_x_basis,
            stripe_y_basis,
            b1g_basis,
            columnar_x_basis,
            columnar_y_basis,
        )
    ):
        raise RuntimeError("fixed-Sz basis order changed between square channels")
    return {
        "neel": neel,
        "stripe_x": stripe_x,
        "stripe_y": stripe_y,
        "b1g": b1g,
        "columnar_x": columnar_x,
        "columnar_y": columnar_y,
    }, basis


def triangular_obc_channel_operators(
    rows: int,
    columns: int,
    physical_to_leaf: Mapping[int, int] | None = None,
    *,
    target_q: int = 0,
) -> dict[str, object]:
    """Return triangular K/M, bond-nematic, and chirality channel operators."""

    n_sites = rows * columns
    mapping = _validated_physical_mapping(n_sites, physical_to_leaf)
    coordinates = triangular_phase_coordinates(rows, columns)
    two_pi = 2.0 * math.pi
    families = triangular_nearest_neighbor_orientation_bonds(rows, columns)
    orientation = tuple(
        weighted_bond_fusion_matrix(
            n_sites, family, mapping, target_q=target_q
        )
        for family in families
    )
    first, second, third = orientation
    return {
        "k": spin_structure_factor_operator(
            coordinates,
            (two_pi / 3.0, two_pi / 3.0),
            mapping,
            target_q=target_q,
        ),
        "m1": spin_structure_factor_operator(
            coordinates, (math.pi, 0.0), mapping, target_q=target_q
        ),
        "m2": spin_structure_factor_operator(
            coordinates, (0.0, math.pi), mapping, target_q=target_q
        ),
        "m3": spin_structure_factor_operator(
            coordinates, (math.pi, math.pi), mapping, target_q=target_q
        ),
        "nematic_1": ((2.0 * first - second - third) / math.sqrt(6.0)).tocsr(),
        "nematic_2": ((second - third) / math.sqrt(2.0)).tocsr(),
        "chirality": uniform_triangular_chirality_fusion_matrix(
            rows, columns, mapping, target_q=target_q
        ),
    }


def triangular_obc_channel_operators_sz(
    rows: int,
    columns: int,
    physical_to_leaf: Mapping[int, int] | None = None,
    *,
    n_up: int | None = None,
) -> tuple[dict[str, object], tuple[tuple[int, ...], ...]]:
    """Return the triangular channel panel in a fixed-``Sz`` sector."""

    n_sites = rows * columns
    mapping = _validated_physical_mapping(n_sites, physical_to_leaf)
    coordinates = triangular_phase_coordinates(rows, columns)
    two_pi = 2.0 * math.pi
    families = triangular_nearest_neighbor_orientation_bonds(rows, columns)
    orientations = [
        weighted_bond_sz_matrix(
            n_sites, family, mapping, n_up=n_up
        )
        for family in families
    ]
    first, second, third = (item[0] for item in orientations)
    basis = orientations[0][1]
    k, k_basis = spin_structure_factor_sz_operator(
        coordinates,
        (two_pi / 3.0, two_pi / 3.0),
        mapping,
        n_up=n_up,
    )
    m1, m1_basis = spin_structure_factor_sz_operator(
        coordinates, (math.pi, 0.0), mapping, n_up=n_up
    )
    m2, m2_basis = spin_structure_factor_sz_operator(
        coordinates, (0.0, math.pi), mapping, n_up=n_up
    )
    m3, m3_basis = spin_structure_factor_sz_operator(
        coordinates, (math.pi, math.pi), mapping, n_up=n_up
    )
    chirality, chirality_basis = uniform_triangular_chirality_sz_matrix(
        rows, columns, mapping, n_up=n_up
    )
    if any(
        candidate != basis
        for candidate in (
            *(item[1] for item in orientations),
            k_basis,
            m1_basis,
            m2_basis,
            m3_basis,
            chirality_basis,
        )
    ):
        raise RuntimeError("fixed-Sz basis order changed between triangular channels")
    return {
        "k": k,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "nematic_1": ((2.0 * first - second - third) / math.sqrt(6.0)).tocsr(),
        "nematic_2": ((second - third) / math.sqrt(2.0)).tocsr(),
        "chirality": chirality,
    }, basis


__all__ = [
    "SPHERICAL_COMPONENTS",
    "SectorRank2Operator",
    "compile_sector_rank2_composite",
    "computational_rank2_component",
    "computational_spherical_spin_component",
    "oriented_triangular_plaquettes",
    "reduced_tensor_sector_matrix",
    "scalar_chirality_computational_matrix",
    "scalar_chirality_fusion_matrix",
    "scalar_chirality_sz_matrix",
    "spin_structure_factor_operator",
    "spin_structure_factor_sz_operator",
    "square_obc_channel_operators",
    "square_obc_channel_operators_sz",
    "square_coordinates",
    "triangular_nearest_neighbor_orientation_bonds",
    "triangular_obc_channel_operators",
    "triangular_obc_channel_operators_sz",
    "triangular_phase_coordinates",
    "uniform_triangular_chirality_fusion_matrix",
    "uniform_triangular_chirality_sz_matrix",
    "weighted_spherical_spin_component",
    "weighted_bond_fusion_matrix",
    "weighted_bond_sz_matrix",
]
