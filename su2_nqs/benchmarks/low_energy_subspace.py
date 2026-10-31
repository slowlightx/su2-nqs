"""Basis-invariant diagnostics inside a closed low-energy subspace.

Near-degenerate eigenvectors may rotate arbitrarily under tiny numerical or
boundary perturbations.  Physical comparisons should therefore use the
projector onto the complete cluster and invariants of operators compressed to
that subspace, rather than attach meaning to one raw eigenvector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from ..operators.permutation import apply_site_permutation_fusion
from ..su2.fusion_path import count_paths
from .block_spin_fcs import block_spin_fcs


def _vector_batch(
    states: np.ndarray, n_sites: int, target_q: int
) -> tuple[np.ndarray, bool]:
    """Validate fusion vectors and return a two-dimensional batch."""

    array = np.asarray(states, dtype=np.complex128)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("states must be a vector or a matrix of column vectors")
    if array.shape[0] != count_paths(n_sites, int(target_q)):
        raise ValueError("state dimension does not match n_sites and target_q")
    if np.any(~np.isfinite(array)):
        raise ValueError("states must be finite")
    return array, was_vector


def _physical_to_leaf_mapping(
    n_sites: int, physical_to_leaf: Mapping[int, int] | None
) -> dict[int, int]:
    mapping = (
        {site: site for site in range(n_sites)}
        if physical_to_leaf is None
        else {int(key): int(value) for key, value in physical_to_leaf.items()}
    )
    if set(mapping) != set(range(n_sites)) or set(mapping.values()) != set(
        range(n_sites)
    ):
        raise ValueError("physical_to_leaf must be a site permutation")
    return mapping


def matrix_free_weighted_bond_actions(
    states: np.ndarray,
    n_sites: int,
    operator_bonds: Mapping[str, Sequence[tuple[int, int, float]]],
    physical_to_leaf: Mapping[int, int] | None = None,
    *,
    target_q: int = 0,
    constants: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Apply several scalar pair operators to a low-rank fusion batch.

    Every row in ``operator_bonds[name]`` means
    ``weight * S_first.S_second``.  A physical pair shared by several named
    operators is applied only once.  Optional constants multiply the
    identity.  No full operator is materialized.
    """

    if isinstance(n_sites, bool) or not isinstance(n_sites, int) or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    batch, was_vector = _vector_batch(states, n_sites, int(target_q))
    mapping = _physical_to_leaf_mapping(n_sites, physical_to_leaf)
    names = tuple(str(name) for name in operator_bonds)
    if not names or len(set(names)) != len(names):
        raise ValueError("operator_bonds must have distinct nonempty names")
    constant_values = {name: 0.0 for name in names}
    if constants is not None:
        if set(map(str, constants)) - set(names):
            raise ValueError("constant supplied for an unknown operator")
        for raw_name, raw_value in constants.items():
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("operator constants must be finite")
            constant_values[str(raw_name)] = value

    by_pair: dict[tuple[int, int], dict[str, float]] = {}
    for raw_name, rows in operator_bonds.items():
        name = str(raw_name)
        combined: dict[tuple[int, int], float] = {}
        for first, second, weight in rows:
            first, second, weight = int(first), int(second), float(weight)
            if first == second or not 0 <= first < n_sites or not 0 <= second < n_sites:
                raise ValueError("bond endpoints must be distinct physical sites")
            if not math.isfinite(weight):
                raise ValueError("bond weights must be finite")
            edge = tuple(sorted((first, second)))
            combined[edge] = combined.get(edge, 0.0) + weight
        for edge, weight in combined.items():
            if abs(weight) > 1.0e-15:
                by_pair.setdefault(edge, {})[name] = weight

    answer = {
        name: constant_values[name] * batch.copy()
        for name in names
    }
    for (first, second), weights in sorted(by_pair.items()):
        image = list(range(n_sites))
        first_leaf, second_leaf = mapping[first], mapping[second]
        image[first_leaf], image[second_leaf] = second_leaf, first_leaf
        permuted = apply_site_permutation_fusion(
            batch, image, target_q=int(target_q)
        )
        pair_action = 0.5 * permuted - 0.25 * batch
        for name, weight in weights.items():
            answer[name] += weight * pair_action
    return {
        name: value[:, 0] if was_vector else value
        for name, value in answer.items()
    }


def matrix_free_chirality_action(
    states: np.ndarray,
    n_sites: int,
    oriented_triangles: Sequence[tuple[int, int, int]],
    physical_to_leaf: Mapping[int, int] | None = None,
    *,
    target_q: int = 0,
    scale: float = 1.0,
) -> np.ndarray:
    """Apply a sum of consistently oriented scalar chiralities.

    The exact identity
    ``chi_ijk = 0.25j * (P_ij P_jk - P_jk P_ij)`` is evaluated on the input
    batch using adjacent F-move swaps.  ``scale`` multiplies the full sum.
    """

    if not math.isfinite(float(scale)):
        raise ValueError("scale must be finite")
    batch, was_vector = _vector_batch(states, n_sites, int(target_q))
    mapping = _physical_to_leaf_mapping(n_sites, physical_to_leaf)

    def permute(value: np.ndarray, first: int, second: int) -> np.ndarray:
        image = list(range(n_sites))
        first_leaf, second_leaf = mapping[first], mapping[second]
        image[first_leaf], image[second_leaf] = second_leaf, first_leaf
        return apply_site_permutation_fusion(
            value, image, target_q=int(target_q)
        )

    answer = np.zeros_like(batch, dtype=np.complex128)
    for raw_triangle in oriented_triangles:
        first, middle, last = (int(site) for site in raw_triangle)
        if len({first, middle, last}) != 3 or any(
            not 0 <= site < n_sites for site in (first, middle, last)
        ):
            raise ValueError("every chirality triangle needs three distinct sites")
        forward = permute(permute(batch, middle, last), first, middle)
        backward = permute(permute(batch, first, middle), middle, last)
        answer += 0.25j * (forward - backward)
    answer *= float(scale)
    return answer[:, 0] if was_vector else answer


def matrix_free_structure_factor_actions(
    states: np.ndarray,
    coordinates: Sequence[tuple[float, float]],
    wavevectors: Mapping[str, tuple[float, float]],
    physical_to_leaf: Mapping[int, int] | None = None,
    *,
    target_q: int = 0,
    normalize_by_n_squared: bool = True,
) -> dict[str, np.ndarray]:
    """Apply a panel of scalar spin structure factors without full matrices."""

    points = tuple((float(x), float(y)) for x, y in coordinates)
    if not points or any(not all(math.isfinite(value) for value in point) for point in points):
        raise ValueError("coordinates must be finite and nonempty")
    momenta = {
        str(name): (float(wavevector[0]), float(wavevector[1]))
        for name, wavevector in wavevectors.items()
    }
    if not momenta or any(
        not all(math.isfinite(component) for component in wavevector)
        for wavevector in momenta.values()
    ):
        raise ValueError("wavevectors must be finite and nonempty")
    n_sites = len(points)
    scale = float(n_sites**2 if normalize_by_n_squared else n_sites)
    operator_bonds: dict[str, list[tuple[int, int, float]]] = {
        name: [] for name in momenta
    }
    for first in range(n_sites):
        for second in range(first + 1, n_sites):
            dx = points[first][0] - points[second][0]
            dy = points[first][1] - points[second][1]
            for name, (qx, qy) in momenta.items():
                coefficient = 2.0 * math.cos(qx * dx + qy * dy) / scale
                if abs(coefficient) > 1.0e-15:
                    operator_bonds[name].append((first, second, coefficient))
    return matrix_free_weighted_bond_actions(
        states,
        n_sites,
        operator_bonds,
        physical_to_leaf,
        target_q=int(target_q),
        constants={name: 0.75 * n_sites / scale for name in momenta},
    )


def orthonormal_subspace(states: np.ndarray, *, tolerance: float = 1.0e-10) -> np.ndarray:
    """Validate and return column-orthonormal subspace coordinates."""

    vectors = np.asarray(states, dtype=np.complex128)
    if vectors.ndim != 2 or vectors.shape[1] < 1:
        raise ValueError("states must be a nonempty matrix of column vectors")
    error = float(
        np.max(
            np.abs(vectors.conj().T @ vectors - np.eye(vectors.shape[1])),
            initial=0.0,
        )
    )
    if error > tolerance:
        raise ValueError("states are not column orthonormal")
    return vectors


def projected_operator(operator: object, states: np.ndarray) -> np.ndarray:
    """Compress one full-space operator into an orthonormal subspace."""

    vectors = orthonormal_subspace(states)
    if getattr(operator, "shape", None) != (vectors.shape[0], vectors.shape[0]):
        raise ValueError("operator shape does not match the subspace")
    result = vectors.conj().T @ (operator @ vectors)
    return np.asarray(result, dtype=np.complex128)


def matrix_free_structure_factor_subspace(
    states: np.ndarray,
    coordinates: Sequence[tuple[float, float]],
    wavevectors: Mapping[str, tuple[float, float]],
    physical_to_leaf: Mapping[int, int] | None = None,
    *,
    target_q: int = 0,
    normalize_by_n_squared: bool = True,
) -> dict[str, np.ndarray]:
    """Compress a structure-factor panel without materializing its operators.

    For orthonormal columns ``V``, this returns ``V.conj().T @ S(k) @ V`` for
    every requested wavevector.  Each physical transposition is applied once
    to the whole low-rank batch and reused for all momenta through

    ``S_i.S_j = P_ij/2 - I/4``.

    The memory associated with the observable is therefore linear in the
    fusion dimension times the subspace rank, rather than quadratic in the
    fusion dimension.  Coordinates and wavevectors use the same direct dot
    product convention as :func:`spin_structure_factor_operator`.
    """

    vectors = orthonormal_subspace(states)
    actions = matrix_free_structure_factor_actions(
        vectors,
        coordinates,
        wavevectors,
        physical_to_leaf,
        target_q=int(target_q),
        normalize_by_n_squared=normalize_by_n_squared,
    )
    return {
        name: vectors.conj().T @ action
        for name, action in actions.items()
    }


def traceless_part(matrix: np.ndarray) -> np.ndarray:
    """Remove the scalar part of a square matrix."""

    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 1:
        raise ValueError("matrix must be nonempty and square")
    return value - np.trace(value) * np.eye(value.shape[0]) / value.shape[0]


def traceless_frobenius_norm_squared(matrix: np.ndarray) -> float:
    """Return the squared Frobenius norm after removing the scalar part."""

    centered = traceless_part(matrix)
    return float(np.vdot(centered, centered).real)


def low_pair_isolation_diagnostics(
    eigenvalues: Sequence[float],
    *,
    degeneracy_floor: float = 1.0e-12,
) -> dict[str, float]:
    """Diagnose whether the two lowest levels form an isolated pair.

    The dimensionless isolation ratio is ``(E2-E1)/(E1-E0)``.  Its
    denominator is floored only to make an exactly degenerate ground-state
    pair finite and JSON-safe; the raw gaps are always returned so that an
    exact symmetry degeneracy cannot be confused with an exponentially split
    topological pair.
    """

    values = np.asarray(tuple(eigenvalues), dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("at least three eigenvalues are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues must be finite")
    if degeneracy_floor <= 0.0 or not np.isfinite(degeneracy_floor):
        raise ValueError("degeneracy_floor must be finite and positive")
    ordering_error = float(np.max(np.maximum(values[:-1] - values[1:], 0.0)))
    if ordering_error > degeneracy_floor:
        raise ValueError("eigenvalues must be ordered increasingly")
    ground_pair_splitting = max(float(values[1] - values[0]), 0.0)
    separation_from_third = max(float(values[2] - values[1]), 0.0)
    return {
        "ground_pair_splitting": ground_pair_splitting,
        "separation_from_third": separation_from_third,
        "isolation_ratio": separation_from_third
        / max(ground_pair_splitting, float(degeneracy_floor)),
        "ordering_error": ordering_error,
    }


def hermitian_bloch_vector(matrix: np.ndarray) -> np.ndarray:
    """Map a two-state Hermitian operator's traceless part to ``(x,y,z)``."""

    value = np.asarray(matrix, dtype=np.complex128)
    if value.shape != (2, 2):
        raise ValueError("Bloch coordinates require a two-state subspace")
    hermiticity = float(np.max(np.abs(value - value.conj().T), initial=0.0))
    if hermiticity > 1.0e-10:
        raise ValueError("matrix must be Hermitian")
    centered = traceless_part(value)
    return np.asarray(
        [centered[0, 1].real, -centered[0, 1].imag, centered[0, 0].real],
        dtype=np.float64,
    )


def two_state_projector_family_analysis(
    projected_projectors: Mapping[object, np.ndarray],
    comparator: np.ndarray,
) -> dict[str, object]:
    """Resolve a complete projector family's contrast and leading axis.

    The total contrast is

    ``sum_a ||P_low P_a P_low - Tr(P_low P_a)/2 I||_F^2``.

    The leading Bloch axis is obtained from the positive frame operator
    ``sum_a v_a v_a^T``.  Its absolute cosine with a compressed physical
    comparator is invariant under rotations of the two low-energy states.
    """

    if not projected_projectors:
        raise ValueError("projected_projectors must be nonempty")
    vectors = []
    traces = []
    maximum_hermiticity = 0.0
    for matrix in projected_projectors.values():
        value = np.asarray(matrix, dtype=np.complex128)
        if value.shape != (2, 2):
            raise ValueError("every compressed projector must be two by two")
        maximum_hermiticity = max(
            maximum_hermiticity,
            float(np.max(np.abs(value - value.conj().T), initial=0.0)),
        )
        vectors.append(hermitian_bloch_vector(value))
        traces.append(float(np.trace(value).real))
    frame = sum((np.outer(vector, vector) for vector in vectors), start=np.zeros((3, 3)))
    frame = 0.5 * (frame + frame.T)
    values, axes = np.linalg.eigh(frame)
    leading = axes[:, -1]
    comparator_vector = hermitian_bloch_vector(comparator)
    comparator_norm = float(np.linalg.norm(comparator_vector))
    cosine = (
        float(abs(np.dot(leading, comparator_vector)) / comparator_norm)
        if comparator_norm > 1.0e-14
        else None
    )
    contrast = float(2.0 * np.trace(frame))
    resolution = sum(
        (np.asarray(matrix, dtype=np.complex128) for matrix in projected_projectors.values()),
        start=np.zeros((2, 2), dtype=np.complex128),
    )
    return {
        "outcome_count": len(projected_projectors),
        "contrast": contrast,
        "frame_eigenvalues": [float(value) for value in values],
        "leading_axis": [float(value) for value in leading],
        "comparator_traceless_norm_squared": traceless_frobenius_norm_squared(comparator),
        "leading_axis_comparator_absolute_cosine": cosine,
        "projector_trace_sum": float(sum(traces)),
        "maximum_hermiticity_error": maximum_hermiticity,
        "resolution_of_identity_error": float(
            np.max(np.abs(resolution - np.eye(2)), initial=0.0)
        ),
    }


def operator_panel_in_subspace(
    operators: Mapping[str, object], states: np.ndarray
) -> dict[str, dict[str, object]]:
    """Compress a named operator panel and report basis-invariant norms."""

    answer = {}
    for name, operator in operators.items():
        matrix = projected_operator(operator, states)
        answer[str(name)] = {
            "matrix_real": matrix.real.tolist(),
            "matrix_imag": matrix.imag.tolist(),
            "trace": float(np.trace(matrix).real),
            "traceless_frobenius_norm_squared": traceless_frobenius_norm_squared(matrix),
            "hermiticity_error": float(
                np.max(np.abs(matrix - matrix.conj().T), initial=0.0)
            ),
        }
    return answer


def two_state_block_spin_projectors(
    full_space_states: np.ndarray,
    groups: Sequence[Sequence[int]],
) -> dict[tuple[int, ...], np.ndarray]:
    """Compress every joint block-spin projector to a two-state subspace.

    Four FCS evaluations and polarization identities recover all matrix
    elements.  This is an exact-small reference that remains independent of
    an arbitrary rotation of the supplied low-energy eigenvectors.
    """

    states = orthonormal_subspace(full_space_states)
    if states.shape[1] != 2:
        raise ValueError("block-spin projector compression requires two states")
    first = states[:, 0]
    second = states[:, 1]
    plus = (first + second) / np.sqrt(2.0)
    plus_i = (first + 1.0j * second) / np.sqrt(2.0)
    distributions = [
        block_spin_fcs(state, groups) for state in (first, second, plus, plus_i)
    ]
    support = tuple(sorted(set().union(*(set(row) for row in distributions))))
    answer = {}
    for key in support:
        diagonal_first = float(distributions[0].get(key, 0.0))
        diagonal_second = float(distributions[1].get(key, 0.0))
        mean_diagonal = 0.5 * (diagonal_first + diagonal_second)
        real = float(distributions[2].get(key, 0.0)) - mean_diagonal
        imaginary = mean_diagonal - float(distributions[3].get(key, 0.0))
        off_diagonal = real + 1.0j * imaginary
        answer[key] = np.asarray(
            [
                [diagonal_first, off_diagonal],
                [off_diagonal.conjugate(), diagonal_second],
            ],
            dtype=np.complex128,
        )
    return answer


__all__ = [
    "hermitian_bloch_vector",
    "low_pair_isolation_diagnostics",
    "matrix_free_chirality_action",
    "matrix_free_structure_factor_actions",
    "matrix_free_structure_factor_subspace",
    "matrix_free_weighted_bond_actions",
    "operator_panel_in_subspace",
    "orthonormal_subspace",
    "projected_operator",
    "traceless_frobenius_norm_squared",
    "traceless_part",
    "two_state_block_spin_projectors",
    "two_state_projector_family_analysis",
]
