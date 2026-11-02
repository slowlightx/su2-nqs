"""Exact small-system dynamics and local-action hierarchy diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvolutionComparison:
    """Errors of one approximate trajectory against exact time evolution."""

    maximum_infidelity: float
    maximum_norm_error: float
    maximum_energy_error: float


def _checked_hamiltonian(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Hamiltonian must be a square matrix")
    if not np.allclose(values, values.conj().T, atol=2.0e-13):
        raise ValueError("Hamiltonian must be Hermitian")
    return values


def _checked_state(state: np.ndarray, dimension: int) -> np.ndarray:
    values = np.asarray(state, dtype=np.complex128)
    if values.shape != (dimension,):
        raise ValueError(f"state must have shape ({dimension},)")
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("state must have finite nonzero norm")
    return values / norm


def ground_state(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the lowest eigenvalue and a normalized eigenvector."""

    hamiltonian = _checked_hamiltonian(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return float(eigenvalues[0]), np.asarray(eigenvectors[:, 0])


def exact_time_evolution(
    matrix: np.ndarray,
    initial_state: np.ndarray,
    times: Sequence[float],
) -> np.ndarray:
    """Evaluate ``exp(-1j*H*t)|psi0>`` by exact Hermitian diagonalization."""

    hamiltonian = _checked_hamiltonian(matrix)
    state = _checked_state(initial_state, hamiltonian.shape[0])
    time_values = np.asarray(times, dtype=np.float64)
    if time_values.ndim != 1 or np.any(~np.isfinite(time_values)):
        raise ValueError("times must be a finite one-dimensional sequence")
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    coefficients = eigenvectors.conj().T @ state
    phases = np.exp(-1j * time_values[:, None] * eigenvalues[None, :])
    return np.asarray((phases * coefficients[None, :]) @ eigenvectors.T)


def crank_nicolson_evolution(
    matrix: np.ndarray,
    initial_state: np.ndarray,
    *,
    time_step: float,
    steps: int,
) -> np.ndarray:
    """Propagate a time-independent Hamiltonian with unitary Crank--Nicolson."""

    hamiltonian = _checked_hamiltonian(matrix)
    state = _checked_state(initial_state, hamiltonian.shape[0])
    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step must be positive and finite")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    identity = np.eye(hamiltonian.shape[0], dtype=np.complex128)
    left = identity + 0.5j * float(time_step) * hamiltonian
    right = identity - 0.5j * float(time_step) * hamiltonian
    propagator = np.linalg.solve(left, right)
    trajectory = [state]
    for _ in range(int(steps)):
        state = propagator @ state
        trajectory.append(state)
    return np.asarray(trajectory)


def expectation_trajectory(states: np.ndarray, operator: np.ndarray) -> np.ndarray:
    """Return normalized expectations for a trajectory of state vectors."""

    vectors = np.asarray(states, dtype=np.complex128)
    observable = np.asarray(operator, dtype=np.complex128)
    if vectors.ndim != 2 or observable.shape != (vectors.shape[1], vectors.shape[1]):
        raise ValueError("states/operator shapes are incompatible")
    numerators = np.einsum("ti,ij,tj->t", vectors.conj(), observable, vectors)
    norms = np.einsum("ti,ti->t", vectors.conj(), vectors)
    return np.real_if_close(numerators / norms)


def state_fidelities(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Gauge-invariant squared overlaps between two state trajectories."""

    first = np.asarray(reference, dtype=np.complex128)
    second = np.asarray(candidate, dtype=np.complex128)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("trajectories must have the same rank-two shape")
    overlaps = np.einsum("ti,ti->t", first.conj(), second)
    first_norm = np.einsum("ti,ti->t", first.conj(), first).real
    second_norm = np.einsum("ti,ti->t", second.conj(), second).real
    return np.abs(overlaps) ** 2 / (first_norm * second_norm)


def bipartite_entropy_trajectory(
    states: np.ndarray,
    left_dimension: int,
    *,
    base: float = np.e,
) -> np.ndarray:
    """Return pure-state bipartite entropies from a product-basis trajectory."""

    vectors = np.asarray(states, dtype=np.complex128)
    if vectors.ndim != 2:
        raise ValueError("states must form a rank-two trajectory")
    if (
        isinstance(left_dimension, bool)
        or not isinstance(left_dimension, (int, np.integer))
        or left_dimension < 1
        or vectors.shape[1] % int(left_dimension)
    ):
        raise ValueError("left_dimension must divide the state dimension")
    if not np.isfinite(base) or base <= 0.0 or base == 1.0:
        raise ValueError("logarithm base must be positive and different from one")
    right_dimension = vectors.shape[1] // int(left_dimension)
    entropies = []
    for state in vectors:
        normalized = _checked_state(state, vectors.shape[1])
        singular_values = np.linalg.svd(
            normalized.reshape(int(left_dimension), right_dimension),
            compute_uv=False,
        )
        probabilities = singular_values**2
        probabilities = probabilities[probabilities > 1.0e-15]
        entropies.append(-np.sum(probabilities * np.log(probabilities)) / np.log(base))
    return np.asarray(entropies, dtype=np.float64)


def compare_trajectories(
    reference: np.ndarray,
    candidate: np.ndarray,
    hamiltonian: np.ndarray,
) -> EvolutionComparison:
    """Summarize fidelity, norm, and conserved-energy discrepancies."""

    fidelities = state_fidelities(reference, candidate)
    candidate_norms = np.sum(np.abs(candidate) ** 2, axis=1)
    reference_energy = expectation_trajectory(reference, hamiltonian)
    candidate_energy = expectation_trajectory(candidate, hamiltonian)
    return EvolutionComparison(
        maximum_infidelity=float(np.max(np.maximum(0.0, 1.0 - fidelities))),
        maximum_norm_error=float(np.max(np.abs(candidate_norms - 1.0))),
        maximum_energy_error=float(np.max(np.abs(candidate_energy - reference_energy))),
    )


def orthonormal_span(
    vectors: Sequence[np.ndarray] | np.ndarray,
    *,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Return a deterministic orthonormal basis for the supplied columns."""

    matrix = np.asarray(vectors, dtype=np.complex128)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("vectors must form a rank-two column matrix")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular_values) or singular_values[0] == 0.0:
        return np.empty((matrix.shape[0], 0), dtype=np.complex128)
    rank = int(np.sum(singular_values > tolerance * singular_values[0]))
    basis = np.asarray(left[:, :rank])
    # Fix arbitrary SVD column phases for deterministic artifacts.
    for column in range(rank):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        phase = basis[pivot, column]
        if phase != 0.0:
            basis[:, column] *= np.conj(phase) / abs(phase)
    return basis


def global_krylov_basis(
    hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    maximum_depth: int,
    *,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Span ``psi,H psi,...,H**maximum_depth psi``."""

    matrix = _checked_hamiltonian(hamiltonian)
    state = _checked_state(initial_state, matrix.shape[0])
    if maximum_depth < 0:
        raise ValueError("maximum_depth must be non-negative")
    columns = [state]
    current = state
    for _ in range(maximum_depth):
        current = matrix @ current
        columns.append(current)
    return orthonormal_span(np.column_stack(columns), tolerance=tolerance)


def _checked_local_action_problem(
    initial_state: np.ndarray,
    local_operators: Sequence[np.ndarray],
    bonds: Sequence[tuple[int, int]],
    maximum_depth: int,
    *,
    initial_operator_indices: Sequence[int] | None = None,
) -> tuple[
    np.ndarray,
    tuple[np.ndarray, ...],
    tuple[tuple[int, int], ...],
    tuple[int, ...],
]:
    operators = tuple(_checked_hamiltonian(value) for value in local_operators)
    checked_bonds = tuple((int(first), int(second)) for first, second in bonds)
    if len(operators) != len(checked_bonds):
        raise ValueError("local_operators and bonds must have equal length")
    if not operators:
        raise ValueError("at least one local operator is required")
    dimension = operators[0].shape[0]
    if any(operator.shape != (dimension, dimension) for operator in operators):
        raise ValueError("all local operators must have the same shape")
    state = _checked_state(initial_state, dimension)
    if maximum_depth < 0:
        raise ValueError("maximum_depth must be non-negative")
    if initial_operator_indices is None:
        initial_indices = tuple(range(len(operators)))
    else:
        initial_indices = tuple(int(value) for value in initial_operator_indices)
        if not initial_indices or any(
            value < 0 or value >= len(operators) for value in initial_indices
        ):
            raise ValueError("initial_operator_indices contain an invalid bond")
    return state, operators, checked_bonds, initial_indices


def _extend_action_components(
    components: tuple[frozenset[int], ...],
    bond: tuple[int, int],
) -> tuple[frozenset[int], ...]:
    support = frozenset(bond)
    touching = tuple(
        index
        for index, component in enumerate(components)
        if not component.isdisjoint(support)
    )
    if not touching:
        return components + (support,)
    merged = support.union(*(components[index] for index in touching))
    result = [
        component
        for index, component in enumerate(components)
        if index not in touching
    ]
    result.append(frozenset(merged))
    return tuple(sorted(result, key=lambda value: (min(value), len(value))))


def _local_action_columns(
    initial_state: np.ndarray,
    local_operators: Sequence[np.ndarray],
    bonds: Sequence[tuple[int, int]],
    maximum_depth: int,
    *,
    maximum_clusters: int | None,
    initial_operator_indices: Sequence[int] | None,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    state, operators, checked_bonds, initial_indices = _checked_local_action_problem(
        initial_state,
        local_operators,
        bonds,
        maximum_depth,
        initial_operator_indices=initial_operator_indices,
    )
    if maximum_clusters is not None and (
        isinstance(maximum_clusters, bool)
        or not isinstance(maximum_clusters, (int, np.integer))
        or maximum_clusters < 1
    ):
        raise ValueError("maximum_clusters must be positive or None")

    sequences: list[tuple[int, ...]] = [()]
    vectors: list[np.ndarray] = [state]
    frontier = [((), (), state)]
    for _ in range(maximum_depth):
        updated = []
        for sequence, components, vector in frontier:
            allowed_indices = initial_indices if not sequence else range(len(operators))
            for index in allowed_indices:
                operator = operators[index]
                bond = checked_bonds[index]
                next_components = _extend_action_components(components, bond)
                if (
                    maximum_clusters is not None
                    and len(next_components) > maximum_clusters
                ):
                    continue
                candidate = sequence + (index,)
                candidate_vector = operator @ vector
                sequences.append(candidate)
                vectors.append(candidate_vector)
                updated.append((candidate, next_components, candidate_vector))
        frontier = updated
    return np.column_stack(vectors), tuple(sequences)


def cluster_local_action_basis(
    initial_state: np.ndarray,
    local_operators: Sequence[np.ndarray],
    bonds: Sequence[tuple[int, int]],
    maximum_depth: int,
    *,
    maximum_clusters: int | None = 1,
    tolerance: float = 1.0e-12,
    initial_operator_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Span local-action strings with a bounded number of spatial clusters.

    ``maximum_clusters=1`` permits only connected supports, while ``None``
    gives the unrestricted resolved-action space.  A local quench can anchor
    the first action with ``initial_operator_indices``.  Repeated actions and
    distinct operator orderings are retained before the final rank reveal.
    """

    columns, sequences = _local_action_columns(
        initial_state,
        local_operators,
        bonds,
        maximum_depth,
        maximum_clusters=maximum_clusters,
        initial_operator_indices=initial_operator_indices,
    )
    return orthonormal_span(columns, tolerance=tolerance), sequences


def principal_local_action_basis(
    initial_state: np.ndarray,
    local_operators: Sequence[np.ndarray],
    bonds: Sequence[tuple[int, int]],
    maximum_depth: int,
    dimension: int,
    *,
    maximum_clusters: int | None = 1,
    tolerance: float = 1.0e-12,
    initial_operator_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Return an initial-state-preserving, rank-limited action-PCA basis.

    Candidate action vectors are normalized, projected perpendicular to the
    initial state, normalized again, and ranked by singular value.  This is a
    target-trajectory-independent control for comparisons at fixed subspace
    dimension; it does not match action-generation cost.
    """

    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, (int, np.integer))
        or dimension < 1
    ):
        raise ValueError("dimension must be a positive integer")
    columns, sequences = _local_action_columns(
        initial_state,
        local_operators,
        bonds,
        maximum_depth,
        maximum_clusters=maximum_clusters,
        initial_operator_indices=initial_operator_indices,
    )
    state = columns[:, 0]
    if dimension == 1 or columns.shape[1] == 1:
        return state[:, None], sequences
    candidates = columns[:, 1:]
    norms = np.linalg.norm(candidates, axis=0)
    candidates = candidates[:, norms > tolerance]
    norms = norms[norms > tolerance]
    if not candidates.shape[1]:
        return state[:, None], sequences
    candidates = candidates / norms
    residuals = candidates - state[:, None] * (state.conj() @ candidates)[None, :]
    residual_norms = np.linalg.norm(residuals, axis=0)
    residuals = residuals[:, residual_norms > tolerance]
    residual_norms = residual_norms[residual_norms > tolerance]
    if not residuals.shape[1]:
        return state[:, None], sequences
    directions = orthonormal_span(
        residuals / residual_norms,
        tolerance=tolerance,
    )
    directions = directions[:, : max(0, min(int(dimension) - 1, directions.shape[1]))]
    return np.column_stack((state, directions)), sequences


def connected_local_action_basis(
    initial_state: np.ndarray,
    local_operators: Sequence[np.ndarray],
    bonds: Sequence[tuple[int, int]],
    maximum_depth: int,
    *,
    tolerance: float = 1.0e-12,
    initial_operator_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Span independently resolved connected sequences of local bond actions."""

    return cluster_local_action_basis(
        initial_state,
        local_operators,
        bonds,
        maximum_depth,
        maximum_clusters=1,
        tolerance=tolerance,
        initial_operator_indices=initial_operator_indices,
    )


def projected_time_evolution(
    hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    basis: np.ndarray,
    times: Sequence[float],
) -> np.ndarray:
    """Evolve in an orthonormal variational subspace and lift to full space."""

    matrix = _checked_hamiltonian(hamiltonian)
    state = _checked_state(initial_state, matrix.shape[0])
    subspace = np.asarray(basis, dtype=np.complex128)
    if subspace.ndim != 2 or subspace.shape[0] != matrix.shape[0]:
        raise ValueError("basis has an incompatible shape")
    gram = subspace.conj().T @ subspace
    if not np.allclose(gram, np.eye(subspace.shape[1]), atol=5.0e-11):
        raise ValueError("basis columns must be orthonormal")
    projected_hamiltonian = subspace.conj().T @ matrix @ subspace
    projected_state = subspace.conj().T @ state
    projected_trajectory = exact_time_evolution(
        projected_hamiltonian, projected_state, times
    )
    return np.asarray(projected_trajectory @ subspace.T)


__all__ = [
    "EvolutionComparison",
    "compare_trajectories",
    "bipartite_entropy_trajectory",
    "cluster_local_action_basis",
    "connected_local_action_basis",
    "crank_nicolson_evolution",
    "exact_time_evolution",
    "expectation_trajectory",
    "global_krylov_basis",
    "ground_state",
    "orthonormal_span",
    "principal_local_action_basis",
    "projected_time_evolution",
    "state_fidelities",
]
