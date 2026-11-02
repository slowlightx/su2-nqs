"""Exact SU(2)-reduced boundary-channel diagnostics for Heisenberg graphs.

For a left-associated autoregressive sweep, a cut separates an unresolved
left block from the physical sites already peeled into the prefix.  Crossing
Heisenberg terms have the exact factorization

``sum_ij J_ij S_i.S_j = sum_alpha sigma_alpha A_alpha.S B_alpha``

with ``A_alpha = sum_i U[i,alpha] S_i`` and
``B_alpha = sum_j V[j,alpha] S_j``.  Each alpha is one irreducible physical
rank-1 vector channel (doubled tensor label ``q_op=2``), although it contains
three magnetic components in an ordinary tensor-product operator Schmidt
decomposition.  Thus ``rank(J_cross)`` is the minimal SU(2)-reduced channel
count and the unreduced operator Schmidt rank is ``3 * rank(J_cross)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class BoundaryBond:
    """One graph coupling with an optional physical-shell channel."""

    first: int
    second: int
    coupling: float = 1.0
    channel: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.first, bool) or isinstance(self.second, bool):
            raise TypeError("bond endpoints must be integer site indices")
        if not isinstance(self.first, int) or not isinstance(self.second, int):
            raise TypeError("bond endpoints must be integer site indices")
        if self.first == self.second:
            raise ValueError("a boundary bond must join distinct sites")
        if not math.isfinite(self.coupling):
            raise ValueError("bond coupling must be finite")
        if isinstance(self.channel, bool) or not isinstance(self.channel, int):
            raise TypeError("bond channel must be an integer")
        if self.channel < 0:
            raise ValueError("bond channel must be non-negative")


def normalize_boundary_bonds(
    bonds: Iterable[
        BoundaryBond | tuple[int, int] | tuple[int, int, float] | tuple[int, int, float, int]
    ],
) -> tuple[BoundaryBond, ...]:
    """Normalize existing graph tuples without changing endpoint orientation."""

    answer = []
    for bond in bonds:
        if isinstance(bond, BoundaryBond):
            answer.append(bond)
        elif len(bond) == 2:
            first, second = bond
            answer.append(BoundaryBond(int(first), int(second)))
        elif len(bond) == 3:
            first, second, coupling = bond
            answer.append(BoundaryBond(int(first), int(second), float(coupling)))
        elif len(bond) == 4:
            first, second, coupling, channel = bond
            answer.append(
                BoundaryBond(
                    int(first), int(second), float(coupling), int(channel)
                )
            )
        else:
            raise ValueError(f"invalid boundary bond: {bond!r}")
    return tuple(answer)


def left_associated_cut_sites(
    n_sites: int, decision: int
) -> tuple[np.ndarray, np.ndarray]:
    """Physical blocks after a root-first left-tree decision.

    Decision zero peels site ``n_sites-1`` from the unresolved block.  The
    final decision resolves the last two physical leaves simultaneously and
    therefore has an empty hidden block.
    """

    if n_sites < 2:
        raise ValueError("a boundary cut requires at least two sites")
    if not 0 <= decision < n_sites - 1:
        raise ValueError("decision lies outside the left-tree sweep")
    hidden_count = 0 if decision == n_sites - 2 else n_sites - decision - 1
    return (
        np.arange(hidden_count, dtype=np.int32),
        np.arange(hidden_count, n_sites, dtype=np.int32),
    )


def boundary_coupling_matrix(
    n_sites: int,
    bonds: Iterable[
        BoundaryBond | tuple[int, int] | tuple[int, int, float] | tuple[int, int, float, int]
    ],
    decision: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hidden sites, prefix sites, and their crossing ``J_ij`` matrix."""

    hidden, prefix = left_associated_cut_sites(n_sites, decision)
    hidden_position = {int(site): index for index, site in enumerate(hidden)}
    prefix_position = {int(site): index for index, site in enumerate(prefix)}
    coupling = np.zeros((len(hidden), len(prefix)), dtype=np.float64)
    for bond in normalize_boundary_bonds(bonds):
        if not 0 <= bond.first < n_sites or not 0 <= bond.second < n_sites:
            raise ValueError("bond endpoint lies outside the system")
        if bond.first in hidden_position and bond.second in prefix_position:
            coupling[
                hidden_position[bond.first], prefix_position[bond.second]
            ] += bond.coupling
        elif bond.second in hidden_position and bond.first in prefix_position:
            coupling[
                hidden_position[bond.second], prefix_position[bond.first]
            ] += bond.coupling
    return hidden, prefix, coupling


def _numerical_rank(singular_values: np.ndarray, tolerance: float) -> int:
    if len(singular_values) == 0:
        return 0
    threshold = tolerance * max(1.0, float(singular_values[0]))
    return int(np.sum(singular_values > threshold))


@dataclass(frozen=True)
class BoundaryRecouplingPlan:
    """Pure-numeric factorization of vector channels at one AR cut.

    ``left_vectors`` and ``right_vectors`` are orthonormal endpoint
    coefficients.  Together with ``singular_values`` they reconstruct the
    physical crossing matrix.  A forward implementation propagates one
    reduced tensor with ``tensor_rank_q == 2`` per channel; allowed doubled
    boundary-spin changes are ``-2, 0, +2`` subject to the triangle mask.

    ``endpoint_right_coefficients`` is a stable, nonminimal alternative gauge:
    one row per active hidden endpoint, containing all couplings from that
    endpoint to the prefix.  It is useful when SVD gauge rotations between
    successive cuts are less desirable than one extra channel.
    """

    n_sites: int
    decision: int
    hidden_sites: np.ndarray
    prefix_sites: np.ndarray
    coupling_matrix: np.ndarray
    singular_values: np.ndarray
    left_vectors: np.ndarray
    right_vectors: np.ndarray
    endpoint_hidden_sites: np.ndarray
    endpoint_right_coefficients: np.ndarray
    hidden_scalar_bonds: tuple[BoundaryBond, ...]
    tensor_rank_q: int = 2
    boundary_delta_q: tuple[int, int, int] = (-2, 0, 2)

    @property
    def channel_rank(self) -> int:
        return int(len(self.singular_values))

    @property
    def ordinary_magnetic_schmidt_rank(self) -> int:
        return 3 * self.channel_rank

    @property
    def endpoint_channel_count(self) -> int:
        return int(len(self.endpoint_hidden_sites))

    @property
    def hidden_scalar_channel_count(self) -> int:
        """One combined ``q_op=0`` channel for all hidden-only bonds."""

        return int(bool(self.hidden_scalar_bonds))

    @property
    def shell_resolved_hidden_scalar_channel_count(self) -> int:
        return len({bond.channel for bond in self.hidden_scalar_bonds})

    def reconstruct_coupling_matrix(self) -> np.ndarray:
        if self.channel_rank == 0:
            return np.zeros_like(self.coupling_matrix)
        return (self.left_vectors * self.singular_values[None, :]) @ self.right_vectors.T

    def singular_weight_fractions(self) -> np.ndarray:
        squared = self.singular_values**2
        total = float(np.sum(squared))
        return np.cumsum(squared) / total if total > 0.0 else squared

    def valid_boundary_q_transitions(self, maximum_q: int) -> np.ndarray:
        """Triangle mask for a physical rank-1 (doubled ``q_op=2``) tensor."""

        if maximum_q < 0:
            raise ValueError("maximum_q must be non-negative")
        old = np.arange(maximum_q + 1, dtype=np.int32)[:, None]
        new = np.arange(maximum_q + 1, dtype=np.int32)[None, :]
        return (
            (np.abs(old - self.tensor_rank_q) <= new)
            & (new <= old + self.tensor_rank_q)
            & ((old + new + self.tensor_rank_q) % 2 == 0)
        )

    def forward_cost_estimate(self, hidden_dim: int = 32) -> dict[str, int]:
        """Count reduced dynamic values and leading real head multiply-adds."""

        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        rank = self.channel_rank
        endpoint_rank = self.endpoint_channel_count
        return {
            "minimal_complex_transition_values": 3 * rank,
            "endpoint_gauge_complex_transition_values": 3 * endpoint_rank,
            "minimal_real_head_madds": 2 * rank * hidden_dim,
            "endpoint_gauge_real_head_madds": 2 * endpoint_rank * hidden_dim,
            "minimal_factor_storage_floats": rank
            * (len(self.hidden_sites) + len(self.prefix_sites) + 1),
            "endpoint_factor_storage_floats": endpoint_rank
            * (1 + len(self.prefix_sites)),
            "hidden_scalar_complex_values": self.hidden_scalar_channel_count,
            "shell_resolved_hidden_scalar_complex_values": (
                self.shell_resolved_hidden_scalar_channel_count
            ),
            "minimal_total_real_head_madds": 2
            * (rank + self.hidden_scalar_channel_count)
            * hidden_dim,
        }


def compile_boundary_recoupling_plan(
    n_sites: int,
    bonds: Iterable[
        BoundaryBond | tuple[int, int] | tuple[int, int, float] | tuple[int, int, float, int]
    ],
    decision: int,
    *,
    rank_tolerance: float = 1.0e-12,
) -> BoundaryRecouplingPlan:
    """Compile the minimal SU(2) vector-channel SVD at one left-tree cut."""

    if rank_tolerance < 0.0:
        raise ValueError("rank_tolerance must be non-negative")
    normalized = normalize_boundary_bonds(bonds)
    hidden, prefix, coupling = boundary_coupling_matrix(
        n_sites, normalized, decision
    )
    hidden_scalar_bonds = partition_boundary_bonds(
        n_sites, normalized, decision
    )["hidden"]
    if coupling.size == 0:
        singular = np.zeros(0, dtype=np.float64)
        left = np.zeros((len(hidden), 0), dtype=np.float64)
        right = np.zeros((len(prefix), 0), dtype=np.float64)
    else:
        full_left, full_singular, full_right_h = np.linalg.svd(
            coupling, full_matrices=False
        )
        rank = _numerical_rank(full_singular, rank_tolerance)
        singular = full_singular[:rank]
        left = full_left[:, :rank]
        right = full_right_h[:rank, :].T
    endpoint_mask = np.any(np.abs(coupling) > rank_tolerance, axis=1)
    return BoundaryRecouplingPlan(
        n_sites=n_sites,
        decision=decision,
        hidden_sites=hidden,
        prefix_sites=prefix,
        coupling_matrix=coupling,
        singular_values=singular,
        left_vectors=left,
        right_vectors=right,
        endpoint_hidden_sites=hidden[endpoint_mask],
        endpoint_right_coefficients=coupling[endpoint_mask],
        hidden_scalar_bonds=hidden_scalar_bonds,
    )


def advance_boundary_recoupling_plan(
    plan: BoundaryRecouplingPlan,
    bonds: Iterable[
        BoundaryBond | tuple[int, int] | tuple[int, int, float] | tuple[int, int, float, int]
    ],
    *,
    rank_tolerance: float = 1.0e-12,
) -> BoundaryRecouplingPlan:
    """Compile the next cut, retaining a deterministic physical endpoint gauge."""

    if plan.decision >= plan.n_sites - 2:
        raise ValueError("the final cut cannot be advanced")
    return compile_boundary_recoupling_plan(
        plan.n_sites,
        bonds,
        plan.decision + 1,
        rank_tolerance=rank_tolerance,
    )


def partition_boundary_bonds(
    n_sites: int,
    bonds: Iterable[
        BoundaryBond | tuple[int, int] | tuple[int, int, float] | tuple[int, int, float, int]
    ],
    decision: int,
) -> dict[str, tuple[BoundaryBond, ...]]:
    """Partition graph terms into hidden, crossing, and prefix scalar sectors."""

    hidden, prefix = left_associated_cut_sites(n_sites, decision)
    hidden_set = set(map(int, hidden))
    prefix_set = set(map(int, prefix))
    parts: dict[str, list[BoundaryBond]] = {
        "hidden": [],
        "crossing": [],
        "prefix": [],
    }
    for bond in normalize_boundary_bonds(bonds):
        endpoints = {bond.first, bond.second}
        if endpoints <= hidden_set:
            parts["hidden"].append(bond)
        elif endpoints <= prefix_set:
            parts["prefix"].append(bond)
        else:
            parts["crossing"].append(bond)
    return {name: tuple(items) for name, items in parts.items()}


def boundary_operator_dense(plan: BoundaryRecouplingPlan) -> np.ndarray:
    """Materialize only the crossing operator in the computational basis.

    This is an exact small-system reference.  Production code should consume
    the numeric endpoint factors and cached reduced 6j tables instead.
    """

    left_n = len(plan.hidden_sites)
    right_n = len(plan.prefix_sites)
    dimension = 1 << plan.n_sites
    if left_n == 0 or right_n == 0 or plan.channel_rank == 0:
        return np.zeros((dimension, dimension), dtype=np.complex128)
    sx = 0.5 * np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = 0.5 * np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sz = 0.5 * np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    def local_operator(size: int, site: int, value: np.ndarray) -> np.ndarray:
        result = np.asarray([[1.0 + 0.0j]])
        identity = np.eye(2, dtype=np.complex128)
        for position in range(size):
            result = np.kron(result, value if position == site else identity)
        return result

    left_spin = [
        tuple(local_operator(left_n, site, component) for component in (sx, sy, sz))
        for site in range(left_n)
    ]
    right_spin = [
        tuple(local_operator(right_n, site, component) for component in (sx, sy, sz))
        for site in range(right_n)
    ]
    answer = np.zeros((dimension, dimension), dtype=np.complex128)
    for channel in range(plan.channel_rank):
        for component in range(3):
            left_value = sum(
                plan.left_vectors[site, channel] * left_spin[site][component]
                for site in range(left_n)
            )
            right_value = sum(
                plan.right_vectors[site, channel] * right_spin[site][component]
                for site in range(right_n)
            )
            answer += plan.singular_values[channel] * np.kron(
                left_value, right_value
            )
    return answer


def ordinary_operator_schmidt_singular_values(
    operator: np.ndarray, left_sites: int, right_sites: int
) -> np.ndarray:
    """Unreduced operator-Schmidt spectrum across a physical site cut."""

    left_dimension = 1 << left_sites
    right_dimension = 1 << right_sites
    concrete = np.asarray(operator, dtype=np.complex128)
    dimension = left_dimension * right_dimension
    if concrete.shape != (dimension, dimension):
        raise ValueError("operator shape does not match the physical bipartition")
    matricized = concrete.reshape(
        left_dimension, right_dimension, left_dimension, right_dimension
    ).transpose(0, 2, 1, 3).reshape(left_dimension**2, right_dimension**2)
    return np.linalg.svd(matricized, compute_uv=False)


@dataclass(frozen=True)
class PrefixSuffixOperatorMatrix:
    """Explicit ``H[new,old]`` matricization by AR transition keys."""

    values: np.ndarray
    prefix_transition_keys: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    suffix_transition_keys: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    prefix_length: int


def prefix_suffix_operator_matrix(
    matrix: object,
    action_sequences: Sequence[Sequence[int]] | np.ndarray,
    prefix_length: int,
) -> PrefixSuffixOperatorMatrix:
    """Matricize an exact small operator by target/source prefix and suffix.

    Rows use ``(new_prefix, old_prefix)`` and columns use
    ``(new_suffix, old_suffix)``.  Sparse columns are traversed in the project
    orientation ``H[new, old]``.  This irregular fusion-path matricization is a
    validation aid; its ordinary rank must not be confused with the reduced
    physical vector-channel count.
    """

    actions = np.asarray(action_sequences)
    if actions.ndim != 2 or not np.issubdtype(actions.dtype, np.integer):
        raise TypeError("action_sequences must be a two-dimensional integer array")
    if not 0 <= prefix_length <= actions.shape[1]:
        raise ValueError("prefix_length lies outside the action sequence")
    csc = matrix.tocsc()
    if csc.shape != (len(actions), len(actions)):
        raise ValueError("operator and action basis dimensions must agree")
    entries = []
    prefix_keys = set()
    suffix_keys = set()
    for old in range(len(actions)):
        start, stop = csc.indptr[old : old + 2]
        for new, element in zip(
            csc.indices[start:stop], csc.data[start:stop], strict=True
        ):
            prefix_key = (
                tuple(map(int, actions[int(new), :prefix_length])),
                tuple(map(int, actions[old, :prefix_length])),
            )
            suffix_key = (
                tuple(map(int, actions[int(new), prefix_length:])),
                tuple(map(int, actions[old, prefix_length:])),
            )
            prefix_keys.add(prefix_key)
            suffix_keys.add(suffix_key)
            entries.append((prefix_key, suffix_key, complex(element)))
    ordered_prefix = tuple(sorted(prefix_keys))
    ordered_suffix = tuple(sorted(suffix_keys))
    prefix_position = {key: index for index, key in enumerate(ordered_prefix)}
    suffix_position = {key: index for index, key in enumerate(ordered_suffix)}
    values = np.zeros(
        (len(ordered_prefix), len(ordered_suffix)), dtype=np.complex128
    )
    for prefix_key, suffix_key, element in entries:
        values[prefix_position[prefix_key], suffix_position[suffix_key]] += element
    return PrefixSuffixOperatorMatrix(
        values, ordered_prefix, ordered_suffix, prefix_length
    )


def reconstruct_prefix_suffix_operator(
    matricization: PrefixSuffixOperatorMatrix,
    action_sequences: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray:
    """Reconstruct dense ``H[new,old]`` entries from a transition matrix."""

    actions = np.asarray(action_sequences)
    prefix_position = {
        key: index
        for index, key in enumerate(matricization.prefix_transition_keys)
    }
    suffix_position = {
        key: index
        for index, key in enumerate(matricization.suffix_transition_keys)
    }
    result = np.zeros((len(actions), len(actions)), dtype=np.complex128)
    length = matricization.prefix_length
    for old in range(len(actions)):
        for new in range(len(actions)):
            prefix_key = (
                tuple(map(int, actions[new, :length])),
                tuple(map(int, actions[old, :length])),
            )
            suffix_key = (
                tuple(map(int, actions[new, length:])),
                tuple(map(int, actions[old, length:])),
            )
            if prefix_key in prefix_position and suffix_key in suffix_position:
                result[new, old] = matricization.values[
                    prefix_position[prefix_key], suffix_position[suffix_key]
                ]
    return result
