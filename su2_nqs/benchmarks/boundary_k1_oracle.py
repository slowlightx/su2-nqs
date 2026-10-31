"""Exhaustive rank-one boundary-channel oracles for autoregressive cuts.

This module deliberately separates two notions which are easy to conflate:

* :func:`causal_centered_features` centers quantities which are already
  computable from a revealed autoregressive prefix.  It is suitable for a
  production causal model.
* :func:`conditional_suffix_oracle` contracts an operator local energy over
  every hidden suffix.  The returned value is prefix measurable, but its
  *construction* uses the complete wavefunction and is therefore only an
  exhaustive small-system oracle.  It is an upper bound for a future reduced
  Wigner--Eckart message, not an implementation of that message.

At a physical cut, one singular component of the crossing coupling matrix is
one SU(2)-reduced vector channel (doubled operator label ``q_op=2``).  Its
fusion-space matrix elements are resolved by the allowed boundary-spin
changes ``delta_q=(-2, 0, +2)``.  Connectivity always follows
``H[new_path, old_path]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .boundary_operator_channels import BoundaryRecouplingPlan
from ..operators.permutation import heisenberg_graph_fusion_matrix


BOUNDARY_DELTA_Q: tuple[int, int, int] = (-2, 0, 2)


@dataclass(frozen=True)
class RankOneBoundaryChannel:
    """One deterministic physical ``q_op=2`` crossing channel.

    ``coupling_matrix`` is expressed in the hidden/prefix site order of the
    parent :class:`BoundaryRecouplingPlan`.  ``left_coefficients`` and
    ``right_coefficients`` include the channel scale symmetrically, so their
    outer product reconstructs ``coupling_matrix``.
    """

    n_sites: int
    decision: int
    gauge: str
    channel_index: int
    hidden_sites: np.ndarray
    prefix_sites: np.ndarray
    coupling_matrix: np.ndarray
    left_coefficients: np.ndarray
    right_coefficients: np.ndarray
    retained_frobenius_fraction: float
    tensor_rank_q: int = 2
    boundary_delta_q: tuple[int, int, int] = BOUNDARY_DELTA_Q

    def reconstruct_coupling_matrix(self) -> np.ndarray:
        return np.outer(self.left_coefficients, self.right_coefficients)

    def weighted_bonds(
        self, *, zero_tolerance: float = 1.0e-14
    ) -> tuple[tuple[int, int, float], ...]:
        """Return physical bonds for this rank-one component."""

        if zero_tolerance < 0.0:
            raise ValueError("zero_tolerance must be non-negative")
        bonds = []
        for left_position, first in enumerate(self.hidden_sites):
            for right_position, second in enumerate(self.prefix_sites):
                coupling = float(
                    self.coupling_matrix[left_position, right_position]
                )
                if abs(coupling) > zero_tolerance:
                    bonds.append((int(first), int(second), coupling))
        return tuple(bonds)


def _canonical_pair(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fix the simultaneous SVD sign by a stable left-then-right pivot."""

    first = np.asarray(left, dtype=np.float64).copy()
    second = np.asarray(right, dtype=np.float64).copy()
    if len(first):
        pivot_value = float(first[int(np.argmax(np.abs(first)))])
    elif len(second):
        pivot_value = float(second[int(np.argmax(np.abs(second)))])
    else:
        pivot_value = 0.0
    if pivot_value < 0.0:
        first *= -1.0
        second *= -1.0
    return first, second


def compile_rank_one_boundary_channel(
    plan: BoundaryRecouplingPlan,
    *,
    gauge: str = "svd",
    channel_index: int = 0,
) -> RankOneBoundaryChannel:
    """Select one exact reduced vector factor from a boundary plan.

    ``gauge='svd'`` selects the requested minimal singular component and fixes
    its otherwise arbitrary sign.  ``gauge='endpoint'`` selects a single
    hidden endpoint (ordered by physical site); this gauge is nonminimal but
    stable as the cut advances.  Both returned objects are rank one.  Neither
    claims to reconstruct a multichannel crossing matrix by itself.
    """

    if gauge not in {"svd", "endpoint"}:
        raise ValueError("gauge must be 'svd' or 'endpoint'")
    if isinstance(channel_index, bool) or not isinstance(channel_index, int):
        raise TypeError("channel_index must be an integer")
    if channel_index < 0:
        raise ValueError("channel_index must be non-negative")

    total_squared = float(np.sum(np.asarray(plan.coupling_matrix) ** 2))
    if gauge == "svd":
        if channel_index >= plan.channel_rank:
            raise ValueError("SVD channel_index lies outside the boundary rank")
        singular = float(plan.singular_values[channel_index])
        scale = np.sqrt(singular)
        left, right = _canonical_pair(
            scale * plan.left_vectors[:, channel_index],
            scale * plan.right_vectors[:, channel_index],
        )
    else:
        if channel_index >= plan.endpoint_channel_count:
            raise ValueError("endpoint channel_index lies outside the boundary rank")
        endpoint_site = int(plan.endpoint_hidden_sites[channel_index])
        positions = np.flatnonzero(plan.hidden_sites == endpoint_site)
        if len(positions) != 1:
            raise RuntimeError("endpoint gauge site is absent or ambiguous")
        left = np.zeros(len(plan.hidden_sites), dtype=np.float64)
        left[int(positions[0])] = 1.0
        right = np.asarray(
            plan.endpoint_right_coefficients[channel_index], dtype=np.float64
        ).copy()
        left, right = _canonical_pair(left, right)

    coupling = np.outer(left, right)
    retained = (
        float(np.sum(coupling**2)) / total_squared if total_squared > 0.0 else 0.0
    )
    return RankOneBoundaryChannel(
        n_sites=plan.n_sites,
        decision=plan.decision,
        gauge=gauge,
        channel_index=channel_index,
        hidden_sites=np.asarray(plan.hidden_sites, dtype=np.int32),
        prefix_sites=np.asarray(plan.prefix_sites, dtype=np.int32),
        coupling_matrix=coupling,
        left_coefficients=left,
        right_coefficients=right,
        retained_frobenius_fraction=retained,
    )


def rank_one_fusion_operator(
    channel: RankOneBoundaryChannel, *, target_q: int = 0
):
    """Build the exact fusion-space scalar product for one vector channel."""

    return heisenberg_graph_fusion_matrix(
        channel.n_sites, channel.weighted_bonds(), target_q=target_q
    )


@dataclass(frozen=True)
class BoundaryDeltaContributions:
    """Local-energy contributions resolved by doubled boundary-spin change."""

    values: np.ndarray
    delta_q: tuple[int, int, int]
    local_energies: np.ndarray
    maximum_reconstruction_error: float
    maximum_forbidden_matrix_element: float


def boundary_delta_local_contributions(
    matrix: object,
    basis_paths: Sequence[Sequence[int]] | np.ndarray,
    amplitudes: Sequence[complex] | np.ndarray,
    boundary_label_index: int | None,
    *,
    fixed_boundary_q: int = 1,
    zero_tolerance: float = 2.0e-12,
) -> BoundaryDeltaContributions:
    """Resolve ``(H psi)(old)/psi(old)`` by ``new_q-old_q``.

    Sparse columns are traversed in the mandatory project orientation
    ``H[new, old]``.  ``boundary_label_index=None`` represents a one-site
    hidden block with fixed doubled spin ``q=1``.
    """

    basis = np.asarray(basis_paths)
    psi = np.asarray(amplitudes, dtype=np.complex128)
    if basis.ndim != 2 or not np.issubdtype(basis.dtype, np.integer):
        raise TypeError("basis_paths must be a doubled-integer matrix")
    if psi.shape != (len(basis),):
        raise ValueError("amplitudes must have one entry per basis path")
    if np.any(~np.isfinite(psi)) or np.any(np.abs(psi) == 0.0):
        raise ValueError("amplitudes must be finite and nonzero")
    if zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be non-negative")
    if boundary_label_index is None:
        boundary_q = np.full(len(basis), fixed_boundary_q, dtype=np.int32)
    else:
        if not 0 <= boundary_label_index < basis.shape[1]:
            raise ValueError("boundary_label_index lies outside the path")
        boundary_q = np.asarray(basis[:, boundary_label_index], dtype=np.int32)

    csc = matrix.tocsc()
    if csc.shape != (len(basis), len(basis)):
        raise ValueError("operator and basis dimensions must agree")
    values = np.zeros((len(basis), len(BOUNDARY_DELTA_Q)), dtype=np.complex128)
    slot = {delta: index for index, delta in enumerate(BOUNDARY_DELTA_Q)}
    maximum_forbidden = 0.0
    for old in range(len(basis)):
        start, stop = csc.indptr[old : old + 2]
        for new, element in zip(
            csc.indices[start:stop], csc.data[start:stop], strict=True
        ):
            delta = int(boundary_q[int(new)] - boundary_q[old])
            if delta not in slot:
                maximum_forbidden = max(maximum_forbidden, abs(complex(element)))
                continue
            values[old, slot[delta]] += complex(element) * psi[int(new)] / psi[old]
    if maximum_forbidden > zero_tolerance:
        raise RuntimeError(
            "a q_op=2 boundary channel contains a forbidden doubled-spin "
            f"transition of magnitude {maximum_forbidden:.3e}"
        )
    local = np.asarray(csc.T @ psi, dtype=np.complex128) / psi
    # csc.T is intentional: matrix columns store H[new,old], while SciPy's
    # matrix-vector product contracts the second index.  For Hermitian real
    # physical matrices this equals H @ psi, but this expression also makes
    # the connectivity orientation explicit and testable on directed inputs.
    error = float(np.max(np.abs(np.sum(values, axis=1) - local), initial=0.0))
    return BoundaryDeltaContributions(
        values=values,
        delta_q=BOUNDARY_DELTA_Q,
        local_energies=local,
        maximum_reconstruction_error=error,
        maximum_forbidden_matrix_element=float(maximum_forbidden),
    )


def _normalized_weights(weights: np.ndarray, size: int) -> np.ndarray:
    concrete = np.asarray(weights, dtype=np.float64)
    if concrete.shape != (size,):
        raise ValueError("probabilities must have one entry per state")
    if np.any(~np.isfinite(concrete)) or np.any(concrete < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(np.sum(concrete))
    if total <= 0.0:
        raise ValueError("probabilities must have positive total")
    return concrete / total


def _conditional_mean(
    values: np.ndarray,
    actions: np.ndarray,
    probabilities: np.ndarray,
    prefix_length: int,
) -> np.ndarray:
    if not 0 <= prefix_length <= actions.shape[1]:
        raise ValueError("prefix_length lies outside the action sequence")
    flattened = values.reshape(len(values), -1)
    totals: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
    for row, value, probability in zip(
        actions, flattened, probabilities, strict=True
    ):
        key = tuple(int(action) for action in row[:prefix_length])
        if key not in totals:
            totals[key] = (0.0, np.zeros(flattened.shape[1], dtype=np.complex128))
        mass, numerator = totals[key]
        totals[key] = (mass + float(probability), numerator + probability * value)
    means = {key: numerator / mass for key, (mass, numerator) in totals.items()}
    result = np.asarray(
        [means[tuple(int(action) for action in row[:prefix_length])] for row in actions]
    )
    return result.reshape(values.shape)


def _prefix_invariance_error(
    values: np.ndarray, actions: np.ndarray, prefix_length: int
) -> float:
    flattened = values.reshape(len(values), -1)
    references: dict[tuple[int, ...], np.ndarray] = {}
    maximum = 0.0
    for row, value in zip(actions, flattened, strict=True):
        key = tuple(int(action) for action in row[:prefix_length])
        if key in references:
            maximum = max(
                maximum,
                float(np.max(np.abs(value - references[key]), initial=0.0)),
            )
        else:
            references[key] = value
    return maximum


def _conditional_centering_error(
    centered: np.ndarray,
    actions: np.ndarray,
    probabilities: np.ndarray,
    prefix_length: int,
) -> float:
    mean = _conditional_mean(centered, actions, probabilities, prefix_length)
    return float(np.max(np.abs(mean), initial=0.0))


@dataclass(frozen=True)
class ConditionalSuffixOracle:
    """Prefix-measurable result of an explicitly noncausal suffix contraction."""

    candidate_messages: np.ndarray
    previous_messages: np.ndarray
    centered_messages: np.ndarray
    decision: int
    maximum_prefix_invariance_error: float
    maximum_valid_action_centering_error: float
    uses_hidden_suffix: bool = True


def conditional_suffix_oracle(
    local_contributions: np.ndarray,
    action_sequences: Sequence[Sequence[int]] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    decision: int,
) -> ConditionalSuffixOracle:
    """Contract hidden suffixes and center over every valid next action.

    If ``z_delta(x)`` is a channel-resolved local contribution, the candidate
    message is ``E[z_delta | a_0,...,a_d]``.  Subtracting its expectation over
    valid ``a_d`` at fixed earlier prefix gives a martingale difference.  The
    result is exactly causal *as a table*, but obtaining that table requires
    all suffixes and amplitudes, hence ``uses_hidden_suffix=True``.
    """

    values = np.asarray(local_contributions, dtype=np.complex128)
    actions = np.asarray(action_sequences)
    if values.ndim < 1 or actions.ndim != 2 or len(values) != len(actions):
        raise ValueError("contributions and actions must share their state axis")
    if not np.issubdtype(actions.dtype, np.integer):
        raise TypeError("autoregressive actions must be integers")
    if not 0 <= decision < actions.shape[1]:
        raise ValueError("decision lies outside the action sequence")
    weights = _normalized_weights(np.asarray(probabilities), len(actions))
    candidate = _conditional_mean(values, actions, weights, decision + 1)
    previous = _conditional_mean(values, actions, weights, decision)
    centered = candidate - previous
    return ConditionalSuffixOracle(
        candidate_messages=candidate,
        previous_messages=previous,
        centered_messages=centered,
        decision=decision,
        maximum_prefix_invariance_error=_prefix_invariance_error(
            candidate, actions, decision + 1
        ),
        maximum_valid_action_centering_error=_conditional_centering_error(
            centered, actions, weights, decision
        ),
    )


def causal_centered_features(
    prefix_features: np.ndarray,
    action_sequences: Sequence[Sequence[int]] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    decision: int,
    *,
    invariance_tolerance: float = 2.0e-12,
) -> np.ndarray:
    """Center genuinely prefix-computable candidate features over valid actions.

    This routine refuses a feature which varies between complete paths sharing
    the revealed prefix.  Unlike :func:`conditional_suffix_oracle`, it performs
    no suffix contraction and is therefore a production-causality gate.
    """

    features = np.asarray(prefix_features)
    actions = np.asarray(action_sequences)
    if features.ndim < 1 or actions.ndim != 2 or len(features) != len(actions):
        raise ValueError("features and actions must share their state axis")
    if not 0 <= decision < actions.shape[1]:
        raise ValueError("decision lies outside the action sequence")
    error = _prefix_invariance_error(features, actions, decision + 1)
    if error > invariance_tolerance:
        raise ValueError(
            "feature reads an unrevealed suffix: within-prefix discrepancy "
            f"{error:.3e}"
        )
    weights = _normalized_weights(np.asarray(probabilities), len(actions))
    return features - _conditional_mean(features, actions, weights, decision)


def real_design(values: np.ndarray) -> np.ndarray:
    """Split arbitrary complex feature columns into real and imaginary parts."""

    concrete = np.asarray(values, dtype=np.complex128).reshape(len(values), -1)
    return np.concatenate((np.real(concrete), np.imag(concrete)), axis=1)
