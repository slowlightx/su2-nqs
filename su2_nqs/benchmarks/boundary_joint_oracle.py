"""Exact hidden/crossing/closed sector oracles at autoregressive cuts.

The physical Hamiltonian is repartitioned at every cut into operators acting
inside the unresolved hidden block, across the cut, and inside the revealed
prefix.  Conditional suffix contractions of these operators provide an exact
small-system upper bound for joint vector/scalar messages.  They are not a
causal production implementation: every returned martingale table depends on
the full wavefunction during construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .boundary_k1_oracle import ConditionalSuffixOracle, conditional_suffix_oracle
from .boundary_operator_channels import (
    BoundaryBond,
    normalize_boundary_bonds,
    partition_boundary_bonds,
)
from ..operators.permutation import heisenberg_graph_fusion_matrix


@dataclass(frozen=True)
class BoundarySectorOperators:
    """Sparse ``H[new,old]`` operators in the three physical cut sectors."""

    n_sites: int
    decision: int
    hidden_bonds: tuple[BoundaryBond, ...]
    crossing_bonds: tuple[BoundaryBond, ...]
    prefix_bonds: tuple[BoundaryBond, ...]
    hidden_operator: object
    crossing_operator: object
    prefix_operator: object
    full_operator: object

    def reconstructed_operator(self):
        return self.hidden_operator + self.crossing_operator + self.prefix_operator


def _bond_tuples(bonds: Sequence[BoundaryBond]) -> tuple[tuple[int, int, float], ...]:
    return tuple((bond.first, bond.second, bond.coupling) for bond in bonds)


def compile_boundary_sector_operators(
    n_sites: int,
    bonds: Iterable[
        BoundaryBond | tuple[int, int] | tuple[int, int, float] | tuple[int, int, float, int]
    ],
    decision: int,
    *,
    target_q: int = 0,
) -> BoundarySectorOperators:
    """Build the exact hidden/crossing/prefix Hamiltonian partition."""

    concrete = normalize_boundary_bonds(bonds)
    parts = partition_boundary_bonds(n_sites, concrete, decision)
    hidden = heisenberg_graph_fusion_matrix(
        n_sites, _bond_tuples(parts["hidden"]), target_q=target_q
    )
    crossing = heisenberg_graph_fusion_matrix(
        n_sites, _bond_tuples(parts["crossing"]), target_q=target_q
    )
    prefix = heisenberg_graph_fusion_matrix(
        n_sites, _bond_tuples(parts["prefix"]), target_q=target_q
    )
    full = heisenberg_graph_fusion_matrix(
        n_sites, _bond_tuples(concrete), target_q=target_q
    )
    return BoundarySectorOperators(
        n_sites=n_sites,
        decision=decision,
        hidden_bonds=parts["hidden"],
        crossing_bonds=parts["crossing"],
        prefix_bonds=parts["prefix"],
        hidden_operator=hidden,
        crossing_operator=crossing,
        prefix_operator=prefix,
        full_operator=full,
    )


def column_local_energies(
    matrix: object, amplitudes: Sequence[complex] | np.ndarray
) -> np.ndarray:
    """Return ``sum_new H[new,old] psi(new)/psi(old)`` for every old state."""

    psi = np.asarray(amplitudes, dtype=np.complex128)
    csc = matrix.tocsc()
    if csc.shape != (len(psi), len(psi)):
        raise ValueError("operator and amplitude dimensions must agree")
    if np.any(~np.isfinite(psi)) or np.any(np.abs(psi) == 0.0):
        raise ValueError("amplitudes must be finite and nonzero")
    return np.asarray(csc.T @ psi, dtype=np.complex128) / psi


@dataclass(frozen=True)
class BoundarySectorSuffixOracles:
    """Leakage-marked exact martingale increments for the cut sectors."""

    hidden: ConditionalSuffixOracle
    crossing: ConditionalSuffixOracle
    prefix: ConditionalSuffixOracle
    full: ConditionalSuffixOracle
    maximum_local_energy_reconstruction_error: float
    maximum_increment_reconstruction_error: float
    uses_hidden_suffix: bool = True

    @property
    def noncrossing_centered_messages(self) -> np.ndarray:
        return self.hidden.centered_messages + self.prefix.centered_messages


def boundary_sector_suffix_oracles(
    sectors: BoundarySectorOperators,
    amplitudes: Sequence[complex] | np.ndarray,
    action_sequences: Sequence[Sequence[int]] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> BoundarySectorSuffixOracles:
    """Contract all hidden suffixes for each physical sector at its cut."""

    psi = np.asarray(amplitudes, dtype=np.complex128)
    actions = np.asarray(action_sequences)
    if actions.ndim != 2 or len(actions) != len(psi):
        raise ValueError("actions must have one row per amplitude")
    decision = sectors.decision
    local_hidden = column_local_energies(sectors.hidden_operator, psi)
    local_crossing = column_local_energies(sectors.crossing_operator, psi)
    local_prefix = column_local_energies(sectors.prefix_operator, psi)
    local_full = column_local_energies(sectors.full_operator, psi)

    def oracle(values: np.ndarray) -> ConditionalSuffixOracle:
        return conditional_suffix_oracle(
            values[:, None], actions, probabilities, decision
        )

    hidden = oracle(local_hidden)
    crossing = oracle(local_crossing)
    prefix = oracle(local_prefix)
    full = oracle(local_full)
    local_error = float(
        np.max(
            np.abs(local_hidden + local_crossing + local_prefix - local_full),
            initial=0.0,
        )
    )
    increment_error = float(
        np.max(
            np.abs(
                hidden.centered_messages
                + crossing.centered_messages
                + prefix.centered_messages
                - full.centered_messages
            ),
            initial=0.0,
        )
    )
    return BoundarySectorSuffixOracles(
        hidden=hidden,
        crossing=crossing,
        prefix=prefix,
        full=full,
        maximum_local_energy_reconstruction_error=local_error,
        maximum_increment_reconstruction_error=increment_error,
    )


def channel_increment_diagnostics(
    target_increment: Sequence[complex] | np.ndarray,
    noncrossing_increment: Sequence[complex] | np.ndarray,
    crossing_channel_increments: np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> dict[str, object]:
    """Measure ordered vector-channel corrections to the scalar sector.

    Channels are assumed to be in descending boundary-coupling singular-value
    order.  The fixed physical coefficient is one; no regression is performed.
    Marginal contributions can be negative because vector channels interfere.
    """

    target = np.asarray(target_increment, dtype=np.complex128)
    scalar = np.asarray(noncrossing_increment, dtype=np.complex128)
    channels = np.asarray(crossing_channel_increments, dtype=np.complex128)
    weights = np.asarray(probabilities, dtype=np.float64)
    if target.ndim != 1 or scalar.shape != target.shape:
        raise ValueError("target and noncrossing increments must be vectors")
    if channels.ndim != 2 or channels.shape[0] != len(target):
        raise ValueError("crossing channels must have shape (states, channels)")
    if weights.shape != target.shape or np.any(weights < 0.0):
        raise ValueError("probabilities must align with target")
    weights = weights / np.sum(weights)

    def norm(values: np.ndarray) -> float:
        return float(np.sum(weights * np.abs(values) ** 2))

    target_norm = norm(target)
    approximation = scalar.copy()
    previous_error = norm(target - approximation)
    rows = []
    for channel in range(channels.shape[1]):
        value = channels[:, channel]
        approximation += value
        error = norm(target - approximation)
        rows.append(
            {
                "channel": channel,
                "channel_norm": norm(value),
                "cumulative_error": error,
                "cumulative_explained_fraction": (
                    1.0 - error / target_norm if target_norm else 1.0
                ),
                "marginal_explained_fraction": (
                    (previous_error - error) / target_norm if target_norm else 0.0
                ),
            }
        )
        previous_error = error
    return {
        "target_norm": target_norm,
        "scalar_only_error": norm(target - scalar),
        "scalar_only_explained_fraction": (
            1.0 - norm(target - scalar) / target_norm if target_norm else 1.0
        ),
        "channels": rows,
        "final_reconstruction_error": previous_error,
        "maximum_pointwise_reconstruction_error": float(
            np.max(np.abs(target - approximation), initial=0.0)
        ),
    }
