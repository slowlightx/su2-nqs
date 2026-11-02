"""Coordinate-free physical-channel ledger for fusion-space local energy.

This exact, small-system benchmark resolves every Heisenberg-bond matrix
element by its physical shell, its life-cycle at a reverse left-associated
autoregressive cut, and the doubled boundary-spin change carried across that
cut.  It introduces no SVD gauge and no learned coordinate system.

Matrices follow the project convention ``H[new_path, old_path]``.  At every
cut their channel-resolved local-energy contributions sum to

``E_loc(old) = sum_new H[new, old] * psi(new) / psi(old)``.

The conditional expectations of those channels give a finer ledger for the
ordinary Doob martingale.  A channel is assigned the signed covariance
``Re E[conj(Delta_channel) Delta_total]``; consequently the channel values at
one cut add exactly to that cut's Doob-increment variance, including negative
interference allocations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence

import numpy as np

from ..su2.fusion_path import enumerate_paths, is_valid_path
from ..su2.fusion_tree import FusionTree
from .causal_residual import compact_tree_action_sequences


CARRIER_DELTA_Q: tuple[int, int, int] = (-2, 0, 2)
PHYSICAL_EVENT_NAMES: tuple[str, str, str, str, str] = (
    "prebirth",
    "birth",
    "transport_open",
    "death",
    "transport_scalar",
)
PREBIRTH_EVENT = 0
BIRTH_EVENT = 1
TRANSPORT_OPEN_EVENT = 2
DEATH_EVENT = 3
TRANSPORT_SCALAR_EVENT = 4


@dataclass(frozen=True)
class PhysicalBondTerm:
    """One coupled bond matrix and its coordinate-free graph metadata.

    ``matrix`` may be a dense square array or a SciPy-like sparse object with
    ``tocsc``.  It must already include the desired coupling and must use
    ``matrix[new, old]`` orientation.  ``first``, ``second``, and ``shell``
    are metadata; all three remain independent of a fusion-tree gauge.
    """

    first: int
    second: int
    matrix: object
    shell: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        for value, label in ((self.first, "first"), (self.second, "second")):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"bond {label} endpoint must be an integer")
        if int(self.first) == int(self.second):
            raise ValueError("a physical bond must join distinct sites")
        if isinstance(self.shell, bool) or not isinstance(self.shell, Integral):
            raise TypeError("bond shell must be an integer")
        if int(self.shell) < 0:
            raise ValueError("bond shell must be non-negative")
        if not isinstance(self.name, str):
            raise TypeError("bond name must be a string")


@dataclass(frozen=True)
class ChannelMartingaleAllocation:
    """Doob increments and signed variance allocations for channel values.

    Channel arrays use axes ``[state, cut, ...physical channels]``.  The
    observable reconstructed at different cuts may differ by a state-
    independent constant; this is the explicitly supported constant gauge.
    """

    normalized_probabilities: np.ndarray
    conditional_channel_messages: np.ndarray
    channel_increments: np.ndarray
    total_messages: np.ndarray
    total_increments: np.ndarray
    signed_variance_allocation: np.ndarray
    cut_variances: np.ndarray
    mean_observable: complex
    observable_variance: float
    explained_variance: float
    maximum_centered_observable_discrepancy: float
    maximum_increment_reconstruction_error: float
    maximum_cut_variance_additivity_error: float
    maximum_increment_orthogonality_error: float
    martingale_variance_error: float


@dataclass(frozen=True)
class PhysicalCarrierMartingale:
    """Exact physical bond-shell/event/``delta_q`` local-energy ledger.

    The channel axes are ``[state, cut, shell, event, delta_q]``.  Per-bond
    event codes are retained for audit.  ``bond_signed_variance_allocation``
    has axes ``[bond, cut, event, delta_q]`` and is evaluated with a streamed
    second pass: no state-by-cut-by-bond tensor is retained.  Thus the spatial
    bond map is exact while the N=16 reference remains modest in memory.
    """

    shell_ids: np.ndarray
    event_names: tuple[str, ...]
    delta_q: tuple[int, ...]
    bond_first: np.ndarray
    bond_second: np.ndarray
    bond_shell_slots: np.ndarray
    bond_names: tuple[str, ...]
    bond_birth_decisions: np.ndarray
    bond_death_decisions: np.ndarray
    bond_event_codes: np.ndarray
    channel_local_energies: np.ndarray
    local_energies: np.ndarray
    normalized_probabilities: np.ndarray
    conditional_channel_messages: np.ndarray
    channel_increments: np.ndarray
    total_messages: np.ndarray
    total_increments: np.ndarray
    signed_variance_allocation: np.ndarray
    bond_signed_variance_allocation: np.ndarray
    cut_variances: np.ndarray
    mean_local_energy: complex
    local_energy_variance: float
    explained_variance: float
    maximum_forbidden_matrix_element: float
    maximum_local_energy_reconstruction_error: float
    maximum_increment_reconstruction_error: float
    maximum_cut_variance_additivity_error: float
    maximum_increment_orthogonality_error: float
    martingale_variance_error: float
    maximum_bond_to_shell_allocation_error: float
    maximum_bond_to_cut_allocation_error: float


def _validate_site(n_sites: int, site: int) -> int:
    if isinstance(site, bool) or not isinstance(site, Integral):
        raise TypeError("bond endpoints must be integer site indices")
    concrete = int(site)
    if not 0 <= concrete < n_sites:
        raise ValueError("a bond endpoint lies outside the system")
    return concrete


def _reveal_decision(n_sites: int, site: int) -> int:
    """Compact decision which reveals ``site`` in the reverse sweep."""

    # The last compact action contains two exact microsteps, revealing sites
    # 1 and 0 in that order.
    return min(n_sites - 2, n_sites - 1 - site)


def classify_reverse_left_bond(
    n_sites: int,
    first: int,
    second: int,
    decision: int,
) -> str:
    """Classify one bond after a reverse left-associated AR decision.

    A bond is hidden (``prebirth``), is sourced when its rightmost endpoint is
    first exposed (``birth``), remains a rank-one open carrier, dies when its
    second endpoint is exposed, and is thereafter transported as a remembered
    scalar.  A bond whose endpoints are both exposed in the final compact
    decision is classified as ``death`` at that cut.  More generally the
    final action's site-1 transport and site-0 closure are represented by its
    completed ``death`` channel.  For a singlet this action is deterministic,
    so its Doob increment and every signed variance allocation are exactly
    zero; no variance is assigned twice to its internal microsteps.
    """

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be an integer")
    n_sites = int(n_sites)
    if n_sites < 2:
        raise ValueError("n_sites must be at least two")
    if isinstance(decision, bool) or not isinstance(decision, Integral):
        raise TypeError("decision must be an integer")
    decision = int(decision)
    if not 0 <= decision < n_sites - 1:
        raise ValueError("decision lies outside the reverse sweep")
    left = _validate_site(n_sites, first)
    right = _validate_site(n_sites, second)
    if left == right:
        raise ValueError("a physical bond must join distinct sites")
    low, high = sorted((left, right))
    birth = _reveal_decision(n_sites, high)
    death = _reveal_decision(n_sites, low)
    if decision < birth:
        return PHYSICAL_EVENT_NAMES[PREBIRTH_EVENT]
    if decision == death:
        return PHYSICAL_EVENT_NAMES[DEATH_EVENT]
    if decision == birth:
        return PHYSICAL_EVENT_NAMES[BIRTH_EVENT]
    if decision < death:
        return PHYSICAL_EVENT_NAMES[TRANSPORT_OPEN_EVENT]
    return PHYSICAL_EVENT_NAMES[TRANSPORT_SCALAR_EVENT]


def _normalize_probabilities(probabilities: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (size,):
        raise ValueError("born_probabilities must have one entry per basis state")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("born_probabilities must be finite and non-negative")
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("born_probabilities must have positive finite total")
    return values / total


def _validate_actions(action_sequences: np.ndarray, size: int) -> np.ndarray:
    actions = np.asarray(action_sequences)
    if actions.ndim != 2 or len(actions) != size:
        raise ValueError("action_sequences must have one row per basis state")
    if not np.issubdtype(actions.dtype, np.integer):
        raise TypeError("compact-tree actions must be integers")
    if actions.shape[1] < 1:
        raise ValueError("at least one autoregressive decision is required")
    keys = {tuple(int(action) for action in row) for row in actions}
    if len(keys) != size:
        raise ValueError("complete action sequences must identify basis states uniquely")
    return np.asarray(actions, dtype=np.int32)


def _conditional_mean(
    values: np.ndarray,
    actions: np.ndarray,
    probabilities: np.ndarray,
    prefix_length: int,
) -> np.ndarray:
    """Born conditional mean for arbitrary trailing value axes."""

    flattened = np.asarray(values, dtype=np.complex128).reshape(len(values), -1)
    totals: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
    for row, value, probability in zip(
        actions, flattened, probabilities, strict=True
    ):
        key = tuple(int(action) for action in row[:prefix_length])
        if key not in totals:
            totals[key] = (
                0.0,
                np.zeros(flattened.shape[1], dtype=np.complex128),
            )
        mass, numerator = totals[key]
        totals[key] = (mass + float(probability), numerator + probability * value)
    means = {
        key: numerator / mass
        for key, (mass, numerator) in totals.items()
        if mass > 0.0
    }
    result = np.empty_like(flattened)
    for index, row in enumerate(actions):
        key = tuple(int(action) for action in row[:prefix_length])
        if key not in means:
            raise ValueError("an action prefix has zero Born mass")
        result[index] = means[key]
    return result.reshape(values.shape)


def allocate_channel_martingale(
    born_probabilities: Sequence[float] | np.ndarray,
    channel_local_energies: np.ndarray,
    action_sequences: np.ndarray,
    *,
    consistency_tolerance: float = 2.0e-11,
) -> ChannelMartingaleAllocation:
    """Allocate exact Doob variance to arbitrary physical local-energy channels.

    ``channel_local_energies`` has shape ``(state, cut, ...)``.  Summing its
    trailing axes must reconstruct the same centered observable at every cut.
    State-independent, cut- and channel-dependent offsets are allowed; this
    is the constant channel gauge, and it leaves every increment and variance
    allocation invariant.
    """

    values = np.asarray(channel_local_energies, dtype=np.complex128)
    if values.ndim < 3:
        raise ValueError(
            "channel_local_energies must have axes (state, cut, channels...)"
        )
    actions = _validate_actions(action_sequences, values.shape[0])
    if values.shape[1] != actions.shape[1]:
        raise ValueError("channel cut count must equal the action-sequence length")
    if not math.isfinite(consistency_tolerance) or consistency_tolerance < 0.0:
        raise ValueError("consistency_tolerance must be finite and non-negative")
    probabilities = _normalize_probabilities(
        np.asarray(born_probabilities), values.shape[0]
    )
    active = probabilities > 0.0
    if np.any(~np.isfinite(values[active])):
        raise ValueError("channel values must be finite on positive-Born states")
    values = np.where(active.reshape((-1,) + (1,) * (values.ndim - 1)), values, 0.0j)

    channel_axes = tuple(range(2, values.ndim))
    totals_by_cut = np.sum(values, axis=channel_axes)
    means_by_cut = probabilities @ totals_by_cut
    centered_by_cut = totals_by_cut - means_by_cut[None, :]
    reference = totals_by_cut[:, -1]
    centered_reference = reference - probabilities @ reference
    centered_discrepancy = float(
        np.max(np.abs(centered_by_cut - centered_reference[:, None]), initial=0.0)
    )
    if centered_discrepancy > consistency_tolerance:
        raise ValueError(
            "channel sums do not represent one observable up to a constant gauge: "
            f"centered discrepancy {centered_discrepancy:.3e}"
        )

    cuts = actions.shape[1]
    messages = np.empty_like(values)
    increments = np.empty_like(values)
    total_messages = np.empty((len(values), cuts + 1), dtype=np.complex128)
    for prefix_length in range(cuts + 1):
        total_messages[:, prefix_length] = _conditional_mean(
            reference, actions, probabilities, prefix_length
        )
    total_increments = total_messages[:, 1:] - total_messages[:, :-1]

    for decision in range(cuts):
        current = _conditional_mean(
            values[:, decision], actions, probabilities, decision + 1
        )
        previous = _conditional_mean(
            values[:, decision], actions, probabilities, decision
        )
        messages[:, decision] = current
        increments[:, decision] = current - previous

    reconstructed_increments = np.sum(increments, axis=channel_axes)
    increment_error = float(
        np.max(
            np.abs(reconstructed_increments - total_increments),
            initial=0.0,
        )
    )
    weighted_total = probabilities[:, None] * total_increments
    allocation = np.real(
        np.sum(
            np.conjugate(increments)
            * weighted_total.reshape(
                (len(values), cuts) + (1,) * (increments.ndim - 2)
            ),
            axis=0,
        )
    )
    cut_variances = np.sum(
        probabilities[:, None] * np.abs(total_increments) ** 2, axis=0
    ).real
    allocation_sums = np.sum(
        allocation, axis=tuple(range(1, allocation.ndim))
    )
    cut_additivity_error = float(
        np.max(np.abs(allocation_sums - cut_variances), initial=0.0)
    )

    gram = np.einsum(
        "s,sd,se->de",
        probabilities,
        np.conjugate(total_increments),
        total_increments,
    )
    off_diagonal = gram - np.diag(np.diag(gram))
    orthogonality_error = float(np.max(np.abs(off_diagonal), initial=0.0))
    variance = float(np.sum(probabilities * np.abs(centered_reference) ** 2))
    explained = float(np.sum(cut_variances))
    return ChannelMartingaleAllocation(
        normalized_probabilities=probabilities,
        conditional_channel_messages=messages,
        channel_increments=increments,
        total_messages=total_messages,
        total_increments=total_increments,
        signed_variance_allocation=np.asarray(allocation, dtype=np.float64),
        cut_variances=np.asarray(cut_variances, dtype=np.float64),
        mean_observable=complex(probabilities @ reference),
        observable_variance=variance,
        explained_variance=explained,
        maximum_centered_observable_discrepancy=centered_discrepancy,
        maximum_increment_reconstruction_error=increment_error,
        maximum_cut_variance_additivity_error=cut_additivity_error,
        maximum_increment_orthogonality_error=orthogonality_error,
        martingale_variance_error=float(abs(variance - explained)),
    )


def _full_left_paths(
    basis_paths_or_labels: np.ndarray,
    n_sites: int,
    target_q: int,
) -> np.ndarray:
    basis = np.asarray(basis_paths_or_labels)
    if basis.ndim != 2:
        raise ValueError("basis paths/labels must be a rank-two array")
    if not np.issubdtype(basis.dtype, np.integer):
        raise TypeError("fusion labels must be doubled integers")
    if basis.shape[1] == n_sites - 1:
        full = np.concatenate(
            (np.ones((len(basis), 1), dtype=np.int32), basis.astype(np.int32)),
            axis=1,
        )
    elif basis.shape[1] == n_sites:
        full = np.asarray(basis, dtype=np.int32)
    else:
        raise ValueError(
            f"expected {n_sites - 1} internal labels or length-{n_sites} paths"
        )
    rows = tuple(tuple(int(q) for q in row) for row in full)
    if any(not is_valid_path(row, target_q=target_q) for row in rows):
        raise ValueError("basis contains an invalid spin-half fusion path")
    if len(set(rows)) != len(rows):
        raise ValueError("basis fusion paths must be unique")
    return full


def _matrix_columns(matrix: object, dimension: int):
    """Yield ``(old, new_indices, values)`` without requiring SciPy."""

    if hasattr(matrix, "tocsc"):
        csc = matrix.tocsc(copy=True)
        if csc.shape != (dimension, dimension):
            raise ValueError("a bond matrix shape does not match the fusion basis")
        if hasattr(csc, "sum_duplicates"):
            csc.sum_duplicates()
        if np.any(~np.isfinite(csc.data)):
            raise ValueError("bond matrix elements must be finite")
        for old in range(dimension):
            start, stop = csc.indptr[old : old + 2]
            yield old, np.asarray(csc.indices[start:stop]), np.asarray(
                csc.data[start:stop], dtype=np.complex128
            )
        return
    dense = np.asarray(matrix, dtype=np.complex128)
    if dense.shape != (dimension, dimension):
        raise ValueError("a bond matrix shape does not match the fusion basis")
    if np.any(~np.isfinite(dense)):
        raise ValueError("bond matrix elements must be finite")
    for old in range(dimension):
        new = np.flatnonzero(dense[:, old] != 0.0)
        yield old, new, dense[new, old]


def _matrix_hermiticity_error(matrix: object, dimension: int) -> tuple[float, float]:
    """Return maximum anti-Hermitian entry and matrix scale without densifying."""

    if hasattr(matrix, "tocsc"):
        concrete = matrix.tocsc(copy=True)
        if concrete.shape != (dimension, dimension):
            raise ValueError("a bond matrix shape does not match the fusion basis")
        if hasattr(concrete, "sum_duplicates"):
            concrete.sum_duplicates()
        if np.any(~np.isfinite(concrete.data)):
            raise ValueError("bond matrix elements must be finite")
        difference = concrete - concrete.getH()
        return (
            float(np.max(np.abs(difference.data), initial=0.0)),
            float(np.max(np.abs(concrete.data), initial=0.0)),
        )
    concrete = np.asarray(matrix, dtype=np.complex128)
    if concrete.shape != (dimension, dimension):
        raise ValueError("a bond matrix shape does not match the fusion basis")
    if np.any(~np.isfinite(concrete)):
        raise ValueError("bond matrix elements must be finite")
    return (
        float(np.max(np.abs(concrete - concrete.conj().T), initial=0.0)),
        float(np.max(np.abs(concrete), initial=0.0)),
    )


def _event_delta(
    full_paths: np.ndarray,
    new: int,
    old: int,
    decision: int,
    event: int,
    low_site: int,
    high_site: int,
) -> int:
    if event in (PREBIRTH_EVENT, TRANSPORT_SCALAR_EVENT):
        return 0
    if event in (BIRTH_EVENT, TRANSPORT_OPEN_EVENT):
        hidden_count = full_paths.shape[1] - decision - 1
        return int(
            full_paths[new, hidden_count - 1]
            - full_paths[old, hidden_count - 1]
        )
    if event == DEATH_EVENT:
        # Sites one and zero are the two microsteps of the final compact
        # decision.  For bond (0, 1) its incoming one-spin boundary has fixed
        # doubled q=1 in bra and ket.  A bond (0, k>1), however, has already
        # carried its right endpoint to that final decision; its incoming
        # *two-spin* boundary is path[1] and can change by +/-2.  Collapsing all
        # low_site==0 deaths into delta_q=0 loses this physical carrier label.
        if low_site == 0:
            if high_site == 1:
                return 0
            return int(full_paths[new, 1] - full_paths[old, 1])
        return int(full_paths[new, low_site] - full_paths[old, low_site])
    raise RuntimeError("unknown physical carrier event")


def _single_bond_channel_values(
    full_paths: np.ndarray,
    psi: np.ndarray,
    matrix: object,
    bond_events: np.ndarray,
    low_site: int,
    high_site: int,
    delta_position: dict[int, int],
    tolerance: float,
    first: int,
    second: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return one bond's cut/event/``delta_q`` values without a bond axis."""

    dimension = len(full_paths)
    cuts = bond_events.shape[0]
    values = np.zeros(
        (dimension, cuts, len(PHYSICAL_EVENT_NAMES), len(CARRIER_DELTA_Q)),
        dtype=np.complex128,
    )
    direct_local = np.zeros((dimension,), dtype=np.complex128)
    maximum_forbidden = 0.0
    for old, new_indices, matrix_values in _matrix_columns(matrix, dimension):
        if len(new_indices) == 0:
            continue
        ratios = psi[new_indices] / psi[old]
        direct_local[old] += np.sum(matrix_values * ratios)
        for decision in range(cuts):
            event = int(bond_events[decision])
            for new, matrix_value, ratio in zip(
                new_indices, matrix_values, ratios, strict=True
            ):
                delta = _event_delta(
                    full_paths,
                    int(new),
                    old,
                    decision,
                    event,
                    low_site,
                    high_site,
                )
                if delta not in delta_position:
                    maximum_forbidden = max(
                        maximum_forbidden, float(abs(matrix_value))
                    )
                    if abs(matrix_value) > tolerance:
                        raise ValueError(
                            "a bond matrix has a forbidden carrier transition: "
                            f"bond=({first},{second}), decision={decision}, "
                            f"event={PHYSICAL_EVENT_NAMES[event]}, delta_q={delta}, "
                            f"|H[new,old]|={abs(matrix_value):.3e}"
                        )
                    continue
                values[
                    old,
                    decision,
                    event,
                    delta_position[delta],
                ] += matrix_value * ratio
    return values, direct_local, maximum_forbidden


def _stream_bond_signed_variance_allocation(
    full_paths: np.ndarray,
    psi: np.ndarray,
    terms: tuple[PhysicalBondTerm, ...],
    bond_events: np.ndarray,
    probabilities: np.ndarray,
    actions: np.ndarray,
    total_increments: np.ndarray,
    delta_position: dict[int, int],
    tolerance: float,
) -> np.ndarray:
    """Stream exact ``Re E[conj(Delta_b) Delta_total]`` bond allocations."""

    cuts = actions.shape[1]
    answer = np.empty(
        (len(terms), cuts, len(PHYSICAL_EVENT_NAMES), len(CARRIER_DELTA_Q)),
        dtype=np.float64,
    )
    weighted_total = probabilities[:, None] * total_increments
    for bond_index, term in enumerate(terms):
        first = int(term.first)
        second = int(term.second)
        low, high = sorted((first, second))
        values, _, _ = _single_bond_channel_values(
            full_paths,
            psi,
            term.matrix,
            bond_events[:, bond_index],
            low,
            high,
            delta_position,
            tolerance,
            first,
            second,
        )
        increments = np.empty_like(values)
        for decision in range(cuts):
            current = _conditional_mean(
                values[:, decision], actions, probabilities, decision + 1
            )
            previous = _conditional_mean(
                values[:, decision], actions, probabilities, decision
            )
            increments[:, decision] = current - previous
        answer[bond_index] = np.real(
            np.sum(
                np.conjugate(increments)
                * weighted_total[:, :, None, None],
                axis=0,
            )
        )
    return answer


def physical_carrier_martingale(
    basis_paths_or_labels: np.ndarray,
    born_probabilities: Sequence[float] | np.ndarray,
    wavefunction: Sequence[complex] | np.ndarray,
    bond_terms: Sequence[PhysicalBondTerm],
    action_sequences: np.ndarray,
    *,
    target_q: int = 0,
    tolerance: float = 2.0e-12,
    born_tolerance: float = 2.0e-11,
) -> PhysicalCarrierMartingale:
    """Build the exact shell/event/``delta_q`` local-energy martingale.

    This routine is intended as an exhaustive reference, not a sampled
    production kernel.  It stores shell-aggregated state-dependent values and
    streams one bond at a time for the final signed covariance map;
    ``bond_event_codes`` independently audits each bond life cycle without a
    prohibitive state-by-cut-by-bond tensor.
    """

    raw_basis = np.asarray(basis_paths_or_labels)
    if raw_basis.ndim != 2:
        raise ValueError("basis paths/labels must be a rank-two array")
    actions = _validate_actions(action_sequences, raw_basis.shape[0])
    n_sites = actions.shape[1] + 1
    if n_sites % 2:
        raise ValueError("the reverse singlet carrier requires an even site count")
    if isinstance(target_q, bool) or not isinstance(target_q, Integral):
        raise TypeError("target_q must be a doubled-integer angular momentum")
    if int(target_q) != 0:
        raise ValueError("the reverse left-associated carrier currently requires target_q=0")
    full_paths = _full_left_paths(
        raw_basis, n_sites, int(target_q)
    )
    dimension = len(full_paths)
    canonical_paths = enumerate_paths(n_sites, int(target_q))
    if tuple(tuple(int(q) for q in row) for row in full_paths) != canonical_paths:
        raise ValueError(
            "basis paths must be the complete canonical left-associated basis"
        )
    canonical_actions = compact_tree_action_sequences(
        FusionTree.left_associated(n_sites),
        int(target_q),
        full_paths[:, 1:],
    )
    if not np.array_equal(actions, canonical_actions):
        raise ValueError(
            "action_sequences must match the supplied paths' canonical "
            "reverse left-associated compact actions"
        )
    terms = tuple(bond_terms)
    if not terms:
        raise ValueError("at least one physical bond term is required")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    if not math.isfinite(born_tolerance) or born_tolerance < 0.0:
        raise ValueError("born_tolerance must be finite and non-negative")

    psi = np.asarray(wavefunction, dtype=np.complex128)
    if psi.shape != (dimension,):
        raise ValueError("wavefunction must have one amplitude per basis state")
    if np.any(~np.isfinite(psi)):
        raise ValueError("wavefunction amplitudes must be finite")
    if np.any(np.abs(psi) == 0.0):
        raise ValueError("local energy requires nonzero wavefunction amplitudes")
    probabilities = _normalize_probabilities(
        np.asarray(born_probabilities), dimension
    )
    expected_born = np.abs(psi) ** 2
    expected_norm = float(np.sum(expected_born))
    if not math.isfinite(expected_norm) or expected_norm <= 0.0:
        raise ValueError("wavefunction has non-finite or zero squared norm")
    expected_born /= expected_norm
    born_error = float(np.max(np.abs(probabilities - expected_born), initial=0.0))
    if born_error > born_tolerance:
        raise ValueError(
            "born_probabilities disagree with normalized |wavefunction|^2: "
            f"maximum discrepancy {born_error:.3e}"
        )

    if any(not isinstance(term, PhysicalBondTerm) for term in terms):
        raise TypeError("bond_terms must contain PhysicalBondTerm instances")
    shell_ids = np.asarray(
        sorted({int(term.shell) for term in terms}), dtype=np.int32
    )
    shell_position = {int(shell): slot for slot, shell in enumerate(shell_ids)}
    cuts = n_sites - 1
    event_position = {name: index for index, name in enumerate(PHYSICAL_EVENT_NAMES)}
    delta_position = {delta: index for index, delta in enumerate(CARRIER_DELTA_Q)}
    channel_values = np.zeros(
        (
            dimension,
            cuts,
            len(shell_ids),
            len(PHYSICAL_EVENT_NAMES),
            len(CARRIER_DELTA_Q),
        ),
        dtype=np.complex128,
    )
    direct_local = np.zeros((dimension,), dtype=np.complex128)
    bond_events = np.empty((cuts, len(terms)), dtype=np.int8)
    birth_decisions = np.empty((len(terms),), dtype=np.int32)
    death_decisions = np.empty((len(terms),), dtype=np.int32)
    maximum_forbidden = 0.0

    for bond_index, term in enumerate(terms):
        first = _validate_site(n_sites, term.first)
        second = _validate_site(n_sites, term.second)
        if first == second:
            raise ValueError("a physical bond must join distinct sites")
        low, high = sorted((first, second))
        hermiticity_error, matrix_scale = _matrix_hermiticity_error(
            term.matrix, dimension
        )
        if hermiticity_error > tolerance * max(1.0, matrix_scale):
            raise ValueError(
                "a per-bond Heisenberg matrix must be Hermitian: maximum "
                f"discrepancy {hermiticity_error:.3e}"
            )
        birth_decisions[bond_index] = _reveal_decision(n_sites, high)
        death_decisions[bond_index] = _reveal_decision(n_sites, low)
        for decision in range(cuts):
            event_name = classify_reverse_left_bond(
                n_sites, first, second, decision
            )
            bond_events[decision, bond_index] = event_position[event_name]

        shell_slot = shell_position[int(term.shell)]
        bond_values, bond_local, bond_forbidden = _single_bond_channel_values(
            full_paths,
            psi,
            term.matrix,
            bond_events[:, bond_index],
            low,
            high,
            delta_position,
            tolerance,
            first,
            second,
        )
        channel_values[:, :, shell_slot] += bond_values
        direct_local += bond_local
        maximum_forbidden = max(maximum_forbidden, bond_forbidden)

    reconstructed = np.sum(channel_values, axis=(2, 3, 4))
    reconstruction_error = float(
        np.max(np.abs(reconstructed - direct_local[:, None]), initial=0.0)
    )
    reconstruction_tolerance = tolerance * max(
        1.0, float(np.max(np.abs(direct_local), initial=0.0))
    )
    if reconstruction_error > reconstruction_tolerance:
        raise ValueError(
            "forbidden carrier transitions prevent exact local-energy "
            f"reconstruction: error {reconstruction_error:.3e} exceeds "
            f"the fixed tolerance {reconstruction_tolerance:.3e}"
        )
    allocation = allocate_channel_martingale(
        probabilities,
        channel_values,
        actions,
        consistency_tolerance=reconstruction_tolerance,
    )
    bond_allocation = _stream_bond_signed_variance_allocation(
        full_paths,
        psi,
        terms,
        bond_events,
        probabilities,
        actions,
        allocation.total_increments,
        delta_position,
        tolerance,
    )
    bond_to_shell = np.zeros_like(allocation.signed_variance_allocation)
    bond_shell_slots = np.asarray(
        [shell_position[int(term.shell)] for term in terms], dtype=np.int32
    )
    for bond_index, shell_slot in enumerate(bond_shell_slots):
        bond_to_shell[:, shell_slot] += bond_allocation[bond_index]
    bond_to_shell_error = float(
        np.max(
            np.abs(bond_to_shell - allocation.signed_variance_allocation),
            initial=0.0,
        )
    )
    bond_to_cut_error = float(
        np.max(
            np.abs(
                np.sum(bond_allocation, axis=(0, 2, 3))
                - allocation.cut_variances
            ),
            initial=0.0,
        )
    )
    return PhysicalCarrierMartingale(
        shell_ids=shell_ids,
        event_names=PHYSICAL_EVENT_NAMES,
        delta_q=CARRIER_DELTA_Q,
        bond_first=np.asarray([int(term.first) for term in terms], dtype=np.int32),
        bond_second=np.asarray([int(term.second) for term in terms], dtype=np.int32),
        bond_shell_slots=bond_shell_slots,
        bond_names=tuple(
            term.name or f"bond_{int(term.first)}_{int(term.second)}"
            for term in terms
        ),
        bond_birth_decisions=birth_decisions,
        bond_death_decisions=death_decisions,
        bond_event_codes=bond_events,
        channel_local_energies=channel_values,
        local_energies=direct_local,
        normalized_probabilities=allocation.normalized_probabilities,
        conditional_channel_messages=allocation.conditional_channel_messages,
        channel_increments=allocation.channel_increments,
        total_messages=allocation.total_messages,
        total_increments=allocation.total_increments,
        signed_variance_allocation=allocation.signed_variance_allocation,
        bond_signed_variance_allocation=bond_allocation,
        cut_variances=allocation.cut_variances,
        mean_local_energy=allocation.mean_observable,
        local_energy_variance=allocation.observable_variance,
        explained_variance=allocation.explained_variance,
        maximum_forbidden_matrix_element=maximum_forbidden,
        maximum_local_energy_reconstruction_error=reconstruction_error,
        maximum_increment_reconstruction_error=(
            allocation.maximum_increment_reconstruction_error
        ),
        maximum_cut_variance_additivity_error=(
            allocation.maximum_cut_variance_additivity_error
        ),
        maximum_increment_orthogonality_error=(
            allocation.maximum_increment_orthogonality_error
        ),
        martingale_variance_error=allocation.martingale_variance_error,
        maximum_bond_to_shell_allocation_error=bond_to_shell_error,
        maximum_bond_to_cut_allocation_error=bond_to_cut_error,
    )


__all__ = [
    "BIRTH_EVENT",
    "CARRIER_DELTA_Q",
    "ChannelMartingaleAllocation",
    "DEATH_EVENT",
    "PHYSICAL_EVENT_NAMES",
    "PREBIRTH_EVENT",
    "PhysicalBondTerm",
    "PhysicalCarrierMartingale",
    "TRANSPORT_OPEN_EVENT",
    "TRANSPORT_SCALAR_EVENT",
    "allocate_channel_martingale",
    "classify_reverse_left_bond",
    "physical_carrier_martingale",
]
