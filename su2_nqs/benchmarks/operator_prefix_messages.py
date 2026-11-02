"""Operator-aligned targets along an autoregressive fusion-tree filtration.

The local energy of a sampled old path ``x`` is assembled with the project
connectivity convention

``E_loc(x) = sum_y H[y, x] * psi(y) / psi(x)``.

Conditional expectations of this quantity over autoregressive prefixes form
the exact Doob martingale of the Hamiltonian residual.  These targets are a
small-system reference for causal operator-message models: unlike a causal
cone count, they retain matrix elements, wavefunction ratios, and interference
signs.  They deliberately do not prescribe a neural encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ConnectedOperatorColumns:
    """Padded Hamiltonian columns for sampled fusion paths.

    ``matrix_elements[s, k]`` is exactly
    ``H[connected_paths[s, k], source_paths[s]]`` and ``psi_ratios[s, k]`` is
    ``psi(connected_paths[s, k]) / psi(source_paths[s])``.  All fusion labels
    are doubled integer angular momenta.
    """

    source_paths: np.ndarray
    connected_paths: np.ndarray
    matrix_elements: np.ndarray
    psi_ratios: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        source = np.asarray(self.source_paths)
        connected = np.asarray(self.connected_paths)
        elements = np.asarray(self.matrix_elements)
        ratios = np.asarray(self.psi_ratios)
        valid = np.asarray(self.valid)
        if source.ndim != 2:
            raise ValueError("source_paths must have shape (samples, labels)")
        expected_paths = (
            (source.shape[0], elements.shape[1], source.shape[1])
            if elements.ndim == 2
            else None
        )
        if connected.ndim != 3 or connected.shape != expected_paths:
            raise ValueError(
                "connected_paths must have shape (samples, connections, labels)"
            )
        if (
            elements.ndim != 2
            or ratios.shape != elements.shape
            or valid.shape != elements.shape
        ):
            raise ValueError(
                "matrix_elements, psi_ratios, and valid must share shape "
                "(samples, connections)"
            )
        if not np.issubdtype(source.dtype, np.integer) or not np.issubdtype(
            connected.dtype, np.integer
        ):
            raise TypeError("fusion paths must contain doubled integer labels")
        if np.any(source < 0) or np.any(connected[valid] < 0):
            raise ValueError("doubled fusion labels must be non-negative")
        if np.any(~np.isfinite(elements[valid])) or np.any(
            ~np.isfinite(ratios[valid])
        ):
            raise ValueError("valid matrix elements and psi ratios must be finite")

    @property
    def n_samples(self) -> int:
        return int(np.asarray(self.source_paths).shape[0])

    @property
    def maximum_connections(self) -> int:
        return int(np.asarray(self.valid).shape[1])


@dataclass(frozen=True)
class PrefixOperatorMessages:
    """Exact prefix messages and martingale increments for sampled columns."""

    local_energies: np.ndarray
    prefix_messages: np.ndarray
    increments: np.ndarray
    normalized_weights: np.ndarray

    @property
    def residuals(self) -> np.ndarray:
        """Centered local energies represented by the complete filtration."""

        return self.local_energies - self.prefix_messages[:, 0]


@dataclass(frozen=True)
class PrefixOperatorTerm:
    """One numeric operator term used by a causal-prefix diagnostic.

    ``matrix`` follows the project convention ``matrix[new, old]`` and must
    provide ``tocsc``.  The optional channel is metadata for physically shared
    shells (for example J1/J2); it does not affect the exact prefix compiler.
    """

    name: str
    matrix: object
    coupling: float = 1.0
    channel: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an operator term needs a non-empty name")
        if not math.isfinite(self.coupling):
            raise ValueError("operator coupling must be finite")
        if isinstance(self.channel, bool) or not isinstance(self.channel, int):
            raise TypeError("operator channel must be an integer")
        if self.channel < 0:
            raise ValueError("operator channel must be non-negative")


@dataclass(frozen=True)
class PrefixOperatorCausality:
    """Exact suffix-independence audit of one operator's prefix columns."""

    causal_at_decision: np.ndarray
    persistently_causal: np.ndarray
    first_available_decision: int
    maximum_coefficient_discrepancy: np.ndarray
    mismatched_signature_groups: np.ndarray


@dataclass(frozen=True)
class CausalRecouplingMessages:
    """Prefix-amplitude operator messages for several numeric terms.

    ``values[x, d, b]`` is zero before term ``b`` becomes suffix-independent.
    Thereafter it is the exact prefix-only quantity

    ``sum_y H_b[y,x] * exp(L_d(y) - L_d(x))``,

    where ``L_d`` is the cumulative autoregressive log amplitude through
    decision ``d``.  The compiler verifies that the aggregated prefix column
    of ``H_b`` is identical for every full path sharing the source prefix.
    """

    values: np.ndarray
    first_available_decisions: np.ndarray
    causality: tuple[PrefixOperatorCausality, ...]
    maximum_message_invariance_error: np.ndarray
    names: tuple[str, ...]
    channels: np.ndarray
    couplings: np.ndarray


def pack_connected_operator_columns(
    source_paths: Sequence[Sequence[int]] | np.ndarray,
    connected_paths: Sequence[Sequence[Sequence[int]]],
    matrix_elements: Sequence[Sequence[complex]],
    psi_ratios: Sequence[Sequence[complex]],
) -> ConnectedOperatorColumns:
    """Pack ragged ``H[new,old]`` columns into fixed-shape numeric arrays."""

    sources = np.asarray(source_paths)
    if sources.ndim != 2:
        raise ValueError("source_paths must have shape (samples, labels)")
    if not (
        len(connected_paths)
        == len(matrix_elements)
        == len(psi_ratios)
        == len(sources)
    ):
        raise ValueError("one connected column is required for every source path")
    counts = tuple(len(column) for column in connected_paths)
    if any(
        len(elements) != count or len(ratios) != count
        for count, elements, ratios in zip(
            counts, matrix_elements, psi_ratios, strict=True
        )
    ):
        raise ValueError("connected paths, elements, and ratios must align per column")
    width = max(counts, default=0)
    paths = np.zeros((len(sources), width, sources.shape[1]), dtype=np.int32)
    elements_array = np.zeros((len(sources), width), dtype=np.complex128)
    ratios_array = np.zeros((len(sources), width), dtype=np.complex128)
    valid = np.zeros((len(sources), width), dtype=bool)
    for sample, (column, elements, ratios) in enumerate(
        zip(connected_paths, matrix_elements, psi_ratios, strict=True)
    ):
        count = len(column)
        if count == 0:
            continue
        concrete = np.asarray(column)
        if concrete.shape != (count, sources.shape[1]):
            raise ValueError("a connected path has the wrong label count")
        paths[sample, :count] = concrete
        elements_array[sample, :count] = np.asarray(elements)
        ratios_array[sample, :count] = np.asarray(ratios)
        valid[sample, :count] = True
    return ConnectedOperatorColumns(
        np.asarray(sources, dtype=np.int32), paths, elements_array, ratios_array, valid
    )


def connected_operator_columns_from_sparse(
    matrix: object,
    basis_paths: Sequence[Sequence[int]],
    amplitudes: Sequence[complex] | np.ndarray,
    *,
    source_indices: Sequence[int] | np.ndarray | None = None,
) -> ConnectedOperatorColumns:
    """Extract sparse ``H[new,old]`` columns and exact wavefunction ratios.

    The sparse object must provide ``tocsc``.  ``source_indices`` may include
    repeats, as it would for a Born sample.  Sampled source amplitudes must be
    nonzero because their local-energy ratios would otherwise be undefined.
    """

    csc = matrix.tocsc()
    basis = np.asarray(basis_paths)
    psi = np.asarray(amplitudes, dtype=np.complex128)
    if basis.ndim != 2 or not np.issubdtype(basis.dtype, np.integer):
        raise TypeError("basis_paths must be a rectangular doubled-integer array")
    dimension = len(basis)
    if csc.shape != (dimension, dimension) or psi.shape != (dimension,):
        raise ValueError("operator, basis, and amplitude dimensions must agree")
    if source_indices is None:
        indices = np.arange(dimension, dtype=np.int64)
    else:
        indices = np.asarray(source_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("source_indices must be a one-dimensional integer array")
        indices = indices.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= dimension):
            raise ValueError("source index lies outside the operator basis")
    if np.any(np.abs(psi[indices]) == 0.0):
        raise ValueError("sampled source amplitudes must be nonzero")

    connected: list[np.ndarray] = []
    elements: list[np.ndarray] = []
    ratios: list[np.ndarray] = []
    for old in indices:
        start, stop = csc.indptr[old : old + 2]
        new = csc.indices[start:stop]
        connected.append(basis[new])
        elements.append(np.asarray(csc.data[start:stop], dtype=np.complex128))
        ratios.append(psi[new] / psi[old])
    return pack_connected_operator_columns(
        basis[indices], connected, elements, ratios
    )


def local_energy_contributions(columns: ConnectedOperatorColumns) -> np.ndarray:
    """Return every padded ``H[new,old] * psi(new) / psi(old)`` contribution."""

    valid = np.asarray(columns.valid)
    return np.where(
        valid,
        np.asarray(columns.matrix_elements) * np.asarray(columns.psi_ratios),
        0.0j,
    )


def connected_local_energies(columns: ConnectedOperatorColumns) -> np.ndarray:
    """Assemble exact local energies from operator columns."""

    return np.sum(local_energy_contributions(columns), axis=1)


def _prefix_key(row: np.ndarray, length: int) -> tuple[int, ...]:
    return tuple(int(value) for value in row[:length])


def _column_prefix_signature(
    csc: object,
    column: int,
    actions: np.ndarray,
    length: int,
    *,
    zero_tolerance: float,
) -> dict[tuple[int, ...], complex]:
    start, stop = csc.indptr[column : column + 2]
    signature: dict[tuple[int, ...], complex] = {}
    for row, element in zip(
        csc.indices[start:stop], csc.data[start:stop], strict=True
    ):
        key = _prefix_key(actions[int(row)], length)
        signature[key] = signature.get(key, 0.0j) + complex(element)
    return {
        key: value
        for key, value in signature.items()
        if abs(value) > zero_tolerance
    }


def operator_prefix_causality(
    matrix: object,
    action_sequences: Sequence[Sequence[int]] | np.ndarray,
    *,
    coefficient_tolerance: float = 2.0e-12,
    zero_tolerance: float = 1.0e-14,
) -> PrefixOperatorCausality:
    """Find when ``matrix[new, old]`` becomes an exact prefix operator.

    At prefix length ``d+1``, every matrix column is aggregated by the target
    action prefix.  The operator is causal at ``d`` only when two old paths
    with the same source prefix have identical aggregated target-prefix
    columns.  This is a direct no-future-label test: it neither assumes that a
    geometric leaf path contains all required spectator spins nor averages
    over the hidden suffix.

    Availability is deliberately stricter than a one-level coincidence.  A
    term first becomes available at the earliest decision for which it is
    causal at that and *every later* decision.  The complete decision always
    passes because it identifies a full basis state.
    """

    actions = np.asarray(action_sequences)
    if actions.ndim != 2 or not np.issubdtype(actions.dtype, np.integer):
        raise TypeError("action_sequences must be a two-dimensional integer array")
    if coefficient_tolerance < 0.0 or zero_tolerance < 0.0:
        raise ValueError("causality tolerances must be non-negative")
    csc = matrix.tocsc()
    if csc.shape != (len(actions), len(actions)):
        raise ValueError("operator and action basis dimensions must agree")

    decisions = actions.shape[1]
    causal = np.ones(decisions, dtype=bool)
    discrepancies = np.zeros(decisions, dtype=np.float64)
    mismatches = np.zeros(decisions, dtype=np.int32)
    for decision in range(decisions):
        length = decision + 1
        references: dict[
            tuple[int, ...], dict[tuple[int, ...], complex]
        ] = {}
        for old, action_row in enumerate(actions):
            source_prefix = _prefix_key(action_row, length)
            signature = _column_prefix_signature(
                csc,
                old,
                actions,
                length,
                zero_tolerance=zero_tolerance,
            )
            reference = references.get(source_prefix)
            if reference is None:
                references[source_prefix] = signature
                continue
            if reference.keys() != signature.keys():
                causal[decision] = False
                mismatches[decision] += 1
                discrepancies[decision] = np.inf
                continue
            if not signature:
                continue
            difference = max(
                abs(signature[key] - reference[key]) for key in signature
            )
            discrepancies[decision] = max(discrepancies[decision], difference)
            if difference > coefficient_tolerance:
                causal[decision] = False
                mismatches[decision] += 1

    persistent = np.logical_and.accumulate(causal[::-1])[::-1]
    available = np.flatnonzero(persistent)
    if len(available) == 0:  # Defensive: complete prefixes must identify columns.
        raise RuntimeError("operator never became prefix causal")
    return PrefixOperatorCausality(
        causal,
        persistent,
        int(available[0]),
        discrepancies,
        mismatches,
    )


def maximum_prefix_invariance_error(
    values: Sequence[complex] | np.ndarray,
    action_sequences: Sequence[Sequence[int]] | np.ndarray,
    decision: int,
) -> float:
    """Maximum spread of a scalar over paths sharing a decision prefix."""

    concrete = np.asarray(values)
    actions = np.asarray(action_sequences)
    if concrete.shape != (len(actions),):
        raise ValueError("values must have one entry per action sequence")
    if not 0 <= decision < actions.shape[1]:
        raise ValueError("decision lies outside the action sequence")
    representatives: dict[tuple[int, ...], complex] = {}
    maximum = 0.0
    for value, row in zip(concrete, actions, strict=True):
        key = _prefix_key(row, decision + 1)
        if key in representatives:
            maximum = max(maximum, abs(complex(value) - representatives[key]))
        else:
            representatives[key] = complex(value)
    return float(maximum)


def causal_prefix_recoupling_messages(
    terms: Sequence[PrefixOperatorTerm],
    action_sequences: Sequence[Sequence[int]] | np.ndarray,
    cumulative_prefix_logs: np.ndarray,
    *,
    coefficient_tolerance: float = 2.0e-12,
    invariance_tolerance: float = 2.0e-10,
) -> CausalRecouplingMessages:
    """Compile exact prefix-available local-recoupling features.

    The full paths are used only to exhaustively *certify* suffix independence.
    A returned feature is a function of the source prefix, target prefixes,
    numeric ``H[new, old]`` elements, and the model's cumulative prefix logs.
    It is never exposed before that certification succeeds.  This provides a
    small-system reference for a future local F-move compiler without silently
    treating a geometric cone as a complete recoupling cone.
    """

    concrete_terms = tuple(terms)
    if not concrete_terms:
        raise ValueError("at least one operator term is required")
    actions = np.asarray(action_sequences)
    logs = np.asarray(cumulative_prefix_logs, dtype=np.complex128)
    if actions.ndim != 2 or not np.issubdtype(actions.dtype, np.integer):
        raise TypeError("action_sequences must be a two-dimensional integer array")
    if logs.shape != actions.shape:
        raise ValueError("cumulative_prefix_logs must match action_sequences")
    if np.any(~np.isfinite(logs)):
        raise ValueError("prefix logs must be finite on the exhaustive basis")

    n_states, n_decisions = actions.shape
    values = np.zeros(
        (n_states, n_decisions, len(concrete_terms)), dtype=np.complex128
    )
    errors = np.zeros((n_decisions, len(concrete_terms)), dtype=np.float64)
    audits: list[PrefixOperatorCausality] = []
    first = np.empty(len(concrete_terms), dtype=np.int32)
    for term_index, term in enumerate(concrete_terms):
        csc = term.matrix.tocsc()
        if csc.shape != (n_states, n_states):
            raise ValueError(f"operator term {term.name!r} has the wrong shape")
        audit = operator_prefix_causality(
            csc,
            actions,
            coefficient_tolerance=coefficient_tolerance,
        )
        audits.append(audit)
        first[term_index] = audit.first_available_decision
        columns = np.repeat(
            np.arange(n_states, dtype=np.int64), np.diff(csc.indptr)
        )
        rows = np.asarray(csc.indices, dtype=np.int64)
        elements = np.asarray(csc.data, dtype=np.complex128)
        for decision in range(audit.first_available_decision, n_decisions):
            contributions = elements * np.exp(
                logs[rows, decision] - logs[columns, decision]
            )
            message = np.zeros(n_states, dtype=np.complex128)
            np.add.at(message, columns, contributions)
            error = maximum_prefix_invariance_error(message, actions, decision)
            errors[decision, term_index] = error
            if error > invariance_tolerance:
                raise RuntimeError(
                    f"term {term.name!r} passed the coefficient audit but its "
                    f"decision-{decision} message varies within a prefix by {error:.3e}"
                )
            values[:, decision, term_index] = message

    return CausalRecouplingMessages(
        values=values,
        first_available_decisions=first,
        causality=tuple(audits),
        maximum_message_invariance_error=errors,
        names=tuple(term.name for term in concrete_terms),
        channels=np.asarray([term.channel for term in concrete_terms], dtype=np.int32),
        couplings=np.asarray(
            [term.coupling for term in concrete_terms], dtype=np.float64
        ),
    )


def _normalized_weights(
    sample_weights: Sequence[float] | np.ndarray, size: int
) -> np.ndarray:
    weights = np.asarray(sample_weights, dtype=np.float64)
    if weights.shape != (size,):
        raise ValueError("sample_weights must have one entry per source path")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("sample weights must be finite and non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("sample weights must have positive total weight")
    return weights / total


def prefix_operator_messages(
    columns: ConnectedOperatorColumns,
    action_sequences: Sequence[Sequence[int]] | np.ndarray,
    sample_weights: Sequence[float] | np.ndarray,
) -> PrefixOperatorMessages:
    """Construct weighted local-energy messages for every AR prefix.

    Column zero of ``prefix_messages`` is the global energy.  Column ``d+1``
    is ``E[E_loc | actions[:d+1]]``.  Consequently ``increments[:, d]`` is
    the exact operator-aligned Doob target exposed by decision ``d``.  If the
    complete action sequence identifies each sampled state, the increments
    sum exactly to ``E_loc - E``.
    """

    actions = np.asarray(action_sequences)
    if actions.ndim != 2 or actions.shape[0] != columns.n_samples:
        raise ValueError("action_sequences must have one row per source path")
    if not np.issubdtype(actions.dtype, np.integer):
        raise TypeError("autoregressive actions must be integers")
    weights = _normalized_weights(sample_weights, columns.n_samples)
    local = connected_local_energies(columns)
    messages = np.empty((columns.n_samples, actions.shape[1] + 1), dtype=np.complex128)
    messages[:, 0] = np.sum(weights * local)
    for length in range(1, actions.shape[1] + 1):
        totals: dict[tuple[int, ...], list[complex | float]] = {}
        for weight, value, row in zip(weights, local, actions, strict=True):
            key = tuple(int(action) for action in row[:length])
            if key not in totals:
                totals[key] = [0.0, 0.0j]
            totals[key][0] += float(weight)
            totals[key][1] += weight * value
        means = {
            key: numerator / probability
            for key, (probability, numerator) in totals.items()
            if probability > 0.0
        }
        messages[:, length] = np.asarray(
            [means[tuple(int(action) for action in row[:length])] for row in actions]
        )
    return PrefixOperatorMessages(local, messages, np.diff(messages, axis=1), weights)


def operator_message_diagnostics(
    messages: PrefixOperatorMessages,
) -> dict[str, object]:
    """Measure residual reconstruction, variance explanation, and target scale."""

    weights = np.asarray(messages.normalized_weights)
    local = np.asarray(messages.local_energies)
    increments = np.asarray(messages.increments)
    centered = local - np.sum(weights * local)
    variance = float(np.sum(weights * np.abs(centered) ** 2))
    increment_variance = np.sum(weights[:, None] * np.abs(increments) ** 2, axis=0)
    reconstructed = np.sum(increments, axis=1)
    error = reconstructed - centered
    explained = float(np.sum(increment_variance))
    cross_error = float(abs(np.sum(weights * np.abs(reconstructed) ** 2) - explained))
    rows = []
    for decision in range(increments.shape[1]):
        squared = np.abs(increments[:, decision]) ** 2
        contribution = float(increment_variance[decision])
        second = float(np.sum(weights * squared**2))
        rows.append(
            {
                "decision": decision,
                "rms_magnitude": float(np.sqrt(contribution)),
                "maximum_magnitude": float(np.max(np.abs(increments[:, decision]))),
                "variance_contribution": contribution,
                "variance_fraction": contribution / variance if variance > 0.0 else 0.0,
                "sampling_ess_fraction": (
                    contribution**2 / second if second > 0.0 else 1.0
                ),
            }
        )
    return {
        "energy_real": float(np.real(np.sum(weights * local))),
        "energy_imag": float(np.imag(np.sum(weights * local))),
        "local_energy_variance": variance,
        "explained_variance": explained,
        "explained_variance_fraction": explained / variance if variance > 0.0 else 1.0,
        "maximum_reconstruction_error": float(np.max(np.abs(error), initial=0.0)),
        "weighted_reconstruction_error": float(np.sum(weights * np.abs(error) ** 2)),
        "increment_orthogonality_error": cross_error,
        "maximum_local_energy_magnitude": float(np.max(np.abs(local), initial=0.0)),
        "decisions": rows,
    }
