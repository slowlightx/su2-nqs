"""Sparse stable-endpoint compiler for causal SU(2) boundary recurrences.

This module compiles graph-local birth, transport, and death events for the
root-first sweep of a left-associated singlet tree.  It deliberately stores
neither revealed fusion paths nor prefix-by-prefix operators.  The only
angular-momentum data are fixed-size local Condon--Shortley tables for adding
one spin-half to a revealed block.

For an active hidden endpoint ``i`` the exact graph object is

``B_i^(1) = sum_{j in revealed} J_ij S_j``.

On revealing ``s``, persistent/new rows receive ``J_is S_s`` and the old row
``i=s`` is contracted as ``S_s.B_s`` into the closed scalar sector.  Learned
models may compress the multiplicity dependence of these objects, but the
event topology and every local Wigner--Eckart coefficient below are exact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable

import numpy as np

from ..su2.reduced_vector import (
    VECTOR_DELTA_Q,
    appended_spin_scalar_contraction_coefficient,
    appended_spin_vector_source_coefficient,
    old_block_vector_advance_coefficient,
)
from .boundary_operator_channels import BoundaryBond, normalize_boundary_bonds


MAX_REVEALS_PER_DECISION = 2


def _reachable_singlet_cut(q: int, block_sites: int, hidden_sites: int) -> bool:
    """Whether a spin-half block irrep can be completed to a total singlet."""

    return (
        0 <= q <= block_sites
        and (block_sites - q) % 2 == 0
        and q <= hidden_sites
        and (hidden_sites - q) % 2 == 0
    )


def _active_endpoints(
    hidden: set[int],
    revealed: set[int],
    coupling: np.ndarray,
    tolerance: float,
) -> tuple[int, ...]:
    return tuple(
        site
        for site in sorted(hidden)
        if any(abs(float(coupling[site, other])) > tolerance for other in revealed)
    )


def _sector_summaries(
    hidden: set[int],
    revealed: set[int],
    bond_first: np.ndarray,
    bond_second: np.ndarray,
    bond_coupling: np.ndarray,
    bond_shell_slot: np.ndarray,
    shell_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    sums = np.zeros((3, shell_count), dtype=np.float64)
    squared = np.zeros_like(sums)
    counts = np.zeros((3, shell_count), dtype=np.int32)
    for first, second, value, shell in zip(
        bond_first,
        bond_second,
        bond_coupling,
        bond_shell_slot,
        strict=True,
    ):
        if int(first) in hidden and int(second) in hidden:
            sector = 0
        elif int(first) in revealed and int(second) in revealed:
            sector = 2
        else:
            sector = 1
        sums[sector, int(shell)] += float(value)
        squared[sector, int(shell)] += float(value) ** 2
        counts[sector, int(shell)] += 1
    return (
        sums[0],
        sums[1],
        sums[2],
        squared[0],
        squared[1],
        squared[2],
        counts[0],
        counts[1],
        counts[2],
    )


@dataclass(frozen=True)
class _RawEvent:
    decision: int
    substep: int
    revealed_site: int
    old_hidden: tuple[int, ...]
    old_revealed: tuple[int, ...]
    new_hidden: tuple[int, ...]
    new_revealed: tuple[int, ...]
    old_endpoints: tuple[int, ...]
    new_endpoints: tuple[int, ...]
    closing_terms: tuple[tuple[int, float, int], ...]


@dataclass(frozen=True)
class SparseCausalRecurrencePlan:
    """Pure-numeric payload for a stable-endpoint causal recurrence.

    The leading axes of all event arrays are ``[decision, substep]``.  Every
    decision has one substep except the last, which reveals physical sites 1
    and 0 in that order.  Endpoint slots are deterministically sorted by
    physical site.  ``persist_old_slot_for_new`` is ``-1`` for births.

    Local tensor arrays use doubled labels and the orientation ``[new, old]``:

    * ``old_block_transport[..., delta_new, delta_old]`` transports an
      existing rank-one reduced message;
    * ``endpoint_source[..., delta_new]`` multiplies ``J_is`` for the newly
      appended physical spin;
    * ``dying_scalar[..., delta_old]`` contracts the dying ``B_s`` with the
      appended ``S_s`` into a scalar.

    No array dimension is a path count.  Compilation cost is polynomial in
    ``n_sites`` and graph size.
    """

    n_sites: int
    target_q: int
    delta_q: tuple[int, int, int]
    maximum_active_endpoints: int
    endpoint_slot_count: int
    maximum_closing_terms: int
    closing_term_slot_count: int
    shell_ids: np.ndarray
    bond_first: np.ndarray
    bond_second: np.ndarray
    bond_coupling: np.ndarray
    bond_shell_slot: np.ndarray
    event_valid: np.ndarray
    decision_substep_count: np.ndarray
    revealed_site: np.ndarray
    revealed_count_before: np.ndarray
    revealed_count_after: np.ndarray
    old_endpoint_sites: np.ndarray
    old_endpoint_mask: np.ndarray
    new_endpoint_sites: np.ndarray
    new_endpoint_mask: np.ndarray
    persist_old_slot_for_new: np.ndarray
    birth_mask: np.ndarray
    source_coupling: np.ndarray
    dying_old_slot: np.ndarray
    dying_endpoint_site: np.ndarray
    death_mask: np.ndarray
    closing_partner_sites: np.ndarray
    closing_couplings: np.ndarray
    closing_shell_slot: np.ndarray
    closing_mask: np.ndarray
    q_transition_valid: np.ndarray
    old_block_transport: np.ndarray
    endpoint_source: np.ndarray
    dying_scalar: np.ndarray
    initial_event: np.ndarray
    hidden_shell_sum_before: np.ndarray
    crossing_shell_sum_before: np.ndarray
    closed_shell_sum_before: np.ndarray
    hidden_shell_sum_after: np.ndarray
    crossing_shell_sum_after: np.ndarray
    closed_shell_sum_after: np.ndarray
    hidden_shell_squared_before: np.ndarray
    crossing_shell_squared_before: np.ndarray
    closed_shell_squared_before: np.ndarray
    hidden_shell_squared_after: np.ndarray
    crossing_shell_squared_after: np.ndarray
    closed_shell_squared_after: np.ndarray
    hidden_shell_count_before: np.ndarray
    crossing_shell_count_before: np.ndarray
    closed_shell_count_before: np.ndarray
    hidden_shell_count_after: np.ndarray
    crossing_shell_count_after: np.ndarray
    closed_shell_count_after: np.ndarray
    newly_closed_shell_sum: np.ndarray
    uses_hidden_suffix: bool = False
    enumerates_revealed_prefixes: bool = False

    @property
    def n_decisions(self) -> int:
        return self.n_sites - 1

    @property
    def q_size(self) -> int:
        return self.n_sites + 1

    @property
    def shell_count(self) -> int:
        return int(len(self.shell_ids))

    @property
    def event_count(self) -> int:
        return int(np.sum(self.event_valid))

    def numeric_payload(self, *, physical_scale: int = 1) -> dict[str, np.ndarray]:
        """Return fixed-shape arrays suitable for ``np.savez`` or JAX.

        ``physical_scale=0`` is the preregistered parameter-matched K=0
        control: graph topology, site identities, masks, and tensor shapes are
        retained while all graph values and local physical coefficients are
        zeroed.  Only the binary values 0 and 1 are accepted so an accidental
        double rescaling of ``J_is`` and its source coefficient is impossible.
        """

        if isinstance(physical_scale, bool) or physical_scale not in (0, 1):
            raise ValueError("physical_scale must be the integer 0 or 1")
        payload = {
            "n_sites": np.asarray(self.n_sites, dtype=np.int32),
            "target_q": np.asarray(self.target_q, dtype=np.int32),
            "delta_q": np.asarray(self.delta_q, dtype=np.int32),
            "maximum_active_endpoints": np.asarray(
                self.maximum_active_endpoints, dtype=np.int32
            ),
            "endpoint_slot_count": np.asarray(self.endpoint_slot_count, dtype=np.int32),
            "maximum_closing_terms": np.asarray(
                self.maximum_closing_terms, dtype=np.int32
            ),
            "shell_ids": np.asarray(self.shell_ids, dtype=np.int32),
        }
        unscaled_names = (
            "bond_first",
            "bond_second",
            "bond_shell_slot",
            "event_valid",
            "decision_substep_count",
            "revealed_site",
            "revealed_count_before",
            "revealed_count_after",
            "old_endpoint_sites",
            "old_endpoint_mask",
            "new_endpoint_sites",
            "new_endpoint_mask",
            "persist_old_slot_for_new",
            "birth_mask",
            "dying_old_slot",
            "dying_endpoint_site",
            "death_mask",
            "closing_partner_sites",
            "closing_shell_slot",
            "closing_mask",
            "q_transition_valid",
            "initial_event",
            "hidden_shell_count_before",
            "crossing_shell_count_before",
            "closed_shell_count_before",
            "hidden_shell_count_after",
            "crossing_shell_count_after",
            "closed_shell_count_after",
        )
        for name in unscaled_names:
            payload[name] = np.asarray(getattr(self, name))
        scaled_names = (
            "bond_coupling",
            "source_coupling",
            "closing_couplings",
            "old_block_transport",
            "endpoint_source",
            "dying_scalar",
            "hidden_shell_sum_before",
            "crossing_shell_sum_before",
            "closed_shell_sum_before",
            "hidden_shell_sum_after",
            "crossing_shell_sum_after",
            "closed_shell_sum_after",
            "hidden_shell_squared_before",
            "crossing_shell_squared_before",
            "closed_shell_squared_before",
            "hidden_shell_squared_after",
            "crossing_shell_squared_after",
            "closed_shell_squared_after",
            "newly_closed_shell_sum",
        )
        for name in scaled_names:
            value = np.asarray(getattr(self, name))
            payload[name] = value if physical_scale else np.zeros_like(value)
        return payload

    def payload_nbytes(self, *, physical_scale: int = 1) -> int:
        """Total bytes of the plain numeric payload (without JAX copies)."""

        return int(
            sum(value.nbytes for value in self.numeric_payload(
                physical_scale=physical_scale
            ).values())
        )

    def reconstruct_endpoint_coefficients(
        self, decision: int, substep: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Causally reconstruct endpoint rows after one requested event.

        This graph-only diagnostic advances the stored ``J_is`` sources and
        never enumerates a fusion path.  Columns follow physical reveal order.
        """

        if not 0 <= decision < self.n_decisions:
            raise ValueError("decision lies outside the recurrence")
        if not 0 <= substep < MAX_REVEALS_PER_DECISION or not self.event_valid[
            decision, substep
        ]:
            raise ValueError("substep is not active for this decision")
        rows: dict[int, dict[int, float]] = {}
        revealed: list[int] = []
        for d in range(self.n_decisions):
            for micro in range(MAX_REVEALS_PER_DECISION):
                if not self.event_valid[d, micro]:
                    continue
                site = int(self.revealed_site[d, micro])
                old_rows = rows
                rows = {}
                for slot in np.flatnonzero(self.new_endpoint_mask[d, micro]):
                    endpoint = int(self.new_endpoint_sites[d, micro, slot])
                    old_slot = int(self.persist_old_slot_for_new[d, micro, slot])
                    previous = (
                        dict(old_rows[int(self.old_endpoint_sites[d, micro, old_slot])])
                        if old_slot >= 0
                        else {}
                    )
                    coupling = float(self.source_coupling[d, micro, slot])
                    if coupling:
                        previous[site] = coupling
                    rows[endpoint] = previous
                revealed.append(site)
                if d == decision and micro == substep:
                    endpoints = np.asarray(sorted(rows), dtype=np.int32)
                    prefix = np.asarray(revealed, dtype=np.int32)
                    matrix = np.zeros(
                        (len(endpoints), len(prefix)), dtype=np.float64
                    )
                    for row, endpoint in enumerate(endpoints):
                        for column, prefix_site in enumerate(prefix):
                            matrix[row, column] = rows[int(endpoint)].get(
                                int(prefix_site), 0.0
                            )
                    return endpoints, prefix, matrix
        raise RuntimeError("requested recurrence event was not reached")


def _aggregate_bonds(
    n_sites: int,
    bonds: Iterable[
        BoundaryBond
        | tuple[int, int]
        | tuple[int, int, float]
        | tuple[int, int, float, int]
    ],
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalized = normalize_boundary_bonds(bonds)
    for bond in normalized:
        if not 0 <= bond.first < n_sites or not 0 <= bond.second < n_sites:
            raise ValueError("a bond endpoint lies outside the system")
    shell_ids = np.asarray(
        sorted({bond.channel for bond in normalized}), dtype=np.int32
    )
    shell_position = {int(shell): slot for slot, shell in enumerate(shell_ids)}
    accumulated: dict[tuple[int, int, int], float] = {}
    for bond in normalized:
        first, second = sorted((bond.first, bond.second))
        key = (first, second, shell_position[bond.channel])
        accumulated[key] = accumulated.get(key, 0.0) + float(bond.coupling)
    retained = sorted(
        (first, second, shell, value)
        for (first, second, shell), value in accumulated.items()
        if abs(value) > tolerance
    )
    first = np.asarray([item[0] for item in retained], dtype=np.int32)
    second = np.asarray([item[1] for item in retained], dtype=np.int32)
    shell = np.asarray([item[2] for item in retained], dtype=np.int32)
    values = np.asarray([item[3] for item in retained], dtype=np.float64)
    combined = np.zeros((n_sites, n_sites), dtype=np.float64)
    for left, right, value in zip(first, second, values, strict=True):
        combined[int(left), int(right)] += float(value)
        combined[int(right), int(left)] += float(value)
    return shell_ids, first, second, values, shell, combined


def compile_sparse_causal_recurrence_plan(
    n_sites: int,
    bonds: Iterable[
        BoundaryBond
        | tuple[int, int]
        | tuple[int, int, float]
        | tuple[int, int, float, int]
    ],
    *,
    target_q: int = 0,
    coefficient_tolerance: float = 1.0e-14,
) -> SparseCausalRecurrencePlan:
    """Compile stable-endpoint events and local SU(2) coefficient tables.

    The current causal reverse-block construction is defined for an even
    spin-half singlet.  Compilation is wavefunction-free and uses
    ``O(n_sites**3 + n_sites * |E|)`` scalar work for the padded local tables;
    it never calls ``enumerate_paths`` or creates a prefix operator.
    """

    if isinstance(n_sites, bool) or not isinstance(n_sites, int):
        raise TypeError("n_sites must be an integer")
    if n_sites < 2 or n_sites % 2:
        raise ValueError("an even number of at least two spin-half sites is required")
    if target_q != 0:
        raise ValueError("the current sparse reverse recurrence requires target_q=0")
    if (
        isinstance(coefficient_tolerance, bool)
        or not isinstance(coefficient_tolerance, Real)
        or not math.isfinite(float(coefficient_tolerance))
        or coefficient_tolerance < 0.0
    ):
        raise ValueError("coefficient_tolerance must be finite and non-negative")
    tolerance = float(coefficient_tolerance)
    (
        shell_ids,
        bond_first,
        bond_second,
        bond_coupling,
        bond_shell_slot,
        combined_coupling,
    ) = _aggregate_bonds(n_sites, bonds, tolerance)

    shell_terms: dict[tuple[int, int], list[tuple[float, int]]] = {}
    for first, second, value, shell in zip(
        bond_first,
        bond_second,
        bond_coupling,
        bond_shell_slot,
        strict=True,
    ):
        shell_terms.setdefault((int(first), int(second)), []).append(
            (float(value), int(shell))
        )

    raw_events: list[_RawEvent] = []
    hidden = set(range(n_sites))
    revealed: set[int] = set()
    for decision in range(n_sites - 1):
        sites = (n_sites - 1 - decision,) if decision < n_sites - 2 else (1, 0)
        for substep, site in enumerate(sites):
            old_hidden = set(hidden)
            old_revealed = set(revealed)
            old_endpoints = _active_endpoints(
                old_hidden, old_revealed, combined_coupling, tolerance
            )
            hidden.remove(site)
            revealed.add(site)
            new_endpoints = _active_endpoints(
                hidden, revealed, combined_coupling, tolerance
            )
            closing = []
            for partner in sorted(old_revealed):
                pair = tuple(sorted((site, partner)))
                for value, shell in shell_terms.get(pair, ()):
                    closing.append((partner, value, shell))
            raw_events.append(
                _RawEvent(
                    decision=decision,
                    substep=substep,
                    revealed_site=site,
                    old_hidden=tuple(sorted(old_hidden)),
                    old_revealed=tuple(sorted(old_revealed)),
                    new_hidden=tuple(sorted(hidden)),
                    new_revealed=tuple(sorted(revealed)),
                    old_endpoints=old_endpoints,
                    new_endpoints=new_endpoints,
                    closing_terms=tuple(closing),
                )
            )

    n_decisions = n_sites - 1
    event_shape = (n_decisions, MAX_REVEALS_PER_DECISION)
    maximum_active = max(
        (
            max(len(event.old_endpoints), len(event.new_endpoints))
            for event in raw_events
        ),
        default=0,
    )
    endpoint_slots = max(1, maximum_active)
    maximum_closing = max((len(event.closing_terms) for event in raw_events), default=0)
    closing_slots = max(1, maximum_closing)
    shell_count = len(shell_ids)
    q_size = n_sites + 1
    delta_count = len(VECTOR_DELTA_Q)

    event_valid = np.zeros(event_shape, dtype=bool)
    decision_substep_count = np.zeros((n_decisions,), dtype=np.int32)
    revealed_site = np.full(event_shape, -1, dtype=np.int32)
    revealed_count_before = np.zeros(event_shape, dtype=np.int32)
    revealed_count_after = np.zeros(event_shape, dtype=np.int32)
    endpoint_shape = event_shape + (endpoint_slots,)
    old_endpoint_sites = np.full(endpoint_shape, -1, dtype=np.int32)
    old_endpoint_mask = np.zeros(endpoint_shape, dtype=bool)
    new_endpoint_sites = np.full(endpoint_shape, -1, dtype=np.int32)
    new_endpoint_mask = np.zeros(endpoint_shape, dtype=bool)
    persist_old_slot_for_new = np.full(endpoint_shape, -1, dtype=np.int32)
    birth_mask = np.zeros(endpoint_shape, dtype=bool)
    source_coupling = np.zeros(endpoint_shape, dtype=np.float64)
    dying_old_slot = np.full(event_shape, -1, dtype=np.int32)
    dying_endpoint_site = np.full(event_shape, -1, dtype=np.int32)
    death_mask = np.zeros(event_shape, dtype=bool)
    closing_shape = event_shape + (closing_slots,)
    closing_partner_sites = np.full(closing_shape, -1, dtype=np.int32)
    closing_couplings = np.zeros(closing_shape, dtype=np.float64)
    closing_shell_slot = np.full(closing_shape, -1, dtype=np.int32)
    closing_mask = np.zeros(closing_shape, dtype=bool)
    q_transition_valid = np.zeros(event_shape + (q_size, q_size), dtype=bool)
    old_block_transport = np.zeros(
        event_shape + (q_size, q_size, delta_count, delta_count),
        dtype=np.float64,
    )
    endpoint_source = np.zeros(
        event_shape + (q_size, q_size, delta_count), dtype=np.float64
    )
    dying_scalar = np.zeros_like(endpoint_source)
    initial_event = np.zeros(event_shape, dtype=bool)

    summary_shape = event_shape + (shell_count,)
    summary_names = (
        "hidden_sum_before",
        "crossing_sum_before",
        "closed_sum_before",
        "hidden_sum_after",
        "crossing_sum_after",
        "closed_sum_after",
        "hidden_squared_before",
        "crossing_squared_before",
        "closed_squared_before",
        "hidden_squared_after",
        "crossing_squared_after",
        "closed_squared_after",
    )
    summaries = {
        name: np.zeros(summary_shape, dtype=np.float64) for name in summary_names
    }
    count_names = (
        "hidden_count_before",
        "crossing_count_before",
        "closed_count_before",
        "hidden_count_after",
        "crossing_count_after",
        "closed_count_after",
    )
    counts = {
        name: np.zeros(summary_shape, dtype=np.int32) for name in count_names
    }
    newly_closed_shell_sum = np.zeros(summary_shape, dtype=np.float64)

    for event_index, event in enumerate(raw_events):
        d, micro = event.decision, event.substep
        event_valid[d, micro] = True
        decision_substep_count[d] += 1
        revealed_site[d, micro] = event.revealed_site
        revealed_count_before[d, micro] = len(event.old_revealed)
        revealed_count_after[d, micro] = len(event.new_revealed)
        old_position = {site: slot for slot, site in enumerate(event.old_endpoints)}
        for slot, site in enumerate(event.old_endpoints):
            old_endpoint_sites[d, micro, slot] = site
            old_endpoint_mask[d, micro, slot] = True
        for slot, site in enumerate(event.new_endpoints):
            new_endpoint_sites[d, micro, slot] = site
            new_endpoint_mask[d, micro, slot] = True
            previous_slot = old_position.get(site, -1)
            persist_old_slot_for_new[d, micro, slot] = previous_slot
            birth_mask[d, micro, slot] = previous_slot < 0
            source_coupling[d, micro, slot] = combined_coupling[
                site, event.revealed_site
            ]
        dying_slot = old_position.get(event.revealed_site, -1)
        if dying_slot >= 0:
            dying_old_slot[d, micro] = dying_slot
            dying_endpoint_site[d, micro] = event.revealed_site
            death_mask[d, micro] = True
        for slot, (partner, value, shell) in enumerate(event.closing_terms):
            closing_partner_sites[d, micro, slot] = partner
            closing_couplings[d, micro, slot] = value
            closing_shell_slot[d, micro, slot] = shell
            closing_mask[d, micro, slot] = True
            newly_closed_shell_sum[d, micro, shell] += value

        before = _sector_summaries(
            set(event.old_hidden),
            set(event.old_revealed),
            bond_first,
            bond_second,
            bond_coupling,
            bond_shell_slot,
            shell_count,
        )
        after = _sector_summaries(
            set(event.new_hidden),
            set(event.new_revealed),
            bond_first,
            bond_second,
            bond_coupling,
            bond_shell_slot,
            shell_count,
        )
        for name, value in zip(
            (
                "hidden_sum_before",
                "crossing_sum_before",
                "closed_sum_before",
                "hidden_squared_before",
                "crossing_squared_before",
                "closed_squared_before",
                "hidden_count_before",
                "crossing_count_before",
                "closed_count_before",
            ),
            before,
            strict=True,
        ):
            (counts if "count" in name else summaries)[name][d, micro] = value
        for name, value in zip(
            (
                "hidden_sum_after",
                "crossing_sum_after",
                "closed_sum_after",
                "hidden_squared_after",
                "crossing_squared_after",
                "closed_squared_after",
                "hidden_count_after",
                "crossing_count_after",
                "closed_count_after",
            ),
            after,
            strict=True,
        ):
            (counts if "count" in name else summaries)[name][d, micro] = value

        old_count = len(event.old_revealed)
        new_count = len(event.new_revealed)
        hidden_before = n_sites - old_count
        hidden_after = n_sites - new_count
        if event_index == 0:
            initial_event[d, micro] = True
        for q_before in range(q_size):
            if not _reachable_singlet_cut(q_before, old_count, hidden_before):
                continue
            for q_after in range(q_size):
                if (
                    not _reachable_singlet_cut(q_after, new_count, hidden_after)
                    or abs(q_after - q_before) != 1
                ):
                    continue
                q_transition_valid[d, micro, q_before, q_after] = True
                for new_slot, delta_new in enumerate(VECTOR_DELTA_Q):
                    q_bra_after = q_after + delta_new
                    if not _reachable_singlet_cut(
                        q_bra_after, new_count, hidden_after
                    ):
                        continue
                    endpoint_source[
                        d, micro, q_before, q_after, new_slot
                    ] = appended_spin_vector_source_coefficient(
                        q_before, q_bra_after, q_after
                    )
                    for old_slot, delta_old in enumerate(VECTOR_DELTA_Q):
                        q_bra_before = q_before + delta_old
                        if old_count == 0 or not _reachable_singlet_cut(
                            q_bra_before, old_count, hidden_before
                        ):
                            continue
                        old_block_transport[
                            d,
                            micro,
                            q_before,
                            q_after,
                            new_slot,
                            old_slot,
                        ] = old_block_vector_advance_coefficient(
                            q_bra_before,
                            q_before,
                            q_bra_after,
                            q_after,
                        )
                if old_count:
                    for old_slot, delta_old in enumerate(VECTOR_DELTA_Q):
                        q_bra_before = q_before + delta_old
                        if _reachable_singlet_cut(
                            q_bra_before, old_count, hidden_before
                        ):
                            dying_scalar[
                                d, micro, q_before, q_after, old_slot
                            ] = appended_spin_scalar_contraction_coefficient(
                                q_bra_before, q_before, q_after
                            )

    return SparseCausalRecurrencePlan(
        n_sites=n_sites,
        target_q=target_q,
        delta_q=VECTOR_DELTA_Q,
        maximum_active_endpoints=maximum_active,
        endpoint_slot_count=endpoint_slots,
        maximum_closing_terms=maximum_closing,
        closing_term_slot_count=closing_slots,
        shell_ids=shell_ids,
        bond_first=bond_first,
        bond_second=bond_second,
        bond_coupling=bond_coupling,
        bond_shell_slot=bond_shell_slot,
        event_valid=event_valid,
        decision_substep_count=decision_substep_count,
        revealed_site=revealed_site,
        revealed_count_before=revealed_count_before,
        revealed_count_after=revealed_count_after,
        old_endpoint_sites=old_endpoint_sites,
        old_endpoint_mask=old_endpoint_mask,
        new_endpoint_sites=new_endpoint_sites,
        new_endpoint_mask=new_endpoint_mask,
        persist_old_slot_for_new=persist_old_slot_for_new,
        birth_mask=birth_mask,
        source_coupling=source_coupling,
        dying_old_slot=dying_old_slot,
        dying_endpoint_site=dying_endpoint_site,
        death_mask=death_mask,
        closing_partner_sites=closing_partner_sites,
        closing_couplings=closing_couplings,
        closing_shell_slot=closing_shell_slot,
        closing_mask=closing_mask,
        q_transition_valid=q_transition_valid,
        old_block_transport=old_block_transport,
        endpoint_source=endpoint_source,
        dying_scalar=dying_scalar,
        initial_event=initial_event,
        hidden_shell_sum_before=summaries["hidden_sum_before"],
        crossing_shell_sum_before=summaries["crossing_sum_before"],
        closed_shell_sum_before=summaries["closed_sum_before"],
        hidden_shell_sum_after=summaries["hidden_sum_after"],
        crossing_shell_sum_after=summaries["crossing_sum_after"],
        closed_shell_sum_after=summaries["closed_sum_after"],
        hidden_shell_squared_before=summaries["hidden_squared_before"],
        crossing_shell_squared_before=summaries["crossing_squared_before"],
        closed_shell_squared_before=summaries["closed_squared_before"],
        hidden_shell_squared_after=summaries["hidden_squared_after"],
        crossing_shell_squared_after=summaries["crossing_squared_after"],
        closed_shell_squared_after=summaries["closed_squared_after"],
        hidden_shell_count_before=counts["hidden_count_before"],
        crossing_shell_count_before=counts["crossing_count_before"],
        closed_shell_count_before=counts["closed_count_before"],
        hidden_shell_count_after=counts["hidden_count_after"],
        crossing_shell_count_after=counts["crossing_count_after"],
        closed_shell_count_after=counts["closed_count_after"],
        newly_closed_shell_sum=newly_closed_shell_sum,
    )


__all__ = [
    "MAX_REVEALS_PER_DECISION",
    "SparseCausalRecurrencePlan",
    "compile_sparse_causal_recurrence_plan",
]
