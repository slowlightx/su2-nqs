"""Independent graph/mask audit for sparse causal recurrence payloads.

This module evaluates only graph partitions and fixed SU(2) masks.  It never
loads a wavefunction, evaluates an energy, or calls a neural model.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .boundary_operator_channels import BoundaryBond, normalize_boundary_bonds
from .sparse_causal_recurrence import (
    MAX_REVEALS_PER_DECISION,
    SparseCausalRecurrencePlan,
)


BondLike = BoundaryBond | tuple[int, int] | tuple[int, int, float] | tuple[int, int, float, int]


def _canonical_bonds(
    n_sites: int, bonds: Iterable[BondLike], tolerance: float
) -> tuple[tuple[int, int, float, int], ...]:
    accumulated: dict[tuple[int, int, int], float] = defaultdict(float)
    for bond in normalize_boundary_bonds(bonds):
        if not 0 <= bond.first < n_sites or not 0 <= bond.second < n_sites:
            raise ValueError("bond endpoint lies outside the system")
        first, second = sorted((bond.first, bond.second))
        accumulated[(first, second, bond.channel)] += float(bond.coupling)
    return tuple(
        (first, second, value, shell)
        for (first, second, shell), value in sorted(accumulated.items())
        if abs(value) > tolerance
    )


def _reachable(q: int, revealed_count: int, hidden_count: int) -> bool:
    return (
        0 <= q <= revealed_count
        and (revealed_count - q) % 2 == 0
        and q <= hidden_count
        and (hidden_count - q) % 2 == 0
    )


def compare_numeric_payloads(
    first: SparseCausalRecurrencePlan, second: SparseCausalRecurrencePlan
) -> dict[str, object]:
    """Compare complete numeric payloads, including graph orientation data."""

    left = first.numeric_payload()
    right = second.numeric_payload()
    mismatched: list[str] = []
    maximum = 0.0
    if set(left) != set(right):
        mismatched.extend(sorted(set(left) ^ set(right)))
    for name in sorted(set(left) & set(right)):
        a, b = left[name], right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            mismatched.append(name)
            continue
        if np.issubdtype(a.dtype, np.inexact):
            error = float(np.max(np.abs(a - b), initial=0.0))
            maximum = max(maximum, error)
            if error != 0.0:
                mismatched.append(name)
        elif not np.array_equal(a, b):
            mismatched.append(name)
    return {
        "maximum_absolute_difference": maximum,
        "mismatched_arrays": sorted(set(mismatched)),
        "exactly_equal": not mismatched and maximum == 0.0,
    }


def audit_sparse_causal_recurrence_plan(
    plan: SparseCausalRecurrencePlan,
    source_bonds: Iterable[BondLike],
    *,
    tolerance: float = 1.0e-12,
) -> dict[str, object]:
    """Audit every event against an independently partitioned source graph."""

    bonds = _canonical_bonds(plan.n_sites, source_bonds, tolerance=1.0e-14)
    shell_ids = tuple(sorted({shell for _, _, _, shell in bonds}))
    shell_position = {shell: slot for slot, shell in enumerate(shell_ids)}
    compiled = tuple(
        (
            int(first),
            int(second),
            float(value),
            int(plan.shell_ids[int(slot)]),
        )
        for first, second, value, slot in zip(
            plan.bond_first,
            plan.bond_second,
            plan.bond_coupling,
            plan.bond_shell_slot,
            strict=True,
        )
    )
    source_by_key = {(i, j, shell): value for i, j, value, shell in bonds}
    compiled_by_key = {(i, j, shell): value for i, j, value, shell in compiled}
    graph_keys = set(source_by_key) | set(compiled_by_key)
    graph_error = max(
        (abs(source_by_key.get(key, 0.0) - compiled_by_key.get(key, 0.0)) for key in graph_keys),
        default=0.0,
    )

    coupling = np.zeros((plan.n_sites, plan.n_sites), dtype=np.float64)
    for first, second, value, _ in bonds:
        coupling[first, second] += value
        coupling[second, first] += value

    expected_valid = np.zeros_like(plan.event_valid)
    expected_sites = np.full_like(plan.revealed_site, -1)
    for decision in range(plan.n_decisions):
        sites = (plan.n_sites - 1 - decision,) if decision < plan.n_decisions - 1 else (1, 0)
        for micro, site in enumerate(sites):
            expected_valid[decision, micro] = True
            expected_sites[decision, micro] = site

    metadata_mismatches = int(np.count_nonzero(plan.event_valid != expected_valid))
    metadata_mismatches += int(np.count_nonzero(plan.revealed_site != expected_sites))
    metadata_mismatches += int(
        np.count_nonzero(
            plan.decision_substep_count
            != np.asarray([1] * (plan.n_decisions - 1) + [2], dtype=np.int32)
        )
    )

    maximum_endpoint_error = 0.0
    maximum_source_error = 0.0
    maximum_partition_error = 0.0
    maximum_partition_count_error = 0
    maximum_closing_error = 0.0
    endpoint_set_mismatches = 0
    lifecycle_mismatches = 0
    closing_metadata_mismatches = 0
    q_mask_mismatches = 0
    maximum_forbidden_qop2_coefficient = 0.0
    revealed: set[int] = set()
    event_rows: list[dict[str, object]] = []
    total_births = total_persists = total_deaths = total_closing = 0
    closing_by_shell = np.zeros(plan.shell_count, dtype=np.int32)

    def sector_arrays(hidden: set[int], shown: set[int]):
        sums = np.zeros((3, plan.shell_count), dtype=np.float64)
        squared = np.zeros_like(sums)
        counts = np.zeros((3, plan.shell_count), dtype=np.int32)
        for first, second, value, shell in bonds:
            if first in hidden and second in hidden:
                sector = 0
            elif first in shown and second in shown:
                sector = 2
            else:
                sector = 1
            slot = shell_position[shell]
            sums[sector, slot] += value
            squared[sector, slot] += value * value
            counts[sector, slot] += 1
        return sums, squared, counts

    for decision in range(plan.n_decisions):
        for micro in range(MAX_REVEALS_PER_DECISION):
            if not plan.event_valid[decision, micro]:
                continue
            site = int(plan.revealed_site[decision, micro])
            old_revealed = set(revealed)
            old_hidden = set(range(plan.n_sites)) - old_revealed
            old_endpoints = tuple(
                endpoint for endpoint in sorted(old_hidden)
                if any(abs(coupling[endpoint, other]) > 1.0e-14 for other in old_revealed)
            )
            revealed.add(site)
            new_revealed = set(revealed)
            new_hidden = set(range(plan.n_sites)) - new_revealed
            new_endpoints = tuple(
                endpoint for endpoint in sorted(new_hidden)
                if any(abs(coupling[endpoint, other]) > 1.0e-14 for other in new_revealed)
            )
            actual_old = tuple(
                int(x) for x in plan.old_endpoint_sites[decision, micro][plan.old_endpoint_mask[decision, micro]]
            )
            actual_new = tuple(
                int(x) for x in plan.new_endpoint_sites[decision, micro][plan.new_endpoint_mask[decision, micro]]
            )
            endpoint_set_mismatches += int(actual_old != old_endpoints) + int(actual_new != new_endpoints)

            endpoints, prefix, rows = plan.reconstruct_endpoint_coefficients(decision, micro)
            expected_rows = coupling[np.ix_(endpoints, prefix)]
            maximum_endpoint_error = max(
                maximum_endpoint_error,
                float(np.max(np.abs(rows - expected_rows), initial=0.0)),
            )
            old_position = {endpoint: slot for slot, endpoint in enumerate(old_endpoints)}
            births: list[int] = []
            persists: list[int] = []
            for slot, endpoint in enumerate(new_endpoints):
                old_slot = int(plan.persist_old_slot_for_new[decision, micro, slot])
                expected_slot = old_position.get(endpoint, -1)
                lifecycle_mismatches += int(old_slot != expected_slot)
                lifecycle_mismatches += int(bool(plan.birth_mask[decision, micro, slot]) != (expected_slot < 0))
                expected_source = coupling[endpoint, site]
                maximum_source_error = max(
                    maximum_source_error,
                    abs(float(plan.source_coupling[decision, micro, slot]) - expected_source),
                )
                (births if expected_slot < 0 else persists).append(endpoint)
            expected_death = site in old_position
            lifecycle_mismatches += int(bool(plan.death_mask[decision, micro]) != expected_death)
            if expected_death:
                lifecycle_mismatches += int(int(plan.dying_old_slot[decision, micro]) != old_position[site])

            expected_closing = {
                (other, shell_position[shell]): value
                for first, second, value, shell in bonds
                for other in ([second] if first == site else [first] if second == site else [])
                if other in old_revealed
            }
            actual_closing = {
                (int(partner), int(shell)): float(value)
                for partner, shell, value in zip(
                    plan.closing_partner_sites[decision, micro][plan.closing_mask[decision, micro]],
                    plan.closing_shell_slot[decision, micro][plan.closing_mask[decision, micro]],
                    plan.closing_couplings[decision, micro][plan.closing_mask[decision, micro]],
                    strict=True,
                )
            }
            closing_metadata_mismatches += len(set(expected_closing) ^ set(actual_closing))
            maximum_closing_error = max(
                maximum_closing_error,
                max(
                    (abs(expected_closing.get(key, 0.0) - actual_closing.get(key, 0.0)) for key in set(expected_closing) | set(actual_closing)),
                    default=0.0,
                ),
            )
            expected_newly_closed = np.zeros(plan.shell_count, dtype=np.float64)
            for (_, shell), value in expected_closing.items():
                expected_newly_closed[shell] += value
                closing_by_shell[shell] += 1
            maximum_closing_error = max(
                maximum_closing_error,
                float(np.max(np.abs(plan.newly_closed_shell_sum[decision, micro] - expected_newly_closed), initial=0.0)),
            )

            for suffix, hidden, shown in (("before", old_hidden, old_revealed), ("after", new_hidden, new_revealed)):
                sums, squared, counts = sector_arrays(hidden, shown)
                for sector, name in enumerate(("hidden", "crossing", "closed")):
                    actual_sum = getattr(plan, f"{name}_shell_sum_{suffix}")[decision, micro]
                    actual_squared = getattr(plan, f"{name}_shell_squared_{suffix}")[decision, micro]
                    actual_count = getattr(plan, f"{name}_shell_count_{suffix}")[decision, micro]
                    maximum_partition_error = max(
                        maximum_partition_error,
                        float(np.max(np.abs(actual_sum - sums[sector]), initial=0.0)),
                        float(np.max(np.abs(actual_squared - squared[sector]), initial=0.0)),
                    )
                    maximum_partition_count_error = max(
                        maximum_partition_count_error,
                        int(np.max(np.abs(actual_count - counts[sector]), initial=0)),
                    )

            old_count = len(old_revealed)
            new_count = len(new_revealed)
            expected_q = np.zeros((plan.q_size, plan.q_size), dtype=bool)
            for q_before in range(plan.q_size):
                for q_after in range(plan.q_size):
                    expected_q[q_before, q_after] = (
                        _reachable(q_before, old_count, plan.n_sites - old_count)
                        and _reachable(q_after, new_count, plan.n_sites - new_count)
                        and abs(q_after - q_before) == 1
                    )
            actual_q = plan.q_transition_valid[decision, micro]
            q_mask_mismatches += int(np.count_nonzero(actual_q != expected_q))
            for q_before, q_after in np.argwhere(actual_q):
                for new_slot, delta_new in enumerate(plan.delta_q):
                    new_allowed = _reachable(q_after + delta_new, new_count, plan.n_sites - new_count)
                    if not new_allowed:
                        maximum_forbidden_qop2_coefficient = max(
                            maximum_forbidden_qop2_coefficient,
                            abs(float(plan.endpoint_source[decision, micro, q_before, q_after, new_slot])),
                        )
                    for old_slot, delta_old in enumerate(plan.delta_q):
                        old_allowed = old_count > 0 and _reachable(
                            q_before + delta_old, old_count, plan.n_sites - old_count
                        )
                        if not (old_allowed and new_allowed):
                            maximum_forbidden_qop2_coefficient = max(
                                maximum_forbidden_qop2_coefficient,
                                abs(float(plan.old_block_transport[decision, micro, q_before, q_after, new_slot, old_slot])),
                            )
                    if old_count:
                        old_allowed = _reachable(
                            q_before + plan.delta_q[new_slot], old_count, plan.n_sites - old_count
                        )
                        if not old_allowed:
                            maximum_forbidden_qop2_coefficient = max(
                                maximum_forbidden_qop2_coefficient,
                                abs(float(plan.dying_scalar[decision, micro, q_before, q_after, new_slot])),
                            )
            invalid_q = ~actual_q
            maximum_forbidden_qop2_coefficient = max(
                maximum_forbidden_qop2_coefficient,
                float(np.max(np.abs(plan.endpoint_source[decision, micro][invalid_q]), initial=0.0)),
                float(np.max(np.abs(plan.dying_scalar[decision, micro][invalid_q]), initial=0.0)),
                float(np.max(np.abs(plan.old_block_transport[decision, micro][invalid_q]), initial=0.0)),
            )

            total_births += len(births)
            total_persists += len(persists)
            total_deaths += int(expected_death)
            total_closing += len(expected_closing)
            event_rows.append({
                "decision": decision,
                "substep": micro,
                "revealed_site": site,
                "revealed_count_before": len(old_revealed),
                "revealed_count_after": len(new_revealed),
                "old_endpoint_count": len(old_endpoints),
                "new_endpoint_count": len(new_endpoints),
                "birth_sites": births,
                "persistent_sites": persists,
                "death": expected_death,
                "closing_terms": len(expected_closing),
                "newly_closed_coupling_by_shell": expected_newly_closed.tolist(),
            })

    source_delta_coverage = np.any(
        np.abs(plan.endpoint_source) > tolerance, axis=(0, 1, 2, 3)
    )
    dying_delta_coverage = np.any(
        np.abs(plan.dying_scalar) > tolerance, axis=(0, 1, 2, 3)
    )
    transport_delta_coverage = np.any(
        np.abs(plan.old_block_transport) > tolerance, axis=(0, 1, 2, 3)
    )
    validations = {
        "source_graph_maximum_error": graph_error,
        "event_metadata_mismatches": metadata_mismatches,
        "endpoint_set_mismatches": endpoint_set_mismatches,
        "maximum_endpoint_reconstruction_error": maximum_endpoint_error,
        "maximum_source_coupling_error": maximum_source_error,
        "lifecycle_mismatches": lifecycle_mismatches,
        "closing_metadata_mismatches": closing_metadata_mismatches,
        "maximum_closing_coupling_error": maximum_closing_error,
        "maximum_partition_sum_or_squared_error": maximum_partition_error,
        "maximum_partition_count_error": maximum_partition_count_error,
        "q_transition_mask_mismatches": q_mask_mismatches,
        "maximum_forbidden_qop2_coefficient": maximum_forbidden_qop2_coefficient,
        "endpoint_source_delta_coverage": source_delta_coverage.tolist(),
        "dying_scalar_delta_coverage": dying_delta_coverage.tolist(),
        "transport_delta_pair_coverage": transport_delta_coverage.tolist(),
        "final_hidden_bond_count": int(np.sum(plan.hidden_shell_count_after[-1, 1])),
        "final_crossing_bond_count": int(np.sum(plan.crossing_shell_count_after[-1, 1])),
    }
    passed = (
        graph_error <= tolerance
        and metadata_mismatches == 0
        and endpoint_set_mismatches == 0
        and maximum_endpoint_error <= tolerance
        and maximum_source_error <= tolerance
        and lifecycle_mismatches == 0
        and closing_metadata_mismatches == 0
        and maximum_closing_error <= tolerance
        and maximum_partition_error <= tolerance
        and maximum_partition_count_error == 0
        and q_mask_mismatches == 0
        and maximum_forbidden_qop2_coefficient <= tolerance
        and bool(np.all(source_delta_coverage))
        and bool(np.all(dying_delta_coverage))
        and validations["final_hidden_bond_count"] == 0
        and validations["final_crossing_bond_count"] == 0
        and tuple(plan.delta_q) == (-2, 0, 2)
        and tuple(plan.shell_ids) == shell_ids
        and not plan.uses_hidden_suffix
        and not plan.enumerates_revealed_prefixes
    )
    shell_rows = []
    for shell, slot in shell_position.items():
        selected = [row for row in bonds if row[3] == shell]
        shell_rows.append({
            "shell_id": shell,
            "bond_count": len(selected),
            "coupling_sum": float(sum(row[2] for row in selected)),
            "closing_term_count": int(closing_by_shell[slot]),
        })
    return {
        "passed": passed,
        "scope": "wavefunction_free_graph_and_fixed_qop2_mask",
        "uses_hidden_suffix": plan.uses_hidden_suffix,
        "enumerates_revealed_prefixes": plan.enumerates_revealed_prefixes,
        "shells": shell_rows,
        "event_count": plan.event_count,
        "maximum_active_endpoints": plan.maximum_active_endpoints,
        "maximum_closing_terms_per_event": plan.maximum_closing_terms,
        "total_births": total_births,
        "total_persistent_transfers": total_persists,
        "total_deaths": total_deaths,
        "total_closing_terms": total_closing,
        "payload_bytes": plan.payload_nbytes(),
        "delta_q": list(plan.delta_q),
        "local_tensor_orientation": "[new_delta_q,old_delta_q]",
        "final_two_microsteps": {
            "decision": plan.n_decisions - 1,
            "substep_count": int(plan.decision_substep_count[-1]),
            "revealed_sites": plan.revealed_site[-1].tolist(),
            "revealed_count_before": plan.revealed_count_before[-1].tolist(),
            "revealed_count_after": plan.revealed_count_after[-1].tolist(),
            "q_transition_pairs": [
                np.argwhere(plan.q_transition_valid[-1, micro]).tolist()
                for micro in range(MAX_REVEALS_PER_DECISION)
            ],
        },
        "validation": validations,
        "events": event_rows,
    }


__all__ = ["audit_sparse_causal_recurrence_plan", "compare_numeric_payloads"]
