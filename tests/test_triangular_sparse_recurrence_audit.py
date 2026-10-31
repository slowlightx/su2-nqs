"""Independent audit of the sparse recurrence on triangular 4x4 PBC."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from su2_nqs.benchmarks.sparse_causal_recurrence import (
    MAX_REVEALS_PER_DECISION,
    SparseCausalRecurrencePlan,
    compile_sparse_causal_recurrence_plan,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.su2.reduced_vector import VECTOR_DELTA_Q


N = 16
J2_OVER_J1 = 0.5
SHELL_COUNTS = np.asarray((48, 48), dtype=np.int32)
SHELL_SUMS = np.asarray((48.0, 24.0), dtype=np.float64)
SHELL_SQUARED_SUMS = np.asarray((48.0, 12.0), dtype=np.float64)


def _two_shell_bonds() -> tuple[tuple[int, int, float, int], ...]:
    j1 = triangular_j1j2_bonds(
        4, 4, j1=1.0, j2=0.0, periodic=True
    )
    j2 = triangular_j1j2_bonds(
        4, 4, j1=0.0, j2=1.0, periodic=True
    )
    assert len(j1) == len(j2) == 48
    return tuple((i, j, value, 0) for i, j, value in j1) + tuple(
        (i, j, J2_OVER_J1 * value, 1) for i, j, value in j2
    )


def _independent_shell_matrices(
    bonds: tuple[tuple[int, int, float, int], ...],
) -> np.ndarray:
    matrices = np.zeros((2, N, N), dtype=np.float64)
    for first, second, coupling, shell in bonds:
        matrices[shell, first, second] += coupling
        matrices[shell, second, first] += coupling
    return matrices


def _active_events(plan: SparseCausalRecurrencePlan):
    for decision in range(plan.n_decisions):
        for microstep in range(MAX_REVEALS_PER_DECISION):
            if plan.event_valid[decision, microstep]:
                yield decision, microstep


def _reachable_singlet_cut(q: int, revealed: int, hidden: int) -> bool:
    return (
        0 <= q <= revealed
        and (revealed - q) % 2 == 0
        and q <= hidden
        and (hidden - q) % 2 == 0
    )


def _rank_one_triangle(ket_q: int, bra_q: int) -> bool:
    """Independent doubled-label triangle rule for ``q_op=2``."""

    return (
        bra_q >= 0
        and abs(ket_q - 2) <= bra_q <= ket_q + 2
        and (ket_q + bra_q + 2) % 2 == 0
    )


@pytest.fixture(scope="module")
def triangular_plan() -> SparseCausalRecurrencePlan:
    return compile_sparse_causal_recurrence_plan(N, _two_shell_bonds())


def test_triangular_endpoint_rows_and_shell_partitions_close_exactly(
    triangular_plan: SparseCausalRecurrencePlan,
) -> None:
    plan = triangular_plan
    shell_matrices = _independent_shell_matrices(_two_shell_bonds())
    coupling = np.sum(shell_matrices, axis=0)

    np.testing.assert_array_equal(plan.shell_ids, (0, 1))
    np.testing.assert_array_equal(
        np.bincount(plan.bond_shell_slot, minlength=2), SHELL_COUNTS
    )
    assert plan.event_count == N
    assert not plan.uses_hidden_suffix
    assert not plan.enumerates_revealed_prefixes

    previous_after: dict[tuple[str, str], np.ndarray] | None = None
    for decision, microstep in _active_events(plan):
        after_count = int(plan.revealed_count_after[decision, microstep])
        endpoints, prefix, rows = plan.reconstruct_endpoint_coefficients(
            decision, microstep
        )

        # The reference is formed directly from the physical two-shell graph,
        # not from any compiler partition or reconstruction helper.
        expected_prefix = np.arange(N - 1, N - 1 - after_count, -1, dtype=np.int32)
        np.testing.assert_array_equal(prefix, expected_prefix)
        hidden = np.setdiff1d(np.arange(N, dtype=np.int32), prefix)
        expected_endpoints = np.asarray(
            [
                site
                for site in hidden
                if np.any(np.abs(coupling[site, prefix]) > 0.0)
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(endpoints, expected_endpoints)
        np.testing.assert_allclose(
            rows, coupling[np.ix_(endpoints, prefix)], atol=0.0, rtol=0.0
        )

        site = int(plan.revealed_site[decision, microstep])
        old_prefix = set(map(int, prefix[:-1]))
        expected_closing = []
        for partner in sorted(old_prefix):
            for shell in range(2):
                value = float(shell_matrices[shell, site, partner])
                if value:
                    expected_closing.append((partner, value, shell))
        actual_closing = [
            (
                int(plan.closing_partner_sites[decision, microstep, slot]),
                float(plan.closing_couplings[decision, microstep, slot]),
                int(plan.closing_shell_slot[decision, microstep, slot]),
            )
            for slot in np.flatnonzero(plan.closing_mask[decision, microstep])
        ]
        assert actual_closing == expected_closing

        totals = {
            "count": SHELL_COUNTS,
            "sum": SHELL_SUMS,
            "squared": SHELL_SQUARED_SUMS,
        }
        current_after: dict[tuple[str, str], np.ndarray] = {}
        for statistic, total in totals.items():
            before = tuple(
                getattr(plan, f"{sector}_shell_{statistic}_before")[
                    decision, microstep
                ]
                for sector in ("hidden", "crossing", "closed")
            )
            after = tuple(
                getattr(plan, f"{sector}_shell_{statistic}_after")[
                    decision, microstep
                ]
                for sector in ("hidden", "crossing", "closed")
            )
            np.testing.assert_allclose(sum(before), total, atol=0.0, rtol=0.0)
            np.testing.assert_allclose(sum(after), total, atol=0.0, rtol=0.0)
            for sector, value in zip(
                ("hidden", "crossing", "closed"), after, strict=True
            ):
                current_after[(statistic, sector)] = value.copy()
            if previous_after is not None:
                for sector, value in zip(
                    ("hidden", "crossing", "closed"), before, strict=True
                ):
                    np.testing.assert_array_equal(
                        value, previous_after[(statistic, sector)]
                    )

        closing_shells = np.asarray(
            [item[2] for item in actual_closing], dtype=np.int32
        )
        closing_values = np.asarray(
            [item[1] for item in actual_closing], dtype=np.float64
        )
        closed_count_increment = (
            plan.closed_shell_count_after[decision, microstep]
            - plan.closed_shell_count_before[decision, microstep]
        )
        closed_sum_increment = (
            plan.closed_shell_sum_after[decision, microstep]
            - plan.closed_shell_sum_before[decision, microstep]
        )
        expected_new_count = np.bincount(closing_shells, minlength=2)
        expected_new_sum = np.bincount(
            closing_shells, weights=closing_values, minlength=2
        )
        np.testing.assert_array_equal(closed_count_increment, expected_new_count)
        np.testing.assert_allclose(
            closed_sum_increment, expected_new_sum, atol=0.0, rtol=0.0
        )
        np.testing.assert_allclose(
            plan.newly_closed_shell_sum[decision, microstep],
            expected_new_sum,
            atol=0.0,
            rtol=0.0,
        )
        previous_after = current_after

    assert previous_after is not None
    np.testing.assert_array_equal(previous_after[("count", "hidden")], (0, 0))
    np.testing.assert_array_equal(previous_after[("count", "crossing")], (0, 0))
    np.testing.assert_array_equal(
        previous_after[("count", "closed")], SHELL_COUNTS
    )
    np.testing.assert_allclose(
        previous_after[("sum", "closed")], SHELL_SUMS, atol=0.0, rtol=0.0
    )


def test_triangular_plan_is_invariant_to_every_bond_orientation(
    triangular_plan: SparseCausalRecurrencePlan,
) -> None:
    reversed_bonds = tuple(
        (second, first, coupling, shell)
        for first, second, coupling, shell in reversed(_two_shell_bonds())
    )
    reversed_plan = compile_sparse_causal_recurrence_plan(N, reversed_bonds)
    for field in fields(SparseCausalRecurrencePlan):
        expected = getattr(triangular_plan, field.name)
        actual = getattr(reversed_plan, field.name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected, err_msg=field.name)
        else:
            assert actual == expected, field.name


def test_final_two_microsteps_and_q_op_two_masks_are_exact(
    triangular_plan: SparseCausalRecurrencePlan,
) -> None:
    plan = triangular_plan
    np.testing.assert_array_equal(
        plan.decision_substep_count, (1,) * (N - 2) + (2,)
    )
    assert np.all(plan.event_valid[:, 0])
    assert not np.any(plan.event_valid[:-1, 1])
    assert plan.event_valid[-1, 1]
    np.testing.assert_array_equal(plan.revealed_site[-1], (1, 0))
    np.testing.assert_array_equal(plan.revealed_count_before[-1], (14, 15))
    np.testing.assert_array_equal(plan.revealed_count_after[-1], (15, 16))
    assert {
        tuple(map(int, pair)) for pair in np.argwhere(plan.q_transition_valid[-1, 0])
    } == {(0, 1), (2, 1)}
    assert {
        tuple(map(int, pair)) for pair in np.argwhere(plan.q_transition_valid[-1, 1])
    } == {(1, 0)}

    assert tuple(plan.delta_q) == tuple(VECTOR_DELTA_Q) == (-2, 0, 2)
    for decision, microstep in _active_events(plan):
        revealed_before = int(plan.revealed_count_before[decision, microstep])
        revealed_after = int(plan.revealed_count_after[decision, microstep])
        hidden_before = N - revealed_before
        hidden_after = N - revealed_after
        for q_before in range(plan.q_size):
            for q_after in range(plan.q_size):
                physical_transition = (
                    _reachable_singlet_cut(
                        q_before, revealed_before, hidden_before
                    )
                    and _reachable_singlet_cut(
                        q_after, revealed_after, hidden_after
                    )
                    and abs(q_after - q_before) == 1
                )
                assert (
                    bool(
                        plan.q_transition_valid[
                            decision, microstep, q_before, q_after
                        ]
                    )
                    == physical_transition
                )
                for new_slot, delta_new in enumerate(plan.delta_q):
                    q_bra_after = q_after + delta_new
                    new_vector_allowed = (
                        physical_transition
                        and _reachable_singlet_cut(
                            q_bra_after, revealed_after, hidden_after
                        )
                        and _rank_one_triangle(q_after, q_bra_after)
                    )
                    if not new_vector_allowed:
                        assert (
                            plan.endpoint_source[
                                decision,
                                microstep,
                                q_before,
                                q_after,
                                new_slot,
                            ]
                            == 0.0
                        )
                    for old_slot, delta_old in enumerate(plan.delta_q):
                        q_bra_before = q_before + delta_old
                        old_vector_allowed = (
                            revealed_before > 0
                            and _reachable_singlet_cut(
                                q_bra_before, revealed_before, hidden_before
                            )
                            and _rank_one_triangle(q_before, q_bra_before)
                        )
                        if not (new_vector_allowed and old_vector_allowed):
                            assert (
                                plan.old_block_transport[
                                    decision,
                                    microstep,
                                    q_before,
                                    q_after,
                                    new_slot,
                                    old_slot,
                                ]
                                == 0.0
                            )
                    q_bra_before = q_before + delta_new
                    dying_allowed = (
                        physical_transition
                        and revealed_before > 0
                        and _reachable_singlet_cut(
                            q_bra_before, revealed_before, hidden_before
                        )
                        and _rank_one_triangle(q_before, q_bra_before)
                    )
                    if not dying_allowed:
                        assert (
                            plan.dying_scalar[
                                decision,
                                microstep,
                                q_before,
                                q_after,
                                new_slot,
                            ]
                            == 0.0
                        )

    # The triangular recurrence exercises all three allowed rank-one changes;
    # the zero-mask assertions above therefore cannot pass vacuously.
    assert np.all(np.any(plan.endpoint_source != 0.0, axis=(0, 1, 2, 3)))
    assert np.all(np.any(plan.dying_scalar != 0.0, axis=(0, 1, 2, 3)))
