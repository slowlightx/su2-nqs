import math

import numpy as np
import pytest

from su2_nqs.benchmarks.sparse_causal_recurrence import (
    MAX_REVEALS_PER_DECISION,
    compile_sparse_causal_recurrence_plan,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.su2.reduced_vector import (
    VECTOR_DELTA_Q,
    advance_weighted_spin_vector,
    compile_weighted_spin_vector,
    contract_reduced_vector_scalar,
)


def _directional_graph(n_sites: int):
    return tuple(
        (site + 1, site, 1.0, 0) for site in range(n_sites - 1)
    ) + tuple(
        (site + 2, site, 0.37, 1) for site in range(n_sites - 2)
    )


def _combined_matrix(plan) -> np.ndarray:
    matrix = np.zeros((plan.n_sites, plan.n_sites), dtype=np.float64)
    for first, second, value in zip(
        plan.bond_first, plan.bond_second, plan.bond_coupling, strict=True
    ):
        matrix[int(first), int(second)] += float(value)
        matrix[int(second), int(first)] += float(value)
    return matrix


@pytest.mark.parametrize("n_sites", (4, 6, 8, 10))
def test_endpoint_events_reconstruct_every_exact_graph_partition(n_sites: int):
    plan = compile_sparse_causal_recurrence_plan(
        n_sites, _directional_graph(n_sites)
    )
    coupling = _combined_matrix(plan)
    assert plan.event_count == n_sites
    assert tuple(plan.decision_substep_count[:-1]) == (1,) * (n_sites - 2)
    assert plan.decision_substep_count[-1] == 2
    assert tuple(plan.revealed_site[-1]) == (1, 0)
    assert not plan.uses_hidden_suffix
    assert not plan.enumerates_revealed_prefixes

    previous_after = None
    for decision in range(plan.n_decisions):
        for micro in range(MAX_REVEALS_PER_DECISION):
            if not plan.event_valid[decision, micro]:
                continue
            endpoints, prefix, rows = plan.reconstruct_endpoint_coefficients(
                decision, micro
            )
            revealed = set(map(int, prefix))
            hidden = set(range(n_sites)) - revealed
            expected_endpoints = np.asarray(
                [
                    site
                    for site in sorted(hidden)
                    if any(abs(coupling[site, other]) > 1.0e-14 for other in revealed)
                ],
                dtype=np.int32,
            )
            np.testing.assert_array_equal(endpoints, expected_endpoints)
            expected_rows = coupling[np.ix_(endpoints, prefix)]
            np.testing.assert_allclose(rows, expected_rows, atol=1.0e-14)

            old_sites = plan.old_endpoint_sites[decision, micro][
                plan.old_endpoint_mask[decision, micro]
            ]
            new_sites = plan.new_endpoint_sites[decision, micro][
                plan.new_endpoint_mask[decision, micro]
            ]
            site = int(plan.revealed_site[decision, micro])
            for new_slot, endpoint in enumerate(new_sites):
                old_slot = int(
                    plan.persist_old_slot_for_new[decision, micro, new_slot]
                )
                assert (old_slot < 0) == bool(
                    plan.birth_mask[decision, micro, new_slot]
                )
                if old_slot >= 0:
                    assert int(old_sites[old_slot]) == int(endpoint)
                assert plan.source_coupling[
                    decision, micro, new_slot
                ] == pytest.approx(coupling[int(endpoint), site])
            if site in set(map(int, old_sites)):
                assert plan.death_mask[decision, micro]
                slot = int(plan.dying_old_slot[decision, micro])
                assert int(old_sites[slot]) == site
            else:
                assert not plan.death_mask[decision, micro]

            before_sums = (
                plan.hidden_shell_sum_before[decision, micro]
                + plan.crossing_shell_sum_before[decision, micro]
                + plan.closed_shell_sum_before[decision, micro]
            )
            after_sums = (
                plan.hidden_shell_sum_after[decision, micro]
                + plan.crossing_shell_sum_after[decision, micro]
                + plan.closed_shell_sum_after[decision, micro]
            )
            total_by_shell = np.bincount(
                plan.bond_shell_slot,
                weights=plan.bond_coupling,
                minlength=plan.shell_count,
            )
            np.testing.assert_allclose(before_sums, total_by_shell, atol=1.0e-14)
            np.testing.assert_allclose(after_sums, total_by_shell, atol=1.0e-14)
            np.testing.assert_allclose(
                plan.closed_shell_sum_after[decision, micro]
                - plan.closed_shell_sum_before[decision, micro],
                plan.newly_closed_shell_sum[decision, micro],
                atol=1.0e-14,
            )
            if previous_after is not None:
                for before, previous in zip(
                    (
                        plan.hidden_shell_sum_before[decision, micro],
                        plan.crossing_shell_sum_before[decision, micro],
                        plan.closed_shell_sum_before[decision, micro],
                    ),
                    previous_after,
                    strict=True,
                ):
                    np.testing.assert_array_equal(before, previous)
            previous_after = (
                plan.hidden_shell_sum_after[decision, micro],
                plan.crossing_shell_sum_after[decision, micro],
                plan.closed_shell_sum_after[decision, micro],
            )

    np.testing.assert_allclose(previous_after[0], 0.0, atol=0.0)
    np.testing.assert_allclose(previous_after[1], 0.0, atol=0.0)


def test_endpoint_events_are_independent_of_input_bond_orientation():
    n_sites = 10
    forward = tuple(
        (min(first, second), max(first, second), value, shell)
        for first, second, value, shell in _directional_graph(n_sites)
    )
    reversed_bonds = tuple(
        (second, first, value, shell)
        for first, second, value, shell in forward
    )
    first = compile_sparse_causal_recurrence_plan(n_sites, forward)
    second = compile_sparse_causal_recurrence_plan(n_sites, reversed_bonds)
    for name in (
        "bond_first",
        "bond_second",
        "bond_coupling",
        "revealed_site",
        "old_endpoint_sites",
        "old_endpoint_mask",
        "new_endpoint_sites",
        "new_endpoint_mask",
        "persist_old_slot_for_new",
        "source_coupling",
        "dying_old_slot",
        "closing_partner_sites",
        "closing_couplings",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


@pytest.mark.parametrize("n_sites", (4, 6, 8, 10))
def test_final_two_microsteps_have_independent_singlet_q_reference(n_sites: int):
    """The last site-1/site-0 closure follows ordinary spin-half triangles."""

    plan = compile_sparse_causal_recurrence_plan(
        n_sites, _directional_graph(n_sites)
    )
    decision = plan.n_decisions - 1
    np.testing.assert_array_equal(plan.revealed_site[decision], (1, 0))
    assert tuple(plan.revealed_count_before[decision]) == (
        n_sites - 2,
        n_sites - 1,
    )
    # Before the last decision the unrevealed two-spin block can carry only
    # q=0 or q=2. Appending site 1 must give q=1; appending site 0 then closes
    # to the fixed q=0 singlet. This reference uses only independent triangle,
    # parity, and remaining-spin completion rules.
    expected_first = {(0, 1), (2, 1)}
    observed_first = set(
        map(tuple, np.argwhere(plan.q_transition_valid[decision, 0]))
    )
    observed_second = set(
        map(tuple, np.argwhere(plan.q_transition_valid[decision, 1]))
    )
    assert observed_first == expected_first
    assert observed_second == {(1, 0)}


def _event_for_old_count(plan, old_count: int) -> tuple[int, int]:
    match = np.argwhere(
        plan.event_valid & (plan.revealed_count_before == old_count)
    )
    assert len(match) == 1
    return tuple(map(int, match[0]))


def _maximum_sparse_difference(first, second) -> float:
    left = first.entry_dict()
    right = second.entry_dict()
    return max(
        (
            abs(left.get(key, 0.0j) - right.get(key, 0.0j))
            for key in set(left) | set(right)
        ),
        default=0.0,
    )


def _singlet_cut_reachable(q: int, block_sites: int, hidden_sites: int) -> bool:
    return (
        0 <= q <= block_sites
        and (block_sites - q) % 2 == 0
        and q <= hidden_sites
        and (hidden_sites - q) % 2 == 0
    )


def test_local_vector_tables_reproduce_exact_recursive_advance_through_n9():
    n_sites = 10
    plan = compile_sparse_causal_recurrence_plan(
        n_sites, _directional_graph(n_sites)
    )
    rng = np.random.default_rng(7182)
    weights = rng.normal(size=n_sites) + 0.19j * rng.normal(size=n_sites)
    assert plan.delta_q == VECTOR_DELTA_Q
    delta_position = {delta: slot for slot, delta in enumerate(plan.delta_q)}

    first_d, first_micro = _event_for_old_count(plan, 0)
    np.testing.assert_allclose(
        plan.endpoint_source[first_d, first_micro, 0, 1],
        np.asarray([0.0, math.sqrt(1.5), 0.0]),
        atol=2.0e-15,
    )
    assert plan.initial_event[first_d, first_micro]
    assert not np.any(plan.old_block_transport[first_d, first_micro])

    table = compile_weighted_spin_vector(weights[:1])
    for old_count in range(1, n_sites):
        d, micro = _event_for_old_count(plan, old_count)
        endpoint_weight = weights[old_count]
        advanced = advance_weighted_spin_vector(
            table, endpoint_weight, zero_tolerance=1.0e-13
        )
        old_entries = table.entry_dict()
        predicted = {}
        for new_path in advanced.paths:
            prefix_new = new_path[:-1]
            for old_path in advanced.paths:
                prefix_old = old_path[:-1]
                q_before = prefix_old[-1]
                q_after = old_path[-1]
                if not plan.q_transition_valid[d, micro, q_before, q_after]:
                    continue
                delta_old = prefix_new[-1] - prefix_old[-1]
                delta_new = new_path[-1] - old_path[-1]
                if delta_old not in delta_position or delta_new not in delta_position:
                    continue
                new_slot = delta_position[delta_new]
                old_slot = delta_position[delta_old]
                value = (
                    plan.old_block_transport[
                        d,
                        micro,
                        q_before,
                        q_after,
                        new_slot,
                        old_slot,
                    ]
                    * old_entries.get((prefix_new, prefix_old), 0.0j)
                )
                if prefix_new == prefix_old:
                    value += (
                        plan.endpoint_source[
                            d, micro, q_before, q_after, new_slot
                        ]
                        * endpoint_weight
                    )
                if abs(value) > 1.0e-13:
                    predicted[(new_path, old_path)] = value
        restricted_exact = {
            key: value
            for key, value in advanced.entry_dict().items()
            if plan.q_transition_valid[
                d, micro, key[1][:-1][-1], key[1][-1]
            ]
            and _singlet_cut_reachable(
                key[0][:-1][-1], old_count, n_sites - old_count
            )
            and _singlet_cut_reachable(
                key[0][-1], old_count + 1, n_sites - old_count - 1
            )
            and (key[0][-1] - key[1][-1]) in delta_position
            and (key[0][:-1][-1] - key[1][:-1][-1]) in delta_position
        }
        maximum_error = max(
            (
                abs(predicted.get(key, 0.0j) - restricted_exact.get(key, 0.0j))
                for key in set(predicted) | set(restricted_exact)
            ),
            default=0.0,
        )
        assert maximum_error < 4.0e-14
        table = advanced

    direct = compile_weighted_spin_vector(weights, zero_tolerance=1.0e-13)
    assert _maximum_sparse_difference(table, direct) < 3.0e-14


@pytest.mark.parametrize("old_count", (1, 2, 3, 4, 5, 6))
def test_dying_scalar_table_matches_exact_reduced_vector_contraction(old_count: int):
    n_sites = 10
    plan = compile_sparse_causal_recurrence_plan(
        n_sites, _directional_graph(n_sites)
    )
    d, micro = _event_for_old_count(plan, old_count)
    rng = np.random.default_rng(8800 + old_count)
    table = compile_weighted_spin_vector(rng.normal(size=old_count))
    spin = compile_weighted_spin_vector([1.0])
    delta_position = {delta: slot for slot, delta in enumerate(plan.delta_q)}
    entries = table.entry_dict()
    for q_after in range(n_sites + 1):
        matrix, basis = contract_reduced_vector_scalar(
            table, spin, q_after, total_m=q_after
        )
        positions = {state: index for index, state in enumerate(basis)}
        for (new_path, old_path), reduced in entries.items():
            q_before = old_path[-1]
            if not plan.q_transition_valid[d, micro, q_before, q_after]:
                continue
            delta_old = new_path[-1] - old_path[-1]
            if delta_old not in delta_position:
                continue
            if not _singlet_cut_reachable(
                new_path[-1], old_count, n_sites - old_count
            ):
                continue
            row = positions.get((new_path, (1,)))
            column = positions.get((old_path, (1,)))
            if row is None or column is None:
                continue
            expected = (
                plan.dying_scalar[
                    d,
                    micro,
                    q_before,
                    q_after,
                    delta_position[delta_old],
                ]
                * reduced
            )
            np.testing.assert_allclose(matrix[row, column], expected, atol=8.0e-14)


def test_n16_payload_is_fixed_shape_polynomial_and_k0_preserves_topology():
    bonds = square_j1j2_bonds(4, 4, j1=1.0, j2=0.5, periodic=False)
    plan = compile_sparse_causal_recurrence_plan(16, bonds)
    assert plan.old_block_transport.shape == (15, 2, 17, 17, 3, 3)
    assert plan.endpoint_source.shape == (15, 2, 17, 17, 3)
    assert plan.dying_scalar.shape == (15, 2, 17, 17, 3)
    assert plan.payload_nbytes() < 2_000_000
    assert plan.maximum_active_endpoints <= 6
    payload = plan.numeric_payload()
    control = plan.numeric_payload(physical_scale=0)
    assert all(isinstance(value, np.ndarray) for value in payload.values())
    for name in (
        "event_valid",
        "revealed_site",
        "old_endpoint_sites",
        "old_endpoint_mask",
        "new_endpoint_sites",
        "new_endpoint_mask",
        "persist_old_slot_for_new",
        "birth_mask",
        "death_mask",
        "q_transition_valid",
    ):
        np.testing.assert_array_equal(payload[name], control[name])
    for name in (
        "bond_coupling",
        "source_coupling",
        "closing_couplings",
        "old_block_transport",
        "endpoint_source",
        "dying_scalar",
        "hidden_shell_sum_after",
        "closed_shell_sum_after",
    ):
        assert np.max(np.abs(control[name]), initial=0.0) == 0.0
    with pytest.raises(ValueError, match="integer 0 or 1"):
        plan.numeric_payload(physical_scale=0.5)
