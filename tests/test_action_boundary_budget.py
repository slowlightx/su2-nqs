import pytest

from su2_nqs.benchmarks.action_boundary_budget import (
    fusion_action_boundary_budget,
    fusion_action_boundary_budgets,
)


def test_action_cut_mapping_and_hidden_scalar_counts() -> None:
    shells = {
        "j1": ((0, 3, 1.0),),
        "j2": ((1, 3, 1.0),),
    }
    weights = {"j1": 1.0, "j2": 0.3}
    before = fusion_action_boundary_budget(4, shells, weights, 0)
    assert before["hidden_site_count"] == 4
    assert before["prefix_site_count"] == 0
    assert before["combined_vector_rank"] == 0
    assert before["hidden_scalar_bond_count"] == 2
    assert before["hidden_scalar_bond_count_by_shell"] == {"j1": 1, "j2": 1}
    assert before["shell_resolved_hidden_scalar_channel_count"] == 2

    after_first = fusion_action_boundary_budget(4, shells, weights, 1)
    assert after_first["decision"] == 0
    assert after_first["hidden_site_count"] == 3
    assert after_first["prefix_site_count"] == 1
    assert after_first["crossing_bond_count"] == 2
    assert after_first["combined_vector_rank"] == 1
    assert after_first["shell_vector_ranks"] == {"j1": 1, "j2": 1}
    assert after_first["shell_separated_vector_rank"] == 2
    assert after_first["hidden_scalar_bond_count"] == 0

    final = fusion_action_boundary_budget(4, shells, weights, 3)
    assert final["hidden_site_count"] == 0
    assert final["prefix_site_count"] == 4
    assert final["combined_vector_rank"] == 0
    assert final["hidden_scalar_bond_count"] == 0


def test_known_two_by_two_crossing_matrix_has_rank_two() -> None:
    shells = {
        "first": ((0, 4, 1.0), (1, 5, 1.0)),
        "second": ((0, 5, 0.4), (1, 4, -0.2)),
    }
    # prefix_length=2 leaves sites 0..3 hidden and sites 4,5 in the prefix.
    row = fusion_action_boundary_budget(
        6, shells, {"first": 1.0, "second": 1.0}, 2
    )
    assert row["combined_vector_rank"] == 2
    assert row["shell_vector_ranks"] == {"first": 2, "second": 2}
    assert row["shell_separated_vector_rank"] == 4
    assert row["combined_rank_r999"] == 2


def test_all_cut_helper_and_inactive_shell_validation() -> None:
    shells = {"active": ((0, 3),), "inactive": ((1, 3),)}
    rows = fusion_action_boundary_budgets(
        4, shells, {"active": 1.0, "inactive": 0.0}
    )
    assert len(rows) == 4
    assert rows[0]["hidden_scalar_bond_count_by_shell"]["inactive"] == 0
    assert rows[1]["shell_vector_ranks"]["inactive"] == 0
    assert rows[0]["shell_resolved_hidden_scalar_channel_count"] == 1
    with pytest.raises(ValueError, match="identical keys"):
        fusion_action_boundary_budget(4, shells, {"active": 1.0}, 1)
