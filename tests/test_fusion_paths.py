from math import comb

import pytest

from su2_nqs.su2.fusion_path import (
    allowed_next_steps,
    count_paths,
    enumerate_paths,
    is_valid_path,
    next_step_mask,
    steps_from_path,
)


@pytest.mark.parametrize(
    ("n", "expected"),
    [(2, 1), (4, 2), (6, 5), (8, 14), (10, 42)],
)
def test_singlet_path_counts(n, expected):
    paths = enumerate_paths(n, 0)
    assert len(paths) == expected == count_paths(n, 0)
    assert all(is_valid_path(path, 0) for path in paths)


@pytest.mark.parametrize("n", [2, 4, 6, 8, 10])
def test_all_total_spin_counts_match_closed_formula(n):
    for target_q in range(n % 2, n + 1, 2):
        k = (n - target_q) // 2
        expected = comb(n, k) - (comb(n, k - 1) if k else 0)
        assert count_paths(n, target_q) == expected
        assert len(enumerate_paths(n, target_q)) == expected


def test_next_step_mask_prunes_prefixes_that_cannot_reach_target():
    assert allowed_next_steps((), 6, 0) == (1,)
    assert allowed_next_steps((1,), 6, 0) == (-1, 1)
    assert allowed_next_steps((1, 0), 6, 0) == (1,)
    assert next_step_mask((1, 2, 3), 6, 0) == (True, False)
    assert allowed_next_steps((1, 2, 3, 4), 6, 0) == ()


def test_path_step_round_trip_and_rejection():
    path = (1, 2, 1, 2, 1, 0)
    assert steps_from_path(path) == (1, -1, 1, -1, -1)
    assert not is_valid_path((1, 0, -1))
    assert not is_valid_path((1, 3))

