"""Exactness and sampling checks for constrained-uniform P8 proposals."""

from __future__ import annotations

from math import comb, exp

import jax
import numpy as np
import pytest

from su2_nqs.su2.fusion_path import count_paths, enumerate_paths, is_valid_path
from su2_nqs.su2.sz_basis import enumerate_sz_configs
from su2_nqs.vmc.constrained_uniform import (
    fusion_completion_count,
    sample_uniform_fusion_paths,
    sample_uniform_sz_configs,
    uniform_fusion_log_probability,
    uniform_fusion_path_probability,
    uniform_sz_log_probability,
)


@pytest.mark.parametrize(
    ("n", "target_q", "expected"),
    [(2, 0, 1), (4, 0, 2), (6, 0, 5), (8, 0, 14), (10, 0, 42)],
)
def test_completion_counts_match_exact_path_multiplicities(
    n: int, target_q: int, expected: int
) -> None:
    assert count_paths(n, target_q) == expected
    assert fusion_completion_count(1, n - 1, target_q) == expected


@pytest.mark.parametrize("n", [2, 4, 6, 8, 10, 12, 14, 16])
def test_every_enumerated_fusion_path_has_probability_one_over_dimension(
    n: int,
) -> None:
    paths = enumerate_paths(n, 0)
    expected = exp(uniform_fusion_log_probability(n, 0))
    assert len(paths) == count_paths(n, 0)
    assert sum(uniform_fusion_path_probability(path, 0) for path in paths) == pytest.approx(1.0)
    assert all(
        uniform_fusion_path_probability(path, 0) == pytest.approx(
            expected, rel=2.0e-14, abs=2.0e-16
        )
        for path in paths
    )


@pytest.mark.parametrize("n", [2, 4, 8, 12, 16])
def test_fixed_magnetization_uniform_probability_matches_combinatorics(n: int) -> None:
    n_up = n // 2
    configurations = enumerate_sz_configs(n, n_up)
    probability = exp(uniform_sz_log_probability(n, n_up))
    assert len(configurations) == comb(n, n_up)
    assert probability == pytest.approx(1.0 / len(configurations))


def test_uniform_samplers_are_legal_reproducible_and_frequency_calibrated() -> None:
    fusion_key = jax.random.PRNGKey(9187)
    first = np.asarray(sample_uniform_fusion_paths(fusion_key, 28_000, 8, 0))
    second = np.asarray(sample_uniform_fusion_paths(fusion_key, 28_000, 8, 0))
    np.testing.assert_array_equal(first, second)
    assert all(is_valid_path(tuple(row), 0) for row in first)
    fusion_counts = {path: 0 for path in enumerate_paths(8, 0)}
    for row in first:
        fusion_counts[tuple(int(value) for value in row)] += 1
    fusion_frequencies = np.asarray(tuple(fusion_counts.values())) / len(first)
    assert np.max(np.abs(fusion_frequencies - 1.0 / len(fusion_counts))) < 0.008

    sz = np.asarray(sample_uniform_sz_configs(jax.random.PRNGKey(2913), 70_000, 8, 4))
    assert np.all(np.sum(sz, axis=1) == 4)
    sz_counts = {configuration: 0 for configuration in enumerate_sz_configs(8, 4)}
    for row in sz:
        sz_counts[tuple(int(value) for value in row)] += 1
    sz_frequencies = np.asarray(tuple(sz_counts.values())) / len(sz)
    assert np.max(np.abs(sz_frequencies - 1.0 / len(sz_counts))) < 0.003


def test_large_sampler_does_not_enumerate_or_overflow_counts() -> None:
    # The exact singlet multiplicity is far beyond int64, while Python integer
    # completion counts and the sequential sampler remain well defined.
    n = 144
    dimension = fusion_completion_count(1, n - 1, 0)
    assert dimension > 2**63
    paths = np.asarray(
        sample_uniform_fusion_paths(jax.random.PRNGKey(122), 32, n, 0)
    )
    assert paths.shape == (32, n)
    assert all(is_valid_path(tuple(row), 0) for row in paths)
    configs = np.asarray(
        sample_uniform_sz_configs(jax.random.PRNGKey(133), 32, n, n // 2)
    )
    assert configs.shape == (32, n)
    assert np.all(np.sum(configs, axis=1) == n // 2)


def test_validation_is_fail_closed() -> None:
    with pytest.raises(TypeError, match="integer"):
        fusion_completion_count(1.0, 3, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unreachable"):
        sample_uniform_fusion_paths(jax.random.PRNGKey(0), 2, 5, 0)
    with pytest.raises(ValueError, match="must not exceed"):
        sample_uniform_sz_configs(jax.random.PRNGKey(0), 2, 4, 5)
