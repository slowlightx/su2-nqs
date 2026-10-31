"""Exact constrained-uniform proposal samplers for tail-aware observers.

The samplers in this module do not enumerate either Hilbert space.  Fusion
actions are drawn with probabilities proportional to the exact number of
valid suffix completions; fixed-magnetization bits use the corresponding
hypergeometric conditional.  Python integers hold the completion counts, so
precomputation cannot silently overflow at the larger sizes used by the P8
campaign.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb, log
from numbers import Integral

import jax
import jax.numpy as jnp

from ..su2.fusion_path import FUSION_STEPS, count_paths, is_valid_path
from ..su2.quantum_numbers import validate_q


def _as_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    answer = int(value)
    if answer < 0:
        raise ValueError(f"{name} must be non-negative")
    return answer


@lru_cache(maxsize=None)
def _fusion_completion_count(current_q: int, remaining: int, target_q: int) -> int:
    if remaining == 0:
        return int(current_q == target_q)
    lower = (
        _fusion_completion_count(current_q - 1, remaining - 1, target_q)
        if current_q > 0
        else 0
    )
    upper = _fusion_completion_count(current_q + 1, remaining - 1, target_q)
    return lower + upper


def fusion_completion_count(current_q: int, remaining: int, target_q: int) -> int:
    """Count valid ``+/-1`` suffixes from ``current_q`` to ``target_q``.

    All spins are spin one half and all labels are doubled integers.  The walk
    is constrained to non-negative intermediate total spin.
    """

    current = _as_nonnegative_integer(current_q, "current_q")
    spins_left = _as_nonnegative_integer(remaining, "remaining")
    validate_q(target_q, name="target_q")
    return _fusion_completion_count(current, spins_left, int(target_q))


def uniform_fusion_action_probabilities(
    current_q: int, remaining_after: int, target_q: int
) -> tuple[float, float]:
    """Return exact-completion probabilities in action order ``(-1,+1)``."""

    current = _as_nonnegative_integer(current_q, "current_q")
    remaining = _as_nonnegative_integer(remaining_after, "remaining_after")
    validate_q(target_q, name="target_q")
    counts = tuple(
        fusion_completion_count(current + action, remaining, int(target_q))
        if current + action >= 0
        else 0
        for action in FUSION_STEPS
    )
    total = counts[0] + counts[1]
    if total == 0:
        raise ValueError("target is unreachable after either fusion action")
    return counts[0] / total, counts[1] / total


def uniform_fusion_log_probability(n: int, target_q: int = 0) -> float:
    """Log probability of every valid path in the uniform proposal."""

    size = _as_nonnegative_integer(n, "n")
    if size < 1:
        raise ValueError("n must be positive")
    validate_q(target_q, name="target_q")
    dimension = count_paths(size, int(target_q))
    if dimension == 0:
        raise ValueError("target_q is unreachable")
    return -log(dimension)


def sample_uniform_fusion_paths(
    key: jax.Array,
    n_samples: int,
    n: int,
    target_q: int = 0,
) -> jax.Array:
    """Draw iid uniform left-associated fusion paths without enumeration."""

    samples = _as_nonnegative_integer(n_samples, "n_samples")
    size = _as_nonnegative_integer(n, "n")
    if samples < 1 or size < 1:
        raise ValueError("n_samples and n must be positive")
    validate_q(target_q, name="target_q")
    if count_paths(size, int(target_q)) == 0:
        raise ValueError("target_q is unreachable")

    current = jnp.ones((samples,), dtype=jnp.int32)
    labels = [current]
    for decision in range(size - 1):
        key, subkey = jax.random.split(key)
        remaining_after = size - decision - 2
        probability_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
        plus_table = jnp.asarray(
            [
                (
                    uniform_fusion_action_probabilities(
                        q, remaining_after, int(target_q)
                    )[1]
                    if fusion_completion_count(q, remaining_after + 1, int(target_q))
                    else 0.0
                )
                for q in range(size + 1)
            ],
            dtype=probability_dtype,
        )
        take_plus = jax.random.uniform(
            subkey, shape=(samples,), dtype=probability_dtype
        ) < plus_table[current]
        current = current + jnp.where(take_plus, 1, -1).astype(jnp.int32)
        labels.append(current)
    return jnp.stack(labels, axis=1)


def uniform_sz_log_probability(n: int, n_up: int) -> float:
    """Log probability of each bitstring in a fixed-magnetization proposal."""

    size = _as_nonnegative_integer(n, "n")
    up = _as_nonnegative_integer(n_up, "n_up")
    if size < 1:
        raise ValueError("n must be positive")
    if up > size:
        raise ValueError("n_up must not exceed n")
    return -log(comb(size, up))


def sample_uniform_sz_configs(
    key: jax.Array,
    n_samples: int,
    n: int,
    n_up: int,
) -> jax.Array:
    """Draw iid uniform fixed-magnetization bitstrings without enumeration."""

    samples = _as_nonnegative_integer(n_samples, "n_samples")
    size = _as_nonnegative_integer(n, "n")
    up_target = _as_nonnegative_integer(n_up, "n_up")
    if samples < 1 or size < 1:
        raise ValueError("n_samples and n must be positive")
    if up_target > size:
        raise ValueError("n_up must not exceed n")

    up_so_far = jnp.zeros((samples,), dtype=jnp.int32)
    bits = []
    for site in range(size):
        key, subkey = jax.random.split(key)
        sites_remaining = size - site
        probability_up = (up_target - up_so_far) / float(sites_remaining)
        bit = jax.random.bernoulli(
            subkey, probability_up, shape=(samples,)
        ).astype(jnp.int32)
        bits.append(bit)
        up_so_far = up_so_far + bit
    return jnp.stack(bits, axis=1)


def uniform_fusion_path_probability(
    path: tuple[int, ...], target_q: int = 0
) -> float:
    """Evaluate the sequential uniform-proposal probability of one path."""

    if not is_valid_path(path, target_q):
        return 0.0
    probability = 1.0
    n = len(path)
    for decision, (left, right) in enumerate(zip(path, path[1:])):
        action = right - left
        probabilities = uniform_fusion_action_probabilities(
            left, n - decision - 2, target_q
        )
        probability *= probabilities[FUSION_STEPS.index(action)]
    return probability


__all__ = [
    "fusion_completion_count",
    "sample_uniform_fusion_paths",
    "sample_uniform_sz_configs",
    "uniform_fusion_action_probabilities",
    "uniform_fusion_log_probability",
    "uniform_fusion_path_probability",
    "uniform_sz_log_probability",
]
