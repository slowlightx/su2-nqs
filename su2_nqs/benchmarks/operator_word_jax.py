"""Batched JAX evaluation of exact scalar-word action plans.

Discrete spin connectivity is built once on the host into a sparse immutable
plan.  Carrier amplitudes for the globally deduplicated terminal states and
the sparse reduction are then evaluated in one JAX program.  This removes the
thousands of small host-to-device launches used by the reference evaluator
while preserving the exact operator convention and coefficients.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.triangular_hnf import IntMatrix, validate_hnf
from ..operators.triangular_a1_scalar import (
    TRIANGULAR_A1_SCALAR_LABELS,
    triangular_a1_scalar_token_connectivity,
)


ScalarWord = tuple[str, ...]
JaxBatchAmplitude = Callable[[jax.Array], jax.Array]


@dataclass(frozen=True)
class FixedCapacityScalarWordKernel:
    """One reusable JIT executable for many sparse plans of fixed capacity."""

    n_sites: int
    amplitude_batch_size: int
    terminal_capacity: int
    entry_capacity: int
    output_count: int
    _kernel: Callable[..., jax.Array] = field(repr=False, compare=False)


@dataclass(frozen=True)
class ScalarWordActionPlan:
    """Sparse terminal-amplitude contraction for several centers and words."""

    terminal_configurations: np.ndarray
    entry_terminal_indices: np.ndarray
    entry_output_indices: np.ndarray
    entry_coefficients: np.ndarray
    sample_count: int
    word_count: int
    maximum_word_depth: int
    raw_terminal_requests: int

    def __post_init__(self) -> None:
        terminals = np.asarray(self.terminal_configurations)
        positions = np.asarray(self.entry_terminal_indices)
        outputs = np.asarray(self.entry_output_indices)
        coefficients = np.asarray(self.entry_coefficients)
        if terminals.ndim != 2 or terminals.shape[0] < 1:
            raise ValueError("terminal_configurations must be nonempty and rank two")
        if any(array.ndim != 1 for array in (positions, outputs, coefficients)):
            raise ValueError("sparse plan entries must be rank-one arrays")
        if not (len(positions) == len(outputs) == len(coefficients)):
            raise ValueError("sparse plan entry arrays must have equal length")
        if len(positions) < 1 or np.any(positions < 0) or np.any(positions >= len(terminals)):
            raise ValueError("terminal indices are out of range")
        output_count = int(self.sample_count) * int(self.word_count)
        if output_count < 1 or np.any(outputs < 0) or np.any(outputs >= output_count):
            raise ValueError("output indices are out of range")
        if np.any(~np.isfinite(coefficients)):
            raise FloatingPointError("sparse plan coefficients must be finite")


def _checked_words(words: Sequence[Sequence[str]]) -> tuple[ScalarWord, ...]:
    checked = tuple(tuple(str(label) for label in word) for word in words)
    if not checked:
        raise ValueError("at least one scalar-operator word is required")
    allowed = set(TRIANGULAR_A1_SCALAR_LABELS)
    if any(label not in allowed for word in checked for label in word):
        raise ValueError("scalar-operator word contains an unknown token")
    return checked


def build_triangular_a1_word_action_plan(
    configurations: np.ndarray,
    supercell: IntMatrix,
    words: Sequence[Sequence[str]],
) -> tuple[ScalarWordActionPlan, dict[str, int | float]]:
    """Build a globally deduplicated sparse action plan on the host."""

    started = time.perf_counter()
    matrix = validate_hnf(supercell)
    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("configurations must be a nonempty rank-two array")
    checked_words = _checked_words(words)
    union: dict[tuple[int, ...], int] = {}
    entry_terminal_indices: list[int] = []
    entry_output_indices: list[int] = []
    entry_coefficients: list[float] = []
    raw_requests = 0

    for row_index, state_array in enumerate(states):
        center = tuple(int(value) for value in state_array)
        connectivity_cache: dict[
            tuple[int, ...],
            dict[str, tuple[tuple[tuple[int, ...], ...], np.ndarray]],
        ] = {}

        def expand(word: ScalarWord) -> dict[tuple[int, ...], float]:
            frontier = {center: 1.0}
            for label in word:
                following: dict[tuple[int, ...], float] = {}
                for current, coefficient in frontier.items():
                    tokens = connectivity_cache.get(current)
                    if tokens is None:
                        tokens = triangular_a1_scalar_token_connectivity(current, matrix)
                        connectivity_cache[current] = tokens
                    connected, elements = tokens[label]
                    for target, element in zip(connected, elements, strict=True):
                        following[target] = following.get(target, 0.0) + float(
                            coefficient * element
                        )
                frontier = {
                    target: coefficient
                    for target, coefficient in following.items()
                    if abs(coefficient) > 1.0e-14
                }
            return frontier

        for word_index, word in enumerate(checked_words):
            expansion = expand(word)
            raw_requests += len(expansion)
            output_index = row_index * len(checked_words) + word_index
            for terminal, coefficient in expansion.items():
                position = union.get(terminal)
                if position is None:
                    position = len(union)
                    union[terminal] = position
                entry_terminal_indices.append(position)
                entry_output_indices.append(output_index)
                entry_coefficients.append(float(coefficient))

    plan = ScalarWordActionPlan(
        terminal_configurations=np.asarray(tuple(union), dtype=np.int8),
        entry_terminal_indices=np.asarray(entry_terminal_indices, dtype=np.int32),
        entry_output_indices=np.asarray(entry_output_indices, dtype=np.int32),
        entry_coefficients=np.asarray(entry_coefficients, dtype=np.float64),
        sample_count=len(states),
        word_count=len(checked_words),
        maximum_word_depth=max(map(len, checked_words)),
        raw_terminal_requests=raw_requests,
    )
    unique = len(plan.terminal_configurations)
    return plan, {
        "sample_count": len(states),
        "word_count": len(checked_words),
        "maximum_word_depth": plan.maximum_word_depth,
        "raw_terminal_configuration_requests": raw_requests,
        "unique_terminal_amplitude_evaluations": unique,
        "terminal_deduplication_fraction": 1.0 - unique / max(1, raw_requests),
        "sparse_contraction_entries": len(plan.entry_coefficients),
        "host_plan_wall_seconds": time.perf_counter() - started,
    }


def evaluate_scalar_word_action_plan_numpy(
    plan: ScalarWordActionPlan,
    amplitude_fn: Callable[[np.ndarray], np.ndarray],
    *,
    amplitude_batch_size: int,
) -> np.ndarray:
    """Reference NumPy execution of a sparse plan."""

    if amplitude_batch_size < 1:
        raise ValueError("amplitude_batch_size must be positive")
    values = []
    for start in range(0, len(plan.terminal_configurations), amplitude_batch_size):
        states = plan.terminal_configurations[start : start + amplitude_batch_size]
        observed = np.asarray(amplitude_fn(states), dtype=np.complex128)
        if observed.shape != (len(states),) or np.any(~np.isfinite(observed)):
            raise FloatingPointError("amplitude_fn returned invalid terminal amplitudes")
        values.append(observed)
    amplitudes = np.concatenate(values)
    actions = np.zeros(plan.sample_count * plan.word_count, dtype=np.complex128)
    np.add.at(
        actions,
        plan.entry_output_indices,
        plan.entry_coefficients * amplitudes[plan.entry_terminal_indices],
    )
    return actions.reshape((plan.sample_count, plan.word_count))


def evaluate_scalar_word_action_plan_jax(
    plan: ScalarWordActionPlan,
    amplitude_fn: JaxBatchAmplitude,
    *,
    amplitude_batch_size: int,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """Evaluate one plan in one JAX program.

    This convenience path compiles for the exact plan capacities.  Production
    loops should use :func:`make_fixed_capacity_scalar_word_kernel` and
    :func:`evaluate_fixed_capacity_scalar_word_kernel` so compilation is
    amortized across all sampled centers.
    """

    if amplitude_batch_size < 1:
        raise ValueError("amplitude_batch_size must be positive")
    output_count = plan.sample_count * plan.word_count
    terminal_capacity = (
        (len(plan.terminal_configurations) + amplitude_batch_size - 1)
        // amplitude_batch_size
    ) * amplitude_batch_size
    compiled = make_fixed_capacity_scalar_word_kernel(
        amplitude_fn,
        n_sites=plan.terminal_configurations.shape[1],
        amplitude_batch_size=amplitude_batch_size,
        terminal_capacity=terminal_capacity,
        entry_capacity=len(plan.entry_coefficients),
        output_count=output_count,
    )
    return evaluate_fixed_capacity_scalar_word_kernel(compiled, plan)


def make_fixed_capacity_scalar_word_kernel(
    amplitude_fn: JaxBatchAmplitude,
    *,
    n_sites: int,
    amplitude_batch_size: int,
    terminal_capacity: int,
    entry_capacity: int,
    output_count: int,
) -> FixedCapacityScalarWordKernel:
    """Create one JIT kernel reusable across plans below registered capacities."""

    if n_sites < 1 or amplitude_batch_size < 1:
        raise ValueError("n_sites and amplitude_batch_size must be positive")
    if terminal_capacity < 1 or terminal_capacity % amplitude_batch_size != 0:
        raise ValueError("terminal_capacity must be a positive batch-size multiple")
    if entry_capacity < 1 or output_count < 1:
        raise ValueError("entry_capacity and output_count must be positive")

    @jax.jit
    def kernel(
        state_chunks: jax.Array,
        terminal_indices: jax.Array,
        output_indices: jax.Array,
        coefficients: jax.Array,
    ) -> jax.Array:
        def body(_, state_chunk):
            amplitudes = jnp.asarray(amplitude_fn(state_chunk))
            return None, amplitudes

        _, chunk_amplitudes = jax.lax.scan(body, None, state_chunks)
        amplitudes = chunk_amplitudes.reshape((terminal_capacity,))
        contributions = coefficients * amplitudes[terminal_indices]
        result = jnp.zeros((output_count,), dtype=contributions.dtype)
        return result.at[output_indices].add(contributions)

    return FixedCapacityScalarWordKernel(
        n_sites=int(n_sites),
        amplitude_batch_size=int(amplitude_batch_size),
        terminal_capacity=int(terminal_capacity),
        entry_capacity=int(entry_capacity),
        output_count=int(output_count),
        _kernel=kernel,
    )


def evaluate_fixed_capacity_scalar_word_kernel(
    compiled: FixedCapacityScalarWordKernel,
    plan: ScalarWordActionPlan,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """Pad one plan to fixed shapes and run a reusable compiled kernel."""

    terminals = plan.terminal_configurations
    output_count = plan.sample_count * plan.word_count
    if terminals.shape[1] != compiled.n_sites:
        raise ValueError("plan site count does not match the compiled kernel")
    if len(terminals) > compiled.terminal_capacity:
        raise ValueError("plan exceeds the compiled terminal capacity")
    if len(plan.entry_coefficients) > compiled.entry_capacity:
        raise ValueError("plan exceeds the compiled sparse-entry capacity")
    if output_count != compiled.output_count:
        raise ValueError("plan output count does not match the compiled kernel")

    terminal_padding = compiled.terminal_capacity - len(terminals)
    padded_terminals = np.pad(
        terminals, ((0, terminal_padding), (0, 0)), mode="edge"
    ).reshape(
        (-1, compiled.amplitude_batch_size, compiled.n_sites)
    )
    entry_padding = compiled.entry_capacity - len(plan.entry_coefficients)
    terminal_indices = np.pad(
        plan.entry_terminal_indices, (0, entry_padding), constant_values=0
    )
    output_indices = np.pad(
        plan.entry_output_indices, (0, entry_padding), constant_values=0
    )
    coefficients = np.pad(
        plan.entry_coefficients, (0, entry_padding), constant_values=0.0
    )

    started = time.perf_counter()
    actions = compiled._kernel(
        jnp.asarray(padded_terminals, dtype=jnp.int8),
        jnp.asarray(terminal_indices, dtype=jnp.int32),
        jnp.asarray(output_indices, dtype=jnp.int32),
        jnp.asarray(coefficients, dtype=jnp.float64),
    )
    actions.block_until_ready()
    wall = time.perf_counter() - started
    observed = np.asarray(actions).reshape((plan.sample_count, plan.word_count))
    if np.any(~np.isfinite(observed)):
        raise FloatingPointError("JAX scalar-word kernel returned non-finite actions")
    unique_count = len(terminals)
    return observed, {
        "jax_backend": jax.default_backend(),
        "amplitude_batch_size": compiled.amplitude_batch_size,
        "amplitude_batch_count": (
            compiled.terminal_capacity // compiled.amplitude_batch_size
        ),
        "terminal_capacity": compiled.terminal_capacity,
        "entry_capacity": compiled.entry_capacity,
        "padded_terminal_count": compiled.terminal_capacity,
        "unique_terminal_count": unique_count,
        "terminal_capacity_utilization": unique_count / compiled.terminal_capacity,
        "entry_capacity_utilization": (
            len(plan.entry_coefficients) / compiled.entry_capacity
        ),
        "jax_kernel_wall_seconds": wall,
        "unique_terminal_amplitudes_per_second": unique_count / wall,
    }


__all__ = [
    "FixedCapacityScalarWordKernel",
    "ScalarWordActionPlan",
    "build_triangular_a1_word_action_plan",
    "evaluate_fixed_capacity_scalar_word_kernel",
    "evaluate_scalar_word_action_plan_jax",
    "evaluate_scalar_word_action_plan_numpy",
    "make_fixed_capacity_scalar_word_kernel",
]
