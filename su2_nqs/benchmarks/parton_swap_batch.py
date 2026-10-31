"""Device-scanned batching of the sealed closed-shell swap evaluator.

Only scheduling changes: the permutation bank, low-rank determinant formula,
condition limit, and direct fallback are inherited without approximation.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .parton_swap_actions import ClosedShellSwapEvaluator
from su2_nqs.operators.projected_permutations import evaluate_swap_bank


class ScannedClosedShellSwapEvaluator(ClosedShellSwapEvaluator):
    """Batch configurations and scan all word chunks inside one compiled call."""

    def __init__(self, *args, sample_batch_size: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if sample_batch_size < 1:
            raise ValueError("positive sample batch size required")
        self.sample_batch_size = sample_batch_size
        chunks = (len(self.c) + self.chunk_size - 1) // self.chunk_size
        def padded(value):
            a = np.zeros((chunks * self.chunk_size, *value.shape[1:]), dtype=value.dtype)
            a[:len(value)] = value
            return jnp.asarray(a.reshape(chunks, self.chunk_size, *value.shape[1:]))
        self.tables = tuple(padded(a) for a in (self.left, self.right, self.c))
        batched = jax.vmap(self.kernel, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))

        @jax.jit
        def summed(state, up_slot, down_slot, up_ratio, down_ratio, base, tables):
            def body(total, words):
                return total + batched(state, up_slot, down_slot, up_ratio,
                                       down_ratio, base, *words), None
            return jax.lax.scan(body, jnp.zeros((len(state), self.c.shape[1])), tables)[0]
        self.summed = summed

    def __call__(self, configurations: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return the same exact images as the reference, including fallbacks."""
        states = np.asarray(configurations)
        n = self.v.shape[1]
        half = n // 2
        if states.ndim != 2 or not len(states) or states.shape[1] != n or np.any((states != 0) & (states != 1)) or np.any(states.sum(axis=1) != half):
            raise ValueError("nonempty half-filled bit configurations required")
        result = np.zeros((len(states), self.c.shape[1]), dtype=complex)
        prepared, positions = [], []
        fallback, maximum = 0, 0.
        for index, state in enumerate(states):
            up, down = np.flatnonzero(state), np.flatnonzero(1-state)
            u, d = self.v[:, up], self.v[:, down]
            condition = float(max(np.max(np.linalg.cond(u)), np.max(np.linalg.cond(d))))
            maximum = max(maximum, condition)
            if not np.isfinite(condition) or condition > self.condition_limit:
                observed, _ = evaluate_swap_bank(state[None], self.direct, self.masks, self.c)
                result[index] = observed[0]
                fallback += 1
                continue
            up_slot, down_slot = np.zeros(n, np.int32), np.zeros(n, np.int32)
            up_slot[up], down_slot[down] = np.arange(half), np.arange(half)
            base = (-1.)**(int(up.sum()) % 2) * np.linalg.det(u) * np.linalg.det(d) * self.w
            prepared.append((state, up_slot, down_slot, self.v @ np.linalg.inv(u), self.v @ np.linalg.inv(d), base))
            positions.append(index)
        for start in range(0, len(prepared), self.sample_batch_size):
            rows = prepared[start:start+self.sample_batch_size]
            count = len(rows)
            rows += [rows[0]] * (self.sample_batch_size-count)
            values = np.asarray(self.summed(*(np.asarray(x) for x in zip(*rows)), self.tables))
            result[positions[start:start+count]] = values[:count]
        return result, {"sample_count": len(states), "word_count": len(self.c),
                        "direct_fallback_count": fallback, "maximum_component_condition": maximum}
