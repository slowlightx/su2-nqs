"""Exact-small schedule comparison for conditional scalar-operator attention.

All arms use the same initialization and number of exact objective updates.
Attention is frozen only during the registered warmup; optimizer state is
not reset. This is not a samples/full-H-call/GPU-hour production benchmark.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..models.reduced_channel_attention import init_channel_attention, channel_attention_coefficients
from ..vmc.adam import _manual_adam_init, _manual_adam_update
from ..vmc.checkpoint import save_checkpoint, load_checkpoint


def train_channel_schedule(
    carrier: np.ndarray, images: np.ndarray, hamiltonian: np.ndarray,
    contexts: np.ndarray, coordinates: np.ndarray, *, seed: int, width: int,
    steps: int, warmup: int, learning_rate: float, schedule: str,
    checkpoint: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Optimize a carrier-nested residual; return final (not best-step) state."""
    if schedule not in ("joint", "staged", "local_only"):
        raise ValueError("unknown schedule")
    if not 0 <= warmup < steps or learning_rate <= 0:
        raise ValueError("invalid training budget")
    jax.config.update("jax_enable_x64", True)
    params = init_channel_attention(jax.random.PRNGKey(seed), width)
    optimizer = _manual_adam_init(params)
    base, bank, h = map(jnp.asarray, (carrier, images, hamiltonian))
    labels, positions = jnp.asarray(contexts), jnp.asarray(coordinates)

    def wavefunction(p):
        v = base + channel_attention_coefficients(p, labels, positions) @ bank
        return v / jnp.linalg.norm(v)

    def objective(p):
        v = wavefunction(p)
        return jnp.vdot(v, h @ v).real

    @jax.jit
    def update(p, state, train_attention):
        energy, grad = jax.value_and_grad(objective)(p)
        grad = dict(grad)
        grad["attention"] = jax.tree_util.tree_map(lambda x: x * train_attention, grad["attention"])
        new, state = _manual_adam_update(p, grad, state, learning_rate)
        return new, state, energy

    @jax.jit
    def metrics(p):
        v = wavefunction(p)
        hv = h @ v
        e = jnp.vdot(v, hv).real
        return e, jnp.vdot(hv - e * v, hv - e * v).real

    history = []
    started = time.perf_counter()
    for step in range(steps + 1):
        if step == 0 or step % 10 == 0 or step == warmup or step == steps:
            e, variance = metrics(params)
            history.append({"step": step, "energy_per_site": float(e) / 16, "variance_total": float(variance)})
        if step == steps:
            break
        enabled = schedule == "joint" or (schedule == "staged" and step >= warmup)
        params, optimizer, _ = update(params, optimizer, jnp.asarray(float(enabled)))
    state = np.asarray(wavefunction(params))
    checkpoint_error = None
    if checkpoint is not None:
        save_checkpoint(checkpoint, params, optimizer_state=optimizer, metadata={"steps": steps, "seed": seed, "schedule": schedule, "history": history})
        replay = np.asarray(wavefunction(load_checkpoint(checkpoint)["params"]))
        checkpoint_error = float(np.max(np.abs(replay - state)))
    return state, {
        "schedule": schedule, "seed": seed, "steps": steps, "warmup": warmup,
        "parameters_allocated": sum(int(x.size) for x in jax.tree_util.tree_leaves(params)),
        "parameters_local": 4,
        "history": history,
        "wall_seconds_including_compile": time.perf_counter() - started,
        "checkpoint_replay_max_error": checkpoint_error,
        "final_coefficients": np.asarray(channel_attention_coefficients(params, labels, positions)).tolist(),
    }
