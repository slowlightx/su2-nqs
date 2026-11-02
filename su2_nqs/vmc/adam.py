"""Adam optimization with Optax when available and a tested local fallback."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time

import jax
import jax.numpy as jnp

from .checkpoint import save_checkpoint
from .local_energy import energy_and_variance
from .sampler import draw_samples
from .gradients import vmc_loss


def _manual_adam_init(params: object) -> tuple[int, object, object]:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return 0, zeros, zeros


def _manual_adam_update(
    params: object,
    gradients: object,
    state: tuple[int, object, object],
    learning_rate: float,
) -> tuple[object, tuple[int, object, object]]:
    step, first, second = state
    step += 1
    first = jax.tree_util.tree_map(lambda m, g: 0.9 * m + 0.1 * g, first, gradients)
    second = jax.tree_util.tree_map(lambda v, g: 0.999 * v + 0.001 * g * g, second, gradients)
    first_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - 0.9**step), first)
    second_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - 0.999**step), second)
    params = jax.tree_util.tree_map(
        lambda value, m, v: value - learning_rate * m / (jnp.sqrt(v) + 1e-8),
        params,
        first_hat,
        second_hat,
    )
    return params, (step, first, second)


def _gradient_norm(gradients: object) -> float:
    leaves = jax.tree_util.tree_leaves(gradients)
    return float(jnp.sqrt(sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in leaves)))


def train_adam(
    params: object,
    key: jax.Array,
    *,
    sample_fn: Callable[[object, jax.Array, int], jax.Array],
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    get_conn_fn: Callable,
    steps: int,
    batch_size: int,
    learning_rate: float = 1e-3,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 100,
    callback: Callable[[int, dict[str, float], object], None] | None = None,
) -> tuple[object, list[dict[str, float]]]:
    """Train with exact autoregressive samples and Adam.

    Optax is used when the optional ``train`` dependencies are installed.  A
    numerically equivalent local Adam implementation keeps the core library
    usable in minimal JAX environments.
    """

    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if checkpoint_path is not None and checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    try:
        import optax  # type: ignore
    except ImportError:
        optax = None

    if optax is not None:
        optimizer = optax.adam(learning_rate)
        optimizer_state = optimizer.init(params)
    else:
        optimizer = None
        optimizer_state = _manual_adam_init(params)

    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        samples, key = draw_samples(sample_fn, params, key, batch_size)
        loss_function = lambda candidate: vmc_loss(
            candidate, samples, log_psi_fn, get_conn_fn
        )
        energy, gradients = jax.value_and_grad(loss_function)(params)
        if optimizer is not None:
            updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
            params = optax.apply_updates(params, updates)
        else:
            params, optimizer_state = _manual_adam_update(
                params, gradients, optimizer_state, learning_rate
            )
        measured_energy, variance = energy_and_variance(
            params, samples, log_psi_fn, get_conn_fn
        )
        metrics = {
            "step": float(step),
            "energy": float(measured_energy),
            "variance": float(variance),
            "gradient_norm": _gradient_norm(gradients),
        }
        history.append(metrics)
        if callback is not None:
            callback(step, metrics, params)
        if checkpoint_path is not None and (
            step % checkpoint_every == 0 or step == steps
        ):
            save_checkpoint(
                checkpoint_path,
                params,
                optimizer_state=optimizer_state,
                metadata={"step": step, "history": history},
            )
    return params, history


def train_exact_adam(
    params: object,
    *,
    loss_fn: Callable[[object], jax.Array],
    steps: int,
    learning_rate: float = 1.0e-3,
    record_every: int | None = None,
) -> tuple[object, dict[str, object]]:
    """Minimize a differentiable exact-support loss with the shared Adam.

    This calibration helper is for small exhaustive objectives.  It uses the
    same Optax/local-fallback update implementation as :func:`train_adam` and
    reports compilation separately from optimizer wall time.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if record_every is None:
        record_every = steps
    if record_every < 1:
        raise ValueError("record_every must be positive")
    try:
        import optax  # type: ignore
    except ImportError:
        optax = None

    if optax is not None:
        optimizer = optax.adam(learning_rate)
        optimizer_state = optimizer.init(params)
    else:
        optimizer = None
        optimizer_state = _manual_adam_init(params)
    loss_and_gradient = jax.jit(jax.value_and_grad(loss_fn))
    compile_started = time.perf_counter()
    initial_loss, _ = loss_and_gradient(params)
    jax.block_until_ready(initial_loss)
    compile_seconds = time.perf_counter() - compile_started

    history: list[dict[str, float]] = []
    training_started = time.perf_counter()
    final_loss = initial_loss
    for step in range(1, steps + 1):
        final_loss, gradients = loss_and_gradient(params)
        if optimizer is not None:
            updates, optimizer_state = optimizer.update(
                gradients, optimizer_state, params
            )
            params = optax.apply_updates(params, updates)
        else:
            params, optimizer_state = _manual_adam_update(
                params, gradients, optimizer_state, learning_rate
            )
        if step % record_every == 0 or step == steps:
            loss_value = float(final_loss)
            gradient_norm = _gradient_norm(gradients)
            history.append(
                {
                    "step": float(step),
                    "loss_before_update": loss_value,
                    "gradient_norm": gradient_norm,
                    "cumulative_training_seconds": (
                        time.perf_counter() - training_started
                    ),
                }
            )
    # The final loss is evaluated before the last update.  Synchronize both it
    # and the returned parameter leaves so optimizer timing cannot leak into a
    # caller's subsequent observable evaluation.
    jax.block_until_ready(final_loss)
    for leaf in jax.tree_util.tree_leaves(params):
        jax.block_until_ready(leaf)
    training_seconds = time.perf_counter() - training_started
    report: dict[str, object] = {
        "initial_loss": float(initial_loss),
        "final_loss_before_last_update": float(final_loss),
        "compile_seconds": compile_seconds,
        "training_seconds": training_seconds,
        "history": history,
    }
    return params, report


__all__ = ["train_adam", "train_exact_adam"]
