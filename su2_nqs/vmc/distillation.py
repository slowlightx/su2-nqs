"""Sampled projection of local imaginary-time targets into normalized NQS."""

from __future__ import annotations

from collections.abc import Callable
import math

import jax
import jax.numpy as jnp


def widen_gru_hidden(params: dict[str, object], new_hidden_dim: int) -> dict[str, object]:
    """Embed a trained GRU exactly into a wider hidden space.

    Extra units start decoupled, so the represented wavefunction is unchanged.
    A subsequent optimizer step can activate them through gradients of the
    widened recurrent/input and head matrices.
    """

    old = int(params["h0"].shape[0])
    if new_hidden_dim < old:
        raise ValueError("new hidden dimension cannot be smaller")
    if new_hidden_dim == old:
        return params
    pad = new_hidden_dim - old
    result = {"embeddings": params["embeddings"]}
    gru = {}
    for name, value in params["gru"].items():
        if name.startswith("w"):
            gru[name] = jnp.pad(value, ((0, 0), (0, pad)))
        elif name.startswith("u"):
            gru[name] = jnp.pad(value, ((0, pad), (0, pad)))
        else:
            gru[name] = jnp.pad(value, ((0, pad),))
    result["gru"] = gru
    result["h0"] = jnp.pad(params["h0"], ((0, pad),))
    for head in ("probability", "phase"):
        extra_head = 1e-3 * jnp.sin(
            jnp.arange(pad * params[head]["w"].shape[1], dtype=jnp.float64)
        ).reshape(pad, params[head]["w"].shape[1])
        result[head] = {
            "w": jnp.concatenate((params[head]["w"], extra_head), axis=0),
            "b": params[head]["b"],
        }
    return result


def sampled_distillation_loss(
    params: object,
    samples: jax.Array,
    importance_weights: jax.Array,
    target_phases: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    *,
    phase_weight: float = 1.0,
    base_logs: jax.Array | None = None,
    target_log_ratios: jax.Array | None = None,
    residual_weight: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Weighted Born cross-entropy plus global-phase-invariant phase loss.

    Samples are drawn from the current/base normalized NQS.  The nonnegative
    weights are ``|psi_target / psi_base|**2``.  ``target_phases`` are phases
    of the complex target amplitudes on the same sampled configurations.
    """

    logs = jax.vmap(lambda sample: log_psi_fn(params, sample))(samples)
    weights = jnp.asarray(importance_weights)
    weights = weights / jnp.maximum(jnp.sum(weights), jnp.finfo(weights.dtype).tiny)
    magnitude_loss = -jnp.sum(weights * (2.0 * jnp.real(logs)))
    phase_alignment = jnp.sum(
        weights * jnp.exp(1j * (jnp.imag(logs) - target_phases))
    )
    phase_loss = 1.0 - jnp.abs(phase_alignment)
    residual_loss = jnp.asarray(0.0, dtype=magnitude_loss.dtype)
    if base_logs is not None and target_log_ratios is not None:
        residual = (logs - base_logs) - target_log_ratios
        # Overall normalization and global phase are physically irrelevant.
        residual -= jnp.sum(weights * residual)
        residual_loss = jnp.sum(weights * jnp.abs(residual) ** 2)
    loss = magnitude_loss + phase_weight * phase_loss + residual_weight * residual_loss
    return loss, {
        "magnitude_loss": magnitude_loss,
        "phase_loss": phase_loss,
        "phase_coherence": jnp.abs(phase_alignment),
        "residual_loss": residual_loss,
        "batch_ess_fraction": 1.0
        / (samples.shape[0] * jnp.sum(weights**2)),
    }


def exact_fidelity_distillation_loss(
    params: object,
    basis: jax.Array,
    target: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Negative log fidelity to a complete-sector target.

    Both the student and target are normalized internally.  The loss is
    invariant under either global phase and couples magnitude and phase
    without introducing a coordinate-dependent pointwise log target.
    """

    states = jnp.asarray(basis)
    desired = jnp.asarray(target)
    if states.ndim < 1 or desired.ndim != 1 or states.shape[0] != desired.shape[0]:
        raise ValueError("basis and target must have the same nonzero leading size")
    logs = jax.vmap(lambda state: log_psi_fn(params, state))(states)
    student = jnp.exp(logs)
    student_norm = jnp.real(jnp.vdot(student, student))
    target_norm = jnp.real(jnp.vdot(desired, desired))
    tiny = jnp.finfo(jnp.real(student).dtype).tiny
    normalized_overlap = jnp.vdot(desired, student) / jnp.sqrt(
        jnp.maximum(student_norm * target_norm, tiny)
    )
    fidelity = jnp.clip(jnp.abs(normalized_overlap) ** 2, 0.0, 1.0)
    target_probability = jnp.abs(desired) ** 2 / jnp.maximum(target_norm, tiny)
    student_probability = jnp.abs(student) ** 2 / jnp.maximum(student_norm, tiny)
    born_kl = jnp.sum(
        jnp.where(
            target_probability > 0.0,
            target_probability
            * (
                jnp.log(jnp.maximum(target_probability, tiny))
                - jnp.log(jnp.maximum(student_probability, tiny))
            ),
            0.0,
        )
    )
    return -jnp.log(jnp.maximum(fidelity, tiny)), {
        "fidelity_squared": fidelity,
        "infidelity": 1.0 - fidelity,
        "student_normalization": student_norm,
        "target_normalization": target_norm,
        "target_to_student_born_kl": born_kl,
    }


def exact_graph_fidelity_distillation_loss(
    params: object,
    basis: jax.Array,
    target: jax.Array,
    hamiltonian: jax.Array,
    graph_scale: float,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Negative fidelity in a Hamiltonian graph norm.

    The positive metric is

    ``G = I + (H - E_target)**2 / graph_scale**2``.

    It gives energy-sensitive spectral errors their physical cost while
    preserving a unique zero at the target ray.  Choosing ``graph_scale`` as
    the target energy standard deviation introduces no fitted loss weight.
    """

    scale = float(graph_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("graph_scale must be positive and finite")
    states = jnp.asarray(basis)
    desired = jnp.asarray(target)
    matrix = jnp.asarray(hamiltonian)
    if states.ndim < 1 or desired.ndim != 1 or states.shape[0] != desired.shape[0]:
        raise ValueError("basis and target must have the same nonzero leading size")
    if matrix.shape != (desired.shape[0], desired.shape[0]):
        raise ValueError("hamiltonian shape must match the target sector")
    logs = jax.vmap(lambda state: log_psi_fn(params, state))(states)
    student = jnp.exp(logs)
    target_norm = jnp.real(jnp.vdot(desired, desired))
    student_norm = jnp.real(jnp.vdot(student, student))
    tiny = jnp.finfo(jnp.real(student).dtype).tiny
    target_energy = jnp.real(jnp.vdot(desired, matrix @ desired)) / jnp.maximum(
        target_norm, tiny
    )

    def apply_graph(vector):
        shifted = matrix @ vector - target_energy * vector
        return vector + (
            matrix @ shifted - target_energy * shifted
        ) / scale**2

    graph_target = apply_graph(desired)
    graph_student = apply_graph(student)
    graph_overlap = jnp.vdot(desired, graph_student)
    graph_target_norm = jnp.real(jnp.vdot(desired, graph_target))
    graph_student_norm = jnp.real(jnp.vdot(student, graph_student))
    graph_fidelity = jnp.clip(
        jnp.abs(graph_overlap) ** 2
        / jnp.maximum(graph_target_norm * graph_student_norm, tiny),
        0.0,
        1.0,
    )
    ordinary_overlap = jnp.vdot(desired, student)
    ordinary_fidelity = jnp.clip(
        jnp.abs(ordinary_overlap) ** 2
        / jnp.maximum(target_norm * student_norm, tiny),
        0.0,
        1.0,
    )
    return -jnp.log(jnp.maximum(graph_fidelity, tiny)), {
        "fidelity_squared": ordinary_fidelity,
        "infidelity": 1.0 - ordinary_fidelity,
        "graph_fidelity_squared": graph_fidelity,
        "graph_infidelity": 1.0 - graph_fidelity,
        "student_normalization": student_norm,
        "target_normalization": target_norm,
        "graph_scale": jnp.asarray(scale),
    }


def train_exact_fidelity_distillation(
    params: object,
    *,
    basis: jax.Array,
    target: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    steps: int,
    learning_rate: float = 1.0e-3,
    record_every: int = 25,
    hamiltonian: jax.Array | None = None,
    graph_scale: float | None = None,
) -> tuple[object, list[dict[str, float]]]:
    """Fit a normalized autoregressive state to an exact finite-sector target.

    The local Adam implementation keeps this exact-reference path independent
    of optional Optax installation.  It is intended for small-system
    representation gates, not production VMC.
    """

    if steps < 1 or record_every < 1:
        raise ValueError("steps and record_every must be positive")
    rate = float(learning_rate)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    states = jnp.asarray(basis)
    desired = jnp.asarray(target)
    if states.ndim < 1 or desired.ndim != 1 or states.shape[0] != desired.shape[0]:
        raise ValueError("basis and target must have the same nonzero leading size")
    target_norm = jnp.real(jnp.vdot(desired, desired))
    if not bool(jnp.isfinite(target_norm)) or float(target_norm) <= 0.0:
        raise ValueError("target must have positive finite norm")
    if (hamiltonian is None) != (graph_scale is None):
        raise ValueError("hamiltonian and graph_scale must be supplied together")
    graph_matrix = None if hamiltonian is None else jnp.asarray(hamiltonian)
    if graph_matrix is not None and graph_matrix.shape != (
        desired.shape[0],
        desired.shape[0],
    ):
        raise ValueError("hamiltonian shape must match the target sector")

    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)

    def objective(candidate):
        if graph_matrix is None:
            return exact_fidelity_distillation_loss(
                candidate, states, desired, log_psi_fn
            )
        assert graph_scale is not None
        return exact_graph_fidelity_distillation_loss(
            candidate,
            states,
            desired,
            graph_matrix,
            graph_scale,
            log_psi_fn,
        )

    value_gradient = jax.value_and_grad(objective, has_aux=True)

    @jax.jit
    def update(candidate, first_moment, second_moment, step):
        (loss, metrics), gradient = value_gradient(candidate)
        first_moment = jax.tree.map(
            lambda moment, value: 0.9 * moment + 0.1 * value,
            first_moment,
            gradient,
        )
        second_moment = jax.tree.map(
            lambda moment, value: 0.999 * moment + 0.001 * value * value,
            second_moment,
            gradient,
        )
        first_hat = jax.tree.map(
            lambda value: value / (1.0 - 0.9**step), first_moment
        )
        second_hat = jax.tree.map(
            lambda value: value / (1.0 - 0.999**step), second_moment
        )
        candidate = jax.tree.map(
            lambda value, mean, square: value
            - rate * mean / (jnp.sqrt(square) + 1.0e-8),
            candidate,
            first_hat,
            second_hat,
        )
        return candidate, first_moment, second_moment, loss, metrics

    history = []
    for step in range(1, steps + 1):
        params, first, second, loss, metrics = update(
            params, first, second, jnp.asarray(step, dtype=jnp.float64)
        )
        if step == 1 or step % record_every == 0 or step == steps:
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss),
                    **{name: float(value) for name, value in metrics.items()},
                }
            )
    return params, history


def train_sampled_distillation(
    params: object,
    key: jax.Array,
    *,
    sample_pool: jax.Array,
    pool_importance_weights: jax.Array,
    pool_target_phases: jax.Array,
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    steps: int,
    batch_size: int,
    learning_rate: float = 1e-3,
    phase_weight: float = 1.0,
    pool_base_logs: jax.Array | None = None,
    pool_target_log_ratios: jax.Array | None = None,
    residual_weight: float = 0.0,
    residual_sample_pool: jax.Array | None = None,
    residual_base_logs: jax.Array | None = None,
    residual_target_log_ratios: jax.Array | None = None,
) -> tuple[object, list[dict[str, float]]]:
    """Adam fit to a fixed teacher pool drawn from the base autoregressive NQS."""

    if steps < 1 or batch_size < 1 or len(sample_pool) < 1:
        raise ValueError("steps, batch_size, and sample pool must be positive")
    import optax

    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(params)

    if pool_base_logs is None:
        pool_base_logs = jnp.zeros((len(sample_pool),), dtype=jnp.complex128)
    if pool_target_log_ratios is None:
        pool_target_log_ratios = jnp.zeros((len(sample_pool),), dtype=jnp.complex128)
    if residual_sample_pool is None:
        residual_sample_pool = sample_pool
        residual_base_logs = pool_base_logs
        residual_target_log_ratios = pool_target_log_ratios
    assert residual_base_logs is not None and residual_target_log_ratios is not None

    def objective(candidate, batch_samples, batch_weights, batch_phases, batch_base, batch_ratio,
                  residual_samples, residual_base, residual_ratio):
        loss, metrics = sampled_distillation_loss(
            candidate,
            batch_samples,
            batch_weights,
            batch_phases,
            log_psi_fn,
            phase_weight=phase_weight,
            base_logs=batch_base,
            target_log_ratios=batch_ratio,
            residual_weight=0.0,
        )
        residual_logs = jax.vmap(lambda sample: log_psi_fn(candidate, sample))(residual_samples)
        residual = (residual_logs - residual_base) - residual_ratio
        residual -= jnp.mean(residual)
        connected_loss = jnp.mean(jnp.abs(residual) ** 2)
        metrics["connected_residual_loss"] = connected_loss
        return loss + residual_weight * connected_loss, metrics

    value_gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))
    history = []
    pool_size = len(sample_pool)
    for step in range(1, steps + 1):
        key, subkey = jax.random.split(key)
        indices = jax.random.randint(subkey, (batch_size,), 0, pool_size)
        key, residual_key = jax.random.split(key)
        residual_indices = jax.random.randint(
            residual_key, (batch_size,), 0, len(residual_sample_pool)
        )
        (loss, metrics), gradients = value_gradient(
            params,
            sample_pool[indices],
            pool_importance_weights[indices],
            pool_target_phases[indices],
            pool_base_logs[indices],
            pool_target_log_ratios[indices],
            residual_sample_pool[residual_indices],
            residual_base_logs[residual_indices],
            residual_target_log_ratios[residual_indices],
        )
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        if step == 1 or step % 50 == 0 or step == steps:
            history.append({
                "step": float(step),
                "loss": float(loss),
                **{name: float(value) for name, value in metrics.items()},
            })
    return params, history
