"""Exact-support SR for a monomial-symmetry-projected TreeGRU state."""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg

from ..models.compact_tree_gru import (
    CompactTreeGRUConfig,
    compact_tree_log_psi,
    fixed_compact_tree_path_data,
    init_compact_tree_gru,
)
from ..models.branching_tree_gru import (
    branching_tree_log_psi,
    init_branching_tree_gru,
)
from ..models.orbit_quotient_gru import (
    OrbitPushforwardSupport,
    TreeOrbitSupport,
    orbit_pushforward_state,
    orbit_quotient_log_psi,
)
from ..models.hamiltonian_conditioned_tree import (
    HamiltonianConditionedTreeGRUConfig,
    condition_existing_tree_gru,
    fixed_conditioned_path_data,
    hamiltonian_conditioned_tree_log_psi,
)
from .model_evaluation import parameter_count


def train_exact_projected_tree_sr(
    config: CompactTreeGRUConfig,
    full_labels: np.ndarray,
    sector_columns: np.ndarray,
    sector_hamiltonian: np.ndarray,
    target_state: np.ndarray,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    damping: float,
    cg_tolerance: float,
    cg_max_iterations: int,
    record_every: int,
    fidelity_threshold: float,
) -> tuple[object, dict[str, Any], np.ndarray]:
    """Optimize ``B^dagger |psi_theta>`` with a wavefunction-space SR metric.

    ``B`` has orthonormal character-sector columns in the complete tree basis.
    The base TreeGRU remains exactly normalized; projection and its scalar norm
    are evaluated exhaustively here.  This routine is therefore a calibration
    instrument, not yet a scalable quotient-space sampler.
    """

    if steps < 1 or record_every < 1:
        raise ValueError("steps and record_every must be positive")
    labels = jnp.asarray(full_labels, dtype=jnp.int32)
    columns = jnp.asarray(sector_columns, dtype=jnp.complex128)
    hamiltonian = jnp.asarray(sector_hamiltonian, dtype=jnp.complex128)
    target = np.asarray(target_state, dtype=np.complex128)
    if labels.shape != (sector_columns.shape[0], config.n - 1):
        raise ValueError("full_labels do not match the tree sector")
    if sector_columns.ndim != 2 or sector_columns.shape[1] != target.size:
        raise ValueError("sector columns do not match the target")
    if hamiltonian.shape != (target.size, target.size):
        raise ValueError("sector Hamiltonian shape does not match the target")
    target /= np.linalg.norm(target)
    target_jax = jnp.asarray(target)

    params = init_compact_tree_gru(jax.random.PRNGKey(seed), config)
    flat, unravel = ravel_pytree(params)

    def reduced_state(values):
        candidate = unravel(values)
        logs = jax.vmap(
            lambda state: compact_tree_log_psi(candidate, state, config)
        )(labels)
        full = jnp.exp(logs)
        reduced = columns.conj().T @ full
        return reduced / jnp.linalg.norm(reduced)

    initial_state = np.asarray(reduced_state(flat), dtype=np.complex128)
    gauge_index = int(np.argmax(np.abs(initial_state)))

    def energy(values):
        state = reduced_state(values)
        return jnp.real(jnp.vdot(state, hamiltonian @ state))

    def features(values):
        state = reduced_state(values)
        anchor = state[gauge_index]
        gauge_fixed = state * jnp.conj(anchor) / jnp.maximum(
            jnp.abs(anchor), 1.0e-14
        )
        return jnp.sqrt(2.0) * jnp.concatenate(
            (jnp.real(gauge_fixed), jnp.imag(gauge_fixed))
        )

    @jax.jit
    def update(values):
        _, pullback = jax.vjp(features, values)

        def fisher_product(vector):
            _, tangent = jax.jvp(features, (values,), (vector,))
            return pullback(tangent)[0] + damping * vector

        gradient = jax.grad(energy)(values)
        direction, _ = cg(
            fisher_product,
            gradient,
            tol=cg_tolerance,
            maxiter=cg_max_iterations,
        )
        return (
            values - learning_rate * direction,
            jnp.linalg.norm(gradient),
            jnp.linalg.norm(direction),
        )

    def metrics(values) -> dict[str, float]:
        state = np.asarray(reduced_state(values), dtype=np.complex128)
        applied = np.asarray(sector_hamiltonian) @ state
        value = float(np.vdot(state, applied).real)
        variance = float(max(np.vdot(applied, applied).real - value**2, 0.0))
        return {
            "energy": value,
            "variance": variance,
            "fidelity": float(abs(np.vdot(target, state)) ** 2),
            "normalization": float(np.vdot(state, state).real),
            "target_residual": float(
                np.linalg.norm(np.asarray(hamiltonian) @ target_jax - np.vdot(target, np.asarray(hamiltonian) @ target).real * target)
            ),
        }

    started = time.perf_counter()
    history = [{"step": 0, **metrics(flat)}]
    executed_steps = 0
    for step in range(1, steps + 1):
        flat, gradient_norm, direction_norm = update(flat)
        executed_steps = step
        if step % record_every == 0 or step == steps:
            item = {
                "step": step,
                **metrics(flat),
                "gradient_norm": float(gradient_norm),
                "natural_direction_norm": float(direction_norm),
            }
            history.append(item)
            if item["fidelity"] >= fidelity_threshold:
                break
    final_state = np.asarray(reduced_state(flat), dtype=np.complex128)
    final = metrics(flat)
    report = {
        "optimizer": "exact_wavefunction_sr",
        "metric": "gauge_fixed_projected_wavefunction",
        "seed": seed,
        "steps": executed_steps,
        "requested_steps": steps,
        "stopped_early": executed_steps < steps,
        "learning_rate": learning_rate,
        "damping": damping,
        "cg_tolerance": cg_tolerance,
        "cg_max_iterations": cg_max_iterations,
        "fidelity_threshold": fidelity_threshold,
        "fidelity_threshold_reached": final["fidelity"] >= fidelity_threshold,
        "gauge_index": gauge_index,
        "parameter_count": parameter_count(unravel(flat)),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "final": final,
    }
    return unravel(flat), report, final_state


__all__ = ["train_exact_projected_tree_sr"]


def _train_exact_state_adam(
    params: object,
    state_function,
    hamiltonian_array: np.ndarray,
    target_array: np.ndarray,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    record_every: int,
    fidelity_threshold: float,
    optimizer_name: str,
    loss_kind: str = "energy",
) -> tuple[object, dict[str, Any], np.ndarray]:
    """Shared first-order optimizer for an exactly evaluated normalized state."""

    if steps < 1 or record_every < 1:
        raise ValueError("steps and record_every must be positive")
    hamiltonian = jnp.asarray(hamiltonian_array, dtype=jnp.complex128)
    target = np.asarray(target_array, dtype=np.complex128)
    target /= np.linalg.norm(target)
    first = jax.tree.map(jnp.zeros_like, params)
    second = jax.tree.map(jnp.zeros_like, params)

    if loss_kind not in {"energy", "fidelity"}:
        raise ValueError("loss_kind must be 'energy' or 'fidelity'")
    target_jax = jnp.asarray(target)

    def objective(candidate):
        state = state_function(candidate)
        if loss_kind == "energy":
            return jnp.real(jnp.vdot(state, hamiltonian @ state))
        fidelity = jnp.abs(jnp.vdot(target_jax, state)) ** 2
        return -jnp.log(jnp.maximum(fidelity, jnp.finfo(jnp.float64).tiny))

    value_and_gradient = jax.value_and_grad(objective)

    @jax.jit
    def update(candidate, first_moment, second_moment, step):
        value, gradient = value_and_gradient(candidate)
        first_moment = jax.tree.map(
            lambda old, new: 0.9 * old + 0.1 * new,
            first_moment,
            gradient,
        )
        second_moment = jax.tree.map(
            lambda old, new: 0.999 * old + 0.001 * new * new,
            second_moment,
            gradient,
        )
        first_hat = jax.tree.map(
            lambda moment: moment / (1.0 - 0.9**step), first_moment
        )
        second_hat = jax.tree.map(
            lambda moment: moment / (1.0 - 0.999**step), second_moment
        )
        candidate = jax.tree.map(
            lambda value, mean, square: value
            - learning_rate * mean / (jnp.sqrt(square) + 1.0e-8),
            candidate,
            first_hat,
            second_hat,
        )
        gradient_norm = jnp.sqrt(
            sum(jnp.sum(value * value) for value in jax.tree.leaves(gradient))
        )
        return candidate, first_moment, second_moment, value, gradient_norm

    target_energy = float(np.vdot(target, np.asarray(hamiltonian_array) @ target).real)

    def metrics(candidate) -> tuple[dict[str, float], np.ndarray]:
        state = np.asarray(state_function(candidate), dtype=np.complex128)
        applied = np.asarray(hamiltonian_array) @ state
        value = float(np.vdot(state, applied).real)
        variance = float(max(np.vdot(applied, applied).real - value**2, 0.0))
        return (
            {
                "energy": value,
                "energy_error": value - target_energy,
                "variance": variance,
                "fidelity": float(abs(np.vdot(target, state)) ** 2),
                "normalization": float(np.vdot(state, state).real),
            },
            state,
        )

    started = time.perf_counter()
    initial, _ = metrics(params)
    history = [{"step": 0, **initial}]
    executed_steps = 0
    for step in range(1, steps + 1):
        params, first, second, _, gradient_norm = update(
            params, first, second, jnp.asarray(step, dtype=jnp.float64)
        )
        executed_steps = step
        if step % record_every == 0 or step == steps:
            item, _ = metrics(params)
            history.append(
                {"step": step, **item, "gradient_norm": float(gradient_norm)}
            )
            if item["fidelity"] >= fidelity_threshold:
                break
    final, final_state = metrics(params)
    report = {
        "optimizer": optimizer_name,
        "loss_kind": loss_kind,
        "seed": seed,
        "steps": executed_steps,
        "requested_steps": steps,
        "stopped_early": executed_steps < steps,
        "learning_rate": learning_rate,
        "fidelity_threshold": fidelity_threshold,
        "fidelity_threshold_reached": final["fidelity"] >= fidelity_threshold,
        "parameter_count": parameter_count(params),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "final": final,
    }
    return params, report, final_state


def train_exact_projected_tree_adam(
    config: CompactTreeGRUConfig,
    full_labels: np.ndarray,
    sector_columns: np.ndarray,
    sector_hamiltonian: np.ndarray,
    target_state: np.ndarray,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    record_every: int,
    fidelity_threshold: float,
) -> tuple[object, dict[str, Any], np.ndarray]:
    """Adam control for the exact-support projected TreeGRU state."""

    labels = jnp.asarray(full_labels, dtype=jnp.int32)
    columns = jnp.asarray(sector_columns, dtype=jnp.complex128)
    if labels.shape != (sector_columns.shape[0], config.n - 1):
        raise ValueError("full_labels do not match the tree sector")
    if sector_columns.ndim != 2 or sector_columns.shape[1] != len(target_state):
        raise ValueError("sector columns do not match the target")

    def reduced_state(candidate):
        logs = jax.vmap(
            lambda state: compact_tree_log_psi(candidate, state, config)
        )(labels)
        reduced = columns.conj().T @ jnp.exp(logs)
        return reduced / jnp.linalg.norm(reduced)

    params = init_compact_tree_gru(jax.random.PRNGKey(seed), config)
    return _train_exact_state_adam(
        params,
        reduced_state,
        sector_hamiltonian,
        target_state,
        seed=seed,
        steps=steps,
        learning_rate=learning_rate,
        record_every=record_every,
        fidelity_threshold=fidelity_threshold,
        optimizer_name="exact_projected_adam",
    )


def train_exact_projected_tree_fidelity_adam(
    config: CompactTreeGRUConfig,
    full_labels: np.ndarray,
    sector_columns: np.ndarray,
    sector_hamiltonian: np.ndarray,
    target_state: np.ndarray,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    record_every: int,
    fidelity_threshold: float,
) -> tuple[object, dict[str, Any], np.ndarray]:
    """Diagnose projected-state capacity by optimizing exact fidelity."""

    labels = jnp.asarray(full_labels, dtype=jnp.int32)
    columns = jnp.asarray(sector_columns, dtype=jnp.complex128)
    if labels.shape != (sector_columns.shape[0], config.n - 1):
        raise ValueError("full_labels do not match the tree sector")

    def reduced_state(candidate):
        logs = jax.vmap(
            lambda state: compact_tree_log_psi(candidate, state, config)
        )(labels)
        reduced = columns.conj().T @ jnp.exp(logs)
        return reduced / jnp.linalg.norm(reduced)

    params = init_compact_tree_gru(jax.random.PRNGKey(seed), config)
    return _train_exact_state_adam(
        params,
        reduced_state,
        sector_hamiltonian,
        target_state,
        seed=seed,
        steps=steps,
        learning_rate=learning_rate,
        record_every=record_every,
        fidelity_threshold=fidelity_threshold,
        optimizer_name="exact_projected_fidelity_adam",
        loss_kind="fidelity",
    )


def train_exact_branching_projected_tree_fidelity_adam(
    config: CompactTreeGRUConfig,
    full_labels: np.ndarray,
    sector_columns: np.ndarray,
    sector_hamiltonian: np.ndarray,
    target_state: np.ndarray,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    record_every: int,
    fidelity_threshold: float,
) -> tuple[object, dict[str, Any], np.ndarray]:
    """Capacity control with hidden states propagated down tree branches."""

    labels = jnp.asarray(full_labels, dtype=jnp.int32)
    columns = jnp.asarray(sector_columns, dtype=jnp.complex128)
    if labels.shape != (sector_columns.shape[0], config.n - 1):
        raise ValueError("full_labels do not match the tree sector")

    def reduced_state(candidate):
        logs = jax.vmap(
            lambda state: branching_tree_log_psi(candidate, state, config)
        )(labels)
        reduced = columns.conj().T @ jnp.exp(logs)
        return reduced / jnp.linalg.norm(reduced)

    params = init_branching_tree_gru(jax.random.PRNGKey(seed), config)
    return _train_exact_state_adam(
        params,
        reduced_state,
        sector_hamiltonian,
        target_state,
        seed=seed,
        steps=steps,
        learning_rate=learning_rate,
        record_every=record_every,
        fidelity_threshold=fidelity_threshold,
        optimizer_name="exact_branching_projected_fidelity_adam",
        loss_kind="fidelity",
    )


def train_exact_quotient_tree_adam(
    config: CompactTreeGRUConfig,
    support: TreeOrbitSupport,
    sector_hamiltonian: np.ndarray,
    target_state: np.ndarray,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    record_every: int,
    fidelity_threshold: float,
) -> tuple[object, dict[str, Any], np.ndarray]:
    """Optimize a TreeGRU normalized directly on symmetry-orbit states."""

    if support.labels.shape[1:] != (config.n - 1,):
        raise ValueError("orbit support does not match the tree")
    if np.shape(sector_hamiltonian) != (support.size, support.size):
        raise ValueError("sector Hamiltonian does not match orbit support")
    indices = jnp.arange(support.size, dtype=jnp.int32)

    def quotient_state(candidate):
        logs = jax.vmap(
            lambda index: orbit_quotient_log_psi(candidate, index, config, support)
        )(indices)
        return jnp.exp(logs)

    params = init_compact_tree_gru(jax.random.PRNGKey(seed), config)
    return _train_exact_state_adam(
        params,
        quotient_state,
        sector_hamiltonian,
        target_state,
        seed=seed,
        steps=steps,
        learning_rate=learning_rate,
        record_every=record_every,
        fidelity_threshold=fidelity_threshold,
        optimizer_name="exact_quotient_adam",
    )


def train_exact_orbit_pushforward_adam(
    config: CompactTreeGRUConfig,
    support: OrbitPushforwardSupport,
    sector_hamiltonian: np.ndarray,
    target_state: np.ndarray,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    record_every: int,
    fidelity_threshold: float,
) -> tuple[object, dict[str, Any], np.ndarray]:
    """Optimize normalized coefficients obtained by character-orbit pooling."""

    if np.shape(sector_hamiltonian) != (support.n_orbits, support.n_orbits):
        raise ValueError("sector Hamiltonian does not match pushforward support")

    def quotient_state(candidate):
        return orbit_pushforward_state(candidate, config, support)

    params = init_compact_tree_gru(jax.random.PRNGKey(seed), config)
    return _train_exact_state_adam(
        params,
        quotient_state,
        sector_hamiltonian,
        target_state,
        seed=seed,
        steps=steps,
        learning_rate=learning_rate,
        record_every=record_every,
        fidelity_threshold=fidelity_threshold,
        optimizer_name="exact_orbit_pushforward_adam",
    )


def train_exact_conditioned_projected_heads_adam(
    config: HamiltonianConditionedTreeGRUConfig,
    base_params: dict[str, object],
    full_labels: np.ndarray,
    sector_columns: np.ndarray,
    sector_hamiltonian: np.ndarray,
    target_state: np.ndarray,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    record_every: int,
    fidelity_threshold: float,
) -> tuple[object, dict[str, Any], np.ndarray]:
    """Optimize only channel heads on a frozen Hamiltonian-conditioned state."""

    labels = jnp.asarray(full_labels, dtype=jnp.int32)
    columns = jnp.asarray(sector_columns, dtype=jnp.complex128)
    if labels.shape != (sector_columns.shape[0], config.n - 1):
        raise ValueError("full_labels do not match the conditioned tree sector")
    initialized = condition_existing_tree_gru(
        base_params, jax.random.PRNGKey(seed), config
    )
    fixed_local = {
        name: value
        for name, value in initialized["local"].items()
        if name not in {"probability_head", "phase_head"}
    }
    trainable = {
        "probability_head": initialized["local"]["probability_head"],
        "phase_head": initialized["local"]["phase_head"],
    }

    def assemble(heads):
        return {"base": base_params, "local": {**fixed_local, **heads}}

    precompute_started = time.perf_counter()
    fixed_params = assemble(trainable)
    path_data = jax.jit(
        jax.vmap(
            lambda state: fixed_conditioned_path_data(fixed_params, state, config)
        )
    )(labels)
    jax.block_until_ready(path_data)
    base_logits, base_phases, masks, actions, features = path_data
    precompute_seconds = time.perf_counter() - precompute_started

    def reduced_state(heads):
        local_logits = jnp.einsum(
            "sdach,ch->sda", features, heads["probability_head"]
        )
        local_phases = jnp.einsum(
            "sdach,ch->sda", features, heads["phase_head"]
        )
        log_probabilities = jax.nn.log_softmax(
            jnp.where(masks, base_logits + local_logits, -1.0e30), axis=-1
        )
        selected_log_probabilities = jnp.take_along_axis(
            log_probabilities, actions[..., None], axis=2
        )[..., 0]
        selected_phases = jnp.take_along_axis(
            base_phases + local_phases, actions[..., None], axis=2
        )[..., 0]
        logs = 0.5 * jnp.sum(selected_log_probabilities, axis=1) + 1.0j * jnp.sum(
            selected_phases, axis=1
        )
        reduced = columns.conj().T @ jnp.exp(logs)
        return reduced / jnp.linalg.norm(reduced)

    trained, report, state = _train_exact_state_adam(
        trainable,
        reduced_state,
        sector_hamiltonian,
        target_state,
        seed=seed,
        steps=steps,
        learning_rate=learning_rate,
        record_every=record_every,
        fidelity_threshold=fidelity_threshold,
        optimizer_name="exact_conditioned_projected_heads_adam",
    )
    report["frozen_base_parameter_count"] = parameter_count(base_params)
    report["stored_conditioner_parameter_count"] = parameter_count(fixed_local)
    report["fixed_feature_precompute_seconds"] = precompute_seconds
    report["fixed_feature_bytes"] = int(
        sum(value.size * value.dtype.itemsize for value in path_data)
    )
    return assemble(trained), report, state


__all__.extend(
    [
        "train_exact_orbit_pushforward_adam",
        "train_exact_branching_projected_tree_fidelity_adam",
        "train_exact_conditioned_projected_heads_adam",
        "train_exact_projected_tree_adam",
        "train_exact_projected_tree_fidelity_adam",
        "train_exact_quotient_tree_adam",
    ]
)


def doob_conditioned_full_state(
    heads: dict[str, jax.Array],
    base_path_data: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    features: jax.Array,
) -> jax.Array:
    """Evaluate a normalized frozen-base state with causal Doob features."""

    base_logits, base_phases, masks, actions = base_path_data
    feature_values = jnp.asarray(features, dtype=jnp.float64)
    expected = base_logits.shape + (heads["probability_head"].shape[-1],)
    if feature_values.shape != expected:
        raise ValueError(f"expected feature shape {expected}, got {feature_values.shape}")
    local_logits = jnp.einsum(
        "sdac,dc->sda", feature_values, heads["probability_head"]
    )
    local_phases = jnp.einsum(
        "sdac,dc->sda", feature_values, heads["phase_head"]
    )
    log_probabilities = jax.nn.log_softmax(
        jnp.where(masks, base_logits + local_logits, -1.0e30), axis=-1
    )
    selected_log_probabilities = jnp.take_along_axis(
        log_probabilities, actions[..., None], axis=2
    )[..., 0]
    selected_phases = jnp.take_along_axis(
        base_phases + local_phases, actions[..., None], axis=2
    )[..., 0]
    logs = 0.5 * jnp.sum(selected_log_probabilities, axis=1) + 1.0j * jnp.sum(
        selected_phases, axis=1
    )
    return jnp.exp(logs)


def train_exact_doob_conditioned_projected_heads_adam(
    config: CompactTreeGRUConfig,
    base_params: dict[str, object],
    full_labels: np.ndarray,
    sector_columns: np.ndarray,
    sector_hamiltonian: np.ndarray,
    target_state: np.ndarray,
    candidate_features: np.ndarray,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    record_every: int,
    fidelity_threshold: float,
) -> tuple[object, dict[str, Any], np.ndarray, np.ndarray]:
    """Fit only cut-resolved heads on exact-small Hamiltonian-Doob features."""

    labels = jnp.asarray(full_labels, dtype=jnp.int32)
    columns = jnp.asarray(sector_columns, dtype=jnp.complex128)
    concrete_features = np.asarray(candidate_features, dtype=np.float64)
    expected = (
        len(full_labels),
        config.n - 1,
        config.action_dim,
    )
    if concrete_features.ndim != 4 or concrete_features.shape[:3] != expected:
        raise ValueError(
            "candidate_features must have shape "
            "(states, decisions, action_dim, features)"
        )
    if sector_columns.shape[0] != len(full_labels):
        raise ValueError("sector columns do not match full_labels")

    precompute_started = time.perf_counter()
    path_data = jax.jit(
        jax.vmap(
            lambda state: fixed_compact_tree_path_data(base_params, state, config)
        )
    )(labels)
    jax.block_until_ready(path_data)
    features = jnp.asarray(concrete_features, dtype=jnp.float64)
    feature_count = int(concrete_features.shape[-1])
    heads = {
        "probability_head": jnp.zeros(
            (config.n - 1, feature_count), dtype=jnp.float64
        ),
        "phase_head": jnp.zeros(
            (config.n - 1, feature_count), dtype=jnp.float64
        ),
    }
    precompute_seconds = time.perf_counter() - precompute_started

    def full_state(candidate):
        return doob_conditioned_full_state(candidate, path_data, features)

    def reduced_state(candidate):
        reduced = columns.conj().T @ full_state(candidate)
        return reduced / jnp.linalg.norm(reduced)

    trained, report, state = _train_exact_state_adam(
        heads,
        reduced_state,
        sector_hamiltonian,
        target_state,
        seed=seed,
        steps=steps,
        learning_rate=learning_rate,
        record_every=record_every,
        fidelity_threshold=fidelity_threshold,
        optimizer_name="exact_doob_conditioned_projected_heads_adam",
    )
    final_full = np.asarray(full_state(trained), dtype=np.complex128)
    report["fixed_base_parameter_count"] = parameter_count(base_params)
    report["feature_count_per_cut"] = feature_count
    report["fixed_feature_precompute_seconds"] = precompute_seconds
    report["fixed_feature_bytes"] = int(
        concrete_features.nbytes
        + sum(value.size * value.dtype.itemsize for value in path_data)
    )
    zero_full = np.asarray(full_state(heads), dtype=np.complex128)
    base_logs = jax.vmap(
        lambda state: compact_tree_log_psi(base_params, state, config)
    )(labels)
    base_full = np.exp(np.asarray(base_logs, dtype=np.complex128))
    report["zero_head_amplitude_error"] = float(
        np.max(np.abs(zero_full - base_full), initial=0.0)
    )
    report["full_autoregressive_normalization_error"] = abs(
        float(np.vdot(final_full, final_full).real) - 1.0
    )
    return trained, report, state, final_full


__all__.extend(
    [
        "doob_conditioned_full_state",
        "train_exact_doob_conditioned_projected_heads_adam",
    ]
)
