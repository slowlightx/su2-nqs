"""Fail-closed timing instrumentation for sampled matrix-free SR.

This module is deliberately independent of the historical training runners.
It exposes the same native-basis interface for fusion-path SU(2) and fixed-Sz
models while separating compile, sampling, basis lookup, local-energy plus
backpropagation, QGT/CG/update, and observer wall times.

The conjugate-gradient loop is implemented explicitly so the executed
iteration and matrix-vector counts are measurements rather than inferred
proxies.  Current JAX/XLA ``cost_analysis`` does not specify whether FLOPs in a
data-dependent ``while`` loop are multiplied by the executed iteration count.
Consequently the full training-core semantic-FLOP field is fail-closed as
unavailable; raw static compiler analyses are retained as diagnostics only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .padded_sparse import padded_connectivity


@dataclass(frozen=True)
class InstrumentedSRResult:
    """Trained parameters, step records, and disjoint timing ledger."""

    params: object
    history: tuple[Mapping[str, Any], ...]
    initial_observation: Mapping[str, Any] | None
    cost_ledger: Mapping[str, Any]
    algorithmic_work: Mapping[str, Any]
    semantic_flops: Mapping[str, Any]


def _finite_positive(value: Real, name: str) -> float:
    answer = float(value)
    if not np.isfinite(answer) or answer <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return answer


def _block_until_ready(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready()
        if hasattr(leaf, "block_until_ready")
        else leaf,
        tree,
    )


def _json_cost_analysis(lowered: Any) -> dict[str, float]:
    """Return finite scalar compiler diagnostics without interpreting loops."""

    try:
        analysis = lowered.cost_analysis()
    except Exception:
        return {}
    if isinstance(analysis, (tuple, list)):
        rows = tuple(row for row in analysis if isinstance(row, Mapping))
        if len(rows) != 1:
            return {}
        analysis = rows[0]
    if not isinstance(analysis, Mapping):
        return {}
    answer = {}
    for name, value in analysis.items():
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(scalar):
            answer[str(name)] = scalar
    return answer


def conjugate_gradient_with_count(
    matrix_vector: Callable[[jax.Array], jax.Array],
    right_hand_side: jax.Array,
    *,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Solve a positive-definite system with exact executed-work counters.

    Returns ``(solution, iterations, matvecs, true_relative_residual)``.
    There is one matrix-vector product per CG iteration and one additional
    product for the independently recomputed true residual.
    """

    tolerance = _finite_positive(tolerance, "tolerance")
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, Integral)
        or int(maximum_iterations) < 1
    ):
        raise ValueError("maximum_iterations must be a positive integer")
    maximum_iterations = int(maximum_iterations)
    rhs = jnp.asarray(right_hand_side)
    if rhs.ndim != 1:
        raise ValueError("right_hand_side must be one-dimensional")
    solution = jnp.zeros_like(rhs)
    residual = rhs
    direction = residual
    squared_residual = jnp.real(jnp.vdot(residual, residual))
    rhs_norm = jnp.linalg.norm(rhs)
    threshold_squared = (tolerance * jnp.maximum(rhs_norm, jnp.finfo(rhs.dtype).tiny)) ** 2

    def cg_condition(state):
        iteration, _, carried_residual, _, value = state
        del carried_residual
        return (iteration < maximum_iterations) & (value > threshold_squared)

    def cg_body(state):
        iteration, current, carried_residual, search, value = state
        image = matrix_vector(search)
        denominator = jnp.real(jnp.vdot(search, image))
        safe = jnp.where(
            jnp.abs(denominator) > jnp.finfo(rhs.dtype).tiny,
            denominator,
            jnp.finfo(rhs.dtype).tiny,
        )
        alpha = value / safe
        changed = current + alpha * search
        next_residual = carried_residual - alpha * image
        next_value = jnp.real(jnp.vdot(next_residual, next_residual))
        beta = next_value / jnp.maximum(value, jnp.finfo(rhs.dtype).tiny)
        next_search = next_residual + beta * search
        return iteration + 1, changed, next_residual, next_search, next_value

    initial = (
        jnp.asarray(0, dtype=jnp.int32),
        solution,
        residual,
        direction,
        squared_residual,
    )
    iterations, solution, _, _, _ = jax.lax.while_loop(
        cg_condition, cg_body, initial
    )
    true_residual = matrix_vector(solution) - rhs
    relative = jnp.linalg.norm(true_residual) / jnp.maximum(
        rhs_norm, jnp.finfo(rhs.dtype).tiny
    )
    matvecs = iterations + jnp.asarray(1, dtype=jnp.int32)
    return solution, iterations, matvecs, relative


def _validated_cost_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    wall_components = (
        "sampling_wall_seconds",
        "basis_index_wall_seconds",
        "local_energy_and_backprop_wall_seconds",
        "qgt_and_cg_update_wall_seconds",
        "training_orchestration_wall_seconds",
    )
    required = (
        *wall_components,
        "training_core_wall_seconds",
        "observer_wall_seconds",
        "compile_wall_seconds",
        "end_to_end_wall_seconds",
        "optimizer_steps_attempted",
        "cg_iterations",
        "qgt_matvecs",
        "true_cg_relative_residual_maximum",
    )
    missing = tuple(name for name in required if name not in ledger)
    if missing:
        raise ValueError(f"instrumented cost ledger missing {missing}")
    numerical = {
        name: float(ledger[name])
        for name in required
        if name
        not in ("optimizer_steps_attempted", "cg_iterations", "qgt_matvecs")
    }
    if any(not np.isfinite(value) or value < 0.0 for value in numerical.values()):
        raise ValueError("instrumented cost values must be finite and non-negative")
    for name in ("optimizer_steps_attempted", "cg_iterations", "qgt_matvecs"):
        value = ledger[name]
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    component_sum = sum(float(ledger[name]) for name in wall_components)
    scale = max(1.0, component_sum, float(ledger["training_core_wall_seconds"]))
    if abs(component_sum - float(ledger["training_core_wall_seconds"])) > 1.0e-9 * scale:
        raise ValueError("training-core timing components overlap or do not close")
    accounted = (
        component_sum
        + float(ledger["observer_wall_seconds"])
        + float(ledger["compile_wall_seconds"])
    )
    if float(ledger["end_to_end_wall_seconds"]) + 1.0e-9 * max(1.0, accounted) < accounted:
        raise ValueError("disjoint timings exceed end-to-end wall")
    if int(ledger["qgt_matvecs"]) != int(ledger["cg_iterations"]) + int(
        ledger["optimizer_steps_attempted"]
    ):
        raise ValueError("qgt_matvec count does not equal iterations plus true-residual checks")
    return {
        "component_sum": component_sum,
        "accounted_wall_seconds": accounted,
        "unassigned_wall_seconds": float(ledger["end_to_end_wall_seconds"])
        - accounted,
    }


def _step_algorithmic_work(
    *,
    n_sites: int,
    autoregressive_steps_per_configuration: int,
    batch_size: int,
    padded_connectivity_width: int,
    valid_connected_entries: int,
    cg_iterations: int,
    qgt_matvecs: int,
) -> dict[str, int]:
    """Count source-level work exactly without calling it backend FLOPs.

    A ``configuration`` count records how many complete autoregressive
    sequences enter the corresponding program stage.  Multiplication by
    ``n_sites`` gives recurrent-token transitions.  Tangent and adjoint
    sweeps are kept separate because neither has a backend-independent scalar
    FLOP weight.  The vector is therefore auditable across representations
    while remaining honest about hardware/compiler differences.
    """

    integer_values = {
        "n_sites": n_sites,
        "autoregressive_steps_per_configuration": autoregressive_steps_per_configuration,
        "batch_size": batch_size,
        "padded_connectivity_width": padded_connectivity_width,
        "valid_connected_entries": valid_connected_entries,
        "cg_iterations": cg_iterations,
        "qgt_matvecs": qgt_matvecs,
    }
    if any(
        isinstance(value, bool) or int(value) != value or int(value) < 0
        for value in integer_values.values()
    ):
        raise ValueError("algorithmic-work inputs must be non-negative integers")
    if (
        n_sites < 1
        or autoregressive_steps_per_configuration < 1
        or batch_size < 1
        or padded_connectivity_width < 1
    ):
        raise ValueError("algorithmic-work dimensions must be positive")
    if qgt_matvecs != cg_iterations + 1:
        raise ValueError("each step requires one true-residual QGT matvec")
    padded = batch_size * padded_connectivity_width
    if valid_connected_entries > padded:
        raise ValueError("valid connectivity cannot exceed padded connectivity")

    sequence_counts = {
        "sampling_forward_configurations": batch_size,
        "local_energy_reference_primal_configurations": batch_size,
        "local_energy_connected_primal_configurations_padded": padded,
        "energy_gradient_primal_configurations": batch_size,
        "energy_gradient_adjoint_configurations": batch_size,
        "qgt_setup_primal_configurations": batch_size,
        "qgt_tangent_configurations": batch_size * qgt_matvecs,
        "qgt_adjoint_configurations": batch_size * qgt_matvecs,
    }
    token_counts = {
        name.replace("configurations", "token_transitions"): value
        * autoregressive_steps_per_configuration
        for name, value in sequence_counts.items()
    }
    return {
        "sample_draws": batch_size,
        "basis_lookup_configurations": batch_size,
        "hamiltonian_columns_sampled": batch_size,
        "hamiltonian_connected_entries_valid": valid_connected_entries,
        "hamiltonian_connected_slots_evaluated_padded": padded,
        "cg_iterations": cg_iterations,
        "qgt_matvecs": qgt_matvecs,
        **sequence_counts,
        **token_counts,
        "recurrent_token_transitions_total": sum(token_counts.values()),
    }


def _aggregate_algorithmic_work(
    rows: Sequence[Mapping[str, int]],
    *,
    n_sites: int,
    autoregressive_steps_per_configuration: int,
    batch_size: int,
    padded_connectivity_width: int,
) -> dict[str, Any]:
    """Sum exact step work and attach closure diagnostics."""

    if not rows:
        raise ValueError("algorithmic-work ledger requires at least one step")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise ValueError("algorithmic-work rows have inconsistent fields")
    totals = {name: sum(int(row[name]) for row in rows) for name in keys}
    transition_components = tuple(
        name
        for name in keys
        if "_token_transitions" in name
        and name != "recurrent_token_transitions_total"
    )
    closed = sum(totals[name] for name in transition_components)
    if closed != totals["recurrent_token_transitions_total"]:
        raise RuntimeError("algorithmic recurrent-transition ledger does not close")
    if totals["qgt_matvecs"] != totals["cg_iterations"] + len(rows):
        raise RuntimeError("algorithmic QGT-matvec ledger does not close")
    return {
        "schema_version": 1,
        "available": True,
        "is_backend_flop_count": False,
        "is_single_weighted_cost": False,
        "interpretation": (
            "Exact source-semantic work vector. Components must remain separate; "
            "wall time is an independent co-primary axis and no component may be "
            "renamed or weighted post hoc as FLOPs."
        ),
        "optimizer_steps": len(rows),
        "n_sites": n_sites,
        "autoregressive_steps_per_configuration": autoregressive_steps_per_configuration,
        "batch_size": batch_size,
        "padded_connectivity_width": padded_connectivity_width,
        "totals": totals,
        "audit": {
            "transition_component_sum": closed,
            "transition_total": totals["recurrent_token_transitions_total"],
            "qgt_matvec_closure": totals["qgt_matvecs"]
            - totals["cg_iterations"],
        },
    }


def train_instrumented_padded_sparse_sr(
    params: object,
    key: jax.Array,
    *,
    matrix: Any,
    basis: Sequence[Sequence[int]],
    sample_fn: Callable[[object, jax.Array, int], jax.Array],
    log_psi_fn: Callable[[object, jax.Array], jax.Array],
    steps: int,
    batch_size: int,
    learning_rate: float,
    damping: float,
    cg_tolerance: float = 1.0e-6,
    cg_max_iterations: int = 400,
    maximum_cg_relative_residual: float = 1.0e-3,
    autoregressive_steps_per_configuration: int | None = None,
    observer: Callable[[object, int], Mapping[str, Any]] | None = None,
    observe_every: int = 1,
) -> InstrumentedSRResult:
    """Run a small instrumented sampled-SR trajectory in either native basis."""

    if isinstance(steps, bool) or not isinstance(steps, Integral) or int(steps) < 1:
        raise ValueError("steps must be a positive integer")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or int(batch_size) < 2
    ):
        raise ValueError("batch_size must be an integer at least two")
    if (
        isinstance(observe_every, bool)
        or not isinstance(observe_every, Integral)
        or int(observe_every) < 1
    ):
        raise ValueError("observe_every must be a positive integer")
    steps, batch_size, observe_every = int(steps), int(batch_size), int(observe_every)
    learning_rate = _finite_positive(learning_rate, "learning_rate")
    damping = _finite_positive(damping, "damping")
    cg_tolerance = _finite_positive(cg_tolerance, "cg_tolerance")
    maximum_cg_relative_residual = _finite_positive(
        maximum_cg_relative_residual, "maximum_cg_relative_residual"
    )
    if (
        isinstance(cg_max_iterations, bool)
        or not isinstance(cg_max_iterations, Integral)
        or int(cg_max_iterations) < 1
    ):
        raise ValueError("cg_max_iterations must be a positive integer")
    cg_max_iterations = int(cg_max_iterations)

    basis_tuple = tuple(tuple(int(value) for value in state) for state in basis)
    if not basis_tuple or matrix.shape != (len(basis_tuple), len(basis_tuple)):
        raise ValueError("matrix and nonempty basis dimensions must agree")
    positions = {state: index for index, state in enumerate(basis_tuple)}
    if len(positions) != len(basis_tuple):
        raise ValueError("basis states must be unique")
    if autoregressive_steps_per_configuration is None:
        autoregressive_steps_per_configuration = len(basis_tuple[0])
    if (
        isinstance(autoregressive_steps_per_configuration, bool)
        or not isinstance(autoregressive_steps_per_configuration, Integral)
        or int(autoregressive_steps_per_configuration) < 1
    ):
        raise ValueError("autoregressive_steps_per_configuration must be positive")
    autoregressive_steps_per_configuration = int(
        autoregressive_steps_per_configuration
    )
    basis_array = jnp.asarray(basis_tuple, dtype=jnp.int32)
    neighbors, elements, valid = padded_connectivity(matrix)
    neighbors_array = jnp.asarray(neighbors)
    elements_array = jnp.asarray(elements)
    valid_array = jnp.asarray(valid)
    flat, unravel = ravel_pytree(params)

    def sample_kernel(values, sample_key):
        return sample_fn(unravel(values), sample_key, batch_size)

    def local_energy_and_gradient(values, samples, indices):
        candidate = unravel(values)
        logs = jax.vmap(lambda state: log_psi_fn(candidate, state))(samples)
        connected_states = basis_array[neighbors_array[indices]]
        connected_logs = jax.vmap(
            jax.vmap(lambda state: log_psi_fn(candidate, state))
        )(connected_states)
        contributions = jnp.where(
            valid_array[indices],
            elements_array[indices]
            * jnp.exp(connected_logs - logs[:, None]),
            0.0,
        )
        local = jnp.sum(contributions, axis=1)
        mean = jnp.mean(local)
        centered_local = jax.lax.stop_gradient(local - mean)

        def objective(candidate_values):
            changed = unravel(candidate_values)
            changed_logs = jax.vmap(
                lambda state: log_psi_fn(changed, state)
            )(samples)
            surrogate = 2.0 * jnp.real(
                jnp.mean(jnp.conj(changed_logs) * centered_local)
            )
            return surrogate

        gradient = jax.grad(objective)(values)
        variance = jnp.mean(jnp.abs(local - mean) ** 2)
        return jnp.real(mean), jnp.real(variance), gradient

    def qgt_cg_update(values, samples, gradient):
        def centered_features(candidate_values):
            candidate = unravel(candidate_values)
            logs = jax.vmap(
                lambda state: log_psi_fn(candidate, state)
            )(samples)
            centered = logs - jnp.mean(logs)
            scale = jnp.sqrt(2.0 / batch_size)
            return jnp.concatenate(
                (scale * jnp.real(centered), scale * jnp.imag(centered))
            )

        _, pullback = jax.vjp(centered_features, values)

        def fisher_product(vector):
            _, tangent = jax.jvp(centered_features, (values,), (vector,))
            return pullback(tangent)[0] + damping * vector

        direction, iterations, matvecs, relative_residual = (
            conjugate_gradient_with_count(
                fisher_product,
                gradient,
                tolerance=cg_tolerance,
                maximum_iterations=cg_max_iterations,
            )
        )
        return (
            values - learning_rate * direction,
            jnp.linalg.norm(gradient),
            jnp.linalg.norm(direction),
            iterations,
            matvecs,
            relative_residual,
        )

    sample_jit = jax.jit(sample_kernel)
    local_jit = jax.jit(local_energy_and_gradient)
    qgt_jit = jax.jit(qgt_cg_update)
    sample_shape = jax.ShapeDtypeStruct(
        (batch_size, len(basis_tuple[0])), jnp.int32
    )
    index_shape = jax.ShapeDtypeStruct((batch_size,), jnp.int32)
    gradient_shape = jax.ShapeDtypeStruct(flat.shape, flat.dtype)

    end_to_end_started = time.perf_counter()
    compiler_build_started = time.perf_counter()
    sample_lowered = sample_jit.lower(flat, key)
    sample_compiled = sample_lowered.compile()
    local_lowered = local_jit.lower(flat, sample_shape, index_shape)
    local_compiled = local_lowered.compile()
    qgt_lowered = qgt_jit.lower(flat, sample_shape, gradient_shape)
    qgt_compiled = qgt_lowered.compile()
    compiler_diagnostics = {
        "sampling_static": _json_cost_analysis(sample_lowered),
        "local_energy_and_backprop_static": _json_cost_analysis(local_lowered),
        "qgt_cg_update_static": _json_cost_analysis(qgt_lowered),
    }
    compiler_build_wall = time.perf_counter() - compiler_build_started

    # Execute every compiled stage once with production shapes.  The warmup
    # key is derived without splitting or advancing the training key, and its
    # candidate parameters are discarded.  This makes later step timings
    # compile-free while preserving the registered training RNG stream and
    # initial parameter state.
    warmup_started = time.perf_counter()
    warmup_key = jax.random.fold_in(key, 0x50384430)
    warmup_samples = _block_until_ready(sample_compiled(flat, warmup_key))
    warmup_concrete = np.asarray(warmup_samples)
    try:
        warmup_indices_np = np.asarray(
            [
                positions[tuple(int(value) for value in state)]
                for state in warmup_concrete
            ],
            dtype=np.int32,
        )
    except KeyError as error:
        raise ValueError("warmup sampler returned a state outside its basis") from error
    warmup_indices = jnp.asarray(warmup_indices_np)
    _, _, warmup_gradient = _block_until_ready(
        local_compiled(flat, warmup_samples, warmup_indices)
    )
    (
        _,
        _,
        _,
        warmup_iterations,
        warmup_matvecs,
        warmup_residual,
    ) = _block_until_ready(qgt_compiled(flat, warmup_samples, warmup_gradient))
    discarded_warmup_wall = time.perf_counter() - warmup_started
    compile_wall = compiler_build_wall + discarded_warmup_wall

    observer_wall = 0.0
    initial_observation = None
    if observer is not None:
        started = time.perf_counter()
        initial_observation = _block_until_ready(dict(observer(unravel(flat), 0)))
        observer_wall += time.perf_counter() - started

    history: list[Mapping[str, Any]] = []
    sampling_wall = 0.0
    basis_index_wall = 0.0
    local_wall = 0.0
    qgt_wall = 0.0
    total_iterations = 0
    total_matvecs = 0
    algorithmic_rows: list[Mapping[str, int]] = []
    maximum_true_residual = 0.0
    accepted_updates = 0
    for step in range(1, steps + 1):
        key, sample_key = jax.random.split(key)
        started = time.perf_counter()
        samples = _block_until_ready(sample_compiled(flat, sample_key))
        step_sampling = time.perf_counter() - started
        sampling_wall += step_sampling

        started = time.perf_counter()
        concrete = np.asarray(samples)
        try:
            indices_np = np.asarray(
                [positions[tuple(int(value) for value in state)] for state in concrete],
                dtype=np.int32,
            )
        except KeyError as error:
            raise ValueError("native sampler returned a state outside its basis") from error
        indices = jnp.asarray(indices_np)
        valid_connected_entries = int(np.sum(valid[indices_np]))
        step_index = time.perf_counter() - started
        basis_index_wall += step_index

        started = time.perf_counter()
        energy, variance, gradient = _block_until_ready(
            local_compiled(flat, samples, indices)
        )
        step_local = time.perf_counter() - started
        local_wall += step_local

        started = time.perf_counter()
        (
            candidate,
            gradient_norm,
            direction_norm,
            iterations,
            matvecs,
            relative_residual,
        ) = _block_until_ready(qgt_compiled(flat, samples, gradient))
        step_qgt = time.perf_counter() - started
        qgt_wall += step_qgt
        iteration_count = int(iterations)
        matvec_count = int(matvecs)
        residual_value = float(relative_residual)
        total_iterations += iteration_count
        total_matvecs += matvec_count
        step_algorithmic_work = _step_algorithmic_work(
            n_sites=len(basis_tuple[0]),
            autoregressive_steps_per_configuration=(
                autoregressive_steps_per_configuration
            ),
            batch_size=batch_size,
            padded_connectivity_width=int(neighbors.shape[1]),
            valid_connected_entries=valid_connected_entries,
            cg_iterations=iteration_count,
            qgt_matvecs=matvec_count,
        )
        algorithmic_rows.append(step_algorithmic_work)
        maximum_true_residual = max(maximum_true_residual, residual_value)
        accepted = bool(
            np.isfinite(residual_value)
            and residual_value <= maximum_cg_relative_residual
        )
        if accepted:
            flat = candidate
            accepted_updates += 1

        observation = None
        step_observer = 0.0
        if observer is not None and (
            step % observe_every == 0 or step == steps or not accepted
        ):
            started = time.perf_counter()
            observation = _block_until_ready(dict(observer(unravel(flat), step)))
            step_observer = time.perf_counter() - started
            observer_wall += step_observer
        history.append(
            {
                "step": step,
                "energy": float(energy),
                "variance": float(variance),
                "gradient_norm": float(gradient_norm),
                "natural_direction_norm": float(direction_norm),
                "cg_iterations": iteration_count,
                "qgt_matvecs": matvec_count,
                "true_cg_relative_residual": residual_value,
                "update_accepted": accepted,
                "sampling_wall_seconds": step_sampling,
                "basis_index_wall_seconds": step_index,
                "local_energy_and_backprop_wall_seconds": step_local,
                "qgt_and_cg_update_wall_seconds": step_qgt,
                "observer_wall_seconds": step_observer,
                "algorithmic_work": step_algorithmic_work,
                "observation": observation,
            }
        )
        if not accepted:
            break

    end_to_end_wall = time.perf_counter() - end_to_end_started
    measured_kernel_wall = sampling_wall + basis_index_wall + local_wall + qgt_wall
    orchestration_wall = end_to_end_wall - compile_wall - observer_wall - measured_kernel_wall
    if orchestration_wall < -1.0e-9:
        raise RuntimeError("instrumented timing partition exceeded end-to-end wall")
    orchestration_wall = max(orchestration_wall, 0.0)
    training_core_wall = measured_kernel_wall + orchestration_wall
    ledger = {
        "optimizer_steps_requested": steps,
        "optimizer_steps_attempted": len(history),
        "optimizer_updates_accepted": accepted_updates,
        "sampling_wall_seconds": sampling_wall,
        "basis_index_wall_seconds": basis_index_wall,
        "local_energy_and_backprop_wall_seconds": local_wall,
        "qgt_and_cg_update_wall_seconds": qgt_wall,
        "training_orchestration_wall_seconds": orchestration_wall,
        "training_core_wall_seconds": training_core_wall,
        "observer_wall_seconds": observer_wall,
        "compile_wall_seconds": compile_wall,
        "compiler_build_wall_seconds": compiler_build_wall,
        "discarded_warmup_wall_seconds": discarded_warmup_wall,
        "discarded_warmup_cg_iterations": int(warmup_iterations),
        "discarded_warmup_qgt_matvecs": int(warmup_matvecs),
        "discarded_warmup_true_cg_relative_residual": float(warmup_residual),
        "discarded_warmup_parameter_update_applied": False,
        "discarded_warmup_advanced_training_rng": False,
        "end_to_end_wall_seconds": end_to_end_wall,
        "cg_iterations": total_iterations,
        "qgt_matvecs": total_matvecs,
        "true_cg_relative_residual_maximum": maximum_true_residual,
    }
    ledger_audit = _validated_cost_ledger(ledger)
    ledger = {**ledger, "audit": ledger_audit}
    algorithmic_work = _aggregate_algorithmic_work(
        algorithmic_rows,
        n_sites=len(basis_tuple[0]),
        autoregressive_steps_per_configuration=(
            autoregressive_steps_per_configuration
        ),
        batch_size=batch_size,
        padded_connectivity_width=int(neighbors.shape[1]),
    )
    semantic = {
        "available": False,
        "training_core_semantic_flops": None,
        "formal_cost_gate_pass": False,
        "reason": (
            "The current JAX/XLA cost analysis reports a static qgt_cg_update "
            "program but does not certify executed FLOPs for its data-dependent "
            "CG while-loop. Multiplying a static body estimate by iterations is "
            "not an audited compiler measurement and is therefore forbidden."
        ),
        "future_acceptance_convention": (
            "A supported backend must provide executed FLOPs separately for "
            "sampling, local-energy/backprop, and QGT/CG/update; their sum is the "
            "training-core semantic FLOP count. Observer FLOPs remain separate."
        ),
        "raw_static_compiler_diagnostics_only": compiler_diagnostics,
        "executed_cg_iterations": total_iterations,
        "executed_qgt_matvecs": total_matvecs,
    }
    return InstrumentedSRResult(
        params=unravel(flat),
        history=tuple(history),
        initial_observation=initial_observation,
        cost_ledger=ledger,
        algorithmic_work=algorithmic_work,
        semantic_flops=semantic,
    )


__all__ = [
    "InstrumentedSRResult",
    "conjugate_gradient_with_count",
    "train_instrumented_padded_sparse_sr",
]
