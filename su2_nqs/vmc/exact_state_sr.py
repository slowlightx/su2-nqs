"""Matrix-free natural-gradient optimization for exact-small wavefunctions.

This module is deliberately model agnostic.  It is a calibration instrument
for ansatzes whose complete small-system amplitude vector can be enumerated;
production VMC must continue to use sampled estimators.
"""

from __future__ import annotations

from collections.abc import Callable
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg


ExactStateFunction = Callable[[object], jax.Array]
HamiltonianAction = Callable[[jax.Array], jax.Array]
ParameterMap = Callable[[jax.Array], jax.Array]


def _validate_real_parameters(params: object) -> None:
    for leaf in jax.tree_util.tree_leaves(params):
        if not np.issubdtype(np.asarray(leaf).dtype, np.floating):
            raise TypeError("exact-state SR currently requires real parameters")


def train_exact_state_sr(
    params: object,
    *,
    state_fn: ExactStateFunction,
    hamiltonian: jax.Array | None = None,
    hamiltonian_action: HamiltonianAction | None = None,
    steps: int,
    learning_rate: float,
    damping: float,
    cg_tolerance: float = 1.0e-6,
    cg_max_iterations: int = 100,
    record_every: int = 10,
) -> tuple[object, dict[str, object]]:
    """Optimize a fully enumerated state with the pure-state tangent metric.

    ``state_fn(params)`` returns an unnormalized real or complex amplitude
    vector in the same order as ``hamiltonian``.  Parameters are presently
    restricted to real arrays.  The metric is the Euclidean pullback of the
    normalized state's real and imaginary components; for real wavefunctions
    this is the real pure-state quantum geometric tensor.  The linear system
    is applied by JVP/VJP and is never materialized.
    """

    if steps < 1 or record_every < 1:
        raise ValueError("steps and record_every must be positive")
    if learning_rate <= 0.0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    if damping <= 0.0 or not np.isfinite(damping):
        raise ValueError("damping must be positive and finite")
    if cg_tolerance <= 0.0 or cg_max_iterations < 1:
        raise ValueError("invalid conjugate-gradient controls")
    _validate_real_parameters(params)
    if (hamiltonian is None) == (hamiltonian_action is None):
        raise ValueError(
            "supply exactly one of hamiltonian or hamiltonian_action"
        )
    matrix = None if hamiltonian is None else jnp.asarray(hamiltonian)
    if matrix is not None and (
        matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError("hamiltonian must be square")

    def apply_hamiltonian(state: jax.Array) -> jax.Array:
        if matrix is not None:
            return matrix @ state
        action = jnp.asarray(hamiltonian_action(state))
        if action.shape != state.shape:
            raise ValueError("hamiltonian_action changed the state-vector shape")
        return action

    flat, unravel = ravel_pytree(params)

    def normalized_state(values: jax.Array) -> jax.Array:
        amplitudes = jnp.ravel(jnp.asarray(state_fn(unravel(values))))
        norm = jnp.real(jnp.vdot(amplitudes, amplitudes))
        return amplitudes / jnp.sqrt(norm)

    def energy(values: jax.Array) -> jax.Array:
        state = normalized_state(values)
        return jnp.real(jnp.vdot(state, apply_hamiltonian(state)))

    def tangent_coordinates(values: jax.Array) -> jax.Array:
        state = normalized_state(values)
        return jnp.concatenate((jnp.real(state), jnp.imag(state)))

    @jax.jit
    def update(
        values: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        _, pullback = jax.vjp(tangent_coordinates, values)

        def metric_product(vector: jax.Array) -> jax.Array:
            _, tangent = jax.jvp(tangent_coordinates, (values,), (vector,))
            return pullback(tangent)[0] + damping * vector

        current_energy, gradient = jax.value_and_grad(energy)(values)
        direction, _ = cg(
            metric_product,
            gradient,
            tol=cg_tolerance,
            maxiter=cg_max_iterations,
        )
        residual = metric_product(direction) - gradient
        relative_residual = jnp.linalg.norm(residual) / jnp.maximum(
            jnp.linalg.norm(gradient),
            jnp.finfo(values.dtype).tiny,
        )
        return (
            values - learning_rate * direction,
            current_energy,
            jnp.linalg.norm(gradient),
            jnp.linalg.norm(direction),
            relative_residual,
        )

    compile_started = time.perf_counter()
    compiled_energy = energy(flat)
    compiled_energy.block_until_ready()
    compiled_update = update(flat)
    jax.block_until_ready(compiled_update)
    compile_seconds = time.perf_counter() - compile_started
    initial_energy = float(compiled_energy)
    history: list[dict[str, float]] = []
    training_started = time.perf_counter()
    for step in range(1, steps + 1):
        (
            flat,
            current_energy,
            gradient_norm,
            direction_norm,
            relative_residual,
        ) = update(flat)
        if step % record_every == 0 or step == steps:
            recorded_energy = energy(flat)
            recorded_energy.block_until_ready()
            history.append(
                {
                    "step": float(step),
                    "energy": float(recorded_energy),
                    "energy_before_update": float(current_energy),
                    "gradient_norm": float(gradient_norm),
                    "natural_direction_norm": float(direction_norm),
                    "cg_relative_residual": float(relative_residual),
                }
            )
    training_seconds = time.perf_counter() - training_started
    params = unravel(flat)
    state = normalized_state(flat)
    h_state = apply_hamiltonian(state)
    final_energy = jnp.real(jnp.vdot(state, h_state))
    variance = jnp.real(jnp.vdot(h_state - final_energy * state, h_state - final_energy * state))
    return params, {
        "optimizer": "exact_pure_state_sr",
        "steps": steps,
        "learning_rate": learning_rate,
        "damping": damping,
        "cg_tolerance": cg_tolerance,
        "cg_max_iterations": cg_max_iterations,
        "initial_energy": initial_energy,
        "final_energy": float(final_energy),
        "final_variance": float(variance),
        "compile_seconds": compile_seconds,
        "training_seconds": training_seconds,
        "history": history,
    }


def train_exact_state_sr_trust_region(
    params: object,
    *,
    state_fn: ExactStateFunction,
    hamiltonian: jax.Array | None = None,
    hamiltonian_action: HamiltonianAction | None = None,
    steps: int,
    initial_learning_rate: float,
    initial_damping: float,
    maximum_fs_infidelity: float,
    cg_tolerance: float = 1.0e-7,
    cg_max_iterations: int = 100,
    maximum_backtracks: int = 12,
    record_every: int = 1,
) -> tuple[object, dict[str, object]]:
    """Exact-state SR with energy-monotone Fubini--Study backtracking.

    This is an exact-small calibration optimizer.  Each natural-gradient
    proposal is shortened until the normalized-state infidelity lies inside
    ``maximum_fs_infidelity`` and the exact variational energy decreases.
    Rejected directions increase the diagonal damping; accepted directions
    relax it back toward ``initial_damping``.  Production VMC must replace
    the exact quantities by independently audited sampled estimators.
    """

    if steps < 1 or record_every < 1 or maximum_backtracks < 1:
        raise ValueError("steps, record_every, and maximum_backtracks must be positive")
    for name, value in (
        ("initial_learning_rate", initial_learning_rate),
        ("initial_damping", initial_damping),
        ("maximum_fs_infidelity", maximum_fs_infidelity),
        ("cg_tolerance", cg_tolerance),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if maximum_fs_infidelity >= 1.0:
        raise ValueError("maximum_fs_infidelity must be below one")
    if cg_max_iterations < 1:
        raise ValueError("cg_max_iterations must be positive")
    _validate_real_parameters(params)
    if (hamiltonian is None) == (hamiltonian_action is None):
        raise ValueError("supply exactly one of hamiltonian or hamiltonian_action")
    matrix = None if hamiltonian is None else jnp.asarray(hamiltonian)
    if matrix is not None and (
        matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError("hamiltonian must be square")

    def apply_hamiltonian(state: jax.Array) -> jax.Array:
        if matrix is not None:
            return matrix @ state
        action = jnp.asarray(hamiltonian_action(state))
        if action.shape != state.shape:
            raise ValueError("hamiltonian_action changed the state-vector shape")
        return action

    flat, unravel = ravel_pytree(params)

    def normalized_state(values: jax.Array) -> jax.Array:
        amplitudes = jnp.ravel(jnp.asarray(state_fn(unravel(values))))
        norm = jnp.real(jnp.vdot(amplitudes, amplitudes))
        return amplitudes / jnp.sqrt(norm)

    def energy(values: jax.Array) -> jax.Array:
        state = normalized_state(values)
        return jnp.real(jnp.vdot(state, apply_hamiltonian(state)))

    def tangent_coordinates(values: jax.Array) -> jax.Array:
        state = normalized_state(values)
        return jnp.concatenate((jnp.real(state), jnp.imag(state)))

    @jax.jit
    def natural_direction(values: jax.Array, damping: jax.Array):
        _, pullback = jax.vjp(tangent_coordinates, values)

        def metric_product(vector: jax.Array) -> jax.Array:
            _, tangent = jax.jvp(tangent_coordinates, (values,), (vector,))
            return pullback(tangent)[0] + damping * vector

        current_energy, gradient = jax.value_and_grad(energy)(values)
        direction, _ = cg(
            metric_product,
            gradient,
            tol=cg_tolerance,
            maxiter=cg_max_iterations,
        )
        residual = metric_product(direction) - gradient
        relative_residual = jnp.linalg.norm(residual) / jnp.maximum(
            jnp.linalg.norm(gradient), jnp.finfo(values.dtype).tiny
        )
        return (
            direction,
            current_energy,
            jnp.linalg.norm(gradient),
            jnp.linalg.norm(direction),
            relative_residual,
        )

    @jax.jit
    def candidate_metrics(reference_state: jax.Array, values: jax.Array):
        candidate_state = normalized_state(values)
        candidate_energy = jnp.real(
            jnp.vdot(candidate_state, apply_hamiltonian(candidate_state))
        )
        overlap = jnp.vdot(reference_state, candidate_state)
        infidelity = jnp.maximum(0.0, 1.0 - jnp.abs(overlap) ** 2)
        return candidate_energy, infidelity

    compile_started = time.perf_counter()
    initial_state = normalized_state(flat)
    initial_energy = energy(flat)
    compiled_direction = natural_direction(
        flat, jnp.asarray(initial_damping, dtype=flat.dtype)
    )
    compiled_candidate = candidate_metrics(initial_state, flat)
    jax.block_until_ready((initial_energy, compiled_direction, compiled_candidate))
    compile_seconds = time.perf_counter() - compile_started

    damping = float(initial_damping)
    history: list[dict[str, float | bool]] = []
    accepted_steps = 0
    rejected_steps = 0
    training_started = time.perf_counter()
    for step in range(1, steps + 1):
        reference_state = normalized_state(flat)
        (
            direction,
            current_energy,
            gradient_norm,
            direction_norm,
            relative_residual,
        ) = natural_direction(flat, jnp.asarray(damping, dtype=flat.dtype))
        step_size = float(initial_learning_rate)
        accepted = False
        chosen_energy = float(current_energy)
        chosen_infidelity = 0.0
        backtracks = 0
        for backtracks in range(maximum_backtracks + 1):
            candidate = flat - step_size * direction
            candidate_energy, infidelity = candidate_metrics(
                reference_state, candidate
            )
            candidate_energy_value = float(candidate_energy)
            infidelity_value = float(infidelity)
            if (
                np.isfinite(candidate_energy_value)
                and np.isfinite(infidelity_value)
                and infidelity_value <= maximum_fs_infidelity
                and candidate_energy_value < float(current_energy) - 1.0e-14
            ):
                flat = candidate
                chosen_energy = candidate_energy_value
                chosen_infidelity = infidelity_value
                accepted = True
                accepted_steps += 1
                damping = max(float(initial_damping), damping / 1.5)
                break
            step_size *= 0.5
        if not accepted:
            rejected_steps += 1
            damping *= 10.0
        if step % record_every == 0 or step == steps or not accepted:
            history.append(
                {
                    "step": float(step),
                    "accepted": accepted,
                    "energy_before_update": float(current_energy),
                    "energy": chosen_energy,
                    "fs_infidelity": chosen_infidelity,
                    "step_size": step_size if accepted else 0.0,
                    "backtracks": float(backtracks),
                    "damping": damping,
                    "gradient_norm": float(gradient_norm),
                    "natural_direction_norm": float(direction_norm),
                    "cg_relative_residual": float(relative_residual),
                }
            )
    training_seconds = time.perf_counter() - training_started
    params = unravel(flat)
    state = normalized_state(flat)
    h_state = apply_hamiltonian(state)
    final_energy = jnp.real(jnp.vdot(state, h_state))
    residual = h_state - final_energy * state
    variance = jnp.real(jnp.vdot(residual, residual))
    return params, {
        "optimizer": "exact_pure_state_sr_fs_trust_region",
        "steps": steps,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "initial_learning_rate": initial_learning_rate,
        "initial_damping": initial_damping,
        "final_damping": damping,
        "maximum_fs_infidelity": maximum_fs_infidelity,
        "maximum_backtracks": maximum_backtracks,
        "cg_tolerance": cg_tolerance,
        "cg_max_iterations": cg_max_iterations,
        "initial_energy": float(initial_energy),
        "final_energy": float(final_energy),
        "final_variance": float(variance),
        "compile_seconds": compile_seconds,
        "training_seconds": training_seconds,
        "history": history,
    }


def train_product_state_sr_trust_region(
    params: object,
    *,
    states_fn: ExactStateFunction,
    hamiltonian_action: HamiltonianAction,
    steps: int,
    initial_learning_rate: float,
    initial_damping: float,
    maximum_mean_fs_infidelity: float,
    maximum_state_fs_infidelity: float | None = None,
    canonicalize_flat: ParameterMap | None = None,
    project_flat_tangent: ParameterMap | None = None,
    cg_tolerance: float = 1.0e-7,
    cg_max_iterations: int = 100,
    linear_solver: str = "cg",
    maximum_linear_relative_residual: float | None = None,
    maximum_backtracks: int = 12,
    record_every: int = 1,
) -> tuple[object, dict[str, object]]:
    """Optimize one parameterization of several independent physical rays.

    ``states_fn(params)`` returns ``(K, D)`` amplitudes and
    ``hamiltonian_action(states)`` returns the corresponding ``H_k |psi_k>``.
    The objective is the mean of the ``K`` Rayleigh energies.  The metric is
    the mean Fubini--Study pullback, with the phase tangent removed *for each
    ray independently*.  This distinction is essential for coupling-
    conditioned states: concatenating the states into one artificial vector
    would assign physical meaning to relative phases between different
    Hamiltonians.

    The optional flat-parameter maps support exact gauge fixing.  A
    canonicalizer is applied to the initial point and every candidate, while
    the tangent projector is applied to gradients, metric products, and SR
    directions.  Both must be linear/idempotent on tangents; callers should
    test that property for their chosen gauge.

    This remains an exact-small calibration routine.  It does not replace a
    sampled product-state SR estimator for production VMC.
    """

    if steps < 1 or record_every < 1 or maximum_backtracks < 1:
        raise ValueError("steps, record_every, and maximum_backtracks must be positive")
    for name, value in (
        ("initial_learning_rate", initial_learning_rate),
        ("initial_damping", initial_damping),
        ("maximum_mean_fs_infidelity", maximum_mean_fs_infidelity),
        ("cg_tolerance", cg_tolerance),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    maximum_state_fs_infidelity = (
        maximum_mean_fs_infidelity
        if maximum_state_fs_infidelity is None
        else maximum_state_fs_infidelity
    )
    if (
        not np.isfinite(maximum_state_fs_infidelity)
        or maximum_state_fs_infidelity <= 0.0
        or maximum_mean_fs_infidelity >= 1.0
        or maximum_state_fs_infidelity >= 1.0
    ):
        raise ValueError("FS infidelity limits must lie in (0,1)")
    if cg_max_iterations < 1:
        raise ValueError("cg_max_iterations must be positive")
    if linear_solver not in {"cg", "dual_dense"}:
        raise ValueError("linear_solver must be 'cg' or 'dual_dense'")
    if maximum_linear_relative_residual is not None and (
        not np.isfinite(maximum_linear_relative_residual)
        or maximum_linear_relative_residual <= 0.0
    ):
        raise ValueError("maximum_linear_relative_residual must be positive")
    _validate_real_parameters(params)

    flat, unravel = ravel_pytree(params)
    identity = lambda value: value
    canonicalize = canonicalize_flat or identity
    project_tangent = project_flat_tangent or identity
    flat = canonicalize(flat)
    if flat.shape != project_tangent(flat).shape or flat.shape != canonicalize(flat).shape:
        raise ValueError("parameter gauge maps must preserve flat shape")

    def normalized_states(values: jax.Array) -> jax.Array:
        amplitudes = jnp.asarray(states_fn(unravel(canonicalize(values))))
        if amplitudes.ndim != 2:
            raise ValueError("states_fn must return a rank-two (K,D) array")
        norms = jnp.real(jnp.sum(jnp.conj(amplitudes) * amplitudes, axis=1))
        return amplitudes / jnp.sqrt(norms)[:, None]

    def actions(states: jax.Array) -> jax.Array:
        output = jnp.asarray(hamiltonian_action(states))
        if output.shape != states.shape:
            raise ValueError("hamiltonian_action changed the state-array shape")
        return output

    def state_energies(values: jax.Array) -> jax.Array:
        states = normalized_states(values)
        return jnp.real(jnp.sum(jnp.conj(states) * actions(states), axis=1))

    def objective(values: jax.Array) -> jax.Array:
        return jnp.mean(state_energies(values))

    def real_coordinates(values: jax.Array) -> jax.Array:
        states = normalized_states(values)
        scale = jnp.sqrt(jnp.asarray(states.shape[0], dtype=states.real.dtype))
        return jnp.concatenate((jnp.real(states), jnp.imag(states)), axis=1) / scale

    def horizontal_coordinates(
        values: jax.Array, vector: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        states, tangents = jax.jvp(
            normalized_states, (values,), (project_tangent(vector),)
        )
        overlaps = jnp.sum(jnp.conj(states) * tangents, axis=1)
        horizontal = tangents - states * overlaps[:, None]
        scale = jnp.sqrt(jnp.asarray(states.shape[0], dtype=states.real.dtype))
        coordinates = jnp.concatenate(
            (jnp.real(horizontal), jnp.imag(horizontal)), axis=1
        ) / scale
        speeds = jnp.real(jnp.sum(jnp.conj(horizontal) * horizontal, axis=1))
        return coordinates, speeds

    @jax.jit
    def natural_direction(values: jax.Array, damping: jax.Array):
        _, pullback = jax.vjp(real_coordinates, values)

        def undamped_metric_product(vector: jax.Array) -> jax.Array:
            horizontal, _ = horizontal_coordinates(values, vector)
            return project_tangent(pullback(horizontal)[0])

        def metric_product(vector: jax.Array) -> jax.Array:
            projected = project_tangent(vector)
            return undamped_metric_product(projected) + damping * projected

        current_energy, gradient = jax.value_and_grad(objective)(values)
        gradient = project_tangent(gradient)
        if linear_solver == "cg":
            direction, _ = cg(
                metric_product,
                gradient,
                tol=cg_tolerance,
                maxiter=cg_max_iterations,
            )
        else:
            # The physical tangent rank is bounded by the number of real
            # normalized-state coordinates, often far below parameter count.
            # Solve the ridge system in that dual space.  The explicit
            # horizontal Jacobian also removes each ray's independent phase.
            states = normalized_states(values)
            coordinate_jacobian = jax.jacrev(real_coordinates)(values)
            dimension = states.shape[1]
            scale = jnp.sqrt(jnp.asarray(states.shape[0], dtype=states.real.dtype))
            jacobian = scale * (
                coordinate_jacobian[:, :dimension, :]
                + 1j * coordinate_jacobian[:, dimension:, :]
            )
            overlaps = jnp.einsum("kd,kdp->kp", jnp.conj(states), jacobian)
            horizontal_jacobian = jacobian - states[:, :, None] * overlaps[:, None, :]
            real_jacobian = jnp.concatenate(
                (jnp.real(horizontal_jacobian), jnp.imag(horizontal_jacobian)), axis=1
            ).reshape((-1, values.size)) / scale
            # Apply the declared parameter-tangent projector to every row.
            real_jacobian = jax.vmap(project_tangent)(real_jacobian)
            dual = real_jacobian @ real_jacobian.T
            dual = dual + damping * jnp.eye(dual.shape[0], dtype=dual.dtype)
            projected_gradient = real_jacobian @ gradient
            correction = jnp.linalg.solve(dual, projected_gradient)
            direction = (gradient - real_jacobian.T @ correction) / damping
        direction = project_tangent(direction)
        residual = metric_product(direction) - gradient
        horizontal, state_speeds = horizontal_coordinates(values, direction)
        metric_speed_squared = jnp.sum(horizontal * horizontal)
        natural_slope = jnp.vdot(gradient, direction).real
        relative_residual = jnp.linalg.norm(residual) / jnp.maximum(
            jnp.linalg.norm(gradient), jnp.finfo(values.dtype).tiny
        )
        return (
            direction,
            current_energy,
            state_energies(values),
            jnp.linalg.norm(gradient),
            jnp.linalg.norm(direction),
            relative_residual,
            metric_speed_squared,
            state_speeds,
            natural_slope,
        )

    @jax.jit
    def candidate_metrics(reference_states: jax.Array, values: jax.Array):
        candidate_states = normalized_states(values)
        energies = jnp.real(
            jnp.sum(jnp.conj(candidate_states) * actions(candidate_states), axis=1)
        )
        overlaps = jnp.sum(jnp.conj(reference_states) * candidate_states, axis=1)
        infidelities = jnp.maximum(0.0, 1.0 - jnp.abs(overlaps) ** 2)
        return energies, infidelities

    compile_started = time.perf_counter()
    initial_states = normalized_states(flat)
    initial_energies = state_energies(flat)
    compiled_direction = natural_direction(
        flat, jnp.asarray(initial_damping, dtype=flat.dtype)
    )
    compiled_candidate = candidate_metrics(initial_states, flat)
    jax.block_until_ready((initial_energies, compiled_direction, compiled_candidate))
    compile_seconds = time.perf_counter() - compile_started

    damping = float(initial_damping)
    history: list[dict[str, object]] = []
    accepted_steps = 0
    rejected_steps = 0
    training_started = time.perf_counter()
    for step in range(1, steps + 1):
        reference_states = normalized_states(flat)
        (
            direction,
            current_energy,
            current_energies,
            gradient_norm,
            direction_norm,
            relative_residual,
            metric_speed_squared,
            state_speeds,
            natural_slope,
        ) = natural_direction(flat, jnp.asarray(damping, dtype=flat.dtype))
        if (
            maximum_linear_relative_residual is not None
            and float(relative_residual) > maximum_linear_relative_residual
        ):
            raise FloatingPointError(
                "product-FS SR linear residual exceeded registered limit: "
                f"{float(relative_residual):.6g} > "
                f"{maximum_linear_relative_residual:.6g}"
            )
        step_size = float(initial_learning_rate)
        accepted = False
        chosen_energies = np.asarray(current_energies, dtype=float)
        chosen_infidelities = np.zeros(len(chosen_energies), dtype=float)
        backtracks = 0
        for backtracks in range(maximum_backtracks + 1):
            candidate = canonicalize(flat - step_size * direction)
            candidate_energies, infidelities = candidate_metrics(
                reference_states, candidate
            )
            candidate_values = np.asarray(candidate_energies, dtype=float)
            infidelity_values = np.asarray(infidelities, dtype=float)
            if (
                np.isfinite(candidate_values).all()
                and np.isfinite(infidelity_values).all()
                and float(np.mean(infidelity_values))
                <= maximum_mean_fs_infidelity
                and float(np.max(infidelity_values))
                <= maximum_state_fs_infidelity
                and float(np.mean(candidate_values))
                < float(current_energy) - 1.0e-14
            ):
                flat = candidate
                chosen_energies = candidate_values
                chosen_infidelities = infidelity_values
                accepted = True
                accepted_steps += 1
                damping = max(float(initial_damping), damping / 1.5)
                break
            step_size *= 0.5
        if not accepted:
            rejected_steps += 1
            damping *= 10.0
        if step % record_every == 0 or step == steps or not accepted:
            history.append(
                {
                    "step": step,
                    "accepted": accepted,
                    "energy_before_update": float(current_energy),
                    "energy": float(np.mean(chosen_energies)),
                    "state_energies_before_update": np.asarray(
                        current_energies, dtype=float
                    ).tolist(),
                    "state_energies": chosen_energies.tolist(),
                    "mean_fs_infidelity": float(np.mean(chosen_infidelities)),
                    "maximum_state_fs_infidelity": float(
                        np.max(chosen_infidelities)
                    ),
                    "state_fs_infidelities": chosen_infidelities.tolist(),
                    "step_size": step_size if accepted else 0.0,
                    "backtracks": backtracks,
                    "damping": damping,
                    "coordinate_gradient_norm": float(gradient_norm),
                    "natural_direction_norm": float(direction_norm),
                    "cg_relative_residual": float(relative_residual),
                    "natural_direction_product_fs_speed": float(
                        jnp.sqrt(metric_speed_squared)
                    ),
                    "natural_direction_state_fs_speeds": np.sqrt(
                        np.asarray(state_speeds, dtype=float)
                    ).tolist(),
                    "damped_natural_gradient_squared": float(natural_slope),
                }
            )
    training_seconds = time.perf_counter() - training_started
    params = unravel(canonicalize(flat))
    final_states = normalized_states(flat)
    final_actions = actions(final_states)
    final_energies = jnp.real(
        jnp.sum(jnp.conj(final_states) * final_actions, axis=1)
    )
    residuals = final_actions - final_energies[:, None] * final_states
    variances = jnp.real(jnp.sum(jnp.conj(residuals) * residuals, axis=1))
    final_direction = natural_direction(
        flat, jnp.asarray(damping, dtype=flat.dtype)
    )
    return params, {
        "optimizer": "exact_product_FS_SR_trust_region",
        "metric": "mean independent-ray Fubini-Study pullback",
        "state_count": int(final_states.shape[0]),
        "steps": steps,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "initial_learning_rate": initial_learning_rate,
        "initial_damping": initial_damping,
        "final_damping": damping,
        "maximum_mean_fs_infidelity": maximum_mean_fs_infidelity,
        "maximum_state_fs_infidelity": maximum_state_fs_infidelity,
        "maximum_backtracks": maximum_backtracks,
        "cg_tolerance": cg_tolerance,
        "cg_max_iterations": cg_max_iterations,
        "linear_solver": linear_solver,
        "maximum_linear_relative_residual": maximum_linear_relative_residual,
        "initial_state_energies": np.asarray(initial_energies, dtype=float).tolist(),
        "final_state_energies": np.asarray(final_energies, dtype=float).tolist(),
        "final_state_variances": np.asarray(variances, dtype=float).tolist(),
        "final_mean_energy": float(jnp.mean(final_energies)),
        "final_coordinate_gradient_norm": float(final_direction[3]),
        "final_damped_natural_gradient_squared": float(final_direction[8]),
        "final_natural_direction_product_fs_speed": float(
            jnp.sqrt(final_direction[6])
        ),
        "compile_seconds": compile_seconds,
        "training_seconds": training_seconds,
        "history": history,
    }


__all__ = [
    "ExactStateFunction",
    "HamiltonianAction",
    "train_exact_state_sr",
    "train_exact_state_sr_trust_region",
    "train_product_state_sr_trust_region",
]
