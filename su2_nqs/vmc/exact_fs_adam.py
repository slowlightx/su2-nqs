"""Bounded exact-state Adam with gauge-invariant Fubini--Study backtracking.

This is a small-support diagnostic optimizer, not a sampled VMC optimizer.
The trust radius is sin(d_FS) = sqrt(1-|<old|new>|^2). It constrains physical
states, not parameter lengths. Equal caps do not imply equal actual distances.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .checkpoint import load_checkpoint, save_checkpoint


def train_exact_fs_adam(
    params: object, *, state_fn: Callable, hamiltonian: np.ndarray,
    identity: dict, checkpoint: Path, steps: int = 256,
    learning_rate: float = 0.001, radius: float = 0.01,
    max_backtracks: int = 12, seconds_limit: float = 300.,
    record_every: int = 32, resume: bool = False, callback: Callable | None = None,
) -> tuple[object, dict]:
    """Jointly train real parameters; save moments and every attempted update.

    Adam moments start at zero in each arm. An uphill momentum direction is
    replaced by the same second-moment-preconditioned gradient (recorded).
    Backtracking accepts only finite, energy-nonincreasing, in-radius states.
    ``steps`` counts attempts, including rejected attempts; no best-seed search.
    The time cap is cumulative active time, checked between complete attempts.
    """
    if steps < 0 or max_backtracks < 0 or record_every < 1:
        raise ValueError("invalid integer optimization budget")
    if not (0 < radius < 1 and learning_rate > 0 and seconds_limit > 0):
        raise ValueError("invalid radius, rate or time cap")
    flat, unravel = ravel_pytree(params)
    if jnp.iscomplexobj(flat):
        raise TypeError("optimizer expects real parameter leaves")
    h = jnp.asarray(hamiltonian)
    contract = dict(identity=identity, steps=steps, learning_rate=learning_rate,
                    radius=radius, max_backtracks=max_backtracks,
                    seconds_limit=seconds_limit, record_every=record_every)

    def evaluate(x):
        psi = state_fn(unravel(x))
        energy = jnp.vdot(psi, h @ psi).real
        return energy, psi

    eval_fn = jax.jit(evaluate)
    grad_fn = jax.jit(jax.value_and_grad(evaluate, has_aux=True))
    report = dict(contract=contract, completed=0, accepted=0, momentum_restarts=0,
                  full_H_calls=0, gradient_calls=0, history=[],
                  active_seconds=0., compile_seconds=0., resumed=resume)
    m, v = jnp.zeros_like(flat), jnp.zeros_like(flat)
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(checkpoint)
        saved = load_checkpoint(checkpoint)
        if saved["metadata"]["contract"] != contract:
            raise ValueError("checkpoint contract mismatch")
        flat = jnp.asarray(saved["params"])
        m, v = map(jnp.asarray, saved["optimizer_state"])
        report = {**saved["metadata"], "resumed": True}
    elif resume:
        raise FileNotFoundError(checkpoint)
    compile_start = perf_counter()
    # Lowering/compiling does not evaluate a new physical state or gradient.
    eval_fn = eval_fn.lower(flat).compile()
    grad_fn = grad_fn.lower(flat).compile()
    report["compile_seconds"] += perf_counter() - compile_start
    previous_active = report["active_seconds"]
    started = perf_counter()

    def persist():
        report["active_seconds"] = previous_active + perf_counter() - started
        save_checkpoint(checkpoint, np.asarray(flat),
                        optimizer_state=(np.asarray(m), np.asarray(v)), metadata=report)

    for step in range(report["completed"], steps):
        if previous_active + perf_counter() - started >= seconds_limit:
            break
        (energy, old), gradient = grad_fn(flat)
        energy = float(energy)
        report["full_H_calls"] += 1
        report["gradient_calls"] += 1
        if not np.isfinite(energy) or not np.isfinite(np.asarray(gradient)).all():
            raise FloatingPointError("nonfinite parent or gradient")
        m = .9*m + .1*gradient
        v = .999*v + .001*gradient**2
        denominator = jnp.sqrt(v/(1-.999**(step+1))) + 1e-8
        direction = -(m/(1-.9**(step+1))) / denominator
        restarted = float(jnp.vdot(gradient, direction).real) >= 0
        if restarted:
            direction = -gradient/denominator
            m = (1-.9**(step+1))*gradient
            report["momentum_restarts"] += 1
        accepted = False
        distance = 0.
        next_energy = energy
        scale = learning_rate
        for backtrack in range(max_backtracks+1):
            candidate = flat + scale*direction
            value, state = eval_fn(candidate)
            report["full_H_calls"] += 1
            value = float(value)
            # Normalize the overlap to make the distance robust to roundoff.
            norm_product = float(jnp.vdot(old, old).real)*float(jnp.vdot(state, state).real)
            fidelity = (abs(complex(jnp.vdot(old, state)))**2 / norm_product
                        if np.isfinite(norm_product) and norm_product > 0 else float("nan"))
            measured = float(np.sqrt(max(0., 1.-fidelity)))
            if np.isfinite(value) and np.isfinite(fidelity) and measured <= radius and value <= energy + 1e-12:
                flat, next_energy, distance, accepted = candidate, value, measured, True
                break
            scale *= .5
        report["completed"] = step+1
        report["accepted"] += int(accepted)
        row = dict(step=step+1, energy=next_energy, accepted=accepted,
                   fs_sine=distance, trial_learning_rate=scale,
                   backtracks=backtrack, momentum_restart=restarted)
        report["history"].append(row)
        # Persist before optional logging/diagnostics so an interrupted callback
        # cannot discard the just-completed update or advance moments twice.
        persist()
        if callback is not None and ((step+1) % record_every == 0 or step+1 == steps):
            callback(unravel(flat), row)
    report["stop"] = "steps" if report["completed"] == steps else "time"
    persist()
    return unravel(flat), report
