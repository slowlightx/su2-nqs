"""Budgeted exact-support calibration, independent of model and Hamiltonian.

Adam explores the full parameter space; L-BFGS then diagnoses whether a
finite-step Adam plateau was an optimizer limitation. Neither is a sampled
production optimizer. Checkpoints retain Adam moments, best/current states,
and evaluation counts. L-BFGS curvature is explicitly restarted on resume.
"""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize

from .adam import _manual_adam_init, _manual_adam_update
from .checkpoint import load_checkpoint, save_checkpoint


def train_bounded_exact(
    params: object,
    *,
    loss_fn: Callable,
    identity: dict,
    checkpoint: Path,
    adam_steps: int,
    lbfgs_steps: int,
    seconds_per_stage: float,
    learning_rate: float = 0.003,
    record_every: int = 50,
    callback: Callable | None = None,
    resume: bool = False,
) -> tuple[object, dict]:
    """Return the lowest evaluated loss state, with both terminal states logged.

    Budgets are cumulative across resumes and checked between evaluations;
    compilation and callback diagnostics are reported separately. One running
    evaluation cannot be preempted. L-BFGS function evaluations are capped at
    five times its iteration budget. Exact objective values, never ED states,
    select the best checkpoint. All parameters receive gradients.
    """
    if min(adam_steps, lbfgs_steps, record_every) < 1:
        raise ValueError("positive step and record budgets required")
    if not np.isfinite(seconds_per_stage) or seconds_per_stage <= 0:
        raise ValueError("positive finite time budget required")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("positive finite learning rate required")
    flat, unravel = ravel_pytree(params)
    if not np.issubdtype(flat.dtype, np.floating):
        raise TypeError("real parameter coordinates required")
    contract = dict(identity=identity, adam_steps=adam_steps, lbfgs_steps=lbfgs_steps,
                    seconds_per_stage=seconds_per_stage, learning_rate=learning_rate,
                    record_every=record_every)
    report = dict(contract=contract, adam_completed=0, lbfgs_completed=0,
                  evaluations=0, history=[], stage_seconds=dict(adam=0., lbfgs=0.),
                  compile_seconds=0., diagnostic_seconds=0., resumed=False,
                  lbfgs_curvature_restarted=False, status="running")
    moments = _manual_adam_init(flat)
    best, best_loss = flat, float("inf")
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(checkpoint)
        saved = load_checkpoint(checkpoint)
        if saved["metadata"]["contract"] != contract:
            raise ValueError("checkpoint contract differs")
        report = saved["metadata"]
        flat = jnp.asarray(saved["params"]["current"])
        best = jnp.asarray(saved["params"]["best"])
        best_loss = float(report["best_loss"])
        moments = jax.tree_util.tree_map(jnp.asarray, saved["optimizer_state"])
        report["resumed"] = True
        report["lbfgs_curvature_restarted"] = report["lbfgs_completed"] > 0
        if report["status"] == "bounded_complete":
            return unravel(best), report

    value_grad = jax.jit(jax.value_and_grad(lambda x: loss_fn(unravel(x))))
    update = jax.jit(_manual_adam_update)
    start = perf_counter()
    value, gradient = value_grad(flat)
    jax.block_until_ready((value, gradient))
    jax.block_until_ready(update(flat, gradient, moments, learning_rate))
    report["compile_seconds"] += perf_counter() - start
    report["evaluations"] += 1
    if not np.isfinite(float(value)) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("nonfinite initial objective/gradient")

    def consider(x, loss):
        nonlocal best, best_loss
        if float(loss) < best_loss:
            best, best_loss = jnp.array(x), float(loss)

    def save():
        report["best_loss"] = best_loss
        save_checkpoint(checkpoint, dict(current=flat, best=best),
                        optimizer_state=moments, metadata=report)

    def record(stage, step, x, loss, grad):
        started = perf_counter()
        row = dict(stage=stage, step=int(step), loss=float(loss),
                   gradient_norm=float(np.linalg.norm(grad)),
                   evaluations=report["evaluations"],
                   active_seconds=sum(report["stage_seconds"].values()))
        if callback is not None:
            row.update(callback(unravel(x), row))
        report["history"].append(row)
        report["diagnostic_seconds"] += perf_counter() - started

    consider(flat, value)
    if not report["history"]:
        record("initial", 0, flat, value, gradient)
    save()
    while (report["adam_completed"] < adam_steps and
           report["stage_seconds"]["adam"] < seconds_per_stage):
        started = perf_counter()
        fraction = report["adam_completed"] / adam_steps
        rate = learning_rate * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * fraction)))
        proposed, next_moments = update(flat, gradient, moments, rate)
        value_new, grad_new = value_grad(proposed)
        jax.block_until_ready((value_new, grad_new))
        report["stage_seconds"]["adam"] += perf_counter() - started
        report["evaluations"] += 1
        if not np.isfinite(float(value_new)) or not np.all(np.isfinite(grad_new)):
            report["adam_stop"] = "nonfinite_proposal_rejected"
            break
        flat, moments, value, gradient = proposed, next_moments, value_new, grad_new
        report["adam_completed"] += 1
        consider(flat, value)
        if report["adam_completed"] % record_every == 0:
            record("adam", report["adam_completed"], flat, value, gradient)
            save()
    report.setdefault("adam_stop", "steps" if report["adam_completed"] == adam_steps else "time")
    # A resumed L-BFGS checkpoint must not relabel its current loss as Adam's
    # terminal result. Preserve the already recorded stage boundary.
    if "adam_terminal_loss" not in report:
        record("adam_terminal", report["adam_completed"], flat, value, gradient)
        report["adam_terminal_loss"] = float(value)
        report["adam_best_loss"] = best_loss
    save()

    # Start deterministic refinement at the best evaluated Adam state, not ED.
    flat = best
    class BudgetReached(Exception):
        pass

    def objective(x):
        if report["stage_seconds"]["lbfgs"] >= seconds_per_stage:
            raise BudgetReached
        started = perf_counter()
        v, g = value_grad(jnp.asarray(x))
        v, g = float(v), np.asarray(g, dtype=float)
        report["stage_seconds"]["lbfgs"] += perf_counter() - started
        report["evaluations"] += 1
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            raise FloatingPointError("nonfinite L-BFGS objective/gradient")
        consider(x, v)
        return v, g

    def accepted(intermediate_result):
        nonlocal flat
        flat = jnp.asarray(intermediate_result.x)
        report["lbfgs_completed"] += 1
        if report["lbfgs_completed"] % record_every == 0:
            v, g = objective(flat)
            record("lbfgs", report["lbfgs_completed"], flat, v, g)
            save()

    remaining = lbfgs_steps - report["lbfgs_completed"]
    if remaining > 0 and report["stage_seconds"]["lbfgs"] < seconds_per_stage:
        try:
            result = minimize(objective, np.asarray(flat), jac=True, method="L-BFGS-B",
                              callback=accepted,
                              options=dict(maxiter=remaining, maxfun=5*remaining,
                                           ftol=1e-14, gtol=1e-8, maxls=25, maxcor=20))
            flat = jnp.asarray(result.x)
            report["lbfgs_stop"] = str(result.message)
            report["lbfgs_terminal_loss"] = float(result.fun)
        except BudgetReached:
            report["lbfgs_stop"] = "time"
        except FloatingPointError as error:
            report["lbfgs_stop"] = str(error)
    report["status"] = "bounded_complete"
    value, gradient = value_grad(best)
    jax.block_until_ready((value, gradient))
    report["evaluations"] += 1
    record("best", report["adam_completed"]+report["lbfgs_completed"], best, value, gradient)
    save()
    return unravel(best), report
