"""Ensemble energy optimization, exact and independent-sample VMC controls.

Different sizes receive equal weight in energy PER SITE; within each size all
couplings receive equal weight. Parameter derivatives are centered separately
for every Hamiltonian, as required by multi-system SR. Optax is not required.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg

from .basis import all_bits, validate_size
from .hamiltonians import IsingCouplings
from .models import GRUConfig, log_psi, sample


@dataclass(frozen=True)
class Task:
    """One physical ensemble member; no class label enters the neural model."""

    n: int
    couplings: IsingCouplings
    family: str = "A"

    def __post_init__(self) -> None:
        validate_size(self.n)
        if self.couplings.h <= 0 or self.couplings.jx < 0:
            raise ValueError("positive-amplitude training requires h>0 and jx>=0")


def local_energies(params: dict, bits: jax.Array, couplings: jax.Array,
                   config: GRUConfig) -> jax.Array:
    """All terms of the Pauli super-Hamiltonian; no MCMC or sign approximation."""
    bits, couplings = jnp.asarray(bits), jnp.asarray(couplings)
    if bits.ndim != 2 or couplings.shape != (4,):
        raise ValueError("bits must be (batch,n), couplings (4,)")
    batch, n = bits.shape
    validate_size(n)
    # uint8 is a common storage dtype: subtracting one before promotion wraps
    # spin -1 to 255 and corrupts antiparallel ZZ bonds.
    z = 2 * bits.astype(jnp.int32) - 1
    diagonal = -couplings[0] * jnp.sum(z * jnp.roll(z, -1, axis=1), axis=1)
    diagonal -= couplings[1] * jnp.sum(z * jnp.roll(z, -2, axis=1), axis=1)
    masks = jnp.eye(n, dtype=jnp.int32)
    single = jnp.bitwise_xor(bits[:, None, :], masks[None, :, :])
    pair_masks = jnp.bitwise_xor(masks, jnp.roll(masks, 1, axis=1))
    pairs = jnp.bitwise_xor(bits[:, None, :], pair_masks[None, :, :])
    old_logs = log_psi(params, bits, couplings, config)
    new_logs = log_psi(params, jnp.concatenate((single, pairs), axis=1).reshape(batch * 2 * n, n), couplings, config)
    ratios = jnp.exp(new_logs.reshape(batch, 2 * n) - old_logs[:, None])
    return diagonal - couplings[3] * ratios[:, :n].sum(axis=1) - couplings[2] * ratios[:, n:].sum(axis=1)


def exact_statistics(params: dict, n: int, couplings: jax.Array, config: GRUConfig) -> tuple[jax.Array, jax.Array]:
    """Exhaustive total energy and variance, n<=10, differentiable in params."""
    if n > 10:
        raise ValueError("exact training restricted to n<=10")
    bits = jnp.asarray(all_bits(n))
    psi = jnp.exp(log_psi(params, bits, couplings, config))
    z = 2 * bits - 1
    diagonal = -couplings[0] * jnp.sum(z * jnp.roll(z, -1, axis=1), axis=1)
    diagonal -= couplings[1] * jnp.sum(z * jnp.roll(z, -2, axis=1), axis=1)
    labels = jnp.arange(1 << n)
    masks = jnp.asarray(1 << np.arange(n - 1, -1, -1))
    pair_masks = masks ^ jnp.roll(masks, -1)
    hpsi = diagonal * psi - couplings[3] * psi[labels[:, None] ^ masks].sum(axis=1)
    hpsi -= couplings[2] * psi[labels[:, None] ^ pair_masks].sum(axis=1)
    energy = psi @ hpsi
    variance = jnp.sum((hpsi - energy * psi) ** 2)
    return energy, variance


@partial(jax.jit, static_argnames=("n", "config"))
def _exact_gradient(params: dict, n: int, couplings: jax.Array, config: GRUConfig):
    def objective(p):
        energy, variance = exact_statistics(p, n, couplings, config)
        return energy / n, (energy, variance)
    (_, statistics), gradient = jax.value_and_grad(objective, has_aux=True)(params)
    return gradient, statistics


@partial(jax.jit, static_argnames=("n", "config"))
def _exact_ensemble_statistics(params: dict, n: int, couplings: jax.Array,
                               config: GRUConfig):
    return jax.vmap(lambda g: exact_statistics(params, n, g, config))(couplings)


@partial(jax.jit, static_argnames=("n", "config"))
def _exact_ensemble_gradient(params: dict, n: int, couplings: jax.Array,
                             config: GRUConfig):
    def objective(p):
        energies, variances = _exact_ensemble_statistics(p, n, couplings, config)
        return energies.mean() / n, (energies, variances)
    (_, statistics), gradient = jax.value_and_grad(objective, has_aux=True)(params)
    return gradient, statistics


@partial(jax.jit, static_argnames=("n", "n_samples", "config"))
def _vmc_gradient(params: dict, key: jax.Array, n: int, n_samples: int,
                  couplings: jax.Array, config: GRUConfig):
    bits = sample(params, key, n_samples, n, couplings, config)
    energies = jax.lax.stop_gradient(local_energies(params, bits, couplings, config))
    centered = energies - energies.mean()
    gradient = jax.grad(lambda p: 2 * jnp.mean(log_psi(p, bits, couplings, config) * centered) / n)(params)
    return gradient, (energies.mean(), jnp.var(energies))


@partial(jax.jit, static_argnames=("config",))
def _parameter_scores(params: dict, bits: jax.Array, couplings: jax.Array, config: GRUConfig):
    flat, unravel = ravel_pytree(params)
    # Independent scalar gradients avoid the quadratic batch-by-batch reverse
    # Jacobian intermediates of differentiating the whole output vector.
    return jax.vmap(jax.grad(lambda p, b: log_psi(unravel(p), b, couplings, config)),
                    in_axes=(None, 0))(flat, bits)


@partial(jax.jit, static_argnames=("n", "config"))
def _exact_sr_blocks(params: dict, n: int, couplings: jax.Array, config: GRUConfig):
    bits = jnp.asarray(all_bits(n))
    labels = jnp.arange(1 << n)
    masks = jnp.asarray(1 << np.arange(n - 1, -1, -1))
    pair_masks = masks ^ jnp.roll(masks, -1)
    z = 2 * bits - 1

    def block(g):
        psi = jnp.exp(log_psi(params, bits, g, config))
        p = psi**2
        diagonal = -g[0] * jnp.sum(z * jnp.roll(z, -1, axis=1), axis=1)
        diagonal -= g[1] * jnp.sum(z * jnp.roll(z, -2, axis=1), axis=1)
        hpsi = diagonal * psi - g[3] * psi[labels[:, None] ^ masks].sum(axis=1)
        hpsi -= g[2] * psi[labels[:, None] ^ pair_masks].sum(axis=1)
        energy = psi @ hpsi
        variance = jnp.sum((hpsi - energy * psi)**2)
        scores = _parameter_scores(params, bits, g, config)
        centered = scores - p @ scores
        root = jnp.sqrt(p)
        return root[:, None] * centered, root * (hpsi / psi - energy) / n, energy, variance

    return jax.vmap(block)(couplings)


def _group_tasks(tasks: list[Task]) -> dict[int, list[Task]]:
    groups: dict[int, list[Task]] = {}
    for task in tasks:
        groups.setdefault(task.n, []).append(task)
    return groups


@partial(jax.jit, static_argnames=("n", "n_samples", "config"))
def _sampled_sr_block(params: dict, key: jax.Array, n: int, n_samples: int,
                      couplings: jax.Array, config: GRUConfig):
    """One compiled Hamiltonian block, centered within its independent draws."""
    bits = sample(params, key, n_samples, n, couplings, config)
    scores = _parameter_scores(params, bits, couplings, config)
    energies = local_energies(params, bits, couplings, config)
    root = 1 / jnp.sqrt(n_samples)
    return (root * (scores - scores.mean(axis=0)),
            root * (energies - energies.mean()) / n,
            energies.mean(), energies.var())


def _ensemble_sr_data(params: dict, tasks: list[Task], keys: list[jax.Array],
                      config: GRUConfig, mode: str, n_samples: int):
    blocks, targets, statistics = [], [], []
    if mode == "exact":
        for n, group in _group_tasks(tasks).items():
            a, target, energies, variances = _exact_sr_blocks(
                params, n, jnp.asarray([t.couplings.vector() for t in group]), config)
            blocks.append(a.reshape(-1, a.shape[-1]) / jnp.sqrt(len(tasks)))
            targets.append(target.reshape(-1) / jnp.sqrt(len(tasks)))
            statistics.extend((n, e, v) for e, v in zip(energies, variances, strict=True))
    else:
        for task, key in zip(tasks, keys, strict=True):
            g = jnp.asarray(task.couplings.vector())
            a, target, energy, variance = _sampled_sr_block(
                params, key, task.n, n_samples, g, config)
            blocks.append(a / jnp.sqrt(len(tasks)))
            targets.append(target / jnp.sqrt(len(tasks)))
            statistics.append((task.n, energy, variance))
    return jnp.concatenate(blocks), jnp.concatenate(targets), statistics


@partial(jax.jit, static_argnames=("solver", "maxiter"))
def _solve_sr_once(a: jax.Array, target: jax.Array, shift: float, *,
              solver: str, tolerance: float, maxiter: int):
    force = a.T @ target
    if solver == "cg":
        # Explicit score rows, but no dense parameter metric or cubic solve.
        diagonal = jnp.sum(a*a, axis=0) + shift
        operator = lambda x: a.T @ (a @ x) + shift*x
        direction, _ = cg(operator, force, tol=tolerance, maxiter=maxiter,
                          M=lambda x: x/diagonal)
    elif a.shape[0] < a.shape[1]:
        direction = a.T @ jnp.linalg.solve(a @ a.T + shift*jnp.eye(a.shape[0]), target)
    else:
        direction = jnp.linalg.solve(a.T @ a + shift*jnp.eye(a.shape[1]), force)
    residual = jnp.linalg.norm(a.T @ (a @ direction) + shift*direction - force)
    residual /= jnp.maximum(jnp.linalg.norm(force), 1e-30)
    return direction, residual, 2*jnp.linalg.norm(force)


@partial(jax.jit, static_argnames=("maxiter",))
def _sr_cg_correction(a: jax.Array, rhs: jax.Array, shift: float,
                      tolerance: float, maxiter: int) -> jax.Array:
    """Solve the recomputed residual equation with the original SR operator."""
    diagonal = jnp.sum(a*a, axis=0) + shift
    operator = lambda x: a.T @ (a @ x) + shift*x
    correction, _ = cg(operator, rhs, tol=tolerance, maxiter=maxiter,
                       M=lambda x: x/diagonal)
    return correction


def _solve_sr_refined(a: jax.Array, target: jax.Array, shift: float, *,
                      solver: str, tolerance: float, maxiter: int):
    """Repair recursive-CG residual drift; retain the original stop threshold.

    Up to three correction solves reuse the same score rows and force. They
    introduce no extra Hamiltonian evaluations or Born draws. Nonfinite or
    persistently inaccurate solutions still reach the caller's failure guard.
    Successful initial solves return their original arrays unchanged.
    """
    direction, residual, gradient = _solve_sr_once(
        a,target,shift,solver=solver,tolerance=tolerance,maxiter=maxiter)
    refinements = 0
    limit = max(1e-7,10*tolerance)
    while (solver == "cg" and refinements < 3 and np.isfinite(float(residual))
           and float(residual) > limit):
        force = a.T @ target
        rhs = force - (a.T @ (a @ direction) + shift*direction)
        direction = direction + _sr_cg_correction(a,rhs,shift,tolerance,maxiter)
        residual = jnp.linalg.norm(a.T @ (a @ direction) + shift*direction - force)
        residual /= jnp.maximum(jnp.linalg.norm(force),1e-30)
        refinements += 1
    return direction,residual,gradient,refinements


def _solve_sr(a: jax.Array, target: jax.Array, shift: float, *,
              solver: str, tolerance: float, maxiter: int):
    """Return the stable SR direction, true residual and gradient norm."""
    return _solve_sr_refined(a,target,shift,solver=solver,tolerance=tolerance,maxiter=maxiter)[:3]


def ensemble_sr_direction(params: dict, tasks: list[Task], keys: list[jax.Array],
                          config: GRUConfig, mode: str, n_samples: int,
                          shift: float = 1e-3, *, solver: str = "dense",
                          tolerance: float = 1e-8, maxiter: int = 500) -> dict:
    """Regularized multi-system natural gradient with per-system centering.

    S=sum_k w_k Cov_k(d_theta log psi), f=sum_k w_k Cov_k(O,E/N).
    Solve (S+shift I) d=f, so imaginary-time step is theta-=learning_rate*d.
    Coupling QGT is deliberately not used as the parameter SR matrix.
    """
    if not tasks or len(keys)!=len(tasks) or mode not in {"exact","vmc"}:
        raise ValueError("invalid SR ensemble, keys or mode")
    if mode=="exact" and max(t.n for t in tasks)>10:
        raise ValueError("exact SR restricted to n<=10")
    if shift <= 0 or not np.isfinite(shift):
        raise ValueError("SR shift must be finite and positive")
    if solver not in {"dense", "cg"} or not 0 < tolerance < 1 or maxiter < 1:
        raise ValueError("invalid SR solver/tolerance/iterations")
    a, target, _ = _ensemble_sr_data(params, tasks, keys, config, mode, n_samples)
    _, unravel = ravel_pytree(params)
    direction, residual, _ = _solve_sr(a, target, shift, solver=solver,
                                      tolerance=tolerance, maxiter=maxiter)
    if not np.isfinite(float(residual)) or float(residual) > max(1e-7, 10*tolerance):
        raise FloatingPointError(f"SR linear solve did not converge: relative residual={float(residual)}")
    return unravel(direction)


def init_optimizer(params: dict) -> dict:
    """Adam state, also storing the global update counter for exact resumes."""
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "step": 0,
            "work": {"gradient_hamiltonians": 0, "hamiltonian_evaluations": 0,
                     "basis_state_evaluations": 0, "born_samples": 0}}


@jax.jit
def _adam_update(params: dict, gradient: dict, m: dict, v: dict, step: int, learning_rate: float):
    m = jax.tree.map(lambda old, g: 0.9 * old + 0.1 * g, m, gradient)
    v = jax.tree.map(lambda old, g: 0.999 * old + 0.001 * g * g, v, gradient)
    params = jax.tree.map(lambda p, first, second: p - learning_rate * (first / (1 - 0.9 ** step))
                         / (jnp.sqrt(second / (1 - 0.999 ** step)) + 1e-8), params, m, v)
    return params, m, v


def train(params: dict, tasks: list[Task], config: GRUConfig, key: jax.Array, *,
          steps: int = 100, learning_rate: float = 0.003, mode: str = "exact",
          n_samples: int = 128, optimizer: str = "adam", sr_shift: float = 1e-3,
          sr_solver: str = "dense", sr_tolerance: float = 1e-8, sr_maxiter: int = 500,
          sr_max_step_norm: float | None = None, sr_backtracking: bool = False,
          tasks_per_step: int | None = None,
          state: dict | None = None, callback: Callable[[dict],None] | None = None) -> tuple[dict, dict, jax.Array, list[dict]]:
    """Optimize one shared state; return resumable params/optimizer/PRNG/history.

    A fixed total VMC sample budget is represented by n_samples PER TASK; the
    config records total=len(tasks)*n_samples. Compare arms at matched totals.
    """
    if not tasks or mode not in {"exact", "vmc"} or optimizer not in {"adam", "sr"}:
        raise ValueError("invalid tasks, mode or optimizer")
    if steps < 1 or n_samples < 2 or learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("positive steps/rate and >=2 samples required")
    if mode == "exact" and max(t.n for t in tasks) > 10:
        raise ValueError("exact training requires all sizes<=10")
    if tasks_per_step is not None and (isinstance(tasks_per_step, bool) or
            not isinstance(tasks_per_step, int) or not 1 <= tasks_per_step <= len(tasks)):
        raise ValueError("tasks_per_step must be between one and ensemble size")
    if sr_solver not in {"dense", "cg"} or not 0 < sr_tolerance < 1 or sr_maxiter < 1:
        raise ValueError("invalid SR solver controls")
    if sr_shift <= 0 or not np.isfinite(sr_shift):
        raise ValueError("SR shift must be finite and positive")
    if sr_max_step_norm is not None and (not np.isfinite(sr_max_step_norm) or sr_max_step_norm <= 0):
        raise ValueError("SR step norm must be finite and positive")
    if sr_backtracking and (mode != "exact" or optimizer != "sr"):
        raise ValueError("deterministic backtracking requires exact SR")
    state = init_optimizer(params) if state is None else state
    history = []
    for iteration in range(steps):
        active_tasks = tasks
        if tasks_per_step is not None:
            key, selection_key = jax.random.split(key)
            if tasks_per_step < len(tasks):
                indices = np.asarray(jax.random.choice(selection_key, len(tasks),
                                                       (tasks_per_step,), replace=False))
                active_tasks = [tasks[int(i)] for i in indices]
        key, *keys = jax.random.split(key, len(active_tasks) + 1)
        gradients, statistics = [], []
        sr_residual, backtracks, effective_rate, sr_refinements = None, 0, learning_rate, 0
        if optimizer == "sr":
            a, target, statistics = _ensemble_sr_data(params, active_tasks, keys, config, mode, n_samples)
            flat, unravel = ravel_pytree(params)
            direction, sr_residual, gradient_norm, sr_refinements = _solve_sr_refined(
                a, target, sr_shift, solver=sr_solver,tolerance=sr_tolerance,maxiter=sr_maxiter)
            if not np.isfinite(float(sr_residual)) or float(sr_residual) > max(1e-7, 10*sr_tolerance):
                raise FloatingPointError(f"SR solve residual={float(sr_residual)}")
        elif mode == "exact":
            for n, group in _group_tasks(active_tasks).items():
                gradient, (energies, variances) = _exact_ensemble_gradient(
                    params, n, jnp.asarray([t.couplings.vector() for t in group]), config)
                gradients.append(jax.tree.map(lambda g: g*len(group)/len(active_tasks), gradient))
                statistics.extend((n, e, v) for e, v in zip(energies, variances, strict=True))
            gradient = jax.tree.map(lambda *gs: sum(gs), *gradients)
            gradient_norm = jnp.linalg.norm(ravel_pytree(gradient)[0])
        else:
            for task, task_key in zip(active_tasks, keys, strict=True):
                g = jnp.asarray(task.couplings.vector())
                gradient, stat = _vmc_gradient(params, task_key, task.n, n_samples, g, config)
                gradients.append(gradient)
                statistics.append((task.n, *stat))
            gradient = jax.tree.map(lambda *gs: sum(gs)/len(gs), *gradients)
            gradient_norm = jnp.linalg.norm(ravel_pytree(gradient)[0])
        step = int(state["step"]) + 1
        if optimizer == "adam":
            params, m, v = _adam_update(params, gradient, state["m"], state["v"], step, learning_rate)
        else:
            if sr_max_step_norm is not None:
                effective_rate = min(learning_rate, sr_max_step_norm/max(float(jnp.linalg.norm(direction)), 1e-30))
            candidate = unravel(flat - effective_rate*direction)
            if sr_backtracking:
                before = float(np.mean([float(e)/n for n, e, _ in statistics]))
                while True:
                    after = []
                    for n, group in _group_tasks(active_tasks).items():
                        energies, _ = _exact_ensemble_statistics(candidate, n,
                            jnp.asarray([t.couplings.vector() for t in group]), config)
                        after.extend(np.asarray(energies)/n)
                    if np.isfinite(after).all() and np.mean(after) <= before + 1e-13:
                        break
                    backtracks += 1
                    effective_rate *= .5
                    candidate = unravel(flat - effective_rate*direction)
                    if backtracks == 12:
                        candidate = params
                        effective_rate = 0.
                        break
            params = candidate
            m, v = state["m"], state["v"]
        trials = (min(backtracks + 1, 12) if sr_backtracking else 0)
        multiplier = 1 + trials
        work = dict(state.get("work", {}))
        for name, increment in {
            "gradient_hamiltonians": len(active_tasks),
            "hamiltonian_evaluations": multiplier*len(active_tasks),
            "basis_state_evaluations": multiplier*sum(1 << t.n for t in active_tasks) if mode == "exact" else 0,
            "born_samples": len(active_tasks)*n_samples if mode == "vmc" else 0}.items():
            work[name] = work.get(name, 0) + increment
        if optimizer == "sr":
            work["sr_refinement_solves"] = work.get("sr_refinement_solves",0) + sr_refinements
        state = {"m": m, "v": v, "step": step, "work": work}
        if iteration == 0 or (iteration + 1) % 25 == 0 or iteration == steps - 1:
            record = {"step": step, "energy_per_site": float(np.mean([float(e)/n for n, e, _ in statistics])),
                      "variance_per_site": float(np.mean([float(v)/n for n, _, v in statistics])),
                      "max_variance_per_site": float(max(float(v)/n for n, _, v in statistics)),
                      "gradient_norm": float(gradient_norm), "tasks_evaluated": len(active_tasks)}
            if optimizer == "sr":
                record.update(sr_relative_residual=float(sr_residual), effective_learning_rate=float(effective_rate),
                              backtracks=backtracks,sr_refinement_solves=sr_refinements)
            record["cumulative_work"] = dict(work)
            if not np.isfinite(record["energy_per_site"] + record["variance_per_site"]):
                raise FloatingPointError("nonfinite training statistics")
            # Statistics are measured BEFORE the update labeled step.
            record["measurement"] = "pre_update"
            history.append(record)
            if callback is not None:
                callback(record)
    return params, state, key, history
