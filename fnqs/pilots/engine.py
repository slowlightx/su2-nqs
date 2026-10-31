"""Fixed-size SR with optional, label-free physical-response regularization.

The h derivative is taken at fixed theta. The sample penalty differentiates
normalized importance weights against a frozen Born proposal, including the
Born-measure derivative. Its finite-batch plug-in estimate is biased; it is
an exploratory objective, not a certified susceptibility error bound.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import math

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from fnqs.diagnostics.response import response_records
from fnqs.training import _sr_cg_correction, init_optimizer
from fnqs.variants.size_features import SizeFeatureConfig, init_model, log_psi, sample


@dataclass(frozen=True)
class Options:
    """Predeclared SR limits; never extend a failed solve automatically."""
    samples: int = 64
    learning_rate: float = .5
    shift: float = .001
    tolerance: float = 1e-7
    maxiter: int = 5000
    max_step: float = .05
    response_weight: float = 0.

    def __post_init__(self):
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 2:
            raise ValueError("at least two Born rows required")
        if isinstance(self.maxiter, bool) or not isinstance(self.maxiter, int) or self.maxiter < 1:
            raise ValueError("positive iteration cap required")
        positive = (self.learning_rate, self.shift, self.tolerance, self.max_step)
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("SR scales must be positive and finite")
        if not math.isfinite(self.response_weight) or self.response_weight < 0:
            raise ValueError("response weight must be finite and nonnegative")


def response_loss(records: dict, weights: jax.Array) -> jax.Array:
    """Squared projected response/source norm, using the same Born measure."""
    p = weights / jnp.sum(weights)
    mean = lambda x: jnp.sum(p*x)
    s, e, hs, v = (records[k] for k in ("score", "energy", "h_score", "source"))
    mu, energy = mean(s), mean(e)
    residual = hs-mu*e-energy*(s-mu)+v
    residual -= mean(residual)
    return mean(abs(residual)**2)/(mean(abs(v-mean(v))**2)+1e-12)


def penalty(params: dict, bits: jax.Array, g: jax.Array, proposal: jax.Array,
            config: SizeFeatureConfig) -> jax.Array:
    """Differentiate theta with proposal log amplitudes held constant."""
    amplitude = lambda p, b, x: log_psi(p, b, x, config)
    rows = response_records(params, bits, g, amplitude)
    weights = jax.nn.softmax(2*(amplitude(params, bits, g)-jax.lax.stop_gradient(proposal)))
    return response_loss(rows, weights)


@partial(jax.jit, static_argnames=("config", "options"))
def sampled_block(params: dict, bits: jax.Array, g: jax.Array,
                  config: SizeFeatureConfig, options: Options):
    """Center parameter scores within one Hamiltonian, never across tasks."""
    flat, unravel = ravel_pytree(params)
    amplitude = lambda p, b, x: log_psi(p, b, x, config)
    scores = jax.vmap(jax.grad(lambda v, b: amplitude(unravel(v), b, g)), (None, 0))(flat, bits)
    rows = response_records(params, bits, g, amplitude)
    centered = scores-scores.mean(axis=0)
    energy = rows["energy"]
    energy_force = centered.T@(energy-energy.mean())/(len(bits)*bits.shape[1])
    value = response_loss(rows, jnp.ones(len(bits)))
    if options.response_weight:
        proposal = amplitude(params, bits, g)
        grad = jax.grad(lambda v: penalty(unravel(v), bits, g, proposal, config))(flat)
        energy_force += .5*options.response_weight*grad
    return centered/jnp.sqrt(len(bits)), energy_force, energy.mean()/bits.shape[1], value


def solve(a: jax.Array, rhs: jax.Array, options: Options) -> tuple[jax.Array, float, int]:
    """Arbitrary RHS SR, with true residual and at most three corrections."""
    direction = _sr_cg_correction(a, rhs, options.shift, options.tolerance, options.maxiter)
    operator = lambda x: a.T@(a@x)+options.shift*x
    norm = jnp.maximum(jnp.linalg.norm(rhs), 1e-30)
    residual = float(jnp.linalg.norm(operator(direction)-rhs)/norm)
    refinements = 0
    while np.isfinite(residual) and residual > 1e-6 and refinements < 3:
        direction += _sr_cg_correction(a, rhs-operator(direction), options.shift,
                                       options.tolerance, options.maxiter)
        residual = float(jnp.linalg.norm(operator(direction)-rhs)/norm)
        refinements += 1
    if not np.isfinite(residual) or residual > 1e-6 or not np.isfinite(np.asarray(direction)).all():
        raise RuntimeError(f"SR true residual guard failed: {residual}")
    return direction, residual, refinements


@partial(jax.jit, static_argnames=("count", "n", "config"))
def draw(params, key, count, n, g, config):
    return sample(params, key, count, n, g, config)


def train(params: dict, key: jax.Array, n: int, schedule: list[list[list[float]]],
          config: SizeFeatureConfig, options: Options, callback=None):
    """Run an explicit fixed-size task schedule, resetting optimizer on entry.

    Work counts two H actions and one d_h H action per sampled task, including
    the energy-only control's response diagnostic. CG corrections reuse rows.
    Born rows and logical H actions do not measure differentiated FLOPs.
    """
    state = init_optimizer(params)
    work = {"updates": 0, "task_evaluations": 0, "hamiltonian_actions": 0,
            "source_actions": 0, "born_samples": 0, "basis_rows": 0,
            "sr_correction_solves": 0}
    history = []
    for step, tasks in enumerate(schedule, 1):
        if not tasks:
            raise ValueError("empty update")
        blocks, forces, energies, losses = [], [], [], []
        for task in tasks:
            g = np.asarray(task, dtype=float)
            if g.shape != (4,) or not np.isfinite(g).all() or g[3] <= 0 or g[2] < 0:
                raise ValueError("invalid stoquastic task")
            key, subkey = jax.random.split(key)
            bits = draw(params, subkey, options.samples, n, jnp.asarray(g), config)
            a, force, energy, loss = sampled_block(params, bits, jnp.asarray(g), config, options)
            blocks.append(a); forces.append(force); energies.append(float(energy)); losses.append(float(loss))
        count = len(tasks)
        a = jnp.concatenate(blocks)/jnp.sqrt(count)
        rhs = jnp.stack(forces).mean(axis=0)
        direction, residual, corrections = solve(a, rhs, options)
        flat, unravel = ravel_pytree(params)
        scale = min(options.learning_rate, options.max_step/max(float(jnp.linalg.norm(direction)), 1e-30))
        params = unravel(flat-scale*direction)
        work["updates"] += 1; work["task_evaluations"] += count
        work["hamiltonian_actions"] += 2*count; work["source_actions"] += count
        work["born_samples"] += options.samples*count
        work["sr_correction_solves"] += corrections
        row = {"step": step, "tasks": tasks, "energy_per_site": energies,
               "response_loss": losses, "true_sr_residual": residual,
               "step_norm": float(scale*jnp.linalg.norm(direction)), "sr_corrections": corrections}
        history.append(row)
        state["step"] = step; state["work"] = dict(work)
        if callback is not None:
            callback(params, state, key, history)
    return params, state, key, history


def model_signature(config: SizeFeatureConfig, n: int) -> dict:
    """Feature policy is part of ancestry identity, even with equal shapes."""
    return {"architecture": "positive_z2_gru", "n": n, **asdict(config)}
