"""Fixed-work energy continuation from arbitrary saved real parameters.

The caller supplies a differentiable cached state_pair(x)=(psi,Hpsi), with
TOTAL-H units, and one fixed positive proposal Q. No carrier, sampler or
Hamiltonian implementation is called here. The full finite-Q Rayleigh ratio
differentiates both bra and ket, and physical FS matching only selects a step
length, never an energy-selected checkpoint or optimization direction.

All fresh arms start with zero Adam moments, including continuation from a
previously trained state. Checkpoints atomically include parameters, moments,
history and run identity. Only trusted locally generated pickle checkpoints
may be loaded. An external workflow remains responsible for cumulative wall
and resource budgets; uncommitted interrupted work cannot be reconstructed
from a completed-step checkpoint and is never presented as exactly counted.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .fixed_proposal_rayleigh import weighted_full_h_loss, weighted_full_h_moments
from .matched_state_step import matched_state_step
from .resumable_checkpoint import (
    canonical_json_sha256, load_resumable_checkpoint, save_resumable_checkpoint,
)


@dataclass(frozen=True)
class CachedEnergyConfig:
    """A fixed endpoint and the unchanged Adam-direction/FS step protocol."""

    steps: int = 60
    n_sites: int = 36
    target_fs: float = .001
    max_alpha: float = .5
    bisection_steps: int = 11
    beta1: float = .9
    beta2: float = .999
    adam_epsilon: float = 1e-8
    fs_relative_tolerance: float = .01
    fs_absolute_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for name, minimum in (('steps', 0), ('n_sites', 1), ('bisection_steps', 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        for name in ('target_fs', 'max_alpha', 'adam_epsilon', 'fs_relative_tolerance', 'fs_absolute_tolerance'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.target_fs >= np.pi/2 or self.fs_relative_tolerance >= 1 or self.fs_absolute_tolerance >= self.target_fs:
            raise ValueError('unresolvable FS radius or matching tolerance')
        for name in ('beta1', 'beta2'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f'{name} must be finite in [0,1)')


@dataclass(frozen=True)
class CachedEnergyResult:
    """Final fixed-step parameters and cached physical values, not an audit."""

    parameters: np.ndarray
    first_moment: np.ndarray
    second_moment: np.ndarray
    amplitudes: np.ndarray
    h_amplitudes: np.ndarray
    history: tuple[dict[str, Any], ...]
    completed_steps: int
    resumed_from_step: int
    final_moments: dict[str, float]
    ledger: dict[str, Any]
    active_seconds: float

    def as_dict(self) -> dict[str, Any]:
        """JSON summary; the caller saves/seals parameter endpoints separately."""
        return dict(parameter_count=int(self.parameters.size), completed_steps=self.completed_steps,
            resumed_from_step=self.resumed_from_step, final_moments=self.final_moments,
            history=list(self.history), ledger=self.ledger, active_seconds=self.active_seconds,
            checkpoint_selection='fixed final step; no energy selection',
            independent_audit_performed=False)


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256(str(contiguous.dtype).encode())
    digest.update(repr(contiguous.shape).encode()); digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _real_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) == 0 or np.iscomplexobj(array):
        raise ValueError(f'{name} must be a nonempty real vector')
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite')
    return array


def _moment_record(psi: jax.Array, hp: jax.Array, q: jax.Array, n_sites: int) -> dict[str, float]:
    moments = weighted_full_h_moments(psi, hp, q)
    result = dict(energy_per_site=float(moments.energy_complex.real/n_sites),
        imaginary_energy_per_site=float(moments.energy_complex.imag/n_sites),
        variance_per_site=float(moments.variance/n_sites),
        second_moment_total=float(moments.second_moment), norm_sum=float(moments.norm_sum))
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError('nonfinite cached state moments')
    return result


def run_cached_energy_continuation(
    initial_parameters: jax.Array,
    state_pair: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    q: np.ndarray,
    config: CachedEnergyConfig,
    checkpoint_path: str | Path,
    run_identity: Mapping[str, Any],
    *,
    resume: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> CachedEnergyResult:
    """Continue one arm with full gradients and atomic deterministic resume.

    ``state_pair`` must be pure, deterministic, and return matching amplitude
    vectors in total-H units on exactly the supplied Q corpus. The caller's
    ``run_identity`` must seal the model, anchor, feature data and numerical
    sources; this helper additionally seals config, initial parameters and Q.
    No RNG or sample index advances. Resume cannot change the endpoint budget.

    ``progress(record)`` runs only AFTER a complete checkpoint commit, once
    per completed step. A callback exception therefore leaves an exact resume
    point. The ledger counts committed steps, not unknown partially executed
    work before an interruption; the caller must enforce its global wall cap.
    """
    if not isinstance(config, CachedEnergyConfig) or not callable(state_pair):
        raise TypeError('config and a callable state_pair are required')
    if not isinstance(run_identity, Mapping) or not run_identity:
        raise ValueError('a nonempty caller run identity is required')
    initial = _real_vector(initial_parameters, 'initial_parameters')
    proposal = _real_vector(q, 'q')
    if np.any(proposal <= 0):
        raise ValueError('q must be strictly positive including nodes')
    identity = dict(schema='cached_energy_continuation_v1', config=asdict(config),
        caller=dict(run_identity), initial_parameters_sha256=_array_hash(initial),
        proposal_sha256=_array_hash(proposal))
    canonical_json_sha256(identity)  # Fail before any write on invalid metadata.
    path = Path(checkpoint_path)
    if path.exists() and not resume:
        raise FileExistsError('checkpoint exists; explicit unchanged-protocol resume required')
    if resume and not path.exists():
        raise FileNotFoundError('resume requested without a committed checkpoint')
    x = jnp.asarray(initial); m = jnp.zeros_like(x); v = jnp.zeros_like(x)
    history: list[dict[str, Any]] = []; prior_seconds = 0.; prior_validations = 0
    if resume:
        saved = load_resumable_checkpoint(path, expected_run_identity=identity)
        if saved['next_sample_index'] != 0:
            raise ValueError('cached fixed-Q continuation cannot advance sample indices')
        completed = saved['completed_steps']
        if completed > config.steps or [row.get('step') for row in saved['history']] != list(range(1, completed+1)):
            raise ValueError('checkpoint history exceeds or disagrees with the fixed endpoint')
        optimizer = saved['optimizer_state']
        if not isinstance(optimizer, dict) or set(optimizer) != {'m', 'v'}:
            raise ValueError('checkpoint needs exactly both Adam moment vectors')
        restored = [_real_vector(value, name) for value, name in
                    ((saved['params'], 'checkpoint parameters'), (optimizer['m'], 'Adam m'), (optimizer['v'], 'Adam v'))]
        if any(value.shape != initial.shape for value in restored) or np.any(restored[2] < 0):
            raise ValueError('checkpoint parameter/moment shape or second-moment sign is invalid')
        x, m, v = map(jnp.asarray, restored); history = [dict(row) for row in saved['history']]
        prior_seconds = float(saved['metadata'].get('active_seconds', 0.))
        prior_validations = int(saved['metadata'].get('completed_invocation_validations', 0))
        if not np.isfinite(prior_seconds) or prior_seconds < 0 or prior_validations < 0:
            raise ValueError('invalid checkpoint cumulative metadata')
    resumed_from = len(history)
    started = perf_counter(); state = jax.jit(state_pair)

    def loss(params):
        psi, hp = state_pair(params)
        return weighted_full_h_loss(psi, hp/config.n_sites, jnp.asarray(proposal))

    gradient_fn = jax.jit(jax.value_and_grad(loss))
    psi, hp = state(x)
    if psi.shape != proposal.shape or hp.shape != proposal.shape:
        raise ValueError('state_pair and fixed Q shapes differ')
    _moment_record(psi, hp, jnp.asarray(proposal), config.n_sites)
    validations = prior_validations+1

    def commit() -> None:
        save_resumable_checkpoint(path, np.asarray(x), optimizer_state=dict(m=np.asarray(m), v=np.asarray(v)),
            completed_steps=len(history), next_sample_index=0, run_identity=identity, history=history,
            metadata=dict(active_seconds=prior_seconds+perf_counter()-started,
                          completed_invocation_validations=validations,
                          optimizer_origin='zero moments at this continuation initial state',
                          uncommitted_interrupted_work_not_reconstructed=True))

    if not resume:
        commit()  # Explicit recoverable zero-moment initial checkpoint.
    for step in range(resumed_from, config.steps):
        objective, gradient = gradient_fn(x); jax.block_until_ready(gradient)
        if not np.isfinite(float(objective)) or not np.isfinite(np.asarray(gradient)).all():
            raise FloatingPointError('nonfinite complete cached-energy gradient')
        next_m = config.beta1*m+(1-config.beta1)*gradient
        next_v = config.beta2*v+(1-config.beta2)*gradient**2
        direction = -(next_m/(1-config.beta1**(step+1)))/(jnp.sqrt(next_v/(1-config.beta2**(step+1)))+config.adam_epsilon)
        direction /= jnp.maximum(jnp.linalg.norm(direction), 1e-300)
        before, _ = state(x)
        matched = matched_state_step(lambda alpha:np.asarray(state(x+alpha*direction)[0]),
            np.asarray(before), proposal, config.target_fs, config.max_alpha, config.bisection_steps,
            relative_tolerance=config.fs_relative_tolerance, absolute_tolerance=config.fs_absolute_tolerance)
        next_x = x+matched.alpha*direction
        next_psi, next_hp = state(next_x)
        moments = _moment_record(next_psi, next_hp, jnp.asarray(proposal), config.n_sites)
        # Commit only after the whole proposed physical state has been checked.
        x, m, v = next_x, next_m, next_v; psi, hp = next_psi, next_hp
        row = dict(step=step+1, loss_before_per_site=float(objective), **moments,
            gradient_norm=float(jnp.linalg.norm(gradient)), physical_step=matched.as_dict(),
            active_seconds=prior_seconds+perf_counter()-started)
        history.append(row); commit()
        if progress is not None:
            progress(dict(stage='cached_energy_step_committed', completed_steps=step+1,
                active_seconds=prior_seconds+perf_counter()-started,
                energy_per_site=moments['energy_per_site'], checkpoint=str(path)))
    ledger = dict(completed_gradient_calls=len(history),
        completed_step_candidate_calls=len(history)*(config.bisection_steps+1),
        completed_cached_state_pair_evaluations=len(history)*(config.bisection_steps+4),
        completed_invocation_validation_state_pair_evaluations=validations,
        this_invocation_gradient_calls=len(history)-resumed_from,
        this_invocation_validation_state_pair_evaluations=1,
        new_samples=0, new_H_queries=0, optimizer_moments_reset_at_start=True,
        count_scope='committed training steps plus invocation validations; partial uncommitted work is unknown and constrained by external cumulative wall budget')
    return CachedEnergyResult(np.asarray(x), np.asarray(m), np.asarray(v), np.asarray(psi), np.asarray(hp),
        tuple(history), len(history), resumed_from, _moment_record(psi, hp, jnp.asarray(proposal), config.n_sites),
        ledger, prior_seconds+perf_counter()-started)


__all__ = ['CachedEnergyConfig', 'CachedEnergyResult', 'run_cached_energy_continuation']
