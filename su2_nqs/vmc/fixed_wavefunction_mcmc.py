"""Reference Metropolis and local-energy estimators for fixed wavefunctions.

This module is deliberately model agnostic.  It is intended for immutable
physical trial states such as projected parton or analytic tower states whose
batched amplitudes are naturally evaluated outside a differentiable JAX VMC
loop.  Fixed-``Sz`` up--down exchanges give a symmetric, ergodic proposal on
the requested sector; no ansatz-specific repair of exact nodes is performed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..operators.spin_config import SpinBond
from .spin_physical import padded_spin_connectivity


BatchAmplitude = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FixedWavefunctionSamples:
    """Autocorrelated samples and proposal diagnostics from parallel chains."""

    configurations: np.ndarray
    amplitudes: np.ndarray
    chain_count: int
    measurement_blocks: int
    acceptance_fraction: float
    burn_acceptance_fraction: float


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _evaluate(amplitude_fn: BatchAmplitude, states: np.ndarray) -> np.ndarray:
    values = np.asarray(amplitude_fn(np.asarray(states, dtype=np.int8)))
    if values.shape != (len(states),):
        raise ValueError("amplitude_fn must return one amplitude per state")
    if np.any(~np.isfinite(values)):
        raise FloatingPointError("fixed wavefunction returned non-finite amplitudes")
    return values.astype(np.complex128, copy=False)


def _initial_states(
    generator: np.random.Generator,
    *,
    chain_count: int,
    n_sites: int,
    n_up: int,
) -> np.ndarray:
    states = np.zeros((chain_count, n_sites), dtype=np.int8)
    for state in states:
        state[generator.choice(n_sites, n_up, replace=False)] = 1
    return states


def _exchange_step(
    generator: np.random.Generator,
    states: np.ndarray,
    amplitudes: np.ndarray,
    amplitude_fn: BatchAmplitude,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proposed = np.array(states, copy=True)
    for row, state in enumerate(states):
        up = np.flatnonzero(state == 1)
        down = np.flatnonzero(state == 0)
        first = int(up[generator.integers(len(up))])
        second = int(down[generator.integers(len(down))])
        proposed[row, first] = 0
        proposed[row, second] = 1
    candidate = _evaluate(amplitude_fn, proposed)
    current_magnitude = np.abs(amplitudes)
    candidate_magnitude = np.abs(candidate)
    log_ratio = np.full(len(states), -np.inf, dtype=np.float64)
    ordinary = (current_magnitude > 0.0) & (candidate_magnitude > 0.0)
    log_ratio[ordinary] = 2.0 * (
        np.log(candidate_magnitude[ordinary])
        - np.log(current_magnitude[ordinary])
    )
    log_ratio[(current_magnitude == 0.0) & (candidate_magnitude > 0.0)] = np.inf
    accepted = np.log(generator.random(len(states))) < np.minimum(0.0, log_ratio)
    states = np.where(accepted[:, None], proposed, states)
    amplitudes = np.where(accepted, candidate, amplitudes)
    return states, amplitudes, accepted


def sample_fixed_wavefunction_exchange(
    amplitude_fn: BatchAmplitude,
    *,
    n_sites: int,
    n_up: int,
    chain_count: int,
    burn_sweeps: int,
    measurement_blocks: int,
    decorrelation_sweeps: int,
    seed: int,
) -> FixedWavefunctionSamples:
    """Sample ``|psi|^2`` using non-null global up--down exchanges.

    One sweep is one proposed up--down exchange per chain.  Returned rows are
    ordered first by measurement block and then by chain, which permits a
    conservative chain-mean uncertainty estimate without reconstructing the
    Markov history.
    """

    n_sites = _positive_integer(n_sites, "n_sites")
    chain_count = _positive_integer(chain_count, "chain_count")
    burn_sweeps = _positive_integer(burn_sweeps, "burn_sweeps")
    measurement_blocks = _positive_integer(
        measurement_blocks, "measurement_blocks"
    )
    decorrelation_sweeps = _positive_integer(
        decorrelation_sweeps, "decorrelation_sweeps"
    )
    if isinstance(n_up, bool) or not isinstance(n_up, Integral):
        raise TypeError("n_up must be an integer")
    if not 0 < int(n_up) < n_sites:
        raise ValueError("n_up must lie strictly between zero and n_sites")
    generator = np.random.default_rng(int(seed))
    states = _initial_states(
        generator,
        chain_count=chain_count,
        n_sites=n_sites,
        n_up=int(n_up),
    )
    amplitudes = _evaluate(amplitude_fn, states)
    burn_accepted = 0
    for _ in range(burn_sweeps):
        states, amplitudes, accepted = _exchange_step(
            generator, states, amplitudes, amplitude_fn
        )
        burn_accepted += int(np.count_nonzero(accepted))

    recorded_states = []
    recorded_amplitudes = []
    measurement_accepted = 0
    for _ in range(measurement_blocks):
        for _ in range(decorrelation_sweeps):
            states, amplitudes, accepted = _exchange_step(
                generator, states, amplitudes, amplitude_fn
            )
            measurement_accepted += int(np.count_nonzero(accepted))
        recorded_states.append(np.array(states, copy=True))
        recorded_amplitudes.append(np.array(amplitudes, copy=True))
    return FixedWavefunctionSamples(
        configurations=np.concatenate(recorded_states, axis=0),
        amplitudes=np.concatenate(recorded_amplitudes, axis=0),
        chain_count=chain_count,
        measurement_blocks=measurement_blocks,
        acceptance_fraction=float(
            measurement_accepted
            / (chain_count * measurement_blocks * decorrelation_sweeps)
        ),
        burn_acceptance_fraction=float(
            burn_accepted / (chain_count * burn_sweeps)
        ),
    )


def fixed_wavefunction_local_energies(
    amplitude_fn: BatchAmplitude,
    configurations: np.ndarray,
    bonds: Iterable[SpinBond],
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """Evaluate exact local energies on supplied physical configurations."""

    batch_size = _positive_integer(batch_size, "batch_size")
    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("configurations must be a nonempty rank-two array")
    checked_bonds = tuple(bonds)
    output = []
    for start in range(0, len(states), batch_size):
        batch = states[start : start + batch_size]
        connected, elements, valid, _ = padded_spin_connectivity(
            batch, checked_bonds
        )
        centers = _evaluate(amplitude_fn, batch)
        connected_values = np.zeros(elements.shape, dtype=np.complex128)
        connected_values[valid] = _evaluate(amplitude_fn, connected[valid])
        actions = np.sum(elements * connected_values, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            local = actions / centers
        if np.any(~np.isfinite(local)):
            raise FloatingPointError("fixed wavefunction produced non-finite local energy")
        output.append(local)
    return np.concatenate(output)


def chain_mean_standard_error(
    values: np.ndarray, *, chain_count: int, measurement_blocks: int
) -> float:
    """Return the standard error across persistent-chain means."""

    chain_count = _positive_integer(chain_count, "chain_count")
    measurement_blocks = _positive_integer(
        measurement_blocks, "measurement_blocks"
    )
    array = np.asarray(values)
    if array.shape != (chain_count * measurement_blocks,):
        raise ValueError("values do not match the recorded block/chain shape")
    chain_means = array.reshape((measurement_blocks, chain_count)).mean(axis=0)
    if chain_count == 1:
        return 0.0
    return float(np.std(chain_means.real, ddof=1) / np.sqrt(chain_count))


__all__ = [
    "BatchAmplitude",
    "FixedWavefunctionSamples",
    "chain_mean_standard_error",
    "fixed_wavefunction_local_energies",
    "sample_fixed_wavefunction_exchange",
]
