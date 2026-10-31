"""Matrix-free JAX Heisenberg action for the strict projected-AGP carrier.

Hamiltonian logic is kept out of :mod:`su2_nqs.models`.  This module evaluates
``psi`` and ``H psi`` directly, without dividing by ``psi`` and without
enumerating a Hilbert space.  The direct-amplitude form remains meaningful at
nodes and is differentiable with respect to the AGP pairing parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import Mapping, NamedTuple, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from ..models.projected_singlet_agp import (
    ProjectedAGPBatch,
    ProjectedSingletAGPConfig,
    projected_singlet_agp_amplitudes,
)


GraphBond = tuple[int, int, float]


@dataclass(frozen=True)
class ProjectedAGPHeisenbergConfig:
    """Static graph contract suitable for checkpoint metadata."""

    n_sites: int
    bonds: tuple[GraphBond, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_sites, bool)
            or not isinstance(self.n_sites, Integral)
            or int(self.n_sites) < 2
        ):
            raise ValueError("n_sites must be an integer at least two")
        n_sites = int(self.n_sites)
        normalized: list[GraphBond] = []
        for raw in self.bonds:
            if len(raw) != 3:
                raise ValueError("every graph bond must contain (i, j, coupling)")
            first, second, coupling = raw
            if (
                isinstance(first, bool)
                or isinstance(second, bool)
                or not isinstance(first, Integral)
                or not isinstance(second, Integral)
            ):
                raise TypeError("bond endpoints must be integers")
            first, second = int(first), int(second)
            coupling = float(coupling)
            if (
                first == second
                or not 0 <= first < n_sites
                or not 0 <= second < n_sites
            ):
                raise ValueError("bond endpoints must be distinct valid sites")
            if not np.isfinite(coupling):
                raise ValueError("bond couplings must be finite")
            normalized.append((first, second, coupling))
        if not normalized:
            raise ValueError("at least one graph bond is required")
        object.__setattr__(self, "n_sites", n_sites)
        object.__setattr__(self, "bonds", tuple(normalized))

    @property
    def bond_count(self) -> int:
        return len(self.bonds)

    def checkpoint_metadata(self) -> dict[str, object]:
        encoded = json.dumps(
            self.bonds, separators=(",", ":"), sort_keys=False
        ).encode("utf-8")
        return {
            "schema": "projected_agp_heisenberg_jax_v1",
            "n_sites": self.n_sites,
            "bond_count": self.bond_count,
            "bonds_sha256": hashlib.sha256(encoded).hexdigest(),
            "action": "direct_Hpsi_no_local_energy_division",
        }


class ProjectedAGPHeisenbergBatch(NamedTuple):
    """Differentiable direct-amplitude action and node-aware local energy."""

    amplitudes: jax.Array
    hamiltonian_actions: jax.Array
    local_energies: jax.Array
    center_exact_nodes: jax.Array
    minimum_connected_cancellation_ratio: jax.Array


class DirectAmplitudeRayleighEstimate(NamedTuple):
    """Self-normalized direct-amplitude importance estimate."""

    energy: jax.Array
    complex_numerator: jax.Array
    denominator: jax.Array
    numerator_imaginary_fraction: jax.Array
    residual_norm: jax.Array
    relative_residual_norm: jax.Array
    state_weight_ess: jax.Array
    state_weight_ess_fraction: jax.Array
    amplitude_rescale: jax.Array
    log_quadrature_rescale: jax.Array
    valid: jax.Array


def _all_bond_flips(
    configurations: jax.Array, first: jax.Array, second: jax.Array
) -> jax.Array:
    def flip_one(left: jax.Array, right: jax.Array) -> jax.Array:
        original_left = configurations[:, left]
        original_right = configurations[:, right]
        return configurations.at[:, left].set(original_right).at[:, right].set(
            original_left
        )

    # vmap produces (bond, batch, site); put batch first for the action sum.
    return jnp.swapaxes(jax.vmap(flip_one)(first, second), 0, 1)


def projected_agp_heisenberg_action(
    parameters: Mapping[str, jax.Array],
    configurations: jax.Array | np.ndarray | Sequence[Sequence[int]],
    carrier_config: ProjectedSingletAGPConfig,
    graph_config: ProjectedAGPHeisenbergConfig,
    *,
    amplitude_microbatch_size: int | None = None,
) -> ProjectedAGPHeisenbergBatch:
    """Evaluate ``psi`` and ``H psi`` for a batch without Hilbert enumeration.

    Each bond contributes one fixed-width swapped configuration.  Parallel-spin
    swaps are retained with zero coefficient; this static representation is
    accelerator friendly and keeps gradients entirely in JAX.
    """

    if graph_config.n_sites != carrier_config.n_sites:
        raise ValueError("carrier and graph must have the same site count")
    states = jnp.asarray(configurations, dtype=jnp.int8)
    if (
        states.ndim != 2
        or states.shape[0] < 1
        or states.shape[1] != carrier_config.n_sites
    ):
        raise ValueError("configurations must have shape (nonzero_batch, n_sites)")
    bonds = np.asarray(graph_config.bonds, dtype=np.float64)
    first = jnp.asarray(bonds[:, 0], dtype=jnp.int32)
    second = jnp.asarray(bonds[:, 1], dtype=jnp.int32)
    couplings = jnp.asarray(bonds[:, 2], dtype=jnp.float64)

    flipped = _all_bond_flips(states, first, second)
    connected = jnp.concatenate((states[:, None, :], flipped), axis=1)
    flat_connected = connected.reshape((-1, carrier_config.n_sites))
    evaluated: ProjectedAGPBatch = projected_singlet_agp_amplitudes(
        parameters,
        flat_connected,
        carrier_config,
        microbatch_size=amplitude_microbatch_size,
    )
    batch_size = states.shape[0]
    width = graph_config.bond_count + 1
    amplitudes = evaluated.amplitudes.reshape((batch_size, width))
    cancellations = evaluated.cancellation_ratios.reshape((batch_size, width))
    exact_nodes = evaluated.exact_nodes.reshape((batch_size, width))

    left = states[:, first]
    right = states[:, second]
    antiparallel = left != right
    diagonal = jnp.sum(
        couplings[None, :] * jnp.where(antiparallel, -0.25, 0.25), axis=1
    )
    off_diagonal = 0.5 * couplings[None, :] * antiparallel
    action = diagonal * amplitudes[:, 0] + jnp.sum(
        off_diagonal * amplitudes[:, 1:], axis=1
    )
    center = amplitudes[:, 0]
    center_nodes = exact_nodes[:, 0]
    local_energy = jnp.where(center_nodes, jnp.nan + 0.0j, action / center)
    return ProjectedAGPHeisenbergBatch(
        amplitudes=center,
        hamiltonian_actions=action,
        local_energies=local_energy,
        center_exact_nodes=center_nodes,
        minimum_connected_cancellation_ratio=jnp.min(cancellations, axis=1),
    )


def direct_amplitude_rayleigh_estimate(
    action: ProjectedAGPHeisenbergBatch,
    proposal_probabilities: jax.Array | np.ndarray | Sequence[float],
    *,
    observation_weights: jax.Array | np.ndarray | Sequence[float] | None = None,
) -> DirectAmplitudeRayleighEstimate:
    r"""Estimate a Rayleigh quotient without ``log(psi)`` or ``Hpsi/psi``.

    For samples drawn from a normalized positive proposal ``q``, omit
    ``observation_weights`` and the empirical weights are ``1/M``.  For exact
    deterministic quadrature, pass normalized observation weights explicitly
    (for example ``q`` when every support point is visited once).  Exact nodes
    are allowed because neither summand divides by ``psi``.

    The finite-sample ratio is generally biased.  Its variance and the state
    weight ESS must be calibrated with the eventual learned proposal plus a
    registered uniform floor; this function does not infer global support from
    a finite batch.
    """

    amplitudes = jnp.asarray(action.amplitudes)
    hamiltonian_actions = jnp.asarray(action.hamiltonian_actions)
    if amplitudes.ndim != 1 or hamiltonian_actions.shape != amplitudes.shape:
        raise ValueError("action amplitudes and Hpsi must be matching vectors")
    host_q = np.asarray(proposal_probabilities, dtype=np.float64)
    if host_q.shape != amplitudes.shape:
        raise ValueError("proposal_probabilities must match the action batch")
    if np.any(~np.isfinite(host_q)) or np.any(host_q <= 0.0):
        raise ValueError("proposal probabilities must be finite and strictly positive")
    q = jnp.asarray(host_q, dtype=jnp.float64)
    if observation_weights is None:
        weights = jnp.ones_like(q) / len(q)
    else:
        host_weights = np.asarray(observation_weights, dtype=np.float64)
        if host_weights.shape != amplitudes.shape:
            raise ValueError("observation_weights must match the action batch")
        if np.any(~np.isfinite(host_weights)) or np.any(host_weights < 0.0):
            raise ValueError("observation weights must be finite and nonnegative")
        total = float(np.sum(host_weights))
        if total <= 0.0:
            raise ValueError("observation weights must have positive total")
        weights = jnp.asarray(host_weights / total, dtype=jnp.float64)

    # Both the Rayleigh quotient and residual ratio are invariant under one
    # global state scale and one global quadrature scale.  Remove both before
    # products/squares so otherwise harmless AGP homogeneity and tiny proposal
    # probabilities do not overflow the estimator.
    amplitude_rescale = jnp.maximum(
        jnp.maximum(
            jnp.max(jnp.abs(amplitudes)),
            jnp.max(jnp.abs(hamiltonian_actions)),
        ),
        jnp.finfo(jnp.float64).tiny,
    )
    scaled_amplitudes = amplitudes / amplitude_rescale
    scaled_actions = hamiltonian_actions / amplitude_rescale
    log_measure = jnp.where(
        weights > 0.0, jnp.log(weights) - jnp.log(q), -jnp.inf
    )
    log_quadrature_rescale = jnp.max(log_measure)
    quadrature_measure = jnp.where(
        jnp.isfinite(log_measure),
        jnp.exp(log_measure - log_quadrature_rescale),
        0.0,
    )
    importance_state_weights = quadrature_measure * jnp.abs(scaled_amplitudes) ** 2
    denominator = jnp.sum(importance_state_weights)
    complex_numerator = jnp.sum(
        quadrature_measure * jnp.conj(scaled_amplitudes) * scaled_actions
    )
    energy = jnp.real(complex_numerator) / denominator
    imaginary_fraction = jnp.abs(jnp.imag(complex_numerator)) / jnp.maximum(
        jnp.abs(jnp.real(complex_numerator)), jnp.finfo(jnp.float64).tiny
    )
    weight_sum = jnp.sum(importance_state_weights)
    ess = weight_sum**2 / jnp.sum(importance_state_weights**2)
    residual_squared = jnp.sum(
        quadrature_measure
        * jnp.abs(scaled_actions - energy * scaled_amplitudes) ** 2
    ) / denominator
    residual_norm = jnp.sqrt(jnp.maximum(residual_squared, 0.0))
    relative_residual_norm = residual_norm / jnp.maximum(
        jnp.abs(energy), jnp.finfo(jnp.float64).tiny
    )
    valid = (
        jnp.isfinite(denominator)
        & (denominator > 0.0)
        & jnp.isfinite(energy)
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(ess)
        & jnp.isfinite(imaginary_fraction)
    )
    return DirectAmplitudeRayleighEstimate(
        energy=energy,
        complex_numerator=complex_numerator,
        denominator=denominator,
        numerator_imaginary_fraction=imaginary_fraction,
        residual_norm=residual_norm,
        relative_residual_norm=relative_residual_norm,
        state_weight_ess=ess,
        state_weight_ess_fraction=ess / len(q),
        amplitude_rescale=amplitude_rescale,
        log_quadrature_rescale=log_quadrature_rescale,
        valid=valid,
    )


def require_valid_direct_amplitude_estimate(
    estimate: DirectAmplitudeRayleighEstimate,
) -> None:
    """Fail closed on a materialized zero/nonfinite direct estimate."""

    if not bool(np.asarray(estimate.valid)):
        raise FloatingPointError(
            "direct-amplitude Rayleigh estimate has zero norm or nonfinite "
            "energy/residual/ESS diagnostics"
        )


__all__ = [
    "DirectAmplitudeRayleighEstimate",
    "ProjectedAGPHeisenbergBatch",
    "ProjectedAGPHeisenbergConfig",
    "direct_amplitude_rayleigh_estimate",
    "projected_agp_heisenberg_action",
    "require_valid_direct_amplitude_estimate",
]
