"""Common fixed-sector exchange Metropolis sampler for physical NQS inputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral

import jax
import jax.numpy as jnp


PhysicalAmplitude = Callable[[object, jax.Array], jax.Array]
ExchangeEdge = tuple[int, int]


def _checked_chain_shape(states: jax.Array) -> jax.Array:
    array = jnp.asarray(states, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError("states must have shape (chains, n_sites>=2)")
    return array


def canonical_exchange_edges(
    edges: Sequence[Sequence[int]], *, n_sites: int
) -> tuple[ExchangeEdge, ...]:
    """Validate, orient and deduplicate undirected physical exchange edges."""

    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 2:
        raise ValueError("n_sites must be an integer at least two")
    canonical: set[ExchangeEdge] = set()
    for edge in edges:
        raw = tuple(edge)
        if len(raw) != 2 or any(
            isinstance(site, bool) or not isinstance(site, Integral) for site in raw
        ):
            raise ValueError("each exchange edge must contain two integer sites")
        first, second = (int(raw[0]), int(raw[1]))
        if not (0 <= first < n_sites and 0 <= second < n_sites):
            raise ValueError("exchange-edge site lies outside the lattice")
        if first == second:
            raise ValueError("exchange edges cannot be self loops")
        canonical.add(tuple(sorted((first, second))))
    if not canonical:
        raise ValueError("at least one exchange edge is required")
    return tuple(sorted(canonical))


def initialize_spin_exchange_chains(
    key: jax.Array,
    *,
    n_chains: int,
    n_sites: int,
    n_up: int,
) -> jax.Array:
    """Draw independent uniform binary configurations in fixed ``Sz``."""

    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in (n_chains, n_sites, n_up)
    ):
        raise TypeError("chain and sector sizes must be integers")
    if n_chains < 1 or n_sites < 2 or not 0 <= n_up <= n_sites:
        raise ValueError("invalid spin-chain or fixed-Sz dimensions")
    scores = jax.random.uniform(key, (int(n_chains), int(n_sites)))
    selected = jnp.argsort(scores, axis=1)[:, : int(n_up)]
    rows = jnp.arange(int(n_chains), dtype=jnp.int32)[:, None]
    return jnp.zeros((int(n_chains), int(n_sites)), dtype=jnp.int32).at[
        rows, selected
    ].set(1)


def initialize_hubbard_exchange_chains(
    key: jax.Array,
    *,
    n_chains: int,
    n_sites: int,
    n_up: int,
    n_down: int,
) -> jax.Array:
    """Draw uniform physical Fock tokens at fixed ``(N_up,N_down)``."""

    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in (n_chains, n_sites, n_up, n_down)
    ):
        raise TypeError("chain and sector sizes must be integers")
    if (
        n_chains < 1
        or n_sites < 2
        or not 0 <= n_up <= n_sites
        or not 0 <= n_down <= n_sites
    ):
        raise ValueError("invalid Hubbard-chain or fixed-sector dimensions")
    up_key, down_key = jax.random.split(key)
    up = initialize_spin_exchange_chains(
        up_key, n_chains=n_chains, n_sites=n_sites, n_up=n_up
    )
    down = initialize_spin_exchange_chains(
        down_key, n_chains=n_chains, n_sites=n_sites, n_up=n_down
    )
    return up + 2 * down


def spin_exchange_proposals(
    key: jax.Array, states: jax.Array
) -> jax.Array:
    """Propose a symmetric two-site swap for every spin chain."""

    array = _checked_chain_shape(states)
    n_chains, n_sites = array.shape
    site_key, offset_key = jax.random.split(key)
    first = jax.random.randint(site_key, (n_chains,), 0, n_sites)
    offset = jax.random.randint(offset_key, (n_chains,), 1, n_sites)
    second = (first + offset) % n_sites
    rows = jnp.arange(n_chains, dtype=jnp.int32)
    first_values = array[rows, first]
    second_values = array[rows, second]
    return array.at[rows, first].set(second_values).at[rows, second].set(first_values)


def spin_bond_exchange_proposals(
    key: jax.Array,
    states: jax.Array,
    exchange_edges: Sequence[Sequence[int]],
) -> jax.Array:
    """Propose a symmetric spin exchange on a physical Hamiltonian edge."""

    array = _checked_chain_shape(states)
    n_chains, n_sites = array.shape
    edges = jnp.asarray(
        canonical_exchange_edges(exchange_edges, n_sites=n_sites), dtype=jnp.int32
    )
    selected = jax.random.randint(key, (n_chains,), 0, edges.shape[0])
    first, second = edges[selected, 0], edges[selected, 1]
    rows = jnp.arange(n_chains, dtype=jnp.int32)
    first_values = array[rows, first]
    second_values = array[rows, second]
    return array.at[rows, first].set(second_values).at[rows, second].set(first_values)


def hubbard_exchange_proposals(
    key: jax.Array, states: jax.Array
) -> jax.Array:
    """Propose a symmetric fixed-spin-channel occupation swap per chain."""

    tokens = _checked_chain_shape(states)
    if not jnp.issubdtype(tokens.dtype, jnp.integer):
        raise TypeError("Hubbard tokens must be integers")
    n_chains, n_sites = tokens.shape
    site_key, offset_key, spin_key = jax.random.split(key, 3)
    first = jax.random.randint(site_key, (n_chains,), 0, n_sites)
    offset = jax.random.randint(offset_key, (n_chains,), 1, n_sites)
    second = (first + offset) % n_sites
    spin = jax.random.randint(spin_key, (n_chains,), 0, 2)
    occupied = jnp.stack((tokens & 1, (tokens >> 1) & 1), axis=-1)
    rows = jnp.arange(n_chains, dtype=jnp.int32)
    first_values = occupied[rows, first, spin]
    second_values = occupied[rows, second, spin]
    occupied = occupied.at[rows, first, spin].set(second_values)
    occupied = occupied.at[rows, second, spin].set(first_values)
    return occupied[..., 0] + 2 * occupied[..., 1]


def hubbard_bond_exchange_proposals(
    key: jax.Array,
    states: jax.Array,
    exchange_edges: Sequence[Sequence[int]],
) -> jax.Array:
    """Propose one spin-resolved fermion hop along a physical bond.

    Occupation swapping is equivalent to a hop when exactly one endpoint is
    occupied and otherwise gives a null proposal.  Uniform selection of an
    undirected bond and spin species makes the proposal kernel symmetric.
    Wavefunction phase (including the signed spatial projector) is evaluated
    by ``amplitude_fn``; canonical hopping-matrix signs belong separately to
    the local-energy connectivity and never enter this proposal probability.
    """

    tokens = _checked_chain_shape(states)
    n_chains, n_sites = tokens.shape
    edges = jnp.asarray(
        canonical_exchange_edges(exchange_edges, n_sites=n_sites), dtype=jnp.int32
    )
    edge_key, spin_key = jax.random.split(key)
    selected = jax.random.randint(edge_key, (n_chains,), 0, edges.shape[0])
    first, second = edges[selected, 0], edges[selected, 1]
    spin = jax.random.randint(spin_key, (n_chains,), 0, 2)
    occupied = jnp.stack((tokens & 1, (tokens >> 1) & 1), axis=-1)
    rows = jnp.arange(n_chains, dtype=jnp.int32)
    first_values = occupied[rows, first, spin]
    second_values = occupied[rows, second, spin]
    occupied = occupied.at[rows, first, spin].set(second_values)
    occupied = occupied.at[rows, second, spin].set(first_values)
    return occupied[..., 0] + 2 * occupied[..., 1]


def metropolis_exchange_step(
    params: object,
    key: jax.Array,
    states: jax.Array,
    amplitudes: jax.Array,
    *,
    amplitude_fn: PhysicalAmplitude,
    particle_kind: str,
    exchange_edges: Sequence[Sequence[int]] | None = None,
    bond_proposal_probability: float = 1.0,
    lazy_probability: float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Apply one detailed-balance exchange step to parallel chains.

    Returns ``states, amplitudes, accepted, nontrivial_proposal``.  The cached
    amplitudes avoid reevaluating the current state at every sweep.  When
    ``exchange_edges`` is supplied, ``bond_proposal_probability`` selects a
    state-independent mixture of the Hamiltonian-bond and global exchange
    kernels.  Both components are symmetric involution mixtures, so the
    ordinary Born-ratio Metropolis acceptance remains valid.
    """

    array = _checked_chain_shape(states)
    current = jnp.asarray(amplitudes)
    if current.shape != (array.shape[0],):
        raise ValueError("one cached amplitude is required per chain")
    if not 0.0 <= float(lazy_probability) < 1.0:
        raise ValueError("lazy_probability must lie in [0, 1)")
    if not 0.0 <= float(bond_proposal_probability) <= 1.0:
        raise ValueError("bond_proposal_probability must lie in [0, 1]")
    proposal_key, acceptance_key, lazy_key, mixture_key = jax.random.split(key, 4)
    if particle_kind == "spin":
        global_proposed = spin_exchange_proposals(proposal_key, array)
        if exchange_edges is None:
            proposed = global_proposed
        else:
            bond_proposed = spin_bond_exchange_proposals(
                proposal_key, array, exchange_edges
            )
            choose_bond = jax.random.uniform(
                mixture_key, (array.shape[0],)
            ) < float(bond_proposal_probability)
            proposed = jnp.where(choose_bond[:, None], bond_proposed, global_proposed)
    elif particle_kind == "hubbard":
        global_proposed = hubbard_exchange_proposals(proposal_key, array)
        if exchange_edges is None:
            proposed = global_proposed
        else:
            bond_proposed = hubbard_bond_exchange_proposals(
                proposal_key, array, exchange_edges
            )
            choose_bond = jax.random.uniform(
                mixture_key, (array.shape[0],)
            ) < float(bond_proposal_probability)
            proposed = jnp.where(choose_bond[:, None], bond_proposed, global_proposed)
    else:
        raise ValueError("particle_kind must be 'spin' or 'hubbard'")
    proposed_amplitudes = jax.vmap(lambda row: amplitude_fn(params, row))(proposed)
    # Compare Born weights in the log domain.  Projected tree amplitudes can
    # be exponentially small even when their *ratios* are perfectly regular;
    # forming ``abs(psi)**2`` first would therefore underflow long before the
    # physical Metropolis decision becomes singular.
    current_magnitude = jnp.abs(current)
    proposed_magnitude = jnp.abs(proposed_amplitudes)
    positive_current = current_magnitude > 0.0
    positive_proposed = proposed_magnitude > 0.0
    safe_current = jnp.where(positive_current, current_magnitude, 1.0)
    safe_proposed = jnp.where(positive_proposed, proposed_magnitude, 1.0)
    log_acceptance_ratio = 2.0 * (
        jnp.log(safe_proposed) - jnp.log(safe_current)
    )
    log_acceptance_probability = jnp.where(
        positive_current,
        jnp.where(
            positive_proposed,
            jnp.minimum(0.0, log_acceptance_ratio),
            -jnp.inf,
        ),
        jnp.where(positive_proposed, 0.0, -jnp.inf),
    )
    # ``log(U)`` is in ``[-inf, 0)``.  The strict comparison also rejects a
    # zero-to-zero proposal, matching the former fail-closed convention.
    log_uniform = jnp.log(
        jax.random.uniform(acceptance_key, log_acceptance_probability.shape)
    )
    accepted = log_uniform < log_acceptance_probability
    if lazy_probability > 0.0:
        # An explicit identity component removes periodic edge cases (for
        # example a two-site, one-up spin sector) without changing the target
        # distribution or requiring a state-dependent Hastings correction.
        accepted = accepted & (
            jax.random.uniform(lazy_key, accepted.shape) >= lazy_probability
        )
    nontrivial = jnp.any(proposed != array, axis=1)
    new_states = jnp.where(accepted[:, None], proposed, array)
    new_amplitudes = jnp.where(accepted, proposed_amplitudes, current)
    return new_states, new_amplitudes, accepted, nontrivial


def run_exchange_sweeps(
    params: object,
    key: jax.Array,
    states: jax.Array,
    *,
    amplitude_fn: PhysicalAmplitude,
    particle_kind: str,
    sweeps: int,
    amplitudes: jax.Array | None = None,
    exchange_edges: Sequence[Sequence[int]] | None = None,
    bond_proposal_probability: float = 1.0,
    lazy_probability: float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Advance persistent chains and return acceptance diagnostics.

    Here ``sweeps`` means proposal attempts *per chain*.  It is not implicitly
    multiplied by the number of sites or bonds.  Cached amplitudes are valid
    only for the supplied ``params`` and ``states`` and must never be reused
    after an optimizer update.
    """

    if isinstance(sweeps, bool) or not isinstance(sweeps, Integral) or sweeps < 1:
        raise ValueError("sweeps must be a positive integer")
    array = _checked_chain_shape(states)
    if amplitudes is None:
        current = jax.vmap(lambda row: amplitude_fn(params, row))(array)
    else:
        current = jnp.asarray(amplitudes)
    keys = jax.random.split(key, int(sweeps) + 1)

    def body(carry, step_key):
        chain_states, chain_amplitudes = carry
        updated = metropolis_exchange_step(
            params,
            step_key,
            chain_states,
            chain_amplitudes,
            amplitude_fn=amplitude_fn,
            particle_kind=particle_kind,
            exchange_edges=exchange_edges,
            bond_proposal_probability=bond_proposal_probability,
            lazy_probability=lazy_probability,
        )
        new_states, new_amplitudes, accepted, nontrivial = updated
        return (new_states, new_amplitudes), (accepted, nontrivial)

    (final_states, final_amplitudes), (accepted, nontrivial) = jax.lax.scan(
        body, (array, current), keys[:-1]
    )
    accepted_moves = accepted & nontrivial
    diagnostics = {
        "acceptance_fraction_including_null": jnp.mean(accepted),
        "nontrivial_proposal_fraction": jnp.mean(nontrivial),
        "accepted_move_fraction": jnp.mean(accepted_moves),
        "conditional_move_acceptance": jnp.sum(accepted_moves)
        / jnp.maximum(jnp.sum(nontrivial), 1),
        "zero_amplitude_chain_fraction": jnp.mean(
            jnp.abs(final_amplitudes) == 0.0
        ),
    }
    return final_states, final_amplitudes, keys[-1], diagnostics


__all__ = [
    "ExchangeEdge",
    "canonical_exchange_edges",
    "hubbard_bond_exchange_proposals",
    "hubbard_exchange_proposals",
    "initialize_hubbard_exchange_chains",
    "initialize_spin_exchange_chains",
    "metropolis_exchange_step",
    "run_exchange_sweeps",
    "spin_bond_exchange_proposals",
    "spin_exchange_proposals",
]
