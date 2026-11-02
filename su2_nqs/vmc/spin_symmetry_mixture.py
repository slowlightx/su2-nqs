"""Reversible cross-basin proposals for fixed-``Sz`` spin sampling.

The kernels in this module are independent of a Hamiltonian or model class.
They combine ordinary exchange moves with triangular-lattice point-group
orbits and a full-support dimer independence proposal.  The latter includes
the exact Hastings correction and a uniform fixed-sector floor.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from numbers import Integral

import jax
import jax.numpy as jnp

from su2_nqs.lattices.physical_route_banks import (
    triangular_dimer_complete_route_orders,
)
from su2_nqs.lattices.triangular_hnf import (
    hnf_point_group_permutations,
    hnf_translation_permutations,
)
from su2_nqs.vmc.physical_exchange import (
    PhysicalAmplitude,
    canonical_exchange_edges,
    spin_bond_exchange_proposals,
    spin_exchange_proposals,
)


SitePermutation = tuple[int, ...]
DimerMatching = tuple[tuple[int, int], ...]


def _checked_spin_states(states: jax.Array) -> jax.Array:
    array = jnp.asarray(states, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError("states must have shape (chains, n_sites>=2)")
    return array


def _canonical_matching(pairs: Sequence[Sequence[int]]) -> DimerMatching:
    matching = tuple(sorted(tuple(sorted((int(pair[0]), int(pair[1])))) for pair in pairs))
    flattened = tuple(site for pair in matching for site in pair)
    if len(flattened) < 2 or len(flattened) % 2 or len(set(flattened)) != len(flattened):
        raise ValueError("a dimer matching must cover distinct sites exactly once")
    if set(flattened) != set(range(len(flattened))):
        raise ValueError("a dimer matching must cover sites 0..N-1")
    return matching


def d6_point_group_images(
    supercell: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[SitePermutation, ...]:
    """Return the twelve active triangular ``D6`` site permutations."""

    images = tuple(image for _name, _operation, image in hnf_point_group_permutations(supercell))
    if len(images) != 12:
        raise ValueError("the requested supercell does not retain the full D6 point group")
    return images


def d6_complete_dimer_matchings(rows: int, columns: int) -> tuple[DimerMatching, ...]:
    """Return the space-group-complete NN dimer orbit on an even square torus.

    Three orientation labels alone miss translated staggerings.  Acting with
    translations and the full triangular ``D6`` point group on one physical
    nearest-neighbour matching produces the complete 12-covering orbit on the
    ``6x6`` torus used by the Route-ViT benchmark.
    """

    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (rows, columns)):
        raise TypeError("rows and columns must be integers")
    if rows != columns or rows < 2 or rows % 2:
        raise ValueError("D6-complete dimer matchings require an even square torus")
    n_sites = int(rows * columns)
    routes = triangular_dimer_complete_route_orders(int(rows), int(columns))
    order = routes["row_dimers"]
    reference = _canonical_matching(
        tuple(tuple(order[index : index + 2]) for index in range(0, n_sites, 2))
    )
    supercell = ((int(columns), 0), (0, int(rows)))
    points = d6_point_group_images(supercell)
    translations = tuple(image for _translation, image in hnf_translation_permutations(supercell))
    orbit: set[DimerMatching] = set()
    for point in points:
        for translation in translations:
            # Active affine action: first apply the point operation and then
            # translate its image.
            image = tuple(translation[point[site]] for site in range(n_sites))
            orbit.add(
                tuple(
                    sorted(
                        tuple(sorted((image[first], image[second])))
                        for first, second in reference
                    )
                )
            )
    result = tuple(sorted(orbit))
    if (rows, columns) == (6, 6) and len(result) != 12:
        raise RuntimeError("the triangular 6x6 dimer orbit must contain 12 coverings")
    return result


def _checked_matchings(matchings: Sequence[Sequence[Sequence[int]]], n_sites: int) -> jax.Array:
    concrete = tuple(_canonical_matching(matching) for matching in matchings)
    if not concrete:
        raise ValueError("at least one dimer matching is required")
    if any(len(matching) * 2 != n_sites for matching in concrete):
        raise ValueError("each matching must cover every site")
    return jnp.asarray(concrete, dtype=jnp.int32)


def dimer_uniform_log_probabilities(
    states: jax.Array,
    matchings: Sequence[Sequence[Sequence[int]]] | jax.Array,
    uniform_floor: float,
) -> jax.Array:
    """Evaluate the exact full-support dimer-mixture proposal density."""

    array = _checked_spin_states(states)
    n_sites = int(array.shape[1])
    if n_sites % 2:
        raise ValueError("dimer proposals require an even site count")
    epsilon = float(uniform_floor)
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("uniform_floor must lie in (0,1]")
    matching_array = jnp.asarray(matchings, dtype=jnp.int32)
    if matching_array.ndim != 3 or matching_array.shape[1:] != (n_sites // 2, 2):
        raise ValueError("matchings must have shape (count,n_sites/2,2)")
    first = array[:, matching_array[:, :, 0]]
    second = array[:, matching_array[:, :, 1]]
    support_count = jnp.sum(jnp.all(first + second == 1, axis=-1), axis=1)
    in_sector = jnp.sum(array, axis=1) == n_sites // 2
    uniform_probability = epsilon / float(math.comb(n_sites, n_sites // 2))
    dimer_probability = (
        (1.0 - epsilon)
        * support_count.astype(jnp.float64)
        / float(matching_array.shape[0] * 2 ** (n_sites // 2))
    )
    probability = jnp.where(in_sector, uniform_probability + dimer_probability, 0.0)
    return jnp.where(probability > 0.0, jnp.log(probability), -jnp.inf)


def sample_dimer_uniform_proposal(
    key: jax.Array,
    n_samples: int,
    matchings: Sequence[Sequence[Sequence[int]]] | jax.Array,
    uniform_floor: float,
) -> tuple[jax.Array, jax.Array]:
    """Draw from the normalized dimer mixture plus fixed-``Sz`` floor.

    Returns ``states, used_uniform_component``.
    """

    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    matching_array = jnp.asarray(matchings, dtype=jnp.int32)
    if matching_array.ndim != 3 or matching_array.shape[2] != 2:
        raise ValueError("matchings must have shape (count,pairs,2)")
    n_sites = int(2 * matching_array.shape[1])
    epsilon = float(uniform_floor)
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("uniform_floor must lie in (0,1]")
    component_key, uniform_key, matching_key, orientation_key = jax.random.split(key, 4)
    used_uniform = jax.random.uniform(component_key, (int(n_samples),)) < epsilon

    scores = jax.random.uniform(uniform_key, (int(n_samples), n_sites))
    selected = jnp.argsort(scores, axis=1)[:, : n_sites // 2]
    rows = jnp.arange(int(n_samples), dtype=jnp.int32)[:, None]
    uniform_states = jnp.zeros((int(n_samples), n_sites), dtype=jnp.int32).at[
        rows, selected
    ].set(1)

    matching_ids = jax.random.randint(
        matching_key, (int(n_samples),), 0, matching_array.shape[0]
    )
    selected_matchings = matching_array[matching_ids]
    orientations = jax.random.bernoulli(
        orientation_key, shape=(int(n_samples), n_sites // 2)
    ).astype(jnp.int32)
    dimer_states = jnp.zeros((int(n_samples), n_sites), dtype=jnp.int32)
    dimer_states = dimer_states.at[rows, selected_matchings[:, :, 0]].set(orientations)
    dimer_states = dimer_states.at[rows, selected_matchings[:, :, 1]].set(1 - orientations)
    states = jnp.where(used_uniform[:, None], uniform_states, dimer_states)
    return states, used_uniform


def d6_orbit_proposals(
    key: jax.Array,
    states: jax.Array,
    images: Sequence[Sequence[int]] | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply a uniformly selected active point-group operation per chain."""

    array = _checked_spin_states(states)
    image_array = jnp.asarray(images, dtype=jnp.int32)
    if image_array.ndim != 2 or image_array.shape[1] != array.shape[1]:
        raise ValueError("images must have shape (group_order,n_sites)")
    selected = jax.random.randint(key, (array.shape[0],), 0, image_array.shape[0])
    chain_images = image_array[selected]
    rows = jnp.arange(array.shape[0], dtype=jnp.int32)[:, None]
    proposed = jnp.zeros_like(array).at[rows, chain_images].set(array)
    return proposed, selected


def _validated_kernel_weights(weights: Sequence[float]) -> tuple[float, float, float]:
    concrete = tuple(float(value) for value in weights)
    if len(concrete) != 3 or any(value < 0.0 for value in concrete):
        raise ValueError("kernel_weights must be three non-negative probabilities")
    if abs(sum(concrete) - 1.0) > 1.0e-12:
        raise ValueError("kernel_weights must sum to one")
    return concrete[0], concrete[1], concrete[2]


def metropolis_spin_symmetry_mixture_step(
    params: object,
    key: jax.Array,
    states: jax.Array,
    amplitudes: jax.Array,
    *,
    amplitude_fn: PhysicalAmplitude,
    exchange_edges: Sequence[Sequence[int]],
    kernel_weights: Sequence[float],
    d6_images: Sequence[Sequence[int]] | jax.Array,
    dimer_matchings: Sequence[Sequence[Sequence[int]]] | jax.Array,
    uniform_floor: float,
    bond_proposal_probability: float = 0.9,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Apply one reversible exchange/orbit/independence mixture step.

    Kernel labels are ``0=exchange``, ``1=D6 orbit``, and
    ``2=dimer+uniform independence``.
    """

    array = _checked_spin_states(states)
    current = jnp.asarray(amplitudes)
    if current.shape != (array.shape[0],):
        raise ValueError("one cached amplitude is required per chain")
    weights = _validated_kernel_weights(kernel_weights)
    if not 0.0 <= float(bond_proposal_probability) <= 1.0:
        raise ValueError("bond_proposal_probability must lie in [0,1]")
    edges = canonical_exchange_edges(exchange_edges, n_sites=int(array.shape[1]))
    (
        global_key,
        bond_key,
        exchange_choice_key,
        orbit_key,
        independence_key,
        kernel_key,
        acceptance_key,
    ) = jax.random.split(key, 7)

    global_proposed = spin_exchange_proposals(global_key, array)
    bond_proposed = spin_bond_exchange_proposals(bond_key, array, edges)
    choose_bond = jax.random.uniform(exchange_choice_key, (array.shape[0],)) < float(
        bond_proposal_probability
    )
    exchange_proposed = jnp.where(choose_bond[:, None], bond_proposed, global_proposed)
    orbit_proposed, _operation_ids = d6_orbit_proposals(orbit_key, array, d6_images)
    independence_proposed, _used_uniform = sample_dimer_uniform_proposal(
        independence_key, int(array.shape[0]), dimer_matchings, uniform_floor
    )

    thresholds = jnp.asarray((weights[0], weights[0] + weights[1]), dtype=jnp.float64)
    draw = jax.random.uniform(kernel_key, (array.shape[0],))
    kernel = jnp.where(draw < thresholds[0], 0, jnp.where(draw < thresholds[1], 1, 2))
    proposed = jnp.where(
        (kernel == 0)[:, None],
        exchange_proposed,
        jnp.where((kernel == 1)[:, None], orbit_proposed, independence_proposed),
    )
    proposed_amplitudes = jax.vmap(lambda row: amplitude_fn(params, row))(proposed)

    current_magnitude = jnp.abs(current)
    proposed_magnitude = jnp.abs(proposed_amplitudes)
    safe_current = jnp.where(current_magnitude > 0.0, current_magnitude, 1.0)
    safe_proposed = jnp.where(proposed_magnitude > 0.0, proposed_magnitude, 1.0)
    log_ratio = 2.0 * (jnp.log(safe_proposed) - jnp.log(safe_current))
    current_log_q = dimer_uniform_log_probabilities(array, dimer_matchings, uniform_floor)
    proposed_log_q = dimer_uniform_log_probabilities(
        proposed, dimer_matchings, uniform_floor
    )
    log_ratio = log_ratio + jnp.where(
        kernel == 2, current_log_q - proposed_log_q, 0.0
    )
    log_acceptance = jnp.where(
        current_magnitude > 0.0,
        jnp.where(proposed_magnitude > 0.0, jnp.minimum(0.0, log_ratio), -jnp.inf),
        jnp.where(proposed_magnitude > 0.0, 0.0, -jnp.inf),
    )
    accepted = jnp.log(jax.random.uniform(acceptance_key, log_acceptance.shape)) < log_acceptance
    nontrivial = jnp.any(proposed != array, axis=1)
    new_states = jnp.where(accepted[:, None], proposed, array)
    new_amplitudes = jnp.where(accepted, proposed_amplitudes, current)
    return new_states, new_amplitudes, accepted, nontrivial, kernel


def run_spin_symmetry_mixture_sweeps(
    params: object,
    master_key: jax.Array,
    proposal_index: int,
    states: jax.Array,
    *,
    amplitude_fn: PhysicalAmplitude,
    sweeps: int,
    exchange_edges: Sequence[Sequence[int]],
    kernel_weights: Sequence[float],
    d6_images: Sequence[Sequence[int]] | jax.Array,
    dimer_matchings: Sequence[Sequence[Sequence[int]]] | jax.Array,
    uniform_floor: float,
    amplitudes: jax.Array | None = None,
    bond_proposal_probability: float = 0.9,
) -> tuple[jax.Array, jax.Array, int, dict[str, jax.Array]]:
    """Advance indexed persistent chains with chunk-size-independent replay."""

    if isinstance(sweeps, bool) or not isinstance(sweeps, Integral) or sweeps < 1:
        raise ValueError("sweeps must be a positive integer")
    if isinstance(proposal_index, bool) or not isinstance(proposal_index, Integral) or proposal_index < 0:
        raise ValueError("proposal_index must be a non-negative integer")
    array = _checked_spin_states(states)
    current = (
        jax.vmap(lambda row: amplitude_fn(params, row))(array)
        if amplitudes is None
        else jnp.asarray(amplitudes)
    )
    indices = jnp.arange(int(proposal_index), int(proposal_index) + int(sweeps), dtype=jnp.uint32)

    def body(carry, index):
        chain_states, chain_amplitudes = carry
        step_key = jax.random.fold_in(master_key, index)
        updated = metropolis_spin_symmetry_mixture_step(
            params,
            step_key,
            chain_states,
            chain_amplitudes,
            amplitude_fn=amplitude_fn,
            exchange_edges=exchange_edges,
            kernel_weights=kernel_weights,
            d6_images=d6_images,
            dimer_matchings=dimer_matchings,
            uniform_floor=uniform_floor,
            bond_proposal_probability=bond_proposal_probability,
        )
        new_states, new_amplitudes, accepted, nontrivial, kernel = updated
        return (new_states, new_amplitudes), (accepted, nontrivial, kernel)

    (final_states, final_amplitudes), (accepted, nontrivial, kernel) = jax.lax.scan(
        body, (array, current), indices
    )
    accepted_moves = accepted & nontrivial
    proposed_by_kernel = jnp.asarray([jnp.sum(kernel == label) for label in range(3)])
    nontrivial_by_kernel = jnp.asarray(
        [jnp.sum((kernel == label) & nontrivial) for label in range(3)]
    )
    accepted_by_kernel = jnp.asarray(
        [jnp.sum((kernel == label) & accepted_moves) for label in range(3)]
    )
    diagnostics = {
        "proposed_by_kernel": proposed_by_kernel,
        "nontrivial_by_kernel": nontrivial_by_kernel,
        "accepted_moves_by_kernel": accepted_by_kernel,
        "conditional_acceptance_by_kernel": accepted_by_kernel
        / jnp.maximum(nontrivial_by_kernel, 1),
        "accepted_move_fraction": jnp.mean(accepted_moves),
        "zero_amplitude_chain_fraction": jnp.mean(jnp.abs(final_amplitudes) == 0.0),
    }
    return final_states, final_amplitudes, int(proposal_index) + int(sweeps), diagnostics


__all__ = [
    "DimerMatching",
    "SitePermutation",
    "d6_complete_dimer_matchings",
    "d6_orbit_proposals",
    "d6_point_group_images",
    "dimer_uniform_log_probabilities",
    "metropolis_spin_symmetry_mixture_step",
    "run_spin_symmetry_mixture_sweeps",
    "sample_dimer_uniform_proposal",
]
