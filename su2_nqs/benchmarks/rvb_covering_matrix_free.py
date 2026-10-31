"""Matrix-free transition-loop estimators for oriented RVB coverings.

The basis vectors are products of canonically oriented Condon--Shortley
singlets.  Two coverings form disjoint even transition loops.  Their overlap
and every Heisenberg-bond matrix element follow from the loop count and the
alternating parity on those loops, so neither a spin-basis wavefunction nor a
many-body Hamiltonian matrix is required.

For exact lattice symmetry we use ``<C|P_A1|D>`` and
``<C|H P_A1|D>``.  Commutation of ``H`` with the space group reduces the
apparent double projector to one relative group sum.  The neural coefficient
is evaluated once per covering; the group order appears only in the cheap
batched loop kernel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.spatial_symmetry import SitePermutation, validate_site_permutation
from .rvb_covering_oracle import DimerMatching, transform_oriented_matching


jax.config.update("jax_enable_x64", True)


WeightedBond = tuple[int, int, float]


def matching_partner_array(
    matching: Sequence[Sequence[int]], *, n_sites: int | None = None
) -> np.ndarray:
    """Return the involutive partner array of one perfect matching."""

    pairs = tuple(tuple(pair) for pair in matching)
    if not pairs:
        raise ValueError("matching must be nonempty")
    inferred = 2 * len(pairs) if n_sites is None else int(n_sites)
    if inferred < 2 or inferred % 2:
        raise ValueError("n_sites must be a positive even integer")
    partner = np.full(inferred, -1, dtype=np.int32)
    for raw in pairs:
        if len(raw) != 2 or any(
            isinstance(site, bool) or not isinstance(site, Integral) for site in raw
        ):
            raise TypeError("every dimer must contain two integer sites")
        first, second = (int(raw[0]), int(raw[1]))
        if not 0 <= first < inferred or not 0 <= second < inferred or first == second:
            raise ValueError("dimer sites must be distinct and lie in 0..N-1")
        if partner[first] >= 0 or partner[second] >= 0:
            raise ValueError("dimers must be site-disjoint")
        partner[first] = second
        partner[second] = first
    if np.any(partner < 0):
        raise ValueError("matching must cover every site exactly once")
    partner.setflags(write=False)
    return partner


def partner_array_matching(partner: Sequence[int]) -> DimerMatching:
    """Convert a checked involutive partner array to canonical dimers."""

    values = np.asarray(partner)
    if values.ndim != 1 or values.size < 2 or values.size % 2:
        raise ValueError("partner must be a positive even-length vector")
    if values.dtype == np.bool_ or not np.issubdtype(values.dtype, np.integer):
        raise TypeError("partner must have integer dtype")
    values = values.astype(np.int64, copy=False)
    sites = np.arange(values.size)
    if (
        np.any(values < 0)
        or np.any(values >= values.size)
        or np.any(values == sites)
        or not np.array_equal(values[values], sites)
    ):
        raise ValueError("partner must be a fixed-point-free involution")
    return tuple((int(site), int(values[site])) for site in sites if site < values[site])


@dataclass(frozen=True)
class TransitionLoopData:
    """Overlap and deterministic component/parity labelling of two coverings."""

    overlap: float
    loop_count: int
    component_labels: np.ndarray
    parity_bits: np.ndarray


def transition_loop_data(
    left_partner: Sequence[int], right_partner: Sequence[int]
) -> TransitionLoopData:
    """Return the exact oriented overlap data for two partner arrays."""

    left = np.asarray(left_partner, dtype=np.int64)
    right = np.asarray(right_partner, dtype=np.int64)
    # Reuse the public conversion for complete API-boundary validation.
    partner_array_matching(left)
    partner_array_matching(right)
    if left.shape != right.shape:
        raise ValueError("left and right matchings must have the same size")
    n_sites = int(left.size)
    labels = np.full(n_sites, -1, dtype=np.int32)
    bits = np.full(n_sites, -1, dtype=np.int8)
    for root in range(n_sites):
        if labels[root] >= 0:
            continue
        stack = [root]
        labels[root] = root
        bits[root] = 0
        while stack:
            site = stack.pop()
            for neighbor in (int(left[site]), int(right[site])):
                expected = 1 - int(bits[site])
                if labels[neighbor] < 0:
                    labels[neighbor] = root
                    bits[neighbor] = expected
                    stack.append(neighbor)
                elif bits[neighbor] != expected:
                    raise RuntimeError("transition graph is not an even-loop graph")
    loop_count = int(np.unique(labels).size)
    mask_left = np.arange(n_sites) < left
    mask_right = np.arange(n_sites) < right
    negative_count = int(np.count_nonzero(mask_left & (bits == 0)))
    negative_count += int(np.count_nonzero(mask_right & (bits == 0)))
    sign = -1.0 if negative_count % 2 else 1.0
    overlap = sign * 2.0 ** (loop_count - n_sites / 2)
    labels.setflags(write=False)
    bits.setflags(write=False)
    return TransitionLoopData(overlap, loop_count, labels, bits)


def valence_bond_overlap(
    left_partner: Sequence[int], right_partner: Sequence[int]
) -> float:
    """Return ``<left|right>`` in the canonical oriented-singlet gauge."""

    return transition_loop_data(left_partner, right_partner).overlap


def valence_bond_hamiltonian_element(
    left_partner: Sequence[int],
    right_partner: Sequence[int],
    bonds: Sequence[WeightedBond | Sequence[float]],
) -> float:
    """Return ``<left|sum J_ij S_i.S_j|right>`` from transition loops."""

    data = transition_loop_data(left_partner, right_partner)
    value = 0.0
    n_sites = data.component_labels.size
    for raw in bonds:
        bond = tuple(raw)
        if len(bond) != 3:
            raise ValueError("weighted bonds must be (first,second,coupling)")
        first, second = int(bond[0]), int(bond[1])
        coupling = float(bond[2])
        if not 0 <= first < second < n_sites or not np.isfinite(coupling):
            raise ValueError("invalid finite weighted bond")
        if data.component_labels[first] == data.component_labels[second]:
            parity = 1.0 if data.parity_bits[first] == data.parity_bits[second] else -1.0
            value += 0.75 * coupling * parity
    return data.overlap * value


def projected_valence_bond_matrix_elements(
    left_partner: Sequence[int],
    right_partner: Sequence[int],
    bonds: Sequence[WeightedBond | Sequence[float]],
    group: Sequence[SitePermutation | Sequence[int]],
) -> tuple[float, float]:
    """Return ``(<C|P_A1|D>, <C|H P_A1|D>)`` exactly."""

    left = np.asarray(left_partner, dtype=np.int32)
    right_matching = partner_array_matching(right_partner)
    images = tuple(validate_site_permutation(image, n_sites=left.size) for image in group)
    if not images:
        raise ValueError("space group must be nonempty")
    overlap = 0.0
    hamiltonian = 0.0
    for image in images:
        transformed, sign = transform_oriented_matching(right_matching, image)
        transformed_partner = matching_partner_array(transformed, n_sites=left.size)
        overlap += sign * valence_bond_overlap(left, transformed_partner)
        hamiltonian += sign * valence_bond_hamiltonian_element(
            left, transformed_partner, bonds
        )
    scale = 1.0 / len(images)
    return scale * overlap, scale * hamiltonian


def _jax_transition_loop_data(left: jax.Array, right: jax.Array):
    n_sites = left.shape[0]
    sites = jnp.arange(n_sites, dtype=jnp.int32)
    labels = sites

    def label_step(_index, current):
        return jnp.minimum(current, jnp.minimum(current[left], current[right]))

    labels = jax.lax.fori_loop(0, n_sites, label_step, labels)
    # A second propagation makes the fixed iteration count robust to the
    # simultaneous-update direction on long transition loops.
    labels = jax.lax.fori_loop(0, n_sites, label_step, labels)
    bits = jnp.where(labels == sites, 0, -1)

    def bit_step(_index, current):
        from_left = jnp.where(current[left] >= 0, 1 - current[left], -1)
        from_right = jnp.where(current[right] >= 0, 1 - current[right], -1)
        candidate = jnp.maximum(from_left, from_right)
        return jnp.where(current >= 0, current, candidate)

    bits = jax.lax.fori_loop(0, n_sites, bit_step, bits)
    loop_count = jnp.sum(labels == sites)
    negative_count = jnp.sum((sites < left) & (bits == 0))
    negative_count += jnp.sum((sites < right) & (bits == 0))
    sign = jnp.where(negative_count % 2, -1.0, 1.0)
    overlap = sign * jnp.exp2(loop_count.astype(jnp.float64) - n_sites / 2.0)
    return overlap, labels, bits


def _jax_overlap_hamiltonian(
    left: jax.Array,
    right: jax.Array,
    bond_first: jax.Array,
    bond_second: jax.Array,
    couplings: jax.Array,
):
    overlap, labels, bits = _jax_transition_loop_data(left, right)
    same_loop = labels[bond_first] == labels[bond_second]
    parity = jnp.where(bits[bond_first] == bits[bond_second], 1.0, -1.0)
    ratio = 0.75 * jnp.sum(couplings * same_loop * parity)
    return overlap, overlap * ratio


def _jax_transform_partner(
    partner: jax.Array, image: jax.Array, inverse: jax.Array
):
    sites = jnp.arange(partner.shape[0], dtype=jnp.int32)
    transformed = image[partner[inverse]]
    reversal_count = jnp.sum(
        (sites < partner) & (image[sites] > image[partner])
    )
    sign = jnp.where(reversal_count % 2, -1.0, 1.0)
    return transformed, sign


def compile_projected_loop_kernel(
    bonds: Sequence[WeightedBond | Sequence[float]],
    group: Sequence[SitePermutation | Sequence[int]],
    *,
    n_sites: int,
):
    """Return a JIT batch kernel for projected overlap and Hamiltonian pairs."""

    if n_sites < 2 or n_sites % 2:
        raise ValueError("n_sites must be a positive even integer")
    checked_bonds = tuple(tuple(bond) for bond in bonds)
    bond_first = jnp.asarray([int(bond[0]) for bond in checked_bonds], dtype=jnp.int32)
    bond_second = jnp.asarray([int(bond[1]) for bond in checked_bonds], dtype=jnp.int32)
    couplings = jnp.asarray([float(bond[2]) for bond in checked_bonds], dtype=jnp.float64)
    images_np = np.asarray(
        [validate_site_permutation(image, n_sites=n_sites) for image in group],
        dtype=np.int32,
    )
    if images_np.shape[0] < 1:
        raise ValueError("space group must be nonempty")
    inverses_np = np.argsort(images_np, axis=1).astype(np.int32)
    images = jnp.asarray(images_np)
    inverses = jnp.asarray(inverses_np)

    def one_pair(left, right):
        transformed, signs = jax.vmap(
            lambda image, inverse: _jax_transform_partner(right, image, inverse)
        )(images, inverses)
        overlap, hamiltonian = jax.vmap(
            lambda candidate: _jax_overlap_hamiltonian(
                left, candidate, bond_first, bond_second, couplings
            )
        )(transformed)
        return jnp.mean(signs * overlap), jnp.mean(signs * hamiltonian)

    return jax.jit(jax.vmap(one_pair))


@dataclass(frozen=True)
class SequentialMatchingSamples:
    """Independent accepted samples and their unnormalized proposal weights."""

    partners: np.ndarray
    log_unnormalized_probabilities: np.ndarray
    proposal_attempts: int
    rejected_dead_ends: int


@dataclass(frozen=True)
class TwoDimerFlip:
    """A symmetric local move between two perfect matchings of four sites."""

    first: tuple[tuple[int, int], tuple[int, int]]
    second: tuple[tuple[int, int], tuple[int, int]]


def enumerate_two_dimer_flips(
    n_sites: int, bonds: Sequence[WeightedBond | Sequence[float]]
) -> tuple[TwoDimerFlip, ...]:
    """Enumerate every graph-supported two-dimer reconnection move."""

    adjacency = _adjacency(n_sites, bonds)
    edges = {
        (first, second)
        for first in range(n_sites)
        for second in adjacency[first]
        if first < second
    }
    flips: set[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = set()
    for sites in combinations(range(n_sites), 4):
        first, second, third, fourth = sites
        patterns = (
            tuple(sorted(((first, second), (third, fourth)))),
            tuple(sorted(((first, third), (second, fourth)))),
            tuple(sorted(((first, fourth), (second, third)))),
        )
        valid = tuple(pattern for pattern in patterns if set(pattern) <= edges)
        for left, right in combinations(valid, 2):
            flips.add(tuple(sorted((left, right))))
    return tuple(TwoDimerFlip(*pair) for pair in sorted(flips))


def flippable_two_dimer_move_indices(
    partner: Sequence[int], flips: Sequence[TwoDimerFlip]
) -> tuple[int, ...]:
    """Return all symmetric two-dimer moves available in one matching."""

    matching = set(partner_array_matching(partner))
    return tuple(
        index
        for index, flip in enumerate(flips)
        if set(flip.first) <= matching or set(flip.second) <= matching
    )


def apply_two_dimer_flip(
    partner: Sequence[int], flip: TwoDimerFlip
) -> np.ndarray:
    """Apply one available symmetric flip and return a new partner array."""

    values = np.asarray(partner, dtype=np.int32)
    matching = set(partner_array_matching(values))
    if set(flip.first) <= matching:
        removed, inserted = flip.first, flip.second
    elif set(flip.second) <= matching:
        removed, inserted = flip.second, flip.first
    else:
        raise ValueError("the requested two-dimer move is not flippable")
    matching.difference_update(removed)
    matching.update(inserted)
    return matching_partner_array(tuple(sorted(matching)), n_sites=values.size)


def _adjacency(
    n_sites: int, bonds: Sequence[WeightedBond | Sequence[float]]
) -> tuple[tuple[int, ...], ...]:
    neighbors = [set() for _ in range(n_sites)]
    for raw in bonds:
        bond = tuple(raw)
        if len(bond) != 3:
            raise ValueError("weighted bonds must be (first,second,coupling)")
        first, second = int(bond[0]), int(bond[1])
        coupling = complex(bond[2])
        if not 0 <= first < second < n_sites:
            raise ValueError("bond sites must satisfy 0 <= first < second < N")
        if not np.isfinite(coupling.real) or not np.isfinite(coupling.imag):
            raise ValueError("bond couplings must be finite")
        if abs(coupling) > 0.0:
            neighbors[first].add(second)
            neighbors[second].add(first)
    return tuple(tuple(sorted(row)) for row in neighbors)


def sequential_matching_log_weight(
    partner: Sequence[int],
    bonds: Sequence[WeightedBond | Sequence[float]],
) -> float:
    """Return the pre-rejection log probability of the canonical proposal."""

    values = np.asarray(partner, dtype=np.int32)
    partner_array_matching(values)
    adjacency = _adjacency(values.size, bonds)
    remaining = set(range(values.size))
    log_probability = 0.0
    while remaining:
        first = min(remaining)
        choices = tuple(site for site in adjacency[first] if site in remaining)
        second = int(values[first])
        if second not in choices:
            return -np.inf
        log_probability -= float(np.log(len(choices)))
        remaining.remove(first)
        remaining.remove(second)
    return log_probability


def sample_sequential_matchings(
    n_sites: int,
    bonds: Sequence[WeightedBond | Sequence[float]],
    *,
    count: int,
    seed: int,
    maximum_attempts: int = 1_000_000,
) -> SequentialMatchingSamples:
    """Sample accepted canonical sequential proposals independently.

    Dead-end constructions are rejected.  The returned probability omits the
    global acceptance normalization, which cancels exactly between numerator
    and denominator of pair-importance estimators.
    """

    if count < 1 or maximum_attempts < count:
        raise ValueError("count must be positive and maximum_attempts >= count")
    adjacency = _adjacency(n_sites, bonds)
    rng = np.random.default_rng(seed)
    partners = []
    log_probabilities = []
    attempts = 0
    rejected = 0
    while len(partners) < count and attempts < maximum_attempts:
        attempts += 1
        remaining = set(range(n_sites))
        partner = np.full(n_sites, -1, dtype=np.int32)
        log_probability = 0.0
        failed = False
        while remaining:
            first = min(remaining)
            choices = tuple(site for site in adjacency[first] if site in remaining)
            if not choices:
                failed = True
                break
            second = int(choices[int(rng.integers(len(choices)))])
            log_probability -= float(np.log(len(choices)))
            partner[first] = second
            partner[second] = first
            remaining.remove(first)
            remaining.remove(second)
        if failed:
            rejected += 1
            continue
        partners.append(partner)
        log_probabilities.append(log_probability)
    if len(partners) != count:
        raise RuntimeError("sequential proposal exhausted maximum_attempts")
    return SequentialMatchingSamples(
        np.stack(partners),
        np.asarray(log_probabilities, dtype=np.float64),
        attempts,
        rejected,
    )


__all__ = [
    "SequentialMatchingSamples",
    "TwoDimerFlip",
    "TransitionLoopData",
    "compile_projected_loop_kernel",
    "apply_two_dimer_flip",
    "enumerate_two_dimer_flips",
    "flippable_two_dimer_move_indices",
    "matching_partner_array",
    "partner_array_matching",
    "projected_valence_bond_matrix_elements",
    "sample_sequential_matchings",
    "sequential_matching_log_weight",
    "transition_loop_data",
    "valence_bond_hamiltonian_element",
    "valence_bond_overlap",
]
