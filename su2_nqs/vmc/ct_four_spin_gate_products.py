"""Reusable full-H products for one physical four-spin gate before A1.

The 70 entries are coefficients of U[new_local_bits,old_local_bits] with
equal doubled local m. No gate parameter, ratio, probability threshold, or
magnetic-sector truncation occurs in this kernel. Hamiltonian work stays
outside the model's raw and exchange-query factories.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from ..models.triangular_mixed_clebsch import MixedClebschConfig


def four_spin_entry_pairs() -> tuple[tuple[int, int], ...]:
    """Lexicographic (new,old) local MSB-first codes with equal doubled m."""
    return tuple((new, old) for new in range(16) for old in range(16)
                 if new.bit_count() == old.bit_count())


def make_gate_products_from_queries(
    raw_query: Callable,
    exchange_query_factory: Callable[[np.ndarray], Callable],
    images: np.ndarray,
    bonds: Sequence[tuple[int, int, float]],
    ordered_sites: Sequence[int],
    *,
    n_sites: int,
    projection_chunk: int = 1,
) -> tuple[Callable, dict]:
    """Return action(bank,bits_batch)->(Phi70,HPhi70,Qcore,Qpush).

    raw_query(bank,bits) is one complex-capable raw amplitude. The exchange
    factory accepts the ordered OUTSIDE-bond pairs and returns a query giving
    (raw, swapped_vector). Outside means neither endpoint is in ordered_sites;
    these swaps commute with replacing the local four bits. Touching swaps
    instead receive a fresh six-predecessor raw query, including null swaps.

    Phi70 is the coherent image mean; HPhi70 is H P U|raw>, not U H P|raw>.
    This optimized implementation requires every image to preserve the full
    weighted H and checks that condition. Images need not form a group for
    the algebra, but a complete group is required for an A1 interpretation.

    Qcore is the INCOHERENT image mean of sum_old|raw(predecessor)|²/k,
    k=binomial(4,number_of_local_up_bits). Qpush sums Qcore(T_b x) with fixed
    input-label probabilities |J_b|/sum|J|, retaining all null and repeated
    swaps. Both have the raw state's norm; the caller divides by known Zraw.
    No sample generation, energy estimate, or normalization is done here.
    """
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 4:
        raise ValueError('integer n_sites>=4 required')
    sites = tuple(ordered_sites)
    if (len(sites) != 4 or len(set(sites)) != 4
            or any(isinstance(a, bool) or not isinstance(a, Integral) or not 0 <= a < n_sites for a in sites)):
        raise ValueError('four distinct ordered integer sites in range required')
    maps = np.asarray(images)
    if (maps.ndim != 2 or maps.shape[1] != n_sites or not len(maps)
            or not np.issubdtype(maps.dtype, np.integer)
            or not np.all(np.sort(maps, axis=1) == np.arange(n_sites))
            or isinstance(projection_chunk, bool) or not isinstance(projection_chunk, Integral)
            or projection_chunk < 1 or len(maps) % projection_chunk):
        raise ValueError('physical permutations and a positive dividing projection_chunk required')
    edges = np.asarray([(a, b) for a, b, _ in bonds])
    values = np.asarray([value for _, _, value in bonds])
    if (edges.ndim != 2 or edges.shape[1] != 2 or not len(edges)
            or not np.issubdtype(edges.dtype, np.integer)
            or np.any(edges < 0) or np.any(edges >= n_sites) or np.any(edges[:, 0] == edges[:, 1])
            or np.iscomplexobj(values) or not np.isfinite(values).all()):
        raise ValueError('finite real weighted integer bonds without self-loops required')
    coefficients = values.astype(np.float64)
    if np.sum(abs(coefficients)) <= 0:
        raise ValueError('positive total absolute bond weight required for Qpush')
    probabilities = abs(coefficients)/np.sum(abs(coefficients))

    def weighted_graph(edge_array):
        graph = {}
        for pair, value in zip(edge_array, coefficients, strict=True):
            key = tuple(sorted(map(int, pair)))
            graph[key] = graph.get(key, 0.)+float(value)
        return {key: value for key, value in graph.items() if value != 0.}

    reference = weighted_graph(edges)
    # Qpush also commutes through the image mean only if the absolute-label
    # push weights are invariant. H invariance alone would miss cancellations.
    def pushed_graph(edge_array):
        graph = {}
        for pair, value in zip(edge_array, probabilities, strict=True):
            key = tuple(sorted(map(int, pair)))
            graph[key] = graph.get(key, 0.)+float(value)
        return {key: value for key, value in graph.items() if value != 0.}

    proposal_reference = pushed_graph(edges)
    for image in maps:
        for expected, actual in ((reference, weighted_graph(image[edges])),
                                 (proposal_reference, pushed_graph(image[edges]))):
            if expected.keys() != actual.keys() or any(
                    not np.isclose(value, actual[key], rtol=0, atol=1e-14)
                    for key, value in expected.items()):
                raise ValueError('every image must preserve both weighted H and the absolute-bond proposal')
    outside = np.asarray([i for i, (a, b) in enumerate(edges) if a not in sites and b not in sites], np.int32)
    touching = np.asarray([i for i in range(len(edges)) if i not in set(outside)], np.int32)
    query = exchange_query_factory(edges[outside].astype(np.int32)) if len(outside) else None
    pairs = four_spin_entry_pairs()
    entry_lookup = np.full((16, 16), -1, np.int32)
    for index, (new, old) in enumerate(pairs):
        entry_lookup[new, old] = index
    bits_table = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.int32)
    old_table = np.empty((16, 6), np.int32)
    valid_table = np.zeros((16, 6), bool)
    for new in range(16):
        old = np.flatnonzero(bits_table.sum(1) == bits_table[new].sum())
        old_table[new] = new
        old_table[new, :len(old)] = old
        valid_table[new, :len(old)] = True
    sites_j, edges_j = jnp.asarray(sites), jnp.asarray(edges)
    bit_codes = jnp.asarray([8, 4, 2, 1])
    bits_table, old_table, valid_table, entry_lookup = map(jnp.asarray,
        (bits_table, old_table, valid_table, entry_lookup))
    h_weights, push_weights = jnp.asarray(coefficients), jnp.asarray(probabilities)
    outside_j, touching_j = jnp.asarray(outside), jnp.asarray(touching)
    chunks = jnp.asarray(maps.reshape(-1, projection_chunk, n_sites))

    def predecessors(bits):
        new = bits[sites_j] @ bit_codes
        old, valid = old_table[new], valid_table[new]
        rows = jnp.broadcast_to(bits[None, :], (6, n_sites)).at[:, sites_j].set(bits_table[old])
        return rows, entry_lookup[new, old], valid

    def scatter(amplitudes, entries, valid):
        return jnp.zeros(70, amplitudes.dtype).at[entries].add(amplitudes*valid)

    def guide(amplitudes, valid):
        return jnp.sum(abs(amplitudes)**2*valid)/jnp.sum(valid)

    def image_products(bank, bits):
        rows, entry_ids, valid = predecessors(bits)
        if query is None:
            raw = jax.vmap(raw_query, in_axes=(None, 0))(bank, rows)
            swapped = jnp.zeros((6, 0), raw.dtype)
        else:
            raw, swapped = jax.vmap(query, in_axes=(None, 0))(bank, rows)
        phi = scatter(raw, entry_ids, valid)
        qcore = guide(raw, valid)
        opposite = bits[edges_j[:, 0]] != bits[edges_j[:, 1]]
        diagonal = jnp.sum(h_weights*jnp.where(opposite, -.25, .25))
        hp = diagonal*phi
        qpush = jnp.asarray(0., qcore.dtype)
        if len(outside):
            changed = jnp.where(opposite[outside_j][None, :], swapped, raw[:, None])
            coefficients_out = jnp.where(opposite[outside_j], .5*h_weights[outside_j], 0.)
            hp = hp+scatter(changed @ coefficients_out, entry_ids, valid)
            outside_density = jnp.sum(abs(changed)**2*valid[:, None], axis=0)/jnp.sum(valid)
            qpush = qpush+jnp.dot(push_weights[outside_j], outside_density)
        if len(touching):
            def touching_product(index):
                a, b = edges_j[index]
                changed_bits = bits.at[a].set(bits[b]).at[b].set(bits[a])
                changed_rows, changed_ids, changed_valid = predecessors(changed_bits)
                changed_raw = jax.vmap(raw_query, in_axes=(None, 0))(bank, changed_rows)
                feature = scatter(changed_raw, changed_ids, changed_valid)
                return (jnp.where(opposite[index], .5*h_weights[index], 0.)*feature,
                        push_weights[index]*guide(changed_raw, changed_valid))
            changed_h, changed_q = jax.lax.map(touching_product, touching_j)
            hp = hp+jnp.sum(changed_h, axis=0)
            qpush = qpush+jnp.sum(changed_q)
        return phi, hp, qcore, qpush

    def one(bank, bits):
        def partial(group):
            values = jax.vmap(image_products, in_axes=(None, 0))(bank, bits[group])
            return jax.tree.map(lambda x: jnp.sum(x, axis=0), values)
        sums = jax.lax.map(jax.checkpoint(partial), chunks)
        return jax.tree.map(lambda x: jnp.sum(x, axis=0)/len(maps), sums)

    @jax.jit
    def action(bank, batch_bits):
        batch = jnp.asarray(batch_bits)
        if batch.ndim != 2 or batch.shape[1] != n_sites or not jnp.issubdtype(batch.dtype, jnp.integer):
            raise ValueError('batch of physical integer-bit configurations required')
        return jax.vmap(one, in_axes=(None, 0))(bank, batch)

    metadata = dict(entry_pairs=[list(pair) for pair in pairs], entry_count=70,
        entry_orientation='U[new_local_MSB_code,old_local_MSB_code]', ordered_sites=list(sites),
        n_sites=int(n_sites), images=len(maps), projection_chunk=int(projection_chunk),
        input_bond_labels=len(edges), outside_bond_labels=len(outside), touching_bond_labels=len(touching),
        fixed_push_probabilities=probabilities.tolist(), maximum_predecessors=6,
        padded_raw_center_queries_per_image=6*(1+len(touching)),
        padded_cached_exchange_queries_per_image=6*len(outside),
        padded_raw_center_queries_per_record=len(maps)*6*(1+len(touching)),
        padded_cached_exchange_queries_per_record=len(maps)*6*len(outside),
        counts_include_null_swaps_and_zero_padded_predecessors=True,
        weighted_H_and_push_graph_image_invariant=True, full_H_not_U_H=True,
        coherent_entry_image_mean=True, guide_square_before_image_mean=True,
        guide_normalization='raw_state_norm_divided_by_caller', no_wavefunction_ratios=True,
        full_H_approximation=False)
    return action, metadata


def make_projected_gate_products(config: MixedClebschConfig, images: np.ndarray,
    bonds: Sequence[tuple[int, int, float]], ordered_sites: Sequence[int], *,
    projection_chunk: int = 1,
) -> tuple[Callable, dict]:
    """CT feature-LUT adapter with native width: rebuild bank inside raw AD."""
    from ..models.triangular_clebsch_feature_lut import make_feature_lut_amplitude
    from ..models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
    return make_gate_products_from_queries(make_feature_lut_amplitude(config),
        lambda edges: make_feature_lut_exchange_queries(config, edges),
        images, bonds, ordered_sites, n_sites=config.length**2,
        projection_chunk=projection_chunk)
