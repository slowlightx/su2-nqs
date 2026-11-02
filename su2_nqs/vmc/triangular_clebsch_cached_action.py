"""Node-safe coherent-A1 full Heisenberg action with ancestor message reuse.

The model query is Hamiltonian-free. Here H commutes through the verified
physical spatial permutations, so each image builds one set of base messages
and reuses unchanged subtrees over every H bond. No single-image ratio is taken.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
from ..models.triangular_mixed_clebsch import MixedClebschConfig
from .exchange_bridge_importance import make_coherent_exchange_bridge_products


def make_cached_projected_heisenberg_bridge(
    config: MixedClebschConfig, images: np.ndarray,
    bonds: Sequence[tuple[int, int, float]], probabilities: Sequence[float], *,
    projection_chunk: int = 4,
    active_top_widths: Sequence[Mapping[int, int]] | None = None,
) -> tuple[Callable, dict]:
    """One full-A1 connection sweep returns psi, Hpsi, and bridge density.

    The callable acts on one prepared endpoint bank and a physical-bit batch.
    Its third output is sum_b p_b |P_A1 psi(T_b x)|^2. Invoke the SAME compiled
    callable for each frozen endpoint; sum these outputs to construct Qpush,
    and sum |psi|^2 to construct Q0. Qmix=(1-f)*Q0+f*Qpush shares the same
    unknown normalization as Q0. This is not an IID sampler or an energy
    approximation. Repeated/null bond labels retain their specified weights.

    Active upper widths require exact fixed-bank closure/repacking as in the
    original action. The unchanged legacy full-H factory below remains the
    default for production; this optional path adds coherent connection
    storage/reindexing whose actual cost must be measured.
    """
    def factory(edges):
        return make_feature_lut_exchange_queries(config, edges,
                                                 active_top_widths=active_top_widths)
    return make_coherent_exchange_bridge_products(factory, images, bonds,
        probabilities, n_sites=config.length**2, projection_chunk=projection_chunk)


def make_cached_projected_heisenberg_action(
    config: MixedClebschConfig, images: np.ndarray,
    bonds: Sequence[tuple[int, int, float]], *, projection_chunk: int = 4,
    active_top_widths: Sequence[Mapping[int, int]] | None = None,
) -> Callable:
    """Return action(prepared_bank,batch_bits)->(P_A1 psi,H P_A1 psi).

    All retained doubled-q/m labels, image phases, weighted bonds and repeated
    bond terms remain unchanged. ``prepared_bank`` must be rebuilt after every
    parameter update, inside the gradient/JVP when differentiating parameters.
    Permutations must preserve the complete weighted Hamiltonian; otherwise
    commuting H through the projector would be invalid and is rejected.
    Optional active upper widths are an exact fixed-bank execution reduction,
    after ``repack_active_clebsch_bank`` verifies inactive-path closure. They
    must not be substituted for full native channel-opening gradients.
    """
    n = config.length**2
    maps = np.asarray(images)
    if (maps.ndim != 2 or maps.shape[1] != n or not len(maps)
            or not np.issubdtype(maps.dtype, np.integer)
            or not np.all(np.sort(maps, axis=1) == np.arange(n))
            or isinstance(projection_chunk, bool) or not isinstance(projection_chunk, int)
            or projection_chunk < 1 or len(maps) % projection_chunk):
        raise ValueError('physical permutations and a positive dividing chunk required')
    edges = np.asarray([(a, b) for a, b, _ in bonds])
    coefficients = np.asarray([c for _, _, c in bonds], np.float64)
    query = make_feature_lut_exchange_queries(config, edges, active_top_widths=active_top_widths)
    if not np.isfinite(coefficients).all():
        raise ValueError('finite real Heisenberg couplings required')
    weights = {}
    for (a, b), value in zip(edges, coefficients, strict=True):
        key = tuple(sorted((int(a), int(b))))
        weights[key] = weights.get(key, 0.)+float(value)
    # Cancelled/explicit zero terms do not impose a fictitious graph symmetry.
    # They remain in the actual action/query list below and contribute zero.
    weights = {key: value for key, value in weights.items() if value != 0.}
    for image in maps:
        transformed = {}
        for (a, b), value in zip(edges, coefficients, strict=True):
            key = tuple(sorted((int(image[a]), int(image[b]))))
            transformed[key] = transformed.get(key, 0.)+float(value)
        transformed = {key: value for key, value in transformed.items() if value != 0.}
        if weights.keys() != transformed.keys() or any(
                not np.isclose(value, transformed[key], rtol=0, atol=1e-14)
                for key, value in weights.items()):
            raise ValueError('spatial permutation does not preserve complete weighted H')
    edges = jnp.asarray(edges)
    coefficients = jnp.asarray(coefficients)
    chunks = jnp.asarray(maps.reshape(-1, projection_chunk, n))

    def image_action(bank, bits):
        psi, swapped = query(bank, bits)
        opposite = bits[edges[:, 0]] != bits[edges[:, 1]]
        diagonal = jnp.sum(coefficients*jnp.where(opposite, -.25, .25))
        off = jnp.sum(coefficients*jnp.where(opposite, .5*swapped, 0.))
        return jnp.stack((psi, diagonal*psi+off))

    def one(bank, bits):
        def partial(group):
            return jnp.sum(jax.vmap(image_action, in_axes=(None, 0))(bank, bits[group]), axis=0)
        result = jnp.sum(jax.lax.map(jax.checkpoint(partial), chunks), axis=0)/len(maps)
        return result[0], result[1]

    return jax.jit(jax.vmap(one, in_axes=(None, 0)))
