"""Unbiased image-subsampled coherent-A1 Heisenberg action for triangular CT.

The exact coherent-A1 action costs ``|G|`` raw-tree evaluations plus
``|G| * n_bonds`` ancestor-path updates per configuration.  This module keeps
the projected amplitude ``A(x) = |G|^-1 sum_g psi(g x)`` EXACT (it needs only
the ``|G|`` cheap raw evaluations) and estimates only the numerator

    (H P psi)(x) = |G|^-1 sum_g (H psi)(g x)

from ``m`` randomly selected images.  H commutes with every verified lattice
permutation, so each image term is the RAW Heisenberg action at ``g x``.

For a configuration-independent-in-``g`` constant ``beta`` (a control
variate; the current energy estimate is a good choice),

    (H P psi)(x) = beta * A(x) + |G|^-1 sum_g [(H psi)(g x) - beta psi(g x)],

and the second sum is estimated by

* ``uniform``: ``m`` distinct images drawn uniformly without replacement,
  estimator ``mean_S r_g``;
* ``amplitude``: ``m`` images drawn with replacement from the defensive
  mixture ``pi_g = (1-eps)|psi(g x)|/sum|psi| + eps/|G|``, estimator
  ``mean_S r_g / (|G| pi_g)``.  The raw amplitudes are already known.

Both are exactly unbiased for every ``beta``.  ``m = |G|`` with ``uniform``
reproduces the exact action.  Only the numerator is random, so the returned
``hpsi`` may be used wherever an exact ``Hpsi`` enters LINEARLY (the energy
numerator and the SR force).  It must NOT be squared to estimate variances:
use the exact action for audits.  No doubled-q label, CG table, image weight
or bond coefficient is altered.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..models.triangular_clebsch_feature_lut import make_feature_lut_amplitude
from ..models.triangular_mixed_clebsch import MixedClebschConfig
from .triangular_clebsch_cached_action import ExchangeKernel, _exchange_queries

ImageScheme = Literal['uniform', 'amplitude']


def _validated_images(config: MixedClebschConfig, images: np.ndarray, chunk: int) -> np.ndarray:
    n = config.length**2
    maps = np.asarray(images)
    if (maps.ndim != 2 or maps.shape[1] != n or not len(maps)
            or not np.issubdtype(maps.dtype, np.integer)
            or not np.all(np.sort(maps, axis=1) == np.arange(n))
            or isinstance(chunk, bool) or not isinstance(chunk, Integral)
            or chunk < 1 or len(maps) % chunk):
        raise ValueError('physical permutations and a positive dividing chunk required')
    return maps.astype(np.int32)


def _validated_bonds(maps: np.ndarray, bonds: Sequence[tuple[int, int, float]]):
    edges = np.asarray([(a, b) for a, b, _ in bonds], np.int32)
    coefficients = np.asarray([c for _, _, c in bonds], np.float64)
    if edges.ndim != 2 or not len(edges) or not np.isfinite(coefficients).all():
        raise ValueError('nonempty finite real Heisenberg bonds required')
    reference = {}
    for (a, b), value in zip(edges, coefficients, strict=True):
        key = tuple(sorted((int(a), int(b))))
        reference[key] = reference.get(key, 0.)+float(value)
    reference = {k: v for k, v in reference.items() if v != 0.}
    for image in maps:
        moved = {}
        for (a, b), value in zip(edges, coefficients, strict=True):
            key = tuple(sorted((int(image[a]), int(image[b]))))
            moved[key] = moved.get(key, 0.)+float(value)
        moved = {k: v for k, v in moved.items() if v != 0.}
        if moved.keys() != reference.keys() or any(
                not np.isclose(v, moved[k], rtol=0, atol=1e-14) for k, v in reference.items()):
            raise ValueError('spatial permutation does not preserve complete weighted H')
    return edges, coefficients


def make_image_resolved_heisenberg(
    config: MixedClebschConfig, images: np.ndarray,
    bonds: Sequence[tuple[int, int, float]], *, projection_chunk: int = 4,
    active_top_widths: Sequence[Mapping[int, int]] | None = None,
    exchange_kernel: ExchangeKernel = 'auto',
) -> dict[str, Callable]:
    """Per-image raw building blocks on ONE prepared feature-LUT bank.

    Returns callables (unjitted, single configuration):
    ``raw_amplitudes(prepared,bits)->(|G|,)`` with entries ``psi(g x)``;
    ``image_actions(prepared,bits,indices)->(psi(g x),(H psi)(g x))`` for the
    selected image indices; ``bond_terms(prepared,bits)->(n_bonds,)`` giving
    the exact projected off-diagonal bond contributions (diagnostics only).
    """
    maps = _validated_images(config, images, projection_chunk)
    edges_np, coefficients_np = _validated_bonds(maps, bonds)
    raw = make_feature_lut_amplitude(config, active_top_widths=active_top_widths)
    query = _exchange_queries(config, edges_np, active_top_widths, exchange_kernel)
    edges, coefficients = jnp.asarray(edges_np), jnp.asarray(coefficients_np)
    table = jnp.asarray(maps)
    chunks = jnp.asarray(maps.reshape(-1, projection_chunk, config.length**2))

    def diagonal(bits):
        opposite = bits[edges[:, 0]] != bits[edges[:, 1]]
        return jnp.sum(coefficients*jnp.where(opposite, -.25, .25)), opposite

    def raw_amplitudes(prepared, bits):
        def block(group):
            return jax.vmap(raw, in_axes=(None, 0))(prepared, bits[group])
        return jax.lax.map(block, chunks).reshape(-1)

    def one_image(prepared, image_bits):
        psi, swapped = query(prepared, image_bits)
        diag, opposite = diagonal(image_bits)
        off = jnp.sum(coefficients*jnp.where(opposite, .5*swapped, 0.))
        return psi, diag*psi+off

    def image_actions(prepared, bits, indices):
        # Bounded batches: a full vmap over many images x all swaps is memory-bound.
        return jax.lax.map(lambda i: one_image(prepared, bits[table[i]]), indices,
                           batch_size=projection_chunk)

    def bond_terms(prepared, bits):
        def block(group):
            def single(image_bits):
                _, swapped = query(prepared, image_bits)
                opposite = image_bits[edges[:, 0]] != image_bits[edges[:, 1]]
                return coefficients*jnp.where(opposite, .5*swapped, 0.)
            return jnp.sum(jax.vmap(single)(bits[group]), axis=0)
        return jnp.sum(jax.lax.map(block, chunks), axis=0)/len(maps)

    return dict(raw_amplitudes=raw_amplitudes, image_actions=image_actions,
                bond_terms=bond_terms, image_count=len(maps), bond_count=len(edges_np))


def image_selection(key: jax.Array, raw: jax.Array, samples: int, scheme: ImageScheme,
                    defensive: float) -> tuple[jax.Array, jax.Array]:
    """Return (indices, per-index estimator weights) for one configuration.

    ``weights`` multiply ``r_g`` so that ``mean(weights*r)`` is unbiased for
    ``|G|^-1 sum_g r_g``.
    """
    count = raw.shape[0]
    if scheme == 'uniform':
        indices = jax.random.choice(key, count, (samples,), replace=False)
        return indices, jnp.ones((samples,), raw.dtype)
    magnitude = jnp.abs(raw)
    total = jnp.sum(magnitude)
    shaped = jnp.where(total > 0, magnitude/jnp.where(total > 0, total, 1.), 1./count)
    probability = (1.-defensive)*shaped+defensive/count
    indices = jax.random.choice(key, count, (samples,), replace=True, p=probability)
    return indices, 1./(count*probability[indices])


def make_subsampled_projected_action(
    config: MixedClebschConfig, images: np.ndarray,
    bonds: Sequence[tuple[int, int, float]], *, image_samples: int,
    scheme: ImageScheme = 'amplitude', defensive: float = .1,
    projection_chunk: int = 4,
    active_top_widths: Sequence[Mapping[int, int]] | None = None,
    exchange_kernel: ExchangeKernel = 'auto',
) -> Callable:
    """Return jitted action(prepared,rows,keys,beta)->(P psi exact, H P psi estimate).

    ``rows``: (batch,N) physical bits; ``keys``: (batch,2) PRNG keys, one per
    row (independent across rows); ``beta``: scalar control-variate energy
    (total, not per site).  The first output equals the exact projected
    amplitude; the second is unbiased for the exact projected action.
    """
    if isinstance(image_samples, bool) or not isinstance(image_samples, Integral) or image_samples < 1:
        raise ValueError('positive integer image sample count required')
    if scheme not in ('uniform', 'amplitude'):
        raise ValueError("scheme must be 'uniform' or 'amplitude'")
    if not 0. < float(defensive) <= 1.:
        raise ValueError('defensive mixture weight must be in (0,1]')
    parts = make_image_resolved_heisenberg(config, images, bonds, projection_chunk=projection_chunk,
                                           active_top_widths=active_top_widths,
                                           exchange_kernel=exchange_kernel)
    if scheme == 'uniform' and image_samples > parts['image_count']:
        raise ValueError('uniform sampling without replacement needs m <= |G|')
    m, eps = int(image_samples), float(defensive)

    def one(prepared, bits, key, beta):
        raw = parts['raw_amplitudes'](prepared, bits)
        amplitude = jnp.mean(raw)
        indices, weights = image_selection(key, jax.lax.stop_gradient(raw), m, scheme, eps)
        psi_s, hpsi_s = parts['image_actions'](prepared, bits, indices)
        correction = jnp.mean(weights*(hpsi_s-beta*psi_s))
        return amplitude, beta*amplitude+correction

    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, None)))


def conditional_estimator_variance(raw: np.ndarray, hraw: np.ndarray, beta: float,
                                   samples: int, scheme: ImageScheme,
                                   defensive: float = .1) -> float:
    """Exact Var of the H P psi estimator given complete per-image data.

    ``raw``/``hraw`` hold psi(g x) and (H psi)(g x) for ALL images. Uses the
    finite-population correction for uniform sampling without replacement.
    """
    raw, hraw = np.asarray(raw, np.float64), np.asarray(hraw, np.float64)
    r = hraw-beta*raw
    count = len(r)
    if scheme == 'uniform':
        if samples >= count:
            return 0.
        s2 = np.var(r, ddof=1)
        return float((1.-samples/count)*s2/samples)
    magnitude = np.abs(raw)
    shaped = magnitude/magnitude.sum() if magnitude.sum() > 0 else np.full(count, 1./count)
    p = (1.-defensive)*shaped+defensive/count
    y = r/(count*p)
    return float((np.sum(p*y*y)-np.sum(p*y)**2)/samples)
