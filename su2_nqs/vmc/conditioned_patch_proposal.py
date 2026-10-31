"""Independent physical-bit proposals from local diagonals, not TTN Born draws.

A product of normalized patch diagonals is conditioned on an exact total
number of up spins by dynamic programming. Optional permutation images and
a uniform fixed-Sz floor remain normalized *classical proposals*. They do
not implement a coherent spatial projection, nor approximate its density
by an incoherent mixture. The target amplitude is specified separately.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from math import log
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from ..su2.reduced_tree_environment import ReducedSingletTree, reduced_tree_environments
from .constrained_uniform import sample_uniform_sz_configs, uniform_sz_log_probability


def physical_leaf_diagonals(
    state: ReducedSingletTree,
    features: Sequence[Mapping[int, np.ndarray]],
    *, tolerance: float = 1e-11,
) -> tuple[np.ndarray, dict]:
    """Return exact raw-tree physical leaf diagonals and norm diagnostics.

    ``features[i][q][bits,a]`` contains the physical block amplitude with
    doubled magnetic label ``m=2*popcount(bits)-n_patch``; site zero is MSB.
    Its columns must be an already validated local SU(2) intertwiner. This
    helper checks magnetic support and every fixed-m Gram against the tree.
    It retains complex multiplicity coherence through the complementary
    environment, rather than drawing virtual labels as a classical mixture.

    All leaves must contain the same number of physical spin-1/2 sites.
    The returned diagonals belong to ONE RAW TREE, not its coherent A1 sum.
    """
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("tolerance must lie in (0,1)")
    if len(features) != len(state.leaf_grams):
        raise ValueError("one physical feature mapping per leaf required")
    environment = reduced_tree_environments(state)
    if environment.maximum_relative_edge_norm_error > tolerance:
        raise ArithmeticError("raw tree edge norms disagree")
    probabilities = []
    max_gram_error = max_imaginary = max_trace_error = 0.0
    dimension = None
    for leaf, gram, block in zip(state.tree.leaves, state.leaf_grams, features, strict=True):
        if set(block) != set(gram):
            raise ValueError("physical feature charges must match leaf Gram charges")
        local = None
        for q in sorted(block):
            values = np.asarray(block[q], dtype=np.complex128)
            if (values.ndim != 2 or values.shape[1] != gram[q].shape[0]
                    or not np.isfinite(values).all()):
                raise ValueError("finite (physical states,multiplicity) features required")
            if dimension is None:
                dimension = values.shape[0]
            if values.shape[0] != dimension or dimension < 2 or dimension & (dimension-1):
                raise ValueError("all physical leaf dimensions must be the same power of two")
            n_patch = dimension.bit_length()-1
            magnetic = np.asarray([2*i.bit_count()-n_patch for i in range(dimension)])
            forbidden = (np.abs(magnetic) > q) | ((magnetic-q) % 2 != 0)
            scale = max(float(np.max(np.abs(values), initial=0)), np.finfo(float).tiny)
            if np.max(np.abs(values[forbidden]), initial=0) > tolerance*scale:
                raise ValueError("physical feature violates doubled magnetic support")
            for m in range(-q, q+1, 2):
                selected = values[magnetic == m]
                reference = np.asarray(gram[q])
                error = float(np.max(np.abs(selected.conj().T @ selected-reference), initial=0))
                max_gram_error = max(max_gram_error, error)
                if error > tolerance*max(float(np.max(np.abs(reference))), np.finfo(float).tiny):
                    raise ValueError("fixed-m physical feature Gram disagrees with raw tree")
            contribution = np.einsum("xa,ab,xb->x", values,
                                      environment.downward[leaf][q], values.conj(), optimize=True)
            local = contribution if local is None else local+contribution
        if local is None:
            raise ValueError("a physical leaf cannot have no charges")
        local = local/environment.norm_squared
        max_imaginary = max(max_imaginary, float(np.max(np.abs(local.imag))))
        if np.max(np.abs(local.imag)) > tolerance or np.min(local.real) < -tolerance:
            raise ArithmeticError("derived physical diagonal is not real and nonnegative")
        trace_error = abs(float(local.real.sum())-1.)
        max_trace_error = max(max_trace_error, trace_error)
        if trace_error > tolerance:
            raise ArithmeticError("physical leaf diagonal is not normalized")
        # Only contraction roundoff of an exact positive double layer is removed.
        diagonal = np.maximum(local.real, 0.)
        probabilities.append(diagonal/diagonal.sum())
    return np.asarray(probabilities), {
        "raw_tree_norm_squared": environment.norm_squared,
        "maximum_relative_edge_norm_error": environment.maximum_relative_edge_norm_error,
        "maximum_fixed_m_gram_error": max_gram_error,
        "maximum_diagonal_imaginary_part": max_imaginary,
        "maximum_diagonal_trace_error": max_trace_error,
        "coherent_A1_diagonal": False,
    }


def make_conditioned_patch_proposal(
    patches: Sequence[Sequence[int]],
    probabilities: Sequence[Sequence[float]],
    images: Sequence[Sequence[int]] | None = None,
    *, n_up: int, uniform_fraction: float = .05, image_batch_size: int = 4,
) -> tuple[Callable, Callable, dict]:
    """Build ``log_density(batch), draw(key,count), diagnostics`` for IID Q.

    ``patches`` partitions sites into equal, ordered physical blocks. Each
    row of ``probabilities`` is normalized in site-zero-MSB bit order. The
    product density is conditioned on exactly ``n_up`` ones. Offline log-DP
    supplies exact completion weights before each categorical draw; invalid
    magnetic sectors are never sampled and then repaired.

    Images use active ``image[old_site]=new_site`` permutations. A uniformly
    selected image pushes the conditional product forward. Any finite list
    of permutations is valid; closure and lattice symmetry are NOT assumed.
    The density averages precisely that same list, using bounded batches.
    The fixed-Sz uniform floor guarantees support only when its fraction is
    positive. It does not certify finite-sample tail coverage or importance
    effective sample size. ``count`` must be static under JIT.

    Save the supplied probability array, patches/images, diagnostics and
    the actual independent draw seed together to reproduce an audit.
    """
    sites = np.asarray(patches)
    if (sites.ndim != 2 or min(sites.shape, default=0) < 1
            or not np.issubdtype(sites.dtype, np.integer)):
        raise ValueError("patches must be a nonempty rectangular integer array")
    n_patches, patch_size = sites.shape
    n_sites = sites.size
    if not np.array_equal(np.sort(sites.ravel()), np.arange(n_sites)):
        raise ValueError("patches must partition sites 0..N-1 exactly once")
    if isinstance(n_up, bool) or not isinstance(n_up, Integral) or not 0 <= n_up <= n_sites:
        raise ValueError("integer n_up in 0..N required")
    if patch_size > 20:
        raise ValueError("explicit physical patch diagonals are limited to 20 spins")
    if np.iscomplexobj(probabilities):
        raise ValueError("physical patch probabilities must be real")
    p = np.asarray(probabilities, dtype=np.float64)
    if (p.shape != (n_patches, 1 << patch_size) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1., atol=1e-12, rtol=0)):
        raise ValueError("normalized nonnegative physical patch probabilities required")
    if not np.isfinite(uniform_fraction) or not 0 <= uniform_fraction <= 1:
        raise ValueError("uniform_fraction must lie in [0,1]")
    if (isinstance(image_batch_size, bool) or not isinstance(image_batch_size, Integral)
            or image_batch_size < 1):
        raise ValueError("positive integer image_batch_size required")
    permutations = np.arange(n_sites)[None, :] if images is None else np.asarray(images)
    if (permutations.ndim != 2 or permutations.shape[1] != n_sites or not len(permutations)
            or not np.issubdtype(permutations.dtype, np.integer)
            or np.any(np.sort(permutations, axis=1) != np.arange(n_sites))):
        raise ValueError("images must be a nonempty array of site permutations")

    up_counts = np.asarray([x.bit_count() for x in range(1 << patch_size)], dtype=np.int32)
    sector_mass = np.stack([np.bincount(up_counts, weights=row, minlength=patch_size+1) for row in p])
    log_p = np.full_like(p, -np.inf)
    np.log(p, out=log_p, where=p > 0)
    log_mass = np.full_like(sector_mass, -np.inf)
    np.log(sector_mass, out=log_mass, where=sector_mass > 0)
    suffix = np.full((n_patches+1, n_sites+1), -np.inf)
    suffix[-1, 0] = 0.
    for i in reversed(range(n_patches)):
        for up in range(n_sites+1):
            choices = [log_mass[i, k]+suffix[i+1, up-k] for k in range(min(patch_size, up)+1)]
            suffix[i, up] = np.logaddexp.reduce(choices)
    log_condition = float(suffix[0, n_up])
    if not np.isfinite(log_condition):
        raise ValueError("product proposal has no support in the requested fixed-Sz sector")
    log_uniform = uniform_sz_log_probability(n_sites, int(n_up))
    log_floor = -np.inf if uniform_fraction == 0 else log(uniform_fraction)+log_uniform
    log_product_fraction = -np.inf if uniform_fraction == 1 else log(1-uniform_fraction)
    patch_indices = jnp.asarray(sites, dtype=jnp.int32)
    image_indices = jnp.asarray(permutations, dtype=jnp.int32)
    log_probabilities, suffix_j = jnp.asarray(log_p), jnp.asarray(suffix)
    up_j = jnp.asarray(up_counts)
    powers = 1 << jnp.arange(patch_size-1, -1, -1, dtype=jnp.int32)
    local_bits = ((np.arange(1 << patch_size)[:, None] >> np.arange(patch_size-1, -1, -1)) & 1).astype(np.int32)
    local_bits_j = jnp.asarray(local_bits)

    def log_density(states: jax.Array) -> jax.Array:
        values = jnp.asarray(states)
        if values.ndim != 2 or values.shape[1] != n_sites:
            raise ValueError("states must be (batch,N) physical-bit arrays")
        valid = jnp.all((values == 0) | (values == 1), axis=1) & (values.sum(axis=1) == n_up)
        bits = jnp.clip(values, 0, 1).astype(jnp.int32)

        def image_log_density(image):
            indices = jnp.sum(bits[:, image][:, patch_indices]*powers, axis=-1)
            return jnp.sum(log_probabilities[jnp.arange(n_patches)[None, :], indices], axis=1)-log_condition

        per_image = jax.lax.map(image_log_density, image_indices, batch_size=int(image_batch_size))
        averaged = logsumexp(per_image, axis=0)-log(len(permutations))
        result = jnp.logaddexp(log_product_fraction+averaged, log_floor)
        return jnp.where(valid, result, -jnp.inf)

    def draw(key: jax.Array, count: int) -> jax.Array:
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError("count must be a positive static integer")
        key, image_key, uniform_key, floor_key = jax.random.split(key, 4)
        remaining = jnp.full((count,), n_up, dtype=jnp.int32)
        product = jnp.zeros((count, n_sites), dtype=jnp.int32)
        for i in range(n_patches):
            key, choose_key = jax.random.split(key)
            suffix_up = remaining[:, None]-up_j[None, :]
            suffix_value = suffix_j[i+1, jnp.clip(suffix_up, 0, n_sites)]
            logits = log_probabilities[i][None, :]+jnp.where(suffix_up >= 0, suffix_value, -jnp.inf)
            selected = jax.random.categorical(choose_key, logits, axis=-1)
            product = product.at[jnp.arange(count)[:, None], patch_indices[i]].set(local_bits_j[selected])
            remaining = remaining-up_j[selected]
        choice = jax.random.randint(image_key, (count,), 0, len(permutations))
        pushed = jnp.zeros_like(product).at[jnp.arange(count)[:, None], image_indices[choice]].set(product)
        uniform = sample_uniform_sz_configs(uniform_key, int(count), n_sites, int(n_up))
        use_floor = jax.random.bernoulli(floor_key, float(uniform_fraction), (count,))
        return jnp.where(use_floor[:, None], uniform, pushed)

    digest = sha256(b"conditioned_patch_proposal_v1")
    for values, dtype in ((sites, "<i8"), (p, "<f8"), (permutations, "<i8")):
        array = np.asarray(values, dtype=dtype)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    digest.update(np.asarray([n_up], dtype="<i8").tobytes())
    digest.update(np.asarray([uniform_fraction], dtype="<f8").tobytes())
    diagnostics = {
        "schema": "conditioned_patch_proposal_v1", "proposal_sha256": digest.hexdigest(),
        "n_sites": n_sites, "n_patches": n_patches, "patch_size": patch_size,
        "n_up": int(n_up), "doubled_total_m": 2*int(n_up)-n_sites,
        "image_count": len(permutations), "image_batch_size": int(image_batch_size),
        "uniform_fraction": float(uniform_fraction),
        "log_product_condition_probability": log_condition,
        "log_uniform_probability": log_uniform,
        "iid_draws": True, "full_fixed_Sz_support": bool(uniform_fraction > 0),
        "coherent_projection_density": False,
        "group_closure_assumed": False, "unseen_tail_coverage_certified": False,
    }
    return log_density, draw, diagnostics


__all__ = ["make_conditioned_patch_proposal", "physical_leaf_diagonals"]
