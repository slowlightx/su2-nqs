"""Coherent spatial projection of identical physical-amplitude patch products.

This is an initialization/reference construction, not a new trained head.  The
parent is explicitly projected BEFORE copies are multiplied.  Copying an
unprojected tensor bank generally gives a different state.  No Hamiltonian,
sampling, or SU(2) approximation is introduced here: a product of singlet
parents and its spatial projection remain singlets.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.spatial_symmetry import validate_site_permutation


def _checked_group(images: Iterable[Sequence[int]], n_sites: int) -> tuple:
    """Certify closure using a generating subset, without a quadratic table."""
    checked = tuple(validate_site_permutation(g, n_sites=n_sites) for g in images)
    if not checked or len(set(checked)) != len(checked):
        raise ValueError('a nonempty, duplicate-free permutation group is required')
    identity = tuple(range(n_sites))
    allowed = set(checked)
    if identity not in allowed:
        raise ValueError('permutation group must contain the identity')
    generated = {identity}
    generators = []
    for candidate in sorted(checked):
        if candidate in generated:
            continue
        generators.append(candidate)
        # Restart from the old subgroup: adding a generator can also make its
        # product with any old element new.  Every product must remain allowed.
        queue = list(generated)
        for current in queue:
            for generator in generators:
                product = tuple(generator[current[i]] for i in range(n_sites))
                if product not in allowed:
                    raise ValueError('permutation images are not a closed group')
                if product not in generated:
                    generated.add(product)
                    queue.append(product)
    if generated != allowed:
        raise ValueError('permutation group could not be generated')
    return tuple(sorted(checked))


@dataclass(frozen=True)
class PatchProductOrbit:
    """Exact A1 orbit compression for identical, source-invariant copies.

    Site maps use ``image[old_site] = new_site``.  Each patch is an ordered list
    in the parent's site convention. ``representatives[g][patch][site]`` is a
    target site index; positive ``counts`` retain every original target image.
    The trivial character is essential: this compression is NOT valid for a
    sign-changing parent irrep or for independent parameters on different copies.
    """
    patches: tuple[tuple[int, ...], ...]
    source_images: tuple[tuple[int, ...], ...]
    representatives: tuple[tuple[tuple[int, ...], ...], ...]
    counts: tuple[int, ...]
    target_group_size: int

    @property
    def n_sites(self) -> int:
        return len(self.patches)*len(self.patches[0])

    @property
    def source_sites(self) -> int:
        return len(self.patches[0])

    @property
    def parent_raw_queries_per_amplitude(self) -> int:
        """Straight evaluation count, before configuration-dependent reuse."""
        return len(self.representatives)*len(self.patches)*len(self.source_images)


def patch_product_orbit(
    patches: Sequence[Sequence[int]],
    source_images: Iterable[Sequence[int]],
    target_images: Iterable[Sequence[int]],
) -> PatchProductOrbit:
    """Validate disjoint ordered patches and deduplicate the exact target A1 sum.

    Canonicalize each patch under the parent's A1 group, then sort identical
    copies.  These are amplitude identities, not an incoherent probability
    average.  The resulting state can be identically zero; normalization and
    Born-sampling checks remain the caller's responsibility.
    """
    raw = tuple(tuple(p) for p in patches)
    if not raw or not raw[0] or any(len(p) != len(raw[0]) for p in raw):
        raise ValueError('patches must be nonempty and have equal sizes')
    if any(isinstance(i, bool) or not isinstance(i, Integral) for p in raw for i in p):
        raise TypeError('patch site labels must be integers')
    raw = tuple(tuple(int(i) for i in p) for p in raw)
    n_sites = len(raw)*len(raw[0])
    if sorted(i for p in raw for i in p) != list(range(n_sites)):
        raise ValueError('patches must partition every target site exactly once')
    source = _checked_group(source_images, len(raw[0]))
    target = _checked_group(target_images, n_sites)
    source_array = np.asarray(source, dtype=np.int64)
    patches_array = np.asarray(raw, dtype=np.int64)
    orbit: Counter = Counter()
    for image in target:
        mapped = np.asarray(image, dtype=np.int64)[patches_array]
        canonical = []
        for patch in mapped:
            versions = patch[source_array]
            # Lexicographic sorting is exact in integer site indices.
            order = np.lexsort(versions[:, ::-1].T)
            canonical.append(tuple(int(i) for i in versions[order[0]]))
        orbit[tuple(sorted(canonical))] += 1
    representatives = tuple(sorted(orbit))
    return PatchProductOrbit(raw, source, representatives,
                             tuple(orbit[g] for g in representatives), len(target))


def make_projected_patch_product(
    raw_parent: Callable,
    orbit: PatchProductOrbit,
) -> Callable:
    """Return ``amplitude(shared_parameters, bits)`` with both coherent A1 sums.

    ``raw_parent(parameters, parent_bits)`` must return one real/complex physical
    amplitude, not a log amplitude.  The function is pure JAX and differentiates
    through every source-image sum and all copies sharing the same parameters.
    Sequential ``lax.map`` bounds simultaneous image storage; this does not make
    the potentially large exact query count cheap.  No implicit normalization
    or magnitude-only operation is applied.
    """
    if not isinstance(orbit, PatchProductOrbit):
        raise TypeError('orbit must be produced by patch_product_orbit')
    source = jnp.asarray(orbit.source_images, dtype=jnp.int32)
    representatives = jnp.asarray(orbit.representatives, dtype=jnp.int32)
    counts = jnp.asarray(orbit.counts)

    def amplitude(parameters: object, bits: jax.Array) -> jax.Array:
        bits = jnp.asarray(bits)
        if bits.shape != (orbit.n_sites,) or not jnp.issubdtype(bits.dtype, jnp.integer):
            raise ValueError('one integer-bit physical configuration is required')
        valid = jnp.all((bits == 0) | (bits == 1))
        bits = jnp.clip(bits, 0, 1)

        def parent(patch_bits):
            values = jax.lax.map(lambda image: raw_parent(parameters, patch_bits[image]), source)
            if values.ndim != 1:
                raise ValueError('raw_parent must return one scalar physical amplitude')
            return jnp.mean(values)

        def product(representative):
            return jnp.prod(jax.lax.map(parent, bits[representative]))

        values = jax.lax.map(product, representatives)
        value = jnp.sum(values*counts)/orbit.target_group_size
        return jnp.where(valid, value, jnp.zeros_like(value))

    return amplitude
