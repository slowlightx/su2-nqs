"""On-demand physical-basis lattice projection for spin and fermion NQSs.

The projector acts on amplitudes, not probabilities and not training data.
It therefore preserves interference between symmetry images.  The fermionic
variant includes the canonical mode-reordering sign for every site map.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from typing import Mapping, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.spatial_symmetry import (
    SitePermutation,
    compose_site_permutations,
    inverse_site_permutation,
    triangular_hnf_little_space_group_character,
    validate_site_permutation,
    validated_permutation_group,
)
from ..su2.hubbard_cg_irrep import DOUBLON, DOWN, UP


class PhysicalAmplitude(Protocol):
    """One physical-configuration amplitude with a static model config."""

    def __call__(self, params: object, state: jax.Array, config: object) -> jax.Array:
        ...


@dataclass(frozen=True)
class PhysicalSpatialProjector:
    """A finite one-dimensional-character projector in the physical basis."""

    site_permutations: tuple[SitePermutation, ...]
    characters: tuple[complex, ...] | None = None
    fermionic: bool = False
    orbit_chunk_size: int | None = None
    rematerialize_orbit: bool = False

    @classmethod
    def triangular_hnf_little_group(
        cls,
        supercell: tuple[tuple[int, int], tuple[int, int]],
        momentum_numerator: tuple[int, int],
        point_characters: Mapping[str, complex],
        *,
        fermionic: bool = False,
        orbit_chunk_size: int | None = None,
        rematerialize_orbit: bool = False,
    ) -> "PhysicalSpatialProjector":
        """Construct an exact scalar-character projector at nonzero momentum.

        Only finite-torus point operations in the reciprocal little group are
        included.  A point operation that moves the momentum to another star
        arm requires a multidimensional sewing representation and is rejected
        by the underlying lattice helper rather than being mislabeled as an
        A1 scalar projection.
        """

        images, characters = triangular_hnf_little_space_group_character(
            supercell, momentum_numerator, point_characters
        )
        return cls(
            images,
            characters,
            fermionic=fermionic,
            orbit_chunk_size=orbit_chunk_size,
            rematerialize_orbit=rematerialize_orbit,
        )

    def __post_init__(self) -> None:
        supplied = tuple(
            validate_site_permutation(image) for image in self.site_permutations
        )
        if len(set(supplied)) != len(supplied):
            raise ValueError("site_permutations must be duplicate free")
        group = validated_permutation_group(supplied)
        object.__setattr__(self, "site_permutations", group)
        if self.characters is None:
            characters = (1.0 + 0.0j,) * len(group)
        else:
            supplied_characters = tuple(complex(value) for value in self.characters)
            if len(supplied_characters) != len(supplied):
                raise ValueError("one character is required per group image")
            by_image = dict(zip(supplied, supplied_characters, strict=True))
            characters = tuple(by_image[image] for image in group)
            if any(
                not np.isfinite(value.real) or not np.isfinite(value.imag)
                for value in characters
            ):
                raise ValueError("projector characters must be finite")
            if any(abs(abs(value) - 1.0) > 1.0e-10 for value in characters):
                raise ValueError("one-dimensional characters must have unit modulus")
            character_by_image = dict(zip(group, characters, strict=True))
            for left, left_character in character_by_image.items():
                for right, right_character in character_by_image.items():
                    product = compose_site_permutations(left, right)
                    if abs(
                        character_by_image[product]
                        - left_character * right_character
                    ) > 1.0e-10:
                        raise ValueError(
                            "characters must form a one-dimensional representation"
                        )
        object.__setattr__(self, "characters", characters)
        if self.orbit_chunk_size is not None:
            if (
                isinstance(self.orbit_chunk_size, bool)
                or not isinstance(self.orbit_chunk_size, Integral)
                or self.orbit_chunk_size < 1
            ):
                raise ValueError("orbit_chunk_size must be a positive integer")
            object.__setattr__(self, "orbit_chunk_size", int(self.orbit_chunk_size))
        if not isinstance(self.rematerialize_orbit, bool):
            raise TypeError("rematerialize_orbit must be boolean")

    @property
    def n_sites(self) -> int:
        return len(self.site_permutations[0])

    @property
    def group_order(self) -> int:
        return len(self.site_permutations)


def spin_pullback_orbit(
    states: jax.Array, projector: PhysicalSpatialProjector
) -> jax.Array:
    """Return physical spin pullbacks with shape ``[..., group, sites]``."""

    if projector.fermionic:
        raise ValueError("spin_pullback_orbit requires a bosonic/spin projector")
    array = jnp.asarray(states, dtype=jnp.int32)
    if array.ndim < 1 or array.shape[-1] != projector.n_sites:
        raise ValueError("spin states and projector site counts disagree")
    images = jnp.asarray(projector.site_permutations, dtype=jnp.int32)
    return jnp.take(array, images, axis=-1)


@lru_cache(maxsize=None)
def _fermionic_inversion_tensors(
    projector: PhysicalSpatialProjector,
) -> np.ndarray:
    """Return ``[group,mode,mode]`` inversion indicators for pullback maps."""

    n_sites = projector.n_sites
    modes = 2 * n_sites
    result = np.zeros((projector.group_order, modes, modes), dtype=np.int8)
    upper = np.arange(modes)[:, None] < np.arange(modes)[None, :]
    for group_index, image in enumerate(projector.site_permutations):
        inverse = inverse_site_permutation(image)
        orbital_map = np.asarray(
            [2 * inverse[site] + spin for site in range(n_sites) for spin in (0, 1)],
            dtype=np.int32,
        )
        result[group_index] = upper & (orbital_map[:, None] > orbital_map[None, :])
    result.setflags(write=False)
    return result


def fermionic_pullback_orbit(
    states: jax.Array, projector: PhysicalSpatialProjector
) -> tuple[jax.Array, jax.Array]:
    """Return Hubbard-token pullbacks and canonical fermionic signs."""

    if not projector.fermionic:
        raise ValueError("fermionic_pullback_orbit requires fermionic=True")
    tokens = jnp.asarray(states, dtype=jnp.int32)
    if tokens.ndim < 1 or tokens.shape[-1] != projector.n_sites:
        raise ValueError("Hubbard states and projector site counts disagree")
    images = jnp.asarray(projector.site_permutations, dtype=jnp.int32)
    pulled = jnp.take(tokens, images, axis=-1)
    occupied_up = (tokens == UP) | (tokens == DOUBLON)
    occupied_down = (tokens == DOWN) | (tokens == DOUBLON)
    occupied = jnp.stack((occupied_up, occupied_down), axis=-1).reshape(
        tokens.shape[:-1] + (2 * projector.n_sites,)
    )
    inversions = jnp.asarray(_fermionic_inversion_tensors(projector), dtype=jnp.int32)
    parity_count = jnp.einsum(
        "...m,gmn,...n->...g",
        occupied.astype(jnp.int32),
        inversions,
        occupied.astype(jnp.int32),
    )
    signs = jnp.where(parity_count % 2, -1.0, 1.0)
    return pulled, signs


def spatial_projected_amplitude(
    params: object,
    state: jax.Array,
    model_config: object,
    projector: PhysicalSpatialProjector,
    *,
    amplitude_fn: PhysicalAmplitude,
) -> jax.Array:
    """Evaluate a coherent character-weighted physical symmetry orbit."""

    characters = jnp.asarray(projector.characters, dtype=jnp.complex128)
    if projector.fermionic:
        orbit, signs = fermionic_pullback_orbit(state, projector)
    else:
        orbit = spin_pullback_orbit(state, projector)
        signs = jnp.ones((projector.group_order,), dtype=jnp.float64)
    chunk_size = projector.orbit_chunk_size or projector.group_order
    total = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)
    evaluate = lambda candidate, row: amplitude_fn(candidate, row, model_config)
    if projector.rematerialize_orbit:
        evaluate = jax.checkpoint(evaluate)
    for start in range(0, projector.group_order, chunk_size):
        stop = min(start + chunk_size, projector.group_order)
        values = jax.vmap(
            lambda row: evaluate(params, row)
        )(orbit[start:stop])
        total = total + jnp.sum(
            jnp.conj(characters[start:stop]) * signs[start:stop] * values
        )
    return total / projector.group_order


def spatial_projected_amplitudes(
    params: object,
    states: jax.Array,
    model_config: object,
    projector: PhysicalSpatialProjector,
    *,
    amplitude_fn: PhysicalAmplitude,
) -> jax.Array:
    """Vectorize :func:`spatial_projected_amplitude` over physical states."""

    array = jnp.asarray(states, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != projector.n_sites:
        raise ValueError("expected a rank-two physical-state batch")
    return jax.vmap(
        lambda row: spatial_projected_amplitude(
            params,
            row,
            model_config,
            projector,
            amplitude_fn=amplitude_fn,
        )
    )(array)


__all__ = [
    "PhysicalAmplitude",
    "PhysicalSpatialProjector",
    "fermionic_pullback_orbit",
    "spatial_projected_amplitude",
    "spatial_projected_amplitudes",
    "spin_pullback_orbit",
]
