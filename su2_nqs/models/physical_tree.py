"""Physical-input bridge facade for Clebsch--Gordan tree amplitudes.

This is the narrow model boundary for the current single-tree exact-small
bridge and physical-``Hpsi`` resource bridge. It is not itself a
scalable-method claim.
Spin models consume computational-basis bits (``0=down``, ``1=up``); Hubbard
models consume physical Fock tokens ``0..3``. Fixed numerical
Clebsch--Gordan coefficients carry magnetic indices and learned tensors carry
only reduced multiplicity features. Direct fusion paths are deliberately
absent from this namespace and remain exact/reference coordinates or legacy
controls.

The current balanced trees, including uses whose leaves follow a Morton site
ordering, are deterministic binary tree coordinates. A leaf order can encode
horizontal or vertical adjacency, but this is *not* the planned multi-cut,
overlapping-cell, multi-leg-boundary fusion network. It must not be presented
as a geometry-native patch construction when interpreting two-dimensional
benchmarks.

The matched Abelian Hubbard tree is exposed here as a same-tree control: it
uses the identical physical token input and merge schedule while enforcing
only fixed ``(Ne, Sz)`` rather than total SU(2).
"""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp

from .hubbard_abelian_tree import (
    HubbardAbelianTreeConfig,
    hubbard_abelian_parameter_count,
    hubbard_abelian_tree_amplitude,
    init_hubbard_abelian_tree,
)
from .hubbard_cg import (
    HubbardCGConfig,
    active_hubbard_cg_amplitude,
    hubbard_cg_parameter_count,
    init_active_hubbard_cg,
)
from .spin_clebsch_tree import (
    SpinClebschTreeConfig,
    init_spin_clebsch_tree,
    spin_clebsch_tree_amplitude,
    spin_clebsch_tree_amplitudes,
    spin_clebsch_tree_parameter_count,
)


class PhysicalTreeConfig(Protocol):
    """Minimum static contract shared by physical-input tree evaluators."""

    n_sites: int


class PhysicalTreeAmplitude(Protocol):
    """Callable protocol for one physical configuration amplitude."""

    def __call__(
        self,
        params: object,
        configuration: jax.Array,
        config: PhysicalTreeConfig,
    ) -> jax.Array: ...


# Canonical physical-input bridge names. The aliases retain compatibility
# with the exact-tested implementation dataclasses.
PhysicalSpinTreeConfig = SpinClebschTreeConfig
PhysicalHubbardTreeConfig = HubbardCGConfig
MatchedHubbardAbelianTreeConfig = HubbardAbelianTreeConfig


def physical_tree_amplitudes(
    params: object,
    configurations: jax.Array,
    config: PhysicalTreeConfig,
    *,
    amplitude_fn: PhysicalTreeAmplitude,
) -> jax.Array:
    """Evaluate any physical-input tree amplitude over a rank-two batch."""

    array = jnp.asarray(configurations, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[1] != config.n_sites:
        raise ValueError(
            f"expected configuration batch shape (batch, {config.n_sites})"
        )
    return jax.vmap(lambda row: amplitude_fn(params, row, config))(array)


def init_physical_spin_tree(
    key: jax.Array,
    config: PhysicalSpinTreeConfig,
) -> dict[str, object]:
    """Initialize a physical-spin exact-SU(2) Clebsch tree."""

    return init_spin_clebsch_tree(key, config)


def physical_spin_tree_amplitude(
    params: dict[str, object],
    spins: jax.Array,
    config: PhysicalSpinTreeConfig,
) -> jax.Array:
    """Evaluate one computational-basis spin configuration."""

    return spin_clebsch_tree_amplitude(params, spins, config)


def physical_spin_tree_amplitudes(
    params: dict[str, object],
    spins: jax.Array,
    config: PhysicalSpinTreeConfig,
) -> jax.Array:
    """Evaluate a batch of computational-basis spin configurations."""

    return spin_clebsch_tree_amplitudes(params, spins, config)


def physical_spin_tree_parameter_count(params: dict[str, object]) -> int:
    """Count trainable scalars in a physical-spin Clebsch tree."""

    return spin_clebsch_tree_parameter_count(params)


def init_physical_hubbard_tree(
    key: jax.Array,
    config: PhysicalHubbardTreeConfig,
) -> dict[str, object]:
    """Initialize the packed physical-Fock exact-SU(2) Hubbard tree."""

    return init_active_hubbard_cg(key, config)


def physical_hubbard_tree_amplitude(
    params: dict[str, object],
    tokens: jax.Array,
    config: PhysicalHubbardTreeConfig,
) -> jax.Array:
    """Evaluate one physical Hubbard Fock-token configuration."""

    return active_hubbard_cg_amplitude(params, tokens, config)


def physical_hubbard_tree_amplitudes(
    params: dict[str, object],
    tokens: jax.Array,
    config: PhysicalHubbardTreeConfig,
) -> jax.Array:
    """Evaluate a batch with the packed physical Hubbard SU(2) tree."""

    return physical_tree_amplitudes(
        params,
        tokens,
        config,
        amplitude_fn=physical_hubbard_tree_amplitude,
    )


def physical_hubbard_tree_parameter_count(params: dict[str, object]) -> int:
    """Count trainable scalars in a physical Hubbard SU(2) tree."""

    return hubbard_cg_parameter_count(params)


def init_matched_hubbard_abelian_tree(
    key: jax.Array,
    config: MatchedHubbardAbelianTreeConfig,
) -> dict[str, object]:
    """Initialize the same-tree fixed-``(Ne,Sz)`` Hubbard control."""

    return init_hubbard_abelian_tree(key, config)


def matched_hubbard_abelian_tree_amplitude(
    params: dict[str, object],
    tokens: jax.Array,
    config: MatchedHubbardAbelianTreeConfig,
) -> jax.Array:
    """Evaluate one physical token configuration with the Abelian control."""

    return hubbard_abelian_tree_amplitude(params, tokens, config)


def matched_hubbard_abelian_tree_amplitudes(
    params: dict[str, object],
    tokens: jax.Array,
    config: MatchedHubbardAbelianTreeConfig,
) -> jax.Array:
    """Evaluate a batch with the same-tree Abelian Hubbard control."""

    return physical_tree_amplitudes(
        params,
        tokens,
        config,
        amplitude_fn=matched_hubbard_abelian_tree_amplitude,
    )


def matched_hubbard_abelian_tree_parameter_count(
    params: dict[str, object],
) -> int:
    """Count trainable scalars in the same-tree Abelian control."""

    return hubbard_abelian_parameter_count(params)


__all__ = [
    "MatchedHubbardAbelianTreeConfig",
    "PhysicalHubbardTreeConfig",
    "PhysicalSpinTreeConfig",
    "PhysicalTreeAmplitude",
    "PhysicalTreeConfig",
    "init_matched_hubbard_abelian_tree",
    "init_physical_hubbard_tree",
    "init_physical_spin_tree",
    "matched_hubbard_abelian_tree_amplitude",
    "matched_hubbard_abelian_tree_amplitudes",
    "matched_hubbard_abelian_tree_parameter_count",
    "physical_hubbard_tree_amplitude",
    "physical_hubbard_tree_amplitudes",
    "physical_hubbard_tree_parameter_count",
    "physical_spin_tree_amplitude",
    "physical_spin_tree_amplitudes",
    "physical_spin_tree_parameter_count",
    "physical_tree_amplitudes",
]
