"""Shared reduced-label attention for overlapping local scalar carrier gates.

Each gate reads a complete spectator ``(q,a)`` label, where ``q=2j`` is an
integer and the magnetic index is absent.  One shared two-token attention
block returns the ten REAL symmetric active multiplicity-generator
coordinates used by ``conditional_carrier_resonance``.  The two tokens are
an active query and a spectator label; a one-key softmax would incorrectly
make its query/key parameters ineffective.

An immutable reference output, evaluated ONCE at initialization, anchors
G(theta)=f(theta)-f(theta_initial).  Thus every local filter initially is the
identity without zeroing the head or suppressing backbone derivatives.
This module does not insert spatial projectors, evaluate Hamiltonians, or
claim that two finite-context attention gates are a scalable full ViT.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .conditional_carrier_resonance import ACTIVE_DIAGONAL_COORDINATES


Restriction = Literal["full", "diagonal_conditioned", "factorized"]
SPECTATOR_LABELS = ((0, 0), (0, 1), (2, 0), (2, 1), (2, 2), (4, 0))


@dataclass(frozen=True)
class ResonanceAttentionConfig:
    """Small real attention dimensions; quantum labels are not model widths."""

    width: int = 8
    hidden_width: int = 16

    def __post_init__(self) -> None:
        for name, value, minimum in (("width", self.width, 2),
                                      ("hidden_width", self.hidden_width, 1)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


def _shapes(config: ResonanceAttentionConfig) -> dict[str, tuple[int, ...]]:
    d, h = config.width, config.hidden_width
    return dict(q_embedding=(3, d), a_embedding=(3, d), query_token=(d,),
                position_kernel=(2, d), query_kernel=(d, d), key_kernel=(d, d),
                value_kernel=(d, d), output_kernel=(d, d),
                norm1_scale=(d,), norm1_bias=(d,), norm2_scale=(d,), norm2_bias=(d,),
                hidden_kernel=(d, h), hidden_bias=(h,), residual_kernel=(h, d),
                residual_bias=(d,), generator_kernel=(d, 10), generator_bias=(10,))


def shared_resonance_parameter_count(config: ResonanceAttentionConfig = ResonanceAttentionConfig()) -> int:
    """Count shared parameters once, independently of the number of gates."""
    return sum(int(np.prod(shape)) for shape in _shapes(config).values())


def init_shared_resonance_attention(
    key: jax.Array, config: ResonanceAttentionConfig = ResonanceAttentionConfig(),
) -> dict[str, jax.Array]:
    """Initialize nonzero backbone AND head; anchoring provides identity."""
    shapes = _shapes(config)
    keys = jax.random.split(key, len(shapes))
    values = {}
    for (name, shape), rng in zip(shapes.items(), keys, strict=True):
        if name.endswith("_scale"):
            value = jnp.ones(shape, dtype=jnp.float64)
        elif name.endswith("_bias"):
            value = jnp.zeros(shape, dtype=jnp.float64)
        else:
            scale = .35 / np.sqrt(shape[0]) if name.endswith("_kernel") else .2
            value = scale * jax.random.normal(rng, shape, dtype=jnp.float64)
        values[name] = value
    return values


def _checked_parameters(parameters: Mapping[str, jax.Array], config: ResonanceAttentionConfig) -> dict[str, jax.Array]:
    shapes = _shapes(config)
    if set(parameters) != set(shapes):
        raise ValueError("parameter keys do not match the shared attention configuration")
    output = {}
    for name, shape in shapes.items():
        value = jnp.asarray(parameters[name])
        if value.shape != shape or jnp.iscomplexobj(value):
            raise ValueError(f"{name} must be real with shape {shape}")
        if not isinstance(value, jax.core.Tracer) and not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"{name} must be finite")
        output[name] = value.astype(jnp.float64)
    return output


def _checked_displacements(displacements: jax.Array) -> jax.Array:
    offsets = jnp.asarray(displacements)
    if offsets.ndim != 2 or offsets.shape[0] < 1 or offsets.shape[1] != 2 or jnp.iscomplexobj(offsets):
        raise ValueError("displacements must be a nonempty real (gate_count, 2) array")
    if not isinstance(offsets, jax.core.Tracer) and not np.isfinite(np.asarray(offsets)).all():
        raise ValueError("displacements must be finite")
    return offsets.astype(jnp.float64)


def _layer_norm(tokens: jax.Array, scale: jax.Array, bias: jax.Array) -> jax.Array:
    centered = tokens - jnp.mean(tokens, axis=-1, keepdims=True)
    variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(variance + 1e-8) * scale + bias


def resonance_attention_raw(
    parameters: Mapping[str, jax.Array], displacements: jax.Array,
    config: ResonanceAttentionConfig = ResonanceAttentionConfig(),
) -> jax.Array:
    """Return ``[gate, spectator(q,a), generator]`` with shared parameters.

    The six spectator labels occur in ``SPECTATOR_LABELS`` order. Positions
    encode the spectator-to-active displacement in fixed lattice coordinates.
    Equal displacements produce exactly equal gate tables; there are no gate
    identifiers or separately learned gate embeddings. Full spatial A1 is
    enforced outside this module, after composing the physical operators.
    """
    p = _checked_parameters(parameters, config)
    offsets = _checked_displacements(displacements)
    q_index = jnp.asarray([q // 2 for q, _ in SPECTATOR_LABELS])
    a_index = jnp.asarray([a for _, a in SPECTATOR_LABELS])
    spectator = (p["q_embedding"][q_index] + p["a_embedding"][a_index])[None]
    spectator = spectator + (offsets @ p["position_kernel"])[:, None]
    query = jnp.broadcast_to(p["query_token"], spectator.shape)
    tokens = jnp.stack((query, spectator), axis=-2)
    normalized = _layer_norm(tokens, p["norm1_scale"], p["norm1_bias"])
    q = normalized @ p["query_kernel"]
    k = normalized @ p["key_kernel"]
    v = normalized @ p["value_kernel"]
    scores = jnp.einsum("...td,...sd->...ts", q, k) / np.sqrt(config.width)
    mixed = jax.nn.softmax(scores, axis=-1) @ v
    tokens = tokens + mixed @ p["output_kernel"]
    normalized = _layer_norm(tokens, p["norm2_scale"], p["norm2_bias"])
    hidden = jax.nn.gelu(normalized @ p["hidden_kernel"] + p["hidden_bias"])
    tokens = tokens + hidden @ p["residual_kernel"] + p["residual_bias"]
    return tokens[..., 0, :] @ p["generator_kernel"] + p["generator_bias"]


def make_resonance_attention_reference(
    parameters: Mapping[str, jax.Array], displacements: jax.Array,
    config: ResonanceAttentionConfig = ResonanceAttentionConfig(),
) -> np.ndarray:
    """Evaluate and freeze the anchor ONCE, outside any gradient transform.

    Persist this array in checkpoints. Recomputing it at the current trainable
    parameters would make the forward generator identically zero; dynamically
    stop-gradienting that recomputation would create an incorrect derivative.
    """
    reference = np.asarray(resonance_attention_raw(parameters, displacements, config)).copy()
    reference.setflags(write=False)
    return reference


def restrict_resonance_generators(generators: jax.Array, restriction: Restriction = "full") -> jax.Array:
    """Apply exactly nested linear masks to already anchored generators.

    ``diagonal_conditioned`` retains all spectator-dependent diagonals and
    averages off-diagonal generator coordinates over the six labels.
    ``factorized`` retains one shared generator plus a spectator-dependent
    scalar identity. These are generator restrictions: exponentiation need
    not leave the resulting filter's off-diagonal entries context-independent.
    """
    values = jnp.asarray(generators)
    if values.ndim != 3 or values.shape[1:] != (6, 10) or jnp.iscomplexobj(values):
        raise ValueError("generators must be real with shape (gate_count, 6, 10)")
    if restriction not in ("full", "diagonal_conditioned", "factorized"):
        raise ValueError("unknown resonance-generator restriction")
    if restriction == "full":
        return values
    diagonals = jnp.asarray(ACTIVE_DIAGONAL_COORDINATES)
    shared = jnp.mean(values, axis=1, keepdims=True)
    if restriction == "diagonal_conditioned":
        return jnp.broadcast_to(shared, values.shape).at[:, :, diagonals].set(values[:, :, diagonals])
    scalar = jnp.mean((values - shared)[:, :, diagonals], axis=-1, keepdims=True)
    identity = jnp.zeros(10, dtype=values.dtype).at[diagonals].set(1.)
    return shared + scalar * identity[None, None]


def shared_resonance_generators(
    parameters: Mapping[str, jax.Array], reference_outputs: jax.Array,
    displacements: jax.Array,
    config: ResonanceAttentionConfig = ResonanceAttentionConfig(),
    *, restriction: Restriction = "full",
) -> jax.Array:
    """Shared anchored generators; restrict AFTER subtracting the anchor."""
    raw = resonance_attention_raw(parameters, displacements, config)
    reference = jnp.asarray(reference_outputs)
    if reference.shape != raw.shape or jnp.iscomplexobj(reference):
        raise ValueError("reference outputs must match the real raw-generator table")
    if not isinstance(reference, jax.core.Tracer) and not np.isfinite(np.asarray(reference)).all():
        raise ValueError("reference outputs must be finite")
    return restrict_resonance_generators(raw - jax.lax.stop_gradient(reference), restriction)


__all__ = ["ResonanceAttentionConfig", "Restriction", "SPECTATOR_LABELS",
           "shared_resonance_parameter_count", "init_shared_resonance_attention",
           "resonance_attention_raw", "make_resonance_attention_reference",
           "restrict_resonance_generators", "shared_resonance_generators"]
