"""Static environment-to-multiplicity compression predictors.

Inputs are physical, gauge-independent scalar observables in a fixed local
Schur frame, never the queried spin configuration.  The output is a reduced
environment matrix; its eigenspace can choose retained copies of one doubled
spin q.  This is an offline compression controller, not a normalized NQS or
a configuration-dependent wavefunction.  All magnetic copies of a retained
reduced vector must be kept by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
from ..su2.quantum_numbers import validate_q


@dataclass(frozen=True)
class EnvironmentCompressionConfig:
    """Predict a real q=1, multiplicity-two block from scalar tokens."""

    token_dim: int
    width: int = 8
    mixing: str = "attention"
    target_q: int = 1
    output_dim: int = 3

    def __post_init__(self) -> None:
        validate_q(self.target_q)
        if any(isinstance(x, bool) or not isinstance(x, Integral)
               for x in (self.token_dim, self.width, self.output_dim)) or self.token_dim < 1 or self.width < 2 or self.output_dim < 1:
            raise ValueError("positive token dimension and width >=2 required")
        if self.target_q != 1:
            raise ValueError("this small controller targets doubled q=1")
        if self.mixing not in ("attention", "pointwise", "hierarchical_attention"):
            raise ValueError("unknown token mixing")


def init_environment_compression(
    key: jax.Array, config: EnvironmentCompressionConfig,
    mean_block: jax.Array,
) -> dict:
    """Use identical parameter trees and initial outputs for the three arms."""
    keys = iter(jax.random.split(key, 8))
    width = config.width

    def weight(rows: int, columns: int) -> jax.Array:
        return jax.random.normal(next(keys), (rows, columns), dtype=jnp.float64) / rows**.5

    return dict(
        embedding=weight(config.token_dim, width),
        query=weight(width, width), key=weight(width, width),
        value=weight(width, width), output=weight(width, width),
        local_in=weight(width, 2*width), local_out=weight(2*width, width),
        readout=jnp.zeros((width, config.output_dim), jnp.float64),
        bias=jnp.asarray(mean_block, jnp.float64),
    )


def predict_environment_block(
    params: dict, tokens: jax.Array, config: EnvironmentCompressionConfig,
    group_pool: jax.Array,
) -> jax.Array:
    """Return static reduced features without magnetic inputs.

    The original output_dim=3 controller predicts
    (sigma00,sigma11,sqrt(2)*sigma01). A caller may instead request the entries
    of a larger reduced frame; the same scalar-token encoder is reused.

    ``group_pool`` has normalized rows for local-region, environment, and
    crossing tokens. Hierarchical mode first encodes complete local tokens
    pointwise, then exchanges information between these three groups.
    The pointwise arm uses every parameter but has no cross-token attention;
    it sees exactly the same input observables and has identical capacity by
    stored parameter count, not identical functional expressivity.
    """
    hidden = jnp.tanh(tokens @ params["embedding"])
    hidden = hidden + jnp.tanh(hidden @ params["local_in"]) @ params["local_out"]
    if config.mixing == "hierarchical_attention":
        hidden = group_pool @ hidden
    query, key, value = (hidden @ params[name] for name in ("query", "key", "value"))
    if config.mixing == "pointwise":
        mixed = jax.nn.sigmoid(query*key / config.width**.5)*value
    else:
        mixed = jax.nn.softmax(query @ key.T / config.width**.5, axis=-1) @ value
    hidden = hidden + jnp.tanh(mixed @ params["output"])
    return hidden.mean(axis=-2) @ params["readout"] + params["bias"]


def reduced_rank_one_projector(block_values: jax.Array) -> jax.Array:
    """Top eigenspace of a real multiplicity block, independent of eigenvector sign.

    Learned matrix positivity is not assumed.  Compression uses only its
    orthogonal eigenspace, and positivity/energy are evaluated on the actual
    physical state afterwards. Degenerate leading eigenvalues do not define
    a preferred subspace and must not be interpreted as a physical choice.
    """
    a, b, c = block_values
    matrix = jnp.array([[a, c/jnp.sqrt(2.)], [c/jnp.sqrt(2.), b]])
    _, vectors = jnp.linalg.eigh(matrix)
    return jnp.outer(vectors[:, -1], vectors[:, -1])


def reduced_frame_projector(values: jax.Array, multiplicity: int, retained: int) -> jax.Array:
    """QR-retract a static learned reduced frame to a rank-r projector.

    This operation never acts on magnetic indices.  It is appropriate for
    any fixed doubled q, provided the caller applies the identical projector
    to all q+1 magnetic copies. The raw frame must have full column rank;
    callers should monitor its smallest singular value during optimization.
    """
    if not 1 <= retained <= multiplicity:
        raise ValueError("retained multiplicity must lie in [1,multiplicity]")
    frame = values.reshape((multiplicity, retained))
    orthogonal, _ = jnp.linalg.qr(frame, mode="reduced")
    return orthogonal @ orthogonal.T


def environment_memory_weights(
    query: jax.Array, memory: jax.Array, metric: jax.Array,
    *, allowed: jax.Array | None = None,
) -> jax.Array:
    """Positive similarity weights for a static environment-memory controller.

    Inputs contain scalar physical observables in ONE fixed Schur frame,
    not virtual gauge-dependent matrix entries or sampled magnetic labels.
    ``metric`` maps scalar features to comparison coordinates. This is a
    Gaussian/Nadaraya--Watson kernel written as attention: its use alone is
    NOT a new spatial-attention architecture. ``allowed`` can exclude every
    example from a queried parent during leave-parent-out training. Every
    query must have at least one allowed memory; validate this offline.
    """
    if query.ndim != 2 or memory.ndim != 2 or metric.ndim != 2:
        raise ValueError("query, memory and metric must be matrices")
    if query.shape[1] != memory.shape[1] or metric.shape[0] != query.shape[1]:
        raise ValueError("query and memory need identical scalar-feature frames")
    delta = query[:, None, :] @ metric - memory[None, :, :] @ metric
    logits = -jnp.sum(delta*delta, axis=-1) / 2
    if allowed is not None:
        if allowed.shape != logits.shape:
            raise ValueError("one parent-exclusion flag per query/memory pair required")
        logits = jnp.where(allowed, logits, -jnp.inf)
    return jax.nn.softmax(logits, axis=-1)


def predict_environment_memory(
    query: jax.Array, memory: jax.Array, metric: jax.Array,
    reduced_densities: jax.Array, *, allowed: jax.Array | None = None,
) -> jax.Array:
    """Convex PSD reduced-density prediction, never a magnetic tensor.

    The caller supplies physical density blocks in the same complete
    multiplicity frame and doubled-q sector. A selected eigenspace must
    subsequently be applied to ALL q+1 magnetic copies. The output is a
    static compression proposal, not a normalized autoregressive NQS.
    """
    if (reduced_densities.ndim != 3
            or reduced_densities.shape[0] != memory.shape[0]
            or reduced_densities.shape[1] != reduced_densities.shape[2]):
        raise ValueError("one square reduced density per memory required")
    weights = environment_memory_weights(query, memory, metric, allowed=allowed)
    return jnp.einsum("bm,mij->bij", weights, reduced_densities)
