"""Shared scalar attention conditioned coherently on multiple Schur labels.

For two disjoint environment cells A,D and one disjoint active cell B,

    K = sum_(z,w) P_z^A tensor U_B(z,w) tensor P_w^D,

where P_(q,a) sums every magnetic component of the complete multiplet and
U_B(z,w)=direct_sum_q [exp(G_q(z,w)) tensor I_(q+1)].  Each factor commutes
with its own cell's spin generators, hence [K,S_total^alpha]=0. Completeness
of the environment projectors and G=0 imply K=I. The learned coefficients
depend on doubled q, multiplicity a, and relative positions, never on m or
computational spin bits. Spatial projection remains the caller's job.

The frozen initialization output anchors identity without suppressing neural
gradients. Multiple environments enter a genuinely joint attention softmax.
The complete two-environment table still has only360 real coordinates: this
is an exact local mechanism, not a claim of long-range entanglement or a
scalable full configuration-dependent ViT. Enumerating E environments costs
6**E; the label-query API does not silently truncate the complete operator.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .conditional_carrier_resonance import conditional_filter_coefficients, local_scalar_components
from .shared_resonance_attention import (
    ResonanceAttentionConfig, SPECTATOR_LABELS, _checked_parameters, _layer_norm,
    init_shared_resonance_attention, shared_resonance_parameter_count,
)

ContextControl = Literal["full", "one_environment", "additive"]


@dataclass(frozen=True)
class MulticontextResonanceAttentionConfig:
    """One shared attention block; environments are not separately parameterized."""

    num_environments: int = 2
    width: int = 8
    hidden_width: int = 16
    max_table_contexts: int = 1296

    def __post_init__(self) -> None:
        ResonanceAttentionConfig(self.width, self.hidden_width)
        if isinstance(self.num_environments, bool) or not isinstance(self.num_environments, Integral) or self.num_environments < 2:
            raise ValueError("at least two integer-count environments are required")
        if isinstance(self.max_table_contexts, bool) or not isinstance(self.max_table_contexts, Integral) or self.max_table_contexts < 1:
            raise ValueError("max_table_contexts must be a positive integer")


def _shared_config(config: MulticontextResonanceAttentionConfig) -> ResonanceAttentionConfig:
    return ResonanceAttentionConfig(config.width, config.hidden_width)


def multicontext_parameter_count(config: MulticontextResonanceAttentionConfig = MulticontextResonanceAttentionConfig()) -> int:
    """Shared neural count; independent of the number of environments."""
    return shared_resonance_parameter_count(_shared_config(config))


def init_multicontext_resonance_attention(
    key: jax.Array, config: MulticontextResonanceAttentionConfig = MulticontextResonanceAttentionConfig(),
) -> dict[str, jax.Array]:
    """Reuse the nonzero shared query/key/value/backbone/head initialization."""
    return init_shared_resonance_attention(key, _shared_config(config))


def _positions(positions: jax.Array, config: MulticontextResonanceAttentionConfig) -> jax.Array:
    value = jnp.asarray(positions)
    if value.shape != (config.num_environments, 2) or jnp.iscomplexobj(value):
        raise ValueError("relative_positions must be real with shape (num_environments,2)")
    if not isinstance(value, jax.core.Tracer) and not np.isfinite(np.asarray(value)).all():
        raise ValueError("relative_positions must be finite")
    return value.astype(jnp.float64)


def permute_environment_positions(positions: jax.Array, permutation: Sequence[int]) -> jax.Array:
    """Reassign positions while keeping physical projector labels fixed.

    Build a separate initialization anchor for this control. Simultaneously
    permuting labels AND positions is only token-order covariance, not a
    physically different control. Equal positions also make this control
    trivial; callers must report that degeneracy instead of claiming a win.
    """
    values = jnp.asarray(positions)
    order = tuple(permutation)
    if values.ndim != 2 or values.shape[1] != 2 or any(isinstance(i, bool) or not isinstance(i, Integral) for i in order):
        raise ValueError("positions and an integer environment permutation are required")
    if sorted(order) != list(range(len(values))):
        raise ValueError("permutation must contain every environment exactly once")
    return values[jnp.asarray(order)]


def multicontext_attention_for_labels(
    parameters: Mapping[str, jax.Array], labels: jax.Array, relative_positions: jax.Array,
    config: MulticontextResonanceAttentionConfig = MulticontextResonanceAttentionConfig(),
) -> jax.Array:
    """Evaluate B explicit contexts ``labels[B,E,2]=(doubled q,a)``.

    This query API does not assume a measured Schur label. A physical operator
    must still coherently sum the complete projector resolution. Invalid
    concrete labels raise; invalid dynamic JIT labels return NaN fail-closed.
    No magnetic-index argument is accepted.
    """
    p = _checked_parameters(parameters, _shared_config(config))
    positions = _positions(relative_positions, config)
    lab = jnp.asarray(labels)
    if lab.ndim != 3 or lab.shape[0] < 1 or lab.shape[1:] != (config.num_environments, 2) or not jnp.issubdtype(lab.dtype, jnp.integer):
        raise ValueError("labels must be integer doubled(q,a) with shape (B,E,2)")
    valid_pair = jnp.any(jnp.all(lab[..., None, :] == jnp.asarray(SPECTATOR_LABELS), axis=-1), axis=-1)
    if not isinstance(lab, jax.core.Tracer) and not np.asarray(valid_pair).all():
        raise ValueError("unreachable four-spin Schur (q,a) label")
    q_index = jnp.clip(lab[..., 0] // 2, 0, 2)
    a_index = jnp.clip(lab[..., 1], 0, 2)
    contexts = p["q_embedding"][q_index] + p["a_embedding"][a_index]
    contexts = contexts + (positions @ p["position_kernel"])[None]
    query = jnp.broadcast_to(p["query_token"], (lab.shape[0], 1, config.width))
    tokens = jnp.concatenate((query, contexts), axis=1)
    normalized = _layer_norm(tokens, p["norm1_scale"], p["norm1_bias"])
    q, k, v = (normalized @ p[name] for name in ("query_kernel", "key_kernel", "value_kernel"))
    scores = jnp.einsum("btd,bsd->bts", q, k) / np.sqrt(config.width)
    tokens = tokens + (jax.nn.softmax(scores, axis=-1) @ v) @ p["output_kernel"]
    normalized = _layer_norm(tokens, p["norm2_scale"], p["norm2_bias"])
    hidden = jax.nn.gelu(normalized @ p["hidden_kernel"] + p["hidden_bias"])
    tokens = tokens + hidden @ p["residual_kernel"] + p["residual_bias"]
    output = tokens[:, 0] @ p["generator_kernel"] + p["generator_bias"]
    return jnp.where(jnp.all(valid_pair, axis=1)[:, None], output, jnp.nan)


def complete_environment_labels(config: MulticontextResonanceAttentionConfig = MulticontextResonanceAttentionConfig()) -> np.ndarray:
    """Complete lexicographic environment tuples; never truncate a table."""
    count = 6 ** config.num_environments
    if count > config.max_table_contexts:
        raise ValueError("complete context table exceeds its explicit budget; use label queries without claiming a truncated operator")
    choices = np.indices((6,) * config.num_environments).reshape(config.num_environments, -1).T
    return np.asarray(SPECTATOR_LABELS, dtype=np.int32)[choices]


def multicontext_attention_raw(
    parameters: Mapping[str, jax.Array], relative_positions: jax.Array,
    config: MulticontextResonanceAttentionConfig = MulticontextResonanceAttentionConfig(),
) -> jax.Array:
    """Return the complete real table ``[6,...,6,10]`` (two contexts:360)."""
    values = multicontext_attention_for_labels(parameters, complete_environment_labels(config), relative_positions, config)
    return values.reshape((6,) * config.num_environments + (10,))


def make_multicontext_reference(
    parameters: Mapping[str, jax.Array], relative_positions: jax.Array,
    config: MulticontextResonanceAttentionConfig = MulticontextResonanceAttentionConfig(),
) -> np.ndarray:
    """Freeze f(theta_initial) once; persist it with positions in checkpoints."""
    value = np.asarray(multicontext_attention_raw(parameters, relative_positions, config)).copy()
    value.setflags(write=False)
    return value


def free_multicontext_generators(
    parameters: jax.Array, config: MulticontextResonanceAttentionConfig = MulticontextResonanceAttentionConfig(),
) -> jax.Array:
    """Unrestricted reduced generator table (flat360 or[6,6,10] for E=2).

    This is an expressive upper-bound/control family, not an equal-parameter
    neural baseline. Zero parameters give exact identity without a reference.
    """
    shape = (6,) * config.num_environments + (10,)
    values = jnp.asarray(parameters)
    if values.shape == (int(np.prod(shape)),):
        values = values.reshape(shape)
    if values.shape != shape or jnp.iscomplexobj(values):
        raise ValueError("free generator coordinates must have the complete real table shape")
    if not isinstance(values, jax.core.Tracer) and not np.isfinite(np.asarray(values)).all():
        raise ValueError("generator coordinates must be finite")
    return values.astype(jnp.float64)


def restrict_multicontext_generators(
    generators: jax.Array, config: MulticontextResonanceAttentionConfig = MulticontextResonanceAttentionConfig(),
    *, control: ContextControl = "full", environment_index: int = 0,
) -> jax.Array:
    """Generator-table controls, applied after subtracting the frozen anchor.

    The additive control is the sum of ALL single-environment marginal tables,
    with duplicate global means removed. It is stronger than selecting just
    one environment. These linear table subspaces are nested; this alone does
    not prove global containment of finite-width neural hypothesis families.
    """
    value = free_multicontext_generators(generators, config)
    if control not in ("full", "one_environment", "additive"):
        raise ValueError("unknown multicontext control")
    if isinstance(environment_index, bool) or not isinstance(environment_index, Integral) or not 0 <= environment_index < config.num_environments:
        raise ValueError("environment_index is out of range")
    if control == "full":
        return value
    def marginal(index):
        axes = tuple(a for a in range(config.num_environments) if a != index)
        return jnp.broadcast_to(jnp.mean(value, axis=axes, keepdims=True), value.shape)
    if control == "one_environment":
        return marginal(environment_index)
    mean = jnp.mean(value, axis=tuple(range(config.num_environments)), keepdims=True)
    return sum(marginal(i) for i in range(config.num_environments)) - (config.num_environments - 1) * mean


def interaction_contrast(
    generators: jax.Array, config: MulticontextResonanceAttentionConfig = MulticontextResonanceAttentionConfig(),
) -> jax.Array:
    """Nonadditive generator component; for E=2 it is double centering.

    A nonzero contrast demonstrates joint conditioning of generator entries,
    not an energy improvement, quantum entanglement measure, or proof that
    nonlinear filters from additive generators have no joint response.
    """
    value = free_multicontext_generators(generators, config)
    return value - restrict_multicontext_generators(value, config, control="additive")


def multicontext_resonance_generators(
    parameters: Mapping[str, jax.Array], reference_outputs: jax.Array, relative_positions: jax.Array,
    config: MulticontextResonanceAttentionConfig = MulticontextResonanceAttentionConfig(),
    *, control: ContextControl = "full", environment_index: int = 0,
) -> jax.Array:
    """Shared attention with fixed identity anchor and explicit strong controls."""
    raw = multicontext_attention_raw(parameters, relative_positions, config)
    reference = free_multicontext_generators(reference_outputs, config)
    return restrict_multicontext_generators(raw - jax.lax.stop_gradient(reference), config,
                                           control=control, environment_index=environment_index)


def multicontext_active_filters(
    generators: jax.Array, config: MulticontextResonanceAttentionConfig = MulticontextResonanceAttentionConfig(),
) -> jax.Array:
    """Exact local active16x16 scalar matrices; no many-site contraction.

    Reuse the validated reduced exponentials and CG convention. K on all
    cells is assembled coherently with COMPLETE environment projectors, not
    with measured or magnetic-index-dependent classical gates.
    """
    value = free_multicontext_generators(generators, config)
    grouped = value.reshape(-1, 6, 10)
    coefficients = jax.vmap(conditional_filter_coefficients)(grouped).reshape(value.shape)
    return jnp.eye(16, dtype=jnp.float64) + jnp.einsum(
        "...r,rij->...ij", coefficients, jnp.asarray(local_scalar_components().active_basis))


__all__ = ["MulticontextResonanceAttentionConfig", "ContextControl", "multicontext_parameter_count",
           "init_multicontext_resonance_attention", "permute_environment_positions",
           "multicontext_attention_for_labels", "complete_environment_labels", "multicontext_attention_raw",
           "make_multicontext_reference", "free_multicontext_generators", "restrict_multicontext_generators",
           "interaction_contrast", "multicontext_resonance_generators", "multicontext_active_filters"]
