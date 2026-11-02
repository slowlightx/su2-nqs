"""Small static controllers for SU(2) reduced-space capacity proposals.

The inputs are scalar features of legal, complete-multiplet actions.  These
models propose real coefficients for a caller's verified tangent dictionary;
they do not define a wavefunction, an SU(2) projector, or an autoregressive
distribution.  Magnetic labels never enter the controller.  Its context
attention is genuine query/key/value token mixing, not a renamed kernel
regressor.  Non-attention controls deliberately use the same input features.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CapacityControllerConfig:
    """Static scalar-token controller; ``width`` is neural, not SU(2) width."""

    token_dim: int
    width: int = 4
    relation_dim: int = 0
    mixing: str = "attention"

    def __post_init__(self) -> None:
        for name in ("token_dim", "width", "relation_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if self.token_dim < 1 or self.width < 1 or self.relation_dim < 0:
            raise ValueError("positive token dimension/width and nonnegative relation dimension required")
        if self.mixing not in ("linear", "bilinear", "pointwise", "pooled", "signed_graph", "attention"):
            raise ValueError("unknown capacity-controller mixing")


def init_capacity_controller(key: jax.Array, config: CapacityControllerConfig) -> dict:
    """Initialize every arm at the identical zero tangent proposal."""
    if config.mixing in ("linear", "bilinear"):
        extra = config.token_dim if config.mixing == "bilinear" else 0
        return {"readout": jnp.zeros(config.token_dim + extra, jnp.float64),
                "bias": jnp.zeros((), jnp.float64)}
    keys = iter(jax.random.split(key, 8))

    def weight(rows: int, columns: int) -> jax.Array:
        return jax.random.normal(next(keys), (rows, columns), dtype=jnp.float64)/rows**.5

    width = config.width
    params = dict(embedding=weight(config.token_dim, width),
                  local_in=weight(width, width), local_out=weight(width, width),
                  readout=jnp.zeros(width, jnp.float64), bias=jnp.zeros((), jnp.float64))
    if config.mixing == "attention":
        params.update(query=weight(width, width), key=weight(width, width),
                      value=weight(width, width), output=weight(width, width),
                      relation=jnp.zeros(config.relation_dim, jnp.float64))
    elif config.mixing in ("pooled", "signed_graph"):
        params.update(context_in=weight(width, width), context_out=weight(width, width))
    return params


def predict_capacity_coefficients(
    params: dict, tokens: jax.Array, config: CapacityControllerConfig,
    *, allowed: jax.Array | None = None, relations: jax.Array | None = None,
) -> jax.Array:
    """Propose one REAL tangent coefficient per legal complete-multiplet token.

    ``relations[i,j]`` may encode fixed J1/J2 inter-block bond counts or tree
    adjacency; Euclidean block-center distances alone can be degenerate on a
    small torus.  Relations must be fixed before examining residual targets.
    At least one token must be legal. Invalid tokens get exactly zero output
    and are excluded before context pooling and before attention softmax.
    """
    if tokens.ndim != 2 or tokens.shape[1] != config.token_dim:
        raise ValueError("tokens must have shape (actions, token_dim)")
    count = tokens.shape[0]
    if allowed is None:
        allowed = jnp.ones(count, dtype=bool)
    if allowed.shape != (count,):
        raise ValueError("one legal-action flag per token required")
    denominator = jnp.maximum(jnp.sum(allowed), 1)
    if config.mixing in ("linear", "bilinear"):
        encoded = tokens
        if config.mixing == "bilinear":
            mean = jnp.sum(jnp.where(allowed[:, None], tokens, 0), axis=0)/denominator
            encoded = jnp.concatenate((tokens, tokens*mean[None, :]), axis=-1)
        output = encoded @ params["readout"] + params["bias"]
    else:
        hidden = jnp.tanh(tokens @ params["embedding"])
        hidden = hidden + jnp.tanh(hidden @ params["local_in"]) @ params["local_out"]
        if config.mixing in ("pooled", "signed_graph"):
            if config.mixing == "pooled":
                context = jnp.sum(jnp.where(allowed[:, None], hidden, 0), axis=0)/denominator
            else:
                if (relations is None or config.relation_dim < 1
                        or relations.shape != (count, count, config.relation_dim)):
                    raise ValueError("signed graph control needs the final screening relation")
                screening = jnp.where(allowed[None, :], relations[..., -1], 0.)
                screening = screening/jnp.maximum(jnp.sum(jnp.abs(screening), axis=-1, keepdims=True), 1e-12)
                context = screening @ hidden
            hidden = hidden + jnp.tanh((hidden+context) @ params["context_in"]) @ params["context_out"]
        elif config.mixing == "attention":
            query, key, value = (hidden @ params[name] for name in ("query", "key", "value"))
            logits = query @ key.T/config.width**.5
            if config.relation_dim:
                if relations is None or relations.shape != (count, count, config.relation_dim):
                    raise ValueError("pair relations need shape (actions, actions, relation_dim)")
                logits = logits + relations @ params["relation"]
            logits = jnp.where(allowed[None, :], logits, -jnp.inf)
            mixed = jax.nn.softmax(logits, axis=-1) @ value
            hidden = hidden + jnp.tanh(mixed @ params["output"])
        output = hidden @ params["readout"] + params["bias"]
    return jnp.where(allowed, output, 0.)


def capacity_quadratic_loss(
    coefficients: jax.Array, physical_gram: jax.Array, residual_force: jax.Array,
    *, ridge: float = .01,
) -> jax.Array:
    """Fit a physical tangent quadratic, not coordinate-MSE teacher labels.

    For real coefficients d and centered log-amplitude tangents O,
    S=Re<Cov(O,O)> and g=Re<Cov(O,E_loc)>; the linear energy slope is 2 g.d.
    This objective is d.S.d+2g.d with a diagonal physical-metric regularizer.
    It is a *tangent surrogate*, not a finite-step energy or variance.
    """
    if physical_gram.shape != (coefficients.size, coefficients.size) or residual_force.shape != coefficients.shape:
        raise ValueError("coefficient, physical metric and force shapes differ")
    diagonal = jnp.maximum(jnp.diag(physical_gram), 0.)
    return (coefficients @ physical_gram @ coefficients
            + 2*residual_force @ coefficients
            + ridge*jnp.sum(diagonal*coefficients**2))
