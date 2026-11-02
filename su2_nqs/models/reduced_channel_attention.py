"""Attention coefficients for scalar operators conditioned on spectator q.

The model has no Hamiltonian or magnetic indices. The operator layer must
interpret its output through SU(2)-scalar spectral projectors, rather than
multiply computational spin amplitudes by an arbitrary diagonal function.
This minimal model conditions on q only; it is not a full multiplicity-(q,a)
autoregressive wavefunction or a scalable spatial projector.
"""

import jax
import jax.numpy as jnp


def init_channel_attention(key: jax.Array, width: int = 8) -> dict:
    """Initialize a single-block real attention network and zero residual."""
    if width < 2:
        raise ValueError("width must be at least two")
    keys = iter(jax.random.split(key, 12))

    def normal(shape: tuple[int, ...]) -> jax.Array:
        return jax.random.normal(next(keys), shape, dtype=jnp.float64) / jnp.sqrt(width)

    return {
        "local": jnp.zeros(4, dtype=jnp.float64),
        "attention": {
            "q_embedding": normal((4, width)),  # q=0,2,4 plus active-site sentinel
            "mode_embedding": normal((4, width)),
            "role_embedding": normal((2, width)),
            "position": normal((2, width)),
            "query": normal((width, width)), "key": normal((width, width)),
            "value": normal((width, width)), "output": normal((width, width)),
            "mlp_in": normal((width, 2 * width)), "mlp_out": normal((2 * width, width)),
            "head": jnp.zeros(width, dtype=jnp.float64),
        },
    }


def channel_attention_coefficients(params: dict, contexts: jax.Array, coordinates: jax.Array) -> jax.Array:
    """Return coefficients for integer rows ``(mode,q_C,q_D)``.

    Cells A,B are active; C,D are spectators. ``q_C,q_D`` are doubled labels
    0,2,4, checked by the operator/context builder before JIT compilation.
    The zero head ensures exactly the same initial physical carrier for all
    seeds and schedules while hidden attention weights remain trainable.
    """
    p = params["attention"]
    mode = contexts[:, 0]
    q = jnp.stack((jnp.full_like(mode, 3), jnp.full_like(mode, 3), contexts[:, 1] // 2, contexts[:, 2] // 2), axis=1)
    roles = jnp.asarray((0, 0, 1, 1))
    x = p["q_embedding"][q] + p["role_embedding"][roles][None, :, :]
    x = x + (coordinates @ p["position"])[None, :, :] + p["mode_embedding"][mode][:, None, :]

    def norm(y: jax.Array) -> jax.Array:
        return (y - y.mean(axis=-1, keepdims=True)) / jnp.sqrt(y.var(axis=-1, keepdims=True) + 1e-8)

    y = norm(x)
    query, key, value = y @ p["query"], y @ p["key"], y @ p["value"]
    weights = jax.nn.softmax(jnp.einsum("btd,bsd->bts", query, key) / jnp.sqrt(x.shape[-1]), axis=-1)
    x = x + (weights @ value) @ p["output"]
    x = x + jax.nn.gelu(norm(x) @ p["mlp_in"]) @ p["mlp_out"]
    residual = norm(x).mean(axis=1) @ p["head"]
    return params["local"][mode] + residual
