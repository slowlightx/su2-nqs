"""Pure-JAX carrier anchoring for complete reduced Schur manifolds.

The learned reduced-label model supplies a full configuration-dependent
residual.  The carrier is only a projective-space origin: removing the
parallel component avoids spending one parameter direction on normalization,
while the additive residual can change every physical-basis amplitude and is
not constrained to preserve carrier nodes.
"""

from __future__ import annotations

import jax.numpy as jnp


def carrier_anchored_additive_state(
    carrier: jnp.ndarray,
    residual: jnp.ndarray,
    residual_scale: jnp.ndarray | float,
) -> jnp.ndarray:
    """Return a carrier plus its orthogonal, trainable additive residual.

    ``carrier`` and ``residual`` are physical state vectors in the same exact
    symmetry sector.  The carrier is normalized internally.  A zero scale
    recovers it exactly; a nonzero scale does not inherit its nodes.
    """

    anchor = jnp.asarray(carrier)
    candidate = jnp.asarray(residual)
    if anchor.ndim != 1 or candidate.shape != anchor.shape:
        raise ValueError("carrier and residual must be equal one-dimensional arrays")
    scale = jnp.asarray(residual_scale)
    if scale.ndim != 0:
        raise ValueError("residual_scale must be scalar")
    anchor = anchor / jnp.linalg.norm(anchor)
    orthogonal = candidate - anchor * jnp.vdot(anchor, candidate)
    return anchor + scale * orthogonal


__all__ = ["carrier_anchored_additive_state"]
