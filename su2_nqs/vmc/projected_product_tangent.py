"""Forward JVP of an existing node-safe projected full-H product kernel.

The connection/image plan is constant. No image/bond is removed and no local
energy ratio is differentiated. A bank tangent must be generated from the
complete parameter-to-bank map; plugging a tangent bank into a nonlinear
tensor contraction as if it were a wavefunction is generally incorrect.
"""
from __future__ import annotations

from collections.abc import Callable

import jax


def make_projected_product_jvp(evaluate: Callable) -> Callable:
    """Return JIT (bank,delta_bank,plan) -> Hphi,phi,counts,Hdphi,dphi.

    ``evaluate`` returns (Hphi,phi,unique_count,padded_count), with TOTAL H.
    Differentiate only the bank, holding the exact-key plan fixed. Forward
    AD supports the dynamic bounded loop of the original device kernel;
    this API does not promise reverse-mode gradients through that loop.
    H(delta phi) is evaluated even where phi or delta phi is zero.
    """
    @jax.jit
    def product_jvp(bank, delta_bank, prepared):
        values, tangents = jax.jvp(lambda b: evaluate(b, prepared), (bank,), (delta_bank,))
        return (*values, tangents[0], tangents[1])
    return product_jvp


__all__ = ['make_projected_product_jvp']
