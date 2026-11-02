"""Diagonal energy/force estimator for exact spin singlet ansatzes.

For an isotropic Heisenberg Hamiltonian and wavefunctions whose values and
parameter tangents are exactly in the total-spin ``S=0`` sector,

``P0 H P0 = 3 P0 Hz P0``.

The local value of ``3 Hz`` is diagonal in the computational basis, so a VMC
force needs only the sampled center amplitudes.  This module deliberately
does not report its sample variance as the physical Hamiltonian variance:
``Var(3 Hz)`` lacks the zero-variance property and is only an estimator-cost
diagnostic.  Callers must use a full-``H`` audit for final energies and
variances.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..operators.spin_config import SpinBond
from .physical_sampled_adam_center_only import (
    PhysicalBatchCenter,
    SampledVMCGradient,
    sampled_vmc_value_and_grad,
)


def three_hz_local_energies(
    samples: Sequence[Sequence[int]] | np.ndarray | jax.Array,
    bonds: Iterable[SpinBond],
) -> jax.Array:
    """Return ``3 sum_ij J_ij Sz_i Sz_j`` for physical spin-bit samples.

    Bit ``0`` is ``m=-1/2`` and bit ``1`` is ``m=+1/2``.  The result is a
    rank-one float64 array and contains no neural-network evaluations.
    """

    array = jnp.asarray(samples, dtype=jnp.int32)
    if array.ndim != 2 or array.shape[0] < 1:
        raise ValueError("samples must be a nonempty rank-two array")
    checked = tuple(bonds)
    if not checked:
        return jnp.zeros((array.shape[0],), dtype=jnp.float64)
    n_sites = int(array.shape[1])
    pairs: list[tuple[int, int]] = []
    couplings: list[float] = []
    for first, second, coupling in checked:
        if (
            isinstance(first, bool)
            or isinstance(second, bool)
            or not isinstance(first, (int, np.integer))
            or not isinstance(second, (int, np.integer))
        ):
            raise TypeError("bond endpoints must be integers")
        first, second = int(first), int(second)
        value = float(coupling)
        if not 0 <= first < n_sites or not 0 <= second < n_sites or first == second:
            raise ValueError("bond endpoints must be distinct valid site indices")
        if not np.isfinite(value):
            raise ValueError("bond couplings must be finite")
        pairs.append((first, second))
        couplings.append(value)
    pair_array = jnp.asarray(pairs, dtype=jnp.int32)
    coupling_array = jnp.asarray(couplings, dtype=jnp.float64)
    magnetic = array.astype(jnp.float64) - 0.5
    products = magnetic[:, pair_array[:, 0]] * magnetic[:, pair_array[:, 1]]
    return 3.0 * jnp.sum(products * coupling_array[None, :], axis=1)


def sampled_singlet_diagonal_value_and_grad(
    params: object,
    batches: Sequence[object],
    *,
    center_fn: PhysicalBatchCenter,
    bonds: Iterable[SpinBond],
) -> SampledVMCGradient:
    """Estimate an exact-singlet energy and force using diagonal ``3 Hz``.

    This has the same finite-sample centering convention and return type as
    the production center-only full-Hamiltonian estimator.  It is valid only
    when the ansatz and all parameter tangents are exact singlets and the
    Hamiltonian is an isotropic sum of Heisenberg exchanges.
    """

    checked_bonds = tuple(bonds)

    def diagonal_hpsi(candidate: object, batch: object):
        centers = jnp.asarray(center_fn(candidate, batch))
        local = three_hz_local_energies(batch, checked_bonds)
        if centers.ndim != 1 or centers.shape != local.shape:
            raise ValueError("center_fn must return one amplitude per sample")
        return centers, centers * local.astype(centers.dtype)

    return sampled_vmc_value_and_grad(
        params,
        batches,
        hpsi_fn=diagonal_hpsi,
        center_fn=center_fn,
    )


__all__ = [
    "sampled_singlet_diagonal_value_and_grad",
    "three_hz_local_energies",
]
