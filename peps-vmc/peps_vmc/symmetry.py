"""Small SU(2)-equivariant PEPS tensors with virtual ``V_0 + V_1/2``.

All Hilbert labels below are doubled integers.  A local tensor is a map
``A[s, v0, ...] : virtual legs -> physical V_1/2``.  Its nonzero entries
obey ``m_phys == sum(m_virtual)``.  Virtual maps are joined by *oriented*
singlet bonds, not by Kronecker deltas.  Consequently a closed network is
an exact physical singlet before any approximate contraction is applied.

The CG data are prepared with the repository's Condon--Shortley convention.
No symbolic operations, CG evaluation, or Python label enumeration happens
inside the optional JAX parameter-to-tensor runtime function.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from numbers import Integral
from typing import Any, Sequence

import numpy as np

from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.quantum_numbers import coupling_range


VIRTUAL_LABELS: tuple[tuple[int, int], ...] = ((0, 0), (1, -1), (1, 1))
"""Virtual indices ``0, 1, 2`` correspond to ``(q,m)=(0,0),(1,-1),(1,1)``."""

PHYSICAL_M: tuple[int, int] = (-1, 1)
"""Physical bit zero is down and bit one is up, as in ``su2_nqs``."""


@dataclass(frozen=True)
class FusionChannel:
    """One multiplicity channel; all entries in both tuples are doubled spins.

    ``fusion_q[k]`` is the total doubled spin after virtual legs ``0..k``.
    Its last entry is one, since the output is physical spin one half.
    ``virtual_q`` also records which virtual legs are in their singlet block.
    """

    virtual_q: tuple[int, ...]
    fusion_q: tuple[int, ...]


def _coordination(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("coordination must be an integer")
    if value < 1:
        raise ValueError("coordination must be positive")
    return int(value)


@lru_cache(maxsize=None)
def intertwiner_labels(coordination: int) -> tuple[FusionChannel, ...]:
    """Enumerate the complete deterministic reduced basis for ``z`` legs.

    Virtual sector tuples are lexicographic, followed by lexicographic legal
    left-associated fusion paths.  No truncation of multiplicity occurs.
    The dimensions for four and six virtual legs are respectively 12 and 76.
    """

    z = _coordination(coordination)
    channels: list[FusionChannel] = []
    for virtual_q in product((0, 1), repeat=z):
        if sum(virtual_q) % 2 == 0:
            continue
        paths = [(virtual_q[0],)]
        for leg_q in virtual_q[1:]:
            paths = [
                path + (out_q,)
                for path in paths
                for out_q in coupling_range(path[-1], leg_q)
            ]
        channels.extend(
            FusionChannel(virtual_q, path)
            for path in paths
            if path[-1] == 1
        )
    return tuple(channels)


@lru_cache(maxsize=None)
def local_intertwiner_basis(coordination: int) -> np.ndarray:
    """Return fixed real intertwiners of shape ``(K,2,3,...,3)``.

    For each physical doubled projection ``m=-1,+1``, the virtual vectors
    form an orthonormal coupled basis.  Thus the Frobenius Gram matrix of
    complete maps is ``2 * I_K``, not ``I_K``.  This avoids an implicit
    dimension-dependent normalization of the physical map.  The returned
    cached array is read-only.
    """

    z = _coordination(coordination)
    channels = intertwiner_labels(z)
    basis = np.zeros((len(channels), 2) + (3,) * z, dtype=np.float64)
    for channel_index, channel in enumerate(channels):
        local_indices = [(0,) if q == 0 else (1, 2) for q in channel.virtual_q]
        for indices in product(*local_indices):
            ms = tuple(VIRTUAL_LABELS[index][1] for index in indices)
            total_m = sum(ms)
            if total_m not in PHYSICAL_M:
                continue
            coefficient = 1.0
            running_m = ms[0]
            for leg in range(1, z):
                next_m = running_m + ms[leg]
                coefficient *= clebsch_gordan(
                    channel.fusion_q[leg - 1],
                    running_m,
                    channel.virtual_q[leg],
                    ms[leg],
                    channel.fusion_q[leg],
                    next_m,
                )
                running_m = next_m
                if coefficient == 0:
                    break
            basis[(channel_index, PHYSICAL_M.index(total_m)) + indices] = coefficient
    basis.setflags(write=False)
    return basis


def virtual_bond_metric() -> np.ndarray:
    """Return the normalized singlet bond in each ``q`` block.

    ``B[m,n]=(-1)^((q-m)/2) delta(n,-m)/sqrt(q+1)`` with matching ``q``.
    In particular ``B[1,2]=-1/sqrt(2)`` and ``B[2,1]=+1/sqrt(2)``.
    Every lattice edge must fix its first/second endpoint.  Reversing an
    edge is *not* innocuous when singlet and half-integer sectors coexist.
    The block normalization can be absorbed into reduced coefficients,
    but this convention is fixed explicitly for reproducible amplitudes.
    """

    bond = np.zeros((3, 3), dtype=np.float64)
    for first, (q, m) in enumerate(VIRTUAL_LABELS):
        for second, (other_q, other_m) in enumerate(VIRTUAL_LABELS):
            if q == other_q and other_m == -m:
                bond[first, second] = (-1.0) ** ((q - m) // 2) / np.sqrt(q + 1)
    return bond


def reduced_tensor(
    coefficients: Any,
    basis: Any,
    *,
    backend: str = "numpy",
) -> Any:
    """Build ``A=sum_k coefficients[k]*basis[k]`` without altering symmetry.

    ``backend='jax'`` supports differentiation of real or complex reduced
    coefficients.  Only fixed numeric ``basis`` data enter this runtime.
    A leading batch dimension is deliberately not inferred or hidden.
    """

    if backend == "numpy":
        xp = np
    elif backend == "jax":
        import jax.numpy as jnp

        xp = jnp
    else:
        raise ValueError("backend must be 'numpy' or 'jax'")
    weights = xp.asarray(coefficients)
    fixed_basis = xp.asarray(basis)
    if fixed_basis.ndim < 3 or fixed_basis.shape[1] != 2 or any(
        dimension != 3 for dimension in fixed_basis.shape[2:]
    ):
        raise ValueError("basis must have shape (K,2,3,...,3)")
    if weights.ndim != 1 or weights.shape[0] != fixed_basis.shape[0]:
        raise ValueError("coefficients must be a vector of length K")
    return xp.tensordot(weights, fixed_basis, axes=((0,), (0,)))


def rvb_coefficients(
    coordination: int,
    leg_weights: Sequence[complex] | None = None,
) -> np.ndarray:
    """Initialize the one-occupied-virtual-leg nearest-neighbor RVB tensor.

    The physical spin is copied onto exactly one half-integer virtual leg.
    After singlet-bond closure, the state is a coherent superposition of
    dimer coverings with the declared bond orientations.  This initializer
    does not activate the complete local reduced basis; additional channels
    are nevertheless present and independently trainable.
    """

    z = _coordination(coordination)
    weights = np.ones(z) if leg_weights is None else np.asarray(leg_weights)
    if weights.shape != (z,):
        raise ValueError("leg_weights must have one entry per virtual leg")
    coefficients = np.zeros(len(intertwiner_labels(z)), dtype=weights.dtype)
    for index, channel in enumerate(intertwiner_labels(z)):
        if sum(channel.virtual_q) == 1:
            coefficients[index] = weights[channel.virtual_q.index(1)]
    return coefficients


def spin_generators(q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``Sx,Sy,Sz`` for doubled spin ``q``, ordered by ascending ``m``.

    These dense matrices are reference/diagnostic helpers, not contraction
    kernels.  They use the same doubled-label convention as the CG tables.
    """

    if isinstance(q, bool) or not isinstance(q, Integral):
        raise TypeError("q must be a nonnegative integer doubled spin")
    if q < 0:
        raise ValueError("q must be nonnegative")
    ms = np.arange(-q, q + 1, 2)
    raising = np.zeros((q + 1, q + 1), dtype=np.complex128)
    for column, m in enumerate(ms[:-1]):
        raising[column + 1, column] = np.sqrt((q - m) * (q + m + 2)) / 2
    lowering = raising.conj().T
    return (
        (raising + lowering) / 2,
        (raising - lowering) / (2j),
        np.diag(ms.astype(np.complex128) / 2),
    )


def virtual_generators() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return generators on the direct sum ``V_0 + V_1/2`` virtual leg."""

    result = []
    for half in spin_generators(1):
        full = np.zeros((3, 3), dtype=np.complex128)
        full[1:, 1:] = half
        result.append(full)
    return tuple(result)  # type: ignore[return-value]
