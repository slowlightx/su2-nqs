"""Exact-small Marshall-sign controls for bipartite spin configurations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bipartite_marshall_signs(
    configurations: np.ndarray,
    sublattice_a: Sequence[int] | None = None,
) -> np.ndarray:
    """Return ``(-1)**N_down(A)`` using the repository bit convention.

    Bit ``0`` is down, bit ``1`` is up, and columns follow physical site order
    with site zero corresponding to the most-significant computational bit.
    If ``sublattice_a`` is omitted, the even sites ``0,2,...`` are used.
    The returned gauge has full support and imposes no charge or spin sector.
    """

    values = np.asarray(configurations)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("configurations must have nonempty shape (states, sites)")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("spin configurations must contain integer bits")
    values = np.asarray(values, dtype=np.int32)
    if np.any((values != 0) & (values != 1)):
        raise ValueError("spin configurations must contain only bits zero and one")
    n_sites = values.shape[1]
    sites = (
        tuple(range(0, n_sites, 2))
        if sublattice_a is None
        else tuple(int(site) for site in sublattice_a)
    )
    if not sites or len(set(sites)) != len(sites):
        raise ValueError("sublattice_a must contain unique sites")
    if any(site < 0 or site >= n_sites for site in sites):
        raise ValueError("a sublattice site lies outside the configuration")
    down_on_a = np.sum(values[:, sites] == 0, axis=1)
    return np.where(down_on_a % 2 == 0, 1.0, -1.0).astype(np.float64)


def diagonal_gauge_transform(
    operator: np.ndarray,
    gauge: np.ndarray,
) -> np.ndarray:
    """Return ``D^dagger O D`` for a diagonal unit-modulus gauge ``D``."""

    matrix = np.asarray(operator, dtype=np.complex128)
    phases = np.asarray(gauge, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be square")
    if phases.shape != (matrix.shape[0],):
        raise ValueError("gauge must have one phase per basis state")
    if np.any(~np.isfinite(phases)) or np.max(np.abs(np.abs(phases) - 1.0)) > 1e-12:
        raise ValueError("gauge entries must have unit modulus")
    return phases.conj()[:, None] * matrix * phases[None, :]


def offdiagonal_stoquastic_audit(operator: np.ndarray) -> dict[str, float]:
    """Audit positive-real and imaginary parts after excluding the diagonal."""

    matrix = np.asarray(operator, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be square")
    offdiagonal = matrix.copy()
    np.fill_diagonal(offdiagonal, 0.0)
    return {
        "maximum_positive_real_offdiagonal": float(
            max(np.max(np.real(offdiagonal), initial=0.0), 0.0)
        ),
        "maximum_absolute_imaginary_offdiagonal": float(
            np.max(np.abs(np.imag(offdiagonal)), initial=0.0)
        ),
    }


__all__ = [
    "bipartite_marshall_signs",
    "diagonal_gauge_transform",
    "offdiagonal_stoquastic_audit",
]
