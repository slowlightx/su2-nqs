"""Exact-small action of local computational-basis gates."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np


def _validated_action(
    wavefunction: np.ndarray,
    gate: np.ndarray,
    sites: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], int]:
    vector = np.asarray(wavefunction, dtype=np.complex128)
    local = np.asarray(gate, dtype=np.complex128)
    selected = tuple(int(site) for site in sites)
    if vector.ndim != 1 or vector.size < 1 or np.any(~np.isfinite(vector)):
        raise ValueError("wavefunction must be a finite nonempty vector")
    n_sites = int(round(math.log2(vector.size)))
    if 2**n_sites != vector.size:
        raise ValueError("wavefunction size must be a power of two")
    if local.shape != (16, 16) or np.any(~np.isfinite(local)):
        raise ValueError("a four-site gate must be a finite 16x16 matrix")
    if len(selected) != 4 or len(set(selected)) != 4 or any(
        not 0 <= site < n_sites for site in selected
    ):
        raise ValueError("sites must be four distinct in-range indices")
    return vector, local, selected, n_sites


def apply_local_gate(
    wavefunction: np.ndarray,
    gate: np.ndarray,
    sites: Sequence[int],
) -> np.ndarray:
    """Apply ``gate`` to ordered physical ``sites`` of a full bit-basis state.

    Tensor axis zero is physical site zero, preserving the repository's
    most-significant-bit convention.  The input is never modified.
    """

    vector, local, selected, n_sites = _validated_action(
        wavefunction, gate, sites
    )
    remainder = tuple(site for site in range(n_sites) if site not in selected)
    order = selected + remainder
    inverse = np.argsort(order)
    tensor = vector.reshape((2,) * n_sites)
    front = np.transpose(tensor, order).reshape(16, -1)
    transformed = (local @ front).reshape((2,) * n_sites)
    return np.transpose(transformed, inverse).reshape(-1)


def apply_local_gate_to_columns(
    vectors: np.ndarray,
    gate: np.ndarray,
    sites: Sequence[int],
) -> np.ndarray:
    """Apply one four-site gate to many full-basis state columns.

    ``vectors`` has shape ``(2**N, batch)``.  This is the exact-small bridge
    used to project local intertwiner generators into a fusion basis without
    constructing a ``2**N``-square operator.
    """

    values = np.asarray(vectors, dtype=np.complex128)
    if values.ndim != 2 or values.shape[1] < 1 or np.any(~np.isfinite(values)):
        raise ValueError("vectors must be a finite nonempty column batch")
    # Reuse the scalar validator for all gate/site/physical-dimension checks.
    _, local, selected, n_sites = _validated_action(values[:, 0], gate, sites)
    remainder = tuple(site for site in range(n_sites) if site not in selected)
    order = selected + remainder + (n_sites,)
    inverse = np.argsort(order)
    tensor = values.reshape((2,) * n_sites + (values.shape[1],))
    front = np.transpose(tensor, order).reshape(16, -1)
    transformed = (local @ front).reshape(
        (2,) * n_sites + (values.shape[1],)
    )
    return np.transpose(transformed, inverse).reshape(values.shape)


def apply_local_gate_layer(
    wavefunction: np.ndarray,
    gates_and_sites: Sequence[tuple[np.ndarray, Sequence[int]]],
) -> np.ndarray:
    """Apply an explicitly ordered layer of four-site gates."""

    result = np.asarray(wavefunction, dtype=np.complex128)
    for gate, sites in gates_and_sites:
        result = apply_local_gate(result, gate, sites)
    return result


__all__ = [
    "apply_local_gate",
    "apply_local_gate_layer",
    "apply_local_gate_to_columns",
]
