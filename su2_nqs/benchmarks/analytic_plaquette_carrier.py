"""Analytic IID samplers for the aligned four-spin ``A1`` carrier."""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral

import numpy as np

from ..su2.fusion_tree import FusionTree
from ..su2.many_cell_schur import ManyCellSchurLayout
from ..su2.recoupling import tree_basis_matrix
from .local_multiplicity_carriers import square_plaquette_singlet_carrier


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@lru_cache(maxsize=1)
def aligned_plaquette_a1_local_data() -> tuple[np.ndarray, np.ndarray]:
    """Return physical amplitudes and positive balanced-Schur coefficients."""

    physical = np.asarray(
        square_plaquette_singlet_carrier().vectors[:, 0], dtype=np.complex128
    )
    balanced, states = tree_basis_matrix(
        FusionTree.balanced(4), target_q=0, target_m=0
    )
    if tuple(state.labels for state in states) != ((0, 0, 0), (2, 2, 0)):
        raise RuntimeError("four-spin balanced singlet convention changed")
    coefficients = np.asarray(balanced.T.conj() @ physical)
    if np.max(np.abs(coefficients.imag), initial=0.0) > 2.0e-14 or np.any(
        coefficients.real <= 0.0
    ):
        raise RuntimeError("local A1 coefficients lost their positive convention")
    if abs(float(np.vdot(coefficients, coefficients).real) - 1.0) > 2.0e-13:
        raise RuntimeError("local A1 Schur coefficients are not normalized")
    physical.setflags(write=False)
    positive = np.asarray(coefficients.real, dtype=np.float64)
    positive.setflags(write=False)
    return physical, positive


def sample_aligned_plaquette_a1_physical(
    layout: ManyCellSchurLayout,
    *,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample physical bits and exact phases from the plaquette product."""

    count = _positive_integer(sample_count, "sample_count")
    physical, _ = aligned_plaquette_a1_local_data()
    probabilities = np.abs(physical) ** 2
    active = np.flatnonzero(probabilities > 0.0)
    active_p = probabilities[active] / np.sum(probabilities[active])
    rng = np.random.default_rng(int(seed))
    choices = rng.choice(
        active, size=(count, layout.n_cells), replace=True, p=active_p
    )
    configurations = np.zeros((count, layout.n_sites), dtype=np.int32)
    phases = np.zeros((count,), dtype=np.float64)
    local_bits = (
        (choices[..., None] >> np.arange(3, -1, -1, dtype=np.int64)) & 1
    ).astype(np.int32)
    for cell_index, cell in enumerate(layout.physical_cells):
        configurations[:, np.asarray(cell, dtype=np.int64)] = local_bits[
            :, cell_index
        ]
        phases += np.angle(physical[choices[:, cell_index]])
    return configurations, phases


def sample_aligned_plaquette_a1_schur(
    layout: ManyCellSchurLayout,
    *,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample reduced labels; every cell has ``q=0`` and intercell ``q=0``."""

    count = _positive_integer(sample_count, "sample_count")
    _, coefficients = aligned_plaquette_a1_local_data()
    rng = np.random.default_rng(int(seed))
    channels = rng.choice(
        2, size=(count, layout.n_cells), replace=True, p=coefficients**2
    )
    labels = np.zeros((count, layout.n_sites - 1), dtype=np.int32)
    for cell, (left, right) in enumerate(layout.cell_pair_indices):
        pair_q = 2 * channels[:, cell]
        labels[:, left] = pair_q
        labels[:, right] = pair_q
    # Cell roots and all inter-cell nodes remain q=0 by construction.
    decoded = layout.decode(labels)
    if (
        np.any(decoded.cell_q != 0)
        or np.any(decoded.cell_a != channels)
        or np.any(decoded.intercell_q != 0)
    ):
        raise RuntimeError("sampled plaquette Schur labels failed exact decoding")
    return labels, np.zeros((count,), dtype=np.float64)


__all__ = [
    "aligned_plaquette_a1_local_data",
    "sample_aligned_plaquette_a1_physical",
    "sample_aligned_plaquette_a1_schur",
]
