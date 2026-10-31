"""Canonical-environment adapter for the existing finite Schur decoder.

Only its RAW tensor tree is represented here: not a spatially projected sum,
not a carrier plus centered residual. No decoder API or trained bank changes.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..su2.local_schur import cell_schur_spec
from ..su2.reduced_tree_environment import ReducedSingletTree
from .cell_multiplicity_clebsch_tree import cell_multiplicity_layouts
from .finite_schur_decoder import FiniteSchurDecoderConfig


def finite_bank_reduced_tree(bank: Mapping, config: FiniteSchurDecoderConfig) -> ReducedSingletTree:
    """Extract active complex kernels and complete orthonormal Schur leaves."""
    if set(bank) != {"nodes"} or len(bank["nodes"]) != config.tree.n_cells-1:
        raise ValueError("finite Schur nodes bank required")
    spec = cell_schur_spec(config.tree.cell_kind)
    leaves = tuple({q: np.eye(spec.multiplicity(q), dtype=np.complex128) for q in spec.q_values}
                   for _ in config.tree.cells)
    kernels = []
    for tensor, layout in zip(bank["nodes"], cell_multiplicity_layouts(config.tree), strict=True):
        array = np.asarray(tensor, dtype=np.complex128)
        expected = (len(layout.keys), config.tree.feature_dim, max(layout.left_widths), max(layout.right_widths))
        if array.shape != expected:
            raise ValueError("unexpected padded finite bank shape")
        kernels.append({key: array[index, :, :lw, :rw].copy()
                        for index, (key, lw, rw) in enumerate(zip(layout.keys, layout.left_widths, layout.right_widths, strict=True))})
    root = np.ones(config.tree.feature_dim, dtype=np.complex128)/np.sqrt(config.tree.feature_dim)
    return ReducedSingletTree(config.tree.cell_tree, leaves, tuple(kernels), root)


def reduced_tree_finite_bank(state: ReducedSingletTree, config: FiniteSchurDecoderConfig) -> dict:
    """Pack a canonical/projected raw tree back into unchanged padded storage.

    Effective rank reduction is NOT smaller allocation or faster contraction:
    widths and shapes remain exactly those of the original decoder.
    """
    if state.tree != config.tree.cell_tree:
        raise ValueError("tree and finite decoder disagree")
    nodes = []
    for kernels, layout in zip(state.kernels, cell_multiplicity_layouts(config.tree), strict=True):
        if set(kernels) != set(layout.keys):
            raise ValueError("charge blocks and finite decoder disagree")
        tensor = np.zeros((len(layout.keys), config.tree.feature_dim,
                           max(layout.left_widths), max(layout.right_widths)), dtype=np.complex128)
        for index, (key, lw, rw) in enumerate(zip(layout.keys, layout.left_widths, layout.right_widths, strict=True)):
            if kernels[key].shape != (config.tree.feature_dim, lw, rw):
                raise ValueError("changed reduced widths cannot use padded original decoder")
            tensor[index, :, :lw, :rw] = kernels[key]
        nodes.append(tensor)
    return {"nodes": tuple(nodes)}


__all__ = ["finite_bank_reduced_tree", "reduced_tree_finite_bank"]
