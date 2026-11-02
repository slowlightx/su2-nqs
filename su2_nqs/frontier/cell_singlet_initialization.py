"""Local low-energy initialization of the existing full-q physical CG decoder.

Only this frontier helper knows about H. The learned model and its exact
magnetic tensors are unchanged. All spin labels are doubled integers.
"""
from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..models.cell_multiplicity_clebsch_tree import (
    SpinCellMultiplicityConfig, cell_multiplicity_layouts,
)
from ..operators.sz_heisenberg import SzHeisenbergGraph
from ..su2.local_schur import cell_schur_tensor


def local_singlet_reference(bonds: Sequence[tuple[int, int, float]]) -> dict:
    """Solve only the two-dimensional q=0 sector of one four-spin cell.

    Physical rows use site-zero-most-significant bits. This is local N=4
    algebra, not a finite-torus ED calculation or an N16 teacher transfer.
    """
    graph = SzHeisenbergGraph(4, tuple(bonds), n_up=2)
    bits = np.asarray(graph.basis, dtype=np.int32)
    indices = bits @ (1 << np.arange(3, -1, -1))
    columns = cell_schur_tensor('square_plaquette')[:, 0, 4, :2]
    h = graph.to_dense()
    reduced = columns[indices].T @ h @ columns[indices]
    values, vectors = np.linalg.eigh(reduced)
    coefficients = vectors[:, 0]
    coefficients *= np.sign(coefficients[np.argmax(abs(coefficients))])
    physical = columns @ coefficients
    residual = np.linalg.norm(h @ physical[indices] - values[0]*physical[indices])
    return dict(coefficients=coefficients, physical=physical,
                energy=float(values[0]), singlet_gap=float(values[1]-values[0]),
                eigen_residual=float(residual), bonds=tuple(bonds))


def cell_product_parameters(config: SpinCellMultiplicityConfig, coefficients: np.ndarray) -> dict:
    """Embed a normalized real local q=0 vector as an exact product state.

    The full parameter structure is retained; only feature zero and the
    (q_left,q_right,q_total)=(0,0,0) maps are initially nonzero. Imaginary
    tensors have the same product maps with zero readout. There is no cutoff.
    """
    c = np.asarray(coefficients, dtype=np.float64)
    if (config.cell_kind != 'square_plaquette' or config.merge_mode != 'full_joint'
            or config.target_q != 0 or config.target_m2 != 0
            or config.maximum_intermediate_q is not None):
        raise ValueError('require the full-q, full_joint four-spin singlet decoder')
    if c.shape != (2,) or not np.isfinite(c).all() or not np.isclose(c@c, 1., atol=1e-12, rtol=0):
        raise ValueError('need normalized real q=0 multiplicity coefficients')
    unique = {x.parameter_index: x for x in cell_multiplicity_layouts(config)}
    merges = []
    for index in range(len(unique)):
        layout = unique[index]; blocks = []
        for channel, left, right in zip(layout.keys, layout.left_widths, layout.right_widths, strict=True):
            value = np.zeros((config.feature_dim, left, right))
            if channel == (0, 0, 0):
                a = c if layout.left_size == 4 else np.eye(1, left)[0]
                b = c if layout.right_size == 4 else np.eye(1, right)[0]
                value[0] = np.outer(a, b)
            blocks.append(jnp.asarray(value))
        merges.append({'joint': tuple(blocks)})
    readout = jnp.asarray(np.eye(1, config.feature_dim)[0])
    return dict(real=dict(merges=tuple(merges), readout=readout),
                imag=dict(merges=tuple(merges), readout=jnp.zeros_like(readout)))


def perturb_cell_product(params: dict, *, seed: int, scale: float) -> dict:
    """Seed every reduced tensor, without changing the manifold or freezing it.

    Add iid Gaussian reduced-map entries with SD scale/sqrt(input fan-in),
    and readout entries with SD scale. Scale is NOT a physical norm ratio.
    Generic perturbation is not a proof of nonzero support on all bit strings;
    runtime node/mixing diagnostics are still required.
    """
    if not np.isfinite(scale) or scale < 0:
        raise ValueError('scale must be finite and nonnegative')
    rng = np.random.default_rng(seed)
    def perturb(value):
        x = np.asarray(value)
        fan = np.prod(x.shape[1:]) if x.ndim > 1 else 1
        return jnp.asarray(x + scale*rng.normal(size=x.shape)/np.sqrt(fan))
    return jax.tree_util.tree_map(perturb, params)


def product_amplitudes(states: np.ndarray, cells: tuple, local: np.ndarray) -> np.ndarray:
    """Independent direct product oracle; no global CG/tree contraction."""
    bits = np.asarray(states, dtype=np.int32)
    if bits.ndim != 2 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError('need a batch of binary physical states')
    indices = bits[:, np.asarray(cells)] @ (1 << np.arange(3, -1, -1))
    return np.prod(np.asarray(local)[indices], axis=1)


def singlet_bond_diagnostics(states: np.ndarray, bonds: Sequence, cells: tuple) -> dict:
    """3Hz estimators for intra/inter-cell energy; NOT physical variance.

    For an exact global singlet, <Si.Sj>=3<Szi Szj> for every pair, including
    after A1 projection. The fixed partition is diagnostic, not a symmetry
    breaking claim. Leading sample dimensions are preserved.
    """
    bits = np.asarray(states)
    if bits.shape[-1] != sum(map(len, cells)) or not np.all((bits == 0) | (bits == 1)):
        raise ValueError('states must match the binary cell partition')
    owner = np.empty(bits.shape[-1], dtype=int)
    for k, cell in enumerate(cells): owner[list(cell)] = k
    z = 2*bits-1; intra = np.zeros(bits.shape[:-1]); inter = np.zeros_like(intra)
    for a, b, coupling in bonds:
        value = .75*coupling*z[..., a]*z[..., b]/bits.shape[-1]
        if owner[a] == owner[b]: intra += value
        else: inter += value
    return dict(intra=intra, inter=inter, total=intra+inter)
