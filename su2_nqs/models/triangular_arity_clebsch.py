"""A finite-rank binary control for the genuine three-input Clebsch node.

Both arms use the SAME triangle cover, nine-site output widths and upper
binary tree. A triangle retains its complete q=1 (two copies) and q=3
Schur spaces. The binary arm then applies W_(3,3->6) and W_(6,3->9),
with an actual finite multiplicity rank on the six-site edge. This is the
same fixed-CG/reduced-W rule used by the square ClebschTree. Complete
lower triangle maps are fixed basis changes; their invertible gauges can
be absorbed into the next W without restricting the state family.

The joint arm has ONE independent coefficient for every legal
(q0,q1,q2,q12,a0,a1,a2,output-copy) channel. Computing its invariant basis
with two CG coefficients does NOT factor its trainable tensor into two
binary maps. In particular q12 is a complete basis index, not a retained
finite-width virtual edge. No attention or Spatial-ViT is involved.
"""
from __future__ import annotations

from numbers import Integral
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.triangle_blocks import triangle_block_tables
from .triangular_mixed_clebsch import MixedClebschConfig

ArityMode = Literal['binary_truncated', 'ternary_joint']
SIX_Q = (0, 2, 4, 6)


def binary_intermediate_widths(rank: int) -> dict[int, int]:
    """Retained SIX-site multiplicities at doubled q=0,2,4,6.

    The complete widths are (5,9,5,1). ``rank>=9`` recovers the complete
    binary parameterization and is not an arity-expressivity comparison.
    """
    if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 1:
        raise ValueError('binary intermediate multiplicity rank must be a positive integer')
    tables = triangle_block_tables()
    return {q: min(int(rank), tables['six'][q].shape[1]) for q in SIX_Q}


def _pieces(config: MixedClebschConfig, q: int) -> tuple[tuple[int, int, int, int], ...]:
    """(q6,q3,full-six-width,three-width), in the existing basis order."""
    data = triangle_block_tables()
    channels = data['binary_channels'][q]
    return tuple((q6, q3, data['six'][q6].shape[1], data['three'][q3].shape[1])
                 for q6 in SIX_Q for q3 in (1, 3)
                 if any(a == q6 and b == q3 for a, b, _, _ in channels))


def binary_to_joint_bank(params: dict, config: MixedClebschConfig, rank: int) -> dict:
    """Fold genuine two-input W maps once, outside all physical/H queries.

    The six-site W is shared across ALL nine-site output charges and third
    triangle channels. Separate per-output factorizations would not define
    one consistent binary tensor network and are deliberately not used.
    """
    widths = binary_intermediate_widths(rank)
    data = triangle_block_tables()
    local = []
    for node, six in zip(params['local'], params['six'], strict=True):
        values = []
        for q, matrix in zip(config.local_q, node, strict=True):
            pieces = []
            offset = 0
            for q6, _, d6, d3 in _pieces(config, q):
                count = widths[q6] * d3
                block = matrix[offset:offset+count].reshape(widths[q6], d3, -1)
                pieces.append(jnp.einsum('ar,rbk->abk', six[q6//2], block).reshape(d6*d3, -1))
                offset += count
            if offset != matrix.shape[0]:
                raise ValueError('binary node layout does not match the declared rank')
            values.append(jnp.asarray(data['binary_overlaps'][q].T) @ jnp.concatenate(pieces))
        local.append(tuple(values))
    return dict(local=tuple(local), top=params['top'])


def arity_clebsch_bank(params: dict, config: MixedClebschConfig,
                      mode: ArityMode, rank: int) -> dict:
    """One shared physical-amplitude bank; magnetic indices are fixed CG."""
    if mode == 'binary_truncated':
        return binary_to_joint_bank(params, config, rank)
    if mode == 'ternary_joint':
        return dict(local=params['local'], top=params['top'])
    raise ValueError('unknown binary/ternary arity arm')


def six_edge_unfoldings(bank: dict, config: MixedClebschConfig) -> tuple[dict[int, np.ndarray], ...]:
    """Offline joint-W matricizations across (triangles 0,1)|(2,output).

    Row dimension is the COMPLETE six-spin multiplicity. All compatible
    output q/copies are concatenated. A binary rank-r node has rank<=r in
    each such matrix, while a joint ternary tensor need not.
    """
    data = triangle_block_tables()
    result = []
    for node in bank['local']:
        sectors = {q: [] for q in SIX_Q}
        for q, weight in zip(config.local_q, node, strict=True):
            binary = data['binary_overlaps'][q] @ np.asarray(weight)
            offset = 0
            for q6, _, d6, d3 in _pieces(config, q):
                block = binary[offset:offset+d6*d3].reshape(d6, -1)
                sectors[q6].append(block)
                offset += d6*d3
        result.append({q: np.concatenate(parts, axis=1) for q, parts in sectors.items() if parts})
    return tuple(result)


def init_arity_pair(bank: dict, config: MixedClebschConfig, rank: int) -> tuple[dict, dict]:
    """Compress one parent ONCE, then lift it exactly into both trainable arms.

    A truncated SVD initializes the binary edge consistently across all
    output charges. The ternary arm starts at this SAME compressed physical
    state, NOT the uncompressed, potentially stronger parent. SVD is offline
    initialization only, never inside a JAX gradient or an H query. Its
    Frobenius error is a local coefficient diagnostic, NOT a global fidelity.
    """
    widths = binary_intermediate_widths(rank)
    data = triangle_block_tables()
    local, six, losses = [], [], []
    for node, unfolded in zip(bank['local'], six_edge_unfoldings(bank, config), strict=True):
        axes = {}
        spectrum = {}
        for q6, matrix in unfolded.items():
            u, s, _ = np.linalg.svd(matrix, full_matrices=False)
            r = widths[q6]
            if u.shape[1] < r:
                # Complete a legal reduced-feature frame without an arbitrary
                # numerical singular-value cutoff or deleting an SU2 charge.
                u = np.linalg.qr(np.column_stack((u, np.eye(len(u)))))[0]
            axes[q6] = u[:, :r]
            spectrum[q6] = dict(singular_values=s.tolist(), retained_width=r,
                               discarded_squared_norm=float(np.sum(s[r:]**2)))
        values = []
        for q, w in zip(config.local_q, node, strict=True):
            binary = data['binary_overlaps'][q] @ np.asarray(w)
            pieces = []
            offset = 0
            for q6, _, d6, d3 in _pieces(config, q):
                block = binary[offset:offset+d6*d3].reshape(d6, d3, -1)
                pieces.append(np.einsum('ar,abk->rbk', axes[q6], block).reshape(widths[q6]*d3, -1))
                offset += d6*d3
            values.append(jnp.asarray(np.concatenate(pieces)))
        local.append(tuple(values))
        # Unused charges still have a deterministic, legal reduced frame.
        six.append(tuple(jnp.asarray(axes.get(q, np.eye(data['six'][q].shape[1])[:, :widths[q]]))
                         for q in SIX_Q))
        losses.append(spectrum)
    binary = dict(local=tuple(local), six=tuple(six), top=bank['top'])
    joint = binary_to_joint_bank(binary, config, rank)
    family = dict(binary_truncated=binary, ternary_joint=joint)
    note = dict(binary_rank=int(rank), six_edge_widths=widths,
                complete_six_edge_widths={q: data['six'][q].shape[1] for q in SIX_Q},
                local_truncation_spectra=losses, same_physical_initial_state=True,
                parent_preserved_exactly=rank >= 9,
                local_Frobenius_error_is_not_global_fidelity=True,
                joint_tensor_factorized=False, square_style_fixed_CG_reduced_W=True,
                local_C3_imposed=False, full_lattice_A1_required=True)
    return family, note
