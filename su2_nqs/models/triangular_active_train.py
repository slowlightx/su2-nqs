"""Explicit fixed-profile trainable Clebsch bank, distinct from native padding.

All local reduced matrices and all declared active upper coefficients are
trainable. This is a free folded-nine-site tree family, not a preservation
of a binary rank-8 factorization or its inactive-channel tangent directions.
Magnetic structure and root charge remain the existing exact SU(2) decoder.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from .triangular_clebsch_active_layout import mixed_active_layout, repack_active_clebsch_bank
from .triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout


def lift_active_train_bank(bank: dict, config: MixedClebschConfig,
        profile: Sequence[Mapping[int, int]]) -> dict:
    """Pure-JAX zero lift to native axes, preserving active-state tangents.

    No QR completion or new trainable parameter is introduced. Taking a
    derivative of the lifted amplitude therefore means derivatives through
    exactly the active coordinates, not all padded native coordinates.
    """
    active = mixed_active_layout(config, profile)
    native = mixed_clebsch_layout(config)
    if len(bank['top']) != len(active):
        raise ValueError('complete active hierarchy required')
    top = []
    for values, row, destination in zip(bank['top'], active, native, strict=True):
        if len(values) != len(row['groups']):
            raise ValueError('active group count mismatch')
        lookup = {}
        for weights, group in zip(values, row['groups'], strict=True):
            if weights.shape != group['shape']:
                raise ValueError('active tensor shape mismatch')
            lookup.update(zip(group['keys'], weights, strict=True))
        buckets = []
        for group in destination['groups']:
            target = group['shape'][1:]
            rows = []
            for key in group['keys']:
                value = lookup.get(key)
                rows.append(jnp.zeros(target, dtype=jnp.float64) if value is None else
                    jnp.pad(value, tuple((0, full-current)
                        for full, current in zip(target, value.shape, strict=True))))
            buckets.append(jnp.stack(rows))
        top.append(tuple(buckets))
    return dict(local=bank['local'], top=tuple(top))


def canonical_active_sampling_bank(bank: dict, config: MixedClebschConfig,
        profile: Sequence[Mapping[int, int]]) -> tuple[dict, dict]:
    """Canonicalize only the exact sampler representation, outside AD.

    Original active optimizer coordinates are unchanged. The established
    block-QR/R-into-parent algorithm is applied at the ACTUAL active widths,
    never to native padded columns (whose QR completions may rotate support).
    Normalized raw sampling and normalized physical-projection products use
    the returned raw norm; training amplitudes retain their original scale.
    """
    layout=mixed_active_layout(config,profile)
    local=[{q:np.asarray(w).copy() for q,w in zip(config.local_q,row,strict=True)} for row in bank['local']]
    top=[]
    for values,row in zip(bank['top'],layout,strict=True):
        mapping={}
        for array,group in zip(values,row['groups'],strict=True):
            if array.shape!=group['shape']:raise ValueError('active tensor shape mismatch')
            mapping.update({key:np.asarray(v).copy() for key,v in zip(group['keys'],array,strict=True)})
        top.append(mapping)
    count=len(local);defects=[]
    for edge in range(count+len(top)-1):
        factors={}
        if edge<count:
            for q,matrix in local[edge].items():
                frame,r=np.linalg.qr(matrix,mode='reduced')
                if frame.shape!=matrix.shape:raise ValueError('local active rank exceeds complete Schur rows')
                local[edge][q]=frame;factors[q]=r
                defects.append(float(np.max(abs(frame.conj().T@frame-np.eye(frame.shape[1])))))
        else:
            node=top[edge-count]
            for q in sorted({key[2] for key in node}):
                keys=[key for key in node if key[2]==q]
                pieces=[node[key].reshape(node[key].shape[0],-1).T for key in keys]
                matrix=np.concatenate(pieces,axis=0)
                frame,r=np.linalg.qr(matrix,mode='reduced')
                if frame.shape!=matrix.shape:raise ValueError('active retained width exceeds child product')
                cursor=0
                for key,piece in zip(keys,pieces,strict=True):
                    node[key]=frame[cursor:cursor+len(piece)].T.reshape(node[key].shape);cursor+=len(piece)
                factors[q]=r
                defects.append(float(np.max(abs(frame.conj().T@frame-np.eye(frame.shape[1])))))
        parent=next(i for i,row in enumerate(layout) if edge in (row['left'],row['right']))
        is_left=layout[parent]['left']==edge
        top[parent]={key:(np.einsum('alr,ul->aur',w,factors[key[0]],optimize=True) if is_left else
            np.einsum('alr,vr->alv',w,factors[key[1]],optimize=True)) for key,w in top[parent].items()}
    canonical=dict(local=tuple(tuple(row[q] for q in config.local_q) for row in local),
        top=tuple(tuple(np.stack([node[key] for key in group['keys']]) for group in row['groups'])
            for node,row in zip(top,layout,strict=True)))
    norm = sum(float(np.vdot(x, x).real) for x in canonical['top'][-1])
    if not np.isfinite(norm) or norm <= 0:
        raise ArithmeticError('active raw tree norm is not finite positive')
    canonical['top'] = (*canonical['top'][:-1],
        tuple(np.asarray(x)/np.sqrt(norm) for x in canonical['top'][-1]))
    # The reverse zero lift/repack checks exact closure without adding QR
    # completion columns to the model or deleting small coefficients.
    active = repack_active_clebsch_bank(lift_active_train_bank(canonical,config,profile),config,profile)
    return active, dict(center='singlet_root', maximum_isometry_defect=max(defects,default=0.),
        raw_tree_only_metric=True, original_raw_norm=norm,
        active_optimizer_coordinates_unchanged=True,
        exact_inactive_parent_closure_verified=True,
        native_inactive_opening_gradients_not_claimed=True,
        variational_family='free_folded_nine_site_fixed_active_upper_profile')
