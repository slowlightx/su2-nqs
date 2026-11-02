"""Numeric exact-small fixed-CG overlaps on a supplied physical bit support."""
from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from typing import Sequence

import numpy as np

from .clebsch_gordan import clebsch_gordan
from .fusion_tree import FusionTree, Leaf
from .quantum_numbers import coupling_range, validate_q
from .tree_rotations import leaf_signature


@lru_cache(maxsize=None)
def _cg_magnetic_table(q_left: int,q_right: int,q_total: int) -> np.ndarray:
    result = np.zeros((q_left+1,q_right+1),dtype=np.float64)
    for left in range(-q_left,q_left+1,2):
        for right in range(-q_right,q_right+1,2):
            if abs(left+right) <= q_total:
                result[(left+q_left)//2,(right+q_right)//2] = clebsch_gordan(
                    q_left,left,q_right,right,q_total,left+right)
    result.setflags(write=False)
    return result


def tree_overlaps_on_bits(tree: FusionTree, labels: Sequence[Sequence[int]],
                          bits: np.ndarray, *, physical_leaf_order: Sequence[int] | None = None,
                          maximum_entries: int = 25_000_000) -> np.ndarray:
    """Return U[physical row, complete doubled-q column], in float64.

    Bit zero is down; site zero is the most significant bit. Magnetic sums
    are doubled integers and the fixed Condon-Shortley CG convention is used.
    This is a reference conversion, not a scalable physical decoder for an
    arbitrary complete configuration-dependent reduced-space network.
    The explicit entry cap rejects large conversions BEFORE allocating them.
    """
    physical = np.asarray(bits)
    rows = np.asarray(labels)
    n = tree.n_leaves
    if physical.ndim != 2 or physical.shape[1] != n or not np.issubdtype(physical.dtype,np.integer):
        raise ValueError("physical bits must be an integer [rows,N] matrix")
    if np.any((physical != 0) & (physical != 1)):
        raise ValueError("physical bits must be zero/down or one/up")
    if rows.ndim != 2 or rows.shape[1] != n-1 or not np.issubdtype(rows.dtype,np.integer):
        raise ValueError("labels must be an integer [columns,N-1] matrix")
    if isinstance(maximum_entries,bool) or not isinstance(maximum_entries,int) or maximum_entries<1:
        raise ValueError("maximum_entries must be a positive integer")
    if len(physical)*len(rows) > maximum_entries:
        raise ValueError("exact-small overlap entry cap: no large conversion attempted")
    order = tuple(range(n)) if physical_leaf_order is None else tuple(physical_leaf_order)
    if any(isinstance(site,(bool,np.bool_)) or not isinstance(site,Integral) for site in order) or sorted(order) != list(range(n)):
        raise ValueError("physical_leaf_order must be a permutation")
    model = physical[:,np.asarray(order)]
    node_index = {node:i for i,node in enumerate(tree.internal_nodes)}
    supports = {node:leaf_signature(node) for node in tree.internal_nodes}
    magnetic = {node:np.sum(2*model[:,np.asarray(support)]-1,axis=1)
                for node,support in supports.items()}
    result = np.zeros((len(physical),len(rows)),dtype=np.float64)
    for column,row in enumerate(rows):
        if row[-1] != 0:
            raise ValueError("this overlap is for a strict root-q=0 sector")
        amplitude = np.ones(len(physical),dtype=np.float64)
        for index,node in enumerate(tree.internal_nodes):
            ql = 1 if isinstance(node.left,Leaf) else int(row[node_index[node.left]])
            qr = 1 if isinstance(node.right,Leaf) else int(row[node_index[node.right]])
            q = int(row[index])
            validate_q(q)
            if q not in coupling_range(ql,qr):
                raise ValueError("invalid doubled-q coupling in supplied label row")
            ml = 2*model[:,node.left.index]-1 if isinstance(node.left,Leaf) else magnetic[node.left]
            mr = 2*model[:,node.right.index]-1 if isinstance(node.right,Leaf) else magnetic[node.right]
            valid = (abs(ml)<=ql)&(abs(mr)<=qr)&(abs(ml+mr)<=q)
            table = _cg_magnetic_table(ql,qr,q)
            amplitude *= np.where(valid,table[np.clip((ml+ql)//2,0,ql),np.clip((mr+qr)//2,0,qr)],0.)
        result[:,column] = amplitude*(magnetic[tree.root] == 0)
    return result


__all__ = ["tree_overlaps_on_bits"]
