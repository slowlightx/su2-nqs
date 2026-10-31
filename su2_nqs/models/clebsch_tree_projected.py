"""Exactly equivalent A1 contraction after folding a root-swap subgroup.

For an S=0 root, swapping equal children multiplies CG by (-1)^q_child.
Replacing W_q by (W_q+(-1)^q W_q^T)/2 therefore implements the root
permutation average. When this permutation belongs to the physical space
group, P_A1 P_swap = P_A1. The PROJECTED state and ALL parameter derivatives
are unchanged; only redundant orbit evaluations are removed. No capacity,
learned parameter or magnetic convention is changed.
"""
from __future__ import annotations

from collections.abc import Callable,Sequence
import jax
import jax.numpy as jnp
import numpy as np

from .literature_clebsch_tree import LiteratureClebschTreeConfig,clebsch_layers,make_clebsch_amplitudes
from ..lattices.spatial_symmetry import validated_permutation_group
from ..su2.fusion_tree import Leaf,TreeNode


def root_swap_bank(params: dict,config: LiteratureClebschTreeConfig) -> dict:
    """Apply the exact root-permutation intertwiner average; q means 2j."""
    if config.target_q != 0: raise ValueError('this root-fold implementation is for S=0 only')
    layer=clebsch_layers(config)[-1]
    parts=[]; cursor=0; bank=params['merges'][-1]
    for (ql,qr,q),shape in zip(layer.keys,layer.shapes,strict=True):
        if q != 0 or ql != qr or shape[1] != shape[2]:
            raise ValueError('root swap requires equivalent left/right reduced channels')
        count=int(np.prod(shape)); block=bank[cursor:cursor+count].reshape(shape)
        parts.append((.5*(block+(-1)**ql*jnp.swapaxes(block,1,2))).reshape(-1))
        cursor+=count
    return dict(params,merges=params['merges'][:-1]+(jnp.concatenate(parts),))


def make_projected_clebsch(config: LiteratureClebschTreeConfig,physical_order: Sequence[int],
                         group_images: Sequence[Sequence[int]],*,orbit_chunk: int=8,
                         fold_root: bool=True) -> tuple[Callable,dict]:
    """Return exact A1 physical batch amplitudes and a verified orbit-size report.

    If the tree-root exchange is not a physical symmetry, use the original
    complete group. The reduced sum is a right-coset sum, NOT a subset
    approximation to Reynolds projection.
    """
    order=np.asarray(physical_order,dtype=np.int32)
    n=config.tree.n_leaves
    if order.shape != (n,) or sorted(order.tolist()) != list(range(n)):
        raise ValueError('physical_order must be a permutation')
    group=np.asarray(validated_permutation_group(tuple(tuple(int(x) for x in row) for row in group_images)),dtype=np.int32)
    if group.shape[1] != n or orbit_chunk < 1: raise ValueError('group sites/chunk mismatch')
    use_fold=False; representatives=group
    # Balanced equal children plus SAME layer-shared bank are essential.
    root=config.tree.root
    left=tuple(leaf.index for leaf in _leaves(root.left))
    right=tuple(leaf.index for leaf in _leaves(root.right))
    if fold_root and config.target_q == 0 and len(left) == len(right):
        swap=np.arange(n,dtype=np.int32)
        swap[order[np.asarray(left)]]=order[np.asarray(right)]
        swap[order[np.asarray(right)]]=order[np.asarray(left)]
        if tuple(swap) in set(map(tuple,group)):
            keys=[min(tuple(g),tuple(g[swap])) for g in group]
            representatives,counts=np.unique(np.asarray(keys),axis=0,return_counts=True)
            if not np.all(counts == 2): raise ArithmeticError('root subgroup cosets incomplete')
            use_fold=True
    if len(representatives)%orbit_chunk:
        raise ValueError('orbit chunk must divide exact quotient order')
    images=jnp.asarray(representatives).reshape(-1,orbit_chunk,n)
    order_jax=jnp.asarray(order); raw=make_clebsch_amplitudes(config)
    def amplitudes(params,bits):
        rows=jnp.asarray(bits,dtype=jnp.int32)
        if rows.ndim != 2 or rows.shape[1] != n: raise ValueError('expected physical [batch, sites]')
        bank=root_swap_bank(params,config) if use_fold else params
        def one(row):
            return jax.lax.map(lambda maps:jnp.sum(raw(bank,row[maps][:,order_jax])),images).sum()/len(representatives)
        return jax.vmap(one)(rows)
    return amplitudes,dict(full_group_order=len(group),evaluation_images=len(representatives),root_fold=use_fold,
                           projected_state_unchanged=True,parameters_unchanged=True)


def _leaves(node: TreeNode) -> tuple[Leaf,...]:
    if isinstance(node,Leaf): return (node,)
    return _leaves(node.left)+_leaves(node.right)
