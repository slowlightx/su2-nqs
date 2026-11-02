"""Physical-input, layer-shared ClebschTree for Vieijra--Nys benchmarks.

This is NOT the autoregressive fusion-path model, nor an attention-generated
tensor bank. Real independent reduced W matrices and fixed Condon--Shortley
CG coefficients define the amplitude. Magnetic indices are never trainable.
Widths follow tau_q'=min(sum tau_left*tau_right, cap_q) with q=2j.
The compiler pads contractions, not the trainable parameter vector, and stores
only reachable numerical CG triples (no subtree-size dense CG allocation).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.fusion_tree import FusionTree, FusionNode, Leaf, TreeNode
from ..su2.quantum_numbers import coupling_range, magnetic_numbers, validate_m, validate_q


@dataclass(frozen=True)
class LiteratureClebschTreeConfig:
    """Exact root (q,m2), doubled intermediate cutoff and reduced width caps."""

    tree: FusionTree
    target_q: int = 0
    target_m2: int = 0
    maximum_q: int = 8
    multiplicity_cap: int = 8
    high_spin_cap: int | None = None
    high_spin_threshold_q: int = 4

    def __post_init__(self) -> None:
        validate_q(self.target_q, name="target_q")
        validate_m(self.target_q, self.target_m2, name="target_m2")
        validate_q(self.maximum_q, name="maximum_q")
        validate_q(self.high_spin_threshold_q, name="high_spin_threshold_q")
        if (self.tree.n_leaves < 2 or not self.tree.target_is_reachable(self.target_q)
                or self.maximum_q < max(1, self.target_q)):
            raise ValueError("unreachable target or insufficient maximum_q")
        for value in (self.multiplicity_cap, self.high_spin_cap):
            if value is not None and (isinstance(value, bool)
                    or not isinstance(value, Integral) or value < 1):
                raise ValueError("multiplicity caps must be positive integers")


@dataclass(frozen=True)
class ClebschLayer:
    """One shared reduced bank and topologically ordered spatial merges."""

    nodes: tuple[FusionNode, ...]
    keys: tuple[tuple[int, int, int], ...]
    widths: tuple[int, ...]
    shapes: tuple[tuple[int, int, int], ...]
    parameter_size: int
    padded_indices: np.ndarray
    cg: np.ndarray
    magnetic_bound: int


def _size(node: TreeNode) -> int:
    return 1 if isinstance(node, Leaf) else _size(node.left) + _size(node.right)


@lru_cache(maxsize=None)
def clebsch_layers(config: LiteratureClebschTreeConfig) -> tuple[ClebschLayer, ...]:
    """Compile shared W entries and numeric CG data outside JAX runtime."""
    bound = config.maximum_q
    width = max(config.multiplicity_cap, config.high_spin_cap or 1)
    groups: dict[tuple[int, int, bool], list[FusionNode]] = {}
    for node in config.tree.internal_nodes:
        groups.setdefault((_size(node.left), _size(node.right), node == config.tree.root), []).append(node)
    groups = dict(sorted(groups.items(), key=lambda item: sum(item[0][:2])))
    widths_by_node: dict[TreeNode, tuple[int, ...]] = {
        leaf: tuple(int(q == 1) for q in range(bound+1)) for leaf in config.tree.leaves}
    result = []
    for (_, _, is_root), nodes in groups.items():
        first = nodes[0]
        left, right = widths_by_node[first.left], widths_by_node[first.right]
        # Equal-size merges must have identical channel structure to share W.
        if any(widths_by_node[n.left] != left or widths_by_node[n.right] != right for n in nodes):
            raise ValueError("same-size subtrees require the same reduced channel layout")
        keys = tuple((a,b,q) for a,ta in enumerate(left) if ta
                     for b,tb in enumerate(right) if tb
                     for q in coupling_range(a,b)
                     if q <= bound and (not is_root or q == config.target_q))
        out = [0]*(bound+1)
        for a,b,q in keys:
            out[q] += left[a]*right[b]
        for q in range(bound+1):
            cap = (config.high_spin_cap if config.high_spin_cap is not None
                   and q > config.high_spin_threshold_q else config.multiplicity_cap)
            out[q] = min(out[q], cap)
        shapes = tuple((out[q], left[a], right[b]) for a,b,q in keys)
        indices = np.full((len(keys),width,width,width), -1, dtype=np.int32)
        cursor = 0
        for k,shape in enumerate(shapes):
            count = int(np.prod(shape))
            indices[k,:shape[0],:shape[1],:shape[2]] = np.arange(cursor,cursor+count).reshape(shape)
            cursor += count
        if not cursor:
            raise ValueError("cutoff removes all root paths")
        cg = np.zeros((len(keys),2*bound+1,2*bound+1), dtype=np.float64)
        for k,(a,b,q) in enumerate(keys):
            for ma in magnetic_numbers(a):
                for mb in magnetic_numbers(b):
                    cg[k,ma+bound,mb+bound] = clebsch_gordan(a,ma,b,mb,q,ma+mb)
        indices.setflags(write=False); cg.setflags(write=False)
        layer = ClebschLayer(tuple(nodes),keys,tuple(out),shapes,cursor,indices,cg,bound)
        result.append(layer)
        for node in nodes:
            widths_by_node[node] = tuple(out)
    return tuple(result)


def make_clebsch_features(config: LiteratureClebschTreeConfig) -> Callable:
    """Return a batch evaluator of all layer features, including magnetic flow."""
    layers = clebsch_layers(config)
    bound = config.maximum_q
    width = max(config.multiplicity_cap, config.high_spin_cap or 1)

    def features(params: dict, spins: jax.Array) -> tuple[jax.Array, ...]:
        rows = jnp.asarray(spins, dtype=jnp.int32)
        if rows.ndim != 2 or rows.shape[1] != config.tree.n_leaves:
            raise ValueError("expected [batch, n_sites] binary physical configurations")
        values = {}
        leaf_features = jnp.zeros((len(rows),bound+1,width)).at[:,1,0].set(1.)
        for leaf in config.tree.leaves:
            values[leaf] = (2*rows[:,leaf.index]-1, leaf_features)
        outputs = []
        for bank,layer in zip(params['merges'],layers,strict=True):
            left_m = jnp.stack([values[n.left][0] for n in layer.nodes],axis=1)
            right_m = jnp.stack([values[n.right][0] for n in layer.nodes],axis=1)
            left_f = jnp.stack([values[n.left][1] for n in layer.nodes],axis=1)
            right_f = jnp.stack([values[n.right][1] for n in layer.nodes],axis=1)
            channels = jnp.asarray(layer.keys)
            padded = jnp.concatenate((bank,jnp.zeros((1,),dtype=bank.dtype)))[jnp.asarray(layer.padded_indices)]
            reduced = jnp.einsum('bnca,bncd,coad->bnco',
                                  left_f[:,:,channels[:,0]],right_f[:,:,channels[:,1]],padded)
            lm = jnp.clip(left_m+bound,0,2*bound)
            rm = jnp.clip(right_m+bound,0,2*bound)
            cg = jnp.asarray(layer.cg)[jnp.arange(len(layer.keys))[None,None,:],lm[:,:,None],rm[:,:,None]]
            cg = jnp.where((abs(left_m) <= bound)[:,:,None] & (abs(right_m) <= bound)[:,:,None],cg,0.)
            out = jnp.zeros((len(rows),len(layer.nodes),bound+1,width),dtype=bank.dtype)
            out = out.at[:,:,channels[:,2],:].add(reduced*cg[:,:,:,None])
            for k,node in enumerate(layer.nodes):
                values[node] = (left_m[:,k]+right_m[:,k],out[:,k])
            outputs.append(out)
        return tuple(outputs)
    return features


def make_clebsch_amplitudes(config: LiteratureClebschTreeConfig) -> Callable:
    """Batch physical amplitudes in exact (S,M); invalid bits give zero."""
    features = make_clebsch_features(config)

    def amplitude(params: dict, spins: jax.Array) -> jax.Array:
        rows = jnp.asarray(spins,dtype=jnp.int32)
        valid = jnp.all((rows == 0) | (rows == 1),axis=1)
        safe = jnp.clip(rows,0,1)
        root = features(params,safe)[-1][:,0,config.target_q,:len(params['readout'])]
        result = root@params['readout']
        return jnp.where(valid & (jnp.sum(2*safe-1,axis=1) == config.target_m2),result,0.)
    return amplitude


def init_literature_clebsch_tree(key: jax.Array, config: LiteratureClebschTreeConfig,
                               calibration_states: np.ndarray | None = None) -> tuple[dict, dict]:
    """Independent real uniform W, optionally positive layer-std calibrated.

    One common positive scalar rescales every W in a layer, preserving its
    relative channels/signs and the initial physical ray. Padding is excluded
    from empirical standard deviations. This initialization is not a fit.
    """
    layers = clebsch_layers(config)
    keys = jax.random.split(key,len(layers)+1)
    params = dict(merges=tuple(jax.random.uniform(k,(l.parameter_size,),minval=-1.,maxval=1.,dtype=jnp.float64)
                              for k,l in zip(keys[:-1],layers,strict=True)),
                  readout=jax.random.uniform(keys[-1],(layers[-1].widths[config.target_q],),
                                             minval=-1.,maxval=1.,dtype=jnp.float64))
    report = dict(uniform_range=[-1.,1.],layer_standardization=calibration_states is not None,
                  doubled_labels=True,magnetic_parameters=False,layers=[])
    if calibration_states is not None:
        rows = np.asarray(calibration_states)
        if rows.ndim != 2 or rows.shape[1] != config.tree.n_leaves or not np.isin(rows,[0,1]).all():
            raise ValueError("calibration must contain valid physical bit rows")
        evaluate = jax.jit(make_clebsch_features(config))
        for index,layer in enumerate(layers):
            out = np.asarray(evaluate(params,jnp.asarray(rows))[index])
            active = np.asarray([[k < t for k in range(out.shape[-1])] for t in layer.widths])
            std = float(np.std(out[:,:,active]))
            if not np.isfinite(std) or std <= 1e-15:
                raise FloatingPointError("layer calibration has vanishing or invalid scale")
            banks = list(params['merges']); banks[index] = banks[index]/std
            params = dict(params,merges=tuple(banks))
            after = float(np.std(np.asarray(evaluate(params,jnp.asarray(rows))[index])[:,:,active]))
            report['layers'].append(dict(index=index,scale=1/std,std_after=after,
                                         widths=list(layer.widths),parameters=layer.parameter_size))
    report['parameter_count'] = sum(l.parameter_size for l in layers)+len(params['readout'])
    return params,report
