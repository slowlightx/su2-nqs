"""Physical-input finite-width triangle/nine-site SU(2) tensor decoders.

Three parameterizations share the SAME effective-bank amplitude kernel:
binary-free (complete six-site intermediate), ternary-free, and C3+T ternary.
Reduced banks are generated once outside physical/projection/H query loops.
This is an unnormalized tensor-tree family, not a normalized autoregressor.
Mixed means binary/ternary tree arity only. This ClebschTree branch contains
no Spatial-ViT, attention, neural tensor generator, or carrier dressing.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from ..su2.quantum_numbers import coupling_range
from ..su2.triangle_blocks import triangle_block_tables, nine_site_output, real_c3_synthesis_basis
from .spin_clebsch_tree import spin_cg_tensor

MixedMode = Literal['binary_free','ternary_free','ternary_c3']


@lru_cache(maxsize=None)
def _packed_tree_cg(left_span: int,right_span: int,keys: tuple) -> np.ndarray:
    """Take only active channels from the original ClebschTree CG reference.

    Spans are bounded retained doubled charges, not the growing physical
    subtree sizes. No attention-model dependency or forbidden runtime block
    is inherited from another decoder.
    """
    reference=spin_cg_tensor(left_span,right_span)
    table=np.stack([reference[ql,:,qr,:,qt] for ql,qr,qt in keys])
    table.setflags(write=False)
    return table


@dataclass(frozen=True)
class MixedClebschConfig:
    """Same complete local Schur content; qmax=2*jmax on retained edges.

    cap>=9 keeps the binary SIX-site intermediate complete (5,9,5,1),
    enabling an identical initial physical state and a lossless local fold.
    The nine-site and higher edges have a finite, explicitly shared budget.
    """
    length: int = 6
    multiplicity_cap: int = 16
    maximum_q: int = 10

    def __post_init__(self) -> None:
        triangular_schur_hierarchy(self.length)
        for name, minimum in (('multiplicity_cap',9),('maximum_q',3)):
            value = getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')

    @property
    def local_q(self) -> tuple[int,...]:
        return tuple(range(1,min(9,self.maximum_q)+1,2))


@lru_cache(maxsize=None)
def mixed_clebsch_layout(config: MixedClebschConfig) -> tuple[dict,...]:
    """Active exact-shape charge buckets; no padding over forbidden q blocks."""
    tree = triangular_schur_hierarchy(config.length)
    widths = [{q:nine_site_output(q,int(config.multiplicity_cap)).width for q in config.local_q}
              for _ in tree.blocks]
    sizes = [9]*len(tree.blocks)
    layouts = []
    for left,right,parent in tree.binary_merges:
        ls,rs = sizes[left],sizes[right]
        root = parent == len(tree.blocks)+len(tree.binary_merges)-1
        candidates = (0,) if root else tuple(range((ls+rs)%2,min(ls+rs,config.maximum_q)+1,2))
        out = {}
        buckets = {}
        for q in candidates:
            pairs = [(ql,qr) for ql in widths[left] for qr in widths[right] if q in coupling_range(ql,qr)]
            capacity = sum(widths[left][ql]*widths[right][qr] for ql,qr in pairs)
            if not capacity:
                continue
            out[q] = 1 if root else min(config.multiplicity_cap,capacity)
            for ql,qr in pairs:
                shape = (out[q],widths[left][ql],widths[right][qr])
                buckets.setdefault(shape,[]).append((ql,qr,q))
        groups=[]
        for shape, triples in sorted(buckets.items()):
            keys=tuple(triples)
            sl=max(q[0] for q in keys);sr=max(q[1] for q in keys)
            groups.append(dict(shape=(len(keys),*shape),keys=keys,left_span=sl,right_span=sr,
                cg=_packed_tree_cg(sl,sr,keys),
                output_groups=tuple((q,tuple(i for i,key in enumerate(keys) if key[2]==q))
                                    for q in out if any(key[2]==q for key in keys))))
        if not groups:
            raise ValueError('retained doubled charges cannot reach the root singlet')
        layouts.append(dict(left=left,right=right,parent=parent,widths=out,groups=tuple(groups)))
        widths.append(out);sizes.append(ls+rs)
    return tuple(layouts)


@lru_cache(maxsize=None)
def local_c3_bases(config: MixedClebschConfig) -> tuple[np.ndarray,...]:
    """Local independent real coefficients; reflection is not imposed."""
    data=triangle_block_tables()
    return tuple(real_c3_synthesis_basis(data['rotations'][q],nine_site_output(q,int(config.multiplicity_cap)))
                 for q in config.local_q)


def mixed_clebsch_bank(params: dict,config: MixedClebschConfig,mode: MixedMode) -> dict:
    """Convert parameters to one common physical-amplitude bank ONCE.

    Binary six-site tensors are folded before any physical queries. Since
    that intermediate is complete, this removes computation, not states.
    C3 synthesis is likewise outside every H-connected/configuration loop.
    """
    if mode not in ('binary_free','ternary_free','ternary_c3'):
        raise ValueError('unknown mixed Clebsch parameterization')
    data=triangle_block_tables()
    local=[]
    for index,node in enumerate(params['local']):
        values=[]
        for k,q in enumerate(config.local_q):
            if mode=='ternary_c3':
                values.append(jnp.einsum('p,pik->ik',node[k],jnp.asarray(local_c3_bases(config)[k])))
            elif mode=='ternary_free':
                values.append(node[k])
            else:
                channels=data['binary_channels'][q]
                pieces=[];offset=0
                for q6 in (0,2,4,6):
                    for q3 in (1,3):
                        count=sum(ql==q6 and qr==q3 for ql,qr,_,_ in channels)
                        if not count:continue
                        d6=data['six'][q6].shape[1];d3=data['three'][q3].shape[1]
                        block=node[k][offset:offset+count].reshape(d6,d3,-1)
                        matrix=params['six'][index][q6//2]
                        pieces.append(jnp.einsum('au,ubk->abk',matrix,block).reshape(count,-1))
                        offset+=count
                folded=jnp.concatenate(pieces,axis=0)
                values.append(jnp.asarray(data['binary_overlaps'][q].T)@folded)
        local.append(tuple(values))
    return dict(local=tuple(local),top=params['top'])


def init_mixed_clebsch_family(key: jax.Array,config: MixedClebschConfig) -> dict[str,dict]:
    """Initialize all three arms at the same physical state and upper tensors."""
    count=len(triangular_schur_hierarchy(config.length).blocks)
    keys=iter(jax.random.split(key,count*len(config.local_q)+sum(len(x['groups']) for x in mixed_clebsch_layout(config))))
    coefficients=[]
    for _ in range(count):
        coefficients.append(tuple(jax.random.normal(next(keys),(len(basis),),dtype=jnp.float64)
                           /np.sqrt(basis.shape[1]) for basis in local_c3_bases(config)))
    top=tuple(tuple(jax.random.normal(next(keys),group['shape'],dtype=jnp.float64)
                    /np.sqrt(group['shape'][-1]*group['shape'][-2])
                    for group in row['groups']) for row in mixed_clebsch_layout(config))
    constrained=dict(local=tuple(coefficients),top=top)
    bank=mixed_clebsch_bank(constrained,config,'ternary_c3')
    free=dict(local=bank['local'],top=top)
    data=triangle_block_tables()
    binary=dict(local=tuple(tuple(jnp.asarray(data['binary_overlaps'][q])@w
                    for q,w in zip(config.local_q,node,strict=True)) for node in bank['local']),
                six=tuple(tuple(jnp.eye(data['six'][q].shape[1],dtype=jnp.float64)
                    for q in (0,2,4,6)) for _ in range(count)),top=top)
    return dict(binary_free=binary,ternary_free=free,ternary_c3=constrained)


def make_mixed_clebsch_amplitude(config: MixedClebschConfig) -> Callable:
    """Return raw(bank,bits) with exact doubled root q=m=0.

    Physical bits set every subtree m; no learned magnetic-index operation
    or configuration-dependent truncation is applied. Space projection is
    attached outside this shared pure-JAX kernel.
    """
    tree=triangular_schur_hierarchy(config.length)
    tables=tuple(jnp.asarray(triangle_block_tables()['nine'][q]) for q in config.local_q)
    sites=tuple(jnp.asarray(block.sites) for block in tree.blocks)
    layout=mixed_clebsch_layout(config)
    powers=jnp.asarray(1<<np.arange(8,-1,-1))

    def amplitude(bank: dict,bits: jax.Array) -> jax.Array:
        bits=jnp.asarray(bits)
        if bits.shape!=(tree.n_sites,) or not jnp.issubdtype(bits.dtype,jnp.integer):
            raise ValueError('one physical integer-bit configuration required')
        valid=jnp.all((bits==0)|(bits==1));bits=jnp.clip(bits,0,1)
        values=[];magnetic=[]
        for i,ids in enumerate(sites):
            local=bits[ids];packed=jnp.dot(local,powers)
            values.append({q:table[packed]@w for q,table,w in zip(config.local_q,tables,bank['local'][i],strict=True)})
            magnetic.append(jnp.sum(2*local-1))
        for index,row in enumerate(layout):
            left,right=values[row['left']],values[row['right']]
            ml,mr=magnetic[row['left']],magnetic[row['right']]
            out={q:jnp.zeros((width,),jnp.float64) for q,width in row['widths'].items()}
            for group,weights in zip(row['groups'],bank['top'][index],strict=True):
                l=jnp.stack([left[ql] for ql,_,_ in group['keys']])
                r=jnp.stack([right[qr] for _,qr,_ in group['keys']])
                sl,sr=group['left_span'],group['right_span']
                cg=jnp.asarray(group['cg'])[:,jnp.clip(ml,-sl,sl)+sl,jnp.clip(mr,-sr,sr)+sr]
                cg=jnp.where((abs(ml)<=sl)&(abs(mr)<=sr),cg,0.)
                terms=jnp.einsum('ca,cb,ckab->ck',l,r,weights)*cg[:,None]
                for q,ids in group['output_groups']:
                    out[q]=out[q]+jnp.sum(terms[jnp.asarray(ids)],axis=0)
            values.append(out);magnetic.append(ml+mr)
        return jnp.where(valid & (magnetic[-1]==0),values[-1][0][0],0.)
    return amplitude


def mixed_parameter_count(params: dict) -> int:
    """Independent stored real scalar count, including binary map gauges."""
    return sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))


def make_mixed_projected_amplitude(config: MixedClebschConfig, images: np.ndarray,
                                   *, chunk: int = 4) -> Callable:
    """Coherent physical A1 mean over explicitly supplied complete images.

    Every image has unit character. The caller supplies the verified lattice
    group; this function never infers global symmetry from a local C3 map.
    Rematerialization bounds reverse-pass storage across projection images.
    """
    maps=np.asarray(images)
    n=config.length**2
    if (maps.ndim!=2 or maps.shape[1]!=n or not np.issubdtype(maps.dtype,np.integer)
            or not np.all(np.sort(maps,axis=1)==np.arange(n))
            or isinstance(chunk,bool) or not isinstance(chunk,int) or chunk<1 or len(maps)%chunk):
        raise ValueError('physical permutations and a positive dividing projection chunk required')
    raw=make_mixed_clebsch_amplitude(config)
    groups=jnp.asarray(maps.reshape(-1,chunk,n))
    def amplitude(bank: dict,bits: jax.Array) -> jax.Array:
        def partial(group):
            return jnp.sum(jax.vmap(raw,in_axes=(None,0))(bank,bits[group]))
        return jnp.sum(jax.lax.map(jax.checkpoint(partial),groups))/len(maps)
    return amplitude
