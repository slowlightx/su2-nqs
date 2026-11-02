"""Opt-in physical cross-branch cycle, fused at its eighteen-site subtree.

The three physical permutations act before nine-site compression. Their
amplitudes are added at the eighteen-site message, BEFORE contracting
unchanged ancestors. This MAY extend a finite compressed family when the
gate crosses incomplete retained child spaces. With complete children, or
an absorbable gate, it can instead be a reparameterization of the old W.
Neither nonzero amplitudes nor this decoder alone prove tangent novelty,
especially after coherent lattice projection.

There is no Hamiltonian or sampling logic here. The same factory supplies raw
and exchanged amplitudes to the existing complete coherent A1/H driver.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from .triangular_clebsch_active_layout import mixed_active_layout
from .triangular_clebsch_exchange_cache import exchange_query_plan
from .triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout


def make_cycle_gated_exchange_queries(
    config: MixedClebschConfig, pairs: Sequence, ordered_sites: Sequence[int], *,
    active_top_widths: Sequence[Mapping[int, int]] | None = None,
) -> tuple[Callable, dict]:
    """Return exact query(bank,bits)->(raw,swaps) for one physical cyclic gate.

    ``bank`` is the differentiable feature-LUT bank plus scalar ``cycle_theta``.
    The gate generator is P_ij P_jk-P_jk P_ij, with the ordered three sites
    crossing the two nine-site children of ONE eighteen-site subtree. Each
    fixed doubled-q multiplet receives the same invariant physical operation.
    The angle remains a JAX input, so its derivative and W derivatives coexist.

    Unaffected messages are reused for each requested physical exchange. All
    old sources, query ordering, complete magnetic components and CG phases
    are unchanged. Optional active widths have fixed-state-only semantics.
    """
    tree=triangular_schur_hierarchy(config.length)
    triple=tuple(ordered_sites)
    if (len(triple)!=3 or len(set(triple))!=3 or any(isinstance(i,bool)
            or not isinstance(i,Integral) or not 0<=i<tree.n_sites for i in triple)):
        raise ValueError('three distinct ordered integer physical sites required')
    layout=(mixed_clebsch_layout(config) if active_top_widths is None
            else mixed_active_layout(config,active_top_widths))
    groups=exchange_query_plan(config,pairs)
    owner={s:i for i,block in enumerate(tree.blocks) for s in block.sites}
    leaves={owner[i] for i in triple}
    target=[k for k,row in enumerate(layout) if row['left']<len(tree.blocks)
            and row['right']<len(tree.blocks) and leaves=={row['left'],row['right']}]
    if len(target)!=1:
        raise ValueError('cycle must cross exactly the two nine-site children of one eighteen-site subtree')
    target=target[0];target_row=layout[target]
    physical_sites=tuple(jnp.asarray(block.sites) for block in tree.blocks)
    powers=jnp.asarray(1<<np.arange(8,-1,-1));sites_j=jnp.asarray(triple)

    def local_message(bank,bits,index):
        values=bits[physical_sites[index]];packed=values@powers
        return ({q:table[packed] for q,table in zip(config.local_q,bank['local_features'][index],strict=True)},
                jnp.sum(2*values-1))

    def plain_merge(bank,left,right,ml,mr,index):
        row=layout[index]
        out={q:jnp.zeros((width,),jnp.float64) for q,width in row['widths'].items()}
        for group,weights in zip(row['groups'],bank['top'][index],strict=True):
            lv=jnp.stack([left[ql] for ql,_,_ in group['keys']])
            rv=jnp.stack([right[qr] for _,qr,_ in group['keys']])
            sl,sr=group['left_span'],group['right_span']
            cg=jnp.asarray(group['cg'])[:,jnp.clip(ml,-sl,sl)+sl,jnp.clip(mr,-sr,sr)+sr]
            cg=jnp.where((abs(ml)<=sl)&(abs(mr)<=sr),cg,0.)
            terms=jnp.einsum('ca,cb,ckab->ck',lv,rv,weights)*cg[:,None]
            for q,ids in group['output_groups']:
                out[q]=out[q]+jnp.sum(terms[jnp.asarray(ids)],axis=0)
        return out

    def merge(bank,bits,values,magnetic,index):
        row=layout[index];left,right=row['left'],row['right']
        out=plain_merge(bank,values[left],values[right],magnetic[left],magnetic[right],index)
        if index==target:
            theta=jnp.asarray(bank['cycle_theta'])
            if theta.ndim!=0:raise ValueError('cycle_theta must be scalar')
            a=jnp.sin(jnp.sqrt(3.)*theta)/jnp.sqrt(3.)
            b=(1-jnp.cos(jnp.sqrt(3.)*theta))/3
            out={q:(1-2*b)*v for q,v in out.items()}
            for order,coefficient in (((1,2,0),a+b),((2,0,1),b-a)):
                changed=bits.at[sites_j].set(bits[sites_j[jnp.asarray(order)]])
                lv,ml=local_message(bank,changed,left);rv,mr=local_message(bank,changed,right)
                branch=plain_merge(bank,lv,rv,ml,mr,index)
                out={q:v+coefficient*branch[q] for q,v in out.items()}
        return out,magnetic[left]+magnetic[right]

    def query(bank:dict,bits:jax.Array) -> tuple[jax.Array,jax.Array]:
        bits=jnp.asarray(bits)
        if bits.shape!=(tree.n_sites,) or not jnp.issubdtype(bits.dtype,jnp.integer):
            raise ValueError('one physical integer-bit configuration required')
        valid=jnp.all((bits==0)|(bits==1));bits=jnp.clip(bits,0,1)
        values=[];magnetic=[]
        for i in range(len(tree.blocks)):
            value,m=local_message(bank,bits,i);values.append(value);magnetic.append(m)
        for k in range(len(layout)):
            value,m=merge(bank,bits,values,magnetic,k);values.append(value);magnetic.append(m)
        legal=valid&(magnetic[-1]==0);raw=jnp.where(legal,values[-1][0][0],0.)
        result=jnp.zeros((len(pairs),),raw.dtype)
        for group in groups:
            def exchanged(pair):
                a,b=pair;changed=bits.at[a].set(bits[b]).at[b].set(bits[a])
                updated=list(values);updated_m=list(magnetic)
                for i in group['leaves']:
                    updated[i],updated_m[i]=local_message(bank,changed,i)
                for k in group['ancestors']:
                    row=layout[k]
                    updated[row['parent']],updated_m[row['parent']]=merge(bank,changed,updated,updated_m,k)
                return jnp.where(legal,updated[-1][0][0],0.)
            result=result.at[jnp.asarray(group['indices'])].set(jax.vmap(exchanged)(jnp.asarray(group['pairs'])))
        return raw,result

    metadata=dict(physical_sites=list(map(int,triple)),gated_merge_index=target,
        gated_parent=target_row['parent'],gated_children=[target_row['left'],target_row['right']],
        finite_gate='exp(theta*(PijPjk-PjkPij)); three permutations fused at18 message',
        angle_bank_key='cycle_theta',coherent_before_upper_contraction=True,
        old_child_product_compression_bypassed=True,physical_gate_before_compression=True,
        may_extend_finite_compressed_family=True,universal_family_extension_guaranteed=False,
        full_A1_tangent_novelty_certified=False,
        no_H_or_sampling_logic=True,pairs=len(pairs),exchange_groups=len(groups),
        fixed_bank_active_layout=active_top_widths is not None)
    return query,metadata
