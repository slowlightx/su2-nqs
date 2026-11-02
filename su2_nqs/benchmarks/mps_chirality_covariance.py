"""Same physical row-chirality C/M matrices from a finite MPS.

The ordinary product is a physical MPO, not a product of operators compressed
into the Schmidt support. No canonicalization, symmetry repair or dense RDM.
"""
from __future__ import annotations

from collections import defaultdict
from numbers import Integral
from typing import Any

import numpy as np

from .chirality_covariance import PAIRS,normalized_coordinates,symmetric_matrix
from .mps_rotation_covariance import finite_mps_prefix_rotation_covariance,charged_replica_matrix


def chirality_terms(edges: tuple[tuple[int,int],...]) -> dict:
    """Return exact ordered-site terms of kappa_z in doubled-spin conventions."""
    if (not edges or any(len(e)!=2 or e[0]==e[1]
            or any(isinstance(i,bool) or not isinstance(i,Integral) or i<0 for i in e) for e in edges)):
        raise ValueError('nonempty oriented physical bonds required')
    terms=defaultdict(complex)
    for a,b in edges:
        terms[tuple(sorted(((a,'Sp'),(b,'Sm'))))]+=.5j
        terms[tuple(sorted(((a,'Sm'),(b,'Sp'))))]-=.5j
    return {k:v for k,v in sorted(terms.items()) if v!=0}


def symmetrized_product_terms(left: dict,right: dict) -> dict:
    """Combine {K_left,K_right}/2, preserving on-site multiplication order."""
    combined=defaultdict(complex)
    for first,second in ((left,right),(right,left)):
        for a,ca in first.items():
            for b,cb in second.items():
                sites=defaultdict(list)
                for i,op in (*a,*b):sites[i].append(op)
                if any(len(ops)==2 and ops[0]==ops[1] for ops in sites.values()):
                    continue  # Spin-half (Sp)^2=(Sm)^2=0, before MPO allocation.
                key=tuple((i,' '.join(ops)) for i,ops in sorted(sites.items()))
                combined[key]+=.5*ca*cb
    return {k:v for k,v in sorted(combined.items()) if v!=0}


def finite_mps_direct_chirality_replica(psi: Any,cut: int,
        edges: tuple[tuple[tuple[int,int],...],...],*,tolerance: float=1e-9) -> dict:
    """Three physical charge-zero Grams, with no Casimir/Schmidt side work.

    This is the identical full-support M, not a Schmidt truncation or a new
    observable. The primitive transfer and cut closure match the separately
    validated older general rotation reader.
    """
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mps import MPSEnvironment
    if psi.bc!='finite' or len(edges)!=3 or not 0<cut<psi.L:
        raise ValueError('three legal finite-prefix chirality groups required')
    for group in edges:
        chirality_terms(group)
        if any(i>=cut for edge in group for i in edge):raise ValueError('operator outside prefix')
    names=list(psi.chinfo.names)
    if names.count('2*Sz')!=1:raise ValueError('explicit physical doubled Sz conservation required')
    index=names.index('2*Sz')
    if int(psi.chinfo.mod[index])!=1:raise ValueError('ordinary doubled Sz, not modular charge, required')
    for i,site in enumerate(psi.sites):
        leg=psi.get_B(i,form=None).get_leg('p');sz=np.asarray(site.get_op('Sz').to_ndarray())
        if (leg.qconj!=1 or sz.shape!=(2,2)
                or not np.allclose(sz,np.diag(np.diag(sz)),atol=tolerance,rtol=0)
                or not np.allclose(leg.to_qflat()[:,index],2*np.diag(sz),atol=tolerance,rtol=0)
                or set(leg.to_qflat()[:,index].tolist())!={-1,1}):
            raise ValueError('actual doubled spin-half physical labels required')
    env=MPSEnvironment(psi,psi);identity=env.init_LP(0)
    grams=[identity.astype(complex)*0. for _ in edges];active={}
    bonds=[(name,min(a,b),max(a,b),1 if a<b else -1)
           for name,group in enumerate(edges) for a,b in group]
    def transfer(i,lp,op=None):
        if op is None:return env._contract_LP(i,lp)
        value=npc.tensordot(lp,psi.get_B(i,form='A'),axes=('vR','vL'))
        value=npc.tensordot(op,value,axes=('p*','p'))
        return npc.tensordot(psi.get_B(i,form='A').conj(),value,
            axes=(['p*','vL*'],['p','vR*']))
    for i in range(cut):
        site=psi.sites[i];grams=[transfer(i,g) for g in grams];next_active={}
        for key,(sp,sm) in active.items():
            name,_,last,orientation=bonds[key]
            if i==last:
                grams[name]+=.5j*orientation*(transfer(i,sp,site.get_op('Sm'))
                                               -transfer(i,sm,site.get_op('Sp')))
            else:next_active[key]=(transfer(i,sp),transfer(i,sm))
        for key,(_,first,_,_) in enumerate(bonds):
            if first==i:next_active[key]=(transfer(i,identity,site.get_op('Sp')),
                                         transfer(i,identity,site.get_op('Sm')))
        active=next_active;identity=transfer(i,identity)
    def close(lp):
        s=psi.get_SR(cut-1)
        if isinstance(s,npc.Array):
            lp=npc.tensordot(s.conj(),lp,axes=['vL*','vR*'])
            return npc.tensordot(lp,s,axes=['vR','vL'])
        return lp.scale_axis(np.asarray(s).conj(),'vR*').scale_axis(s,'vR')
    l=close(identity).transpose(['vR*','vR']).to_ndarray()
    r=env.get_RP(cut-1,store=False).transpose(['vL','vL*']).to_ndarray()
    g={str(i):close(a).transpose(['vR*','vR']).to_ndarray() for i,a in enumerate(grams)}
    virtual=psi.get_B(cut,form=None).get_leg('vL')
    left=psi.get_B(0,form=None).get_leg('vL');right=psi.get_B(psi.L-1,form=None).get_leg('vR')
    if virtual.qconj!=1 or left.qconj!=1 or right.qconj!=-1 or left.ind_len!=1 or right.ind_len!=1:
        raise ValueError('explicit finite boundary charges required')
    offset=sum(int(psi.get_B(i,form=None).qtotal[index]) for i in range(cut))
    offset-=int(psi.get_B(0,form=None).get_leg('vL').to_qflat()[0,index])
    charges=virtual.to_qflat()[:,index]+offset
    norm=complex(np.einsum('ij,ji->',l,r))
    if norm.real<=0 or abs(norm.imag)>tolerance*norm.real:raise ValueError('physical norm unresolved')
    purity=0j
    for q in np.unique(charges):
        ids=np.flatnonzero(charges==q);select=np.ix_(ids,ids)
        rl=r[select]@l[select];purity+=np.einsum('ij,ji->',rl,rl)
    purity/=norm.real**2
    if abs(purity.imag)>tolerance or not 0<purity.real<=1+tolerance:
        raise ValueError('physical purity unresolved')
    for value in g.values():
        if np.linalg.norm(value-value.conj().T)>tolerance*max(np.linalg.norm(value),1e-300):
            raise ValueError('physical chirality Gram lost Hermiticity')
    matrix=charged_replica_matrix(r,g,charges,delta_m=0)/(norm.real**2*purity.real)
    if np.max(abs(matrix.imag))>tolerance:raise ValueError('Hermitian replica matrix has imaginary remainder')
    return dict(replica_matrix=matrix.real.tolist(),purity=float(purity.real),
        physical_norm_without_external_mps_norm=float(norm.real),
        replica_method='three_direct_charge_zero_Grams',full_schmidt_support=True)


def finite_mps_chirality_covariances(psi: Any,cut: int,
        edges: tuple[tuple[tuple[int,int],...],...],*,tolerance: float=1e-9,
        progress: Any=None,replica_method: str='legacy_six_probe',cache: dict | None=None,
        include_C2: bool=False) -> dict:
    """Measure three same-operator row sums, with their complete physical support.

    A sum-of-terms MPO is contracted only up to the prefix; its completed
    identity channel is closed with the ACTUAL complement environment and
    singular factors. This is valid for complex, noncanonical/gauged MPSs.
    """
    if (psi.bc!='finite' or len(edges)!=3 or isinstance(cut,bool) or not isinstance(cut,Integral)
            or not 0<cut<psi.L or any(i>=cut for row in edges for pair in row for i in pair)):
        raise ValueError('three physical chirality groups within a finite prefix required')
    if not isinstance(include_C2,bool):raise ValueError('include_C2 must be an explicit Boolean')
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mps import MPSEnvironment
    from tenpy.networks.mpo import MPOEnvironment,MPOGraph
    from tenpy.networks.terms import TermList
    from time import perf_counter
    start=perf_counter()
    cached={} if cache is None else dict(cache)
    if replica_method not in ('legacy_six_probe','three_direct_charge_zero_Grams'):
        raise ValueError('unknown exact replica implementation')
    if 'replica' in cached:
        replica=cached['replica'];response=cached.get('replica_probe_channels',{})
    elif replica_method=='three_direct_charge_zero_Grams':
        replica=finite_mps_direct_chirality_replica(psi,cut,edges,tolerance=tolerance);response={}
    else:
        groups={f'row{i}':e for i,e in enumerate(edges)}
        groups.update({f'rows{i}{j}':edges[i]+edges[j] for i,j in PAIRS[3:]})
        legacy=finite_mps_prefix_rotation_covariance(psi,cut,tolerance=tolerance,
            leading_levels=1,measured_charges=(0,),chirality_groups=groups,chirality_z_response=True)
        response=legacy['two_copy_chirality_z_response']['channels']
        entries=np.array([response[name]['purity_normalized'] for name in groups])
        entries[3:]=(entries[3:]-entries[[0,0,1]]-entries[[1,2,2]])/2
        replica=dict(replica_matrix=symmetric_matrix(entries).tolist(),purity=legacy['purity'],replica_method=replica_method)
    m=np.asarray(replica['replica_matrix'],float)
    if m.shape!=(3,3) or not np.isfinite(m).all():raise ValueError('invalid cached physical replica matrix')
    cached.update(replica=replica,replica_probe_channels=response)
    if progress is not None:progress(dict(stage='replica_matrix_complete',seconds=perf_counter()-start,readout_cache=cached))
    env=MPSEnvironment(psi,psi)
    rp=env.get_RP(cut-1,store=False)
    def close_cut(lp):
        s=psi.get_SR(cut-1)
        if isinstance(s,npc.Array):
            lp=npc.tensordot(s.conj(),lp,axes=['vL*','vR*'])
            return npc.tensordot(lp,s,axes=['vR','vL'])
        return lp.scale_axis(np.asarray(s).conj(),'vR*').scale_axis(s,'vR')
    def contract_closed(lp,complement):
        return complex(npc.inner(lp,complement,
            axes=[['vR*','vR'],['vL*','vL']],do_conj=False))
    identity=env.get_LP(cut,store=False)
    closed_identity=close_cut(identity)
    norm=contract_closed(closed_identity,rp)
    if norm.real<=0 or abs(norm.imag)>tolerance*norm.real:
        raise ValueError('actual noncanonical parent norm unresolved')
    escort_rp=None;purity_numerator=None
    if include_C2:
        # A physical nonorthogonal prefix is A, L=A†A, and rho=A R A†/Z.
        # Tr(rho² O)/Tr(rho²)=Tr(R L R L_O)/Tr(R L R L).
        # L_O is the SAME full physical product-MPO Gram, including action
        # outside the Schmidt support. No canonical/diagonal-s^4 assumption.
        l=closed_identity.transpose(['vR*','vR']).to_ndarray()
        ordered_rp=rp.transpose(['vL','vL*']);r=ordered_rp.to_ndarray()
        index=list(psi.chinfo.names).index('2*Sz')
        charges=psi.get_B(cut,form=None).get_leg('vL').to_qflat()[:,index]
        escort=np.zeros_like(r,dtype=complex)
        for charge in np.unique(charges):
            ids=np.flatnonzero(charges==charge);select=np.ix_(ids,ids)
            escort[select]=r[select]@l[select]@r[select]
        escort_rp=npc.Array.from_ndarray(escort,ordered_rp.legs,
            qtotal=ordered_rp.qtotal,labels=ordered_rp.get_leg_labels())
        purity_numerator=contract_closed(closed_identity,escort_rp)
        if (purity_numerator.real<=0 or abs(purity_numerator.imag)>tolerance
                or abs(purity_numerator/norm.real**2-replica['purity'])>tolerance):
            raise ValueError('escort and replica physical purity contractions disagree')
        del l,r,escort
    mpo_dimensions=[]
    def expectation(terms,renyi=False):
        if not terms:return (0j,0j) if renyi else 0j
        term_list=TermList([[(op,i) for i,op in key] for key in terms],list(terms.values()))
        mpo=MPOGraph.from_term_list(term_list,psi.sites,'finite',unit_cell_width=psi.unit_cell_width).build_MPO()
        mpo_dimensions.append(max(mpo.chi))
        mpo_env=MPOEnvironment(psi,mpo,psi)
        lp=mpo_env.init_LP(0)
        for i in range(cut):lp=mpo_env._contract_LP(i,lp)
        # The finite MPO has completed every term before the cut. Its IdR
        # channel alone continues as identity through the complement.
        lp=lp.take_slice(mpo.get_IdR(cut),'wR')
        closed=close_cut(lp)
        ordinary=contract_closed(closed,rp)/norm
        if renyi:return ordinary,contract_closed(closed,escort_rp)/purity_numerator
        return ordinary
    linear=[chirality_terms(e) for e in edges]
    cached.setdefault('means',{});cached.setdefault('products',{});cached.setdefault('renyi_products',{})
    def observable(key,terms,section):
        values=cached[section]
        renyi=include_C2 and section=='products'
        if key in values and (not renyi or key in cached['renyi_products']):
            value=complex(*values[key])
            if not np.isfinite(value):raise ValueError('nonfinite cached expectation')
            return value
        measured=expectation(terms,renyi=renyi)
        if renyi:
            value,c2=measured;cached['renyi_products'][key]=[c2.real,c2.imag]
        else:value=measured
        values[key]=[value.real,value.imag]
        if progress is not None:progress(dict(stage='physical_scalar_complete',section=section,key=key,
            seconds=perf_counter()-start,readout_cache=cached))
        return value
    means=np.array([observable(str(i),k,'means') for i,k in enumerate(linear)])
    c=[]
    for i,j in PAIRS:
        c.append(observable(str(i)+str(j),symmetrized_product_terms(linear[i],linear[j]),'products'))
        if progress is not None:progress(dict(stage='ordinary_product_complete',pair=[i,j],seconds=perf_counter()-start,readout_cache=cached))
    c=np.asarray(c)
    if max(np.max(abs(c.imag)),np.max(abs(means.imag)))>tolerance:
        raise ValueError('Hermitian physical chirality expectations have an imaginary remainder')
    raw=symmetric_matrix(c.real)
    connected=raw-np.outer(means.real,means.real)
    result=dict(ordinary_matrix=raw.tolist(),ordinary_connected_matrix=connected.tolist(),
        one_point=means.real.tolist(),one_point_imaginary=means.imag.tolist(),
        replica_matrix=m.tolist(),ordinary_coordinates=normalized_coordinates(raw).tolist(),
        replica_coordinates=normalized_coordinates(m).tolist(),
        purity=replica['purity'],physical_norm_without_external_mps_norm=norm.real,
        ordinary_product_imaginary=c.imag.tolist(),mpo_maximum_bond_dimensions=mpo_dimensions,
        exact_old_row1_replay_value=float(m[1,1]),exact_old_full18_replay_value=float(m.sum()),
        replica_probe_channels=response,wall_seconds=perf_counter()-start,
        replica_method=replica_method,resumed_cached_observables=cache is not None,
        operator_outside_schmidt_support_included=True,schmidt_operator_square_shortcut=False,
        strict_SU2_assumed=False,PSD_clipping=False,mirror_repair=False,
        C2_measured=include_C2,new_training=0,new_samples=0,new_ED=0,
        definitions=dict(C='Re <{kappa_i,kappa_j}/2>',connected='C-<kappa_i><kappa_j>',
            M='Re Tr(rho kappa_i rho kappa_j)/Tr(rho^2)'),
        interpretation='Same-operator cross-method readout, not a phase, QSL or entanglement witness.')
    if include_C2:
        c2=np.array([complex(*cached['renyi_products'][str(i)+str(j)]) for i,j in PAIRS])
        if np.max(abs(c2.imag))>tolerance:raise ValueError('Hermitian escort moment has imaginary remainder')
        c2=symmetric_matrix(c2.real);d=c2-m
        result.update(renyi_matched_matrix=c2.tolist(),commutator_metric=d.tolist(),
            raw_commutator_eigenvalues=np.linalg.eigvalsh(d).tolist(),
            escort_state='rho_A²/Tr(rho_A²), not rho_A',
            C2_definition='Re Tr(rho_A² {kappa_i,kappa_j}/2)/Tr(rho_A²)',
            nonorthogonal_escort_contraction='Tr(R L R L_O)/Tr(R L R L)',
            C2_new_MPO_queries=0)
    return result
