"""One physical cross-nine-block gate on a frozen CT12 merge initializer.

Exact raw144 energy is obtained from changed physical18 block operators and
unchanged ancestors. No coherentA1 norm/energy, H-squared, variance or new
production state is inferred. Physicalm0/m2 slices avoid a full18 tensor.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import signal
from time import perf_counter,process_time

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.operators.physical_precompression import cyclic_three_spin_action
from su2_nqs.operators.su2_sector_merge import compact_reduced_block,validate_reduced_block
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.triangle_blocks import triangle_block_tables
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint
from .triangle_boundary_response_enrichment import TREE
from .triangle_boundary_response_enrichment import active_environment
from .triangle_hierarchy_initialization import split_bond_ledger
from .triangle_lower_path import reproject
from .triangle_sector_transfer import load_source

ROOT=Path(__file__).resolve().parents[3]


def physical_slice(state,source,m):
    """Exact saved18 multiplet columns at one doubledm, no truncation."""
    n=18;bits=np.array([i for i in range(1<<n) if 2*i.bit_count()-n==m],np.int64)
    index=np.full(1<<n,-1,np.int64);index[bits]=np.arange(len(bits))
    bits9=np.arange(512);mag9=np.array([2*i.bit_count()-9 for i in bits9])
    table=triangle_block_tables()['nine'];widths=state['blocks'][16].q_widths
    children=[]
    for leaf in (0,1):
        children.append({q:table[q]@np.asarray(source['local'][leaf][k])
                         for k,q in enumerate((1,3,5,7,9))})
    result={q:np.zeros((len(bits),d),float) for q,d in widths.items() if q>=abs(m)}
    for q,output in result.items():
        rows=state['channels'][0][q];W=state['maps'][0][q]
        for ml in range(-9,10,2):
            mr=m-ml
            if abs(mr)>9:continue
            li=np.flatnonzero(mag9==ml);ri=np.flatnonzero(mag9==mr)
            positions=index[(li[:,None]*512+ri).ravel()]
            block=np.zeros((len(li),len(ri),W.shape[1]))
            for ql in children[0]:
                if ql<abs(ml):continue
                for qr in children[1]:
                    if qr<abs(mr):continue
                    ids=[i for i,(a,_,b,_) in enumerate(rows) if a==ql and b==qr]
                    if not ids:continue
                    cg=clebsch_gordan(ql,ml,qr,mr,q,m)
                    if cg==0:continue
                    left=children[0][ql][li];right=children[1][qr][ri]
                    C=W[ids].reshape(left.shape[1],right.shape[1],-1)*cg
                    block+=np.einsum('ia,abk,jb->ijk',left,C,right,optimize=True)
            output[positions]=block.reshape(-1,W.shape[1])
    whole=np.column_stack(list(result.values()))
    defect=float(np.max(abs(whole.T@whole-np.eye(whole.shape[1]))))
    if defect>2e-10:raise ArithmeticError('actual physical18 slice lost common multiplet isometry')
    return bits,result,defect


def physical_h_action(vectors,bits,bonds):
    """Exact S_i dot S_j = P_ij/2-I/4 in a complete18-spin msector."""
    out=np.zeros_like(vectors)
    for i,j,g in bonds:
        mi=1<<(17-i);mj=1<<(17-j)
        different=((bits&mi)!=0)!=((bits&mj)!=0)
        swapped=bits^np.where(different,mi|mj,0)
        pos=np.searchsorted(bits,swapped)
        if np.any(pos==len(bits)) or not np.array_equal(bits[pos],swapped):
            raise ArithmeticError('Hamiltonian row support incomplete')
        out+=g*(.5*vectors[pos]-.25*vectors)
    return out


def retained_child_capture(vectors,bits,source,*,return_matrices=False):
    """Exact squared projection onto OLD9x9 spaces, not only old18 W."""
    table=triangle_block_tables()['nine'];b9=np.arange(512)
    mag=np.array([2*i.bit_count()-9 for i in b9]);index=np.full(1<<18,-1,np.int64)
    index[bits]=np.arange(len(bits));capture={q:np.zeros((v.shape[1],v.shape[1])) for q,v in vectors.items()}
    for ml in range(-9,10,2):
        li=np.flatnonzero(mag==ml);ri=np.flatnonzero(mag==-ml)
        bases=[]
        for leaf,rows,m in ((0,li,ml),(1,ri,-ml)):
            bases.append(np.column_stack([(table[q]@np.asarray(source['local'][leaf][k]))[rows]
                for k,q in enumerate((1,3,5,7,9)) if q>=abs(m)]))
        pos=index[(li[:,None]*512+ri).ravel()]
        for q,v in vectors.items():
            piece=v[pos].reshape(len(li),len(ri),v.shape[1])
            c=np.einsum('ia,ijk,jb->abk',bases[0],piece,bases[1],optimize=True)
            c=c.reshape(-1,v.shape[1]);capture[q]+=c.T@c
    if return_matrices:return capture
    return {q:float(np.trace(c).real)/len(c) for q,c in capture.items()}


def run(output:Path,seconds_cap:float=900.):
    output.mkdir(parents=True,exist_ok=False)
    parent=ROOT/'projects/clebsch_tree_nqs/results/ct_lower_path_20261026/nested18/state.pkl'
    source,provenance=load_source(.125);state=load_checkpoint(parent)['params']
    triple=(133,2,134);support=TREE.coarse_supports[16]
    sites=tuple(support.index(s) for s in triple);position={s:i for i,s in enumerate(support)}
    _,joins,bonds=split_bond_ledger(12,.125)
    internal=[(position[i],position[j],g) for i,j,g in bonds if i in position and j in position]
    touched=[(i,j,g) for i,j,g in internal if i in sites or j in sites]
    registration=dict(parent=str(parent),parent_sha256=file_sha256(parent),source=provenance,
        physical_sites=triple,ordered_local_sites=sites,angles=[-.06,-.03,0.,.03,.06],
        generator='P_ij P_jk-P_jk P_ij; NNtriangle spanningtwo9blocks, physical before child compression',
        old_multiplets_whole=True,only_changed_node=16,all_ancestors_W_frozen=True,
        changed_operator_ancestors=[16,24,28,30],maximum_seconds=seconds_cap,
        raw_energy_only=True,coherent_A1_energy_known=False,physical_variance_known=False,
        full_H_MC_queries=0,source_sha256={str(p):file_sha256(p) for p in
            (Path(__file__),ROOT/'su2_nqs/operators/physical_precompression.py')})
    atomic_json(output/'registration.json',registration)
    out=dict(registration=registration,status='running',arms={});started=perf_counter();cpu=process_time()
    def timeout(*_):raise TimeoutError('bounded physical precompression probe exhausted')
    old=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,seconds_cap)
    try:
        b0,U0,e0=physical_slice(state,source,0);b2,U2,e2=physical_slice(state,source,2)
        out['physical_slice_checks']=dict(Gram_error=max(e0,e2),m0_rows=len(b0),m2_rows=len(b2),
            m0_columns=sum(u.shape[1] for u in U0.values()),m2_columns=sum(u.shape[1] for u in U2.values()))
        block=compact_reduced_block(state['blocks'][16]);before=float(state['blocks'][30].h_by_q[0][0,0].real)
        parent_h={q:U.T@physical_h_action(U,b0,internal) for q,U in U0.items()}
        error=max(float(np.max(abs(parent_h[q]-block.h_by_q[q]))) for q in U0)
        if error>2e-9:raise ArithmeticError('physical18 H replay differs from saved projected block')
        out['parent_internal_H_error']=error;out['raw144_parent_energy_per_site']=before/144
        T0={q:physical_h_action(u,b0,touched) for q,u in U0.items()}
        for theta in registration['angles']:
            tick=perf_counter()
            V0={q:cyclic_three_spin_action(u,b0,n_sites=18,sites=sites,theta=theta) for q,u in U0.items()}
            V2={q:cyclic_three_spin_action(u,b2,n_sites=18,sites=sites,theta=theta) for q,u in U2.items()}
            newh={q:np.asarray(block.h_by_q[q])+v.T@physical_h_action(v,b0,touched)-U0[q].T@T0[q]
                  for q,v in V0.items()}
            spins=[dict(s) for s in block.site_reduced_spin]
            replay=0.
            for site in sites:
                new={}
                for qo in block.q_widths:
                    for qi in block.q_widths:
                        if abs(qo-qi)>2 or qo+qi<2:continue
                        m=2 if qo==qi else 0;vectors=V2 if m==2 else V0;rows=b2 if m==2 else b0
                        cg=clebsch_gordan(qi,m,2,0,qo,m)
                        z=(((rows>>(17-site))&1)-.5)
                        value=vectors[qo].T@(z[:,None]*vectors[qi])/cg
                        new[qo,qi]=value
                        if theta==0:replay=max(replay,float(np.max(abs(value-spins[site][qo,qi]))))
                spins[site]=new
            changed=replace(block,h_by_q=newh,site_reduced_spin=tuple(spins))
            check=validate_reduced_block(changed)
            candidate=deepcopy(state);candidate['blocks'][16]=changed
            for node in (8,12,14):reproject(candidate,node,joins)
            energy=float(candidate['blocks'][30].h_by_q[0][0,0].real)
            gram=max(float(np.max(abs(v.T@v-np.eye(v.shape[1])))) for v in V0.values())
            # A changed local18 span cannot be absorbed by changing only W18
            # in the old retained9x9 source if this separate projector leaks.
            # Full old18 projector leakage is weaker: W18 rotation alone may
            # explain it, hence record it with exactly that limited meaning.
            old18_capture={q:float(np.linalg.norm(U0[q].T@v)**2/v.shape[1]) for q,v in V0.items()}
            child_capture=retained_child_capture(V0,b0,source)
            out['arms'][str(theta)]=dict(raw144_energy_per_site=energy/144,
                raw144_delta_energy_per_site=(energy-before)/144,physical18_Gram_error=gram,
                changed_boundary_validation=check,parent_boundary_replay_error=replay,
                old18_subspace_capture=old18_capture,old9_product_subspace_capture=child_capture,
                old9_product_mean_leakage={q:1-c for q,c in child_capture.items()},
                wall_seconds=perf_counter()-tick,
                number_of_changed_physical_boundary_spins=len(sites))
            if theta==0 and (abs(energy-before)>2e-8 or replay>2e-9):raise ArithmeticError('identity gate replay failed')
            save_checkpoint(output/('angle_'+str(theta).replace('.','p')+'.pkl'),
                dict(parent=str(parent),parent_sha256=file_sha256(parent),physical_sites=triple,theta=theta,
                    changed_physical_block=changed),metadata=dict(raw_only=True,
                        requires_physical_gate_decoder=True,not_native_CT_bank=True,registration=registration))
            atomic_json(output/'progress.json',out);print(theta,energy/144,round(perf_counter()-tick,3),flush=True)
        out['status']='complete_exact_raw144_physical_gate_probe_not_A1_audit'
    except TimeoutError as e:out.update(status='bounded_partial',error=str(e))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        out.update(wall_seconds=perf_counter()-started,CPU_seconds=process_time()-cpu,
                   full_A1_H_queries=0,new_MC_records=0)
        atomic_json(output/'results.json',out)
    return out


def review_occupation(directory:Path):
    """ZeroH follow-up: actual raw144 occupation-weighted child-space loss.

    The original run's registered source snapshots remain untouched. This
    numerical helper revision adds a projector matrix return, and fixes only
    unsigned row validation in the scalar gate. Both revisions are sealed.
    """
    import json
    result=json.loads((directory/'results.json').read_text());reg=result['registration']
    source,_=load_source(.125);state=load_checkpoint(reg['parent'])['params']
    environment,nodes,_=active_environment(state,source)
    sigma={q:(q+1)*np.asarray(e)/environment.norm_squared
           for q,e in environment.downward[nodes[16]].items()}
    if abs(sum(np.trace(s).real for s in sigma.values())-1)>2e-11:
        raise ArithmeticError('raw18 occupation does not normalize')
    started=perf_counter();bits,U,_=physical_slice(state,source,0)
    out=dict(parent_sha256=file_sha256(Path(reg['parent'])),raw18_sector_probabilities=
        {q:float(np.trace(s).real) for q,s in sigma.items()},arms={},
        full_A1_H_queries=0,raw_H_queries=0,new_MC_records=0,
        full_A1_novelty_proved=False,only_old_physical_leaf_product_projector=True,
        source_sha256={str(p):file_sha256(p) for p in
            (Path(__file__),ROOT/'su2_nqs/operators/physical_precompression.py')})
    for theta in reg['angles']:
        V={q:cyclic_three_spin_action(u,bits,n_sites=18,sites=reg['ordered_local_sites'],theta=theta)
           for q,u in U.items()}
        captured=retained_child_capture(V,bits,source,return_matrices=True)
        old18={q:(U[q].T@v).T@(U[q].T@v) for q,v in V.items()}
        child_weight=float(sum(np.trace(captured[q]@s).real for q,s in sigma.items()))
        old18_weight=float(sum(np.trace(old18[q]@s).real for q,s in sigma.items()))
        out['arms'][str(theta)]=dict(raw144_weight_outside_old9_product=1-child_weight,
            raw144_weight_outside_old18_subspace=1-old18_weight,
            q_resolved_lost_weight={q:float(np.trace((np.eye(len(s))-captured[q])@s).real)
                                   for q,s in sigma.items()})
        if theta==0 and abs(1-child_weight)>2e-11:raise ArithmeticError('identity child-space leakage is notzero')
    out['wall_seconds']=perf_counter()-started
    atomic_json(directory/'occupation_review.json',out)
    return out


def review_double_discarded(directory:Path):
    """Analytic physical gate tangent outside the COMPLETE old rawCT tangent.

    Every first derivative of any old tree tensor has at least one of the
    two9 blocks in its old retained physical space. Their double-complement
    projector therefore annihilates the full rawW tangent. This exact raw
    certificate does NOT survive a Reynolds sum by the same argument.
    """
    import json
    reg=json.loads((directory/'results.json').read_text())['registration']
    source,_=load_source(.125);state=load_checkpoint(reg['parent'])['params']
    env,nodes,_=active_environment(state,source);bits,U,_=physical_slice(state,source,0)
    sigma={q:(q+1)*np.asarray(s)/env.norm_squared for q,s in env.downward[nodes[16]].items()}
    angle=np.pi/(2*np.sqrt(3));dV={q:np.sqrt(3)/2*(
        cyclic_three_spin_action(u,bits,n_sites=18,sites=reg['ordered_local_sites'],theta=angle)
        -cyclic_three_spin_action(u,bits,n_sites=18,sites=reg['ordered_local_sites'],theta=-angle))
        for q,u in U.items()}
    table=triangle_block_tables()['nine'];b9=np.arange(512)
    mag=np.array([2*i.bit_count()-9 for i in b9]);index=np.full(1<<18,-1,np.int64)
    index[bits]=np.arange(len(bits));grams={q:np.zeros((u.shape[1],u.shape[1])) for q,u in U.items()}
    parent_error=0.;orthogonality=0.
    for ml in range(-9,10,2):
        li=np.flatnonzero(mag==ml);ri=np.flatnonzero(mag==-ml)
        bases=[]
        for leaf,rows,m in ((0,li,ml),(1,ri,-ml)):
            bases.append(np.column_stack([(table[q]@np.asarray(source['local'][leaf][k]))[rows]
                for k,q in enumerate((1,3,5,7,9)) if q>=abs(m)]))
        L,R=bases;pos=index[(li[:,None]*512+ri).ravel()]
        for q,derivative in dV.items():
            x=derivative[pos].reshape(len(li),len(ri),derivative.shape[1])
            pl=np.einsum('ia,ja,jbk->ibk',L,L,x,optimize=True)
            pr=np.einsum('ijk,jb,lb->ilk',x,R,R,optimize=True)
            plpr=np.einsum('ia,ja,jbk->ibk',L,L,pr,optimize=True)
            qq=x-pl-pr+plpr;flat=qq.reshape(-1,x.shape[-1]);grams[q]+=flat.T@flat
            orthogonality=max(orthogonality,float(np.max(abs(np.einsum('ia,ijk->ajk',L,qq,optimize=True)))),
                              float(np.max(abs(np.einsum('ijk,jb->ibk',qq,R,optimize=True)))))
            original=U[q][pos].reshape(x.shape)
            old_qq=original-np.einsum('ia,ja,jbk->ibk',L,L,original,optimize=True)
            parent_error=max(parent_error,float(np.max(abs(old_qq))))
    parts={q:float(np.trace(g@sigma[q]).real) for q,g in grams.items()}
    total=sum(float(np.trace(dV[q].T@dV[q]@sigma[q]).real) for q in dV)
    result=dict(parent_sha256=file_sha256(Path(reg['parent'])),
        actual_raw144_double_discarded_tangent_norm_squared=sum(parts.values()),
        actual_raw144_full_gate_tangent_norm_squared=total,
        double_discarded_fraction_of_unprojected_gate_tangent=sum(parts.values())/total,
        q_resolved_double_discarded_norm_squared=parts,
        parent_in_retained9_support_error=parent_error,QQ_projector_orthogonality_error=orthogonality,
        raw_all_W_tangent_novelty=True,full_A1_tangent_novelty_proved=False,
        analytic_derivative_not_small_step_difference=True,full_A1_H_queries=0,raw_H_queries=0,new_MC_records=0,
        source_sha256={str(p):file_sha256(p) for p in
            (Path(__file__),ROOT/'su2_nqs/operators/physical_precompression.py')})
    if sum(parts.values())<=1e-10 or max(parent_error,orthogonality)>1e-10:
        raise ArithmeticError('double-discarded raw tangent certificate failed')
    atomic_json(directory/'double_discarded_review.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seconds-cap',type=float,default=900.);p.add_argument('--review-occupation',action='store_true')
    p.add_argument('--review-double-discarded',action='store_true');a=p.parse_args()
    if a.review_double_discarded:review_double_discarded(a.output)
    elif a.review_occupation:review_occupation(a.output)
    else:run(a.output,a.seconds_cap)
