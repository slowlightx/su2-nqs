"""Physical crossbranch gate plus a real original-W compression direction.

The18-site compression varies in its actual fixed126-site complement.
Ancestors are coherently refactored, not independently unrestricted. Every
state in the two-dimensional center span has the ORIGINAL multiplicity
widths. Physical gate action is before the retained9x9 compression.
Raw-H proposal construction is not a full-A1 quality/variance measurement.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass,replace
from pathlib import Path
import resource
import shutil
import signal
from time import perf_counter,process_time

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.triangular_clebsch_gauge import triangular_raw_overlap
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.operators.physical_precompression import cyclic_three_spin_action
from su2_nqs.operators.su2_complement import hierarchy_singlet_complement,restore_singlet_parent
from su2_nqs.operators.su2_sector_merge import SectorMergeOperator,compact_reduced_block,validate_reduced_block
from su2_nqs.operators.su2_three_block_center import ThreeBlockSingletOperator
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint
from .triangle_boundary_response_enrichment import TREE
from .triangle_hierarchy_initialization import native_bank_from_state,split_bond_ledger
from .triangle_lower_path import clone,cross_bonds,real_maps,reproject
from .triangle_precompression_probe import physical_slice,physical_h_action
from .triangle_sector_transfer import load_source

ROOT=Path(__file__).resolve().parents[3]
CFG=MixedClebschConfig(12,20,10)


@dataclass
class _FactorView:
    """Only the testedfactor method's inputs; no preparedH or matvec cache."""
    blocks: tuple
    channels: tuple
    dimension: int
    q_indices: dict

    def factor(self,coefficients,*,widths):
        return ThreeBlockSingletOperator.factor(self,coefficients,widths=widths)


def export_frozen_center(context_path:Path,coefficients:np.ndarray,output:Path,*,theta:float=0.):
    """Export a selectedfinite same-width W state, with the physicalgate external.

    This uses exactcenterlinear combination, NOT interpolation of several
    parameter tensors. The physicalgate is never smuggled into a nativeCTbank.
    It remains mandatory metadata for the opt-in physicalgate decoder.
    """
    context=load_checkpoint(context_path)['params'];parent=context['parent'];channels=context['channels']
    c=np.asarray(coefficients)
    if c.shape!=(2,) or np.iscomplexobj(c) or not np.isfinite(c).all() or abs(np.linalg.norm(c)-1)>2e-11:
        raise ValueError('normalizedreal2state coefficientsrequired')
    if isinstance(theta,(bool,complex)) or not np.isfinite(theta):raise ValueError('finitegateanglerequired')
    E=context['complement'].complement
    op=_FactorView((parent['blocks'][0],parent['blocks'][1],E),channels,len(channels),
        {q:np.asarray([k for k,row in enumerate(channels) if row[4]==q]) for q in E.q_widths})
    vector=context['center_basis']@c;_,joins,_=split_bond_ledger(12,.125)
    state,bank,proof=restore_center(parent,context['complement'],op,vector,context['source'],joins)
    expected=float(np.vdot(c,context['rawH']@c).real)
    if abs(proof['raw_energy']-expected)>2e-8:raise ArithmeticError('selectedsamewidthrootHreplayfailed')
    save_checkpoint(output,bank,metadata=dict(model=dict(length=12,multiplicity_cap=20,maximum_q=10),
        embedding=proof['embedding'],center_context=str(context_path),context_sha256=file_sha256(context_path),
        coefficients=c.tolist(),external_physical_gate_theta=float(theta),gate_sites=[133,2,134],
        requires_opt_in_physical_gate_decoder=bool(theta!=0),raw_only=True,
        complete_A1_quality_unknown=True,independent_root_H_error=abs(proof['raw_energy']-expected)))
    return bank,proof


def freeze_raw_paths(directory:Path)->dict:
    """Freeze registeredRitz directions after exactrawFS cap.02; noHqueries.

    Three overlap values determine the EXACTsin/cos overlap of a physical
    three-cycle gate. H at a new angle is NOT inferred from these three H
    matrices. Full-A1normcalibration remains separate and bounded0<=s<=1.
    """
    import json
    report=json.loads((directory/'results.json').read_text());kernels=report['raw_kernels']
    gram=np.load(directory/'raw_center_basis.npz')['Gram'].real
    angle=.03;x=np.sqrt(3)*angle
    overlap0=np.array(kernels['0.0']['overlap_parent_to_basis'])
    plus=np.array(kernels[str(angle)]['overlap_parent_to_basis'])
    minus=np.array(kernels[str(-angle)]['overlap_parent_to_basis'])
    sine=(plus-minus)/(2*np.sin(x));cosine=(plus+minus-2*overlap0)/(2*(np.cos(x)-1))
    def overlap(theta):return overlap0+sine*np.sin(np.sqrt(3)*theta)+cosine*(np.cos(np.sqrt(3)*theta)-1)
    def distance(theta,t):
        c=np.array([np.cos(t),np.sin(t)]);value=float(overlap(theta)@c)/np.sqrt(gram[0,0]*float(c@gram@c))
        return float(np.arccos(np.clip(abs(value),0.,1.)))
    rays={}
    for key in ('0.0','-0.03','0.03'):
        h=np.array(kernels[key]['H']);values,axes=np.linalg.eigh((h+h.T)/2)
        v=axes[:,0]
        if v[0]<0:v=-v
        rays[key]=dict(theta=float(key),t=float(np.arctan2(v[1],v[0])),raw_Ritz_energy=float(values[0]))
    joint=min((rays['-0.03'],rays['0.03']),key=lambda p:p['raw_Ritz_energy'])
    out=dict(parent_sha256=report['registration']['parent_sha256'],
        source_results_sha256=file_sha256(directory/'results.json'),
        source_context_sha256=file_sha256(directory/'context.pkl'),
        choice='existingregisteredRitzrays; rawFS cap.02, thenfullA1calibrationcap.01withs<=1',
        exact_physical_three_permutation_overlap=True,H_at_new_angle_not_interpolated=True,
        raw_FS_cap=.02,full_A1_FS_target=.01,full_A1_path_scale_bounds=[0.,1.],paths={},
        raw_original_rays=rays,source_sha256=file_sha256(Path(__file__)),
        full_A1_H_queries=0,raw_H_queries=0,new_MC_records=0)
    for name,ray in (('original_W',rays['0.0']),('joint_gate_compression',joint)):
        lo=0.;hi=1.
        if distance(ray['theta'],ray['t'])<.02:raise ArithmeticError('registeredRitzraydoesnotreachrawFS cap')
        for _ in range(70):
            mid=(lo+hi)/2
            if distance(mid*ray['theta'],mid*ray['t'])<.02:lo=mid
            else:hi=mid
        scale=(lo+hi)/2;theta=scale*ray['theta'];t=scale*ray['t']
        out['paths'][name]=dict(theta=theta,t=t,raw_cap_scale=scale,
            raw_FS=distance(theta,t),raw_Ritz_direction=ray,
            future_path='U(s*theta)[cos(s*t)psi0+sin(s*t)D],0<=s<=1')
    atomic_json(directory/'frozen_raw_paths.json',out)
    return out


def restore_center(parent,complement,operator,center,source,joins):
    """Exactly restore an arbitrary9x9/E center at unchanged original widths."""
    state=clone(parent);widths=parent['blocks'][16].q_widths
    maps,coeff,proof=operator.factor(center,widths=widths)
    state['maps'][0]=real_maps(maps);details=[dict(node=16,**proof)]
    lookup={p:k for k,(_,_,p) in enumerate(TREE.binary_merges)}
    for node,side,step in reversed(complement.steps):
        if side!='left':raise ValueError('this frozen target16 path requires left-child ancestors')
        k=lookup[node];maps,coeff,proof=restore_singlet_parent(step,coeff,
            {q:w.shape[1] for q,w in state['maps'][lookup[16]].items()} if node==24 else
            {q:w.shape[1] for q,w in state['maps'][lookup[24]].items()},
            parent_widths={q:w.shape[1] for q,w in parent['maps'][k].items()})
        state['maps'][k]=real_maps(maps);details.append(dict(node=int(node),**proof))
    # target16 has ancestors24,28 and then the global singlet root30.
    state['maps'][14][0]=real_maps({0:np.array([coeff[q][a,b]
        for q,a,qr,b in parent['channels'][14][0]])[:,None]})[0]
    for node in (0,8,12,14):reproject(state,node,joins)
    bank,embedding=native_bank_from_state(source,state)
    return state,bank,dict(restoration=details,embedding=embedding,
        original18_widths_preserved={q:w.shape[1] for q,w in state['maps'][0].items()},
        raw_energy=float(state['blocks'][30].h_by_q[0][0,0].real))


def gated_physical_block(state,source,theta,cache=None):
    """Exact physical18 conjugation, not an exponential of projected spins."""
    if cache is None:
        b0,U0,g0=physical_slice(state,source,0);b2,U2,g2=physical_slice(state,source,2)
        support=TREE.coarse_supports[16];pos={s:i for i,s in enumerate(support)}
        sites=tuple(pos[i] for i in (133,2,134));bonds=split_bond_ledger(12,.125)[2]
        touched=[(pos[i],pos[j],g) for i,j,g in bonds if i in pos and j in pos
                 and (pos[i] in sites or pos[j] in sites)]
        block=compact_reduced_block(state['blocks'][16])
        T0={q:physical_h_action(u,b0,touched) for q,u in U0.items()}
        cache=dict(b0=b0,b2=b2,U0=U0,U2=U2,sites=sites,touched=touched,block=block,T0=T0,
                   physical_Gram_error=max(g0,g2))
    b0,b2,U0,U2,sites,block=(cache[k] for k in ('b0','b2','U0','U2','sites','block'))
    V0={q:cyclic_three_spin_action(u,b0,n_sites=18,sites=sites,theta=theta) for q,u in U0.items()}
    V2={q:cyclic_three_spin_action(u,b2,n_sites=18,sites=sites,theta=theta) for q,u in U2.items()}
    H={q:np.asarray(block.h_by_q[q])+v.T@physical_h_action(v,b0,cache['touched'])
         -U0[q].T@cache['T0'][q] for q,v in V0.items()}
    spins=[dict(row) for row in block.site_reduced_spin]
    for site in sites:
        row={}
        for qo in block.q_widths:
            for qi in block.q_widths:
                if abs(qo-qi)>2 or qo+qi<2:continue
                m=2 if qo==qi else 0;bits=b2 if m==2 else b0;vectors=V2 if m==2 else V0
                cg=clebsch_gordan(qi,m,2,0,qo,m)
                z=((bits>>(17-site))&1)-.5
                row[qo,qi]=vectors[qo].T@(z[:,None]*vectors[qi])/cg
        spins[site]=row
    changed=replace(block,h_by_q=H,site_reduced_spin=tuple(spins));check=validate_reduced_block(changed)
    overlap={q:U0[q].T@v for q,v in V0.items()}
    return changed,overlap,cache,check


def run(output:Path,seconds_cap:float=900.):
    """Create two real same-width states and exactraw finite-gate2x2 kernels."""
    if not 60<=seconds_cap<=900:raise ValueError('60..900second scientific limit required')
    output.mkdir(parents=True,exist_ok=False)
    parent_path=ROOT/'projects/clebsch_tree_nqs/results/ct_lower_path_20261026/nested18/state.pkl'
    files=[Path(__file__),Path(__file__).with_name('triangle_precompression_probe.py'),
        ROOT/'su2_nqs/operators/physical_precompression.py',ROOT/'su2_nqs/operators/su2_complement.py',
        ROOT/'su2_nqs/operators/su2_three_block_center.py']
    registration=dict(task='CT12_samewidth_compression_direction_plus_physical_gate_v1',
        parent=str(parent_path),parent_sha256=file_sha256(parent_path),gate_sites=[133,2,134],
        gate_angles=[-.03,0.,.03],center_target_sites=18,actual_fixed_complement_sites=126,
        changed_maps=[0,8,12,14],all_original_multiplicity_widths_unchanged=True,
        independent_unrestricted_ancestor_optimization=False,
        physical_gate_before_retained_child_compression=True,
        center_H_matvecs_planned=2,total_CPU_seconds_cap=seconds_cap,
        complete_A1_H_queries=0,new_MC_records=0,raw_H_is_not_A1=True,
        source_sha256={str(p):file_sha256(p) for p in files})
    atomic_json(output/'registration.json',registration);snap=output/'registered_sources';snap.mkdir()
    for p in files:shutil.copy2(p,snap/p.name)
    result=dict(registration=registration,status='running',stages=[],raw_kernels={})
    start=perf_counter();cpu=process_time();ops=[]
    def stage(name,**data):
        result['stages'].append(dict(stage=name,wall_seconds=perf_counter()-start,**data))
        atomic_json(output/'progress.json',result);print(name,round(perf_counter()-start,2),flush=True)
        if process_time()-cpu>seconds_cap:raise TimeoutError('scientificCPUcap reached')
    def stop(*_):raise TimeoutError('scientificwallcap reached')
    old=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,seconds_cap)
    try:
        source,provenance=load_source(.125);parent=load_checkpoint(parent_path)['params']
        _,joins,bonds=split_bond_ledger(12,.125);supports=TREE.coarse_supports
        complement=hierarchy_singlet_complement(blocks=parent['blocks'],merges=TREE.binary_merges,
            maps=parent['maps'],supports=supports,bonds=bonds,target=16)
        E=complement.complement;outside=complement.complement_sites
        op=ThreeBlockSingletOperator(parent['blocks'][0],parent['blocks'][1],E,joins[0],
            cross_bonds(supports[0],outside,bonds),cross_bonds(supports[1],outside,bonds),
            max_dimension=200_000,memory_limit_bytes=768*1024**2)
        ops.append(op);c=op.encode(parent['maps'][0],complement.coefficients)
        tick=perf_counter();hc=op.matvec(c);elapsed=perf_counter()-tick
        energy=float(np.vdot(c,hc).real);error=abs(energy-parent['blocks'][30].h_by_q[0][0,0])
        if error>2e-8:raise ArithmeticError('whole126complementH disagreeswithactualparent')
        residual=hc-energy*c;norm=float(np.linalg.norm(residual))
        if norm<1e-10:raise ArithmeticError('no compressiondirection in the declaredcenter')
        direction=-residual/norm;direction-=c*np.vdot(c,direction);direction/=np.linalg.norm(direction)
        stage('whole126_complement_direction',center_dimension=op.dimension,H_seconds=elapsed,
              parent_H_error=float(error),residual_norm=norm)
        if 1.5*elapsed+150>seconds_cap-(perf_counter()-start):
            result['status']='cost_gate_before_secondcenterH';return result
        hd=op.matvec(direction);B=np.column_stack((c,direction));HB=np.column_stack((hc,hd))
        rawH=B.conj().T@HB
        if np.max(abs(rawH-rawH.conj().T))>2e-8:raise ArithmeticError('two-statecenterHnotHermitian')
        states=[];banks=[];coefficients=[];proofs=[]
        for k,vector in enumerate((c,direction)):
            state,bank,proof=restore_center(parent,complement,op,vector,source,joins)
            discrepancy=abs(proof['raw_energy']-rawH[k,k].real)
            if discrepancy>2e-8:raise ArithmeticError('independentrawrootHfailed')
            proof['independent_root_H_error']=float(discrepancy)
            save_checkpoint(output/f'basis_{k}.pkl',bank,metadata=dict(model=dict(length=12,multiplicity_cap=20,
                maximum_q=10),embedding=proof['embedding'],registration=registration))
            save_checkpoint(output/f'state_{k}.pkl',state,metadata=dict(raw_only=True,registration=registration))
            states.append(state);banks.append(bank);proofs.append(proof)
            _,coeff,_=op.factor(vector,widths=parent['blocks'][16].q_widths);coefficients.append(coeff)
            stage('samewidth_state_'+str(k),raw_energy=proof['raw_energy'])
        gram=np.array([[triangular_raw_overlap(a,b,CFG) for b in banks] for a in banks])
        if np.max(abs(gram-np.eye(2)))>2e-10:raise ArithmeticError('restorednativebasisnotorthonormal')
        # The union is only an offlineexactkernelbasis. Physicalstates remain
        # width10/8 and refactor from the originalcenter for any coefficient.
        expanded=clone(parent);embeddings=[]
        for q in parent['maps'][0]:
            joint=np.column_stack((states[0]['maps'][0][q],states[1]['maps'][0][q]))
            frame,r=np.linalg.qr(joint,mode='reduced');expanded['maps'][0][q]=frame
            d=states[0]['maps'][0][q].shape[1]
            embeddings.append((q,r[:,:d]@coefficients[0][q],r[:,d:]@coefficients[1][q]))
        reproject(expanded,0,joins)
        doubled=SectorMergeOperator(expanded['blocks'][16],E,
            cross_bonds(supports[16],outside,bonds),0,memory_limit_bytes=768*1024**2)
        z0={q:a for q,a,b in embeddings};z1={q:b for q,a,b in embeddings}
        Z=np.column_stack([np.array([z[q][a,b] for q,a,qr,b in doubled.channels]) for z in (z0,z1)])
        if np.max(abs(Z.T@Z-np.eye(2)))>2e-10:raise ArithmeticError('unionbasisembeddinglostGram')
        cache=None
        for theta in registration['gate_angles']:
            changed,overlap,cache,check=gated_physical_block(expanded,source,theta,cache)
            gateop=SectorMergeOperator(changed,E,cross_bonds(supports[16],outside,bonds),0,
                memory_limit_bytes=768*1024**2);ops.append(gateop)
            HZ=np.column_stack([gateop.matvec(Z[:,k]) for k in range(2)])
            little=Z.T@HZ
            if np.max(abs(little-little.T))>2e-8:raise ArithmeticError('gatedrawHkernelnotHermitian')
            O=np.array([[sum(np.trace(z0[q].T@overlap[q]@z[q]).real for q in z0) for z in (z0,z1)]])
            result['raw_kernels'][str(theta)]=dict(H= little.real.tolist(),overlap_parent_to_basis=O[0].tolist(),
                physical_block_checks=check,raw_parent_gate_energy_per_site=float(little[0,0].real)/144)
            if theta==0 and np.max(abs(little-rawH))>2e-8:raise ArithmeticError('gatedzeroangleHkernelnotreplayed')
            stage('exactraw_gate_kernel_'+str(theta))
        save_checkpoint(output/'context.pkl',dict(parent=parent,complement=complement,center_basis=B,
            center_Hbasis=HB,channels=op.channels,rawH=rawH,source=source),metadata=dict(registration=registration))
        np.savez(output/'raw_center_basis.npz',basis=B,Hbasis=HB,Gram=gram,H=rawH)
        result.update(status='complete_samewidth_rawcenter_gate_kernels_pending_A1',basis=proofs,
            center_dimension=op.dimension,maximum_native_Gram_error=float(np.max(abs(gram-np.eye(2)))),
            basis_checkpoint_sha256=[file_sha256(output/f'basis_{k}.pkl') for k in range(2)],
            no_full_A1_quality_claim=True,context_sha256=file_sha256(output/'context.pkl'))
    except TimeoutError as error:result.update(status='bounded_partial',error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        result.update(wall_seconds=perf_counter()-start,CPU_seconds=process_time()-cpu,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            coarse_H_matvecs_actual=sum(op.matvec_count for op in ops),complete_A1_H_queries=0,new_MC_records=0)
        atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds-cap',type=float,default=900.);a=p.parse_args();run(a.output,a.seconds_cap)
