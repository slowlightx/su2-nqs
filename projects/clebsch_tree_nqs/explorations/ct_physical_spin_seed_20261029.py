"""Bounded physical triplet seeds and full-W phase-aware initialization.

This is NOT excited-state energy minimization or a measured excitation gap.
The frozen complete-A1 singlet supplies S_k^z|GS>, a strict triplet trial.
Six supervised full-W natural-gradient steps minimize a ray infidelity, not H.
Fit/calibration data are training feedback; the final128 full-H rows stay
untouched until the last checkpoint has been frozen. No remote writes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint
from su2_nqs.vmc.complex_importance_amplitude_sr import make_complex_importance_amplitude_sr
from su2_nqs.vmc.finite_candidate_calibration import common_proposal_FS
from su2_nqs.vmc.full_support_proposal import draw_full_support_sz0,full_support_sz0_density
from su2_nqs.vmc.spin_source_action import make_spin_source_products
from projects.clebsch_tree_nqs.workflows.ct_sector_excitation_pilot import deadline,family_norms
from projects.clebsch_tree_nqs.workflows.triangle_iid_orbit_audit import paired_iid_moments

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'projects/clebsch_tree_nqs/results/ct_sector_physical_seed_20261029'
PARENTS=ROOT/'projects/clebsch_tree_nqs/results/ct_sector_excitation_pilot_20261028/completed_CT6'


def fidelity(psi:np.ndarray,target:np.ndarray,Q:np.ndarray)->float:
    """Empirical complex ray fidelity under a fixed positive importance law."""
    z=np.sum(abs(psi)**2/Q);t=np.sum(abs(target)**2/Q)
    if z<=0 or t<=0:raise ValueError('positive sampled target/model norms required')
    return float(abs(np.sum(target.conj()*psi/Q))**2/(z*t))


def projector_training_action(psi:np.ndarray,target:np.ndarray,Q:np.ndarray)->np.ndarray:
    """Apply empirical I-|target><target|/<target|target>, NOT physical H."""
    if (psi.shape!=target.shape or Q.shape!=psi.shape or np.any(Q<=0)
            or not all(np.isfinite(x).all() for x in (psi,target,Q))):
        raise ValueError('finite matching amplitudes and positive importance weights required')
    norm=np.sum(abs(target)**2/Q)
    if norm<=0:raise ValueError('nonzero target norm required')
    return psi-target*np.sum(target.conj()*psi/Q)/norm


def overlap_report(psi:np.ndarray,target:np.ndarray,Q:np.ndarray,groups:int=16)->dict:
    """Point and independent-delete-group errors, no overlap-based state selection."""
    labels=np.repeat(np.arange(groups),len(Q)//groups)
    point=fidelity(psi,target,Q)
    leave=np.asarray([fidelity(psi[labels!=g],target[labels!=g],Q[labels!=g]) for g in range(groups)])
    se=np.sqrt((groups-1)/groups*np.sum((leave-leave.mean())**2))
    return dict(fidelity=point,group_jackknife_SE=float(se),records=len(Q),groups=groups,
        finite_sample_ratio_not_certified_population_fidelity=True)


def run(j2:float,output:Path,*,seed_offset:int=0)->dict:
    """Two frozen parents, fixed six attempts and independent terminal readout."""
    if j2 not in (0.,.125) or output.exists():raise ValueError('registered J2/new output required')
    output.mkdir(parents=True)
    tag='ct6_j0' if j2==0 else 'ct6_j0125';arity='ternary_joint' if j2==0 else 'binary_truncated'
    info=json.loads((PARENTS/'independent_review.json').read_text())['runs'][tag]
    parent=PARENTS/tag/'checkpoint.pkl'
    if file_sha256(parent)!=info['checkpoint_sha256']:raise ValueError('audited parent checkpoint differs')
    saved=load_checkpoint(parent)
    if pytree_sha256(saved['params'])!=info['parameter_sha256']:raise ValueError('audited parent params differ')
    if isinstance(seed_offset,bool) or not isinstance(seed_offset,int) or not 0<=seed_offset<=100000:
        raise ValueError('explicit bounded fresh stream offset required')
    offset=(0 if j2==0 else 100)+seed_offset
    seeds={name:3920291001+offset+i for i,name in enumerate(('fit','calibration','audit','statistics'))}
    old=saved['metadata']['rng']
    if set(seeds.values())&set(old['consumed_role_seeds']+old['excluded_seeds']):
        raise ValueError('new roles collide with inherited random history')
    registration=dict(task='CT6_physical_triplet_seed_and_supervised_initialization',j2=j2,
        parent_global_step=info['absolute_step'],parent_parameter_sha256=info['parameter_sha256'],
        parent_checkpoint_sha256=info['checkpoint_sha256'],fit_records=64,calibration_records=64,
        untouched_audit_records=128,supervised_attempts=6,physical_energy_updates=0,
        seeds=seeds,wall_cap_seconds=1000,FS_guard=.04,supervised_tangent_radius=.03,
        target_q=2,target_m2=0,source_channels=['K','M1','M2','M3'],
        initialization_changes_no_production_defaults=True)
    atomic_json(output/'registration.json',registration)
    started=perf_counter();report=dict(registration)
    sources=transitive_local_source_manifest([Path(__file__)],root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    atomic_json(output/'source_manifest.json',sources)
    from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
    from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank
    from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
    from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
    from su2_nqs.models.triangular_sector_amplitude import (make_sector_prepare,pack_sector_parameters,
        make_prepared_sector_amplitude,make_prepared_sector_exchange_queries)
    from su2_nqs.models.triangular_target_root import collective_spin_root_coefficients
    from projects.clebsch_tree_nqs.workflows.ct6_native_direction_calibration import normalized_raw_bank,draw_orbit_rows
    try:
        with deadline(1000):
            cfg=MixedClebschConfig(6,28,10)
            bonds=tuple(triangular_j1j2_bonds(6,6,j2=j2,periodic=True))
            images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
            bank,note=normalized_raw_bank(saved['params'],dict(cfg=cfg,
                bank=lambda p:arity_clebsch_bank(p,cfg,arity,8)))
            prepared=prepare_feature_lut_bank(bank,cfg)
            coords=np.asarray([divmod(i,6) for i in range(36)])
            momenta=np.asarray([[1/3,1/3],[.5,0.],[0.,.5],[.5,.5]])
            phases=np.exp(2j*np.pi*momenta@coords.T)
            prob=np.abs([c for a,b,c in bonds]);prob/=prob.sum()
            action,meta=make_spin_source_products(lambda e:make_feature_lut_exchange_queries(cfg,e),
                images,bonds,prob,phases,n_sites=36,projection_chunk=4)
            report['source_kernel']=meta;context=dict(cfg=cfg,images=images,bonds=bonds)
            data={}
            # Terminal seed is registered now, but its draw/products are NOT
            # exposed until the full-W initialization endpoint is frozen.
            def query_role(role,count):
                rows,draw=draw_full_support_sz0(lambda n,s:draw_orbit_rows([bank],context,n,s,bridge=True),
                    count,n_sites=36,seed=seeds[role],epsilon=.125)
                serial=jax.tree.map(lambda x:np.asarray(x).tolist()
                    if isinstance(x,(np.ndarray,np.generic)) else x,draw)
                atomic_json(output/(role+'_draw.json'),serial)
                parts=[];tick=perf_counter()
                for i in range(count):
                    parts.append(tuple(np.asarray(v) for v in jax.block_until_ready(action(prepared,jnp.asarray(rows[i:i+1])))))
                    if i%32==0:print(tag,role,i+1,'/',count,flush=True)
                phi,hp,qo,qpush=[np.concatenate([p[k] for p in parts]) for k in range(4)]
                Q=full_support_sz0_density(qo,qpush,n_sites=36,epsilon=.125)
                np.savez_compressed(output/(role+'_source_products.npz'),rows=rows,psi=phi,hpsi=hp,proposal=Q)
                report.setdefault('source_query_wall_seconds',{})[role]=perf_counter()-tick
                return rows,phi,hp,Q
            data['fit']=query_role('fit',64);data['calibration']=query_role('calibration',64)
            prepare=jax.jit(make_sector_prepare(cfg,arity=arity,binary_rank=8))
            amplitude=make_prepared_sector_amplitude(cfg,2,0,((1,3),(1,3)),chunk=4)
            batch=jax.jit(lambda p,x:jax.lax.map(lambda row:amplitude(prepare(p),row),x))
            native=arity_clebsch_bank(saved['params'],cfg,arity,8)
            params=pack_sector_parameters(saved['params'],tuple(jnp.asarray(x) for x in collective_spin_root_coefficients(native,cfg,2)))
            zeros=jax.tree.map(jnp.zeros_like,params);moments=(jnp.asarray(0),zeros,zeros)
            initial=deepcopy(params)
            update=make_complex_importance_amplitude_sr(lambda p,x:amplitude(prepare(p),x),params,
                damping=.05,score_chunk=1,parameter_chunk_size=8192,sr_tangent_fs_radius=.03,
                prepare=prepare,prepared_amplitude=amplitude)
            transitions=[]
            fit,cal=data['fit'],data['calibration']
            for step in range(1,7):
                current=np.asarray(batch(params,jnp.asarray(fit[0])))
                fake=projector_training_action(current,fit[1][:,1],fit[3])
                candidate,proposed_moments,sr=jax.block_until_ready(update(params,moments,
                    jnp.asarray(fit[0]),jnp.asarray(fake),jnp.asarray(fit[3]),1.))
                old=np.asarray(batch(params,jnp.asarray(cal[0])))
                oldF=fidelity(old,cal[1][:,1],cal[3]);chosen=params;selected=0.;guard=None
                for factor in (1.,.5,.25,.125):
                    p=jax.tree.map(lambda a,b:a+factor*(b-a),params,candidate)
                    new=np.asarray(batch(p,jnp.asarray(cal[0])))
                    geom=common_proposal_FS(np.column_stack((old,new)),cal[3],groups=16)
                    if (geom['nominal_upper']<=.04 and fidelity(new,cal[1][:,1],cal[3])>oldF):
                        chosen,selected,guard=p,factor,geom;break
                delta=jax.tree.map(lambda a,b:a-b,chosen,params)
                params=chosen
                if selected:moments=proposed_moments
                transitions.append(dict(attempt=step,accepted=bool(selected),fraction=selected,
                    calibration_F_before=oldF,calibration_F_after=fidelity(np.asarray(batch(params,jnp.asarray(cal[0]))),cal[1][:,1],cal[3]),
                    geometry=guard,SR_residual=float(sr['relative_SR_residual']),
                    actual_changed_families=family_norms(delta)))
                save_checkpoint(output/'initialized.pkl',params,optimizer_state=moments,
                    metadata=dict(registration_sha256=canonical_sha256(registration),attempt=step,
                        objective='supervised_infidelity_NOT_energy',parameter_sha256=pytree_sha256(params),
                        final_audit_seed_reserved_unconsumed=seeds['audit']))
                atomic_json(output/'initialization_progress.json',dict(attempt=step,transitions=transitions))
                print(tag,'supervised',step,'F_cal',transitions[-1]['calibration_F_after'],flush=True)
            atomic_json(output/'endpoint_frozen.json',dict(parameter_sha256=pytree_sha256(params),
                checkpoint_sha256=file_sha256(output/'initialized.pkl'),fixed_attempts=6,
                no_terminal_data_seen=True))
            audit=query_role('audit',128);rows,phi,hp,Q=audit
            groups=np.repeat(np.arange(16),8)
            report['source_seed_moments']={name:paired_iid_moments(phi[:,[0,k]],hp[:,[0,k]],Q,groups,
                n_sites=36,seed=seeds['statistics']+k) for k,name in enumerate(registration['source_channels'],1)}
            report['initial_and_final_F_with_physical_K_seed']={
                name:overlap_report(np.asarray(batch(p,jnp.asarray(rows))),phi[:,1],Q)
                for name,p in (('raw_cut',initial),('full_W_supervised',params))}
            report['transitions']=transitions
            report['fixed_attempts_completed']=True
            report['final_params_sha256']=pytree_sha256(params)
            report['source_queries_centers']=256
            report['not_a_lowest_triplet_or_excitation_gap']=True
            report['initialized_CT_energy_not_audited']=True
            report['status']='complete_physical_seed_and_heldout_full_W_initialization'
    except TimeoutError as error:
        report.update(status='bounded_timeout_partial_not_final',error=str(error))
    report['wall_seconds']=perf_counter()-started
    report['source_seals_still_match']=all(file_sha256(ROOT/p)==s for p,s in sources.items())
    atomic_json(output/'results.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--j2',type=float,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed-offset',type=int,default=0)
    args=parser.parse_args(); r=run(args.j2,args.output,seed_offset=args.seed_offset)
    print(json.dumps({k:r[k] for k in ('status','j2','wall_seconds')}),flush=True)
