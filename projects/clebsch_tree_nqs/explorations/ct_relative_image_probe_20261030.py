"""Exact one-sided projector identities and a fixed raw-Born norm oracle.

P=P†=P² and [H,P]=0 reduce integrated double-image moments to one relative
image. Signed overlaps remain: no wavefunction, projector, node or H changes.
The actual-CT part makes no H query and fits no attention model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
from time import perf_counter,process_time

import numpy as np
from su2_nqs.frontier.artifacts import atomic_json

ROOT=Path(__file__).resolve().parents[3]


def exact_identities() -> dict:
    """Independent six-spin Pauli/permutation reference, including derivatives."""
    n=6; rng=np.random.default_rng(26103051); dim=2**n
    single=(np.array([[0,1],[1,0]])/2,np.array([[0,1j],[-1j,0]])/2,np.diag([-.5,.5]))
    def site_operator(matrix,site):
        result=np.ones((1,1))
        for i in range(n):result=np.kron(result,matrix if i==site else np.eye(2))
        return result
    spin=[[site_operator(m,i) for i in range(n)] for m in single]
    def bond(i,j):return sum(s[i]@s[j] for s in spin)
    h=sum(bond(i,(i+1)%n)+.23*bond(i,(i+2)%n) for i in range(n))
    hbad=bond(0,1)+.17*h
    bits=((np.arange(dim)[:,None]>>np.arange(n-1,-1,-1))&1)
    powers=1<<np.arange(n-1,-1,-1)
    def projector(maps):
        matrices=[]
        for mapping in maps:
            indices=bits[:,mapping]@powers; u=np.zeros((dim,dim));u[np.arange(dim),indices]=1
            matrices.append(u)
        return np.mean(matrices,axis=0)
    maps=[np.roll(np.arange(n),i) for i in range(n)]+[np.roll(np.arange(n)[::-1],i) for i in range(n)]
    psi=np.zeros(dim,complex); delta=np.zeros(dim,complex); mask=bits.sum(1)==3
    psi[mask]=rng.normal(size=mask.sum())+1j*rng.normal(size=mask.sum());psi/=np.linalg.norm(psi)
    delta[mask]=rng.normal(size=mask.sum())+1j*rng.normal(size=mask.sum())
    from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
    from su2_nqs.su2.fusion_path import enumerate_paths
    u=coupled_basis_matrix(enumerate_paths(6,0),0,0)
    c=rng.normal(size=5)+1j*rng.normal(size=5); c/=np.linalg.norm(c)
    singlet=u@c; ds=u@(rng.normal(size=5)+1j*rng.normal(size=5))
    cases=[]
    for name,p,v,d in (('full_D6_fixed_M0',projector(maps),psi,delta),
                       ('C3_strict_singlet',projector([np.roll(np.arange(n),i) for i in (0,2,4)]),singlet,ds)):
        pv=p@v; pd=p@d
        left=np.array([np.vdot(pv,pv),np.vdot(pv,h@pv),np.vdot(h@pv,h@pv)])
        right=np.array([np.vdot(v,p@v),np.vdot(v,h@p@v),np.vdot(h@v,p@h@v)])
        dl=2*np.real([np.vdot(pd,pv),np.vdot(pd,h@pv),np.vdot(h@pd,h@pv)])
        dr=2*np.real([np.vdot(d,p@v),np.vdot(d,h@p@v),np.vdot(h@d,p@h@v)])
        quotient_derivative=(dl[1:]*left[0].real-left[1:].real*dl[0])/left[0].real**2
        def normalized(theta):
            a=p@(v+theta*d);return np.array([np.vdot(a,h@a).real,np.vdot(h@a,h@a).real])/np.vdot(a,a).real
        fd=(normalized(1e-6)-normalized(-1e-6))/(2e-6)
        discrepancy=float(abs(np.vdot(pv,hbad@pv)-np.vdot(v,hbad@p@v)))
        cases.append(dict(case=name,P_idempotence_max=float(np.max(abs(p@p-p))),
            H_commutator_max=float(np.max(abs(h@p-p@h))),
            moment_max_error=float(np.max(abs(left-right))),derivative_max_error=float(np.max(abs(dl-dr))),
            quotient_finite_difference_max_error=float(np.max(abs(fd-quotient_derivative))),
            normalized_energy_H2_derivatives=quotient_derivative.tolist(),
            noncommuting_H_negative_control_discrepancy=discrepancy))
    return dict(n_sites=6,cases=cases,H_queries_to_CT=0,
                convention='spin_half_Pauli_in_bit0_down_basis; explicit physical site permutations')


def one_sided_stats(a:np.ndarray,q:np.ndarray,identity:int) -> dict:
    """Exact image-conditional noise; outer variance is only a small-panel estimate."""
    a,q=np.asarray(a),np.asarray(q)
    if a.ndim!=2 or q.shape!=a.shape or np.any(q<=0) or np.max(abs(q.sum(1)-1))>1e-12:
        raise ValueError('matching amplitude matrix and positive normalized image proposal required')
    center=a[:,identity]
    if np.any(center==0):raise ValueError('raw-Born center cannot be repaired or discarded')
    ratios=a/center[:,None]; mean=ratios.mean(1)
    conditional=(abs(ratios)**2/q).sum(1)/a.shape[1]**2-abs(mean)**2
    if conditional.min()<-1e-10:raise ArithmeticError('negative conditional variance beyond roundoff')
    conditional=np.maximum(conditional,0.)
    outer=float(np.sum(abs(mean-mean.mean())**2)/(len(mean)-1))
    result=dict(norm_overlap_mean_real=float(mean.mean().real),norm_overlap_mean_imaginary=float(mean.mean().imag),
        finite_raw_Born_panel_outer_variance=outer,mean_image_conditional_variance=float(conditional.mean()),
        per_row_exact_mean_real=mean.real.tolist(),per_row_exact_mean_imaginary=mean.imag.tolist(),
        per_row_conditional_variance=conditional.tolist(),raw_Born_rows=len(a),images=a.shape[1],k={})
    for k in (8,32):
        total=outer+float(conditional.mean())/k
        result['k'][str(k)]=dict(estimated_total_record_variance=total,
            raw_image_count_reduction=a.shape[1]/k,
            conditional_noise_over_outer_variance=float(conditional.mean())/(k*outer) if outer>0 else None,
            finite_panel_variance_times_image_count_gain=(a.shape[1]/k)*outer/total if total>0 else None)
    return result


def run(output:Path) -> dict:
    """One resource-bounded probe; no row, image, proposal or endpoint selection."""
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('use a fresh directory')
    start=perf_counter();cpu=process_time()
    def stop(signum,frame):raise TimeoutError('fixed90-second resource limit reached')
    signal.signal(signal.SIGALRM,stop);signal.alarm(90)
    signal.signal(signal.SIGXCPU,stop);resource.setrlimit(resource.RLIMIT_CPU,(85,90))
    reg=dict(task='one_relative_group_norm_oracle_after_exact_projector_validation',
        seeds=dict(ct6_j0=26103061,ct6_j0125=26103062),rows_per_parent=16,images=432,
        proposal_q_floor_fraction=.05,raw_draw_images='identity_only',bridge=False,
        no_training=True,new_H_queries=0,no_wavefunction_or_node_floor=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),CPU_cap=90,wall_cap=90)
    atomic_json(output/'registration.json',reg);result=dict(registration=reg,parents={},status='started')
    def progress(stage):
        result['stage']=stage;result['wall_seconds']=perf_counter()-start;result['CPU_seconds']=process_time()-cpu
        atomic_json(output/'progress.json',result)
        if result['wall_seconds']>82 or result['CPU_seconds']>80:raise TimeoutError('save margin within90-second budget')
    try:
        exact=exact_identities();atomic_json(output/'exact_identity_reference.json',exact)
        if any(c['moment_max_error']>1e-12 or c['derivative_max_error']>1e-11
               or c['noncommuting_H_negative_control_discrepancy']<1e-6 for c in exact['cases']):
            raise ArithmeticError('exact projector/H/negative-control gate failed')
        result['exact_reference']=exact;progress('exact_identities_and_noncommuting_negative_control_passed')
        import jax
        jax.config.update('jax_enable_x64',True)
        import jax.numpy as jnp
        from su2_nqs.frontier.artifacts import pytree_sha256
        from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
        from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
        from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
        from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
        from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank,make_feature_lut_amplitude
        from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
        from su2_nqs.vmc.checkpoint import load_checkpoint
        from projects.clebsch_tree_nqs.workflows.ct6_native_direction_calibration import normalized_raw_bank,draw_orbit_rows
        base=ROOT/'projects/clebsch_tree_nqs/results/ct_sector_excitation_pilot_20261028/completed_CT6'
        reviewed=json.loads((base/'independent_review.json').read_text())['runs']
        images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
        identity=int(np.flatnonzero(np.all(images==np.arange(36),axis=1))[0]);cfg=MixedClebschConfig(6,28,10)
        for name,arity,j2 in (('ct6_j0','ternary_joint',0.),('ct6_j0125','binary_truncated',.125)):
            t=perf_counter();path=base/name/'checkpoint.pkl';saved=load_checkpoint(path);info=reviewed[name]
            if (hashlib.sha256(path.read_bytes()).hexdigest()!=info['checkpoint_sha256']
                    or pytree_sha256(saved['params'])!=info['parameter_sha256']):raise ValueError('original parent identity mismatch')
            bank,note=normalized_raw_bank(saved['params'],dict(cfg=cfg,bank=lambda p:arity_clebsch_bank(p,cfg,arity,8)))
            prepared=prepare_feature_lut_bank(bank,cfg);prep_seconds=perf_counter()-t;progress(name+'_prepared')
            t=perf_counter();rows,draw=draw_orbit_rows([bank],dict(cfg=cfg,images=np.arange(36,dtype=np.int32)[None,:],
                bonds=triangular_j1j2_bonds(6,6,j2=j2,periodic=True)),16,reg['seeds'][name],bridge=False)
            draw_seconds=perf_counter()-t
            atomic_json(output/(name+'_raw_draw.json'),dict(seed=reg['seeds'][name],rows=rows.tolist(),
                maximum_Born_replay_error=draw['maximum_Born_replay_error'],
                latent=draw['latent'].tolist(),source='exact normalized raw-Born IID; no orbit image or H push'))
            progress(name+'_all16_raw_Born_rows_frozen')
            transformed=rows[:,images].reshape(-1,36);t=perf_counter();score=np.ones(len(transformed))
            for b,block in enumerate(triangular_schur_hierarchy(6).blocks):
                ids=transformed[:,block.sites]@(1<<np.arange(8,-1,-1))
                local=sum(np.sum(abs(np.asarray(table))**2,axis=1) for table in prepared['local_features'][b])
                score*=local[ids]
            score=score.reshape(16,432);score/=score.sum(1,keepdims=True)
            cheap_q=.95*score+.05/432;cheap_q_seconds=perf_counter()-t
            raw=make_feature_lut_amplitude(cfg);batch=jax.jit(jax.vmap(raw,in_axes=(None,0)))
            chunks=[];times=[]
            for first in range(0,len(transformed),16):
                t=perf_counter();value=np.asarray(jax.block_until_ready(batch(prepared,jnp.asarray(transformed[first:first+16]))))
                chunks.append(value);times.append(perf_counter()-t)
                if first==0:progress(name+'_raw_kernel_compiled')
                if perf_counter()-start>82 or process_time()-cpu>80:raise TimeoutError('fixed90-second cap between raw batches')
            amplitude=np.concatenate(chunks).reshape(16,432)
            magnitude=abs(amplitude);optimal=.95*magnitude/magnitude.sum(1,keepdims=True)+.05/432
            np.savez_compressed(output/(name+'_images.npz'),rows=rows,images=images,raw_group_amplitudes=amplitude,
                cheap_q=cheap_q,oracle_q=optimal)
            arms={label:one_sided_stats(amplitude,q,identity) for label,q in
                (('uniform',np.full_like(amplitude,1/432)),('cheap_local_context',cheap_q),('magnitude_oracle',optimal))}
            warm=float(np.median(times[1:]));full_warm=warm*(len(transformed)/16)
            for label,arm in arms.items():
                qcost=cheap_q_seconds if label=='cheap_local_context' else 0.
                for k in (8,32):
                    sample_cost=draw_seconds+qcost+warm*k
                    full_cost=draw_seconds+full_warm
                    variance_ratio=arm['k'][str(k)]['estimated_total_record_variance']/arm['finite_raw_Born_panel_outer_variance']
                    arm['k'][str(k)]['finite_panel_gain_including_measured_draw_and_batch_cost']=full_cost/sample_cost/variance_ratio
            result['parents'][name]=dict(original_parent_step=info['absolute_step'],arity=arity,
                parent_parameter_sha256=info['parameter_sha256'],parent_checkpoint_sha256=info['checkpoint_sha256'],
                parent_is_not_newest_remote_endpoint=True,preparation_seconds=prep_seconds,
                normalization_cross_overlap_error=note['raw_cross_overlap_error'],raw_Born_16_draw_seconds=draw_seconds,
                raw_Born_density_replay_error=draw['maximum_Born_replay_error'],cheap_all_image_context_q_seconds=cheap_q_seconds,
                first16_raw_kernel_seconds=times[0],warm16_raw_kernel_median_seconds=warm,
                all6912_raw_image_seconds=sum(times),arms=arms,
                true_H_image_labels_available=False,oracle_acquisition_is_all432_images_per_row=True)
            progress(name+'_complete16_raw_Born_norm_oracle')
        result['status']='complete_exact_and_actual_raw_Born_norm_oracle_no_H_or_attention_training'
    except TimeoutError as error:
        result['status']='censored_at_registered_budget_no_extension';result['error']=str(error)
    finally:
        signal.alarm(0);result['wall_seconds']=perf_counter()-start;result['CPU_seconds']=process_time()-cpu
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['peak_RSS_GiB']=rss/(1024**3 if sys.platform=='darwin' else 1024**2)
        result['limits']=['Only16 fresh exact raw-Born rows per parent: variance-times-cost estimates are noisy and not a certified throughput gain.',
            'All image amplitudes are acquired for retrospective oracle scoring; no H, gradient, energy or variance precision is measured.',
            'Positive image q leaves the coherent target unchanged; ratios and finite-sample metric estimates remain delicate.',
            'H2 requires full-support physical proposals even where raw psi vanishes; raw-Born norm probing is not an H2 proposal certificate.',
            'Fixed474/500 parents are not relabeled as newer checkpoints; no remote or production changes.']
        atomic_json(output/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.output)
