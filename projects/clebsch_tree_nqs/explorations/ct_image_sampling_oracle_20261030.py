"""Bounded actual-CT image-weight oracle, amplitude only, no learned speed claim.

All spatial images are evaluated and retained. The hypothetical random
double-image estimator keeps coherent cross terms; proposal floors do not
alter amplitudes, exact zeros, or the target state. No H-image labels exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
from time import perf_counter, process_time

import numpy as np
from su2_nqs.frontier.artifacts import atomic_json

ROOT=Path(__file__).resolve().parents[3]


def norm_pair_oracle(amplitudes: np.ndarray, *, uniform_fraction: float = .05) -> dict:
    """Exact variance for IID pair proposals; not an all-algorithm lower bound.

    With S=sum_g a_g and q_g>0, Z=a_g* a_h/(q_g q_h) has mean |S|².
    For independent g,h its relative variance is
    (sum_g |a_g|²/q_g / |S|²)²-1. Group-average factors cancel in this ratio.
    The q∝|a| limit minimizes this particular one-pair second moment.
    """
    a=np.asarray(amplitudes)
    if (a.ndim!=1 or not len(a) or not np.isfinite(a).all()
            or not 0<uniform_fraction<1): raise ValueError('finite amplitude vector and strict proposal fraction required')
    count=len(a); scale=float(np.max(abs(a)))
    if scale==0: return dict(status='zero_raw_amplitudes',images=count,zero_images=count,
                              relative_errors_defined=False)
    v=a/scale; total=np.sum(v); l1=np.sum(abs(v)); power=np.mean(abs(v)**2)
    if total==0: return dict(status='exact_projected_node',images=count,zero_images=int(np.sum(a==0)),
                             relative_errors_defined=False)
    c1=float(abs(total)/l1); c2=float(abs(v.mean())**2/power)
    proposals=dict(uniform=np.full(count,1/count),
                   magnitude_oracle_with_5percent_uniform=(1-uniform_fraction)*abs(v)/l1+uniform_fraction/count)
    limits=max(c1**(-4)-1,0.)
    outcomes={}
    for name,q in proposals.items():
        relative_variance=max(float((np.sum(abs(v)**2/q)/abs(total)**2)**2-1),0.)
        outcomes[name]=dict(pair_relative_variance=relative_variance,
            relative_SE_k8=float(np.sqrt(relative_variance/8)),relative_SE_k32=float(np.sqrt(relative_variance/32)),
            minimum_positive_q=float(q.min()))
    mass=np.sort(abs(v))[::-1]
    return dict(status='finite_nonzero_projected_amplitude',images=count,zero_images=int(np.sum(a==0)),
        relative_errors_defined=True,L1_sign_coherence=c1,L2_squared_coherence=c2,
        top8_L1_mass=float(mass[:8].sum()/l1),top32_L1_mass=float(mass[:32].sum()/l1),
        limiting_magnitude_oracle_pair_relative_variance=limits,
        limiting_magnitude_oracle_relative_SE_k8=float(np.sqrt(limits/8)),
        limiting_magnitude_oracle_relative_SE_k32=float(np.sqrt(limits/32)),proposals=outcomes,
        interpretation='conditional single-configuration norm oracle; no H, ratio, learning or total-cost guarantee')


def run(output: Path) -> dict:
    """Fixed rows and endpoints; hard stop at60 wall/CPU seconds, no extension."""
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()): raise FileExistsError('use a new empty directory')
    start=perf_counter(); cpu_start=process_time()
    def stop(signum,frame): raise TimeoutError('fixed60-second resource cap reached')
    signal.signal(signal.SIGALRM,stop); signal.alarm(60)
    signal.signal(signal.SIGXCPU,stop); resource.setrlimit(resource.RLIMIT_CPU,(55,60))
    registration=dict(task='actual_CT_image_importance_oracle_amplitudes_only',rows_per_current_parent=4,
        uniform_M0_row_seeds=dict(ct6_j0=26103041,ct6_j0125=26103042),
        parent_steps=dict(ct6_j0=474,ct6_j0125=500),current_CT6_images=432,
        no_row_selection=True,old_CT12_images=1728,old_CT12_is_not_current_parent=True,
        new_H_queries=0,learned_model_fits=0,no_target_or_exact_zero_floor=True,
        pair_q_uniform_fraction=.05,wall_seconds_cap=60,cpu_seconds_cap=60,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(output/'registration.json',registration)
    result=dict(registration=registration,old_CT12={},current_CT6={},status='started')
    def progress(stage):
        result['stage']=stage; result['wall_seconds']=perf_counter()-start
        result['CPU_seconds']=process_time()-cpu_start; atomic_json(output/'progress.json',result)
        if result['wall_seconds']>55 or result['CPU_seconds']>52: raise TimeoutError('save margin at fixed60-second cap')
    try:
        old=ROOT/'projects/clebsch_tree_nqs/results/ct_conditional_nested_initialization_20261023/support_probe_32'
        for path in sorted(old.glob('*group_amplitudes.npz')):
            with np.load(path) as z: arrays=z['raw_group_amplitudes']
            result['old_CT12'][path.name]=dict(source=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                endpoints=['old8_joint','nested10_joint'],
                source_scope='posthoc support diagnostic, old20261023 states, only one fixed chain0 configuration per file',
                rows=[norm_pair_oracle(a) for a in arrays])
        progress('old_CT12_actual_amplitude_cache_reprocessed')
        import jax
        jax.config.update('jax_enable_x64',True)
        import jax.numpy as jnp
        from su2_nqs.frontier.artifacts import pytree_sha256
        from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
        from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
        from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank,make_feature_lut_amplitude
        from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
        from su2_nqs.vmc.checkpoint import load_checkpoint
        from projects.clebsch_tree_nqs.workflows.ct6_native_direction_calibration import normalized_raw_bank
        base=ROOT/'projects/clebsch_tree_nqs/results/ct_sector_excitation_pilot_20261028/completed_CT6'
        reviewed=json.loads((base/'independent_review.json').read_text())['runs']
        images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
        if images.shape!=(432,36): raise ValueError('complete432-image physical projector required')
        for name,arity in (('ct6_j0','ternary_joint'),('ct6_j0125','binary_truncated')):
            tick=perf_counter(); path=base/name/'checkpoint.pkl'; saved=load_checkpoint(path); info=reviewed[name]
            if (hashlib.sha256(path.read_bytes()).hexdigest()!=info['checkpoint_sha256']
                    or pytree_sha256(saved['params'])!=info['parameter_sha256']): raise ValueError('frozen parent identity mismatch')
            cfg=MixedClebschConfig(6,28,10)
            bank,note=normalized_raw_bank(saved['params'],dict(cfg=cfg,
                bank=lambda p:arity_clebsch_bank(p,cfg,arity,8)))
            prepared=prepare_feature_lut_bank(bank,cfg)
            prep_seconds=perf_counter()-tick; progress(name+'_normalized_raw_bank_ready')
            rng=np.random.default_rng(registration['uniform_M0_row_seeds'][name])
            rows=np.asarray([rng.permutation(np.r_[np.zeros(18,np.int32),np.ones(18,np.int32)]) for _ in range(4)])
            transformed=rows[:,images].reshape(-1,36)
            raw=make_feature_lut_amplitude(cfg); batch=jax.jit(jax.vmap(raw,in_axes=(None,0)))
            pieces=[]; times=[]
            for first in range(0,len(transformed),16):
                t=perf_counter(); values=np.asarray(jax.block_until_ready(batch(prepared,jnp.asarray(transformed[first:first+16]))))
                times.append(perf_counter()-t); pieces.append(values)
                if first==0: progress(name+'_first_raw_batch_compiled')
                if perf_counter()-start>55 or process_time()-cpu_start>52: raise TimeoutError('fixed resource cap between raw batches')
            amplitudes=np.concatenate(pieces).reshape(4,432)
            np.savez_compressed(output/(name+'_image_amplitudes.npz'),rows=rows,images=images,raw_group_amplitudes=amplitudes)
            result['current_CT6'][name]=dict(parent_step=info['absolute_step'],arity=arity,
                parent_parameter_sha256=info['parameter_sha256'],parent_checkpoint_sha256=info['checkpoint_sha256'],
                parent_not_latest20261030_endpoint=True,
                normalized_raw_bank_seconds=prep_seconds,normalization_cross_overlap_error=note['raw_cross_overlap_error'],
                first16_raw_image_kernel_seconds=times[0],all1728_raw_image_seconds=sum(times),
                warm16_raw_image_median_seconds=float(np.median(times[1:])),
                rows=[norm_pair_oracle(a) for a in amplitudes],rows_distribution='fresh uniform M0, NOT target Born')
            progress(name+'_all_four_frozen_rows_complete')
        result['status']='complete_fixed_amplitude_oracle_gate_no_H_or_learning'
    except TimeoutError as error:
        result['status']='censored_fixed_budget_no_extension'; result['error']=str(error)
    finally:
        signal.alarm(0)
        result['wall_seconds']=perf_counter()-start;result['CPU_seconds']=process_time()-cpu_start
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['peak_RSS_GiB']=rss/(1024**3 if sys.platform=='darwin' else 1024**2)
        result['limits']=['No image-resolved Hpsi cache was found; this cannot establish energy/force variance or H cost.',
            'The magnitude oracle reads every image amplitude; its teacher/acquisition cost is not free.',
            'Four fixed uniform-M0 rows per parent do not represent target Born or H-residual weight.',
            'Pair estimator mean is unbiased; normalized ratios retain finite-sample bias and possible sign instability.',
            'Fresh probes retain old474/500 parent identities, not later running endpoints.']
        atomic_json(output/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.output)
