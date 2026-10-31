"""One bounded physical-energy update, anchored exactly at S_K^z|GS_A1>.

Psi(W)=source+c*(CT(W)-CT(Wref)); c is fixed from fit norms, not a head.
All native nonroot W and the strict-triplet root remain trainable. Reference
and source are frozen. Fit/guard were previously consumed initialization data;
the128 terminal IID rows and their full H are fresh and cannot select a state.
This is an opt-in local physics pilot, never a lowest-gap claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.complex_importance_amplitude_sr import make_complex_importance_amplitude_sr
from su2_nqs.vmc.finite_candidate_calibration import common_proposal_FS
from su2_nqs.vmc.full_support_proposal import full_support_sz0_density
from su2_nqs.vmc.spin_source_action import make_spin_source_products
from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_projected_amplitude, prepare_feature_lut_bank
from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from projects.clebsch_tree_nqs.workflows.ct_sector_excitation_pilot import make_runtime, deadline, family_norms
from projects.clebsch_tree_nqs.workflows.ct6_native_direction_calibration import normalized_raw_bank
from projects.clebsch_tree_nqs.workflows.triangle_iid_orbit_audit import paired_iid_moments
from .ct_physical_spin_seed_20261029 import PARENTS


def anchored_amplitude(source, current, reference, scale):
    """Fixed-scale affine physical amplitude; source is exact at current=reference."""
    return source + scale * (current - reference)


def run(directory: Path) -> dict:
    """One full-W proposal, independent finite guard, then fresh fixed audit."""
    out = directory / 'protected_physical_update_v1'
    if out.exists():
        raise FileExistsError('new immutable pilot directory required')
    old = json.loads((directory/'results.json').read_text())
    frozen = json.loads((directory/'endpoint_frozen.json').read_text())
    assert file_sha256(directory/'initialized.pkl') == frozen['checkpoint_sha256']
    saved = load_checkpoint(directory/'initialized.pkl'); params = saved['params']
    assert pytree_sha256(params) == frozen['parameter_sha256']
    tag = 'ct6_j0' if old['j2'] == 0 else 'ct6_j0125'
    arity = 'ternary_joint' if old['j2'] == 0 else 'binary_truncated'
    parent = load_checkpoint(PARENTS/tag/'checkpoint.pkl')
    assert pytree_sha256(parent['params']) == old['parent_parameter_sha256']
    plan = dict(model=dict(length=6,multiplicity_cap=28,maximum_q=10),j2=old['j2'],
        arity=arity,binary_rank=8,epsilon=.125,projection_chunk=4,H_chunk=1,action_chunk=1,
        sector=dict(target_q=2,target_m2=0,momentum=((1,3),(1,3))))
    data = {}
    for role in ('fit','calibration'):
        with np.load(directory/(role+'_source_products.npz')) as raw:
            data[role] = (raw['rows'],raw['psi'][:,1],raw['hpsi'][:,1],raw['proposal'])
    seed = 3940291701 + (0 if old['j2'] == 0 else 100)
    assert seed not in old['seeds'].values()
    old_rng=parent['metadata']['rng']
    assert not {seed,seed+1,seed+2}.intersection(old_rng['consumed_role_seeds']+old_rng['excluded_seeds'])
    out.mkdir()
    registration = dict(task='exact_physical_triplet_anchor_one_full_W_energy_update',
        j2=old['j2'],fixed_attempts=1,fit_records=64,guard_records=64,terminal_records=128,
        reused_initialization_fit_and_guard_explicitly_training_feedback=True,
        final_audit_seed=seed,statistics_seed=seed+1,maximum_guard_FS=.015,tangent_FS=.01,
        damping=.05,wall_cap_seconds=1000,initialized_reference_checkpoint_sha256=frozen['checkpoint_sha256'],
        parent_parameter_sha256=old['parent_parameter_sha256'],no_new_head_or_trainable_scalar=True,
        target_q=2,target_m2=0,momentum=[[1,3],[1,3]],no_remote_changes=True)
    atomic_json(out/'registration.json',registration)
    started=perf_counter()
    with deadline(1000):
        runtime=make_runtime(plan,parent); cfg=runtime['cfg']
        bank,_=normalized_raw_bank(parent['params'],dict(cfg=cfg,
            bank=lambda p:arity_clebsch_bank(p,cfg,arity,8)))
        ground=prepare_feature_lut_bank(bank,cfg)
        images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
        parent_amp=make_feature_lut_projected_amplitude(cfg,images,chunk=4)
        coordinates=np.asarray([divmod(i,6) for i in range(36)])
        phases=jnp.asarray(np.exp(2j*np.pi*np.sum(coordinates,axis=1)/3))
        def source(row):
            row=jnp.asarray(row,jnp.int32)
            spin=(2*row-1)/2
            return jnp.dot(phases,spin)/6*parent_amp(ground,row)
        ref=runtime['prepare'](params)
        fit=data['fit']; guard=data['calibration']
        source_batch=jax.jit(jax.vmap(source))
        source_replay=float(np.max(abs(np.asarray(source_batch(jnp.asarray(fit[0])))-fit[1])))
        np.testing.assert_allclose(np.asarray(source_batch(jnp.asarray(fit[0]))),fit[1],rtol=2e-11,atol=2e-16)
        ref_fit=runtime['amplitudes'](params,fit[0])
        scale=float(np.sqrt(np.sum(abs(fit[1])**2/fit[3])/np.sum(abs(ref_fit)**2/fit[3])))
        def amplitude(prepared,row):
            return anchored_amplitude(source(row),runtime['amplitude'](prepared,row),
                runtime['amplitude'](ref,row),scale)
        batch=jax.jit(lambda p,x:jax.lax.map(lambda row:amplitude(runtime['prepare'](p),row),x))
        update=make_complex_importance_amplitude_sr(lambda p,x:amplitude(runtime['prepare'](p),x),
            params,damping=.05,score_chunk=1,parameter_chunk_size=8192,sr_tangent_fs_radius=.01,
            prepare=runtime['prepare'],prepared_amplitude=amplitude)
        tick=perf_counter()
        candidate,moments,sr=jax.block_until_ready(update(params,saved['optimizer_state'],
            jnp.asarray(fit[0]),jnp.asarray(fit[2]),jnp.asarray(fit[3]),1.))
        sr_seconds=perf_counter()-tick
        assert bool(sr['numerically_valid']) and float(sr['relative_SR_residual']) < 1e-7
        chosen=params;fraction=0.;geometry=None;selection=None
        for factor in (1.,.5,.25,.125):
            proposal=jax.tree.map(lambda a,b:a+factor*(b-a),params,candidate)
            psi=np.asarray(batch(proposal,jnp.asarray(guard[0])))
            g=common_proposal_FS(np.column_stack((guard[1],psi)),guard[3],groups=16)
            if g['nominal_upper']<=.015:
                rp,rh=runtime['H'](params,guard[0]);cp,ch=runtime['H'](proposal,guard[0])
                hp=anchored_amplitude(guard[2],ch,rh,scale)
                assert np.allclose(psi,anchored_amplitude(guard[1],cp,rp,scale),rtol=2e-11,atol=2e-14)
                selection=paired_iid_moments(np.column_stack((guard[1],psi)),
                    np.column_stack((guard[2],hp)),guard[3],np.repeat(np.arange(16),4),n_sites=36,seed=seed+2)
                d=selection['candidate_minus_control'];s=selection['paired_independent_block_jackknife_SE']
                if d[0]<=0 and d[2]<=2*s[2]:chosen,fraction,geometry=proposal,factor,g
                break
        save_checkpoint(out/'checkpoint.pkl',chosen,optimizer_state=moments if fraction else saved['optimizer_state'],
            metadata=dict(registration=registration,reference_checkpoint=str(directory/'initialized.pkl'),
                accepted_updates=int(bool(fraction)),fixed_scale=scale,audit_seed_reserved=seed))
        atomic_json(out/'endpoint_frozen.json',dict(parameter_sha256=pytree_sha256(chosen),
            checkpoint_sha256=file_sha256(out/'checkpoint.pkl'),no_audit_rows_seen=True))
        rows,draw=runtime['draw'](128,seed)
        atomic_json(out/'audit_draw.json',jax.tree.map(lambda x:np.asarray(x).tolist()
            if isinstance(x,(np.ndarray,np.generic)) else x,draw))
        bonds=runtime['bonds'];prob=np.abs([c for a,b,c in bonds]);prob/=prob.sum()
        action,_=make_spin_source_products(lambda e:make_feature_lut_exchange_queries(cfg,e),images,
            bonds,prob,np.asarray(phases)[None,:],n_sites=36,projection_chunk=4)
        parts=[]
        for i in range(0,128,8):
            parts.append(tuple(np.asarray(x) for x in jax.block_until_ready(action(ground,jnp.asarray(rows[i:i+8])))))
            print(tag,'protected audit sourceH',i+8,'/128',flush=True)
        phi,hp,qo,qpush=[np.concatenate([x[k] for x in parts]) for k in range(4)]
        Q=full_support_sz0_density(qo,qpush,n_sites=36,epsilon=.125)
        rp,rh=runtime['H'](params,rows);cp,ch=runtime['H'](chosen,rows)
        candidate_psi=anchored_amplitude(phi[:,1],cp,rp,scale)
        candidate_H=anchored_amplitude(hp[:,1],ch,rh,scale)
        moments_final=paired_iid_moments(np.column_stack((phi[:,1],candidate_psi)),
            np.column_stack((hp[:,1],candidate_H)),Q,np.repeat(np.arange(16),8),n_sites=36,seed=seed+1)
        np.savez_compressed(out/'terminal_products.npz',rows=rows,proposal=Q,
            psi=np.column_stack((phi[:,1],candidate_psi)),hpsi=np.column_stack((hp[:,1],candidate_H)))
        result=dict(status='completed_one_protected_physical_update',registration=registration,
            accepted_updates=int(bool(fraction)),fraction=fraction,finite_guard=geometry,selection=selection,
            SR_relative_residual=float(sr['relative_SR_residual']),SR_wall_seconds=sr_seconds,
            changed_families=family_norms(jax.tree.map(lambda a,b:a-b,chosen,params)),fixed_scale=scale,
            source_amplitude_replay_maximum_error=source_replay,moments=moments_final,
            wall_seconds=perf_counter()-started,not_a_converged_excitation_gap=True,
            final_audit_unused_for_update_or_endpoint_choice=True)
        atomic_json(out/'results.json',result)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();r=run(args.directory)
    print(json.dumps(dict(status=r['status'],accepted=r['accepted_updates'],
        energy=r['moments']['endpoint_values'],delta=r['moments']['candidate_minus_control'],wall=r['wall_seconds']),indent=2))
