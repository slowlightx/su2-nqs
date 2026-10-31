"""One all-native-W SR direction, supported finite-grid calibration and H/JVP check.

This isolated workflow never edits a production checkpoint or launches a job.
Fit4096/256, fresh4096 amplitude-only IID calibration/guard, then at most2048
fresh paired all-H-bridge records. No optimizer update is committed and no
training continuation is released. A nonzero candidate is provisional.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
import gc
import json
import os
from pathlib import Path
import shutil
from time import perf_counter,process_time,time

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from su2_nqs.models.triangular_clebsch_gauge import canonicalize_triangular_bank,triangular_raw_overlap
from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_amplitude,prepare_feature_lut_bank
from su2_nqs.models.triangular_mixed_clebsch import mixed_clebsch_layout
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint
from su2_nqs.vmc.exchange_bridge_importance import make_orbit_exchange_bridge_products
from su2_nqs.vmc.finite_candidate_calibration import select_finite_grid,bilateral_direction_moments
from su2_nqs.vmc.linear_orbit_refinement import make_orbit_amplitude_products
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.physical_sr_adam import make_physical_sr_adam
from su2_nqs.vmc.triangular_raw_born import make_triangular_raw_born_sampler
from . import triangle_clebsch_continuation as legacy
from .triangle_hierarchy_initialization import _node_timeout
from .triangle_iid_orbit_audit import paired_iid_moments

ROOT=Path(__file__).resolve().parents[3]
GRID=(1.,.5,.25,.125,0.)
TASK='CT6_one_all_native_W_supported_finite_grid_v1'


def validate_plan(plan:dict,*,executing:bool=False) -> None:
    """Reject altered work counts, stale/unsealed parents and accidental launches."""
    if (plan.get('task')!=TASK or plan.get('model')!=dict(length=6,multiplicity_cap=28,maximum_q=10)
            or plan.get('j2_over_j1')!=.125 or plan.get('arity')!='binary_truncated'
            or plan.get('binary_rank')!=8 or plan.get('fractions')!=list(GRID)
            or plan.get('fit')!=dict(samples=4096,chains=256,refresh_proposals=1476,spacing_proposals=36)
            or plan.get('calibration_records')!=4096 or plan.get('calibration_split')!=2048
            or plan.get('confirmation_records')!=2048 or plan.get('damping')!=.5
            or plan.get('target_FS')!=.001 or plan.get('maximum_FS')!=.005
            or plan.get('wall_cap_seconds')!=14400 or plan.get('maximum_peak_device_bytes')!=70*1024**3
            or plan.get('automatic_continuation') is not False):
        raise ValueError('only the fixed J.125 width28 one-direction4096/4096/2048 protocol is allowed')
    seeds=plan.get('seeds',{})
    if (set(seeds)!={'fit','selection','guard','confirmation','statistics'}
            or len(set(seeds.values()))!=5 or any(isinstance(v,bool) or not isinstance(v,int) or not 0<=v<2**32 for v in seeds.values())
            or set(seeds.values())&set(plan.get('parent_registered_seeds',[]))):
        raise ValueError('five distinct new uint32 streams, disjoint from all declared parent seeds, required')
    parent=plan.get('parent',{})
    if (parent.get('global_step')!=476 or set(parent.get('sealed_files',{}))!=
            {'checkpoint.pkl','registration.json','results.json','terminal_audit.json'}
            or any(not isinstance(v,str) or len(v)!=64 for v in parent['sealed_files'].values())
            or not isinstance(parent.get('params_sha256'),str) or len(parent['params_sha256'])!=64
            or not isinstance(parent.get('terminal_products_sha256'),str) or len(parent['terminal_products_sha256'])!=64):
        raise ValueError('actual audited global476 parent seals required')
    for field in ('projection_chunk','action_chunk','score_chunk','confirmation_chunk'):
        value=plan.get(field)
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError('positive integer execution chunks required')
    if executing and (plan.get('execution_authorized') is not True or not plan.get('authorization_note')):
        raise PermissionError('explicit root launch authority required; validation alone cannot release execution')


def checked_parent(plan:dict) -> tuple[dict,dict]:
    """Load full optimizer/RNG history while leaving the sealed parent immutable."""
    if not jax.config.x64_enabled:
        raise RuntimeError('enable JAXx64 BEFORE loading the trusted parent pickle; float32 reconstruction changes seals')
    directory=Path(plan['parent']['directory'])
    for name,digest in plan['parent']['sealed_files'].items():
        if file_sha256(directory/name)!=digest:raise ValueError('parent seal mismatch: '+name)
    if file_sha256(directory/'terminal_audit.npz')!=plan['parent']['terminal_products_sha256']:
        raise ValueError('parent terminal physical products seal mismatch')
    reg,seal=legacy._sealed_json(directory/'registration.json')
    result=json.loads((directory/'results.json').read_text());audit=json.loads((directory/'terminal_audit.json').read_text())
    state,meta=legacy.shared.restore_state(directory/'checkpoint.pkl',seal)
    if (result['status']!='fixed_steps_complete' or result['absolute_completed_steps']!=476
            or result['checkpoint_sha256']!=plan['parent']['sealed_files']['checkpoint.pkl']
            or result['params_sha256']!=plan['parent']['params_sha256']
            or audit['params_sha256']!=result['params_sha256'] or result['terminal_audit']!=audit
            or not audit['independent_from_training_and_calibration']
            or pytree_sha256(state[0])!=result['params_sha256'] or meta['absolute_completed_steps']!=476
            or reg['model']!=plan['model'] or reg['arity']!=plan['arity'] or reg['j2_over_j1']!=.125):
        raise ValueError('parent endpoint/audit/native state identity mismatch')
    return load_checkpoint(directory/'checkpoint.pkl'),dict(registration_sha256=seal,
        parent_optimizer_sha256=pytree_sha256(state[1]),parent_sampler_rng_sha256=pytree_sha256(state[2:]),
        native_parameter_count=sum(np.size(x) for x in jax.tree.leaves(state[0])))


def runtime_plan(plan:dict) -> dict:
    """Use production kernels without invoking or modifying the sealed campaign."""
    return dict(task=legacy.EVIDENCE_GUARDED_TASK,model=plan['model'],arity=plan['arity'],binary_rank=8,
        j2_over_j1=.125,projection_chunk=plan['projection_chunk'],action_chunk=plan['action_chunk'],
        score_chunk=plan['score_chunk'],damping=.5,maximum_fs_sine=.005,minimum_ess_fraction=.98,
        parameter_chunk_size=8192,sr_solver='dual_force',sr_use_adam=False,sr_tangent_fs_radius=.003,
        training=plan['fit'],calibration=dict(samples=2048,chains=128,refresh_proposals=1440,spacing_proposals=36,
            target_fs_sine=.001,relative_tolerance=.02,max_factor=2.,bracket_steps=8,bisection_steps=12,minimum_ess_fraction=.98))


def normalized_raw_bank(native:dict,rt:dict) -> tuple[dict,dict]:
    """Physical-state-preserving QR is sampler-only; never replace native W."""
    folded=rt['bank'](native);norm=triangular_raw_overlap(folded,folded,rt['cfg'])
    if not np.isfinite(norm) or abs(norm.imag)>1e-12*abs(norm.real) or norm.real<=0:
        raise ArithmeticError('finite positive exact raw norm required')
    canonical,notes=canonicalize_triangular_bank(folded,rt['cfg'])
    canonical=dict(canonical,top=tuple(canonical['top'][:-1])+
        (jax.tree.map(lambda x:np.asarray(x)/np.sqrt(norm.real),canonical['top'][-1]),))
    check=triangular_raw_overlap(canonical,canonical,rt['cfg'])
    cross=triangular_raw_overlap(canonical,folded,rt['cfg'])/np.sqrt(norm.real)
    if max(abs(check-1),abs(cross-1))>2e-11:raise ArithmeticError('sampler-only canonical gauge changed physical state')
    return canonical,dict(notes,original_raw_norm=float(norm.real),normalized_raw_norm=float(check.real),
        raw_cross_overlap_error=float(abs(cross-1)),native_optimizer_coordinates_unchanged=True)


def draw_orbit_rows(banks:list[dict],rt:dict,count:int,seed:int,*,bridge:bool=False) -> tuple[np.ndarray,dict]:
    """Exact normalized raw Born, uniform inverse image, optional all-edge push."""
    profile=[row['widths'] for row in mixed_clebsch_layout(rt['cfg'])]
    cache={};samplers=[make_triangular_raw_born_sampler(b,rt['cfg'],active_top_widths=profile,
        memory_limit_bytes=2*1024**3,shared_tensor_cache=cache) for b in banks]
    images=np.asarray(rt['images']);inverse=np.argsort(images,axis=1)
    edges=np.asarray([(a,b) for a,b,_ in rt['bonds']]);prob=np.abs([c for _,_,c in rt['bonds']]);prob/=prob.sum()
    rng=np.random.default_rng(seed);rows=[];latent=[];errors=[]
    for _ in range(count):
        source=int(rng.integers(len(banks)));draw=samplers[source].draw(rng)
        image=int(rng.integers(len(images)));row=draw.bits[inverse[image]].copy();use=False;bond=-1
        if bridge:
            use=bool(rng.random()<.5);bond=int(rng.choice(len(edges),p=prob))
            if use:
                a,b=edges[bond];row[a],row[b]=row[b],row[a]
        rows.append(row);latent.append((source,image,int(use),bond));errors.append(draw.log_probability_replay_error)
    report=dict(latent=np.asarray(latent),Born_replay=np.asarray(errors),rng_state=rng.bit_generator.state,
        maximum_Born_replay_error=max(errors),sampler_diagnostics=[s.diagnostics for s in samplers])
    return np.asarray(rows,np.int32),report


def _device_peak() -> int:
    return max(int((d.memory_stats() or {}).get('peak_bytes_in_use',0)) for d in jax.local_devices())


def allocation_science_seconds(cap:float,end_time:str|None,now:float) -> float:
    """Respect the scheduler's absolute deadline, reserving120s for saving.

    This does not enlarge the registered science cap. A missing scheduler
    deadline keeps that cap; an invalid or exhausted deadline fails closed.
    Explicit clock arguments make the arithmetic independently testable.
    """
    if not np.isfinite([cap,now]).all() or cap<=0:raise ValueError('finite positive science cap and clock required')
    available=float(cap)
    if end_time is not None:
        try:end=float(end_time)
        except (TypeError,ValueError) as error:raise ValueError('invalid SLURM_JOB_END_TIME') from error
        if not np.isfinite(end):raise ValueError('finite SLURM_JOB_END_TIME required')
        available=min(available,end-now-120.)
    if available<=0:raise TimeoutError('allocation has no science time after120s serialization margin')
    return available


def audited_parent_replay(rt:dict,parent:dict,path:Path) -> dict:
    """Two saved physical rows, same complete A1/H, no newly sampled data."""
    with np.load(path) as stored:
        rows=stored['bits'][:2]
        expected=stored['local_energies'].reshape(-1)[:2]
    if rows.shape!=(2,36) or expected.shape!=(2,) or not np.isfinite(expected).all():
        raise ValueError('two finite saved CT6 full-H energy records required')
    psi,hp=jax.block_until_ready(rt['action'](rt['prepare'](parent),rows))
    if np.any(np.asarray(psi)==0):raise ArithmeticError('saved audit parent unexpectedly at a node')
    actual=np.asarray(hp/psi)
    relative=float(np.max(abs(actual-expected))/max(float(np.max(abs(expected))),1e-300))
    if not np.isfinite(actual).all() or relative>2e-11:
        raise ArithmeticError('saved complete-A1 parent H replay exceeds2e-11')
    return dict(records=2,full_H_centers=2,maximum_relative_error=relative,
        expected_total_local_energies=expected.tolist(),actual_total_local_energies=actual.real.tolist(),
        maximum_imaginary_component=float(np.max(abs(actual.imag))),tolerance=2e-11)


def native_projected_amplitude(rt:dict,params:dict,rows:np.ndarray|jax.Array) -> jax.Array:
    """Batch the ONE-row orbit API, converting host bits before traced indexing."""
    bits=jnp.asarray(rows,dtype=jnp.int32)
    if bits.ndim!=2 or bits.shape[1]!=36:raise ValueError('CT6 physical bit rows must have shape(batch,36)')
    return jax.vmap(rt['projected'],in_axes=(None,0))(rt['prepare'](params),bits)


def finish_frozen_calibration(plan:dict,output:Path,rt:dict,saved:dict,direction:dict,
    native_candidates:list,banks:list,calrows:np.ndarray,normalization:list,r:dict,
    timed:Callable,start:float,available:float) -> dict:
    """Same frozen-grid amplitude calibration and fresh confirmation in both paths."""
    parent=saved['params'];moments=saved['optimizer_state'];delta=direction['params']
    next_moments=direction['optimizer_state'];key=direction['metadata']['fit_key'];states=direction['metadata']['fit_states']
    prepared=[prepare_feature_lut_bank(b,rt['cfg']) for b in banks]
    orbit=make_orbit_amplitude_products(make_feature_lut_amplitude(rt['cfg']),rt['images'],projection_chunk=plan['projection_chunk'])
    # Scalar microbatches share compilation while keeping full image sums bounded.
    query=jax.jit(lambda bank,rows:jax.lax.map(lambda row:jnp.stack(orbit(bank,row[None]))[:,0],rows,batch_size=plan['action_chunk']))
    calp=[];calguide=[]
    for i,b in enumerate(prepared):
        r['work']['calibration_bank_centers_requested']=(i+1)*4096
        values=timed('calibration_bank_'+str(i),lambda b=b:query(b,calrows))
        calp.append(np.asarray(values[:,0]));calguide.append(np.asarray(values[:,1]).real)
        r['work']['calibration_bank_centers']+=4096
    calp=np.stack(calp,axis=1);guides=np.stack(calguide,axis=1);density=guides.mean(1)
    np.savez(output/'calibration_products.npz',rows=calrows,phi=calp,guide_densities=guides,proposal=density)
    replay_errors=[]
    for i,p in enumerate(native_candidates):
        value=timed('native_calibration_replay_'+str(i),lambda p=p:native_projected_amplitude(rt,p,calrows[:2]))
        value=np.asarray(value)/np.sqrt(normalization[i]['original_raw_norm'])
        error=float(np.max(abs(value-calp[:2,i]))/max(float(np.max(abs(calp[:2,i]))),1e-300))
        r['work']['native_calibration_replay_amplitude_centers']+=2;replay_errors.append(error)
        if not np.isfinite(error) or error>2e-11:raise ArithmeticError('native finite/canonical coherent amplitudes disagree')
    r['native_calibration_replay_relative_errors']=replay_errors
    selected,calibration=select_finite_grid(calp,density,np.asarray(GRID),split=2048)
    r['calibration']=calibration;r['selected_index']=selected
    candidate=native_candidates[selected]
    save_checkpoint(output/'provisional_candidate.pkl',candidate,optimizer_state=moments,
        metadata=dict(parent_params_sha256=plan['parent']['params_sha256'],parent_checkpoint_sha256=plan['parent']['sealed_files']['checkpoint.pkl'],
            selected_fraction=GRID[selected],original_parent_metadata=saved['metadata'],fit_key=np.asarray(key),fit_states=np.asarray(states),
            proposed_optimizer_state=next_moments,production_commit_requires_separate_authority=True,
            optimizer_state_is_unchanged_parent_snapshot=True,calibration_only_not_energy_audit=True))
    r['candidate_sha256']=file_sha256(output/'provisional_candidate.pkl')
    atomic_json(output/'candidate_frozen.json',dict(r,wall_seconds=perf_counter()-start))
    if selected==4:
        r['status']='geometry_parent_fallback_no_confirmation_H';return r
    print('candidate frozen; independent paired bridge/H-JVP validation next',flush=True)
    endpoints=[banks[4],banks[selected]];rows,draw=timed('confirmation_IID_draws',lambda:draw_orbit_rows(
        endpoints,rt,2048,plan['seeds']['confirmation'],bridge=True))
    np.savez(output/'confirmation_rows.npz',rows=rows,latent=draw.pop('latent'),Born_replay=draw.pop('Born_replay'))
    atomic_json(output/'confirmation_draw.json',draw)
    probability=np.abs([c for _,_,c in rt['bonds']]);probability/=probability.sum()
    bridge,bridge_meta=make_orbit_exchange_bridge_products(lambda edges:make_feature_lut_exchange_queries(rt['cfg'],edges),
        rt['images'],rt['bonds'],probability,n_sites=36,projection_chunk=plan['projection_chunk'])
    r['bridge_kernel']=bridge_meta
    endpoint_prepared=[prepared[4],prepared[selected]]
    jvp=jax.jit(lambda p,d,x:jax.jvp(lambda z:rt['action'](rt['prepare'](z),x),(p,),(d,)))
    actual_delta=jax.tree.map(lambda a:GRID[selected]*a,delta)
    raw_scale=np.sqrt(normalization[4]['original_raw_norm'])
    data=[];derivatives=[];batch_times=[];chunk=plan['confirmation_chunk']
    for offset in range(0,len(rows),chunk):
        x=rows[offset:offset+chunk];tick=perf_counter();values=[]
        for b in endpoint_prepared:
            r['work']['confirmation_scalar_H_centers_requested']=r['work']['confirmation_scalar_H_centers']+len(x)
            values.append(np.stack([np.asarray(v) for v in bridge(b,x)],axis=1))
            r['work']['confirmation_scalar_H_centers']+=len(x)
        r['work']['confirmation_H_JVP_centers_requested']=r['work']['confirmation_H_JVP_centers']+len(x)
        (pp,hh),(dp,hd)=jax.block_until_ready(jvp(parent,actual_delta,x))
        r['work']['confirmation_H_JVP_centers']+=len(x)
        value=np.stack(values,axis=1);replay=np.stack((np.asarray(pp),np.asarray(hh)),axis=1)/raw_scale
        if np.max(abs(replay-value[:,0,:2]))>2e-11*max(float(np.max(abs(value[:,0,:2]))),1e-300):
            raise ArithmeticError('native JVP parent and canonical bridge scalar products disagree')
        data.append(value);derivatives.append(np.stack((np.asarray(dp),np.asarray(hd)),axis=1)/raw_scale)
        batch_times.append(perf_counter()-tick)
        if _device_peak()>plan['maximum_peak_device_bytes']:raise MemoryError('confirmation peak exceeded70GiB')
        if offset==chunk:
            forecast=perf_counter()-start+batch_times[1]*((len(rows)-offset-len(x))/len(x))+120
            r['confirmation_cost_gate']=dict(cold_batch_seconds=batch_times[0],warm_batch_seconds=batch_times[1],batch_records=len(x),
                conservative_total_seconds=forecast,peak_device_bytes=_device_peak())
            atomic_json(output/'confirmation_cost_gate.json',r['confirmation_cost_gate'])
            if forecast>available:
                r['status']='cost_gate_stopped_confirmation_incomplete';break
        if (offset//chunk)%16==0:
            atomic_json(output/'progress.json',dict(r,wall_seconds=perf_counter()-start))
            print('paired completeH/JVP records',offset+len(x),'/2048',flush=True)
    data=np.concatenate(data);derivatives=np.concatenate(derivatives)
    q=(data[:,:,2].real+data[:,:,3].real).mean(1)/2
    np.savez(output/'confirmation_products.npz',rows=rows[:len(data)],psi=data[:,:,:1][:,:,0],hpsi=data[:,:,1],
        guide_components=data[:,:,2:],proposal=q,dpsi=derivatives[:,0],hdpsi=derivatives[:,1])
    r['timings']['confirmation_batches']=batch_times
    if len(data)!=2048:return r
    group=np.repeat(np.arange(128),16)
    r['finite_step_moments']=paired_iid_moments(data[:,:,0],data[:,:,1],q,group,n_sites=36,seed=plan['seeds']['statistics'])
    r['bilateral_direction']=bilateral_direction_moments(data[:,0,0],data[:,0,1],derivatives[:,0],derivatives[:,1],q)
    slope=r['bilateral_direction']['values'][2];slope_se=r['bilateral_direction']['independent_group_jackknife_SE'][2]
    delta_e=r['finite_step_moments']['candidate_minus_control'][0];delta_se=r['finite_step_moments']['paired_iid_delta_SE'][0]
    r.update(status='complete_provisional_native_W_validation_no_training_release',
        resolved_ascent=bool(slope-2*slope_se>0),resolved_finite_energy_increase=bool(delta_e-2*delta_se>0),
        actual_all_native_W_changed=pytree_sha256(candidate)!=pytree_sha256(parent),
        no_automatic_short_training=True,no_final_untouched_energy_claim=True)
    return r


def run(plan:dict,output:Path) -> dict:
    """Prepare one provisional state only; hard deadline, complete work receipts."""
    validate_plan(plan,executing=True)
    if jax.default_backend()!='gpu' or not any('A100' in d.device_kind for d in jax.devices()):
        raise RuntimeError('actual4096-score run requires the approved A100 runtime; local smoke is test-only')
    if output.exists() and any(output.iterdir()):raise FileExistsError('fresh isolated output required')
    saved,parent_info=checked_parent(plan);output.mkdir(parents=True,exist_ok=True)
    sources=transitive_local_source_manifest([Path(__file__)],root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    registration=dict(plan=plan,plan_sha256=canonical_sha256(plan),sources=sources,parent=parent_info,
        runtime=legacy.numerical_runtime_fingerprint(),not_a_terminal_energy_audit=True,no_optimizer_updates_committed=True)
    atomic_json(output/'registration.json',registration)
    for name in (*plan['parent']['sealed_files'],'terminal_audit.npz'):
        target=output/'parent_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(plan['parent']['directory'])/name,target)
    for name in sources:
        target=output/'registered_sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    start=perf_counter();cpu=process_time();r=dict(status='started',registration=registration,timings={},
        work=dict(parent_replay_H_centers=0,fit_H_centers=0,calibration_bank_centers=0,
            native_calibration_replay_amplitude_centers=0,confirmation_scalar_H_centers=0,confirmation_H_JVP_centers=0))
    def timed(label,call):
        tick=perf_counter();value=call();jax.block_until_ready(value)
        r['timings'][label]=perf_counter()-tick
        if _device_peak()>plan['maximum_peak_device_bytes']:raise MemoryError('registered70GiB device ceiling exceeded')
        atomic_json(output/'progress.json',dict(r,operation=label,wall_seconds=perf_counter()-start))
        return value
    try:
        available=allocation_science_seconds(plan['wall_cap_seconds'],os.environ.get('SLURM_JOB_END_TIME'),time())
        r['effective_science_wall_cap_seconds']=available
        r['scheduler_serialization_margin_seconds']=120
        with _node_timeout(available):
            parent=saved['params'];moments=saved['optimizer_state'];rp=runtime_plan(plan);rt=legacy.runtime(rp,parent)
            r['work']['parent_replay_H_centers_requested']=2
            r['parent_numerical_replay']=timed('parent_complete_H_replay',lambda:audited_parent_replay(
                rt,parent,output/'parent_snapshot'/'terminal_audit.npz'))
            r['work']['parent_replay_H_centers']=2
            atomic_json(output/'parent_numerical_replay.json',r['parent_numerical_replay'])
            key,init=jax.random.split(jax.random.PRNGKey(plan['seeds']['fit']))
            states=initialize_spin_exchange_chains(init,n_chains=256,n_sites=36,n_up=18)
            sampler=legacy.shared.make_record_sampler(rt['projected'],plan['fit'],rt['edges'])
            key,states,fitrows=timed('fit_sampling',lambda:sampler(rt['prepare'](parent),key,states))
            r['work']['fit_H_centers_requested']=4096
            psi,hp=timed('fit_full_H',lambda:rt['action'](rt['prepare'](parent),fitrows));r['work']['fit_H_centers']=4096
            if np.any(np.asarray(psi)==0) or not np.isfinite(np.asarray(hp/psi)).all():raise ArithmeticError('invalid fit Born ratio; no floor')
            update=make_physical_sr_adam(lambda p,x:rt['projected'](rt['prepare'](p),x),parent,
                damping=.5,score_chunk=plan['score_chunk'],maximum_fs_sine=.005,minimum_ess_fraction=.98,
                prepare=rt['prepare'],prepared_amplitude=rt['projected'],sr_solver='dual_force',parameter_chunk_size=8192,
                sr_use_adam=False,sr_tangent_fs_radius=.003)
            r['work']['full_native_score_rows_requested']=4096
            r['work']['native_score_parameter_count']=parent_info['native_parameter_count']
            proposed,next_moments,sr=timed('all_native_SR',lambda:update(parent,moments,fitrows,hp/psi,.03,warm=True))
            r['fit_SR']=jax.tree.map(lambda x:np.asarray(x).item(),sr)
            delta=jax.tree.map(lambda b,a:b-a,proposed,parent)
            r['native_direction_block_norms']={key:float(np.sqrt(sum(np.vdot(np.asarray(x),np.asarray(x)).real
                for x in jax.tree.leaves(value)))) for key,value in delta.items()}
            save_checkpoint(output/'frozen_direction.pkl',delta,optimizer_state=next_moments,
                metadata=dict(parent_params_sha256=plan['parent']['params_sha256'],fit_key=np.asarray(key),fit_states=np.asarray(states),
                    original_parent_optimizer_unchanged=True,proposed_optimizer_uncommitted=True,all_native_parameter_columns=True))
            np.savez(output/'fit_products.npz',rows=fitrows,psi=psi,hpsi=hp)
            if not bool(sr['accepted']):
                r['status']='fit_SR_rejected_no_calibration_or_confirmation';return r
            native_candidates=[jax.tree.map(lambda a,d:a+s*d,parent,delta) for s in GRID]
            banks=[];normalization=[]
            for i,p in enumerate(native_candidates):
                bank,note=normalized_raw_bank(p,rt);banks.append(bank);normalization.append(note)
                save_checkpoint(output/f'finite_native_{i}.pkl',p,optimizer_state=moments,
                    metadata=dict(fraction=GRID[i],provisional_uncommitted=True,parent_optimizer_snapshot=True))
                save_checkpoint(output/f'finite_canonical_{i}.pkl',bank,metadata=note)
            r['normalization']=normalization
            atomic_json(output/'finite_grid_frozen.json',dict(r,direction_sha256=file_sha256(output/'frozen_direction.pkl'),
                native_checkpoint_sha256=[file_sha256(output/f'finite_native_{i}.pkl') for i in range(5)]))
            print('fullnative direction and5finite candidates frozen',flush=True)
            calrows=[]
            for label in ('selection','guard'):
                rows,draw=timed(label+'_IID_draws',lambda label=label:draw_orbit_rows(banks,rt,2048,plan['seeds'][label]))
                np.savez(output/f'{label}_rows.npz',rows=rows,latent=draw.pop('latent'),Born_replay=draw.pop('Born_replay'))
                atomic_json(output/f'{label}_draw.json',draw);calrows.append(rows)
            return finish_frozen_calibration(plan,output,rt,saved,load_checkpoint(output/'frozen_direction.pkl'),
                native_candidates,banks,np.concatenate(calrows),normalization,r,timed,start,available)
    except (TimeoutError,MemoryError) as error:
        r.update(status='bounded_incomplete',error=repr(error))
    except Exception as error:
        r.update(status='failed_validation',error=repr(error))
        raise
    finally:
        r.update(wall_seconds=perf_counter()-start,CPU_seconds=process_time()-cpu,peak_device_bytes=_device_peak())
        r['source_seals_unchanged']={name:file_sha256(ROOT/name)==seal for name,seal in sources.items()}
        seals=dict(plan['parent']['sealed_files'],**{'terminal_audit.npz':plan['parent']['terminal_products_sha256']})
        r['parent_seals_unchanged']={name:file_sha256(Path(plan['parent']['directory'])/name)==seal for name,seal in seals.items()}
        intact=all(r['source_seals_unchanged'].values()) and all(r['parent_seals_unchanged'].values())
        if not intact:r.update(status='failed_immutable_source_or_parent_seal',immutable_seal_failure=True)
        atomic_json(output/'results.json',r)
        if not intact:raise ArithmeticError('registered numerical sources or sealed parent changed during execution')
    return r


def resume_source_manifest(prior:Path) -> dict[str,str]:
    """Read-only seal of ALL files in one immutable failed diagnostic output."""
    if not prior.is_dir():raise FileNotFoundError(prior)
    paths=sorted(prior.rglob('*'))
    if any(p.is_symlink() for p in paths):raise ValueError('resume source must not contain mutable symbolic links')
    return {str(p.relative_to(prior)):file_sha256(p) for p in paths if p.is_file()}


def checked_resume_source(prior:Path,manifest:dict[str,str]) -> tuple[dict,dict,dict]:
    """Only the known post-calibration/pre-replay failure is resumable here."""
    if resume_source_manifest(prior)!=manifest:raise ValueError('immutable resume file set/seals changed')
    reg=json.loads((prior/'registration.json').read_text())
    failed=json.loads((prior/'results.json').read_text())
    grid=json.loads((prior/'finite_grid_frozen.json').read_text());plan=reg['plan']
    validate_plan(plan,executing=True)
    work=failed['work']
    if (failed['status']!='failed_validation' or not failed.get('error','').startswith('TracerArrayConversionError(')
            or failed['registration']!=reg or grid['registration']!=reg
            or canonical_sha256(plan)!=reg['plan_sha256']
            or not failed['fit_SR']['accepted']
            or work['fit_H_centers']!=4096 or work['calibration_bank_centers']!=20480
            or work['native_calibration_replay_amplitude_centers']!=0
            or work['confirmation_scalar_H_centers']!=0 or work['confirmation_H_JVP_centers']!=0
            or not all(failed['source_seals_unchanged'].values())
            or not all(failed['parent_seals_unchanged'].values())
            or any((prior/name).exists() for name in ('candidate_frozen.json','provisional_candidate.pkl',
                'confirmation_rows.npz','confirmation_draw.json','confirmation_products.npz','calibration_products.npz'))):
        raise ValueError('not the registered unreselected failed calibration boundary')
    if manifest['frozen_direction.pkl']!=grid['direction_sha256']:
        raise ValueError('saved direction differs from original frozen seal')
    for i,digest in enumerate(grid['native_checkpoint_sha256']):
        if manifest[f'finite_native_{i}.pkl']!=digest:raise ValueError('saved native finite state differs from original frozen seal')
    for name,digest in reg['sources'].items():
        if manifest['registered_sources/'+name]!=digest:raise ValueError('original source archive seal mismatch')
    return reg,failed,grid


def resume(prior:Path,output:Path,manifest:dict[str,str],*,wall_cap_seconds:float=10800.) -> dict:
    """Resume fixed saved rows only: zero refits/SR/calibration draws.

    The old attempt remains read-only. Its full incurred work is carried in
    the new receipt; repeated20480 amplitude-bank centers are charged again.
    Only the original, previously unused confirmation seed may draw new rows.
    """
    if not jax.config.x64_enabled:raise RuntimeError('enable x64 before resume checkpoint loading')
    if not np.isfinite(wall_cap_seconds) or not 0<wall_cap_seconds<=10800:
        raise ValueError('resume ceiling is at most3h')
    if jax.default_backend()!='gpu' or not any('A100' in d.device_kind for d in jax.devices()):
        raise RuntimeError('approved A100 runtime required for resume')
    prior=prior.resolve();output=output.resolve()
    if output==prior or prior in output.parents or output in prior.parents:
        raise ValueError('resume output must be isolated from the immutable failed output')
    if output.exists() and any(output.iterdir()):raise FileExistsError('new empty resume output required')
    old,failed,grid=checked_resume_source(prior,manifest);plan=old['plan']
    parent_plan=dict(plan,parent=dict(plan['parent'],directory=str(prior/'parent_snapshot')))
    saved,parent_info=checked_parent(parent_plan)
    sources=transitive_local_source_manifest([Path(__file__)],root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    changed=sorted(name for name in set(sources)|set(old['sources']) if sources.get(name)!=old['sources'].get(name))
    if changed!=[str(Path(__file__).resolve().relative_to(ROOT))]:
        raise ValueError('only this workflow may change for the execution-only resume repair: '+repr(changed))
    output.mkdir(parents=True,exist_ok=True)
    registration=dict(plan=plan,plan_sha256=old['plan_sha256'],sources=sources,parent=parent_info,
        runtime=legacy.numerical_runtime_fingerprint(),not_a_terminal_energy_audit=True,no_optimizer_updates_committed=True,
        resume_from=str(prior),resume_source_manifest=manifest,resume_source_manifest_sha256=canonical_sha256(manifest),
        prior_sources=old['sources'],source_version_transition=changed,resume_wall_cap_seconds=wall_cap_seconds,
        no_refit_no_SR_no_new_calibration_draws=True,original_unused_confirmation_seed=plan['seeds']['confirmation'])
    for name in manifest:
        if name.startswith('registered_sources/'):
            destination=output/'prior_registered_sources'/name.removeprefix('registered_sources/')
        elif name in ('registration.json','results.json','progress.json','parent_numerical_replay.json'):
            destination=output/'prior_attempt'/name
        else:destination=output/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(prior/name,destination)
        if file_sha256(destination)!=manifest[name]:raise ValueError('resume copy changed a sealed input: '+name)
    for name in sources:
        destination=output/'registered_sources'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    atomic_json(output/'registration.json',registration)
    start=perf_counter();cpu=process_time()
    r=dict(status='resume_started',registration=registration,timings={},prior_attempt_wall_seconds=failed['wall_seconds'],
        prior_attempt_CPU_seconds=failed['CPU_seconds'],prior_attempt_work=failed['work'],
        work=dict(parent_replay_H_centers=0,fit_H_centers=0,full_native_score_rows_requested=0,
            calibration_bank_centers=0,native_calibration_replay_amplitude_centers=0,
            confirmation_scalar_H_centers=0,confirmation_H_JVP_centers=0),
        fit_SR=failed['fit_SR'],native_direction_block_norms=failed['native_direction_block_norms'],
        normalization=failed['normalization'],prior_parent_numerical_replay=failed['parent_numerical_replay'],
        calibration_rows_reused=4096,new_calibration_draws=0)
    def timed(label,call):
        tick=perf_counter();value=call();jax.block_until_ready(value);r['timings'][label]=perf_counter()-tick
        if _device_peak()>plan['maximum_peak_device_bytes']:raise MemoryError('registered70GiB device ceiling exceeded')
        atomic_json(output/'progress.json',dict(r,operation=label,wall_seconds=perf_counter()-start));return value
    try:
        total_remaining=plan['wall_cap_seconds']-failed['wall_seconds']-120.
        if total_remaining<=0:raise TimeoutError('original4h cumulative budget is exhausted')
        available=allocation_science_seconds(min(wall_cap_seconds,total_remaining),os.environ.get('SLURM_JOB_END_TIME'),time())
        r['effective_science_wall_cap_seconds']=available;r['scheduler_serialization_margin_seconds']=120
        with _node_timeout(available):
            direction=load_checkpoint(output/'frozen_direction.pkl')
            if direction['metadata']['parent_params_sha256']!=plan['parent']['params_sha256']:
                raise ValueError('saved direction is not about the frozen audited parent')
            native=[];banks=[];native_errors=[]
            for i in range(5):
                item=load_checkpoint(output/f'finite_native_{i}.pkl')
                if item['metadata']['fraction']!=GRID[i] or pytree_sha256(item['optimizer_state'])!=parent_info['parent_optimizer_sha256']:
                    raise ValueError('native finite state or uncommitted optimizer differs')
                expected=jax.tree.map(lambda p,d:p+GRID[i]*d,saved['params'],direction['params'])
                check=legacy.compare_replay(expected,item['params'],tolerance=2e-11)
                if not check['passed']:raise ArithmeticError('saved native finite direction algebra does not replay')
                native_errors.append(check);native.append(item['params'])
                item=load_checkpoint(output/f'finite_canonical_{i}.pkl')
                if item['metadata']!=failed['normalization'][i]:raise ValueError('canonical normalization receipt changed')
                banks.append(item['params'])
            calrows=[]
            for label in ('selection','guard'):
                with np.load(output/f'{label}_rows.npz') as draw:
                    rows=draw['rows']
                    if (rows.shape!=(2048,36) or not np.issubdtype(rows.dtype,np.integer)
                            or not np.all((rows==0)|(rows==1)) or not np.all(rows.sum(1)==18)
                            or not np.isfinite(draw['Born_replay']).all()):raise ValueError('invalid saved calibration rows')
                    calrows.append(rows.copy())
            r['saved_native_direction_replay']=native_errors
            rt=legacy.runtime(runtime_plan(plan),saved['params'])
            r['work']['parent_replay_H_centers_requested']=2
            r['parent_numerical_replay']=timed('resume_parent_complete_H_replay',lambda:audited_parent_replay(
                rt,saved['params'],output/'parent_snapshot'/'terminal_audit.npz'))
            r['work']['parent_replay_H_centers']=2
            atomic_json(output/'parent_numerical_replay.json',r['parent_numerical_replay'])
            return finish_frozen_calibration(plan,output,rt,saved,direction,native,banks,np.concatenate(calrows),
                failed['normalization'],r,timed,start,available)
    except (TimeoutError,MemoryError) as error:r.update(status='bounded_incomplete',error=repr(error))
    except Exception as error:
        r.update(status='failed_validation',error=repr(error));raise
    finally:
        r.update(wall_seconds=perf_counter()-start,CPU_seconds=process_time()-cpu,peak_device_bytes=_device_peak())
        r['cumulative_wall_seconds']=failed['wall_seconds']+r['wall_seconds']
        r['cumulative_completed_work']={name:failed['work'].get(name,0)+r['work'].get(name,0) for name in (
            'parent_replay_H_centers','fit_H_centers','calibration_bank_centers','native_calibration_replay_amplitude_centers',
            'confirmation_scalar_H_centers','confirmation_H_JVP_centers')}
        r['source_seals_unchanged']={name:file_sha256(ROOT/name)==digest for name,digest in sources.items()}
        r['prior_attempt_all_file_seals_unchanged']=resume_source_manifest(prior)==manifest
        seals=dict(plan['parent']['sealed_files'],**{'terminal_audit.npz':plan['parent']['terminal_products_sha256']})
        r['parent_seals_unchanged']={name:file_sha256(output/'parent_snapshot'/name)==digest for name,digest in seals.items()}
        intact=all(r['source_seals_unchanged'].values()) and r['prior_attempt_all_file_seals_unchanged'] and all(r['parent_seals_unchanged'].values())
        if not intact:r.update(status='failed_immutable_source_or_parent_seal',immutable_seal_failure=True)
        atomic_json(output/'results.json',r)
        if not intact:raise ArithmeticError('resume source or parent seals changed')
    return r


def replay(output:Path) -> dict:
    """Separate-process eight-record numerical/RNG replay, never another fit.

    Charges16 scalarH and8 full-H JVP centers, plus8 exact raw draws. Original
    fit, candidate selection and confirmation remain unchanged. This validates
    saved-state/runtime replay, not the scientific reliability of the update.
    """
    if not jax.config.x64_enabled:raise RuntimeError('x64 must be enabled before replay loading')
    if (output/'process_replay.json').exists():raise FileExistsError('immutable replay already exists')
    record=json.loads((output/'results.json').read_text());reg=record['registration'];plan=reg['plan']
    if record['status']!='complete_provisional_native_W_validation_no_training_release':
        raise ValueError('completed fixed confirmation required for physical replay')
    for name,digest in reg['sources'].items():
        if file_sha256(ROOT/name)!=digest:raise ValueError('registered numerical source changed: '+name)
    replay_plan=dict(plan,parent=dict(plan['parent'],directory=str(output/'parent_snapshot')))
    saved,info=checked_parent(replay_plan);parent=saved['params'];start=perf_counter()
    remaining=int(allocation_science_seconds(plan['wall_cap_seconds']-record.get('cumulative_wall_seconds',record['wall_seconds']),
        os.environ.get('SLURM_JOB_END_TIME'),time()))
    if remaining<60:raise TimeoutError('no replay budget remains within the SAME4h numerical allocation')
    result=dict(status='started',optimizer_updates_committed=0,full_H_scalar_centers=0,full_H_JVP_centers=0,
        wall_cap_seconds=remaining,shared_original_four_hour_budget=True)
    try:
        with _node_timeout(remaining):
            rt=legacy.runtime(runtime_plan(plan),parent);selected=record['selected_index']
            checkpoint=load_checkpoint(output/'provisional_candidate.pkl')
            if file_sha256(output/'provisional_candidate.pkl')!=record['candidate_sha256']:
                raise ValueError('provisional checkpoint changed')
            direction=load_checkpoint(output/'frozen_direction.pkl')['params']
            expected=jax.tree.map(lambda p,d:p+GRID[selected]*d,parent,direction)
            parameter_replay=legacy.compare_replay(expected,checkpoint['params'],tolerance=2e-11)
            if not parameter_replay['passed']:raise ArithmeticError('native finite parameter curve replay failed')
            if pytree_sha256(checkpoint['optimizer_state'])!=info['parent_optimizer_sha256']:
                raise ArithmeticError('uncommitted parent optimizer snapshot changed')
            banks=[load_checkpoint(output/f'finite_canonical_{i}.pkl')['params'] for i in (4,selected)]
            rows,draw=draw_orbit_rows(banks,rt,8,plan['seeds']['confirmation'],bridge=True)
            original=np.load(output/'confirmation_products.npz')
            if not np.array_equal(rows,original['rows'][:8]):raise ArithmeticError('exact raw IID RNG trajectory changed')
            probability=np.abs([c for _,_,c in rt['bonds']]);probability/=probability.sum()
            bridge,_=make_orbit_exchange_bridge_products(lambda edges:make_feature_lut_exchange_queries(rt['cfg'],edges),
                rt['images'],rt['bonds'],probability,n_sites=36,projection_chunk=plan['projection_chunk'])
            values=np.stack([np.stack([np.asarray(v) for v in bridge(prepare_feature_lut_bank(b,rt['cfg']),rows)],axis=1)
                for b in banks],axis=1);result['full_H_scalar_centers']=16
            delta=jax.tree.map(lambda d:GRID[selected]*d,direction)
            jvp=jax.jit(lambda p,d,x:jax.jvp(lambda z:rt['action'](rt['prepare'](z),x),(p,),(d,)))
            (psi,hp),(dp,hd)=jax.block_until_ready(jvp(parent,delta,rows));result['full_H_JVP_centers']=8
            scale=np.sqrt(record['normalization'][4]['original_raw_norm'])
            actual=dict(psi=values[:,:,0],hpsi=values[:,:,1],guide_components=values[:,:,2:],
                proposal=(values[:,:,2].real+values[:,:,3].real).mean(1)/2,dpsi=np.asarray(dp)/scale,hdpsi=np.asarray(hd)/scale)
            errors={k:float(np.linalg.norm(v-original[k][:8])/max(np.linalg.norm(original[k][:8]),1e-300)) for k,v in actual.items()}
            if max(errors.values())>2e-11:raise ArithmeticError('fullH/JVP/guide replay changed')
            result.update(status='passed',physical_product_relative_errors=errors,native_parameter_replay=parameter_replay,
                exact_sampler_rows_replayed=8,maximum_Born_replay_error=draw['maximum_Born_replay_error'],
                parent_optimizer_and_rng_snapshot_unchanged=True,runtime=legacy.numerical_runtime_fingerprint(),
                candidate_sha256=record['candidate_sha256'],registered_result_sha256=file_sha256(output/'results.json'))
    except Exception as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        result['wall_seconds']=perf_counter()-start;atomic_json(output/'process_replay.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--validate-only',action='store_true');parser.add_argument('--replay',action='store_true')
    parser.add_argument('--resume-from',type=Path);parser.add_argument('--resume-manifest',type=Path)
    parser.add_argument('--wall-cap-seconds',type=float,default=10800.)
    args=parser.parse_args()
    if args.resume_from:
        if args.plan or args.replay or args.validate_only or not args.output or not args.resume_manifest:
            parser.error('resume requires --output and --resume-manifest; original registered plan is reused, no --plan/replay/validate-only')
        print(resume(args.resume_from,args.output,json.loads(args.resume_manifest.read_text()),wall_cap_seconds=args.wall_cap_seconds)['status'],flush=True)
    elif args.plan is None:parser.error('--plan required outside the bounded resume entrypoint')
    elif args.validate_only:
        validate_plan(json.loads(args.plan.read_text()));print('valid_unlaunched_plan')
    elif args.output is None:parser.error('--output required for an authorized execution')
    elif args.replay:print(replay(args.output)['status'],flush=True)
    else:print(run(json.loads(args.plan.read_text()),args.output)['status'],flush=True)
