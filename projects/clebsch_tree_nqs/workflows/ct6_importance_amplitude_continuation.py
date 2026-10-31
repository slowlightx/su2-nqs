"""Bounded native-W CT6 optimization using globally supported IID amplitude SR.

Start at an immutable audited parent, never an unproved diagnostic candidate.
The original J.125 profile permits at most16 updates in14h, with a separate
two-update interim. The J0 joint-ternary profile permits exactly two updates
in3.5h, without an interim or automatic continuation. Both use separate
prepare/replay/train processes and a fresh fixed-endpoint terminal audit.
No MCMC fit, architecture change, source-run edit or job submission is owned.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
from time import perf_counter,process_time,time

import jax
jax.config.update('jax_enable_x64',True)
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint
from su2_nqs.vmc.finite_candidate_calibration import select_finite_grid
from su2_nqs.vmc.full_support_proposal import draw_full_support_sz0,full_support_sz0_density
from su2_nqs.vmc.importance_amplitude_sr import make_importance_amplitude_sr
from . import ct6_native_direction_calibration as diagnostic
from .triangle_iid_orbit_audit import paired_iid_moments

ROOT=Path(__file__).resolve().parents[3]
TASK='CT6_full_support_IID_amplitude_SR16_v1'
J0_TASK='CT6_J0_full_support_IID_amplitude_SR2_v1'
FACTORS=np.array([2.,1.,.5,.25,0.]);RELATIVE_GRID=FACTORS/2.
EPSILON=1/8


def protocol_profile(plan:dict) -> dict:
    """Return one closed, preregistered model/budget profile, not free knobs."""
    if plan.get('task')==TASK:
        return dict(task=TASK,arity='binary_truncated',j2_over_j1=.125,
            parent_global_step=476,updates=16,interim_after_updates=2,interim_records=4096,
            terminal_audit_records=16384,wall_cap_seconds=50400,audit_reserve_seconds=9000,
            interim_reserve_seconds=3600)
    if plan.get('task')==J0_TASK:
        return dict(task=J0_TASK,arity='ternary_joint',j2_over_j1=0.,
            parent_global_step=456,updates=2,interim_after_updates=0,interim_records=0,
            terminal_audit_records=8192,wall_cap_seconds=12600,audit_reserve_seconds=3600,
            interim_reserve_seconds=0)
    raise ValueError('unknown immutable CT6 importance-training profile')


@contextmanager
def budget_timeout(seconds:float):
    """Nested foreground deadlines restore the OUTER remaining timer exactly."""
    if not np.isfinite(seconds) or seconds<=0:raise TimeoutError('no remaining science budget')
    previous=signal.getsignal(signal.SIGALRM);timer=signal.getitimer(signal.ITIMER_REAL);started=perf_counter()
    def stop(*_):raise TimeoutError('cumulative or reserved-audit wall budget reached')
    signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_REAL,min(seconds,timer[0]) if timer[0]>0 else seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        if timer[0]>0:signal.setitimer(signal.ITIMER_REAL,max(.000001,timer[0]-(perf_counter()-started)),timer[1])


def validate_plan(plan:dict,*,executing:bool=False) -> None:
    """One immutable parent, fixed budgets/roles, and explicit execution authority."""
    profile=protocol_profile(plan)
    required=dict(model=dict(length=6,multiplicity_cap=28,maximum_q=10),
        binary_rank=8,fit_records=4096,calibration_records=4096,calibration_split=2048,
        epsilon=EPSILON,damping=.5,learning_rate=.03,target_FS=.001,maximum_FS=.005,
        native_factors=FACTORS.tolist(),maximum_peak_device_bytes=70*1024**3,automatic_next_block=False,
        campaign_deadline_utc='2026-10-28T20:44:00+00:00')
    required.update({k:v for k,v in profile.items() if k!='parent_global_step'})
    if any(plan.get(k)!=v for k,v in required.items()):raise ValueError('fixed4096IID/w28/q10 registered profile required')
    seeds=plan.get('seeds',{})
    if set(seeds)!={'training','interim','interim_statistics','audit','audit_statistics'}:
        raise ValueError('five separate registered training/interim/audit streams required')
    values=list(seeds.values())
    if (any(isinstance(v,bool) or not isinstance(v,int) or not 0<=v<2**32 for v in values)
            or len(set(values))!=len(values) or set(values)&set(plan.get('parent_registered_seeds',[]))):
        raise ValueError('fresh distinct uint32 seeds required')
    parent=plan.get('parent',{})
    if (parent.get('global_step')!=profile['parent_global_step'] or set(parent.get('sealed_files',{}))!=
            {'checkpoint.pkl','registration.json','results.json','terminal_audit.json'}
            or any(not isinstance(v,str) or len(v)!=64 for v in parent['sealed_files'].values())
            or len(parent.get('params_sha256',''))!=64 or len(parent.get('terminal_products_sha256',''))!=64):
        raise ValueError('matching independently audited parent seals required')
    for key in ('projection_chunk','action_chunk','score_chunk','H_chunk'):
        if isinstance(plan.get(key),bool) or not isinstance(plan.get(key),int) or plan[key]<1:
            raise ValueError('positive integer kernel chunks required')
    if executing and (plan.get('execution_authorized') is not True or not plan.get('authorization_note')):
        raise PermissionError('separate explicit root authority required')


def kernel_plan(plan:dict) -> dict:
    """Only construct unchanged native/full432-image/full-H runtime kernels."""
    return dict(plan,fit=dict(samples=4096,chains=256,refresh_proposals=1476,spacing_proposals=36),
        confirmation_chunk=plan['H_chunk'])


def checked_parent(plan:dict) -> tuple[dict,dict]:
    """Verify the exact endpoint/optimizer/audit identity for either profile."""
    if plan['task']==TASK:
        return diagnostic.checked_parent(kernel_plan(plan))
    if not jax.config.x64_enabled:raise RuntimeError('x64 must precede trusted checkpoint load')
    if not all(input_seals(plan).values()):raise ValueError('parent input seals do not match')
    directory=Path(plan['parent']['directory']);expected=plan['parent']['global_step']
    reg,seal=diagnostic.legacy._sealed_json(directory/'registration.json')
    result=json.loads((directory/'results.json').read_text())
    audit=json.loads((directory/'terminal_audit.json').read_text())
    state,meta=diagnostic.legacy.shared.restore_state(directory/'checkpoint.pkl',seal)
    if (result['status']!='fixed_steps_complete' or result['absolute_completed_steps']!=expected
            or result['checkpoint_sha256']!=plan['parent']['sealed_files']['checkpoint.pkl']
            or result['params_sha256']!=plan['parent']['params_sha256']
            or audit['params_sha256']!=result['params_sha256'] or result['terminal_audit']!=audit
            or not audit['independent_from_training_and_calibration']
            or pytree_sha256(state[0])!=result['params_sha256'] or meta['absolute_completed_steps']!=expected
            or reg['model']!=plan['model'] or reg['arity']!=plan['arity']
            or reg['j2_over_j1']!=plan['j2_over_j1'] or reg['binary_rank']!=plan['binary_rank']
            or seal!=plan['parent'].get('registration_self_sha256')):
        raise ValueError('parent endpoint/audit/native state identity mismatch')
    return load_checkpoint(directory/'checkpoint.pkl'),dict(registration_sha256=seal,
        parent_optimizer_sha256=pytree_sha256(state[1]),parent_sampler_rng_sha256=pytree_sha256(state[2:]),
        native_parameter_count=sum(np.size(x) for x in jax.tree.leaves(state[0])))


def runtime_plan(plan:dict) -> dict:
    """Pass explicit arity/H to unchanged physical kernels; no global mutation."""
    return dict(diagnostic.runtime_plan(kernel_plan(plan)),arity=plan['arity'],
        binary_rank=plan['binary_rank'],j2_over_j1=plan['j2_over_j1'])


def make_runtime(plan:dict,parent:dict) -> dict:
    """No chain sampler is invoked; the old runtime contributes physical kernels."""
    rt=diagnostic.legacy.runtime(runtime_plan(plan),parent)
    rt['native_update']=make_importance_amplitude_sr(
        lambda p,x:rt['projected'](rt['prepare'](p),x),parent,damping=.5,score_chunk=plan['score_chunk'],
        parameter_chunk_size=8192,sr_tangent_fs_radius=.003,prepare=rt['prepare'],prepared_amplitude=rt['projected'])
    orbit=diagnostic.make_orbit_amplitude_products(diagnostic.make_feature_lut_amplitude(rt['cfg']),
        rt['images'],projection_chunk=plan['projection_chunk'])
    rt['orbit_query']=jax.jit(lambda bank,rows:jax.lax.map(
        lambda row:jax.numpy.stack(orbit(bank,row[None]))[:,0],rows,batch_size=plan['action_chunk']))
    probabilities=np.abs([c for _,_,c in rt['bonds']]);probabilities/=probabilities.sum()
    rt['bridge'],rt['bridge_metadata']=diagnostic.make_orbit_exchange_bridge_products(
        lambda edges:diagnostic.make_feature_lut_exchange_queries(rt['cfg'],edges),rt['images'],rt['bonds'],
        probabilities,n_sites=36,projection_chunk=plan['projection_chunk'])
    return rt


def initial_rng(plan:dict) -> dict:
    """Store the actual master PCG64 state/cursor, not only its initial seed."""
    rng=np.random.default_rng(plan['seeds']['training'])
    return dict(generator=rng.bit_generator.state,next_step=1,
        consumed_role_seeds=[],excluded_seeds=sorted(set(plan['seeds'].values())|set(plan['parent_registered_seeds'])))


def role_seeds(state:dict,index:int) -> tuple[dict,dict]:
    """Freeze three distinct next-step streams before drawing any physical row."""
    if state['next_step']!=index:raise ValueError('training RNG cursor does not match transition')
    rng=np.random.default_rng();rng.bit_generator.state=state['generator']
    used=set(state['excluded_seeds'])|set(state['consumed_role_seeds']);values=[]
    while len(values)<3:
        seed=int(rng.integers(0,2**32,dtype=np.uint32))
        if seed not in used:values.append(seed);used.add(seed)
    following=dict(state,generator=rng.bit_generator.state,next_step=index+1,
        consumed_role_seeds=state['consumed_role_seeds']+values)
    return dict(zip(('fit','selection','guard'),values)),following


def save_training_state(path:Path,state:tuple,metadata:dict) -> None:
    """Persist original W, full optimizer and exact unused next IID RNG state."""
    params,moments,rng=state
    save_checkpoint(path,params,optimizer_state=moments,metadata=dict(metadata,training_rng=rng,
        params_sha256=pytree_sha256(params),optimizer_sha256=pytree_sha256(moments),rng_sha256=canonical_sha256(rng)))


def load_training_state(path:Path) -> tuple[tuple,dict]:
    """Fail closed on parameter/moment/RNG precision or cursor changes."""
    if not jax.config.x64_enabled:raise RuntimeError('x64 must precede checkpoint load')
    saved=load_checkpoint(path);meta=saved['metadata'];state=(saved['params'],saved['optimizer_state'],meta['training_rng'])
    if (pytree_sha256(state[0])!=meta['params_sha256'] or pytree_sha256(state[1])!=meta['optimizer_sha256']
            or canonical_sha256(state[2])!=meta['rng_sha256']):raise ValueError('training state seal mismatch')
    return state,meta


def save_draw(path:Path,rows:np.ndarray,report:dict) -> None:
    """Keep all actual draw/substream records, without large JSON array lists."""
    arrays={'rows':np.asarray(rows)}
    def encode(value,prefix='draw'):
        if isinstance(value,np.ndarray):arrays[prefix]=value;return dict(npz_array=prefix)
        if isinstance(value,dict):return {k:encode(v,prefix+'__'+k) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [encode(v,prefix+'__'+str(i)) for i,v in enumerate(value)]
        if isinstance(value,np.generic):return value.item()
        return value
    receipt=encode(report);np.savez(path.with_suffix('.npz'),**arrays);atomic_json(path.with_suffix('.json'),receipt)


def draw_rows(banks:list,rt:dict,count:int,seed:int) -> tuple[np.ndarray,dict]:
    """Exact raw-orbit/H bridge plus an explicit uniform Sz=0 component."""
    def base(number,base_seed):return diagnostic.draw_orbit_rows(banks,rt,number,base_seed,bridge=True)
    return draw_full_support_sz0(base,count,n_sites=36,seed=seed,epsilon=EPSILON)


def _product_batches(plan:dict,rt:dict,banks:list,rows:np.ndarray,report:dict,label:str,remaining=None) -> np.ndarray:
    prepared=[diagnostic.prepare_feature_lut_bank(b,rt['cfg']) for b in banks]
    pieces=[];times=[];chunk=plan['H_chunk'];report.setdefault('scalar_H_centers',0)
    def preserve():
        report['H_batch_seconds']=times;report['complete_product_records']=sum(len(x) for x in pieces)
        report['evaluated_H_centers_without_complete_endpoint_tuple']=report['scalar_H_centers']-len(banks)*report['complete_product_records']
        prefix=report.get('partial_artifact_prefix')
        if prefix is not None:
            if pieces:np.savez(Path(prefix).with_suffix('.npz'),rows=rows[:report['complete_product_records']],products=np.concatenate(pieces))
            atomic_json(Path(prefix).with_suffix('.json'),report)
    try:
        for offset in range(0,len(rows),chunk):
            tick=perf_counter();pair=[];bits=rows[offset:offset+chunk]
            for b in prepared:
                report['scalar_H_centers_requested']=report['scalar_H_centers']+len(bits)
                pair.append(np.stack([np.asarray(v) for v in jax.block_until_ready(rt['bridge'](b,bits))],axis=1))
                report['scalar_H_centers']+=len(bits)
            pieces.append(np.stack(pair,axis=1));times.append(perf_counter()-tick)
            if diagnostic._device_peak()>plan['maximum_peak_device_bytes']:raise MemoryError('registered70GiB ceiling exceeded')
            if remaining is not None and len(times)==2:
                required=times[-1]*(len(rows)-offset-len(bits))/len(bits)+120
                report['cost_gate']=dict(cold_batch_seconds=times[0],warm_batch_seconds=times[1],
                    batch_records=len(bits),forecast_remaining_seconds=required,available_seconds=remaining())
                if required>remaining():raise TimeoutError(label+' cannot finish within reserved wall time')
            if offset//chunk%64==0:
                preserve();print(label,offset+len(bits),'/',len(rows),flush=True)
        return np.concatenate(pieces)
    finally:preserve()


def restore_uncommitted_fit(cache:dict,report:dict,records:int,scale:float,output:Path) -> tuple:
    """Reuse sealed fit/H work after an SR allocation failure, never an audit.

    The exact parent, training RNG cursor and all three role seeds must match.
    Selection/guard data are NOT restored. Only a failed, pre-SR full panel is
    accepted, so an optimized or selected candidate cannot become fit data.
    """
    directory=Path(cache['directory']);seals=cache['sealed_files']
    names={'fit_products.npz','fit_draw.npz','fit_draw.json','streams_frozen.json','transition.json'}
    if set(seals)!=names or any(file_sha256(directory/n)!=seals[n] for n in names):
        raise ValueError('sealed uncommitted fit cache changed')
    old=json.loads((directory/'transition.json').read_text())
    frozen=json.loads((directory/'streams_frozen.json').read_text())
    for key in ('index','parent_params_sha256','seeds','rng_before_sha256','rng_after_sha256'):
        if old.get(key)!=report[key] or frozen.get(key)!=report[key]:
            raise ValueError('cached fit parent/training stream mismatch: '+key)
    if (old.get('accepted') is not False or old.get('SR') is not None
            or old['work'].get('fit_draws')!=records or old['work'].get('fit_H_centers')!=records
            or old['work'].get('full_native_jacobian_rows')!=0
            or (directory/'raw_SR_direction.pkl').exists()):
        raise ValueError('only complete fit products before an uncommitted SR failure may be recovered')
    with np.load(directory/'fit_products.npz',allow_pickle=False) as f:
        rows=f['rows'].copy();phi=f['psi'].copy();hp=f['hpsi'].copy()
        density=f['proposal'].copy();products=f['canonical_products'].copy()
        saved_scale=float(f['exact_native_to_canonical_scale'])
    with np.load(directory/'fit_draw.npz',allow_pickle=False) as f:
        same_rows=np.array_equal(rows,f['rows'])
    if (rows.shape!=(records,36) or not same_rows or not np.isin(rows,(0,1)).all()
            or not np.all(rows.sum(1)==18) or products.shape!=(records,4)
            or any(x.shape!=(records,) for x in (phi,hp,density))
            or not np.isfinite(np.r_[phi,hp,density,products.ravel(),saved_scale]).all()
            or not np.isclose(saved_scale,scale,rtol=2e-13,atol=0)):
        raise ValueError('cached fit support/shape/normalization mismatch')
    expected=full_support_sz0_density(products[:,2].real,products[:,3].real,n_sites=36,epsilon=EPSILON)
    if (np.any(density<=0) or not np.allclose(density,expected,rtol=2e-13,atol=0)
            or not np.allclose(phi,scale*products[:,0],rtol=2e-13,atol=0)
            or not np.allclose(hp,scale*products[:,1],rtol=2e-13,atol=0)
            or np.iscomplexobj(phi) or np.iscomplexobj(hp)):
        raise ValueError('cached native amplitudes/H/proposal do not agree')
    # Preserve the actual original draw receipt, not a newly simulated draw.
    for name in ('fit_draw.npz','fit_draw.json'):shutil.copy2(directory/name,output/name)
    report['fit_cache_recovery']=dict(directory=str(directory),sealed_files=seals,
        reused_fit_records=records,reused_full_H_centers=records,not_independent_audit=True)
    report['work']['reused_fit_records']=records
    report['work']['reused_fit_H_centers']=records
    return rows,phi,hp,density,products


def finite_step(plan:dict,rt:dict,state:tuple,index:int,output:Path,*,replay_fit_from:dict|None=None) -> tuple[tuple,dict]:
    """One complete IID native direction and fresh finite-state geometry.

    Historical closed plans still require4096 fit/geometry records. An
    independently validated continuation controller may explicitly request a
    larger complete fit panel; no cost gate silently reduces its size.
    """
    if not 1<=index<=plan['updates'] or output.exists():raise ValueError('fresh registered transition required')
    output.mkdir(parents=True);parent,moments,rng=state;seeds,next_rng=role_seeds(rng,index);start=perf_counter()
    report=dict(index=index,accepted=False,parent_params_sha256=pytree_sha256(parent),seeds=seeds,
        rng_before_sha256=canonical_sha256(rng),rng_after_sha256=canonical_sha256(next_rng),
        timings={},work=dict(fit_draws=0,fit_H_centers=0,fit_native_H_replay_centers=0,
            full_native_jacobian_rows=0,calibration_draws=0,calibration_bank_centers=0,native_amplitude_replay_centers=0))
    atomic_json(output/'streams_frozen.json',report)
    def timed(name,call):
        tick=perf_counter();value=call();jax.block_until_ready(value);report['timings'][name]=perf_counter()-tick
        if diagnostic._device_peak()>plan['maximum_peak_device_bytes']:raise MemoryError('registered70GiB ceiling exceeded')
        atomic_json(output/'progress.json',report);return value
    try:
        bank,note=timed('parent_sampler_canonicalization',lambda:diagnostic.normalized_raw_bank(parent,rt))
        records=plan.get('fit_records',4096)
        split=plan.get('calibration_split',2048)
        scale=np.sqrt(note['original_raw_norm'])
        if replay_fit_from is not None:
            rows,native_phi,native_hp,density,products=restore_uncommitted_fit(
                replay_fit_from,report,records,scale,output)
        else:
            rows,draw=timed('fit_full_support_IID',lambda:draw_rows([bank],rt,records,seeds['fit']))
            save_draw(output/'fit_draw',rows,draw);report['work']['fit_draws']=records
            product_report=dict(partial_artifact_prefix=str(output/'fit_partial'))
            report['fit_products_work']=product_report
            products=timed('fit_full_H',lambda:_product_batches(plan,rt,[bank],rows,product_report,'IID fitH'))[:,0]
            report['work']['fit_H_centers']=product_report['scalar_H_centers']
            density=full_support_sz0_density(products[:,2].real,products[:,3].real,n_sites=36,epsilon=EPSILON)
            native_phi=scale*products[:,0];native_hp=scale*products[:,1]
        if any(np.max(abs(v.imag))>1e-13*max(np.max(abs(v)),1e-300) for v in (native_phi,native_hp)):
            raise ArithmeticError('real native family acquired complex physical products')
        native_phi=native_phi.real;native_hp=native_hp.real
        np.savez(output/'fit_products.npz',rows=rows,psi=native_phi,hpsi=native_hp,proposal=density,
            canonical_products=products,exact_native_to_canonical_scale=scale)
        replay=timed('native_fit_H_scale_replay',lambda:rt['action'](rt['prepare'](parent),rows[:2]))
        expected=np.stack((native_phi[:2],native_hp[:2]));actual=np.stack([np.asarray(v) for v in replay])
        errors=np.max(abs(actual-expected),axis=1)/np.maximum(np.max(abs(expected),axis=1),1e-300)
        report['native_fit_H_scale_replay_relative_errors']=errors.tolist();report['work']['fit_native_H_replay_centers']=2
        if not np.isfinite(errors).all() or max(errors)>2e-11:raise ArithmeticError('native psi/J/Hpsi scale contract replay failed')
        proposed,next_moments,sr=timed('full_native_importance_amplitude_SR',lambda:rt['native_update'](
            parent,moments,rows,native_hp,density,.03))
        report['work']['full_native_jacobian_rows']=records
        report['SR']=jax.tree.map(lambda x:np.asarray(x).item(),sr)
        delta=jax.tree.map(lambda a,b:a-b,proposed,parent)
        save_checkpoint(output/'raw_SR_direction.pkl',delta,optimizer_state=next_moments,
            metadata=dict(parent_params_sha256=report['parent_params_sha256'],next_training_rng=next_rng,
                importance_proposal_frozen=True,all_native_columns=True,proposed_optimizer_uncommitted=True))
        if not bool(sr['accepted']):report['reason']='numerically_invalid_amplitude_SR';return state,report
        candidates=[jax.tree.map(lambda p,d:p+factor*d,parent,delta) for factor in FACTORS]
        banks=[];notes=[]
        for i,p in enumerate(candidates):
            b,n=diagnostic.normalized_raw_bank(p,rt);banks.append(b);notes.append(n)
            save_checkpoint(output/f'finite_native_{i}.pkl',p,metadata=dict(native_factor=float(FACTORS[i])))
            save_checkpoint(output/f'finite_canonical_{i}.pkl',b,metadata=n)
        atomic_json(output/'finite_grid_frozen.json',dict(native_factors=FACTORS.tolist(),
            native_sha256=[file_sha256(output/f'finite_native_{i}.pkl') for i in range(5)],
            raw_direction_sha256=file_sha256(output/'raw_SR_direction.pkl')))
        panels=[]
        for role in ('selection','guard'):
            # Existing exact five-state orbit proposal already covers these
            # finite endpoint norms; the fit metric separately has full support.
            bits,draw=timed(role+'_IID',lambda role=role:diagnostic.draw_orbit_rows(banks,rt,split,seeds[role]))
            save_draw(output/(role+'_draw'),bits,draw);panels.append(bits);report['work']['calibration_draws']+=split
        bits=np.concatenate(panels);columns=[]
        for i,b in enumerate(banks):
            prepared=diagnostic.prepare_feature_lut_bank(b,rt['cfg'])
            columns.append(np.asarray(timed('orbit_bank_'+str(i),lambda p=prepared:rt['orbit_query'](p,bits))))
            report['work']['calibration_bank_centers']+=2*split
        products=np.stack(columns,axis=1);phi=products[:,:,0];q=products[:,:,1].real.mean(1)
        np.savez(output/'calibration_products.npz',rows=bits,phi=phi,guide_densities=products[:,:,1],proposal=q)
        errors=[]
        for i,p in enumerate(candidates):
            value=np.asarray(timed('native_replay_'+str(i),lambda p=p:diagnostic.native_projected_amplitude(rt,p,bits[:2])))
            value=value/np.sqrt(notes[i]['original_raw_norm'])
            errors.append(float(np.max(abs(value-phi[:2,i]))/max(np.max(abs(phi[:2,i])),1e-300)))
            report['work']['native_amplitude_replay_centers']+=2
        if not np.isfinite(errors).all() or max(errors)>2e-11:raise ArithmeticError('finite native/canonical replay failed')
        selected,calibration=select_finite_grid(phi,q,RELATIVE_GRID,split=split)
        report.update(selected_index=selected,calibration=calibration,native_amplitude_replay_errors=errors,
            actual_native_factor=float(FACTORS[selected]),no_post_IID_training_metric_shrink=True)
        if selected==4:report['reason']='independent_finite_geometry_rejected_no_retry';return state,report
        if pytree_sha256(candidates[selected])==report['parent_params_sha256']:
            report['reason']='zero_native_step_not_counted_as_update';return state,report
        candidate=(candidates[selected],next_moments,next_rng)
        save_training_state(output/'candidate_state.pkl',candidate,dict(completed_updates=index,
            absolute_completed_steps=plan['parent']['global_step']+index,provisional_until_transaction_commit=True))
        report.update(accepted=True,reason='importance_direction_and_finite_geometry_passed',
            candidate_params_sha256=pytree_sha256(candidate[0]),candidate_sha256=file_sha256(output/'candidate_state.pkl'))
        return candidate,report
    finally:
        report['work']['fit_H_centers']=report.get('fit_products_work',{}).get('scalar_H_centers',report['work']['fit_H_centers'])
        report['peak_device_bytes']=diagnostic._device_peak()
        report['wall_seconds']=perf_counter()-start;atomic_json(output/'transition.json',report)


def paired_audit(plan:dict,rt:dict,parent:dict,state:tuple,output:Path,count:int,seed:int,statistics:int,remaining) -> dict:
    """Fresh fixed-pair complete-H moments with a fully supported known proposal."""
    allowed={plan['terminal_audit_records']}
    if plan['interim_records']>0:allowed.add(plan['interim_records'])
    if count not in allowed:raise ValueError('registered interim/final record count required')
    output.mkdir();start=perf_counter();report=dict(status='started',records=0,scalar_H_centers=0,
        seed=seed,statistics_seed=statistics,
        comparison_parent_global_step=plan.get('comparison_parent_global_step',plan['parent']['global_step']),
        endpoint_params_sha256=pytree_sha256(state[0]),full_support_epsilon=EPSILON,
        no_endpoint_selection=True,partial_artifact_prefix=str(output/'partial_products'))
    try:
        if remaining()<=120:raise TimeoutError('insufficient time for fixed audit serialization')
        banks=[];notes=[]
        for native in (parent,state[0]):
            b,n=diagnostic.normalized_raw_bank(native,rt);banks.append(b);notes.append(n)
        rows,draw=draw_rows(banks,rt,count,seed);save_draw(output/'draw',rows,draw)
        report['draw_records']=count;report['normalization']=notes
        products=_product_batches(plan,rt,banks,rows,report,'paired auditH',remaining)
        q=full_support_sz0_density(products[:,:,2].real.mean(1),products[:,:,3].real.mean(1),n_sites=36,epsilon=EPSILON)
        np.savez(output/'products.npz',rows=rows,psi=products[:,:,0],hpsi=products[:,:,1],
            guide_components=products[:,:,2:],proposal=q)
        report['moments']=paired_iid_moments(products[:,:,0],products[:,:,1],q,
            np.repeat(np.arange(128),count//128),n_sites=36,seed=statistics)
        report.update(status='complete',records=count)
        return report
    finally:
        report['wall_seconds']=perf_counter()-start;atomic_json(output/'audit.json',report)


def interim_decision(report:dict) -> dict:
    """One nonadaptive safety decision; an unresolved decrease is not a win."""
    if report.get('status')!='complete' or report.get('records')!=4096:
        raise ValueError('completed fixed4096-record interim required')
    m=report['moments'];de,dv=(m['candidate_minus_control'][i] for i in (0,2))
    se,sv=(m['paired_iid_delta_SE'][i] for i in (0,2))
    if not np.isfinite([de,dv,se,sv]).all() or min(se,sv)<0:raise ArithmeticError('nonfinite interim moments')
    return dict(continue_training=bool(de<=2*se and dv<=2*sv),energy_difference_per_site=de,
        energy_difference_SE=se,variance_difference_per_site=dv,variance_difference_SE=sv,
        resolved_energy_improvement=bool(de+2*se<0),
        rule='no resolved2SE energy ascent and no resolved2SE variance harm; unresolved improvement is not a win',
        interim_consumed_as_release_data_not_terminal_audit=True)


def compare_transition(expected:tuple,actual:tuple,first:dict,second:dict,left:Path,right:Path) -> dict:
    replay=diagnostic.legacy.compare_replay(expected[:2],actual[:2],tolerance=2e-11)
    if not replay['passed'] or canonical_sha256(expected[2])!=canonical_sha256(actual[2]):
        raise ArithmeticError('full native params/moments/RNG replay failed')
    for key in ('accepted','selected_index','actual_native_factor','seeds','rng_after_sha256'):
        if first.get(key)!=second.get(key):raise ArithmeticError('transition choice/stream replay differs: '+key)
    for name in ('fit_products.npz','selection_draw.npz','guard_draw.npz'):
        with np.load(left/name) as a,np.load(right/name) as b:
            if not np.array_equal(a['rows'],b['rows']):raise ArithmeticError('IID replay rows differ')
    numerical=lambda x:jax.tree.map(lambda a:np.frombuffer(a.encode(),np.uint8) if isinstance(a,str) else a,x)
    calibration=diagnostic.legacy.compare_replay(numerical(first['calibration']),numerical(second['calibration']),tolerance=2e-11)
    if not calibration['passed']:raise ArithmeticError('finite geometry replay differs')
    return dict(status='passed',parameters_optimizer_rng_all_compared=True,native=replay,calibration=calibration,
        physical_rows_identical=True,updates_committed_by_replay=0)


def remaining_seconds(plan:dict,started:float,now:float,end_time:str|None) -> float:
    deadline=datetime.fromisoformat(plan['campaign_deadline_utc']).timestamp()
    cap=min(plan['wall_cap_seconds']-(now-started),deadline-now-120.)
    if not np.isfinite([started,now,cap]).all() or now<started or cap<=0:
        raise TimeoutError('cumulative allocation/campaign deadline exhausted')
    return diagnostic.allocation_science_seconds(cap,end_time,now)


def input_seals(plan:dict) -> dict:
    p=plan['parent'];names=dict(p['sealed_files'],**{'terminal_audit.npz':p['terminal_products_sha256']})
    return {k:file_sha256(Path(p['directory'])/k)==v for k,v in names.items()}


def registration(plan:dict,output:Path,info:dict,*,fresh:bool) -> dict:
    if fresh:
        if output.exists() and any(output.iterdir()):raise FileExistsError('isolated fresh campaign output required')
        output.mkdir(parents=True,exist_ok=True)
        sources=transitive_local_source_manifest([Path(__file__)],root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
        reg=dict(plan=plan,plan_sha256=canonical_sha256(plan),sources=sources,parent=info,started_epoch=time(),
            runtime=diagnostic.legacy.numerical_runtime_fingerprint(),initial_global_step=plan['parent']['global_step'],
            declared_maximum_endpoint_global_step=plan['parent']['global_step']+plan['updates'],
            diagnostic477_not_imported=True)
        for name in (*plan['parent']['sealed_files'],'terminal_audit.npz'):
            dest=output/'parent_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(Path(plan['parent']['directory'])/name,dest)
        for name in sources:
            dest=output/'registered_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        atomic_json(output/'registration.json',reg)
    else:
        reg=json.loads((output/'registration.json').read_text())
        if reg['plan']!=plan or reg['plan_sha256']!=canonical_sha256(plan):raise ValueError('plan changed across processes')
        for name,digest in reg['sources'].items():
            if file_sha256(ROOT/name)!=digest or file_sha256(output/'registered_sources'/name)!=digest:
                raise ValueError('immutable numerical source changed: '+name)
    if not all(input_seals(plan).values()):raise ValueError('audited parent input changed')
    return reg


def commit(output:Path,state:tuple,completed:int,history:list,*,parent_global_step:int=476) -> None:
    save_training_state(output/'checkpoint.pkl',state,dict(completed_updates=completed,
        absolute_completed_steps=parent_global_step+completed,history=history,original_parent_global_step=parent_global_step,
        diagnostic477_not_imported=True,full_support_IID_amplitude_SR=True))


def run(plan:dict,phase:str,output:Path) -> dict:
    """Cumulative prepare/replay/train; no silent retries after a failed phase."""
    validate_plan(plan,executing=True)
    if phase not in ('prepare','replay','train'):raise ValueError('unknown phase')
    if jax.default_backend()!='gpu' or not any('A100' in d.device_kind for d in jax.devices()):
        raise RuntimeError('approved A100 runtime required; local toy mocks only')
    phase_path=output/('phase_'+phase+'.json')
    saved,info=checked_parent(plan)
    reg=registration(plan,output,info,fresh=phase=='prepare' and not phase_path.exists())
    if phase_path.exists():return json.loads(phase_path.read_text())
    state=(saved['params'],saved['optimizer_state'],initial_rng(plan))
    if phase!='prepare':
        ready=json.loads((output/'phase_prepare.json').read_text())
        if ready['status']!='prepared_first_transition_uncommitted':raise ValueError('prepare did not release replay/train')
        state,_=load_training_state(output/'initial_state.pkl')
        if phase=='train' and json.loads((output/'phase_replay.json').read_text())['status']!='replay_passed':
            raise ValueError('full first transition requires separate-process replay')
    start=perf_counter();cpu=process_time();report=dict(status='started',phase=phase,completed_updates=0,
        work={},runtime=dict(diagnostic.legacy.numerical_runtime_fingerprint(),process_id=os.getpid()))
    remaining=lambda:remaining_seconds(plan,reg['started_epoch'],time(),os.environ.get('SLURM_JOB_END_TIME'))
    try:
        with budget_timeout(remaining()):
            rt=make_runtime(plan,saved['params'])
            report['parent_numerical_replay']=diagnostic.audited_parent_replay(rt,saved['params'],output/'parent_snapshot'/'terminal_audit.npz')
            report['work']['parent_replay_H_centers']=2
            if phase=='prepare':
                parent_step=plan['parent']['global_step']
                save_training_state(output/'initial_state.pkl',state,dict(completed_updates=0,absolute_completed_steps=parent_step))
                commit(output,state,0,[],parent_global_step=parent_step)
                proposed,step=finite_step(plan,rt,state,1,output/'transition_01');report['transition']=step
                if not step['accepted']:
                    report['status']=f'first_transition_rejected_original{parent_step}_preserved';return report
                # Reserve replay, one further actual update, interim and terminal.
                required=2*step['wall_seconds']+plan['interim_reserve_seconds']+plan['audit_reserve_seconds']+120
                report['cost_gate']=dict(observed_first_seconds=step['wall_seconds'],required_to_two_updates_and_audits=required,
                    available_seconds=remaining(),maximum_not_guaranteed_updates=plan['updates'])
                report['status']=('prepared_first_transition_uncommitted' if required<=remaining()
                    else f'cost_gate_original{parent_step}_preserved')
            elif phase=='replay':
                if ready['runtime']['process_id']==report['runtime']['process_id']:raise ValueError('fresh replay process required')
                expected,_=load_training_state(output/'transition_01'/'candidate_state.pkl')
                actual,step=finite_step(plan,rt,state,1,output/'transition_01_replay')
                first=json.loads((output/'transition_01'/'transition.json').read_text())
                report['replay']=compare_transition(expected,actual,first,step,output/'transition_01',output/'transition_01_replay')
                report.update(status='replay_passed',transition=step)
            else:
                state,_=load_training_state(output/'transition_01'/'candidate_state.pkl')
                parent_step=plan['parent']['global_step']
                history=[dict(index=1,global_step=parent_step+1)]
                commit(output,state,1,history,parent_global_step=parent_step);report['completed_updates']=1
                elapsed=[json.loads((output/'transition_01'/'transition.json').read_text())['wall_seconds']]
                stop=None;interim=None
                for index in range(2,plan['updates']+1):
                    reserve=plan['audit_reserve_seconds']+(plan['interim_reserve_seconds'] if interim is None else 0)
                    if 1.15*max(elapsed[-3:])+reserve+120>remaining():stop='time_reserved_for_independent_audits';break
                    try:
                        with budget_timeout(remaining()-reserve):
                            proposed,step=finite_step(plan,rt,state,index,output/f'transition_{index:02d}')
                    except TimeoutError as error:
                        report['interrupted_uncommitted_transition']=dict(index=index,error=repr(error))
                        stop='step_timeout_final_audit_reserve_protected';break
                    elapsed.append(step['wall_seconds'])
                    if not step['accepted']:stop='numerical_or_finite_geometry_rejection_no_retry';break
                    state=proposed;history.append(dict(index=index,global_step=parent_step+index))
                    commit(output,state,index,history,parent_global_step=parent_step)
                    report['completed_updates']=index;atomic_json(output/'training_progress.json',report)
                    if plan['interim_after_updates']>0 and index==plan['interim_after_updates']:
                        interim_remaining=lambda:remaining()-plan['audit_reserve_seconds']
                        try:
                            with budget_timeout(interim_remaining()):
                                interim=paired_audit(plan,rt,saved['params'],state,output/'interim_audit',plan['interim_records'],
                                    plan['seeds']['interim'],plan['seeds']['interim_statistics'],interim_remaining)
                        except TimeoutError as error:
                            report['interim_incomplete_error']=repr(error)
                            stop='interim_incomplete_final_audit_reserve_protected';break
                        decision=interim_decision(interim);report.update(interim_audit=interim,interim_decision=decision)
                        atomic_json(output/'interim_decision.json',decision)
                        if not decision['continue_training']:stop='interim_resolved_harm_fixed_endpoint';break
                report.update(stop_reason=stop,absolute_completed_steps=parent_step+report['completed_updates'],
                    checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),params_sha256=pytree_sha256(state[0]))
                atomic_json(output/'endpoint_frozen.json',report)
                final=paired_audit(plan,rt,saved['params'],state,output/'terminal_audit',plan['terminal_audit_records'],
                    plan['seeds']['audit'],plan['seeds']['audit_statistics'],remaining)
                report['terminal_audit']=final
                report['status']=(f"fixed{plan['updates']}_updates_complete_independent_audit"
                    if report['completed_updates']==plan['updates'] else 'bounded_prefix_independently_audited')
    except (TimeoutError,MemoryError) as error:report.update(status='bounded_incomplete',error=repr(error))
    except Exception as error:report.update(status='failed_validation',error=repr(error));raise
    finally:
        report.update(wall_seconds=perf_counter()-start,CPU_seconds=process_time()-cpu,
            cumulative_block_wall_seconds=time()-reg['started_epoch'],peak_device_bytes=diagnostic._device_peak())
        report['input_seals_unchanged']=input_seals(plan)
        report['source_seals_unchanged']={name:file_sha256(ROOT/name)==seal for name,seal in reg['sources'].items()}
        intact=all(report['input_seals_unchanged'].values()) and all(report['source_seals_unchanged'].values())
        if not intact:report['status']='failed_immutable_source_or_input_seal'
        atomic_json(phase_path,report)
        if phase=='train':atomic_json(output/'results.json',report)
        if not intact:raise ArithmeticError('immutable parent/source seal changed')
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','replay','train'],required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());result=run(plan,args.phase,args.output)
    print(json.dumps(dict(status=result['status'],phase=args.phase,completed_updates=result.get('completed_updates'))))
    okay={'prepare':'prepared_first_transition_uncommitted','replay':'replay_passed',
        'train':f"fixed{plan['updates']}_updates_complete_independent_audit"}
    if result['status']!=okay[args.phase] and not(args.phase=='train' and result['status']=='bounded_prefix_independently_audited'):
        raise SystemExit(2)


if __name__=='__main__':main()
