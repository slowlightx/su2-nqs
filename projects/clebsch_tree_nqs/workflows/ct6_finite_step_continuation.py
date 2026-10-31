"""Fixed same-width CT6 continuation with supported finite-state geometry.

Import one independently confirmed update476→477, then15 fresh native updates
with a separate full first-new-transition replay and fresh paired endpoint
audit. The original476 state remains the audit comparator. This workflow does
not submit jobs or alter old runs; import requires explicit new authority.
"""
from __future__ import annotations

import argparse
from datetime import datetime
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
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint
from su2_nqs.vmc.finite_candidate_calibration import select_finite_grid
from su2_nqs.vmc.physical_sr_adam import make_physical_sr_adam
from . import ct6_native_direction_calibration as diagnostic
from .triangle_hierarchy_initialization import _node_timeout
from .triangle_iid_orbit_audit import paired_iid_moments

ROOT=Path(__file__).resolve().parents[3]
TASK='CT6_supported_finite_FS_same_width16_v1'
FACTORS=np.array([2.,1.5,1.,.5,0.])
RELATIVE_GRID=FACTORS/2.


def diagnostic_plan(plan:dict) -> dict:
    """Use unchanged physical kernels; do not invoke the old transition guard."""
    return dict(task=diagnostic.TASK,model=plan['model'],j2_over_j1=.125,arity='binary_truncated',binary_rank=8,
        parent=plan['parent'],parent_registered_seeds=[s for s in plan['parent_registered_seeds']
            if s!=plan['inherited_fit_seed']],fractions=list(diagnostic.GRID),
        fit=dict(samples=4096,chains=256,refresh_proposals=1476,spacing_proposals=36),
        calibration_records=4096,calibration_split=2048,confirmation_records=2048,damping=.5,
        target_FS=.001,maximum_FS=.005,wall_cap_seconds=14400,maximum_peak_device_bytes=70*1024**3,
        automatic_continuation=False,projection_chunk=plan['projection_chunk'],action_chunk=plan['action_chunk'],
        score_chunk=plan['score_chunk'],confirmation_chunk=plan['H_chunk'],
        seeds=dict(fit=plan['inherited_fit_seed'],selection=plan['step_seeds'][0]['selection'],
            guard=plan['step_seeds'][0]['guard'],confirmation=plan['seeds']['audit'],statistics=plan['seeds']['audit_statistics']),
        execution_authorized=plan['execution_authorized'],authorization_note=plan.get('authorization_note',''))


def validate_plan(plan:dict,*,executing:bool=False) -> None:
    """Fixed scope, finite-grid support, distinct data roles and hard cost bound."""
    if (plan.get('task')!=TASK or plan.get('updates')!=16 or plan.get('native_factors')!=FACTORS.tolist()
            or plan.get('model')!=dict(length=6,multiplicity_cap=28,maximum_q=10)
            or plan.get('calibration_records')!=4096 or plan.get('calibration_split')!=2048
            or plan.get('reused_diagnostic_confirmation_records')!=2048 or plan.get('terminal_audit_records')!=16384
            or plan.get('target_FS')!=.001 or plan.get('maximum_FS')!=.005
            or plan.get('wall_cap_seconds')!=50400 or plan.get('audit_reserve_seconds')!=9000
            or plan.get('automatic_next_block') is not False
            or plan.get('initial_verified_update_counts_toward_16') is not True
            or plan.get('training_rng_policy')!='inherit_verified_diagnostic_fit_key_and_states'
            or plan.get('campaign_deadline_utc')!='2026-10-28T20:44:00+00:00'
            or plan.get('parent',{}).get('global_step')!=476):
        raise ValueError('only the registered samew28/rank8/q10 16-update finite-FS protocol is allowed')
    seeds=plan.get('seeds',{})
    if set(seeds)!={'audit','audit_statistics'}:
        raise ValueError('fresh final audit and statistics streams required')
    rows=plan.get('step_seeds',[])
    if len(rows)!=15 or any(set(row)!={'selection','guard'} for row in rows):
        raise ValueError('fifteen fresh finite-selection and guard seed pairs required')
    values=list(seeds.values())+[v for row in rows for v in row.values()]
    if (any(isinstance(v,bool) or not isinstance(v,int) or not 0<=v<2**32 for v in values)
            or len(set(values))!=len(values) or set(values)&set(plan.get('parent_registered_seeds',[]))):
        raise ValueError('all fit/calibration/confirmation/audit seeds must be fresh and distinct')
    diagnostic.validate_plan(diagnostic_plan(plan),executing=executing)
    source=plan.get('source_diagnostic',{})
    if set(source.get('sealed_files',{}))!={'registration.json','results.json','confirmation_products.npz',
            'provisional_candidate.pkl','frozen_direction.pkl'}:
        raise ValueError('completed diagnostic products, native candidate and direction must be sealed')
    if any(not isinstance(v,str) or len(v)!=64 for v in source['sealed_files'].values()):
        raise ValueError('source diagnostic requires complete SHA256 seals')
    inherited=plan.get('inherited_fit_seed')
    if (isinstance(inherited,bool) or not isinstance(inherited,int) or inherited not in plan['parent_registered_seeds']
            or inherited in values):raise ValueError('explicit consumed diagnostic fit seed must be inherited, not redrawn')
    if executing and plan.get('release_confirmed_diagnostic_update') is not True:
        raise PermissionError('root must explicitly release the completed diagnostic update')


def make_runtime(plan:dict,parent:dict) -> dict:
    """Full native SR and complete coherent432-image/all216-bond physical H."""
    base=diagnostic_plan(plan);rt=diagnostic.legacy.runtime(diagnostic.runtime_plan(base),parent)
    rt['next_sampler']=diagnostic.legacy.shared.make_record_sampler(rt['projected'],base['fit'],rt['edges'])
    rt['native_update']=make_physical_sr_adam(lambda p,x:rt['projected'](rt['prepare'](p),x),parent,
        damping=.5,score_chunk=plan['score_chunk'],maximum_fs_sine=.005,minimum_ess_fraction=.98,
        prepare=rt['prepare'],prepared_amplitude=rt['projected'],sr_solver='dual_force',parameter_chunk_size=8192,
        sr_use_adam=False,sr_tangent_fs_radius=.003)
    orbit=diagnostic.make_orbit_amplitude_products(diagnostic.make_feature_lut_amplitude(rt['cfg']),
        rt['images'],projection_chunk=plan['projection_chunk'])
    rt['orbit_query']=jax.jit(lambda bank,rows:jax.lax.map(
        lambda row:jnp.stack(orbit(bank,row[None]))[:,0],rows,batch_size=plan['action_chunk']))
    probabilities=np.abs([v for _,_,v in rt['bonds']]);probabilities/=probabilities.sum()
    rt['bridge'],rt['bridge_metadata']=diagnostic.make_orbit_exchange_bridge_products(
        lambda edges:diagnostic.make_feature_lut_exchange_queries(rt['cfg'],edges),rt['images'],rt['bonds'],
        probabilities,n_sites=36,projection_chunk=plan['projection_chunk'])
    return rt


def save_training_state(path:Path,state:tuple,metadata:dict) -> None:
    """Save original native coordinates, all moment arrays and the live sampler."""
    params,moments,key,chains=state
    save_checkpoint(path,params,optimizer_state=moments,metadata=dict(metadata,
        training_key=np.asarray(key),training_states=np.asarray(chains),params_sha256=pytree_sha256(params),
        optimizer_sha256=pytree_sha256(moments),training_rng_sha256=pytree_sha256((key,chains))))


def load_training_state(path:Path) -> tuple[tuple,dict]:
    """Reject precision loss or changed optimizer/RNG before any new sample."""
    if not jax.config.x64_enabled:raise RuntimeError('x64 must be enabled BEFORE loading native state')
    saved=load_checkpoint(path);meta=saved['metadata']
    state=(saved['params'],saved['optimizer_state'],jnp.asarray(meta['training_key']),jnp.asarray(meta['training_states']))
    if (pytree_sha256(state[0])!=meta['params_sha256'] or pytree_sha256(state[1])!=meta['optimizer_sha256']
            or pytree_sha256(state[2:])!=meta['training_rng_sha256']):raise ValueError('native optimizer/RNG state seal mismatch')
    return state,meta


def checked_diagnostic_import(plan:dict,parent:dict) -> tuple[tuple,dict]:
    """Promote exactly ONE released provisional update, never the old snapshot.

    The diagnostic started a registered fresh fit chain. Its terminal fit key
    and configurations, rather than the old production chain or acceptance
    data, seed the continuation. Confirmation is release data, not final audit.
    """
    source=plan['source_diagnostic'];directory=Path(source['directory'])
    for name,digest in source['sealed_files'].items():
        if file_sha256(directory/name)!=digest:raise ValueError('diagnostic seal mismatch: '+name)
    reg=json.loads((directory/'registration.json').read_text())
    result=json.loads((directory/'results.json').read_text());old=reg['plan']
    diagnostic.validate_plan(old)
    if (canonical_sha256(old)!=reg['plan_sha256'] or old['parent']['params_sha256']!=plan['parent']['params_sha256']
            or old['parent']['sealed_files']!=plan['parent']['sealed_files'] or old['model']!=plan['model']
            or old['seeds']['fit']!=plan['inherited_fit_seed']
            or not set(old['seeds'].values())<=set(plan['parent_registered_seeds'])
            or result['status']!='complete_provisional_native_W_validation_no_training_release'
            or result['selected_index']!=0 or result['candidate_sha256']!=source['sealed_files']['provisional_candidate.pkl']
            or result['work']['confirmation_scalar_H_centers']!=4096
            or result['work']['confirmation_H_JVP_centers']!=2048
            or not all(result['source_seals_unchanged'].values()) or not all(result['parent_seals_unchanged'].values())):
        raise ValueError('completed diagnostic identity/counts do not authorize this import')
    step=result['finite_step_moments'];slope=result['bilateral_direction']
    de,dv=(step['candidate_minus_control'][i] for i in (0,2))
    se,sv=(step['paired_iid_delta_SE'][i] for i in (0,2))
    ds=slope['values'][2];ss=slope['independent_group_jackknife_SE'][2]
    if (not np.isfinite([de,dv,se,sv,ds,ss]).all() or min(se,sv,ss)<0
            or de>=0 or ds>=0 or dv-2*sv>0 or result['resolved_ascent']
            or result['resolved_finite_energy_increase']):
        raise ValueError('diagnostic does not satisfy fixed negative-mean/no-resolved-harm release boundary')
    candidate=load_checkpoint(directory/'provisional_candidate.pkl');direction=load_checkpoint(directory/'frozen_direction.pkl')
    meta=candidate['metadata'];dm=direction['metadata'];moments=meta['proposed_optimizer_state']
    expected=jax.tree.map(lambda p,d:p+d,parent['params'],direction['params'])
    algebra=diagnostic.legacy.compare_replay(expected,candidate['params'],tolerance=2e-11)
    if (not algebra['passed'] or meta['selected_fraction']!=1
            or meta['parent_params_sha256']!=plan['parent']['params_sha256']
            or dm['parent_params_sha256']!=plan['parent']['params_sha256']
            or pytree_sha256(candidate['optimizer_state'])!=pytree_sha256(parent['optimizer_state'])
            or pytree_sha256(moments)!=pytree_sha256(direction['optimizer_state'])
            or int(moments[0])!=int(parent['optimizer_state'][0])+1
            or pytree_sha256(moments[1:])!=pytree_sha256(parent['optimizer_state'][1:])
            or pytree_sha256((meta['fit_key'],meta['fit_states']))!=pytree_sha256((dm['fit_key'],dm['fit_states']))):
        raise ValueError('candidate/native direction/proposed optimizer/fit RNG lineage mismatch')
    key=jnp.asarray(meta['fit_key']);chains=jnp.asarray(meta['fit_states'])
    if (key.shape!=(2,) or key.dtype!=jnp.uint32 or chains.shape!=(256,36)
            or not np.isin(chains,[0,1]).all() or not np.all(np.asarray(chains).sum(1)==18)):
        raise ValueError('saved diagnostic fit chain/key shape or physical sector mismatch')
    with np.load(directory/'confirmation_products.npz') as products:
        if products['rows'].shape!=(2048,36) or products['psi'].shape!=(2048,2):
            raise ValueError('complete2048-record paired confirmation required')
    state=(candidate['params'],moments,key,chains)
    return state,dict(completed_updates=1,absolute_completed_steps=477,
        params_sha256=pytree_sha256(state[0]),optimizer_sha256=pytree_sha256(moments),
        training_rng_sha256=pytree_sha256(state[2:]),native_direction_algebra=algebra,
        provisional_parent_optimizer_replaced_by_proposed_optimizer_once=True,
        fit_chain_was_reinitialized_by_diagnostic=True,fit_chain_continued_without_new_initialization=True,
        confirmation_role='consumed_release_not_terminal_quality',source_work=result['work'],
        prior_failed_attempt_work=result.get('prior_attempt_work',{}),
        source_wall_seconds=result['wall_seconds'],prior_failed_attempt_wall_seconds=result.get('prior_attempt_wall_seconds',0),
        release_diagnostics=dict(finite_energy_difference_per_site=de,finite_energy_difference_SE=se,
            variance_difference_per_site=dv,variance_difference_SE=sv,slope_per_site=ds,slope_SE=ss,
            criterion='negative means and no resolved2SE variance harm; root authorization also required'))


def science_seconds(plan:dict,started_epoch:float,now:float,end_time:str|None) -> float:
    """One cumulative14h block and original campaign deadline, not per phase."""
    deadline=datetime.fromisoformat(plan['campaign_deadline_utc']).timestamp()
    if not np.isfinite([started_epoch,now]).all() or now<started_epoch:raise ValueError('invalid monotonic wall chronology')
    cap=min(plan['wall_cap_seconds']-(now-started_epoch),deadline-now-120.)
    if cap<=0:raise TimeoutError('cumulative science cap or original campaign deadline exhausted')
    return diagnostic.allocation_science_seconds(cap,end_time,now)


def compare_transition(expected:tuple,actual:tuple,first:dict,second:dict,left:Path,right:Path) -> dict:
    """Check full native state, discrete choice and actual sampled/calibration rows."""
    replay=diagnostic.legacy.compare_replay(expected,actual,tolerance=2e-11)
    if not replay['passed']:raise ArithmeticError('full params/moments/RNG process replay failed')
    for name in ('accepted','selected_index','actual_native_factor'):
        if first.get(name)!=second.get(name):raise ArithmeticError('finite selection differs in replay: '+name)
    numeric=lambda tree:jax.tree.map(lambda x:np.frombuffer(x.encode(),np.uint8) if isinstance(x,str) else x,tree)
    metrics=diagnostic.legacy.compare_replay(numeric(first['calibration']),numeric(second['calibration']),tolerance=2e-11)
    if not metrics['passed']:raise ArithmeticError('finite calibration metrics differ in replay')
    for name in ('fit_products.npz','selection_rows.npz','guard_rows.npz'):
        with np.load(left/name) as a,np.load(right/name) as b:
            if not np.array_equal(a['rows'],b['rows']):raise ArithmeticError('process replay generated different bit rows')
    return dict(status='passed',full_state=replay,calibration=metrics,bit_rows_identical=True,
        parameters_optimizer_keys_states_all_compared=True,optimizer_updates_committed_by_replay=0)


def finite_step(plan:dict,rt:dict,state:tuple,index:int,output:Path) -> tuple[tuple,dict]:
    """One actual full-W direction and one nonadaptive supported finite-grid decision.

    Parent fallback aborts this attempted transition; it never retries with new
    rows. The old training-sample metric does not shrink the independently
    calibrated finite candidate again. The safe raw-SR proposal itself retains
    the existing ridge, tangent cap and numerical checks.
    """
    if not 1<=index<16 or output.exists():raise ValueError('fresh post-import transition directory/index1..15 required')
    output.mkdir(parents=True);parent,moments,key,chains=state;start=perf_counter()
    report=dict(index=index,accepted=False,timings={},work=dict(fit_H_centers=0,score_rows=0,calibration_draws=0,
        calibration_bank_centers=0,native_amplitude_replay_centers=0),parent_params_sha256=pytree_sha256(parent))
    def timed(name,call):
        tick=perf_counter();value=call();jax.block_until_ready(value);report['timings'][name]=perf_counter()-tick
        if diagnostic._device_peak()>70*1024**3:raise MemoryError('registered70GiB device ceiling exceeded')
        atomic_json(output/'progress.json',report);return value
    try:
        key,chains,rows=timed('fit_sampling',lambda:rt['next_sampler'](rt['prepare'](parent),key,chains))
        report['work']['fit_H_centers_requested']=4096
        psi,hp=timed('fit_H',lambda:rt['action'](rt['prepare'](parent),rows));report['work']['fit_H_centers']=4096
        if np.any(np.asarray(psi)==0) or not np.isfinite(np.asarray(hp/psi)).all():raise ArithmeticError('invalid Born fit ratio; no floor')
        np.savez(output/'fit_products.npz',rows=rows,psi=psi,hpsi=hp)
        report['work']['score_rows_requested']=4096
        proposed,next_moments,sr=timed('full_native_SR',lambda:rt['native_update'](parent,moments,rows,hp/psi,.03,warm=True))
        report['work']['score_rows']=4096;report['SR']=jax.tree.map(lambda x:np.asarray(x).item(),sr)
        delta=jax.tree.map(lambda a,b:a-b,proposed,parent)
        save_checkpoint(output/'raw_SR_direction.pkl',delta,optimizer_state=next_moments,
            metadata=dict(parent_params_sha256=report['parent_params_sha256'],fit_key=np.asarray(key),fit_states=np.asarray(chains),
                proposed_optimizer_uncommitted=True))
        if not bool(sr['accepted']):report['reason']='raw_SR_rejected';return state,report
        candidates=[jax.tree.map(lambda p,d:p+factor*d,parent,delta) for factor in FACTORS]
        banks=[];notes=[]
        for i,p in enumerate(candidates):
            bank,note=diagnostic.normalized_raw_bank(p,rt);banks.append(bank);notes.append(note)
            save_checkpoint(output/f'finite_native_{i}.pkl',p,metadata=dict(native_factor=float(FACTORS[i]),parent_params_sha256=report['parent_params_sha256']))
            save_checkpoint(output/f'finite_canonical_{i}.pkl',bank,metadata=note)
        report['normalization']=notes
        atomic_json(output/'finite_grid_frozen.json',dict(report,native_factors=FACTORS.tolist(),
            native_sha256=[file_sha256(output/f'finite_native_{i}.pkl') for i in range(5)],
            raw_direction_sha256=file_sha256(output/'raw_SR_direction.pkl')))
        panels=[]
        for role in ('selection','guard'):
            bits,draw=timed(role+'_IID_draws',lambda role=role:diagnostic.draw_orbit_rows(banks,rt,2048,plan['step_seeds'][index-1][role]))
            np.savez(output/(role+'_rows.npz'),rows=bits,latent=draw.pop('latent'),Born_replay=draw.pop('Born_replay'))
            atomic_json(output/(role+'_draw.json'),draw);panels.append(bits);report['work']['calibration_draws']+=2048
        bits=np.concatenate(panels);products=[]
        for i,bank in enumerate(banks):
            prepared=diagnostic.prepare_feature_lut_bank(bank,rt['cfg'])
            report['work']['calibration_bank_centers_requested']=(i+1)*4096
            products.append(np.asarray(timed('orbit_bank_'+str(i),lambda p=prepared:rt['orbit_query'](p,bits))))
            report['work']['calibration_bank_centers']+=4096
        products=np.stack(products,axis=1);phi=products[:,:,0];guides=products[:,:,1].real;density=guides.mean(1)
        np.savez(output/'calibration_products.npz',rows=bits,phi=phi,guide_densities=guides,proposal=density)
        errors=[]
        for i,p in enumerate(candidates):
            actual=np.asarray(timed('native_replay_'+str(i),lambda p=p:diagnostic.native_projected_amplitude(rt,p,bits[:2])))
            actual=actual/np.sqrt(notes[i]['original_raw_norm'])
            error=float(np.max(abs(actual-phi[:2,i]))/max(np.max(abs(phi[:2,i])),1e-300));errors.append(error)
            report['work']['native_amplitude_replay_centers']+=2
        if not np.isfinite(errors).all() or max(errors)>2e-11:raise ArithmeticError('finite native/canonical amplitude replay failed')
        selected,calibration=select_finite_grid(phi,density,RELATIVE_GRID,split=2048)
        report.update(selected_index=selected,calibration=calibration,native_replay_relative_errors=errors,
            selected_fraction_is_relative_to_two_times_raw_SR_delta=True,actual_native_factor=float(FACTORS[selected]),
            no_post_IID_training_metric_shrink=True)
        if selected==4:report['reason']='independent_finite_geometry_rejected';return state,report
        candidate=candidates[selected];next_state=(candidate,next_moments,key,chains)
        save_training_state(output/'candidate_state.pkl',next_state,dict(completed_updates=index+1,
            absolute_completed_steps=477+index,parent_params_sha256=report['parent_params_sha256'],
            proposed_until_guard_and_replay=True,actual_native_factor=float(FACTORS[selected])))
        report.update(accepted=True,reason='finite_geometry_passed',candidate_sha256=file_sha256(output/'candidate_state.pkl'),
            candidate_params_sha256=pytree_sha256(candidate))
        return next_state,report
    finally:
        report['wall_seconds']=perf_counter()-start;atomic_json(output/'transition.json',report)


def final_audit(plan:dict,rt:dict,parent:dict,state:tuple,output:Path,remaining) -> dict:
    """Fresh fixed-endpoint IID paired complete-H audit, retaining exact nodes.

    Q=one quarter of parent/candidate raw orbit densities and their all-edge
    pushes. Known raw norms are removed before constructing either component.
    The target is the full coherent432-image state; H is not restricted to the
    retained TTN space. No history-based endpoint selection is performed.
    """
    output.mkdir();start=perf_counter();products=[];rows=None
    report=dict(status='started',records=0,scalar_H_centers=0,
        draw_records=0,seed=plan['seeds']['audit'],statistics_seed=plan['seeds']['audit_statistics'],
        comparison_parent_global_step=476,endpoint_params_sha256=pytree_sha256(state[0]),
        independent_from_fit_calibration_and_diagnostic_confirmation=True)
    try:
        banks=[];normalization=[]
        for native in (parent,state[0]):
            bank,note=diagnostic.normalized_raw_bank(native,rt);banks.append(bank);normalization.append(note)
        report['normalization']=normalization
        rows,draw=diagnostic.draw_orbit_rows(banks,rt,16384,plan['seeds']['audit'],bridge=True)
        np.savez(output/'rows.npz',rows=rows,latent=draw.pop('latent'),Born_replay=draw.pop('Born_replay'))
        atomic_json(output/'draw.json',draw);report['draw_records']=16384
        prepared=[diagnostic.prepare_feature_lut_bank(bank,rt['cfg']) for bank in banks]
        times=[];chunk=plan['H_chunk']
        for offset in range(0,len(rows),chunk):
            bits=rows[offset:offset+chunk];tick=perf_counter();pair=[]
            for bank in prepared:
                report['scalar_H_centers_requested']=report['scalar_H_centers']+len(bits)
                values=jax.block_until_ready(rt['bridge'](bank,bits))
                pair.append(np.stack([np.asarray(v) for v in values],axis=1))
                report['scalar_H_centers']+=len(bits)
            products.append(np.stack(pair,axis=1));times.append(perf_counter()-tick);report['records']+=len(bits)
            if diagnostic._device_peak()>70*1024**3:raise MemoryError('terminal audit exceeded70GiB')
            if len(times)==2:
                required=times[-1]*(len(rows)-report['records'])/len(bits)+120.
                report['cost_gate']=dict(cold_batch_seconds=times[0],warm_batch_seconds=times[1],
                    batch_records=len(bits),forecast_remaining_seconds=required,available_seconds=remaining())
                atomic_json(output/'cost_gate.json',report['cost_gate'])
                if required>remaining():
                    report['status']='cost_gate_stopped_fixed_audit_incomplete';break
            if (offset//chunk)%32==0:
                atomic_json(output/'progress.json',report);print('fresh endpoint audit',report['records'],'/16384',flush=True)
        data=np.concatenate(products);q=(data[:,:,2].real+data[:,:,3].real).mean(1)/2
        np.savez(output/'products.npz',rows=rows[:len(data)],psi=data[:,:,0],hpsi=data[:,:,1],
            guide_components=data[:,:,2:],proposal=q)
        report['H_batch_seconds']=times
        if len(data)==16384:
            report['moments']=paired_iid_moments(data[:,:,0],data[:,:,1],q,np.repeat(np.arange(128),128),
                n_sites=36,seed=plan['seeds']['audit_statistics'])
            report['status']='complete'
        return report
    finally:
        if products and not (output/'products.npz').exists():
            data=np.concatenate(products);q=(data[:,:,2].real+data[:,:,3].real).mean(1)/2
            np.savez(output/'products.npz',rows=rows[:len(data)],psi=data[:,:,0],hpsi=data[:,:,1],
                guide_components=data[:,:,2:],proposal=q)
        report['wall_seconds']=perf_counter()-start;atomic_json(output/'audit.json',report)


def input_seals(plan:dict) -> dict[str,bool]:
    """Check original external parent and diagnostic files without mutating them."""
    result={}
    for label in ('parent','source_diagnostic'):
        item=plan[label];seals=dict(item['sealed_files'])
        if label=='parent':seals['terminal_audit.npz']=item['terminal_products_sha256']
        result.update({label+'/'+name:file_sha256(Path(item['directory'])/name)==digest for name,digest in seals.items()})
    return result


def _new_registration(plan:dict,output:Path,parent_info:dict,import_info:dict) -> dict:
    if output.exists() and any(output.iterdir()):raise FileExistsError('new isolated campaign directory required')
    output.mkdir(parents=True,exist_ok=True)
    sources=transitive_local_source_manifest([Path(__file__)],root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    registration=dict(plan=plan,plan_sha256=canonical_sha256(plan),sources=sources,parent=parent_info,
        imported_update=import_info,started_epoch=time(),runtime=diagnostic.legacy.numerical_runtime_fingerprint(),
        final_audit_comparator_global_step=476,declared_endpoint_global_step=492,
        diagnostic_confirmation_is_consumed_release_data=True)
    for label in ('parent','source_diagnostic'):
        item=plan[label];names=list(item['sealed_files'])
        if label=='parent':names.append('terminal_audit.npz')
        for name in names:
            target=output/(label+'_snapshot')/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(Path(item['directory'])/name,target)
    for name in sources:
        target=output/'registered_sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    atomic_json(output/'registration.json',registration);return registration


def _read_registration(plan:dict,output:Path) -> dict:
    registration=json.loads((output/'registration.json').read_text())
    if registration['plan_sha256']!=canonical_sha256(plan) or registration['plan']!=plan:
        raise ValueError('continuation plan changed between processes')
    for name,digest in registration['sources'].items():
        if file_sha256(ROOT/name)!=digest or file_sha256(output/'registered_sources'/name)!=digest:
            raise ValueError('registered numerical source/archive changed: '+name)
    if not all(input_seals(plan).values()):raise ValueError('external immutable input changed')
    return registration


def _commit(output:Path,state:tuple,completed:int,history:list) -> None:
    save_training_state(output/'checkpoint.pkl',state,dict(completed_updates=completed,
        absolute_completed_steps=476+completed,history=history,
        original_parent_global_step=476,imported_verified_diagnostic_updates=1,
        direct_SR_moment_arrays_preserved_counter_incremented_once_per_committed_update=True))


def work_accounting(output:Path,current_phase:dict) -> dict:
    """Actual campaign work, charging the uncommitted full process replay too."""
    transitions={str(path.parent.name):json.loads(path.read_text()) for path in output.glob('transition_*/transition.json')}
    totals={}
    for row in transitions.values():
        for name,value in row['work'].items():totals[name]=totals.get(name,0)+value
    phases={path.stem.removeprefix('phase_'):json.loads(path.read_text()) for path in output.glob('phase_*.json')}
    phases[current_phase['phase']]=current_phase
    parent=sum(row.get('work',{}).get('parent_replay_H_centers',0) for row in phases.values())
    imported=sum(row.get('work',{}).get('imported_replay_H_centers',0) for row in phases.values())
    audit_path=output/'terminal_audit'/'audit.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    return dict(transition_attempts=len(transitions),transition_work_including_uncommitted_replay=totals,
        parent_numerical_replay_H_centers=parent,imported_numerical_replay_H_centers=imported,
        terminal_scalar_H_centers=audit.get('scalar_H_centers',0),
        terminal_IID_draw_records=audit.get('draw_records',0),new_H_JVP_centers=0,
        total_new_scalar_H_centers=parent+imported+totals.get('fit_H_centers',0)+audit.get('scalar_H_centers',0),
        imported_diagnostic_costs_recorded_separately_in_registration=True,
        committed_optimizer_updates_not_equal_query_or_replay_attempts=True)


def run(plan:dict,phase:str,output:Path) -> dict:
    """One allocation, separate prepare/replay/train processes; no job submission.

    Repeating a finished phase only returns its saved receipt. A failed phase
    cannot consume another draw or silently retry an optimizer step.
    """
    validate_plan(plan,executing=True)
    if phase not in ('prepare','replay','train'):raise ValueError('unknown phase')
    if jax.default_backend()!='gpu' or not any('A100' in d.device_kind for d in jax.devices()):
        raise RuntimeError('approved A100 runtime required; local mocks are tests only')
    phase_path=output/('phase_'+phase+'.json')
    if phase_path.exists():
        _read_registration(plan,output);return json.loads(phase_path.read_text())
    saved,parent_info=diagnostic.checked_parent(diagnostic_plan(plan))
    if phase=='prepare':
        imported,import_info=checked_diagnostic_import(plan,saved)
        reg=_new_registration(plan,output,parent_info,import_info)
    else:
        reg=_read_registration(plan,output)
        ready=json.loads((output/'phase_prepare.json').read_text())
        if ready['status']!='prepared_first_new_transition_uncommitted':raise ValueError('prepare did not release replay/train')
        imported,_=load_training_state(output/'imported_477.pkl')
        if phase=='train' and json.loads((output/'phase_replay.json').read_text())['status']!='replay_passed':
            raise ValueError('separate first-new-transition replay must pass before commit')
    start=perf_counter();cpu=process_time();report=dict(phase=phase,status='started',work={},
        runtime=dict(diagnostic.legacy.numerical_runtime_fingerprint(),process_id=os.getpid()),completed_updates=1)
    remaining=lambda:science_seconds(plan,reg['started_epoch'],time(),os.environ.get('SLURM_JOB_END_TIME'))
    try:
        with _node_timeout(remaining()):
            rt=make_runtime(plan,saved['params'])
            report['parent_numerical_replay']=diagnostic.audited_parent_replay(
                rt,saved['params'],output/'parent_snapshot'/'terminal_audit.npz')
            report['work']['parent_replay_H_centers']=2
            atomic_json(output/(phase+'_parent_replay.json'),report['parent_numerical_replay'])
            if phase=='prepare':
                # The checkpoint remains in original native coordinates. Its
                # canonical normalized copy is only an independent decoder check.
                _,normalization=diagnostic.normalized_raw_bank(imported[0],rt)
                with np.load(output/'source_diagnostic_snapshot'/'confirmation_products.npz') as p:
                    report['work']['imported_replay_H_centers_requested']=2
                    actual=np.stack([np.asarray(v) for v in jax.block_until_ready(rt['action'](
                        rt['prepare'](imported[0]),jnp.asarray(p['rows'][:2],jnp.int32)))],axis=1)
                    expected=np.stack((p['psi'][:2,1],p['hpsi'][:2,1]),axis=1)
                    actual=actual/np.sqrt(normalization['original_raw_norm'])
                errors=np.max(abs(actual-expected),axis=0)/np.maximum(np.max(abs(expected),axis=0),1e-300)
                if not np.isfinite(errors).all() or max(errors)>2e-11:raise ArithmeticError('imported477 native A1/H replay failed')
                report['imported_native_psi_Hpsi_replay_relative_errors']=errors.tolist()
                report['work']['imported_replay_H_centers']=2
                save_training_state(output/'imported_477.pkl',imported,reg['imported_update'])
                _commit(output,imported,1,[dict(kind='verified_diagnostic_import',global_step=477)])
                state,first=finite_step(plan,rt,imported,1,output/'transition_01')
                report['transition']=first
                if not first['accepted']:
                    report['status']='first_new_transition_rejected_imported477_preserved';return report
                # Includes one replay plus all14 remaining transitions. This is
                # an observed cold-cost forecast, not an assumed speedup.
                required=15*first['wall_seconds']+plan['audit_reserve_seconds']+120
                report['cost_gate']=dict(observed_first_transition_seconds=first['wall_seconds'],
                    remaining_transitions_including_replay=15,forecast_remaining_seconds=required,
                    available_seconds=remaining(),audit_reserve_seconds=plan['audit_reserve_seconds'])
                report['status']=('prepared_first_new_transition_uncommitted' if required<=remaining()
                    else 'cost_gate_stopped_imported477_preserved')
            elif phase=='replay':
                if ready['runtime']['process_id']==report['runtime']['process_id']:
                    raise ValueError('first-new-transition replay requires a fresh process')
                expected,_=load_training_state(output/'transition_01'/'candidate_state.pkl')
                actual,second=finite_step(plan,rt,imported,1,output/'transition_01_replay')
                first=json.loads((output/'transition_01'/'transition.json').read_text())
                replay=compare_transition(expected,actual,first,second,output/'transition_01',output/'transition_01_replay')
                replay['prepare_process_id']=ready['runtime']['process_id'];replay['replay_process_id']=report['runtime']['process_id']
                atomic_json(output/'process_replay.json',replay);report['transition']=second;report['status']='replay_passed'
            else:
                state,_=load_training_state(output/'transition_01'/'candidate_state.pkl')
                history=[dict(kind='verified_diagnostic_import',global_step=477),
                    dict(kind='new_finite_FS_update',index=1,global_step=478)]
                _commit(output,state,2,history);report['completed_updates']=2
                elapsed_steps=[json.loads((output/'transition_01'/'transition.json').read_text())['wall_seconds']]
                stopped=None
                for index in range(2,16):
                    required=1.15*max(elapsed_steps[-3:])+plan['audit_reserve_seconds']+120
                    if required>remaining():stopped='remaining_time_reserved_for_fixed_audit';break
                    proposed,step=finite_step(plan,rt,state,index,output/f'transition_{index:02d}')
                    elapsed_steps.append(step['wall_seconds'])
                    if not step['accepted']:stopped='finite_geometry_or_SR_rejection_no_retry';break
                    state=proposed;history.append(dict(kind='new_finite_FS_update',index=index,global_step=477+index))
                    _commit(output,state,index+1,history);report['completed_updates']=index+1
                    atomic_json(output/'training_progress.json',dict(report,absolute_completed_steps=477+index))
                report['stop_reason']=stopped
                report['absolute_completed_steps']=476+report['completed_updates']
                report['checkpoint_sha256']=file_sha256(output/'checkpoint.pkl');report['params_sha256']=pytree_sha256(state[0])
                atomic_json(output/'endpoint_frozen.json',report)
                audit=final_audit(plan,rt,saved['params'],state,output/'terminal_audit',remaining);report['terminal_audit']=audit
                report['status']=('fixed16_updates_complete_independent_audit' if report['completed_updates']==16
                    else 'incomplete_prefix_independently_audited') if audit['status']=='complete' else 'endpoint_audit_incomplete'
    except (TimeoutError,MemoryError) as error:
        report.update(status='bounded_incomplete',error=repr(error))
    except Exception as error:
        report.update(status='failed_validation',error=repr(error));raise
    finally:
        report.update(wall_seconds=perf_counter()-start,CPU_seconds=process_time()-cpu,
            cumulative_block_wall_seconds=time()-reg['started_epoch'],peak_device_bytes=diagnostic._device_peak())
        report['source_seals_unchanged']={name:file_sha256(ROOT/name)==seal for name,seal in reg['sources'].items()}
        report['input_seals_unchanged']=input_seals(plan)
        intact=all(report['source_seals_unchanged'].values()) and all(report['input_seals_unchanged'].values())
        if not intact:report['status']='failed_immutable_source_or_input_seal'
        report['campaign_work_accounting']=work_accounting(output,report)
        atomic_json(phase_path,report)
        if phase=='train':atomic_json(output/'results.json',report)
        if not intact:raise ArithmeticError('immutable source/input changed')
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','replay','train'],required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());result=run(plan,args.phase,args.output)
    print(json.dumps(dict(status=result['status'],phase=args.phase,completed_updates=result.get('completed_updates')),sort_keys=True))
    success={'prepare':'prepared_first_new_transition_uncommitted','replay':'replay_passed',
        'train':'fixed16_updates_complete_independent_audit'}
    if result['status']!=success[args.phase]:raise SystemExit(2)


if __name__=='__main__':main()
