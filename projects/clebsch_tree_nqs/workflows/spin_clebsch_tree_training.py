"""Resumable physical-spin CT VMC: SR+Adam warmup, then clipped Adam.

Run a two-step sampled pilot before releasing the registered1000-step square8
accuracy reproduction. Reuse the PASSED cost-gate parameters; no parameter
reinitialization, carrier, head, ED or global fusion-coefficient conversion.
"""
from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.vmc.physical_sr_adam import make_physical_sr_adam
from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains,run_exchange_sweeps
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.checkpoint import save_checkpoint,load_checkpoint
from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest,numerical_runtime_identity
from su2_nqs.models.clebsch_tree_projected import make_projected_clebsch
from su2_nqs.models.clebsch_tree_orbit_dag import make_clebsch_execution
from .spin import _bonds
from .spin_clebsch_tree_sampled import components,ROOT

jax.config.update('jax_enable_x64',True)


def _validate_optimizer_condition_change(old: dict,new: dict,continuation: dict) -> None:
    """Remove only explicitly registered optimizer changes from a comparison.

    Physical model, Hamiltonian, symmetry, sampling kernel and random seed are
    never allowable optimizer conditions. The old single-radius registration
    remains valid. Moment resets are opt-in import events, not resume events.
    """
    changed=continuation.get('optimizer_condition_change')
    allowed={'maximum_FS_sine','sr_warm_steps','warm_samples','adam_samples',
             'sr_damping','warm_lr','adam_lr','sr_use_adam','sr_tangent_fs_radius',
             'strict_nonzero_updates'}
    if changed is not None:
        if not isinstance(changed,dict) or not changed or set(changed)-allowed:
            raise ValueError('unsupported registered optimizer condition change')
        for name,values in changed.items():
            default=(True if name=='sr_use_adam' else False if name=='strict_nonzero_updates' else None)
            if values != dict(from_value=old.get(name,default),to_value=new.get(name,default)):
                raise ValueError('registered optimizer condition values disagree')
            old.pop(name,None); new.pop(name,None)
    reset=continuation.get('reset_optimizer_moments',False)
    if not isinstance(reset,bool):
        raise ValueError('optimizer moment reset must be boolean')
    if reset and not continuation.get('optimizer_reset_reason'):
        raise ValueError('optimizer moment reset needs a registered reason')


def _validate_endpoint_budget_change(old: dict,new: dict,continuation: dict) -> None:
    """Permit one explicit bounded endpoint/time extension, never hidden work.

    This only changes the registered stopping rule and quality-save locations.
    It cannot alter the Hamiltonian, width, sampler, initializer or optimizer.
    The imported checkpoint is checked against ``from_completed_steps`` by
    ``train`` before any update. Old historical work is always retained.
    """
    change=continuation.get('endpoint_budget_change')
    if change is None:return
    if set(change)!={'from_completed_steps','additional_updates','time_extension_seconds',
                    'registered_steps','registered_milestones'}:
        raise ValueError('explicit endpoint, time and milestone budget declaration required')
    start=change['from_completed_steps']; extra=change['additional_updates']
    extension=change['time_extension_seconds']
    if (any(isinstance(v,bool) or not isinstance(v,int) for v in (start,extra,extension))
            or start<0 or not 1<=extra<=1000 or extension<0
            or old['steps']<start or new['steps']!=start+extra
            or change['registered_steps']!=dict(from_value=old['steps'],to_value=new['steps'])
            or change['registered_milestones']!=dict(from_value=old['milestones'],to_value=new['milestones'])
            or new['maximum_training_seconds']!=old['maximum_training_seconds']+extension
            or not set((start,start+2,start+4,start+extra))<=set(new['milestones'])
            or any(isinstance(v,bool) or not isinstance(v,int) or not 0<=v<=new['steps']
                   for v in new['milestones'])):
        raise ValueError('bounded continuation endpoint/time/milestone values disagree')
    for key in ('steps','milestones','maximum_training_seconds'):
        old.pop(key);new.pop(key)


def _execution_settings(spec: dict) -> tuple[str,int,int,int]:
    """Validate pure execution options; absent kernel preserves legacy behavior."""
    execution=spec.get('execution',{})
    if (not isinstance(execution,dict)
            or set(execution)-{'projection_chunk','score_chunk','H_sample_chunk','kernel'}):
        raise ValueError('unsupported execution settings')
    kernel=execution.get('kernel','legacy')
    if not isinstance(kernel,str) or kernel not in ('legacy','orbit_dag'):
        raise ValueError('execution kernel must be legacy or orbit_dag')
    batches={k:v for k,v in execution.items() if k != 'kernel'}
    if any(not isinstance(v,int) or isinstance(v,bool) or v < 1 for v in batches.values()):
        raise ValueError('execution settings require positive integer batch sizes')
    orbit_chunk=execution.get('projection_chunk',spec['projection_chunk'])
    score_chunk=execution.get('score_chunk',spec['score_chunk'])
    hchunk=execution.get('H_sample_chunk',spec['H_sample_chunk'])
    if kernel != 'legacy' and not spec.get('fold_root_swap',False):
        raise ValueError('orbit-DAG requires the explicitly validated projected execution route')
    if not spec.get('fold_root_swap',False) and orbit_chunk != spec['projection_chunk']:
        raise ValueError('projection execution tuning requires the validated root fold')
    return kernel,orbit_chunk,score_chunk,hchunk


def train(spec: dict,output: Path,until_step: int,session_seconds: float) -> dict:
    """Bounded genuine Born sampling and full-H training; atomic step recovery."""
    if not isinstance(spec.get('strict_nonzero_updates',False),bool):
        raise ValueError('strict_nonzero_updates must be an explicit boolean')
    if spec['require_gpu'] and jax.default_backend() != 'gpu': raise RuntimeError('GPU required')
    cost_directory=ROOT/Path(spec['cost_parent']['directory'])
    gate=json.loads((cost_directory/'results.json').read_text())
    parent=cost_directory/'checkpoint.pkl'
    if (gate['status'] != 'passed' or file_sha256(parent) != spec['cost_parent']['checkpoint_sha256']
            or gate['registration_sha256'] != spec['cost_parent']['registration_sha256']):
        raise ValueError('cost-gate parent not passed/sealed')
    for key in ('geometry','rows','columns','j2_over_j1','multiplicity_cap','high_spin_cap','maximum_q','projection_chunk'):
        if spec.get(key) != gate['config'].get(key): raise ValueError('training changed physical model '+key)
    cfg,raw,amp,bonds,images,order=components(spec)
    execution=spec.get('execution',{})
    kernel,orbit_chunk,score_chunk,hchunk=_execution_settings(spec)
    if spec.get('fold_root_swap',False):
        fast,_=make_clebsch_execution(cfg,order,images,orbit_chunk=orbit_chunk,kernel=kernel)
        amp=lambda z,row:fast(z,row[None])[0]
    nearest=[(a,b) for a,b,_ in _bonds(spec['geometry'],spec['rows'],spec['columns'],0.)]
    sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
                    package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    # Exact amplitude/compiler and physical operator/sampler sources match the
    # original PASSED gate; optimizer is a new, independently tested module.
    parent_reg=json.loads((cost_directory/'registration.json').read_text())
    if (parent_reg['sha256'] != canonical_sha256({k:v for k,v in parent_reg.items() if k != 'sha256'})
            or parent_reg['sha256'] != spec['cost_parent']['registration_sha256']
            or parent_reg['config'] != gate['config']):
        raise ValueError('cost-gate registration/content changed')
    for name,digest in parent_reg['source_manifest'].items():
        if file_sha256(ROOT/name) != digest: raise ValueError('passed cost-gate numerical source changed: '+name)
    optimizer_description=('tested local Adam beta(.9,.999),eps1e-8; explicit phase reset'
        if spec.get('sr_use_adam',True) else
        'plain real SR with uniform ridge and tangent/finite-FS guards; Adam moment arrays retained only for checkpoint continuity during direct-SR updates')
    identity=dict(config=spec,source_manifest=sources,runtime=numerical_runtime_identity(),
                  optimizer=optimizer_description)
    seal=canonical_sha256(identity); output.mkdir(parents=True,exist_ok=True)
    if (output/'registration.json').exists():
        if json.loads((output/'registration.json').read_text()) != dict(identity,sha256=seal):
            raise ValueError('immutable training identity changed')
    else: atomic_json(output/'registration.json',dict(identity,sha256=seal))
    p=load_checkpoint(parent)['params']; optimizer=_manual_adam_init(p)
    key=jax.random.PRNGKey(spec['training_seed']); states=initialize_spin_exchange_chains(key,n_chains=spec['chains'],n_sites=cfg.tree.n_leaves,n_up=cfg.tree.n_leaves//2)
    history=[]; completed=0; active=0.
    continuation=spec.get('training_resume_from')
    if continuation is not None and not (output/'checkpoint.pkl').exists():
        old_dir=ROOT/Path(continuation['directory']); old_reg=json.loads((old_dir/'registration.json').read_text())
        old_path=old_dir/'checkpoint.pkl'; previous=load_checkpoint(old_path)
        if (old_reg['sha256'] != continuation['registration_sha256']
                or file_sha256(old_path) != continuation['checkpoint_sha256']
                or previous['metadata']['registration_sha256'] != old_reg['sha256']):
            raise ValueError('immutable trained CT continuation parent changed')
        for name,digest in old_reg['source_manifest'].items():
            if file_sha256(Path(continuation['source_workspace'])/name) != digest:
                raise ValueError('old trained source snapshot changed')
        old=old_reg['config'].copy(); new=spec.copy()
        for obj in (old,new):
            for k in ('output','release','interpretation','fold_root_swap','training_resume_from','execution'):obj.pop(k,None)
        _validate_optimizer_condition_change(old,new,continuation)
        _validate_endpoint_budget_change(old,new,continuation)
        if old != new: raise ValueError('continuation changed physical model/SR/Adam/sampling protocol')
        if continuation.get('same_projected_physical_state_proved') is not True:
            raise ValueError('kernel continuation needs verified physical equivalence')
        p=previous['params']; optimizer=previous['optimizer_state']; meta=previous['metadata']
        states=jnp.asarray(meta['sampler_states']); key=jnp.asarray(meta['rng'])
        history=meta['history']; completed=meta['completed_steps']; active=meta['active_seconds']
        if pytree_sha256(p) != meta['params_sha256'] or int(optimizer[0]) != meta['optimizer_step']:
            raise ValueError('trained parent optimizer/parameters inconsistent')
        budget_change=continuation.get('endpoint_budget_change')
        if budget_change is not None and completed!=budget_change['from_completed_steps']:
            raise ValueError('bounded continuation imported a different completed endpoint')
        if continuation.get('reset_optimizer_moments',False):
            # Preserve the complete original checkpoint. Reset once, only on
            # importing into this separately registered condition arm. Future
            # calls restore this arm's optimizer unchanged from its checkpoint.
            optimizer=_manual_adam_init(p)
    if (output/'checkpoint.pkl').exists():
        saved=load_checkpoint(output/'checkpoint.pkl'); meta=saved['metadata']
        if (meta['registration_sha256'] != seal or meta['completed_steps'] != len(meta['history'])
                or int(saved['optimizer_state'][0]) != meta['optimizer_step']
                or pytree_sha256(saved['params']) != meta['params_sha256']
                or ('optimizer_sha256' in meta and pytree_sha256(saved['optimizer_state'])!=meta['optimizer_sha256'])
                or ('sampler_rng_sha256' in meta and pytree_sha256((jnp.asarray(meta['rng']),
                    jnp.asarray(meta['sampler_states'])))!=meta['sampler_rng_sha256'])):
            raise ValueError('training optimizer/RNG checkpoint mismatch')
        p=saved['params']; optimizer=saved['optimizer_state']; states=jnp.asarray(meta['sampler_states'])
        key=jnp.asarray(meta['rng']); history=meta['history']; completed=meta['completed_steps']; active=meta['active_seconds']
        if (states.shape != (spec['chains'],cfg.tree.n_leaves)
                or not np.isin(np.asarray(states),[0,1]).all()
                or not np.all(np.asarray(states).sum(axis=1) == cfg.tree.n_leaves//2)):
            raise ValueError('saved sampler is not in the physical M0 support')
    action=make_amplitude_heisenberg_action(amp,bonds,n_sites=cfg.tree.n_leaves)
    # Pure execution tuning must preserve the same projected physical state.
    # Replay on actual saved chains before allowing any new Born draw/update.
    if execution and not (output/'execution_replay.json').exists():
        slow,_=make_projected_clebsch(cfg,order,images,orbit_chunk=spec['projection_chunk'])
        slow_amp=lambda z,row:slow(z,row[None])[0]
        centers=states[:2]
        old_psi,old_h=jax.block_until_ready(make_amplitude_heisenberg_action(
            slow_amp,bonds,n_sites=cfg.tree.n_leaves)(p,centers))
        new_psi,new_h=jax.block_until_ready(action(p,centers))
        def relative(a,b):
            return float(jnp.linalg.norm(a-b)/jnp.maximum(jnp.linalg.norm(a),1e-25))
        errors=dict(amplitude=relative(old_psi,new_psi),H_product=relative(old_h,new_h))
        if kernel == 'orbit_dag':
            # This compares genuinely different implementations at the ACTUAL
            # continuation boundary, not a patched alias of the same backend.
            direction=jax.tree.map(lambda v:jnp.ones_like(v)/jnp.sqrt(
                sum(x.size for x in jax.tree.leaves(p))),p)
            old_jvp=jax.jvp(lambda z:slow(z,centers),(p,),(direction,))[1]
            new_jvp=jax.jvp(lambda z:fast(z,centers),(p,),(direction,))[1]
            old_grad=jax.grad(lambda z:jnp.mean(jnp.log(jnp.abs(slow(z,centers)))))(p)
            new_grad=jax.grad(lambda z:jnp.mean(jnp.log(jnp.abs(fast(z,centers)))))(p)
            errors['parameter_JVP']=relative(old_jvp,new_jvp)
            def flat(z):return jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(z)])
            errors['full_parameter_gradient']=relative(flat(old_grad),flat(new_grad))
        if max(errors.values()) > 2e-11:
            raise ArithmeticError('execution batching changed projected amplitude/H action')
        atomic_json(output/'execution_replay.json',dict(execution=execution,
            parent_params_sha256=pytree_sha256(p),maximum_relative_errors=errors,
            independent_legacy_backend=True,
            new_samples=0,new_parameter_updates=0))
    @jax.jit
    def batch_action(params,batch):
        # One dynamic-parameter compilation reused across all updates. Do not
        # create a new host lambda closing over a different p each step.
        psi,hpsi=jax.lax.map(lambda block:action(params,block),batch.reshape(-1,hchunk,cfg.tree.n_leaves))
        return psi.reshape(-1),hpsi.reshape(-1)
    update=make_physical_sr_adam(amp,p,damping=spec['sr_damping'],score_chunk=score_chunk,
        maximum_fs_sine=spec['maximum_FS_sine'],sr_use_adam=spec.get('sr_use_adam',True),
        sr_tangent_fs_radius=spec.get('sr_tangent_fs_radius'))
    def sample(p,key,states,proposals):
        return run_exchange_sweeps(p,key,states,amplitude_fn=amp,particle_kind='spin',sweeps=proposals,exchange_edges=nearest)
    sample=jax.jit(sample,static_argnames=('proposals',))
    stop=[False]
    def request_stop(signum,frame): stop[0]=True
    handlers={s:signal.getsignal(s) for s in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT)}
    for s in handlers: signal.signal(s,request_stop)
    started=perf_counter()
    def save():
        meta=dict(registration_sha256=seal,completed_steps=completed,history=history,
            sampler_states=np.asarray(states),rng=np.asarray(key),active_seconds=active,
            optimizer_step=int(optimizer[0]),params_sha256=pytree_sha256(p),
            optimizer_sha256=pytree_sha256(optimizer),sampler_rng_sha256=pytree_sha256((key,states)))
        save_checkpoint(output/'checkpoint.pkl',p,optimizer_state=optimizer,metadata=meta)
        if completed in spec['milestones']:
            save_checkpoint(output/'quality'/f'step{completed:05d}.pkl',p,optimizer_state=optimizer,metadata=meta)
    try:
        if completed == 0:
            states,_,key,_=sample(p,key,states,proposals=spec['burn_proposals'])
            jax.block_until_ready(states); active+=perf_counter()-started; save()
        elif not (output/'checkpoint.pkl').exists():
            # Archive an imported optimizer/RNG boundary before its first new
            # update; it remains recoverable even if compilation is interrupted.
            save()
        while completed < min(until_step,spec['steps']) and active < spec['maximum_training_seconds'] and perf_counter()-started < session_seconds and not stop[0]:
            if spec.get('strict_nonzero_updates',False) and history:
                next_count=spec['warm_samples'] if completed<spec['sr_warm_steps'] else spec['adam_samples']
                reserve=max(60.,min(1200.,1.25*history[-1]['seconds']*next_count/history[-1]['samples']))
                if (perf_counter()-started+reserve>=session_seconds
                        or active+reserve>=spec['maximum_training_seconds']):
                    break  # Keep the last atomic state rather than start a doomed full update.
            tick=perf_counter(); warm=completed < spec['sr_warm_steps']
            if completed == spec['sr_warm_steps'] and spec['reset_moments_at_phase_switch']:
                optimizer=_manual_adam_init(p)
            samples=[]; count=spec['warm_samples'] if warm else spec['adam_samples']
            if count%spec['chains']: raise ValueError('snapshots must preserve independent chain labels')
            for _ in range(count//spec['chains']):
                states,_,key,diagnostics=sample(p,key,states,proposals=spec['proposals_between_snapshots']); samples.append(states)
            batch=jnp.stack(samples).reshape(count,cfg.tree.n_leaves)
            sample_seconds=perf_counter()-tick; hstart=perf_counter()
            # Vectorized batches, not one host callback for every connection.
            if count%hchunk: raise ValueError('H sample chunks must divide the sample count')
            psi,hpsi=batch_action(p,batch); jax.block_until_ready(hpsi)
            energies=hpsi/psi
            if not np.isfinite(np.asarray(energies)).all(): raise FloatingPointError('nonfinite full-H Born ratios')
            hseconds=perf_counter()-hstart; ustart=perf_counter()
            candidate,next_optimizer,step_info=update(p,optimizer,batch,energies,spec['warm_lr'] if warm else spec['adam_lr'],warm=warm)
            if spec.get('strict_nonzero_updates',False) and (
                    not bool(step_info['accepted']) or not np.isfinite(float(step_info['FS_sine']))
                    or not 1e-10<float(step_info['FS_sine'])<=spec['maximum_FS_sine']
                    or float(step_info['ESS_fraction'])<.95
                    or pytree_sha256(candidate)==pytree_sha256(p)):
                raise ArithmeticError('rejected/zero/nonfinite physical update; previous checkpoint preserved')
            p,optimizer=candidate,next_optimizer
            jax.block_until_ready(p)
            elapsed=perf_counter()-tick; active+=elapsed; completed+=1
            row=dict(step=completed,samples=count,full_H_calls=count,
                energy_before_update_per_site=float(jnp.mean(energies))/cfg.tree.n_leaves,
                variance_before_update_per_site=float(jnp.var(energies))/cfg.tree.n_leaves,
                sampling_seconds=sample_seconds,H_seconds=hseconds,gradient_update_seconds=perf_counter()-ustart,
                seconds=elapsed,optimizer_phase=('SR_then_Adam' if spec.get('sr_use_adam',True)
                    else 'direct_SR') if warm else 'clipped_Adam',
                **{k:np.asarray(v).item() for k,v in step_info.items()})
            history.append(row); save()
            atomic_json(output/'progress.json',dict(completed_steps=completed,last=row,active_seconds=active,
                params_sha256=pytree_sha256(p),checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),registration_sha256=seal))
            print(row,flush=True)
        result=dict(completed_steps=completed,status='fixed_steps_complete' if completed == spec['steps'] else 'safe_pause',
            params_sha256=pytree_sha256(p),checkpoint_sha256=file_sha256(output/'checkpoint.pkl'),history=history,
            active_seconds=active,training_moments_not_independent_audit=True,full_A1_and_exact_S0=True,registration_sha256=seal)
        atomic_json(output/'training.json',result); return result
    finally:
        for s,h in handlers.items():signal.signal(s,h)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True); parser.add_argument('--until-step',type=int,default=2)
    parser.add_argument('--session-seconds',type=float,default=3300.)
    args=parser.parse_args(); spec=json.loads(args.config.read_text())
    result=train(spec,ROOT/spec['output'],args.until_step,args.session_seconds)
    print(result['status'],result['completed_steps'])
