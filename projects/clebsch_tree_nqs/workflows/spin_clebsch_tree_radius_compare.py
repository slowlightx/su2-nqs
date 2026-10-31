"""One-step, same-parent physical trust-radius comparison for square8 CT.

Both arms inherit identical frozen parameters, optimizer moments, chains and
RNG. They perform the same full-H/SR update except for the declared radius,
then each receives a fresh-chain full-H energy and variance audit. No longer
training phase or energy-selected checkpoint is released automatically.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,numerical_runtime_identity,transitive_local_source_manifest
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint
from .spin_clebsch_tree_sampled import ROOT
from .spin_clebsch_tree_training import train
from .spin_clebsch_tree_audit import audit


def held_direction_statistics(energies: np.ndarray,log_derivative: np.ndarray,
                              chain_labels: np.ndarray,*,n_sites: int) -> dict:
    """Energy slope and tangent distance for one frozen real-state direction.

    Labels identify complete correlated chains; jackknife deletions recenter
    both observables. This is an energy derivative, NOT a variance derivative.
    The latter also needs the derivative of the Hamiltonian product.
    """
    e=np.asarray(energies,dtype=np.float64);d=np.asarray(log_derivative,dtype=np.float64)
    labels=np.asarray(chain_labels)
    if (e.ndim != 1 or d.shape != e.shape or labels.shape != e.shape or n_sites < 1
            or not np.isfinite(e).all() or not np.isfinite(d).all()):
        raise ValueError('finite same-shape energies/direction and valid chain labels required')
    groups=np.unique(labels)
    if len(groups)<2: raise ValueError('independent-chain deletion needs two or more chains')
    slope=2*np.mean((e-e.mean())*(d-d.mean()))/n_sites
    leaves=np.array([2*np.mean((e[labels!=g]-e[labels!=g].mean())
                              *(d[labels!=g]-d[labels!=g].mean()))/n_sites for g in groups])
    se=np.sqrt((len(leaves)-1)/len(leaves)*np.sum((leaves-leaves.mean())**2))
    return dict(energy_slope_per_site=float(slope),conditional_chain_jackknife_SE=float(se),
                tangent_FS_sine_linear=float(np.std(d)),independent_chains=int(len(groups)),
                records=int(len(e)),variance_derivative_claim=False)


def diagnose_optimizer_directions(plan: dict,*,plan_sha256: str) -> dict:
    """Cross-chain same-parent direction test reusing an archived audit stream.

    The old independent audit becomes KNOWN diagnostic data here. Its native
    samples are replayed with the literal legacy sampler, then a few complete
    H queries verify record identity. No optimized state or new Born stream is
    generated; the configurations were already part of the archived audit.
    """
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from time import perf_counter
    from su2_nqs.vmc.adam import _manual_adam_init,_manual_adam_update
    from su2_nqs.vmc.dual_sampled_sr import sampled_overlap_diagnostics
    from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains,run_exchange_sweeps
    from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
    from su2_nqs.models.clebsch_tree_projected import make_projected_clebsch
    from su2_nqs.models.clebsch_tree_orbit_dag import make_clebsch_execution
    from .spin_clebsch_tree_sampled import components
    from .spin import _bonds
    jax.config.update('jax_enable_x64',True)
    if canonical_sha256(plan) != plan_sha256: raise ValueError('registered direction plan changed')
    if plan['require_gpu'] and jax.default_backend()!='gpu':raise RuntimeError('GPU required')
    output=ROOT/Path(plan['output'])
    if output.exists():raise FileExistsError('completed direction probes are immutable')
    parent=ROOT/Path(plan['parent']['directory']);cp=load_checkpoint(parent/'checkpoint.pkl')
    reg=json.loads((parent/'registration.json').read_text());p=cp['params']
    if (file_sha256(parent/'checkpoint.pkl') != plan['parent']['checkpoint_sha256']
            or reg['sha256'] != plan['parent']['registration_sha256']
            or pytree_sha256(p) != cp['metadata']['params_sha256']):
        raise ValueError('direction parent changed')
    audit_dir=ROOT/Path(plan['audit_directory']);a=json.loads((audit_dir/'audit.json').read_text())
    if (a['checkpoint_sha256'] != plan['parent']['checkpoint_sha256']
            or file_sha256(audit_dir/'audit.json') != plan['audit_json_sha256']
            or file_sha256(audit_dir/'local_energies.npz') != plan['audit_raw_sha256']):
        raise ValueError('archived direction audit changed')
    for name,sha in a['source_manifest'].items():
        if file_sha256(Path(plan['audit_source_workspace'])/name)!=sha:
            raise ValueError('archived independent audit source changed: '+name)
    cfg,_,_,bonds,images,order=components(reg['config']);physical=reg['config']
    chunk=physical.get('execution',{}).get('projection_chunk',physical['projection_chunk'])
    literal,_=make_projected_clebsch(cfg,order,images,orbit_chunk=chunk)
    fast,_=make_clebsch_execution(cfg,order,images,orbit_chunk=chunk,kernel='orbit_dag')
    amp=lambda z,row:fast(z,row[None])[0]
    literal_amp=lambda z,row:literal(z,row[None])[0]
    nearest=[(i,j) for i,j,_ in _bonds(physical['geometry'],physical['rows'],physical['columns'],0.)]
    def sample(key,states,proposals):
        return run_exchange_sweeps(p,key,states,amplitude_fn=literal_amp,particle_kind='spin',
                                   sweeps=proposals,exchange_edges=nearest)
    sample=jax.jit(sample,static_argnames=('proposals',))
    specification=a['audit_config'];rows=[];started=perf_counter()
    for stream in range(specification['streams']):
        key=jax.random.PRNGKey(specification['audit_seeds'][stream])
        states=initialize_spin_exchange_chains(key,n_chains=specification['chains_per_stream'],
            n_sites=cfg.tree.n_leaves,n_up=cfg.tree.n_leaves//2)
        states,_,key,_=sample(key,states,specification['burn_proposals'])
        for _ in range(specification['snapshots_per_stream']):
            states,_,key,_=sample(key,states,specification['proposals_between_snapshots']);rows.append(np.asarray(states))
    bits=jnp.asarray(np.stack(rows).reshape(-1,cfg.tree.n_leaves))
    with np.load(audit_dir/'local_energies.npz') as z:energy=np.asarray(z['local_energies']).ravel()
    labels=np.tile(np.arange(specification['chains_per_stream']),len(rows))
    labels+=np.repeat(np.arange(specification['streams'])*specification['chains_per_stream'],
                      specification['snapshots_per_stream']*specification['chains_per_stream'])
    selected=np.asarray(plan['H_replay_indices'],dtype=np.int32)
    psi,hpsi=jax.block_until_ready(make_amplitude_heisenberg_action(amp,bonds,
        n_sites=cfg.tree.n_leaves)(p,bits[selected]))
    error=float(np.max(abs(np.asarray(hpsi/psi)-energy[selected])))
    if error>2e-11:raise ArithmeticError('literal sampler does not replay archived H records')
    theta,unravel=ravel_pytree(p)
    score=jax.grad(lambda values,row:jnp.log(jnp.abs(amp(unravel(values),row))))
    scores=jax.jit(lambda values,rows:jax.lax.map(lambda row:score(values,row),rows,
        batch_size=plan['score_chunk']))(theta,bits)
    jax.block_until_ready(scores)
    if not bool(jnp.all(jnp.isfinite(scores))):raise FloatingPointError('nonfinite physical scores')
    amplitude_chunk=plan.get('amplitude_chunk',8)
    if len(bits)%amplitude_chunk:raise ValueError('amplitude chunk must divide archived records')
    @jax.jit
    def batch_amplitude(parameters,rows):
        # An unchunked1024-row orbit-DAG contraction can materialize>65GiB
        # intermediate features during XLA autotuning. Chunk ONLY execution;
        # retain every archived configuration and the complete coherent A1.
        return jax.lax.map(lambda block:fast(parameters,block),
            rows.reshape(-1,amplitude_chunk,cfg.tree.n_leaves)).reshape(-1)
    old=batch_amplitude(p,bits);folds=[]
    for fold in (0,1):
        fit=labels%2==fold;held=~fit;f=jnp.asarray(np.flatnonzero(fit));target=jnp.asarray(energy[fit])
        centered=(scores[f]-scores[f].mean(0))/jnp.sqrt(len(f))
        g=2*centered.T@(target-target.mean())/jnp.sqrt(len(f));ridge=plan['sr_damping']
        dual=jax.scipy.linalg.solve(ridge*jnp.eye(len(f))+centered@centered.T,centered@g,assume_a='pos')
        sr=(g-centered.T@dual)/ridge
        residual=float(jnp.linalg.norm(centered.T@(centered@sr)+ridge*sr-g)/jnp.linalg.norm(g))
        directions={'gradient':-g,'direct_SR':-sr}
        if plan.get('diagonal_scaled_regularizer') is not None:
            from su2_nqs.vmc.coordinate_sr import diagonal_regularized_real_sr_direction
            regularizer = plan['diagonal_scaled_regularizer']
            diagonal_direction, diagonal_info = diagonal_regularized_real_sr_direction(
                scores[f], target, **regularizer)
            if not bool(diagonal_info['valid']):
                raise FloatingPointError('invalid fit-only diagonal regularizer')
            directions['diagonal_scaled_SR'] = diagonal_direction
        balancing=None
        if plan.get('layer_balanced_SR',False):
            from su2_nqs.models.clebsch_tree_transfer import calibrate_clebsch_parameters
            from su2_nqs.models.literature_clebsch_tree import make_clebsch_amplitudes
            from su2_nqs.vmc.coordinate_sr import rescaled_real_sr_direction
            # D is selected ONLY using this fold's fit-chain activations.
            balanced,calibration=calibrate_clebsch_parameters(cfg,p,
                np.asarray(bits)[fit][:,np.asarray(order)])
            scales=dict(merges=tuple(jnp.ones_like(v)*r['positive_scale'] for v,r in
                zip(p['merges'],calibration['layers'],strict=True)),readout=jnp.ones_like(p['readout']))
            coordinate,_=ravel_pytree(scales);new_theta,_=ravel_pytree(balanced)
            balanced_direction,balance_info=rescaled_real_sr_direction(scores[f],target,coordinate,
                damping=ridge)
            if not bool(balance_info['valid']):raise FloatingPointError('invalid positive layer coordinates')
            directions['layer_balanced_SR']=balanced_direction
            # Independent full AD and raw/projected amplitude checks validate
            # the coordinate change, rather than relying on a scale argument.
            probe=f[:amplitude_chunk]
            new_scores=jax.jit(lambda values,rows:jax.lax.map(lambda row:score(values,row),rows,
                batch_size=plan['score_chunk']))(new_theta,bits[probe[:4]])
            score_error=float(jnp.linalg.norm(new_scores-scores[probe[:4]]/coordinate)
                /jnp.maximum(jnp.linalg.norm(new_scores),1e-25))
            raw=make_clebsch_amplitudes(cfg)
            a=raw(p,bits[probe][:,jnp.asarray(order)]);b=raw(balanced,bits[probe][:,jnp.asarray(order)])
            raw_error=float(jnp.linalg.norm(a/jnp.linalg.norm(a)-b/jnp.linalg.norm(b)))
            projected_balanced=batch_amplitude(balanced,bits[probe]);projected_old=old[probe]
            projected_error=float(jnp.linalg.norm(projected_balanced/jnp.linalg.norm(projected_balanced)
                -projected_old/jnp.linalg.norm(projected_old)))
            if max(score_error,raw_error,projected_error)>2e-10:
                raise ArithmeticError('positive coordinate change altered physical ray or score rule')
            balancing=dict(calibration=calibration,fit_chains_only=True,
                raw_normalized_ray_error=raw_error,full_A1_normalized_ray_error=projected_error,
                score_coordinate_rule_relative_error=score_error,
                relative_SR_residual=float(balance_info['relative_SR_residual']),
                coordinate_scale_min=float(jnp.min(coordinate)),coordinate_scale_max=float(jnp.max(coordinate)))
        for name,raw,lr in (('clipped_Adam',g*jnp.minimum(1.,.001/jnp.linalg.norm(g)),.001),
                            ('SR_then_Adam',sr,.005)):
            proposed,_=_manual_adam_update(p,unravel(raw),_manual_adam_init(p),lr)
            directions[name]=ravel_pytree(proposed)[0]-theta
        evaluations={};physical_derivatives={}
        for name,direction in directions.items():
            norm=jnp.std(scores[f]@direction)
            if not np.isfinite(float(norm)) or float(norm)<1e-15:
                evaluations[name]=dict(status='no_resolved_fit_tangent',fit_tangent_norm=float(norm))
                continue
            direction=direction*(plan['fit_tangent_FS_sine']/norm)
            if plan.get('match_finite_FS',False):
                # Match actual physical overlap, not a parameter norm or only
                # its tangent approximation. Held chains never set this scale.
                for _ in range(4):
                    fit_candidate=batch_amplitude(unravel(theta+direction),bits)
                    fs=sampled_overlap_diagnostics(jnp.log((fit_candidate[fit]/old[fit]).astype(jnp.complex128)))['sampled_fs_sine']
                    if not np.isfinite(float(fs)) or float(fs)<=1e-15:
                        raise FloatingPointError('finite physical radius cannot be calibrated')
                    direction*=plan['fit_tangent_FS_sine']/fs
            derivative=np.asarray(scores@direction)
            physical_derivatives[name]=derivative
            candidate=batch_amplitude(unravel(theta+direction),bits)
            finite={}
            for label,mask in (('fit',fit),('held',held)):
                diagnostics=sampled_overlap_diagnostics(jnp.log((candidate[mask]/old[mask]).astype(jnp.complex128)))
                finite[label]={k:float(v) for k,v in diagnostics.items()}
            if plan.get('match_finite_FS',False) and abs(finite['fit']['sampled_fs_sine']-plan['fit_tangent_FS_sine'])>2e-6:
                raise ArithmeticError('directions did not reach the same actual fit-state distance')
            evaluations[name]=dict(fit=held_direction_statistics(energy[fit],derivative[fit],labels[fit],n_sites=cfg.tree.n_leaves),
                held=held_direction_statistics(energy[held],derivative[held],labels[held],n_sites=cfg.tree.n_leaves),
                finite_step_overlap=finite,parameter_direction_norm=float(jnp.linalg.norm(direction)))
        paired={}
        if balancing is not None:
            paired['paired_balanced_minus_direct_SR']=held_direction_statistics(energy[held],
                (physical_derivatives['layer_balanced_SR']-physical_derivatives['direct_SR'])[held],
                labels[held],n_sites=cfg.tree.n_leaves)
        if 'diagonal_scaled_SR' in physical_derivatives:
            paired['paired_diagonal_minus_direct_SR'] = held_direction_statistics(energy[held],
                (physical_derivatives['diagonal_scaled_SR']-physical_derivatives['direct_SR'])[held],
                labels[held], n_sites=cfg.tree.n_leaves)
        folds.append(dict(fit_parity=fold,gradient_norm=float(jnp.linalg.norm(g)),
            relative_SR_residual=residual,directions=evaluations,
                **paired, **({'layer_balancing':balancing} if balancing is not None else {})))
    result=dict(status='complete',plan_sha256=plan_sha256,parent=plan['parent'],
        elapsed_seconds=perf_counter()-started,archived_audit_is_now_KNOWN_diagnostic=True,
        replayed_existing_samples=int(len(bits)),new_Born_streams=0,new_full_H_center_queries=int(len(selected)),
        H_record_replay_max_absolute_error=error,folds=folds,
        no_optimized_state_created=True,energy_slope_not_variance_derivative=True,
        source_manifest=transitive_local_source_manifest((Path(__file__),),root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')),runtime=numerical_runtime_identity())
    atomic_json(output/'results.json',result);return result


def compare_condition_arm(plan: dict,arm: str,*,plan_sha256: str) -> dict:
    """Run one preregistered multi-step optimizer recipe from a common parent.

    Arms may deliberately differ in SR stage or learning rate. They are recipe
    comparisons, not single-factor or matched-actual-distance experiments.
    Architecture and the complete coherent physical amplitude are unchanged.
    """
    if canonical_sha256(plan) != plan_sha256:
        raise ValueError('registered condition plan changed')
    evidence=plan.get('pre_release_direction_evidence')
    if evidence is not None:
        path=ROOT/Path(evidence['path'])
        if file_sha256(path)!=evidence['sha256']:
            raise ValueError('fixed-parent direction evidence changed')
        probe=json.loads(path.read_text())
        if probe['status']!='complete' or not probe['no_optimized_state_created']:
            raise ValueError('direction evidence is not the registered completed probe')
        for fold in probe['folds']:
            for name in evidence['negative_held_directions']:
                d=fold['directions'][name]
                if (d['held']['energy_slope_per_site']>=0
                        or d['finite_step_overlap']['held']['sampled_fs_sine']>evidence['maximum_held_FS_sine']
                        or d['finite_step_overlap']['held']['reweighting_ess_fraction']<.95):
                    raise ValueError('direction evidence does not support the registered bounded trial')
    item=plan['arms'][arm]
    config_path=ROOT/Path(item['training_config'])
    if file_sha256(config_path) != item['training_config_sha256']:
        raise ValueError('registered arm training config changed')
    spec=json.loads(config_path.read_text()); directory=ROOT/Path(spec['output'])
    parent=plan['parent']; continuation=spec['training_resume_from']
    for name in ('directory','checkpoint_sha256','registration_sha256'):
        if continuation[name] != parent[name]:
            raise ValueError('condition arms do not share the frozen parent')
    expected_reset=plan.get('require_import_moment_reset',True)
    if continuation.get('reset_optimizer_moments',False) is not expected_reset:
        raise ValueError('condition import-moment policy differs from its registration')
    initial_optimizer_step=plan.get('initial_optimizer_step',0)
    if (item['midpoint_step'] <= parent['step']
            or item['endpoint_step']-parent['step'] != plan['updates_per_arm']):
        raise ValueError('invalid frozen condition endpoints')
    source=transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    identity=dict(plan_sha256=plan_sha256,arm=arm,source_manifest=source,
                  runtime=numerical_runtime_identity())
    directory.mkdir(parents=True,exist_ok=True)
    registration=directory/'condition_registration.json'
    if registration.exists():
        if json.loads(registration.read_text()) != identity:
            raise ValueError('condition source/runtime changed after registration')
    else: atomic_json(registration,identity)
    middle_path=directory/'condition_midpoint.pkl'
    current=(load_checkpoint(directory/'checkpoint.pkl')['metadata']['completed_steps']
             if (directory/'checkpoint.pkl').exists() else parent['step'])
    if current <= item['midpoint_step']:
        mid=train(spec,directory,item['midpoint_step'],plan['session_seconds_per_half'])
        if mid['completed_steps'] != item['midpoint_step']:
            raise RuntimeError('bounded condition paused before its midpoint; checkpoint preserved')
        saved=load_checkpoint(directory/'checkpoint.pkl')
        if not middle_path.exists():
            save_checkpoint(middle_path,saved['params'],optimizer_state=saved['optimizer_state'],
                            metadata=saved['metadata'])
    elif not middle_path.exists():
        raise ValueError('condition advanced without an archived midpoint')
    # This second call genuinely reloads parameters, optimizer, chains and RNG.
    # Small exact tests independently compare this split path to uninterrupted
    # updates, including the explicit once-only moment reset at import.
    middle=load_checkpoint(middle_path)
    middle_accepted=sum(bool(row['accepted']) for row in
        middle['metadata']['history'][parent['step']:item['midpoint_step']])
    if (int(middle['optimizer_state'][0]) != initial_optimizer_step+middle_accepted
            or middle['metadata']['completed_steps'] != item['midpoint_step']):
        raise ValueError('condition midpoint optimizer/state boundary is inconsistent')
    trained=train(spec,directory,item['endpoint_step'],plan['session_seconds_per_half'])
    if trained['completed_steps'] != item['endpoint_step']:
        raise RuntimeError('bounded condition paused before its endpoint; checkpoint preserved')
    final=load_checkpoint(directory/'checkpoint.pkl')
    final_accepted=sum(bool(row['accepted']) for row in trained['history'][parent['step']:])
    if int(final['optimizer_state'][0]) != initial_optimizer_step+final_accepted:
        raise ValueError('condition optimizer reset more than once or did not resume')
    audit_path=ROOT/Path(item['audit_config'])
    if file_sha256(audit_path) != item['audit_config_sha256']:
        raise ValueError('registered independent audit config changed')
    audit_spec=json.loads(audit_path.read_text()); audit_dir=ROOT/Path(audit_spec['output'])
    if (audit_spec['training_directory'] != spec['output']
            or audit_spec['endpoint_step'] != item['endpoint_step']):
        raise ValueError('independent audit is not the frozen condition endpoint')
    if (audit_dir/'audit.json').exists():
        measured=json.loads((audit_dir/'audit.json').read_text())
        if (measured['audit_config'] != audit_spec
                or measured['checkpoint_sha256'] != trained['checkpoint_sha256']):
            raise ValueError('existing condition audit changed')
    else: measured=audit(audit_spec,audit_dir)
    rows=trained['history'][parent['step']:]
    result=dict(status='complete',arm=arm,plan_sha256=plan_sha256,
        parent=parent,training_config_sha256=item['training_config_sha256'],
        checkpoint_sha256=trained['checkpoint_sha256'],
        completed_new_updates=len(rows),optimizer_steps=int(final['optimizer_state'][0]),
        checkpoint_resume=dict(midpoint_completed_step=middle['metadata']['completed_steps'],
            midpoint_optimizer_steps=int(middle['optimizer_state'][0]),
            final_optimizer_steps=int(final['optimizer_state'][0]),
            explicit_once_only_import_reset=expected_reset,
            initial_optimizer_step=initial_optimizer_step),
        new_active_training_seconds=trained['active_seconds']-plan['parent']['active_seconds'],
        accepted=sum(bool(row['accepted']) for row in rows),
        cumulative_sampled_FS_path=float(sum(row['FS_sine'] for row in rows)),
        median_sampled_FS_sine=float(np.median([row['FS_sine'] for row in rows])),
        median_step_seconds=float(np.median([row['seconds'] for row in rows])),
        median_relative_SR_residual=float(np.median([row['relative_SR_residual'] for row in rows])),
        new_training_full_H_calls=sum(row['full_H_calls'] for row in rows),
        audit=measured['statistics'],audit_config_sha256=item['audit_config_sha256'],
        independent_execution_replay=measured['independent_execution_replay'],
        recipe_comparison_not_single_factor=True,actual_FS_distances_not_assumed_equal=True,
        no_automatic_long_training=True)
    atomic_json(directory/'condition_result.json',result)
    return result


def compare(plan: dict,*,plan_sha256: str) -> dict:
    """Run fixed two-arm continuation and independent audit from a sealed state."""
    if canonical_sha256(plan) != plan_sha256: raise ValueError('registered radius plan changed')
    parent=ROOT/Path(plan['parent']['directory'])
    reg=json.loads((parent/'registration.json').read_text())
    checkpoint_path=parent/'checkpoint.pkl'; checkpoint=load_checkpoint(checkpoint_path)
    training=json.loads((parent/'training.json').read_text())
    if (reg['sha256'] != plan['parent']['registration_sha256']
            or file_sha256(checkpoint_path) != plan['parent']['checkpoint_sha256']
            or training['checkpoint_sha256'] != plan['parent']['checkpoint_sha256']
            or training['completed_steps'] != plan['parent']['step']
            or checkpoint['metadata']['completed_steps'] != plan['parent']['step']):
        raise ValueError('radius parent is not the registered completed state')
    if set(plan['arms']) != {'radius005','radius020'}: raise ValueError('expected two frozen radius arms')
    results={}
    for name in ('radius005','radius020'):
        item=plan['arms'][name]
        spec=copy.deepcopy(reg['config']); old_radius=spec['maximum_FS_sine']; radius=item['maximum_FS_sine']
        spec['maximum_FS_sine']=radius
        spec['output']=item['training_output']
        spec['interpretation']='One-step same-parent physical trust-radius comparison, not a long accuracy run.'
        spec['release']=dict(authorized_endpoint=plan['parent']['step']+1,
                             automatic_long_release=False,maximum_GPU_hours_for_pair=1)
        continuation=dict(directory=str(parent),checkpoint_sha256=plan['parent']['checkpoint_sha256'],
            registration_sha256=plan['parent']['registration_sha256'],
            source_workspace=plan['parent']['source_workspace'],same_projected_physical_state_proved=True)
        if radius != old_radius:
            continuation['optimizer_condition_change']={'maximum_FS_sine':dict(from_value=old_radius,to_value=radius)}
        spec['training_resume_from']=continuation
        config_path=ROOT/item['generated_training_config']
        if config_path.exists():
            if json.loads(config_path.read_text()) != spec: raise FileExistsError('generated arm config differs')
        else: atomic_json(config_path,spec)
        trained=train(spec,ROOT/spec['output'],plan['parent']['step']+1,plan['maximum_training_session_seconds'])
        if trained['completed_steps'] != plan['parent']['step']+1:
            raise RuntimeError('one-step arm paused before its frozen endpoint')
        audit_spec=dict(training_directory=spec['output'],output=item['audit_output'],
            endpoint_step=plan['parent']['step']+1,require_gpu=plan['require_gpu'],
            streams=plan['audit']['streams'],chains_per_stream=plan['audit']['chains_per_stream'],
            snapshots_per_stream=plan['audit']['snapshots_per_stream'],
            burn_proposals=plan['audit']['burn_proposals'],
            proposals_between_snapshots=plan['audit']['proposals_between_snapshots'],
            audit_seeds=plan['audit']['audit_seeds'],bootstrap_seed=plan['audit']['bootstrap_seed'],
            H_sample_chunk=plan['audit']['H_sample_chunk'])
        audit_path=ROOT/item['generated_audit_config']
        if audit_path.exists():
            if json.loads(audit_path.read_text()) != audit_spec: raise FileExistsError('generated audit differs')
        else: atomic_json(audit_path,audit_spec)
        audit_dir=ROOT/item['audit_output']
        if audit_dir.exists():
            measured=json.loads((audit_dir/'audit.json').read_text())
            if (measured['checkpoint_sha256'] != trained['checkpoint_sha256']
                    or measured['audit_config'] != audit_spec):
                raise ValueError('existing arm audit changed')
        else: measured=audit(audit_spec,audit_dir)
        results[name]=dict(training=trained,audit=measured['statistics'],
                           audit_checkpoint_sha256=measured['checkpoint_sha256'])
    a=results['radius005']; b=results['radius020']
    oldrow=a['training']['history'][-1]; newrow=b['training']['history'][-1]
    if (not np.isclose(oldrow['energy_before_update_per_site'],newrow['energy_before_update_per_site'],atol=1e-12,rtol=0)
            or oldrow['samples'] != newrow['samples']
            or oldrow['full_H_calls'] != newrow['full_H_calls']):
        raise ArithmeticError('arms did not use identical pre-update physical batch')
    e_delta=b['audit']['energy_per_site']-a['audit']['energy_per_site']
    v_delta=b['audit']['variance_per_site']-a['audit']['variance_per_site']
    result=dict(status='complete',plan_sha256=plan_sha256,parent_checkpoint_sha256=plan['parent']['checkpoint_sha256'],
        same_parent_sample_and_full_H_calls=True,arms=results,
        radius020_minus_radius005_energy_per_site=e_delta,
        radius020_minus_radius005_variance_per_site=v_delta,
        energy_standard_error_combined=float(np.hypot(a['audit']['energy_chain_SE'],b['audit']['energy_chain_SE'])),
        variance_standard_error_combined=float(np.hypot(a['audit']['variance_chain_SE'],b['audit']['variance_chain_SE'])),
        no_automatic_long_training=True,source_manifest=transitive_local_source_manifest((Path(__file__),),root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')),runtime=numerical_runtime_identity())
    atomic_json(ROOT/plan['comparison_output'],result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--arm',help='run one registered multi-step optimizer-condition arm')
    parser.add_argument('--diagnose-directions',action='store_true',help='run one fixed-parent archived-stream direction probe')
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    result=(diagnose_optimizer_directions(plan,plan_sha256=args.plan_sha256) if args.diagnose_directions
            else compare_condition_arm(plan,args.arm,plan_sha256=args.plan_sha256)
            if args.arm is not None else compare(plan,plan_sha256=args.plan_sha256))
    print(json.dumps(result,
                     default=str)[:2000],flush=True)
