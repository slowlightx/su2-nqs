"""Serial, bounded repairs of the complete Schur-space model.

Replay stages never train on an audit or alter its frozen endpoint. Numerical
source snapshots and output directories are distinct from historical runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,numerical_runtime_identity
from su2_nqs.vmc.checkpoint import load_checkpoint,save_checkpoint
from su2_nqs.vmc.many_cell_tree import compile_many_cell_device_hamiltonian,evaluate_many_cell_device_local_energy
from .spin_reduced_vit_viability import model_config,neural_bonds
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds


def joint_log_state_matching_loss(candidate: jax.Array,reference: jax.Array) -> jax.Array:
    """Joint normalized-Born KL and gauge-free known-neural-phase regression.

    Inputs are finite complete-support normalized AR logs. A smooth phase
    lift supplied by an old neural network is legitimate for initializer
    transfer; it is NOT an arbitrarily assigned lift of unknown ED signs.
    """
    if candidate.ndim!=1 or candidate.shape!=reference.shape:
        raise ValueError('same complete-support log vectors required')
    p=jnp.exp(2*reference.real);p=p/jnp.sum(p)
    difference=candidate.imag-reference.imag
    difference=difference-jnp.sum(p*difference)
    return 2*jnp.sum(p*(reference.real-candidate.real))+jnp.sum(p*difference**2)


def hermitian_rayleigh_surrogate(c: jax.Array,hc: jax.Array) -> jax.Array:
    """Exact Rayleigh value/first derivative for a fixed Hermitian H.

    The explicit factor TWO retains BOTH coherent energy terms. No derivative
    of the fixed H gather is needed. This is a first-gradient objective, not
    a Hessian surrogate: use the actual H/JVP for second derivatives.
    """
    norm=jnp.vdot(c,c).real
    e=jax.lax.stop_gradient(jnp.vdot(c,hc).real/norm)
    return (2*jnp.vdot(c,jax.lax.stop_gradient(hc)).real-e*norm)/jax.lax.stop_gradient(norm)


def coherent_a1_vap_probe(spec: dict, output: Path) -> dict:
    """Two-update projection/gradient/Born/checkpoint check, NOT an N16 accuracy run.

    Reuse the already validated complete N16 CG conversion. No new ED, no
    N36 projection claim, no comparison with ordinary-ViT production. This
    factors physical orbit sums exactly; it does not average probabilities.
    """
    from su2_nqs.benchmarks.clebsch_tree_exact import (
        graph_hamiltonian_coo, physical_permutation_index_table,
    )
    from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
    from su2_nqs.models.many_cell_schur_vit import many_cell_schur_log_amplitudes
    from su2_nqs.vmc.adam import _manual_adam_init, _manual_adam_update
    from su2_nqs.vmc.exact_reduced_projector import compile_exact_a1_orbit_projector
    from scipy import sparse
    if spec['steps'] != 2:
        raise ValueError('this correctness probe has exactly two updates, not an energy-tail budget')
    for name in ('parent', 'conversion'):
        if file_sha256(Path(spec[name])) != spec[name+'_sha256']:
            raise ValueError('changed coherent projection input')
    loaded = load_checkpoint(spec['parent'])
    params = jax.tree_util.tree_map(jnp.asarray, loaded['params'])
    config = model_config(loaded['metadata']['config'])
    data = np.load(spec['conversion']); u = data['isometry']; bits = data['bits']
    labels = data['labels']; x = jnp.asarray(labels)
    start = time.perf_counter()
    group = triangular_hnf_gamma_a1_group(((4,0),(0,4)))
    plan = compile_exact_a1_orbit_projector(labels, u, bits, group)
    plan_seconds = time.perf_counter()-start
    row,col,val,basis = graph_hamiltonian_coo(16,0,
        triangular_j1j2_bonds(4,4,j2=.125,periodic=True))
    if not np.array_equal(bits,basis):
        raise ValueError('CG conversion and full-H physical basis orders disagree')
    # Exact-small physical row gathers, not repeated-index atomic scatters.
    # This prevents a known GPU roundoff/replay problem; it is not the N36
    # reduced-space action or a new claimed accelerator optimization.
    h = sparse.coo_matrix((val,(row,col)),shape=(len(bits),len(bits))).tocsr()
    h.sort_indices(); width=int(np.diff(h.indptr).max())
    neighbors=np.zeros((len(bits),width),np.int32)
    weights=np.zeros((len(bits),width),np.float64)
    for i in range(len(bits)):
        start,stop=h.indptr[i:i+2]; count=stop-start
        neighbors[i,:count]=h.indices[start:stop]; weights[i,:count]=h.data[start:stop]
    uj = jnp.asarray(u); neighbors=jnp.asarray(neighbors); weights=jnp.asarray(weights)
    indices = physical_permutation_index_table(bits,group)
    def action(c):
        physical = uj@c
        return uj.conj().T@jnp.sum(weights*physical[neighbors],axis=1)
    def vector(p):
        native = jnp.exp(many_cell_schur_log_amplitudes(p,x,config))
        return plan.apply(native)
    def energy(p):
        c = vector(p)
        return hermitian_rayleigh_surrogate(c,action(c))
    def direct_energy(p):
        c=vector(p)
        return jnp.vdot(c,action(c)).real/jnp.vdot(c,c).real
    direct=jax.jit(jax.value_and_grad(direct_energy))
    direct_first=direct(params);direct_second=direct(jax.tree_util.tree_map(lambda a:jnp.array(np.asarray(a)),params))
    direct_replay=max(float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(
        jax.tree_util.tree_leaves(direct_first),jax.tree_util.tree_leaves(direct_second),strict=True))
    actual=jax.value_and_grad(energy)(params)
    gradient_error=max(float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(
        jax.tree_util.tree_leaves(direct_first),jax.tree_util.tree_leaves(actual),strict=True))
    if gradient_error>1e-10:
        raise RuntimeError(f'analytic Hermitian gradient disagrees with full AD: {gradient_error}')
    @jax.jit
    def update(p,state):
        value,grad = jax.value_and_grad(energy)(p)
        new,state = _manual_adam_update(p,grad,state,spec['learning_rate'])
        return new,state,value
    def describe(p):
        c = np.asarray(vector(p)); norm = float(np.vdot(c,c).real)
        if not np.isfinite(c).all() or norm <= 1e-24:
            raise FloatingPointError('unresolved projected endpoint')
        c = c/np.sqrt(norm); hc = np.asarray(action(jnp.asarray(c)))
        e = float(np.vdot(c,hc).real); physical = u@c
        return dict(energy_per_site=e/16,
            variance_per_site=float(np.vdot(hc-e*c,hc-e*c).real/16),
            projection_norm_squared=norm,
            full_A1_physical_max_error=float(np.max(abs(physical[indices]-physical))),
            projection_idempotence_error=float(np.max(abs(np.asarray(plan.apply(c))-c))))
    initial = describe(params); state = _manual_adam_init(params)
    start = time.perf_counter(); compiled = update.lower(params,state).compile()
    compile_seconds = time.perf_counter()-start
    start = time.perf_counter(); history=[]
    for step in range(1,3):
        params,state,value = compiled(params,state); jax.block_until_ready(value)
        record=describe(params); history.append(dict(step=step,**record))
        if record['full_A1_physical_max_error'] > 1e-10:
            raise RuntimeError('training left coherent spatial sector')
    updates_seconds = time.perf_counter()-start
    rng = jax.random.PRNGKey(spec['seed']); rng,draw_key=jax.random.split(rng)
    c = vector(params); indices_drawn = plan.sample_indices(draw_key,c,spec['audit_samples'])
    c = c/jnp.linalg.norm(c); hc=action(c); local=np.asarray(hc[indices_drawn]/c[indices_drawn])
    e=float(jnp.vdot(c,hc).real); probabilities=np.asarray(abs(c)**2)
    frequency=np.bincount(np.asarray(indices_drawn),minlength=len(labels))/spec['audit_samples']
    checkpoint=output/'step0002.pkl'
    save_checkpoint(checkpoint,jax.tree_util.tree_map(np.asarray,params),
        optimizer_state=jax.tree_util.tree_map(np.asarray,state),
        metadata=dict(step=2,rng=np.asarray(rng),config=loaded['metadata']['config'],
            exact_SU2=True,full_A1=True,projection='exact-small physical orbit factor',
            born_sampler='coherent projected categorical',optimizer_new_at_stage_start=True))
    restored=load_checkpoint(checkpoint)
    rp=jax.tree_util.tree_map(jnp.asarray,restored['params'])
    rs=jax.tree_util.tree_map(jnp.asarray,restored['optimizer_state'])
    first=compiled(params,state); second=compiled(rp,rs);jax.block_until_ready(first)
    replay=max(float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(
        jax.tree_util.tree_leaves(first),jax.tree_util.tree_leaves(second),strict=True))
    _,next_key=jax.random.split(rng);_,restored_key=jax.random.split(jnp.asarray(restored['metadata']['rng']))
    next_draw=plan.sample_indices(next_key,vector(first[0]),64)
    replay_draw=plan.sample_indices(restored_key,vector(second[0]),64)
    if replay!=0 or not np.array_equal(next_draw,replay_draw):
        raise RuntimeError(f'projected continuation mismatch: update_max_error={replay}; samples_equal={np.array_equal(next_draw,replay_draw)}')
    np.savez_compressed(output/'audit.npz',indices=np.asarray(indices_drawn),
        labels=labels[np.asarray(indices_drawn)],local_energies=local,probabilities=probabilities)
    return dict(status='complete',stage='coherent_a1',exact_SU2=True,full_A1=True,
        n=16,group_order=len(group),orbit_count=len(plan.orbit_sizes),
        plan_shape=list(plan.factor.shape),plan_seconds=plan_seconds,
        gram_error=plan.gram_error,closure_error=plan.closure_error,
        compile_seconds=compile_seconds,two_updates_seconds=updates_seconds,
        initial=initial,endpoint=history[-1],history=history,
        audit=dict(independent_samples=spec['audit_samples'],
            energy_per_site=float(local.real.mean()/16),
            iid_energy_standard_error=float(local.real.std(ddof=1)/np.sqrt(len(local))/16),
            residual_second_moment_per_site=float(np.mean(abs(local-e)**2)/16),
            maximum_frequency_error=float(np.max(abs(frequency-probabilities)))),
        next_update_replay_max_error=replay,next_sample_replay_exact=True,
        checkpoint_sha256=file_sha256(checkpoint),
        direct_full_H_AD_replay_max_error=direct_replay,
        analytic_dual_coherent_gradient_max_error=gradient_error,
        new_ED=0,new_N36_projected_calls=0,N36_full_A1_available=False,
        N16_correctness_not_a_variational_win=True,
        limitation='Enumerates exact-small physical support; cannot release N36 projected production.')


def migrate_saved_neural_state(spec: dict,output: Path) -> dict:
    """Repair state migration, not a small-Hilbert-space energy-tail search.

    The target is the saved OLD NEURAL state, with its own smooth phase lift,
    not ED signs. Fixed full-support L-BFGS learns magnitudes/phase jointly.
    This initializer preserves parameters at exit, not a resumable SciPy
    optimizer claim. Later VMC starts a newly registered optimizer.
    """
    from scipy.optimize import minimize
    from jax.flatten_util import ravel_pytree
    from su2_nqs.models.many_cell_schur_vit import many_cell_schur_log_amplitudes
    for name in ('parent','old_neural','conversion'):
        if file_sha256(Path(spec[name]))!=spec[name+'_sha256']:raise ValueError('changed migration input')
    old=load_checkpoint(spec['old_neural'])['params'];params=load_checkpoint(spec['parent'])['params']
    params=jax.tree_util.tree_map(jnp.asarray,params)
    data=np.load(spec['conversion']);labels=jnp.asarray(data['labels'])
    legacy=model_config(spec['model_settings']);joint=model_config({**spec['model_settings'],'conditional_mode':'aligned_joint'})
    target_logs=many_cell_schur_log_amplitudes(old,labels,legacy)
    target=jnp.exp(target_logs)
    def objective(p):
        logs=many_cell_schur_log_amplitudes(p,labels,joint)
        return joint_log_state_matching_loss(logs,target_logs)
    def describe(p):
        c=jnp.exp(many_cell_schur_log_amplitudes(p,labels,joint))
        return dict(old_neural_fidelity=float(abs(jnp.vdot(target,c))**2),
            saved_physics_teacher_fidelity=float(abs(jnp.vdot(jnp.asarray(data['reference']),c))**2),
            native_norm_error=float(abs(jnp.vdot(c,c).real-1)),objective=float(objective(p)))
    flat,unravel=ravel_pytree(params);fg=jax.jit(jax.value_and_grad(lambda x:objective(unravel(x))))
    compile_start=time.perf_counter();jax.block_until_ready(fg(flat));compile_seconds=time.perf_counter()-compile_start
    start=time.perf_counter();history=[];evaluations=0
    def fun(x):
        nonlocal evaluations
        if time.perf_counter()-start>spec['wall_seconds']:raise TimeoutError('fixed migration budget')
        value,gradient=fg(jnp.asarray(x));evaluations+=1
        return float(value),np.asarray(gradient,dtype=float)
    initial=describe(params)
    fit=minimize(fun,np.asarray(flat),jac=True,method='L-BFGS-B',
        callback=lambda x:history.append(float(fun(x)[0])),
        options=dict(maxiter=spec['iterations'],maxfun=1500,maxls=30,ftol=1e-12,gtol=1e-6))
    trained=unravel(jnp.asarray(fit.x));endpoint=describe(trained)
    save_checkpoint(output/'initializer.pkl',jax.tree_util.tree_map(np.asarray,trained),
        optimizer_state=dict(algorithm='L-BFGS-B initializer; new VMC optimizer required',
            iterations=int(fit.nit),inverse_hessian_s=np.asarray(fit.hess_inv.sk),
            inverse_hessian_y=np.asarray(fit.hess_inv.yk)),
        metadata=dict(config={**spec['model_settings'],'conditional_mode':'aligned_joint'},
            optimizer_resume_supported=False,rng=np.asarray(jax.random.PRNGKey(spec['seed'])),
            exact_SU2=True,full_A1=False,teacher_is_saved_old_neural_state=True))
    return dict(status='complete',stage='migration',initial=initial,endpoint=endpoint,
        iterations=int(fit.nit),function_evaluations=evaluations,gradient_norm=float(np.linalg.norm(fit.jac)),
        optimizer_success=bool(fit.success),optimizer_message=str(fit.message),
        compile_seconds=compile_seconds,optimization_seconds=time.perf_counter()-start,
        source_old_neural_sha256=spec['old_neural_sha256'],
        checkpoint_sha256=file_sha256(output/'initializer.pkl'),history=history,
        no_ED=True,no_N36_energy_claim=True,full_A1=False,
        migration_passed=bool(endpoint['old_neural_fidelity']>=.95))


def parameterization_probe(spec: dict,output: Path) -> dict:
    """Joint signed teacher fit: representation/optimization, NOT new physics.

    Reuse the saved N16 wavefunction and optimized old backbone. All1430
    coefficients are training data. No new ED or small-system accuracy target.
    """
    from dataclasses import asdict
    from scipy import sparse
    from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis,graph_hamiltonian_coo
    from su2_nqs.benchmarks.fidelity_factorization import fidelity_factors
    from su2_nqs.models.many_cell_schur_vit import init_many_cell_schur_vit,many_cell_schur_log_amplitudes
    from su2_nqs.su2.reduced_physical_overlap import tree_overlaps_on_bits
    from su2_nqs.vmc.adam import _manual_adam_init,_manual_adam_update
    from .spin_reduced_vit_viability import physical_reflection
    from .spin_reduced_vit_diagnosis import state_readout
    prior=Path(spec['padding_gate'])
    if file_sha256(prior)!=spec['padding_gate_sha256']:raise ValueError('changed padding gate')
    gate=json.loads(prior.read_text())
    if gate['warm_speedup']<=1 or gate['maximum_local_energy_difference']>1e-9:
        raise ValueError('serial H-throughput gate not passed')
    for key in ('checkpoint','teacher'):
        if file_sha256(Path(spec[key]))!=spec[key+'_sha256']:raise ValueError('changed '+key)
    loaded=load_checkpoint(spec['checkpoint']);old=jax.tree_util.tree_map(jnp.asarray,loaded['params'])
    settings=spec.get('model_settings') or loaded['metadata'].get('config',loaded['metadata'].get('model_config'))
    raw=np.load(spec['teacher']);labels=raw['labels'];bits=np.asarray(fixed_m_spin_basis(16,0),np.int32)
    config=model_config({**settings,'lx':4,'ly':4})
    order=np.array(physical_reflection(4,4))[np.asarray(config.layout.site_order)]
    conversion_start=time.perf_counter()
    u=tree_overlaps_on_bits(config.layout.tree,labels,bits,physical_leaf_order=order)
    reference=u.T@raw['reconstructed_physical_state']
    error=float(np.max(abs(u@reference-raw['reconstructed_physical_state'])))
    if error>2e-12 or abs(np.vdot(reference,reference).real-1)>2e-12:
        raise RuntimeError('saved teacher and physical frame disagree')
    row,col,val,_=graph_hamiltonian_coo(16,0,triangular_j1j2_bonds(4,4,j2=.125,periodic=True))
    h=sparse.coo_matrix((val,(row,col)),shape=(len(bits),len(bits))).tocsr()
    target=jnp.asarray(reference);probability=abs(target)**2
    entropy=-jnp.sum(jnp.where(probability>0,probability*jnp.log(jnp.where(probability>0,probability,1.)),0.))
    np.savez_compressed(output/'conversion.npz',isometry=u,bits=bits,labels=labels,reference=reference)
    result=dict(status='running',stage='parameterization',new_ED=0,new_N36_training=0,
        all_teacher_rows_are_training_data=True,teacher_reconstruction_error=error,
        conversion_seconds=time.perf_counter()-conversion_start,
        fixed_steps_per_arm=spec['steps'],arms={})
    for mode in ('legacy','aligned_joint'):
        cfg=model_config({**settings,'lx':4,'ly':4,'conditional_mode':mode})
        params=old
        if mode!='legacy':
            template=init_many_cell_schur_vit(jax.random.PRNGKey(spec['seed']),cfg)
            params={**old,'logit_feature':template['logit_feature']}
        x=jnp.asarray(labels)
        def loss(p):
            logs=many_cell_schur_log_amplitudes(p,x,cfg);c=jnp.exp(logs)
            return -2*jnp.sum(probability*logs.real)-entropy-jnp.log(jnp.maximum(abs(jnp.vdot(target,c))**2,1e-30))
        @jax.jit
        def update(p,state):
            value,gradient=jax.value_and_grad(loss)(p)
            new,state=_manual_adam_update(p,gradient,state,spec['learning_rate'])
            return new,state,value
        def describe(p):
            c=np.exp(np.asarray(many_cell_schur_log_amplitudes(p,x,cfg)))
            g=jax.grad(loss)(p)
            return dict(fidelity=asdict(fidelity_factors(reference,c)),
                norm_error=float(abs(np.vdot(c,c).real-1)),
                gradient_norm=float(np.sqrt(sum(np.sum(np.asarray(t)**2) for t in jax.tree_util.tree_leaves(g)))),
                physical=state_readout(c,labels,cfg,u,h))
        initial=describe(params);state=_manual_adam_init(params)
        compilation=time.perf_counter();compiled=update.lower(params,state).compile()
        compile_seconds=time.perf_counter()-compilation;start=time.perf_counter();history=[]
        for step in range(1,spec['steps']+1):
            params,state,value=compiled(params,state);jax.block_until_ready(value)
            if not np.isfinite(float(value)):raise FloatingPointError('nonfinite joint loss')
            if step%50==0:history.append(dict(step=step,loss=float(value)))
        duration=time.perf_counter()-start
        checkpoint=output/(mode+'.pkl');key=np.asarray(jax.random.PRNGKey(spec['seed']+1))
        save_checkpoint(checkpoint,jax.tree_util.tree_map(np.asarray,params),
            optimizer_state=jax.tree_util.tree_map(np.asarray,state),
            metadata=dict(step=spec['steps'],rng=key,config={**settings,'lx':4,'ly':4,'conditional_mode':mode},
                deterministic_teacher_fit=True,exact_SU2=True,full_A1=False))
        restored=load_checkpoint(checkpoint);rp=jax.tree_util.tree_map(jnp.asarray,restored['params'])
        rs=jax.tree_util.tree_map(jnp.asarray,restored['optimizer_state'])
        first=compiled(params,state);second=compiled(rp,rs);jax.block_until_ready(first)
        replay=max(float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(
            jax.tree_util.tree_leaves(first),jax.tree_util.tree_leaves(second),strict=True))
        if replay!=0 or not np.array_equal(restored['metadata']['rng'],key):raise RuntimeError('checkpoint replay failed')
        result['arms'][mode]=dict(initial=initial,endpoint=describe(params),
            parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(params)),
            training_seconds=duration,compile_seconds=compile_seconds,history=history,
            checkpoint_sha256=file_sha256(checkpoint),next_update_replay_max_error=replay)
        atomic_json(output/'progress.json',result)
        print(json.dumps(dict(arm=mode,endpoint=result['arms'][mode]['endpoint'])),flush=True)
    result['status']='complete';result['no_variational_or_scaling_win_claim']=True
    return result


def padding_replay(spec: dict,output: Path) -> dict:
    """Compare identical full-H amplitudes on cached rows, no new Born draws."""
    parent=Path(spec['checkpoint']);rows=Path(spec['cached_rows'])
    for path,expected in ((parent,spec['checkpoint_sha256']),(rows,spec['cached_rows_sha256'])):
        if file_sha256(path)!=expected: raise ValueError('changed frozen input')
    loaded=load_checkpoint(parent)
    settings=loaded['metadata'].get('config',loaded['metadata'].get('model_config'))
    config=model_config(settings)
    params=jax.tree_util.tree_map(jnp.asarray,loaded['params'])
    labels=np.load(rows)['labels'][:spec['samples']]
    bonds=neural_bonds(triangular_j1j2_bonds(6,6,j2=.125,periodic=True),6,6)
    timings={};energies={};ledger={}
    before=pytree_sha256(params)
    for mode in spec.get('modes',['padded_vmap','active_map']):
        kernel=compile_many_cell_device_hamiltonian(config,bonds,
            connected_capacity=spec['capacity'],amplitude_tile_size=spec['tile'],
            skip_padded_amplitudes=mode=='active_map',bucket_padded_amplitudes=mode=='bucketed_batch')
        records=[]
        for repeat in range(spec['repeats']+1):
            start=time.perf_counter();parts=[];calls=0;raw=0;width=[]
            for offset in range(0,len(labels),spec['batch']):
                result=evaluate_many_cell_device_local_energy(kernel,params,labels[offset:offset+spec['batch']])
                parts.extend(result.local_energies);calls+=result.evaluated_connected_amplitudes
                raw+=result.full_h_term_calls;width.extend(result.unique_widths.tolist())
            records.append(time.perf_counter()-start)
        energies[mode]=np.asarray(parts)
        timings[mode]=dict(first_including_compile=records[0],warm_seconds=records[1:])
        ledger[mode]=dict(evaluated_connected_amplitudes=calls,
            unique_connections=int(sum(width)),maximum_width=max(width),
            full_H_calls=len(labels),bond_term_calls=raw)
    challenger=spec.get('challenger','active_map')
    error=float(np.max(abs(energies['padded_vmap']-energies[challenger])))
    if error>1e-9: raise RuntimeError('active full-H action differs from reference')
    if before!=pytree_sha256(params): raise RuntimeError('replay mutated parent')
    return dict(status='complete',stage='padding',exact_SU2=True,full_A1=False,
        samples=len(labels),new_Born_samples=0,new_updates=0,
        input_params_sha256=before,maximum_local_energy_difference=error,
        timing=timings,work=ledger,
        warm_speedup=float(np.median(timings['padded_vmap']['warm_seconds'])/
            np.median(timings[challenger]['warm_seconds'])),
        redundant_amplitude_reduction=ledger['padded_vmap']['evaluated_connected_amplitudes']/
            ledger[challenger]['evaluated_connected_amplitudes'],
        unchanged_state_and_overflow_checks=True,
        not_a_new_energy_or_variational_claim=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();spec=json.loads(args.config.read_text())
    output=Path(spec['output'])
    if output.exists(): raise FileExistsError('use a new output directory')
    output.mkdir(parents=True)
    registration=dict(spec=spec,config_sha256=file_sha256(args.config),
        source_sha256=file_sha256(Path(__file__)),runtime=numerical_runtime_identity())
    atomic_json(output/'registration.json',registration)
    if spec.get('require_gpu') and jax.default_backend()!='gpu': raise RuntimeError('GPU required')
    if spec['stage']=='padding':result=padding_replay(spec,output)
    elif spec['stage']=='parameterization':result=parameterization_probe(spec,output)
    elif spec['stage']=='migration':result=migrate_saved_neural_state(spec,output)
    elif spec['stage']=='coherent_a1':result=coherent_a1_vap_probe(spec,output)
    else:raise ValueError('unimplemented stage')
    result['registration_sha256']=file_sha256(output/'registration.json')
    atomic_json(output/'results.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
