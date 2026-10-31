"""Same-backbone soft-spin constraint and matrix-free S2 cost preflight.

The old VAP sources/seals stay immutable. Reuse its optimizer, exact-space
metrics, physical ViT, checkpoints and independent projection diagnostics.
P0 is never inside the constraint-arm objective or its deployed amplitudes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.krylov_anchored_full_vit import a1_orbit_hamiltonian
from su2_nqs.benchmarks.spin_sector import singlet_projector_action
from su2_nqs.benchmarks.spatial_vit_sampling import projected_log_function, make_full_heisenberg_ratios
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact, save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit
from su2_nqs.operators.spin_squared_ratios import make_spin_squared_ratios
from su2_nqs.vmc.exact_fs_adam import train_exact_fs_adam
from .spin import _prepare_case, _groups, _bonds
from .spin_full_vit_budget import prepare_objective
from .spin_full_vit_vap import state_metrics
from .spin_full_vit_optimization import load_parent

ROOT=Path(__file__).resolve().parents[3]
CONFIG=ROOT/"projects/clebsch_tree_nqs/configs/validation/spin_full_vit_constraint.json"
jax.config.update("jax_enable_x64",True)


def prepare_constraint(data: dict, model: FixedSzSpatialViTConfig, penalty: float) -> dict:
    """Exact-small objective without a spin projector; no diagonalization."""
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    raw=prepare_objective(data,model)
    h=a1_orbit_hamiltonian(data['h_rows'],data['h_columns'],data['h_values'],raw['a1'])
    coo=data['spin_squared'].tocoo()
    s2=a1_orbit_hamiltonian(coo.row,coo.col,coo.data,raw['a1'])
    return dict(raw=raw,h=h,s2=s2,objective=h+penalty*s2)


def seal(path: Path, result: dict) -> dict:
    """Seal derived results without altering ancestors."""
    result={**result}
    result['result_sha256']=canonical_sha256(result)
    atomic_json(path,result)
    return result


def registration(spec: dict) -> dict:
    """Seal training parents, comparison controls, frozen N36 probes and code."""
    files={}
    for trial in spec['parents']:
        for filename in ('results.json','parameters.json','parameters.npz'):
            path=ROOT/spec['parent_root']/trial/filename
            files[str(path.relative_to(ROOT))]=file_sha256(path)
        for arm in ('raw','vap'):
            path=ROOT/spec['reference_root']/f'{trial}_{arm}'/'results.json'
            files[str(path.relative_to(ROOT))]=file_sha256(path)
    path=ROOT/spec['reference_root']/'registration.json'
    files[str(path.relative_to(ROOT))]=file_sha256(path)
    prior_path=ROOT/spec['ratio_preflight']['n36_parent_config']
    files[str(prior_path.relative_to(ROOT))]=file_sha256(prior_path)
    prior=json.loads(prior_path.read_text())
    for geometry in ('triangular','square'):
        load_parent(prior,geometry)
        for filename in ('results.json','checkpoint.pkl'):
            path=ROOT/prior['parent_directory']/geometry/filename
            files[str(path.relative_to(ROOT))]=file_sha256(path)
    result=dict(config=spec,input_files=files,runtime=numerical_runtime_identity(),
        source_manifest=transitive_local_source_manifest((Path(__file__),),root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    result['registration_sha256']=canonical_sha256(result)
    return result


def verified(config_path: Path) -> tuple[dict,dict,Path]:
    """Reject changed inputs, sources or runtime on every invocation."""
    spec=json.loads(config_path.read_text()); output=ROOT/spec['output']
    record=json.loads((output/'registration.json').read_text())
    if record != registration(spec):
        raise ValueError('registration/source/input/runtime mismatch')
    return spec,record,output


def run(config_path: Path, trial: str, arm_id: str, *, resume: bool=False) -> dict:
    """Run one fixed constraint arm; pure H determines all reported energy wins."""
    spec,record,output=verified(config_path)
    if trial not in spec['parents']:
        raise ValueError('unregistered parent')
    arm=next(a for a in spec['arms'] if a['id']==arm_id)
    penalty=float(arm['penalty'])
    destination=output/f'{trial}_{arm_id}'
    if (destination/'results.json').exists(): raise FileExistsError(destination/'results.json')
    started=perf_counter()
    folder=ROOT/spec['parent_root']/trial
    parent=json.loads((folder/'results.json').read_text())
    model=FixedSzSpatialViTConfig(**parent['model_config'])
    if model.n_sites != 16: raise ValueError('this exhaustive training gate is N16 only')
    template=init_fixed_sz_spatial_vit(jax.random.PRNGKey(0),model)
    params=load_parameter_artifact(folder/'parameters.json',template,expected_identity=dict(trial=trial))
    if pytree_sha256(params)!=parent['final_params_sha256']: raise ValueError('parent mismatch')
    data=_prepare_case(parent['case'],'fixed_m0')
    problem=prepare_constraint(data,model,penalty)
    checks=problem['raw']['validate'](params)
    state_fn=problem['raw']['state_fn']
    initial_vector=np.asarray(state_fn(params))
    initial=state_metrics(initial_vector,problem,data)
    if abs(initial['energy_per_site']-parent['metrics']['energy_per_site'])>2e-11:
        raise RuntimeError('parent physical energy not replayed')
    # Independent physical-matrix objective and analytic autodiff audit.
    physical=problem['raw']['physical'](initial_vector)
    indices=np.argsort(-abs(physical)**2,kind='stable')[:spec['ratio_preflight']['n16_centers']]
    log_fn,_=projected_log_function(model,data['groups']['gamma_a1'])
    ratio=make_spin_squared_ratios(log_fn,n_sites=16)(params,jnp.asarray(data['basis'])[indices])
    expected_ratio=(data['spin_squared']@physical)[indices]/physical[indices]
    checks['packed_S2_ratio_scaled_error']=float(np.max(abs(np.asarray(ratio)-expected_ratio)/(1+abs(expected_ratio))))
    if checks['packed_S2_ratio_scaled_error']>1e-10:
        raise RuntimeError('N16 packed S2 ratio differs from physical sparse action')
    expected=float(np.vdot(physical,(data['hamiltonian']+penalty*data['spin_squared'])@physical).real)
    loss=lambda p:jnp.vdot(state_fn(p),jnp.asarray(problem['objective'])@state_fn(p)).real
    checks['penalized_objective_replay_error']=abs(float(loss(params))-expected)
    tangent=jax.tree_util.tree_map(lambda x:jnp.ones_like(x)*.01,params)
    derivative=float(jax.jvp(loss,(params,),(tangent,))[1])
    shifted=[jax.tree_util.tree_map(lambda p,t:p+sign*1e-4*t,params,tangent) for sign in (-1,1)]
    checks['directional_gradient_error']=abs(derivative-float((loss(shifted[1])-loss(shifted[0]))/2e-4))
    if checks['penalized_objective_replay_error']>1e-10 or checks['directional_gradient_error']>2e-7:
        raise RuntimeError(f'constraint exact validation failed {checks}')
    trajectory=(json.loads((destination/'diagnostics.json').read_text())
                if resume and (destination/'diagnostics.json').exists() else [])

    def callback(p,row):
        metrics=state_metrics(np.asarray(state_fn(p)),problem,data)
        trajectory.append(dict(step=row['step'],**metrics))
        atomic_json(destination/'diagnostics.json',trajectory)
        print(trial,arm_id,row['step'],'dE/N',metrics['energy_error_per_site'],
              'Var/N',metrics['variance_per_site'],'S2',metrics['spin_squared'],flush=True)

    print(trial,arm_id,'prepared',perf_counter()-started,flush=True)
    trained,training=train_exact_fs_adam(params,state_fn=state_fn,hamiltonian=problem['objective'],
        identity=dict(registration_sha256=record['registration_sha256'],trial=trial,arm=arm_id),
        checkpoint=destination/'checkpoint.pkl',resume=resume,callback=callback,**spec['optimization'])
    vector=np.asarray(state_fn(trained)); final=state_metrics(vector,problem,data,cuts=True)
    physical=problem['raw']['physical'](vector)
    projected=singlet_projector_action(physical,data['spin_squared'],maximum_spin=8)
    projected/=np.linalg.norm(projected)
    from su2_nqs.benchmarks.krylov_anchored_full_vit import a1_orbit_coordinates
    projected_vector=np.asarray(a1_orbit_coordinates(jnp.asarray(projected),problem['raw']['a1']))
    pav=state_metrics(projected_vector,problem,data,cuts=True)
    artifact=destination/'parameters.json'
    if artifact.exists() and resume:
        saved=load_parameter_artifact(artifact,template,expected_identity=dict(trial=trial,arm=arm_id))
        if pytree_sha256(saved)!=pytree_sha256(trained): raise ValueError('partial finalization mismatch')
    else:
        save_parameter_artifact(artifact,trained,identity=dict(trial=trial,arm=arm_id))
    saved=load_parameter_artifact(artifact,template,expected_identity=dict(trial=trial,arm=arm_id))
    checks['checkpoint_amplitude_replay_max']=float(np.max(abs(np.asarray(state_fn(saved))-vector)))
    temporary=destination/'states.pending.npz'
    np.savez_compressed(temporary,basis=np.asarray(data['basis'],dtype=np.int8),
        parent=problem['raw']['physical'](initial_vector),terminal=physical,postprojected=projected)
    temporary.replace(destination/'states.npz')
    objective_calls=training['full_H_calls']
    result=dict(trial=trial,arm=arm_id,penalty=penalty,case=parent['case'],seed=parent['seed'],
        parameter_count=parent['parameter_count'],parent_params_sha256=parent['final_params_sha256'],
        final_params_sha256=pytree_sha256(trained),registration_sha256=record['registration_sha256'],
        initial=initial,final=final,postprojection_diagnostic=pav,trajectory=trajectory,training=training,
        physical_final_energy_total=16*final['energy_per_site'],
        final_penalized_loss=16*final['energy_per_site']+penalty*final['spin_squared'],
        checks=checks,physical_states_sha256=file_sha256(destination/'states.npz'),
        objective_calls=objective_calls,logical_H_component_calls=objective_calls,
        logical_S2_component_calls=objective_calls,training_spin_projection_calls=0,
        n16_ratio_preflight_centers=len(indices),
        actual_exact_kernel='one preassembled H+lambda*S2 matvec/objective; reverse-mode cost in timing',
        history_energy_meaning='training.history.energy is total penalized loss, NOT physical H energy',
        exact_SU2=False,exact_A1=True,all_parameters_trained=True,wall_seconds=perf_counter()-started,
        hamiltonian_diagonalization_performed=False,remote_submitted=False)
    result=seal(destination/'results.json',result)
    print(trial,arm_id,'DONE',final['energy_error_per_site'],'wall',result['wall_seconds'],flush=True)
    return result


def ratio_preflight(config_path: Path) -> dict:
    """Frozen N36 pointwise action/cost test, not sampling or training."""
    spec,record,output=verified(config_path)
    path=output/'ratio_preflight.json'
    if path.exists(): raise FileExistsError(path)
    started=perf_counter(); entries=[]; arrays={}
    prior=json.loads((ROOT/spec['ratio_preflight']['n36_parent_config']).read_text())
    for geometry in ('triangular','square'):
        checkpoint,result=load_parent(prior,geometry)
        params=checkpoint['params']['model']
        model=FixedSzSpatialViTConfig(**result['model_config'])
        bits=jnp.asarray(checkpoint['params']['chains'][:spec['ratio_preflight']['n36_centers']])
        log_fn,images=projected_log_function(model,_groups(geometry,6,6)['gamma_a1'])
        s2=make_spin_squared_ratios(log_fn,n_sites=36)
        all_pairs=tuple((i,j,2.) for i in range(36) for j in range(i+1,36))
        oracle=make_full_heisenberg_ratios(log_fn,all_pairs)
        h=make_full_heisenberg_ratios(log_fn,_bonds(geometry,6,6,.125 if geometry=='triangular' else .5))
        timings={}; values={}
        # Separately measure lowering/compilation and warm execution; GPU-ready
        # JAX is tested on CPU here, so no accelerator speedup is inferred.
        for label,kernel in (('H',h),('S2_packed',s2),('S2_all_pairs',oracle)):
            begin=perf_counter(); executable=kernel.lower(params,bits).compile()
            compilation=perf_counter()-begin
            first=np.asarray(executable(params,bits))
            elapsed=[]
            for _ in range(spec['ratio_preflight']['timing_repeats']):
                tick=perf_counter(); measured=np.asarray(executable(params,bits))
                elapsed.append(perf_counter()-tick)
                np.testing.assert_allclose(measured,first,rtol=0,atol=0)
            values[label]=first;timings[label]=dict(compile_seconds=compilation,execution_seconds=elapsed)
        expected=.75*36+values['S2_all_pairs']
        difference=np.max(abs(values['S2_packed']-expected))
        scaled=np.max(abs(values['S2_packed']-expected)/(1+abs(expected)))
        if not np.isfinite(values['S2_packed']).all() or scaled>1e-10:
            raise RuntimeError('N36 packed S2 oracle mismatch')
        entries.append(dict(geometry=geometry,centers=len(bits),images=images,
            parent_params_sha256=pytree_sha256(params),pointwise_absolute_error=float(difference),
            pointwise_scaled_error=float(scaled),timings=timings,
            packed_exchanges_per_center=324,all_pairs_per_center=630,
            interpretation='frozen checkpoint probes, NOT a Born expectation or training benefit'))
        arrays.update({geometry+'_states':np.asarray(bits),geometry+'_H':values['H'],
                       geometry+'_S2_packed':values['S2_packed'],geometry+'_S2_oracle':expected})
        print('N36',geometry,'S2 scaled error',scaled,'timings',timings,flush=True)
    np.savez_compressed(output/'ratio_preflight.npz',**arrays)
    return seal(path,dict(registration_sha256=record['registration_sha256'],entries=entries,
        raw_sha256=file_sha256(output/'ratio_preflight.npz'),wall_seconds=perf_counter()-started,
        n36_training_updates=0,new_samples=0,remote_submitted=False))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--trial');parser.add_argument('--arm')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--ratio-preflight',action='store_true')
    args=parser.parse_args()
    if args.register:
        spec=json.loads(args.config.read_text());path=ROOT/spec['output']/'registration.json'
        if path.exists(): raise FileExistsError(path)
        record=registration(spec);atomic_json(path,record);print(record['registration_sha256'])
    elif args.ratio_preflight:
        ratio_preflight(args.config)
    else:
        run(args.config,args.trial,args.arm,resume=args.resume)


if __name__=='__main__': main()
