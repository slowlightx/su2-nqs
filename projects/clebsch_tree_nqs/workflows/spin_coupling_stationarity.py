"""Read-only N16 stationarity probes from sealed coupling-transfer endpoints.

Five full parameter energy gradients define an explicitly limited tangent
span. We also report its training-J-only three-direction subspace. Fixed
rho=.001 probes are evaluated but NEVER committed as trained checkpoints.
All Hamiltonians are read from the sealed 153-dimensional operator file.
"""
from __future__ import annotations
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.benchmarks.krylov_anchored_full_vit import A1OrbitReduction,a1_orbit_reduction,a1_orbit_coordinates
from su2_nqs.benchmarks.coupling_conditioning_capacity import affine_bias_rank_bound,conditioning_tangent_report
from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256,numerical_runtime_identity,transitive_local_source_manifest
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig,init_fixed_sz_spatial_vit,fixed_sz_spatial_vit_amplitudes
from su2_nqs.models.coupling_conditioned_spatial_vit import wrap_spatial_vit_parameters,effective_parameters
from su2_nqs.vmc.matched_state_step import matched_state_step
from su2_nqs.vmc.tangent_residual_diagnostic import project_residual_on_tangents,gradient_conflict_matrix,projector_secant_alignment
from .spin_overlap_attention_cost import _save_npz

jax.config.update('jax_enable_x64',True)
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'projects/clebsch_tree_nqs/results/spin_coupling_transfer_v1'
OUTPUT=ROOT/'projects/clebsch_tree_nqs/results/spin_coupling_stationarity_v1'
PROTOCOL=dict(task='spin_coupling_stationarity',couplings=[.2,.35,.5,.65,.8],training_indices=[0,2,4],
    gradient_probe_count=5,subspaces=['training_J_only','all_five_J'],all_endpoints=True,
    target_fs=.001,max_normalized_alpha=.002,bisection_steps=16,relative_rank_cutoff=1e-10,
    conditioning_bias_scope='two shared singlet-VAP endpoints at all five J, 16 effective-bias directions',
    projector_secant_scope='shared vs pointwise singlet states, central three J, full coarse span .3',
    wall_seconds_cap=180,training_updates_committed=0,new_ED_calls=0,new_samples=0,
    new_H_constructions=0,probe_selection='all fixed endpoints and both subspaces; no energy selection',
    interpretation='posthoc stationarity diagnostic; finite tangent spans give lower bounds, not full-manifold certification')


def _register():
    result=json.loads((SOURCE/'results.json').read_text());unsigned=dict(result);seal=unsigned.pop('result_sha256')
    if result['status']!='complete' or canonical_sha256(unsigned)!=seal:raise ValueError('invalid parent result')
    if file_sha256(SOURCE/'registration.json')!=result['registration_sha256']:raise ValueError('parent registration changed')
    oldreg=json.loads((SOURCE/'registration.json').read_text())
    for rel,sha in oldreg['sources'].items():
        if file_sha256(ROOT/rel)!=sha:raise ValueError('sealed source changed: '+rel)
    for rel,sha in result['artifacts'].items():
        if file_sha256(SOURCE/rel)!=sha:raise ValueError('sealed artifact changed: '+rel)
    inputs={str((SOURCE/rel).relative_to(ROOT)):sha for rel,sha in result['artifacts'].items()}
    for name in ('results.json','registration.json'):inputs[str((SOURCE/name).relative_to(ROOT))]=file_sha256(SOURCE/name)
    registration=dict(protocol=PROTOCOL,inputs=inputs,runtime=numerical_runtime_identity(),
        sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    registration['registration_sha256']=canonical_sha256(registration)
    return result,registration


def _saved_operator_state(cfg,operators,unravel):
    """Same ViT and exact spatial orbit map, without constructing any H."""
    basis=operators['basis'];packed=basis@(1<<np.arange(15,-1,-1));lookup=np.full(1<<16,-1,int);lookup[packed]=np.arange(len(basis))
    grids=basis.reshape(-1,cfg.rows,cfg.columns);images=[]
    for dr in range(0,cfg.rows,cfg.patch_rows):
        for dc in range(0,cfg.columns,cfg.patch_columns):
            moved=np.roll(grids,(dr,dc),(1,2)).reshape(-1,16)
            images.append(lookup[moved@(1<<np.arange(15,-1,-1))])
    coarse=a1_orbit_reduction(np.asarray(images));_,representatives=np.unique(coarse.orbit_id,return_index=True)
    a1=A1OrbitReduction(operators['orbit_id'],operators['orbit_sizes']);reduced=jnp.asarray(basis[representatives])
    p0=jnp.asarray(operators['p0']);coarse_ids=jnp.asarray(coarse.orbit_id)
    def state(x,j,project):
        parameters=effective_parameters(unravel(x),j,center=.5)
        raw=fixed_sz_spatial_vit_amplitudes(parameters,reduced,cfg)
        v=a1_orbit_coordinates(raw[coarse_ids],a1);v/=jnp.linalg.norm(v)
        v=jax.lax.cond(project,lambda y:p0@y,lambda y:y,v)
        return v/jnp.linalg.norm(v)
    return state


def run():
    source,registration=_register()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):raise FileExistsError('immutable stationarity attempt exists')
    OUTPUT.mkdir(parents=True,exist_ok=True);atomic_json(OUTPUT/'registration.json',registration)
    start=perf_counter();records={};conflicts={};saved={}
    def timeout(*_):raise TimeoutError('fixed diagnostic cap; no extension')
    handler=signal.signal(signal.SIGALRM,timeout);signal.alarm(PROTOCOL['wall_seconds_cap'])
    try:
        with np.load(SOURCE/'operators.npz',allow_pickle=False) as a:op={k:a[k].copy() for k in a.files}
        with np.load(SOURCE/'states.npz',allow_pickle=False) as a:physical={k:a[k].copy() for k in a.files}
        cfg=FixedSzSpatialViTConfig(**source['model_config'])
        template=wrap_spatial_vit_parameters(init_fixed_sz_spatial_vit(jax.random.PRNGKey(0),cfg));_,unravel=ravel_pytree(template)
        state_impl=_saved_operator_state(cfg,op,unravel);state=jax.jit(state_impl)
        h0,h2=jnp.asarray(op['h0']),jnp.asarray(op['h2'])
        def energy(x,j,project):
            p=state_impl(x,j,project);return jnp.vdot(p,(h0+j*h2)@p).real/16
        gradient=jax.jit(jax.grad(energy))
        tangents=jax.jit(lambda x,j,project,directions:jax.vmap(
            lambda d:jax.jvp(lambda y:state_impl(y,j,project),(x,),(d,))[1])(directions))
        coupling_tangent=jax.jit(lambda x,j,project:jax.jvp(lambda jj:state_impl(x,jj,project),
            (j,),(jnp.ones_like(j),))[1])
        bias_probe_tree=jax.tree.map(jnp.zeros_like,template)
        bias_probe_tree['backbone']['patch_embedding_bias']=jnp.ones(16)
        flat_bias,_=ravel_pytree(bias_probe_tree);bias_indices=np.flatnonzero(np.asarray(flat_bias))
        bias_directions_np=np.zeros((len(bias_indices),len(flat_bias)))
        bias_directions_np[np.arange(len(bias_indices)),bias_indices]=1
        bias_directions=jnp.asarray(bias_directions_np)
        total_probes=0;max_replay=0.;max_slope_error=0.
        for name,metadata in source['training'].items():
            parameters=load_parameter_artifact(SOURCE/name/'parameters.json',template,
                expected_identity=dict(run=name,registration_sha256=source['registration_sha256']))
            if pytree_sha256(parameters)!=metadata['final_parameters_sha256']:raise ValueError('endpoint mismatch')
            x,_=ravel_pytree(parameters);project=metadata['representation']=='singlet_vap'
            grads=np.asarray([gradient(x,j,project) for j in PROTOCOL['couplings']])
            lengths=np.linalg.norm(grads,axis=1);directions=np.divide(grads,lengths[:,None],out=np.zeros_like(grads),where=lengths[:,None]>0)
            conflicts[name]=dict(all_J=gradient_conflict_matrix(grads),training_J=gradient_conflict_matrix(grads[[0,2,4]]),
                couplings=PROTOCOL['couplings'],representation=metadata['representation'],kind=metadata['kind'])
            saved[name+'__gradients']=grads;saved[name+'__parameter_probe_directions']=directions
            audit_js=PROTOCOL['couplings'] if metadata['kind']=='shared' else metadata['training_couplings']
            shared_data={}
            for j in audit_js:
                key=name+'__'+str(j);p=np.asarray(state(x,j,project));hp=(op['h0']+j*op['h2'])@p
                replay=np.max(abs(p[op['orbit_id']]/np.sqrt(op['orbit_sizes'][op['orbit_id']])-physical[key]))
                max_replay=max(max_replay,float(replay))
                if replay>2e-10:raise ValueError('physical endpoint replay failed')
                t=np.asarray(tangents(x,j,project,jnp.asarray(directions))).T
                saved[key+'__state']=p;saved[key+'__hstate']=hp;saved[key+'__tangents']=t
                horizontal=t-p[:,None]*(p.conj()@t)[None,:]
                slope=2*np.real(horizontal.conj().T@(hp-np.vdot(p,hp).real*p))/16
                parameter_slope=grads[PROTOCOL['couplings'].index(j)]@directions.T
                max_slope_error=max(max_slope_error,float(np.max(abs(slope-parameter_slope))))
                shared_data[j]=(p,hp,t)
                row=dict(coupling=j,held_out_from_original_shared=j in source['config']['held_out_couplings'],
                    representation=metadata['representation'],kind=metadata['kind'],
                    per_probe_energy_slopes=slope.tolist(),per_probe_FS_speeds=np.linalg.norm(horizontal,axis=0).tolist(),subspaces={})
                if metadata['kind']=='shared' and project:
                    bias_jacobian=np.asarray(tangents(x,j,project,bias_directions)).T
                    actual_coupling=np.asarray(coupling_tangent(x,jnp.asarray(j),project))
                    residual=hp-np.vdot(p,hp).real*p
                    row['conditioning_capacity']=conditioning_tangent_report(p,bias_jacobian,np.asarray(parameters['coupling_bias']),
                        target_derivative=residual,actual_coupling_derivative=actual_coupling)
                    actual_horizontal=actual_coupling-p*np.vdot(p,actual_coupling);speed=np.linalg.norm(actual_horizontal)
                    row['conditioning_capacity']['learned_positive_J_unit_FS_energy_slope_per_site']=float(
                        2*np.real(np.vdot(actual_horizontal,residual))/(16*speed)) if speed>1e-14 else None
                    row['conditioning_capacity']['target_semantics']='energy residual, not validated ground-state response'
                    saved[key+'__bias_jacobian']=bias_jacobian;saved[key+'__coupling_derivative']=actual_coupling
                for label,indices in [('training_J_only',[0,2,4]),('all_five_J',list(range(5)))]:
                    analysis=project_residual_on_tangents(p,hp,t[:,indices],n_sites=16,relative_cutoff=PROTOCOL['relative_rank_cutoff'])
                    info=dict(analysis.metrics);saved[key+'__'+label+'__residual_projection']=analysis.projected_residual
                    if analysis.metrics['unit_FS_speed']>0:
                        direction=directions[indices].T@analysis.descent_coefficients
                        matched=matched_state_step(lambda alpha:np.asarray(state(x+alpha*jnp.asarray(direction),j,project)),
                            p,np.ones(len(p)),PROTOCOL['target_fs'],PROTOCOL['max_normalized_alpha'],PROTOCOL['bisection_steps'])
                        trial=matched.amplitudes;htrial=(op['h0']+j*op['h2'])@trial;e=float(np.vdot(trial,htrial).real)
                        variance=float(np.linalg.norm(htrial-e*trial)**2/16)
                        info.update(physical_step=matched.as_dict(),probe_energy_per_site=e/16,probe_variance_per_site=variance,
                            delta_energy_per_site=e/16-analysis.metrics['energy_per_site'],
                            delta_variance_per_site=variance-analysis.metrics['variance_per_site'],
                            predicted_linear_delta_energy=matched.alpha*analysis.metrics['unit_FS_descent_slope_per_site'],
                            parameter_step_norm=float(matched.alpha*np.linalg.norm(direction)))
                        saved[key+'__'+label+'__probe_state']=trial;total_probes+=1
                    else:info.update(physical_step=None,zero_declared_span_descent=True)
                    row['subspaces'][label]=info
                records[key]=row
            if metadata['kind']=='shared':
                ps=[];hs=[];ts=[];mean_e=np.mean([np.vdot(shared_data[j][0],shared_data[j][1]).real for j in (.2,.5,.8)])
                for j in (.2,.5,.8):
                    p,hp,t=shared_data[j];r=hp-np.vdot(p,hp).real*p
                    horizontal_training=(t-p[:,None]*(p.conj()@t)[None,:])[:,PROTOCOL['training_indices']]
                    ps.append(p/np.sqrt(3));hs.append((r+mean_e*p)/np.sqrt(3));ts.append(horizontal_training/np.sqrt(3))
                joint=project_residual_on_tangents(np.concatenate(ps),np.concatenate(hs),np.concatenate(ts),n_sites=16)
                conflicts[name]['mean_training_objective_product_FS_training_gradient_span']=joint.metrics
                conflicts[name]['joint_metric_semantics']='equal-weight direct product of three physical rays; residual norm is mean individual variance, not variance of one common Hamiltonian'
            atomic_json(OUTPUT/'partial_results.json',dict(records=records,conflicts=conflicts,seconds=perf_counter()-start))
            print(json.dumps(dict(endpoint=name,state_count=len(records),seconds=perf_counter()-start)),flush=True)
        secants={}
        for parent in source['config']['parents']:
            for index in (1,2,3):
                before,after=PROTOCOL['couplings'][index-1],PROTOCOL['couplings'][index+1]
                sh=[physical[parent+'__singlet_vap__shared__'+str(j)] for j in (before,after)]
                pt=[physical[parent+f'__singlet_vap__point_{j:.2f}__'+str(j)] for j in (before,after)]
                secants[parent+'__'+str(PROTOCOL['couplings'][index])]=dict(projector_secant_alignment(*sh,*pt,coupling_span=after-before),
                    pointwise_is_variational_proxy_not_exact=True)
        _save_npz(OUTPUT/'directions_and_probes.npz',**saved)
        result=dict(status='complete',protocol=PROTOCOL,records=records,gradient_conflicts=conflicts,
            max_physical_endpoint_replay=max_replay,max_directional_derivative_identity_error=max_slope_error,
            seconds=perf_counter()-start,training_updates_committed=0,new_ED_calls=0,new_samples=0,new_H_constructions=0,
            full_parameter_gradient_objectives=5*len(source['training']),state_tangent_JVPs=5*len(records),
            conditioning_bias_JVPs=160,coupling_input_JVPs=10,
            affine_bias_rank_bound=affine_bias_rank_bound(16,PROTOCOL['couplings']),saved_state_projector_secants=secants,
            finite_state_probes=total_probes,FS_candidate_state_evaluations=17*total_probes,
            explicit_saved_matrix_H_actions_outside_gradients=len(records)+total_probes,finite_direction_lower_bounds_only=True,
            independent_evidence=False,variational_win_claimed=False,registration_sha256=file_sha256(OUTPUT/'registration.json'),
            artifacts={'directions_and_probes.npz':file_sha256(OUTPUT/'directions_and_probes.npz')})
        result['result_sha256']=canonical_sha256(result);atomic_json(OUTPUT/'results.json',result)
        return result
    except Exception as error:
        atomic_json(OUTPUT/'failure.json',dict(error=repr(error),seconds=perf_counter()-start,completed_states=len(records)))
        raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,handler)


if __name__=='__main__':run()
