"""Bounded energy-only, same-backbone coupling transfer on exact N16 support.

The computational-spin ViT is reused without widening/deepening. Exact SU2
is a small-support VAP ablation, not a many-cell decoder. No ED, no new Monte
Carlo samples, no phase labels and no target-wavefunction loss are used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.benchmarks.krylov_anchored_full_vit import a1_orbit_hamiltonian
from su2_nqs.benchmarks.singlet_vap import physical_cut_metrics
from su2_nqs.benchmarks.su2_entanglement_spectrum import reduced_density_matrix
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact, save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.coupling_conditioned_spatial_vit import effective_parameters, wrap_spatial_vit_parameters
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit
from .spin import _prepare_case, _bonds, heisenberg_sz_graph_sparse
from .spin_full_vit_vap import prepare_vap
from .spin_overlap_attention_cost import _save_npz, peak_rss_gib
from .spin_symmetry_readout import density_readout

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_coupling_transfer.json'


def validate_protocol(spec: dict) -> None:
    """Fix split, work and scope before training, without inspecting outcomes."""
    fixed = dict(task='spin_coupling_transfer', train_couplings=[.2,.5,.8],
        held_out_couplings=[.35,.65], all_couplings=[.2,.35,.5,.65,.8],
        representations=['fixed_sz','singlet_vap'], shared_steps=300, pointwise_steps=180,
        learning_rate=.001, coupling_center=.5, record_every=30, wall_seconds_cap=1800,
        response_fd_step=.0001, heldout_in_objective=False, remote_allowed=False,
        new_ED_allowed=False, new_sampling_allowed=False)
    if any(spec.get(k)!=v for k,v in fixed.items()):
        raise ValueError('changed registered coupling-transfer protocol')
    if spec['parents'] != ['square_d16_l2_s2026092901','square_d16_l2_s2026092902']:
        raise ValueError('both original parents must be retained')


def register(spec: dict) -> dict:
    validate_protocol(spec)
    inputs = {}
    for name in spec['parents']:
        folder = ROOT/spec['parent_root']/name
        parent = json.loads((folder/'results.json').read_text())
        unsigned=dict(parent);seal=unsigned.pop('result_sha256')
        if canonical_sha256(unsigned)!=seal:
            raise ValueError('parent result checksum mismatch')
        for file in ('results.json','parameters.json','parameters.npz'):
            path=folder/file;inputs[str(path.relative_to(ROOT))]=file_sha256(path)
    record=dict(config=spec,inputs=inputs,runtime=numerical_runtime_identity(),
        sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    record['registration_sha256']=canonical_sha256(record)
    return record


def rayleigh_metrics(psi: np.ndarray, matrix: np.ndarray, n_sites: int=16) -> dict:
    """Full-support Rayleigh energy and nonnegative residual variance."""
    p=np.asarray(psi)/np.linalg.norm(psi);h=matrix@p;e=np.vdot(p,h).real
    return dict(energy_per_site=float(e/n_sites),
                variance_per_site=float(np.linalg.norm(h-e*p)**2/n_sites))


def response_metrics(state, parameters: jax.Array, coupling: float,
                     h0: jax.Array, h2: jax.Array, delta: float) -> dict:
    """Differentiate the LEARNED branch; HF defect is not a response certificate.

    No finite-field optimization occurs. FD is an independent differentiation
    check of this branch, not a ground-state susceptibility reference.
    """
    def energy(j):
        v=state(parameters,j)
        return jnp.vdot(v,(h0+j*h2)@v).real/16
    def bond(j):
        v=state(parameters,j)
        return jnp.vdot(v,h2@v).real/16
    j=jnp.asarray(coupling,dtype=jnp.float64)
    v,dv=jax.jvp(lambda x:state(parameters,x),(j,),(jnp.ones_like(j),))
    derivative=float(jax.grad(energy)(j));curvature=float(jax.grad(jax.grad(energy))(j))
    bond_value=float(bond(j));bond_derivative=float(jax.grad(bond)(j))
    tangent=dv-v*jnp.vdot(v,dv)
    residual=(h0+j*h2)@v-jnp.vdot(v,(h0+j*h2)@v).real*v
    metric=float(jnp.vdot(tangent,tangent).real)
    residual_hf=float(2*jnp.vdot(tangent,residual).real/16)
    em,e0,ep=map(float,(energy(j-delta),energy(j),energy(j+delta)))
    bminus,bplus=map(float,(bond(j-delta),bond(j+delta)))
    return dict(energy_derivative_per_site=derivative,bond_H2_per_site=bond_value,
        hellmann_feynman_defect=derivative-bond_value,energy_curvature_per_site=curvature,
        bond_response_per_site=bond_derivative,
        curvature_minus_bond_response=curvature-bond_derivative,
        fidelity_susceptibility=metric,
        hellmann_feynman_residual_identity_error=abs(derivative-bond_value-residual_hf),
        hellmann_feynman_residual_bound=float(2*jnp.sqrt(metric*jnp.vdot(residual,residual).real)/16),
        energy_derivative_fd_error=abs(derivative-(ep-em)/(2*delta)),
        energy_curvature_fd_error=abs(curvature-(ep-2*e0+em)/delta**2),
        bond_response_fd_error=abs(bond_derivative-(bplus-bminus)/(2*delta)),
        interpretation='learned branch derivative, not independently validated ground-state response')


def run(config: Path=CONFIG, *, resume: bool=False) -> dict:
    spec=json.loads(config.read_text());registered=register(spec);out=ROOT/spec['output']
    out.mkdir(parents=True,exist_ok=True);regpath=out/'registration.json'
    if regpath.exists():
        if not resume or json.loads(regpath.read_text())!=registered:
            raise ValueError('existing experiment requires unchanged registration and explicit resume')
    else:
        if any(out.iterdir()):raise FileExistsError('output is not empty')
        atomic_json(regpath,registered)
    if (out/'results.json').exists():raise FileExistsError('completed attempt is immutable')
    old=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
    elapsed=float(old.get('seconds',0.));start=perf_counter()
    if elapsed>=spec['wall_seconds_cap']:raise TimeoutError('registered cumulative budget exhausted')
    def progress(stage,**fields):
        record=dict(stage=stage,seconds=elapsed+perf_counter()-start,peak_rss_gib=peak_rss_gib(),**fields)
        atomic_json(out/'progress.json',record);print(json.dumps(record),flush=True)
    def timeout(*_):raise TimeoutError('registered cumulative cap; no automatic extension')
    handler=signal.signal(signal.SIGALRM,timeout);signal.alarm(int(spec['wall_seconds_cap']-elapsed))
    try:
        # The known J=.5 scalar is never substituted as a reference at another J.
        parent0=json.loads((ROOT/spec['parent_root']/spec['parents'][0]/'results.json').read_text())
        data=_prepare_case(parent0['case'],'fixed_m0')
        cfg=FixedSzSpatialViTConfig(**parent0['model_config']);problem=prepare_vap(data,cfg)
        matrices=[]
        for j in (0.,1.):
            operator,basis=heisenberg_sz_graph_sparse(16,_bonds('square',4,4,j),n_up=8)
            if tuple(basis)!=data['basis']:raise ValueError('H-family basis order mismatch')
            coo=operator.tocoo()
            matrices.append(a1_orbit_hamiltonian(coo.row,coo.col,coo.data,problem['raw']['a1']))
        h0=matrices[0];h2=matrices[1]-h0;p0=problem['p0'];s2=problem['s2']
        checks=dict(problem['checks'],hamiltonian_family_replay=float(np.max(abs(h0+.5*h2-problem['h']))),
            H0_P0_commutator=float(np.max(abs(h0@p0-p0@h0))),
            H2_P0_commutator=float(np.max(abs(h2@p0-p0@h2))))
        if max(checks.values())>1e-9:raise RuntimeError(checks)
        prepared=out/'operators.npz'
        operators=dict(h0=h0,h2=h2,p0=p0,s2=s2,
            basis=np.asarray(data['basis']),orbit_id=problem['raw']['a1'].orbit_id,
            orbit_sizes=problem['raw']['a1'].orbit_sizes)
        if not prepared.exists():_save_npz(prepared,**operators)
        else:
            with np.load(prepared,allow_pickle=False) as saved:
                if set(saved.files)!=set(operators) or any(not np.array_equal(saved[k],v) for k,v in operators.items()):
                    raise ValueError('saved operators differ from deterministic reconstruction')
        matrices_jax=(jnp.asarray(h0),jnp.asarray(h2));projector=jnp.asarray(p0)
        template=init_fixed_sz_spatial_vit(jax.random.PRNGKey(0),cfg)
        x_template,unravel=ravel_pytree(wrap_spatial_vit_parameters(template))
        def state_impl(x,j,project):
            raw=problem['raw']['state_fn'](effective_parameters(unravel(x),j,center=spec['coupling_center']))
            v=jax.lax.cond(project,lambda r:projector@r,lambda r:r,raw)
            return v/jnp.linalg.norm(v)
        def loss(x,js,project):
            vectors=jax.vmap(lambda j:state_impl(x,j,project))(js)
            hvs=jax.vmap(lambda j,v:(matrices_jax[0]+j*matrices_jax[1])@v)(js,vectors)
            return jnp.mean(jnp.sum(vectors.conj()*hvs,axis=1).real)/16
        value_grad=jax.jit(jax.value_and_grad(loss))
        state_jit=jax.jit(state_impl)
        @jax.jit
        def update(x,m,v,t,js,project):
            value,g=value_grad(x,js,project)
            m=.9*m+.1*g;v=.999*v+.001*g*g
            step=spec['learning_rate']*(m/(1-.9**t))/(jnp.sqrt(v/(1-.999**t))+1e-8)
            return x-step,m,v,value,jnp.linalg.norm(g)
        records={};parents={}
        for parent_name in spec['parents']:
            folder=ROOT/spec['parent_root']/parent_name
            parent=json.loads((folder/'results.json').read_text())
            params=load_parameter_artifact(folder/'parameters.json',template,expected_identity=dict(trial=parent_name))
            if pytree_sha256(params)!=parent['final_params_sha256']:raise ValueError('parent parameters changed')
            x0,_=ravel_pytree(wrap_spatial_vit_parameters(params))
            parents[parent_name]=dict(inherited_training=parent['training'],inherited_wall_seconds=parent['wall_seconds'],
                original_parameter_count=parent['parameter_count'],conditioned_parameter_count=len(x0),
                replay=problem['raw']['validate'](params),parameters_sha256=pytree_sha256(params))
            # Only TRAINING couplings enter gradient validation or shared fitting.
            for representation in spec['representations']:
                project=representation=='singlet_vap'
                def state(x,j):return state_jit(x,j,project)
                direction=jnp.sin(jnp.arange(len(x0)))/np.sqrt(len(x0))
                js=jnp.asarray(spec['train_couplings'])
                ad=float(jnp.vdot(value_grad(x0,js,project)[1],direction))
                fd=float((loss(x0+1e-5*direction,js,project)-loss(x0-1e-5*direction,js,project))/2e-5)
                if abs(ad-fd)>1e-7:raise RuntimeError('full-gradient FD validation failed')
                checks[parent_name+'_'+representation+'_gradient_fd_error']=abs(ad-fd)
                tasks=([('shared',spec['train_couplings'],spec['shared_steps'])]+
                    [(f'point_{j:.2f}',[j],spec['pointwise_steps']) for j in spec['all_couplings']])
                for task,training_js,steps in tasks:
                    name=parent_name+'__'+representation+'__'+task;destination=out/name
                    destination.mkdir(exist_ok=True)
                    identity=dict(run=name,registration_sha256=file_sha256(regpath))
                    if (destination/'training.json').exists():
                        record=json.loads((destination/'training.json').read_text())
                        endpoint=load_parameter_artifact(destination/'parameters.json',unravel(x0),expected_identity=identity)
                        if pytree_sha256(endpoint)!=record['final_parameters_sha256']:
                            raise ValueError('completed endpoint no longer matches training record')
                        records[name]=record;continue
                    x=x0;m=jnp.zeros_like(x);v=jnp.zeros_like(x);first=0;history=[];active=0.
                    ckpt=destination/'checkpoint.json'
                    cp_template=dict(x=x,m=m,v=v,step=jnp.asarray(0),active_seconds=jnp.asarray(0.))
                    if ckpt.exists():
                        restored=load_parameter_artifact(ckpt,cp_template,expected_identity=identity)
                        x,m,v=restored['x'],restored['m'],restored['v'];first=int(restored['step']);active=float(restored['active_seconds'])
                        history=json.loads((destination/'history.json').read_text())
                        history=[r for r in history if r['step']<=first]
                    before=perf_counter();couplings=jnp.asarray(training_js)
                    for step in range(first+1,steps+1):
                        x,m,v,value,gnorm=update(x,m,v,step,couplings,project)
                        if not np.isfinite(float(value)) or not np.isfinite(float(gnorm)) or not np.isfinite(np.asarray(x)).all():
                            raise FloatingPointError('nonfinite training iterate')
                        if step%spec['record_every']==0 or step==steps:
                            endpoint_loss=float(loss(x,couplings,project))
                            history.append(dict(step=step,mean_energy_per_site=endpoint_loss,gradient_norm=float(gnorm)))
                            atomic_json(destination/'history.json',history)
                            save_parameter_artifact(ckpt,dict(x=x,m=m,v=v,step=jnp.asarray(step),
                                active_seconds=jnp.asarray(active+perf_counter()-before)),identity=identity,overwrite=ckpt.exists())
                            progress('training',run=name,step=step,total_steps=steps,energy=endpoint_loss)
                    active+=perf_counter()-before
                    if (destination/'parameters.json').exists():
                        saved=load_parameter_artifact(destination/'parameters.json',unravel(x),expected_identity=identity)
                        if pytree_sha256(saved)!=pytree_sha256(unravel(x)):
                            raise ValueError('endpoint and optimizer checkpoint disagree')
                    else:save_parameter_artifact(destination/'parameters.json',unravel(x),identity=identity)
                    record=dict(name=name,parent=parent_name,representation=representation,kind=task,
                        training_couplings=training_js,steps=steps,full_H_training_objectives=steps*len(training_js),
                        energy_monitor_H_objectives=len(history)*len(training_js),active_seconds=active,
                        history=history,final_parameters_sha256=pytree_sha256(unravel(x)),parameter_count=len(x),
                        own_initial_training_energy=float(loss(x0,couplings,project)),
                        final_training_energy=float(loss(x,couplings,project)),optimizer='Adam zero moments, fixed final step')
                    atomic_json(destination/'training.json',record);records[name]=record
        endpoints={name:{ext:file_sha256(out/name/('parameters.'+ext)) for ext in ('json','npz')} for name in records}
        atomic_json(out/'endpoints_seal.json',endpoints)
        progress('all_endpoints_sealed_before_transfer_evaluation')
        audit_start=perf_counter();audit={};saved_states={};saved_rhos={}
        for name,record in records.items():
            identity=dict(run=name,registration_sha256=file_sha256(regpath))
            params=load_parameter_artifact(out/name/'parameters.json',unravel(x_template),expected_identity=identity)
            if pytree_sha256(params)!=record['final_parameters_sha256']:raise ValueError('endpoint changed before audit')
            x,_=ravel_pytree(params);project=record['representation']=='singlet_vap'
            def state(x,j):return state_jit(x,j,project)
            js=spec['all_couplings'] if record['kind']=='shared' else record['training_couplings']
            row={}
            for j in js:
                vector=np.asarray(state(x,j));physical=problem['raw']['physical'](vector)
                report=rayleigh_metrics(vector,h0+j*h2)
                report.update(spin_squared=float(np.vdot(vector,s2@vector).real),
                    singlet_weight=float(np.linalg.norm(p0@vector)**2),
                    norm_error=float(abs(np.linalg.norm(vector)-1)),
                    exact_energy_error_per_site=(report['energy_per_site']-data['exact_energy']/16) if j==.5 else None,
                    coupling_held_out_from_shared=j in spec['held_out_couplings'],
                    cuts={cut:physical_cut_metrics(physical,np.asarray(data['basis']),sites)
                        for cut,sites in dict(plaquette=(0,1,4,5),row=(0,1,2,3)).items()})
                if project:
                    rho=reduced_density_matrix(physical,np.asarray(data['basis']),(0,1,4,5))
                    report['multiplicity_readout'],_=density_readout(rho)
                    saved_rhos[name+'__'+str(j)]=rho
                if record['kind']=='shared':
                    before=perf_counter()
                    report['response']=response_metrics(state,x,j,*matrices_jax,spec['response_fd_step'])
                    report['response']['seconds']=perf_counter()-before
                row[str(j)]=report;saved_states[name+'__'+str(j)]=physical
            audit[name]=row
            progress('evaluated',run=name)
        comparisons=[]
        for parent in spec['parents']:
            for j in spec['all_couplings']:
                arms={}
                for rep in spec['representations']:
                    prefix=parent+'__'+rep+'__'
                    shared=audit[prefix+'shared'][str(j)];point=audit[prefix+f'point_{j:.2f}'][str(j)]
                    arms[rep]=dict(shared_minus_pointwise_energy=shared['energy_per_site']-point['energy_per_site'],
                        shared_minus_pointwise_variance=shared['variance_per_site']-point['variance_per_site'])
                    if rep=='singlet_vap':
                        rho_difference=saved_rhos[prefix+'shared__'+str(j)]-saved_rhos[prefix+f'point_{j:.2f}__'+str(j)]
                        arms[rep]['plaquette_trace_distance_to_pointwise']=float(np.sum(abs(np.linalg.eigvalsh(rho_difference)))/2)
                        sr,pr=shared['multiplicity_readout'],point['multiplicity_readout']
                        arms[rep]['p_q_TV_to_pointwise']=float(sum(abs(sr['sectors'][str(q)]['p']-pr['sectors'][str(q)]['p']) for q in (0,2,4))/2)
                        arms[rep]['multiplicity_entropy_minus_pointwise']=sr['multiplicity_entropy']-pr['multiplicity_entropy']
                su=audit[parent+'__singlet_vap__shared'][str(j)];raw=audit[parent+'__fixed_sz__shared'][str(j)]
                comparisons.append(dict(parent=parent,coupling=j,held_out=j in spec['held_out_couplings'],
                    shared_SU2_minus_fixedSz_energy=su['energy_per_site']-raw['energy_per_site'],
                    shared_SU2_minus_fixedSz_variance=su['variance_per_site']-raw['variance_per_site'],arms=arms))
        _save_npz(out/'states.npz',**saved_states,basis=np.asarray(data['basis']))
        artifacts={f:str(file_sha256(out/f)) for f in ('operators.npz','states.npz','endpoints_seal.json')}
        for name in records:
            for file in ('training.json','history.json','parameters.json','parameters.npz','checkpoint.json','checkpoint.npz'):
                artifacts[name+'/'+file]=file_sha256(out/name/file)
        result=dict(status='complete',task=spec['task'],config=spec,model_config=asdict(cfg),parents=parents,
            checks=checks,training=records,audit=audit,comparisons=comparisons,
            exact_reference_couplings=[.5],ground_state_response_accuracy_certified=False,
            physical_phase_claim=False,scalable_SU2_decoder=False,new_ED_calls=0,new_samples=0,remote_released=False,
            total_training_H_objectives=sum(r['full_H_training_objectives'] for r in records.values()),
            transfer_audit_seconds=perf_counter()-audit_start,
            seconds=elapsed+perf_counter()-start,peak_rss_gib=peak_rss_gib(),
            registration_sha256=file_sha256(regpath),artifacts=artifacts,
            caveats=['Historical J=.5 warmstarts, not new independent random seeds.',
                'Projection changes physical initializer; same parameters and backbone, not same initial wavefunction.',
                'Only shared models withhold two J points. Pointwise controls intentionally train there.',
                'Equal aggregate training H objectives, not equal steps per J or wall time; training and audit wall times recorded, projection overhead not isolated.',
                '4008 stored parameters; pointwise coupling_bias is redundant with patch bias and is inactive at J=.5.',
                'Exact support removes MC uncertainty but not variational/optimization error.',
                'HF consistency and finite-difference agreement are necessary diagnostics, not ground-state response accuracy.'])
        result['result_sha256']=canonical_sha256(result);atomic_json(out/'results.json',result)
        progress('complete');return result
    except Exception as error:
        progress('failed',error=repr(error));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    run(args.config,resume=args.resume)
