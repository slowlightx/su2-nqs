"""Sealed fixed-L coverage/budget experiment and conditional family retest.

The primary comparison uses the SAME central development points, inside
both training intervals. Wider-domain extrapolation is secondary. Two
milestones share one trajectory; count its terminal work only once.
"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from fnqs.artifacts import write_json,load_checkpoint
from fnqs.basis import all_bits
from fnqs.diagnostics.response import summarize_response
from fnqs.hamiltonians import IsingCouplings
from fnqs.observables import correlations_from_state
from fnqs.sectors import parity_reference
from fnqs.variants.size_features import SizeFeatureConfig,log_psi,parameter_counts
from .engine import Options,train,init_model,model_signature
from .measurements import action,amplitudes
from .lineage import ROOT,sha,sources,verify_sources,cost,check_cost,save_node,read_node,validate_splits

PRIMARY = [-.03,-.01,.01,.03]
SECONDARY = [-.145,-.125,-.105,-.085,-.065,-.055,-.045,.045,.055,.065,.085,.105,.125,.145]
RESERVED = [-.035,-.015,.015,.035]
SEEDS = [113,127,139]


def points(center: float, offsets: list[float], j2: float=0., jx: float=0.) -> list[list[float]]:
    """Coupling order (j1,j2,jx,h); deterministic rounded physical points."""
    return [[1.,j2,jx,round(center+x,8)] for x in offsets]


def seal(directory: Path, declaration: dict) -> None:
    """Snapshot all imports and this launcher before any numerical work."""
    if directory.exists():
        raise FileExistsError("use a new experiment directory")
    launchers=('run_coupling_factorial.py','run_factorial_solution.py')
    manifest={**sources(),**{f'fnqs/scripts/{name}':sha(ROOT/'fnqs/scripts'/name) for name in launchers}}
    directory.mkdir(parents=True)
    for name in manifest:
        destination=directory/'runtime'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    write_json(directory/'protocol.json',{**declaration,'source_sha256':manifest})


def prepare(directory: Path) -> None:
    """Declare 6 cold trajectories, 12 endpoints; only two design factors."""
    prior=ROOT/'fnqs/results/fixed_size_pilots_v11/protocol.json'
    old=json.loads(prior.read_text());center=next(c['finite_size_peak'] for c in old['cuts'] if c['family']=='A')
    trajectories=[]
    for window,width in [('narrow',.04),('wide',.15)]:
        training=points(center,[-width,-width/2,0,width/2,width])
        primary,secondary,reserved=(points(center,o) for o in (PRIMARY,SECONDARY,RESERVED))
        validate_splits(training,primary+secondary,reserved)
        for seed in SEEDS:
            schedule=[[training[i%5]] for i in range(600)]
            trajectories.append(dict(name=f'{window}_seed_{seed}',window=window,half_width=width,seed=seed,
                train_points=training,schedule=schedule,milestones=[150,600],
                endpoint_costs={str(step):cost(schedule[:step],64) for step in (150,600)}))
    seal(directory,dict(kind='coupling_coverage_budget_factorial_v12',n=12,family='A',center=center,seeds=SEEDS,
        model=asdict(SizeFeatureConfig(8)),options=asdict(Options(max_step=.1)),trajectories=trajectories,
        primary_points=primary,secondary_points=secondary,reserved_points=reserved,
        original_input={str(prior):sha(prior)},parameter_counts=parameter_counts(SizeFeatureConfig(8)),
        primary_metric='worst total-chi relative error on SAME four central development points',
        decision_minimum_improvement=.05,
        decision_rule='An effect is consistent if improvement >= .05 absolute error for ALL three paired seeds at either other-factor setting.',
        solution_rule='If coverage has a consistent effect choose wide/600; otherwise if budget has a consistent effect choose narrow/600. Retest ABC versus family-wise controls under a NEW bounded cold protocol.',
        followup_maximum_tasks=10800,followup_samples_per_task=64,
        unchanged_observable_limits=dict(chi_relative_error=.1,max_correlation_absolute_error=.03,energy_relative_error=.002),
        audit_strategy='Exhaustive 4096 Born rows; full C(r), state overlap, JVP/FD at fixed theta; independent parity ED and JW cross-check.',
        reference_labels_in_training=False,no_automatic_extension_of_declared_trajectories=True,
        reserved_confirmation_measured=False,phase0_scientific_gate=False,universality_established=False))


@partial(jax.jit,static_argnames=('config',))
def finite_scores(params: dict,bits: jax.Array,g: jax.Array,config: SizeFeatureConfig) -> jax.Array:
    """Two central FD steps, on the same exhaustive rows, at fixed theta."""
    direction=jnp.array([0.,0.,0.,1.])
    return jnp.stack([(log_psi(params,bits,g+delta*direction,config)-
                      log_psi(params,bits,g-delta*direction,config))/(2*delta)
                     for delta in (1e-3,5e-4)],axis=1)


def get_reference(directory: Path,n: int,g: list[float]) -> tuple[dict,str]:
    """Save independent ED state/response and all correlations, with a hash."""
    label='_'.join(str(x) for x in g);path=directory/'references'/f'L{n}_{label}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            ref={k:np.asarray(data[k]) for k in data.files}
        if not np.array_equal(ref['couplings'],g):raise ValueError('reference labels mismatch')
        return ref,sha(path)
    reference=parity_reference(n,IsingCouplings(*g))
    ref=dict(couplings=np.asarray(g),energy=np.asarray(reference.energy),state=reference.state,
             chi=np.asarray(reference.chi_total),correlations=correlations_from_state(reference.state,n),
             state_residual=np.asarray(reference.state_residual),response_residual=np.asarray(reference.response_relative_residual))
    if g[1]==0 and g[2]==0:
        from fnqs.exact import TFIMReference
        jw=TFIMReference(n,g[3]); discrepancies=[abs(reference.energy-jw.energy()),
            abs(reference.chi_total-jw.fidelity_susceptibility()),
            np.max(abs(ref['correlations']-np.array([jw.correlation(r) for r in range(n)])))]
        if max(discrepancies)>2e-8:raise RuntimeError('ED/JW oracle disagreement')
        ref['jw_largest_discrepancy']=np.asarray(max(discrepancies))
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**ref)
    return ref,sha(path)


def audit(params: dict,n: int,g: list[float],config: SizeFeatureConfig,ref: dict,archive: Path) -> dict:
    """Measure full C(r), physical response and FD, using no optimizer labels."""
    bits=jnp.asarray(all_bits(n));logs,score=amplitudes(params,bits,jnp.asarray(g),config)
    psi,score=np.exp(np.asarray(logs)),np.asarray(score);fd=np.asarray(finite_scores(params,bits,jnp.asarray(g),config))
    normalization=abs(float(np.sum(psi**2))-1)
    if normalization>1e-10:raise RuntimeError('normalization guard failed')
    rows=dict(score=score,energy=action(psi,n,g)/psi,h_score=action(psi*score,n,g)/psi,
              source=action(psi,n,[0,0,0,1])/psi,fd_scores=fd)
    summary=summarize_response(rows,psi**2)
    if max(summary['fd_score_relative_rms_error'])>5e-4:raise RuntimeError('input AD/FD guard failed')
    corr=correlations_from_state(psi,n)
    archive.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(archive,bits=np.asarray(bits),psi=psi,couplings=np.asarray(g),**rows)
    return {**summary,'n':n,'couplings':g,'normalization_error':normalization,
        'chi_relative_error':abs(summary['chi_total']/float(ref['chi'])-1),
        'chi_reference_ratio':summary['chi_total']/float(ref['chi']),
        'energy_relative_error':abs(summary['energy_total']-float(ref['energy']))/abs(float(ref['energy'])),
        'energy_per_site_absolute_error':abs(summary['energy_total']-float(ref['energy']))/n,
        'correlations':corr.tolist(),'reference_correlations':ref['correlations'].tolist(),
        'max_correlation_absolute_error':float(np.max(abs(corr-ref['correlations']))),
        'state_infidelity':float(max(0.,1-abs(np.vdot(ref['state'],psi))**2)),
        'reference_chi_total':float(ref['chi']),'archive_sha256':sha(archive),
        'born_samples':0,'basis_rows':len(bits),'hamiltonian_actions':2,'source_actions':1,
        'base_amplitude_jvp_batches':1,'fd_amplitude_batches':4}


def endpoint_audit(directory: Path,folder: Path,protocol: dict,primary: list,secondary: list) -> dict:
    """Audit a sealed endpoint; immutable checkpoint identity is required."""
    read_node(folder/'node.json')
    path=folder/'audit.json'
    if path.exists():
        result=json.loads(path.read_text())
        if result['checkpoint_sha256']!=sha(folder/'checkpoint.npz') or result['protocol_sha256']!=sha(directory/'protocol.json'):
            raise ValueError('audit binding mismatch')
        return result
    params,_,_,_=load_checkpoint(folder/'checkpoint.npz');config=SizeFeatureConfig(**protocol['model'])
    output=[]
    for split,point_list in [('primary',primary),('secondary',secondary)]:
        for g in point_list:
            ref,digest=get_reference(directory,protocol['n'],g)
            point=audit(params,protocol['n'],g,config,ref,folder/f'point_{len(output)}.npz')
            output.append({**point,'split':split,'reference_sha256':digest})
    result=dict(points=output,checkpoint_sha256=sha(folder/'checkpoint.npz'),protocol_sha256=sha(directory/'protocol.json'))
    write_json(path,result);return result


def metrics(points_out: list[dict],split: str='primary') -> dict:
    """Worst errors remain primary; RMS is descriptive, not a replacement."""
    rows=[p for p in points_out if p['split']==split]
    return dict(chi=max(p['chi_relative_error'] for p in rows),
        chi_rms=float(np.sqrt(np.mean([p['chi_relative_error']**2 for p in rows]))),
        correlation=max(p['max_correlation_absolute_error'] for p in rows),
        energy=max(p['energy_relative_error'] for p in rows),
        infidelity=max(p['state_infidelity'] for p in rows),rho=max(p['response_relative_residual'] for p in rows))


def decision(rows: list[dict],minimum: float=.05) -> dict:
    """Predeclared paired development effects; no three-seed significance claim."""
    indexed={(r['window'],r['step'],r['seed']):r for r in rows}
    if set(indexed)!={(w,t,s) for w in ('narrow','wide') for t in (150,600) for s in SEEDS}:
        raise ValueError('factorial endpoints incomplete or duplicated')
    if len(rows)!=12:raise ValueError('duplicate endpoint labels')
    coverage={str(t):[indexed['narrow',t,s]['primary']['chi']-indexed['wide',t,s]['primary']['chi'] for s in SEEDS] for t in (150,600)}
    budget={w:[indexed[w,150,s]['primary']['chi']-indexed[w,600,s]['primary']['chi'] for s in SEEDS] for w in ('narrow','wide')}
    covered=any(min(values)>=minimum for values in coverage.values())
    optimized=any(min(values)>=minimum for values in budget.values())
    chosen='wide' if covered else 'narrow' if optimized else None
    interaction=[(indexed['narrow',150,s]['primary']['chi']-indexed['wide',150,s]['primary']['chi'])-
                 (indexed['narrow',600,s]['primary']['chi']-indexed['wide',600,s]['primary']['chi']) for s in SEEDS]
    return dict(coverage_improvements=coverage,budget_improvements=budget,interaction=interaction,
        coverage_consistent=covered,budget_consistent=optimized,minimum_paired_improvement=minimum,
        identified_under_declared_budget=covered or optimized,selected_window=chosen,selected_steps=600 if chosen else None,
        selection_points_are_development=True,formal_scientific_gate=False)


def run(directory: Path) -> dict:
    """Execute each cold trajectory once, seal both milestones, audit after all."""
    protocol=json.loads((directory/'protocol.json').read_text());verify_sources(protocol['source_sha256'])
    for name,digest in protocol['original_input'].items():
        if sha(Path(name))!=digest:raise ValueError('original input changed')
    config=SizeFeatureConfig(**protocol['model']);options=Options(**protocol['options']);began=time.perf_counter()
    for trajectory in protocol['trajectories']:
        base=directory/'trajectories'/trajectory['name']
        if (base/'step_600/node.json').exists():read_node(base/'step_600/node.json');continue
        if (base/'started.json').exists():raise RuntimeError('unfinished trajectory preserved; declare a new version')
        write_json(base/'started.json',dict(protocol_sha256=sha(directory/'protocol.json')))
        seed=trajectory['seed'];params=init_model(jax.random.PRNGKey(seed),config)
        def callback(p,state,key,history):
            step=state['step']
            if step in trajectory['milestones']:
                check_cost(state['work'],trajectory['endpoint_costs'][str(step)])
                save_node(base/f'step_{step}',p,state,key,signature=model_signature(config,protocol['n']),
                    protocol_path=directory/'protocol.json',train_points=trajectory['train_points'],history=history,
                    parents=[],stage='factorial_cold_cumulative')
                print(trajectory['name'],'sealed',step,flush=True)
        train(params,jax.random.PRNGKey(seed+1000),protocol['n'],trajectory['schedule'],config,options,callback)
    endpoint_seal={str(directory/'trajectories'/t['name']/f'step_{step}'/'node.json'):
        sha(directory/'trajectories'/t['name']/f'step_{step}'/'node.json') for t in protocol['trajectories'] for step in t['milestones']}
    write_json(directory/'endpoint_seal.json',dict(nodes=endpoint_seal,reference_opened=False))
    output=[]
    for trajectory in protocol['trajectories']:
        for step in trajectory['milestones']:
            folder=directory/'trajectories'/trajectory['name']/f'step_{step}'
            result=endpoint_audit(directory,folder,protocol,protocol['primary_points'],protocol['secondary_points'])
            output.append(dict(window=trajectory['window'],seed=trajectory['seed'],step=step,
                primary=metrics(result['points']),secondary=metrics(result['points'],'secondary'),audit_sha256=sha(folder/'audit.json')))
    report=dict(kind='coverage_budget_factorial_review',rows=output,decision=decision(output),
        actual_training_cost=cost([row for t in protocol['trajectories'] for row in t['schedule']],options.samples),
        audit_cost=dict(born_samples=0,basis_rows=12*18*4096,hamiltonian_actions=12*18*2,source_actions=12*18),
        reserved_confirmation_measured=False,formal_scientific_gate=False,universality_established=False,
        protocol_sha256=sha(directory/'protocol.json'),elapsed_seconds=time.perf_counter()-began)
    write_json(directory/'review.json',report);print(report['decision'],flush=True);return report
