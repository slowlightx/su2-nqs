"""Fixed-budget energy continuation and one endpoint-blind independent audit.

Reuse the sealed feature compiler, sampler, and portable checkpoints. Worker
processes bound accumulated JAX caches without changing states or resampling.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.multicontext_resonance_features import multicontext_feature_coefficients
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact, save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.additive_environment import additive_environment_generators
from su2_nqs.vmc.cached_energy_continuation import CachedEnergyConfig, run_cached_energy_continuation
from su2_nqs.vmc.cutset_training import paired_chain_audit, weighted_state_distance
from su2_nqs.vmc.transition_moments import estimate_transition_moments, joint_block_bootstrap
from .spin_overlap_attention_cost import _save_npz, peak_rss_gib
from .spin_resonance_training import FixedCarrierFeatures, build_corpus
from .spin_resonance_distillation import load_saved_problem, load_corpus, state_from_coefficients
from .spin_schur_carrier_control import draw

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_resonance_continuation.json'
MODULE = 'projects.clebsch_tree_nqs.workflows.spin_resonance_continuation'
STARTS = ('old_endpoint', 'original_initialization', 'independent_initialization')
ARMS = ('continued_additive110', 'continued_free360',
        *(name for i in range(3) for name in (f'matched_free{i}', f'attention{i}')))


def validate_protocol(spec: dict) -> None:
    """Reject scope/budget changes before reaching numerical runtime."""
    expected = dict(task='spin_resonance_continuation', starts=list(STARTS),
        distillation_teacher='additive110', training_steps=60, step_fs_radius=.001,
        step_max_alpha=.5, step_bisections=11, audit_seed=2026101601,
        bootstrap_seed=2026101602, bootstrap_replicates=5000, chains=16, audit_blocks=8,
        feature_batches_per_worker=8, max_feature_workers=16, remote_allowed=False)
    if any(spec.get(k) != v for k,v in expected.items()):
        raise ValueError('not the fixed continuation/audit protocol')
    for key, cap in [('wall_seconds_cap',1800),('peak_rss_gib_cap',4.)]:
        if not np.isfinite(spec[key]) or not 0 < spec[key] <= cap:
            raise ValueError('invalid finite resource cap')


def registration(spec: dict) -> dict:
    """Verify both old experiments and seal all immutable reused numerical data."""
    validate_protocol(spec)
    inputs = {}
    for run_name in ('source_run','distillation_run'):
        source=ROOT/spec[run_name]
        result=json.loads((source/'results.json').read_text())
        unsigned=dict(result);seal=unsigned.pop('result_sha256')
        if canonical_sha256(unsigned) != seal or result['status'] != 'complete':
            raise ValueError('invalid source result seal')
        oldreg=json.loads((source/'registration.json').read_text())
        if file_sha256(source/'registration.json') != result['registration_sha256']:
            raise ValueError('invalid source registration seal')
        for path,sha in oldreg['sources'].items():
            if file_sha256(ROOT/path) != sha: raise ValueError('sealed source changed: '+path)
        for name,sha in result['artifacts'].items():
            if file_sha256(source/name) != sha: raise ValueError('sealed artifact changed: '+name)
        for file in source.glob('*.json'):
            inputs[str(file.relative_to(ROOT))]=file_sha256(file)
        for name in result['artifacts']:
            inputs[str((source/name).relative_to(ROOT))]=file_sha256(source/name)
    registered=dict(config=spec, inputs=inputs, runtime=numerical_runtime_identity(),
        sources=transitive_local_source_manifest((Path(__file__),), root=ROOT,
            package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    registered['registration_sha256']=canonical_sha256(registered)
    return registered


def initial_arms(spec: dict) -> tuple[dict, dict]:
    """Use all three frozen students; matched free table is exactly their G."""
    source=ROOT/spec['source_run']; distilled=ROOT/spec['distillation_run']
    _,_,_,_,_,nn_table,old=load_saved_problem(source)
    arms={
        'continued_additive110':(old['additive110']['parameters'],additive_environment_generators,'initial_additive'),
        'continued_free360':(old['free360']['parameters'],lambda x:x.reshape(6,6,10),'initial_free'),
    }
    initial={'initial_additive':old['additive110'],'initial_free':old['free360']}
    for i,start in enumerate(STARTS):
        name='additive110__'+start+'__physical'
        path=distilled/(name+'.json')
        seal=json.loads((distilled/'endpoints_seal.json').read_text())[name]
        if any(file_sha256(path.with_suffix('.'+ext)) != sha for ext,sha in seal.items()):
            raise ValueError('distilled endpoint changed')
        template=dict(parameters=jnp.zeros(730),generator_table=jnp.zeros((6,6,10)),coefficients=jnp.zeros(1176))
        values=load_parameter_artifact(path,template,expected_identity=dict(fit=name,
            selection='optimizer final accepted iterate',frozen_reference=str(source/'initial.json')))
        if np.max(abs(np.asarray(nn_table(values['parameters']))-np.asarray(values['generator_table']))) > 1e-12:
            raise ValueError('distilled network changed its frozen anchor')
        arms[f'matched_free{i}']=(values['generator_table'].reshape(-1),lambda x:x.reshape(6,6,10),f'initial{i}')
        arms[f'attention{i}']=(values['parameters'],nn_table,f'initial{i}')
        initial[f'initial{i}']=values
    return arms,initial


def _seal_endpoints(output: Path) -> dict:
    names=[*(arm+'_endpoint' for arm in ARMS),'initial_additive','initial_free',*(f'initial{i}' for i in range(3))]
    return {name:{ext:file_sha256(output/(name+'.'+ext)) for ext in ('json','npz')} for name in names}


def require_endpoint_seal(output: Path) -> dict:
    """Never create or open audit samples before all fixed states are sealed."""
    sealed=json.loads((output/'endpoints_seal.json').read_text())
    if _seal_endpoints(output) != sealed: raise ValueError('initial/final checkpoint seal changed')
    return sealed


def primary_audit(psi: np.ndarray, hpsi: np.ndarray, q: np.ndarray, groups: np.ndarray,
                  names: list[str], spec: dict) -> dict:
    """Bootstrap correlated pair contrasts; never bootstrap the three starts."""
    width=len(names);vectors=np.concatenate((psi,hpsi),axis=1)
    estimate=estimate_transition_moments(vectors,vectors[:,None,:],q)
    pairs=[(names.index(f'matched_free{i}'),names.index(f'attention{i}')) for i in range(3)]
    def transform(moment):
        g=moment[0];norms=np.diag(g)[:width].real
        e=np.asarray([g[a,width+a]/norms[a] for a in range(width)])
        v=(np.diag(g)[width:].real/norms-abs(e)**2)/36
        changes=np.asarray([[(e[n].real-e[f].real)/36,v[n]-v[f]] for f,n in pairs])
        return np.concatenate((changes.ravel(),changes.mean(axis=0)))
    labels=[f'pair{i}.{metric}' for i in range(3) for metric in ('energy','variance')]+['mean.energy','mean.variance']
    b=joint_block_bootstrap(estimate,groups,bootstrap_resamples=spec['bootstrap_replicates'],
        seed=spec['bootstrap_seed'],nonlinear_transform=transform,nonlinear_names=labels).nonlinear
    adjusted=np.quantile(b.replicate_values[:,:6],[.05/(2*6),1-.05/(2*6)],axis=0).T
    return dict(metrics={name:dict(value=float(b.point_values[i]),
        chain_bootstrap_se=float(b.standard_errors[i]),chain_bootstrap_95=b.percentile_interval[i].tolist(),
        **({'bonferroni_six_comparisons_interval':adjusted[i].tolist()} if i<6 else {})) for i,name in enumerate(labels)},
        bootstrap_covariance=b.covariance.tolist(),family_size=6,adjusted_individual_confidence=1-.05/6,
        primary='mean.energy',interpretation='joint whole-chain bootstrap; three fixed warmstarts are not independent replicas')


class FeatureBoundary(Exception):
    """A committed worker batch boundary, not a physics failure."""


def run_stage(config: Path, stage: str, *, resume: bool=False) -> dict:
    """One bounded stage in a clean process; resume preserves all committed data."""
    spec=json.loads(config.read_text()); reg=registration(spec);output=ROOT/spec['output']
    output.mkdir(parents=True,exist_ok=True);regpath=output/'registration.json'
    if regpath.exists():
        if not resume: raise FileExistsError('existing attempt; explicit --resume required')
        if json.loads(regpath.read_text()) != reg: raise ValueError('registered sources/config/inputs changed')
    else:
        if stage != 'train' or any(output.iterdir()): raise ValueError('first stage must be a fresh training registration')
        atomic_json(regpath,reg)
    if (output/'external_budget_exhausted.json').exists():
        raise TimeoutError('external process deadline exhausted the registered budget; resume denied')
    if (output/'results.json').exists(): return json.loads((output/'results.json').read_text())
    old_progress=json.loads((output/'progress.json').read_text()) if (output/'progress.json').exists() else {}
    elapsed=float(old_progress.get('seconds',0.)); started=perf_counter()
    if elapsed>=spec['wall_seconds_cap']: raise TimeoutError('cumulative registered cap exhausted')
    def progress(label, **fields):
        record=dict(stage=label,seconds=elapsed+perf_counter()-started,peak_rss_gib=peak_rss_gib(),**fields)
        atomic_json(output/'progress.json',record);print(json.dumps(record),flush=True)
        if record['seconds']>spec['wall_seconds_cap']: raise TimeoutError('cumulative wall cap')
        if record['peak_rss_gib']>spec['peak_rss_gib_cap']: raise MemoryError('post-commit RSS guard')
    def timeout(*_): raise TimeoutError('registered cap; no extension')
    handler=signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(spec['wall_seconds_cap']-elapsed)))
    try:
        source=ROOT/spec['source_run']
        oldspec=json.loads((source/'registration.json').read_text())['config']
        scale=json.loads((source/'scale.json').read_text())['carrier_scale']
        if stage=='train':
            if (output/'endpoints_seal.json').exists():
                require_endpoint_seal(output);return dict(status='training_already_sealed')
            if (output/'audit_samples.npz').exists(): raise ValueError('audit exists before endpoint sealing')
            arms,initial=initial_arms(spec);corpus=load_corpus(source,'train')
            summaries={};initial_match={}
            for name,values in initial.items():
                path=output/(name+'.json')
                identity=dict(arm=name,selection='immutable source initializer')
                if path.exists():
                    restored=load_parameter_artifact(path,values,expected_identity=identity)
                    if any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(restored),jax.tree.leaves(values))):
                        raise ValueError('saved initialization differs from its immutable source')
                else: save_parameter_artifact(path,values,identity=identity)
            training_config=CachedEnergyConfig(steps=60,n_sites=36,target_fs=.001,max_alpha=.5,bisection_steps=11)
            for arm in ARMS:
                x0,table,parent=arms[arm]
                def state(x):
                    t=table(x);return state_from_coefficients(multicontext_feature_coefficients(t,t),corpus)
                p0,h0=state(x0);pt,ht=state_from_coefficients(initial[parent]['coefficients'],corpus)
                mismatch=max(float(jnp.max(abs(p0-pt))),float(jnp.max(abs(h0-ht))))
                if mismatch>1e-10: raise ValueError('initial physical state does not match its saved parent')
                initial_match[arm]=mismatch
                def callback(record):
                    if record['completed_steps']%10==0:progress('training_'+arm,completed=record['completed_steps'])
                checkpoint=output/(arm+'_checkpoint.ckpt')
                fitted=run_cached_energy_continuation(x0,state,corpus['q'],training_config,checkpoint,
                    dict(arm=arm,registration_sha256=file_sha256(regpath),parent=parent),
                    resume=checkpoint.exists(),progress=callback)
                x=fitted.parameters;t=table(x);coef=multicontext_feature_coefficients(t,t)
                values=dict(parameters=x,generator_table=t,coefficients=coef)
                path=output/(arm+'_endpoint.json');identity=dict(arm=arm,completed_steps=60,selection='fixed final step')
                if path.exists():
                    saved=load_parameter_artifact(path,values,expected_identity=identity)
                    if any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(saved),jax.tree.leaves(values))):
                        raise ValueError('completed endpoint differs from checkpoint')
                else:save_parameter_artifact(path,values,identity=identity)
                summaries[arm]=dict(parent=parent,parameter_count=len(x),history=fitted.history,ledger=fitted.ledger,
                    cumulative_optimizer_active_seconds=fitted.active_seconds,
                    train_endpoint_distance=weighted_state_distance(np.asarray(p0),np.asarray(fitted.amplitudes),np.asarray(corpus['q'])))
                atomic_json(output/(arm+'_history.json'),summaries[arm]);progress('sealed_'+arm)
            atomic_json(output/'training.json',dict(arms=summaries,initial_replay_max=initial_match))
            atomic_json(output/'endpoints_seal.json',_seal_endpoints(output));progress('all_initial_and_final_states_sealed')
            return dict(status='training_complete')
        require_endpoint_seal(output)
        if stage=='samples':
            path=output/'audit_samples.npz'
            if path.exists():
                info=json.loads((output/'audit_sampling.json').read_text())
                if file_sha256(path)!=info['sha256']:raise ValueError('audit sample hash mismatch')
                return dict(status='samples_already_saved')
            engine=FixedCarrierFeatures(oldspec)
            data,info=draw(oldspec,36,engine.carrier,engine.bonds,scale,spec['audit_seed'],spec['audit_blocks'])
            _save_npz(path,**data);info.update(sha256=file_sha256(path),seed=spec['audit_seed'],
                endpoint_seal_sha256=file_sha256(output/'endpoints_seal.json'))
            atomic_json(output/'audit_sampling.json',info);progress('new_audit_samples_committed')
            return dict(status='samples_complete')
        sampling=json.loads((output/'audit_sampling.json').read_text())
        if (file_sha256(output/'audit_samples.npz')!=sampling['sha256'] or
            file_sha256(output/'endpoints_seal.json')!=sampling['endpoint_seal_sha256']):
            raise ValueError('audit changed its samples or frozen endpoints')
        with np.load(output/'audit_samples.npz',allow_pickle=False) as s:data={k:s[k].copy() for k in s.files}
        if stage=='features':
            worker_files=list(output.glob('feature_worker_*.json'))
            if len(worker_files)>=spec['max_feature_workers']:raise TimeoutError('feature worker limit reached')
            workerpath=output/f'feature_worker_{len(worker_files):02d}.json'
            atomic_json(workerpath,dict(status='started',seconds_at_start=elapsed))
            engine=FixedCarrierFeatures(oldspec);committed=0
            def batch_progress(label,**fields):
                nonlocal committed
                committed+=1;progress(label,**fields)
                # Let the final batch reach build_corpus's independent saved
                # C/HC replay check instead of yielding just before validation.
                if committed>=spec['feature_batches_per_worker'] and fields['completed']<fields['total']:
                    raise FeatureBoundary()
            try:
                build_corpus(engine,data,scale,output/'audit_features.npz',batch_progress)
                status='complete'
            except FeatureBoundary:status='committed_batch_boundary'
            except MemoryError:
                atomic_json(workerpath,dict(status='post_commit_memory_guard',seconds=perf_counter()-started,
                    committed_batches=committed,peak_rss_gib=peak_rss_gib()))
                raise
            atomic_json(workerpath,dict(status=status,seconds=perf_counter()-started,committed_batches=committed,peak_rss_gib=peak_rss_gib()))
            progress('feature_worker_finished',committed_batches=committed)
            return dict(status=status)
        if stage!='report':raise ValueError('unknown stage')
        corpus=load_corpus(output,'audit')
        training=json.loads((output/'training.json').read_text())
        names=['carrier','initial_additive','initial_free',*(f'initial{i}' for i in range(3)),*ARMS]
        coefficients=[np.zeros(1176)]
        for name in names[1:]:
            final=name in ARMS
            n=training['arms'][name]['parameter_count'] if final else (110 if name=='initial_additive' else 360 if name=='initial_free' else 730)
            template=dict(parameters=jnp.zeros(n),generator_table=jnp.zeros((6,6,10)),coefficients=jnp.zeros(1176))
            values=load_parameter_artifact(output/(name+('_endpoint' if final else '')+'.json'),template)
            coefficients.append(np.asarray(values['coefficients']))
        coefs=np.stack(coefficients,axis=1)
        psi=np.asarray(corpus['carrier'])[:,None]+np.asarray(corpus['features'])@coefs
        hpsi=np.asarray(corpus['hcarrier'])[:,None]+np.asarray(corpus['hfeatures'])@coefs
        _save_npz(output/'audit_values.npz',psi=psi,hpsi=hpsi,coefficients=coefs,q=data['q'],groups=data['groups'],names=np.asarray(names))
        audit=paired_chain_audit(psi,hpsi,data['q'],data['groups'],36,spec['bootstrap_replicates'],spec['bootstrap_seed'],names)
        primary=primary_audit(psi,hpsi,data['q'],data['groups'],names,spec)
        distances={arm:weighted_state_distance(psi[:,names.index(training['arms'][arm]['parent'])],psi[:,names.index(arm)],data['q']) for arm in ARMS}
        require_endpoint_seal(output);progress('audit_complete')
        result=dict(status='complete',config=spec,training=training,audit=audit,primary_audit=primary,
            audit_endpoint_distances=distances,sampling=sampling,fresh_audit=True,
            physical_H_feature_centers=128,additional_cheap_carrier_H_centers=128,
            all_arms_share_full_H_features=True,variational_win_claimed=False,remote_released=False,
            equilibrium_certified=False,seconds=elapsed+perf_counter()-started,
            registration_sha256=file_sha256(regpath),
            artifacts={p.name:file_sha256(p) for p in output.iterdir() if p.suffix in ('.npz','.ckpt')})
        result['result_sha256']=canonical_sha256(result);atomic_json(output/'results.json',result)
        return result
    except Exception as error:
        atomic_json(output/'failure.json',dict(error=repr(error),stage=stage,seconds=elapsed+perf_counter()-started,
            peak_rss_gib=peak_rss_gib(),automatic_budget_extension=False))
        # Preserve cumulative time also when failure is outside a progress callback.
        atomic_json(output/'progress.json',dict(stage='failed_'+stage,seconds=elapsed+perf_counter()-started,peak_rss_gib=peak_rss_gib()))
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,handler);gc.collect()


def orchestrate(config: Path, *, resume: bool=False) -> None:
    """Run isolated workers with a fixed external wall deadline and retry count."""
    spec=json.loads(config.read_text());validate_protocol(spec);output=ROOT/spec['output'];start=perf_counter()
    def exhaust_external_budget(reason):
        output.mkdir(parents=True,exist_ok=True)
        atomic_json(output/'external_budget_exhausted.json',dict(reason=reason,
            wall_seconds_cap=spec['wall_seconds_cap'],automatic_extension=False,
            accounting='fail closed: uncommitted timeout work cannot be recovered from worker progress'))
    def worker(stage,allow_resume):
        remaining=spec['wall_seconds_cap']-(perf_counter()-start)
        if remaining<=0:
            exhaust_external_budget('outer invocation wall deadline');raise TimeoutError('orchestrator wall deadline')
        command=[sys.executable,'-m',MODULE,'--config',str(config.resolve()),'--stage',stage]
        if allow_resume:command.append('--resume')
        try:done=subprocess.run(command,cwd=ROOT,timeout=remaining,check=False)
        except subprocess.TimeoutExpired:
            exhaust_external_budget('subprocess timeout with potentially uncommitted work');raise
        if done.returncode not in (0,75):raise RuntimeError(f'{stage} worker failed ({done.returncode})')
        if done.returncode==75 and stage!='features':raise MemoryError('non-feature worker exceeded memory guard')
    worker('train',resume);worker('samples',True)
    for _ in range(spec['max_feature_workers']):
        path=output/'audit_features.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as f:
                if len(f['carrier'])==128:break
        worker('features',True)
    with np.load(output/'audit_features.npz',allow_pickle=False) as f:
        if len(f['carrier'])!=128:raise TimeoutError('maximum feature workers exhausted before complete corpus')
    worker('report',True)
    atomic_json(output/'orchestration.json',dict(wall_seconds=perf_counter()-start,
        scope='this orchestrator invocation including process launches; cumulative worker cap stored in results/progress'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stage',choices=('train','samples','features','report'))
    args=parser.parse_args()
    try:
        if args.stage:run_stage(args.config,args.stage,resume=args.resume)
        else:orchestrate(args.config,resume=args.resume)
    except MemoryError:
        sys.exit(75)
