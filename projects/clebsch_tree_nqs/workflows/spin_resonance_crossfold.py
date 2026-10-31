"""Registered cached-only four-fold, one-step direction/FS diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.multicontext_resonance_features import multicontext_feature_coefficients
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.vmc.cached_crossfold import CrossfoldConfig, CachedCrossfoldOracle, whole_chain_folds
from .spin_overlap_attention_cost import _save_npz
from .spin_resonance_continuation import initial_arms, registration as validate_parent_registration
from .spin_resonance_distillation import load_corpus, state_from_coefficients

jax.config.update('jax_enable_x64', True)
ROOT=Path(__file__).resolve().parents[3]
CONFIG=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_resonance_crossfold.json'


def validate_protocol(spec: dict) -> None:
    """No samples, H evaluation, extra directions, or adaptive radius choices."""
    fixed=dict(task='spin_resonance_crossfold',folds=4,starts=[0,1,2],arms=['matched_free','attention'],
        direction='fresh_adam_first_step',adam_epsilon=1e-8,target_fs=.001,normalized_max_alpha=.002,
        bisection_steps=16,relative_fs_tolerance=.01,wall_seconds_cap=120,
        new_H_calls=0,new_samples=0,new_ED_calls=0,remote_allowed=False)
    if any(spec.get(k)!=v for k,v in fixed.items()):raise ValueError('fixed cached-only protocol changed')


def register(spec: dict, config: Path) -> tuple[dict,dict]:
    """Validate source seals and freeze the new protocol before numeric work."""
    validate_protocol(spec);source=ROOT/spec['continuation_run']
    old=json.loads((source/'results.json').read_text());unsigned=dict(old);seal=unsigned.pop('result_sha256')
    if old['status']!='complete' or canonical_sha256(unsigned)!=seal:raise ValueError('invalid continuation seal')
    if file_sha256(source/'registration.json')!=old['registration_sha256']:raise ValueError('source registration changed')
    oldreg=json.loads((source/'registration.json').read_text())
    if validate_parent_registration(old['config'])!=oldreg:raise ValueError('parent provenance changed')
    for name,sha in old['artifacts'].items():
        if file_sha256(source/name)!=sha:raise ValueError('source artifact changed: '+name)
    inputs=dict(oldreg['inputs'])
    for file in ('results.json','registration.json','endpoints_seal.json'):
        inputs[str((source/file).relative_to(ROOT))]=file_sha256(source/file)
    for i in range(3):
        for ext in ('json','npz'):
            p=source/f'initial{i}.{ext}';inputs[str(p.relative_to(ROOT))]=file_sha256(p)
    reg=dict(config=spec,config_sha256=file_sha256(config),inputs=inputs,runtime=numerical_runtime_identity(),
        sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    reg['registration_sha256']=canonical_sha256(reg)
    return reg,old['config']


def run(config: Path=CONFIG) -> dict:
    """Evaluate all 24 independently initialized fold trials, never select one."""
    spec=json.loads(config.read_text());reg,parentspec=register(spec,config);out=ROOT/spec['output']
    if out.exists() and any(out.iterdir()):raise FileExistsError('do not replace or extend a registered attempt')
    out.mkdir(parents=True,exist_ok=True);atomic_json(out/'registration.json',reg)
    started=perf_counter()
    def timeout(*_):raise TimeoutError('fixed numerical wall cap exhausted; no automatic extension')
    prior=signal.signal(signal.SIGALRM,timeout);signal.alarm(spec['wall_seconds_cap'])
    records={};replay={};arrays={}
    try:
        arms,initials=initial_arms(parentspec);corpus=load_corpus(ROOT/parentspec['source_run'],'train')
        with np.load(ROOT/parentspec['source_run']/'train_samples.npz',allow_pickle=False) as data:
            groups=data['groups'].copy()
        folds=whole_chain_folds(groups,spec['folds'])
        if len(groups)!=128 or len(np.unique(groups))!=16 or np.bincount(folds).tolist()!=[32]*4:
            raise ValueError('unexpected sealed whole-chain training corpus')
        cfg=CrossfoldConfig(n_sites=36,target_fs=spec['target_fs'],normalized_max_alpha=spec['normalized_max_alpha'],
            bisection_steps=spec['bisection_steps'],relative_tolerance=spec['relative_fs_tolerance'],adam_epsilon=spec['adam_epsilon'])
        oracles={}
        for family in spec['arms']:
            table=arms[family+'0'][1]
            def state(x,table=table):
                t=table(x);return state_from_coefficients(multicontext_feature_coefficients(t,t),corpus)
            oracles[family]=CachedCrossfoldOracle(state,np.asarray(corpus['q']),cfg)
        for i in spec['starts']:
            matched=[]
            for family in spec['arms']:
                name=family+str(i);x0,_,parent=arms[name]
                p,h=map(np.asarray,oracles[family].state(x0));pt,ht=map(np.asarray,state_from_coefficients(initials[parent]['coefficients'],corpus))
                error=max(float(np.max(abs(p-pt))),float(np.max(abs(h-ht))))
                if error>1e-10:raise ValueError('warmstart replay failure')
                replay[name]=error;matched.append((p,h))
                for fold in range(spec['folds']):
                    key=f'{name}__fold{fold}'
                    row,saved=oracles[family].trial(np.asarray(x0),folds!=fold)
                    row.update(start=i,family=family,fold=fold,fit_chains=np.unique(groups[folds!=fold]).tolist(),
                        held_out_chains=np.unique(groups[folds==fold]).tolist(),parameter_count=len(x0))
                    records[key]=row
                    for k,v in saved.items():arrays[key+'__'+k]=v
                    atomic_json(out/'partial_results.json',dict(records=records,replay=replay,seconds=perf_counter()-started))
                    print(json.dumps(dict(trial=key,matched=row['matched'],held_delta_e=row['held_out']['delta_energy_per_site'],
                        held_delta_var=row['held_out']['delta_variance_per_site'],seconds=perf_counter()-started)),flush=True)
            if max(np.max(abs(a-b)) for a,b in zip(*matched))>1e-10:raise ValueError('NN/free not same physical initial')
        summaries={}
        for family in spec['arms']:
            fs=[]
            for fold in range(spec['folds']):
                rows=[records[f'{family}{i}__fold{fold}'] for i in spec['starts']]
                fs.append(dict(fold=fold,mean_held_energy_slope=float(np.mean([r['held_out']['initial']['energy_slope_per_site'] for r in rows])),
                    mean_held_variance_slope=float(np.mean([r['held_out']['initial']['variance_slope_per_site'] for r in rows])),
                    mean_held_delta_energy=float(np.mean([r['held_out']['delta_energy_per_site'] for r in rows])),
                    mean_held_delta_variance=float(np.mean([r['held_out']['delta_variance_per_site'] for r in rows]))))
            allmatch=all(r['matched'] for r in records.values() if r['family']==family)
            down=sum(x['mean_held_energy_slope']<0 and x['mean_held_delta_energy']<0 for x in fs)
            summaries[family]=dict(folds=fs,all_fit_FS_matched=allmatch,downhill_held_folds=down,
                diagnostic_success=allmatch and down>=3)
        _save_npz(out/'trials.npz',**arrays,q=np.asarray(corpus['q']),groups=groups,folds=folds)
        result=dict(status='complete',config=spec,optimizer=asdict(cfg),records=records,initial_replay_max=replay,
            summaries=summaries,seconds=perf_counter()-started,new_H_queries=0,new_samples=0,new_ED_calls=0,
            independent_evidence=False,variational_win_claimed=False,remote_released=False,
            caveat='Frozen parents and all source records were previously fitted/viewed. Conditional diagnostic, not unbiased new validation.',
            registration_sha256=file_sha256(out/'registration.json'),artifacts={'trials.npz':file_sha256(out/'trials.npz')})
        result['result_sha256']=canonical_sha256(result);atomic_json(out/'results.json',result)
        return result
    except Exception as error:
        atomic_json(out/'failure.json',dict(error=repr(error),seconds=perf_counter()-started,completed_trials=len(records)))
        raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,prior)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=CONFIG)
    run(parser.parse_args().config)
