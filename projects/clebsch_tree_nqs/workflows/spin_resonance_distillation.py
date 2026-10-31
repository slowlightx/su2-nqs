"""Bounded teacher reproduction using already sealed C/HC/F/HF only.

There is deliberately no sampler, carrier evaluator or physical H constructor
in this workflow. Saved full-H features are linear data, not new H calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import signal
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize

from su2_nqs.benchmarks.multicontext_resonance_features import multicontext_feature_coefficients
from su2_nqs.benchmarks.resonance_attention_capacity import head_rank_certificate
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact, save_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.multicontext_resonance_attention import (
    MulticontextResonanceAttentionConfig, init_multicontext_resonance_attention,
    make_multicontext_reference, multicontext_resonance_generators,
)
from su2_nqs.vmc.fixed_proposal_rayleigh import weighted_full_h_moments
from su2_nqs.vmc.resonance_distillation import (
    gauge_invariant_table_loss, table_distillation_metrics, physical_distillation_metrics,
)

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_resonance_distillation.json'


def validate_protocol(spec: dict) -> None:
    """Reject changing the scientific question or silently adding physics work."""
    expected = dict(task='spin_resonance_distillation', teachers=['additive110', 'free360'],
        starts=['old_endpoint', 'original_initialization', 'independent_initialization'],
        independent_initialization_seed=2026101501, max_iterations_per_fit=1000,
        max_function_evaluations_per_fit=1500, physical_loss_weights=[.5, .5], fit_count=14,
        new_physical_H_calls=0, new_carrier_queries=0, new_samples=0, remote_allowed=False)
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('not the bounded saved-feature reproduction protocol')
    for key, limit in [('wall_seconds_cap', 900), ('peak_rss_gib_cap', 4.)]:
        if not np.isfinite(spec[key]) or not 0 < spec[key] <= limit:
            raise ValueError('invalid bounded resource cap')


def registration(spec: dict) -> dict:
    """Seal inputs and verify the parent without rewriting any old artifacts."""
    validate_protocol(spec)
    source = ROOT/spec['source_run']
    parent = json.loads((source/'results.json').read_text())
    unsigned = dict(parent); seal = unsigned.pop('result_sha256')
    if canonical_sha256(unsigned) != seal or parent['status'] != 'complete':
        raise ValueError('invalid parent result seal')
    for name, sha in parent['artifacts'].items():
        if file_sha256(source/name) != sha: raise ValueError('parent artifact changed: '+name)
    oldreg = json.loads((source/'registration.json').read_text())
    if file_sha256(source/'registration.json') != parent['registration_sha256']:
        raise ValueError('parent registration changed')
    for name, sha in oldreg['sources'].items():
        if file_sha256(ROOT/name) != sha: raise ValueError('parent numerical source changed: '+name)
    files = ['results.json', 'registration.json', 'initial.json', 'initial.npz',
             'endpoints_seal.json', 'train_features.npz', 'audit_features.npz',
             'train_samples.npz', 'audit_samples.npz']
    files += [arm+'_endpoint.'+ext for arm in [*spec['teachers'], 'attention730'] for ext in ['json', 'npz']]
    result = dict(config=spec, runtime=numerical_runtime_identity(),
        inputs={str((source/name).relative_to(ROOT)):file_sha256(source/name) for name in files},
        sources=transitive_local_source_manifest((Path(__file__),), root=ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    result['registration_sha256'] = canonical_sha256(result)
    return result


def load_saved_problem(source: Path):
    """Validate portable teacher artifacts and retain the original fixed anchor."""
    old = json.loads((source/'registration.json').read_text())['config']
    cfg = MulticontextResonanceAttentionConfig()
    positions = jnp.asarray(old['positions'])
    initial = init_multicontext_resonance_attention(jax.random.PRNGKey(old['initialization_seed']), cfg)
    reference = make_multicontext_reference(initial, positions, cfg)
    restored = load_parameter_artifact(source/'initial.json', {'attention':initial, 'reference':reference},
        expected_identity={'seed':old['initialization_seed'], 'same_physical_state':'G=0 carrier'})
    initial, reference = restored['attention'], restored['reference']
    flat, unravel = ravel_pytree(initial)
    endpoints = {}
    seals = json.loads((source/'endpoints_seal.json').read_text())
    for arm, n in [('additive110',110), ('free360',360), ('attention730',730)]:
        for ext in ['json', 'npz']:
            if file_sha256(source/(arm+'_endpoint.'+ext)) != seals[arm][ext]:
                raise ValueError('teacher endpoint seal changed')
        template = dict(parameters=jnp.zeros(n), generator_table=jnp.zeros((6,6,10)), coefficients=jnp.zeros(1176))
        endpoints[arm] = load_parameter_artifact(source/(arm+'_endpoint.json'), template,
            expected_identity=dict(arm=arm, completed_steps=60, selection='fixed final step'))
    table = jax.jit(lambda x:multicontext_resonance_generators(unravel(x), reference, positions, cfg))
    return cfg, positions, initial, reference, flat, table, endpoints


def load_corpus(source: Path, phase: str) -> dict:
    """Use sealed features in total-H units, checking sample/feature alignment."""
    with np.load(source/(phase+'_features.npz'), allow_pickle=False) as f, np.load(source/(phase+'_samples.npz'), allow_pickle=False) as s:
        seal = canonical_sha256({k:s[k].tolist() for k in ('states','q','groups')})
        if str(f['sample_seal']) != seal: raise ValueError('feature/sample alignment failed')
        result = {k:jnp.asarray(f[k]) for k in ['carrier','hcarrier','features','hfeatures']}
        result['q'] = jnp.asarray(s['q'])
    if result['features'].shape != (128,1176) or not np.all(np.asarray(result['q']) > 0):
        raise ValueError('unexpected corpus shape or invalid positive proposal')
    return result


def state_from_coefficients(coef: jax.Array, corpus: dict):
    """Reconstruct amplitudes and total-H action without querying physics."""
    return (corpus['carrier']+corpus['features']@coef,
            corpus['hcarrier']+corpus['hfeatures']@coef)


def _json_metrics(metrics) -> dict:
    return {key:np.asarray(value).item() for key,value in metrics._asdict().items()}


def run(config: Path = CONFIG) -> dict:
    """Run all registered fits; evaluate seen transfer only after endpoint sealing."""
    spec = json.loads(config.read_text()); registered = registration(spec)
    output = ROOT/spec['output']
    if output.exists() and any(output.iterdir()): raise FileExistsError('refusing to replace an existing attempt')
    output.mkdir(parents=True, exist_ok=True); atomic_json(output/'registration.json', registered)
    started = perf_counter()
    def progress(stage, **fields):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform == 'darwin' else 1024**2)
        record = dict(stage=stage, seconds=perf_counter()-started, peak_rss_gib=rss, **fields)
        atomic_json(output/'progress.json', record); print(json.dumps(record), flush=True)
        if record['seconds'] > spec['wall_seconds_cap']: raise TimeoutError('registered wall cap')
        if rss > spec['peak_rss_gib_cap']: raise MemoryError('registered RSS cap')
    def timeout(*_): raise TimeoutError('registered wall cap; no extension')
    previous = signal.signal(signal.SIGALRM, timeout); signal.alarm(int(spec['wall_seconds_cap']))
    try:
        source = ROOT/spec['source_run']
        cfg, positions, initial, reference, initial_flat, table, teachers = load_saved_problem(source)
        fresh = init_multicontext_resonance_attention(jax.random.PRNGKey(spec['independent_initialization_seed']), cfg)
        starts = dict(old_endpoint=teachers['attention730']['parameters'],
            original_initialization=initial_flat, independent_initialization=ravel_pytree(fresh)[0])
        train = load_corpus(source, 'train')
        coef_fn = jax.jit(lambda x:multicontext_feature_coefficients(table(x), table(x)))
        train_state = jax.jit(lambda x:state_from_coefficients(coef_fn(x), train))
        records = {}; endpoints = {}; certificates = {}
        replay = float(np.max(abs(np.asarray(table(starts['old_endpoint']))-np.asarray(teachers['attention730']['generator_table']))))
        if replay > 1e-12: raise ValueError('saved NN endpoint does not replay')
        progress('validated_saved_inputs', neural_table_replay_max=replay)

        def fit(name, x0, objective):
            vg = jax.jit(jax.value_and_grad(objective)); hist = []; calls = 0; tick = perf_counter()
            latest = {}
            def fun(x):
                nonlocal calls
                if calls >= spec['max_function_evaluations_per_fit']:
                    # scipy normally stops at an iteration boundary; a strict
                    # wrapper cap below retains the last accepted iterate.
                    raise StopIteration('strict function budget')
                loss, grad = vg(jnp.asarray(x)); loss=float(loss); grad=np.asarray(grad)
                calls += 1
                if not np.isfinite(loss) or not np.isfinite(grad).all():
                    raise FloatingPointError('invalid chart or nonfinite fitting objective')
                latest.update(loss=loss, gradient_norm=float(np.linalg.norm(grad)))
                return loss, grad
            accepted = np.asarray(x0).copy()
            initial_loss, initial_gradient = fun(accepted)
            def callback(x):
                nonlocal accepted
                accepted = np.asarray(x).copy()
                hist.append(dict(iteration=len(hist)+1, function_calls=calls, **latest))
                if len(hist)%100 == 0:
                    atomic_json(output/(name+'_progress.json'), dict(history=hist, parameters=accepted.tolist()))
                    progress(name, iteration=len(hist), function_calls=calls)
            try:
                opt = minimize(fun, accepted, jac=True, method='L-BFGS-B', callback=callback,
                    options=dict(maxiter=spec['max_iterations_per_fit'], maxfun=spec['max_function_evaluations_per_fit'],
                        maxls=spec['max_line_search_steps'], maxcor=spec['lbfgs_memory'], ftol=spec['ftol'], gtol=spec['gtol']))
                x = opt.x; message=str(opt.message); success=bool(opt.success)
            except StopIteration:
                x=accepted; message='strict maximum function evaluations reached'; success=False
            final_loss, final_grad = vg(jnp.asarray(x))
            t=table(x); coef=coef_fn(x)
            save_parameter_artifact(output/(name+'.json'), dict(parameters=jnp.asarray(x), generator_table=t, coefficients=coef),
                identity=dict(fit=name, selection='optimizer final accepted iterate', frozen_reference=str(source/'initial.json')))
            records[name] = dict(initial_loss=initial_loss, initial_gradient_norm=float(np.linalg.norm(initial_gradient)),
                final_loss=float(final_loss), final_gradient_norm=float(jnp.linalg.norm(final_grad)),
                iterations=len(hist), optimizer_function_evaluations=calls, final_diagnostic_evaluations=1,
                termination=message, optimizer_success=success, seconds=perf_counter()-tick, history=hist)
            endpoints[name] = dict(parameters=np.asarray(x), generator_table=np.asarray(t), coefficients=np.asarray(coef))
            atomic_json(output/(name+'_history.json'), records[name]); progress(name+'_complete', loss=float(final_loss))
            return jnp.asarray(x)

        for teacher_name in spec['teachers']:
            teacher = teachers[teacher_name]; desired = teacher['generator_table']
            certificates[teacher_name] = head_rank_certificate(np.asarray(reference), np.asarray(desired), head_width=8)
            target, htarget = state_from_coefficients(teacher['coefficients'], train)
            def table_loss(x): return gauge_invariant_table_loss(table(x), desired)
            def physical_loss(x):
                psi,hpsi = train_state(x)
                m=physical_distillation_metrics(psi,hpsi,target,htarget,train['carrier'],train['hcarrier'],train['q'])
                return .5*m.innovation_relative_mse+.5*m.h_response_relative_mse
            for start_name in spec['starts']:
                name=teacher_name+'__'+start_name
                x = fit(name+'__table', starts[start_name], table_loss)
                fit(name+'__physical', x, physical_loss)
            fit(teacher_name+'__direct_physical', starts['old_endpoint'], physical_loss)
        endpoint_seals = {name:{ext:file_sha256(output/(name+'.'+ext)) for ext in ['json','npz']} for name in records}
        atomic_json(output/'endpoints_seal.json', endpoint_seals)
        progress('all_fourteen_endpoints_sealed_before_transfer')
        corpora = dict(train=train, seen_transfer=load_corpus(source,'audit'))
        evaluations={}
        for teacher_name in spec['teachers']:
            teacher=teachers[teacher_name]; evaluations[teacher_name]={}
            candidates={'teacher':teacher, 'old_attention':teachers['attention730'],
                **{name:values for name,values in endpoints.items() if name.startswith(teacher_name+'__')}}
            for name, values in candidates.items():
                item=dict(table=_json_metrics(table_distillation_metrics(values['generator_table'],teacher['generator_table'])))
                for phase,corpus in corpora.items():
                    p,h=state_from_coefficients(values['coefficients'],corpus)
                    t,ht=state_from_coefficients(teacher['coefficients'],corpus)
                    m=physical_distillation_metrics(p,h,t,ht,corpus['carrier'],corpus['hcarrier'],corpus['q'])
                    moments=weighted_full_h_moments(p,h,corpus['q'])
                    item[phase]=dict(**_json_metrics(m),energy_per_site=float(moments.energy_complex.real/36),
                        variance_per_site=float(moments.variance/36))
                evaluations[teacher_name][name]=item
        for name, seal in endpoint_seals.items():
            if any(file_sha256(output/(name+'.'+ext)) != sha for ext,sha in seal.items()):
                raise ValueError('endpoint changed during transfer reporting')
        result=dict(status='complete', config=spec, fits=records, evaluations=evaluations,
            rank_certificates=certificates, neural_table_replay_max=replay,
            new_physical_H_calls=0,new_carrier_queries=0,new_samples=0,
            reused_train_records=128,reused_seen_transfer_records=128,
            fresh_audit=False, variational_win_claimed=False, remote_released=False,
            seconds=perf_counter()-started, registration_sha256=file_sha256(output/'registration.json'),
            artifacts={p.name:file_sha256(p) for p in output.glob('*.npz')})
        result['result_sha256']=canonical_sha256(result)
        atomic_json(output/'results.json',result);progress('complete')
        return result
    except Exception as error:
        atomic_json(output/'failure.json',dict(error=repr(error),seconds=perf_counter()-started,automatic_extension=False))
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    run(parser.parse_args().config)
