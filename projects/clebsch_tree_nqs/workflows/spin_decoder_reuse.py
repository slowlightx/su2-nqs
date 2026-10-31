"""Same N36 physical state, exact query reuse, gradient and proposal costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.benchmarks.spatial_vit_sampling import make_full_heisenberg_ratios, make_log_exchange_sampler
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest, numerical_runtime_identity
from su2_nqs.vmc.projected_spin_work import (
    make_chunked_a1_amplitudes, make_deduplicated_a1_heisenberg, make_frozen_vmc_force,
)
from .spin import _groups, _bonds
from .spin_decoder_cost import ROOT, physical_config, amplitude_functions, peak_rss_bytes
from .spin_vap_decoder import init_decoder

jax.config.update('jax_enable_x64', True)
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_decoder_reuse.json'


def registration(spec: dict) -> dict:
    """Bind the new evaluator to immutable parent states and parameter artifacts."""
    parent = ROOT/spec['parent']
    paths = [parent/'registration.json']
    for case in spec['cases']:
        paths += [parent/case['geometry']/name for name in
                  ('parameters.json', 'parameters.npz', 'values.npz', 'results.json')]
    record = dict(config=spec, inputs={str(p.relative_to(ROOT)):file_sha256(p) for p in paths},
        runtime=numerical_runtime_identity(), sources=transitive_local_source_manifest((Path(__file__),),
            root=ROOT, package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')))
    record['registration_sha256'] = canonical_sha256(record)
    return record


def measure(fn, args: tuple, repeats: int) -> tuple[dict, object, object]:
    """Block device output, without host-copying large dedup plans into timings."""
    before = peak_rss_bytes(); start = perf_counter()
    lowered = jax.jit(fn).lower(*args); lowering = perf_counter()-start
    start = perf_counter(); executable = lowered.compile(); compilation = perf_counter()-start
    memory = executable.memory_analysis()
    fields = ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes',
              'alias_size_in_bytes','generated_code_size_in_bytes')
    estimates = {k:int(getattr(memory,k)) for k in fields if memory is not None and hasattr(memory,k)}
    start = perf_counter(); first = jax.block_until_ready(executable(*args)); initial = perf_counter()-start
    times = []
    for _ in range(repeats):
        start = perf_counter(); last = jax.block_until_ready(executable(*args)); times.append(perf_counter()-start)
    first_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(first)]
    last_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(last)]
    replay = max(float(np.max(abs(a.astype(complex)-b.astype(complex)))) for a,b in zip(first_leaves,last_leaves))
    return dict(lowering_seconds=lowering, compile_seconds=compilation, first_seconds=initial,
        execution_seconds=times, finite=all(np.isfinite(x).all() for x in first_leaves), replay_max=replay,
        compiler_memory_estimates=estimates, peak_rss_before_bytes=before,
        peak_rss_after_bytes=peak_rss_bytes()), executable, first


def scaled_error(actual, reference) -> float:
    return float(np.max(abs(np.asarray(actual)-np.asarray(reference))/(1+abs(np.asarray(reference)))))


def run(path: Path, geometry: str) -> dict:
    """Run a bounded serial cost/replay test, never optimizer steps."""
    spec = json.loads(path.read_text()); output = ROOT/spec['output']
    reg = json.loads((output/'registration.json').read_text())
    if reg != registration(spec): raise ValueError('registration mismatch')
    case = next(c for c in spec['cases'] if c['geometry']==geometry)
    destination = output/geometry; parent = ROOT/spec['parent']/geometry
    if (destination/'progress.json').exists(): raise FileExistsError(destination)
    start = perf_counter(); measurements = {}; validation = {}; values = {}
    def progress(status):
        atomic_json(destination/'progress.json', dict(status=status, geometry=geometry,
            registration_sha256=reg['registration_sha256'], measurements=measurements, validation=validation,
            elapsed_seconds=perf_counter()-start, peak_rss_bytes=peak_rss_bytes()))
        print(geometry, status, round(perf_counter()-start,3), 's',
              round(peak_rss_bytes()/2**30,3), 'GiB', flush=True)
        if peak_rss_bytes()>spec['peak_rss_limit_gib']*2**30: raise MemoryError('registered RSS cap')
    def alarm(_signum,_frame): raise TimeoutError('registered wall cap')
    previous = signal.signal(signal.SIGALRM,alarm); signal.alarm(spec['wall_limit_seconds_per_process'])
    try:
        progress('started'); config = physical_config(6,6,2)
        params = load_parameter_artifact(parent/'parameters.json',init_decoder(2026093003,config))
        parameter_hash = pytree_sha256(params)
        with np.load(parent/'values.npz') as a:
            states = jnp.asarray(a['states']); parent_amp=a['a1_amplitude_b8']; parent_H=a['a1_H_b8']
        old = amplitude_functions(config,_groups(geometry,6,6)['gamma_a1'])
        bonds = _bonds(geometry,6,6,case['j2_over_j1'])
        old_log = lambda p,x:jnp.log(old['a1'](p,x))
        old_H = make_full_heisenberg_ratios(old_log,bonds)
        def timed(name,fn,args):
            progress(name+'_begin')
            record,exe,value = measure(fn,args,spec['timing_repeats'])
            measurements[name]=record;progress(name+'_done');return exe,value
        exe, reference = timed('old_H',old_H,(params,states)); values['old_H']=np.asarray(reference)
        del exe
        plan,evaluate,action = make_deduplicated_a1_heisenberg(old['raw'],old['projector'],bonds,
                                                             chunk_size=spec['chunk_size'])
        exe,prepared = timed('plan',plan,(states,));del exe
        exe,cached = timed('cached_H',evaluate,(params,prepared));del exe
        exe,actual = timed('end_to_end_H',action,(params,states));del exe
        ratios,amp,unique_count,evaluated_count = actual
        values.update(states=np.asarray(states),H=np.asarray(ratios),amplitudes=np.asarray(amp),
                      unique_count=np.asarray(unique_count),evaluated_count=np.asarray(evaluated_count))
        validation['H_old_scaled_error']=scaled_error(ratios,reference)
        validation['H_parent_scaled_error']=scaled_error(ratios,parent_H)
        validation['H_cached_scaled_error']=scaled_error(ratios,cached[0])
        validation['amplitude_relative_error']=float(np.linalg.norm(np.asarray(amp)-parent_amp)/np.linalg.norm(parent_amp))
        del prepared
        chunked = make_chunked_a1_amplitudes(old['raw'],old['projector'],chunk_size=spec['chunk_size'])
        amp_exe,chunked_values = timed('chunked_amplitude',chunked,(params,states))
        validation['chunked_amplitude_relative_error']=float(np.linalg.norm(np.asarray(chunked_values)-parent_amp)/np.linalg.norm(parent_amp))
        force = make_frozen_vmc_force(chunked,jnp.asarray(parent_H),36)
        force_fn = lambda p,x:ravel_pytree(force(p,x))[0]
        exe,gradient = timed('score_force',force_fn,(params,states));del exe
        values['gradient']=np.asarray(gradient)
        flat,unravel = ravel_pytree(params)
        direction=jnp.sin(jnp.arange(len(flat))+.3);direction/=jnp.linalg.norm(direction)
        epsilon=1e-6;centered=parent_H-np.mean(parent_H)
        def frozen_loss(theta):
            logs=np.log(np.asarray(amp_exe(unravel(theta),states)))
            return 2*np.real(np.mean(np.conj(logs)*centered))/36
        fd=(frozen_loss(flat+epsilon*direction)-frozen_loss(flat-epsilon*direction))/(2*epsilon)
        analytic=float(jnp.dot(gradient,direction))
        validation['force_direction_scaled_error']=abs(fd-analytic)/(1+abs(fd))
        validation['force_direction_analytic']=analytic;validation['force_direction_finite_difference']=fd
        values['gradient_direction']=np.asarray(direction)
        # Same proposal algorithm/key; only amplitude implementation changes.
        key=jax.random.PRNGKey(spec['sampler_seed'])
        new_log=lambda p,x:jnp.log(chunked(p,x[None])[0])
        for proposals in spec['proposal_counts']:
            sampler=make_log_exchange_sampler(new_log,proposals)
            exe,sampled=timed(f'sample_{proposals}',sampler,(params,key,states));del exe
            old_sampler=jax.jit(make_log_exchange_sampler(old_log,proposals))
            before=perf_counter();old_sampled=jax.block_until_ready(old_sampler(params,key,states))
            validation[f'sample_{proposals}_reference_wall_seconds']=perf_counter()-before
            validation[f'sample_{proposals}_identical']=all(np.array_equal(a,b)for a,b in zip(sampled,old_sampled))
            final,finalkey,accept,moves=sampled
            values[f'sample_{proposals}_states']=np.asarray(final)
            values[f'sample_{proposals}_key']=np.asarray(finalkey)
            values[f'sample_{proposals}_accept_move']=np.asarray([accept,moves])
            validation[f'sample_{proposals}_sector_preserved']=bool(np.all(np.asarray(final).sum(axis=1)==18))
            del old_sampler
        validation['parameters_unchanged']=parameter_hash==pytree_sha256(params)
        passed=all(m['finite']for m in measurements.values()) and validation['parameters_unchanged']
        for k,v in validation.items():
            if k.endswith('error'): passed=passed and np.isfinite(v) and v<(1e-6 if k.startswith('force') else 1e-9)
            if k.endswith(('identical','preserved')): passed=passed and v
        np.savez_compressed(destination/'values.npz',**values)
        result=dict(geometry=geometry,case=case,registration_sha256=reg['registration_sha256'],
            parameters_sha256=parameter_hash,measurements=measurements,validation=validation,
            raw_values_sha256=file_sha256(destination/'values.npz'),passed=bool(passed),
            raw_query_count=8*(len(bonds)+1)*old['projector'].group_order,
            unique_count=int(unique_count),evaluated_count=int(evaluated_count),
            spatial_images=old['projector'].group_order,chunk_size=spec['chunk_size'],
            parameter_count=len(flat),backend=jax.default_backend(),peak_rss_bytes=peak_rss_bytes(),
            wall_seconds=perf_counter()-start,training_updates=0,new_ED=False,remote_submitted=False,
            equilibrium_sampling_claim=False,proposal_counts=spec['proposal_counts'])
        result['result_sha256']=canonical_sha256(result);atomic_json(destination/'results.json',result)
        progress('complete'if passed else'validation_failed');return result
    except Exception as error:
        atomic_json(destination/'failure.json',dict(type=type(error).__name__,message=str(error),
            measurements=measurements,validation=validation,elapsed_seconds=perf_counter()-start))
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--geometry',choices=('triangular','square'))
    args=parser.parse_args();spec=json.loads(args.config.read_text())
    if args.register:
        target=ROOT/spec['output']/'registration.json'
        if target.exists():raise FileExistsError(target)
        r=registration(spec);atomic_json(target,r);print(r['registration_sha256'])
    else:run(args.config,args.geometry)


if __name__=='__main__':main()
