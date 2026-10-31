"""Bounded exact physical raw-TTN sampling preflight for frozen CT12 endpoints."""
from __future__ import annotations
import argparse
from pathlib import Path
import resource
import shutil
import sys
from time import perf_counter, process_time

import jax
jax.config.update('jax_enable_x64', True)
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.triangular_clebsch_active_layout import repack_active_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_amplitude, prepare_feature_lut_bank
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.triangular_raw_born import make_triangular_raw_born_sampler
from .triangle_hierarchy_audit import checked_active_widths
from .triangle_hierarchy_initialization import _node_timeout


def run(parent: Path, output: Path, *, seconds_cap: float = 120.) -> dict:
    """Four exact IID raw draws per endpoint, no coherent-A1/H query or MC chain."""
    if not 10 <= seconds_cap <= 120:
        raise ValueError('registered sampler preflight cap is10..120seconds')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()): raise FileExistsError('fresh immutable preflight directory required')
    paths = [parent/'common_layout/old18_bank.pkl', parent/'matched_radius/initialized_bank.pkl']
    registration = dict(task='exact_physical_raw_TTN_sampler_preflight_v1',
        checkpoints=[str(p.resolve()) for p in paths], checkpoint_sha256=[file_sha256(p) for p in paths],
        workflow_sha256=file_sha256(Path(__file__)), samples_per_endpoint=4,
        seed=2610269003, seconds_cap=seconds_cap, tensor_memory_cap_bytes=768*1024**2,
        all_internal_q_multiplicity_coherences_retained=True, physical_H_queries=0, complete_A1_queries=0)
    atomic_json(output/'registration.json', registration)
    shutil.copy2(__file__, output/'registered_workflow.py')
    start=perf_counter();cpu=process_time();result=dict(registration=registration,status='started',endpoints=[])
    cache={}; rng=np.random.default_rng(registration['seed']); cfg=MixedClebschConfig(12,20,10)
    try:
        with _node_timeout(seconds_cap):
            for path in paths:
                saved=load_checkpoint(path);native=saved['params']
                profile=checked_active_widths(saved['metadata']['embedding']['active_top_widths'],nodes=15)
                bank=repack_active_clebsch_bank(native,cfg,profile)
                tick=perf_counter()
                sampler=make_triangular_raw_born_sampler(bank,cfg,active_top_widths=profile,
                    memory_limit_bytes=registration['tensor_memory_cap_bytes'],shared_tensor_cache=cache)
                setup=perf_counter()-tick
                tick=perf_counter();draws=[sampler.draw(rng) for _ in range(4)];draw_seconds=perf_counter()-tick
                raw=jax.jit(make_feature_lut_amplitude(cfg,active_top_widths=profile))
                prepared=prepare_feature_lut_bank(bank,cfg)
                actual=np.asarray([complex(raw(prepared,d.bits)) for d in draws])
                expected=np.asarray([d.amplitude for d in draws])
                errors=abs(actual-expected)/np.maximum(abs(expected),np.finfo(float).tiny)
                if not np.isfinite(errors).all() or max(errors)>2e-10:
                    raise ArithmeticError('exact sampler failed independent native raw-amplitude replay')
                result['endpoints'].append(dict(checkpoint=str(path),setup_seconds=setup,
                    four_draw_seconds=draw_seconds,seconds_per_draw=draw_seconds/4,
                    maximum_sampler_vs_decoder_relative_error=float(max(errors)),
                    maximum_log_probability_replay_error=max(d.log_probability_replay_error for d in draws),
                    sampler=sampler.diagnostics))
                np.savez(output/f'draws_{len(result["endpoints"])-1}.npz',
                    bits=np.asarray([d.bits for d in draws]),log_probability=np.asarray([d.log_probability for d in draws]),
                    sampler_amplitude=expected,decoder_amplitude=actual)
                atomic_json(output/'progress.json',result)
            result['status']='complete_exact_raw_sampler_preflight'
    except (TimeoutError,MemoryError) as error:
        result.update(status='bounded_sampler_preflight_failed',error=str(error))
    finally:
        result.update(wall_seconds=perf_counter()-start,CPU_seconds=process_time()-cpu,
            peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
        atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds-cap',type=float,default=120.)
    args=parser.parse_args();run(args.parent,args.output,seconds_cap=args.seconds_cap)
