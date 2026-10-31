"""Bounded accelerator-only SR-kernel memory test; no wavefunction training."""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.vmc.parameter_block_sr import solve_parameter_block_sr


def run(output: Path, *, parameters: int = 7_856_880, samples: int = 256) -> dict:
    """Exercise the actual N144 score dimensions, with bounded analytic input.

    This is a kernel memory certificate only. The complete CT preparation,
    full H, gradient-score generation, and checkpoint path need their own
    actual-state pilot and may have a higher peak.
    """
    jax.config.update('jax_enable_x64', True)
    if jax.default_backend() != 'gpu':
        raise RuntimeError('actual GPU memory measurement required')
    if output.exists(): raise FileExistsError(output)
    started = perf_counter()
    @jax.jit
    def build():
        rows = jnp.arange(samples*2, dtype=jnp.float64).reshape(samples, 2, 1)
        columns = jnp.arange(parameters, dtype=jnp.float64)[None, None, :]
        return (jnp.sin(rows*.17)*jnp.cos(columns*.003)
                +jnp.cos(rows*.11)*jnp.sin(columns*.005)
                +jnp.sin(rows*.07)*jnp.cos(columns*.009))/parameters**.5
    scores = jax.block_until_ready(build())
    force = jnp.sin(jnp.arange(2*samples, dtype=jnp.float64)*.27)/samples**.5
    function = jax.jit(partial(solve_parameter_block_sr, damping=.1,
        score_scale=(2/samples)**.5, parameter_chunk_size=8192))
    lowered = function.lower(scores, force).compile()
    memory = lowered.memory_analysis()
    tick = perf_counter()
    first = jax.block_until_ready(lowered(scores, force)); cold = perf_counter()-tick
    tick = perf_counter()
    second = jax.block_until_ready(lowered(scores, force)); warm = perf_counter()-tick
    error = float(jnp.linalg.norm(first[0]-second[0])/jnp.maximum(jnp.linalg.norm(first[0]),1e-30))
    stats = [d.memory_stats() or {} for d in jax.local_devices()]
    peak = max(s.get('peak_bytes_in_use',0) for s in stats)
    result = dict(status='passed' if bool(jnp.isfinite(second[0]).all())
        and float(second[2]['relative_linear_residual']) < 1e-7 and error <= 2e-11
        and 0 < peak < 70*1024**3 else 'failed',
        actual_gpu=[d.device_kind for d in jax.local_devices()],
        score_shape=list(scores.shape),raw_score_bytes=int(scores.size*scores.dtype.itemsize),
        parameter_chunk_size=8192,peak_device_bytes=peak,
        compiler_temporary_bytes=int(memory.temp_size_in_bytes),
        compiler_output_bytes=int(memory.output_size_in_bytes),
        first_execution_seconds=cold,warm_execution_seconds=warm,
        residual=float(second[2]['relative_linear_residual']),repeat_relative_error=error,
        elapsed_seconds=perf_counter()-started,full_H_calls=0,training_updates=0,
        actual_CT_training_memory_certified=False)
    atomic_json(output,result)
    if result['status'] != 'passed': raise RuntimeError(json.dumps(result))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.output)),flush=True)
