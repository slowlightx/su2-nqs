"""Bounded full-H batch timing of an already validated triangular ClebschTree.

This thin wrapper does not change any sealed decoder, projection or H kernel.
Its independent oracle uses combined computational-basis connectivity, retaining
all nonzero matrix elements. Inputs are synthetic, never Born energy samples.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
from time import perf_counter
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.triangle_scale_cost import balanced_block_probe_states
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, make_mixed_projected_amplitude, mixed_parameter_count,
)
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action

ROOT = Path(__file__).resolve().parents[3]


def checked_cost_parent(request: dict, *, root: Path = ROOT) -> tuple[dict, dict]:
    """Load trusted completed cost arrays without resigning their old sources."""
    directory = Path(request['directory'])
    paths = {name: directory/filename for name, filename in (
        ('checkpoint', 'cost_checkpoint.pkl'), ('results', 'results.json'),
        ('registration', 'registration.json'))}
    for name, path in paths.items():
        if file_sha256(path) != request[name+'_sha256']:
            raise ValueError('completed cost '+name+' seal mismatch')
    registration = json.loads(paths['registration'].read_text())
    result = json.loads(paths['results'].read_text())
    digest = result['result_sha256']
    if digest != canonical_sha256({k: v for k, v in result.items() if k != 'result_sha256'}):
        raise ValueError('completed cost result canonical identity mismatch')
    if (result.get('status') != 'completed' or result.get('stage') != 'full'
            or result.get('censored') or not result.get('H_finite')
            or not all(result.get('gradient_finite', {}).get(a, False)
                       for a in ('binary_truncated', 'ternary_joint'))
            or not result.get('checkpoint_replay_exact')
            or not result.get('sampler_RNG_checkpoint_replay_exact')
            or result['config_sha256'] != registration['config_sha256']):
        raise ValueError('completed full-H/gradient/checkpoint gate required')
    for name, expected in registration['sources'].items():
        if file_sha256(root/name) != expected:
            raise ValueError('sealed cost source changed: '+name)
    saved = load_checkpoint(paths['checkpoint'])
    if (saved['metadata'].get('trained') is not False
            or saved['metadata']['source_checkpoint_sha256']
            != registration['config']['parent']['checkpoint_sha256']):
        raise ValueError('completed cost checkpoint identity mismatch')
    for arm in ('binary_truncated', 'ternary_joint'):
        if mixed_parameter_count(saved['params'][arm]) != result['parameter_counts'][arm]:
            raise ValueError('completed cost parameter count mismatch')
    return saved['params'], registration['config']


def relative_vector_error(observed: np.ndarray, reference: np.ndarray) -> float:
    """Scale-free norm discrepancy, retaining genuine exact-zero references."""
    observed, reference = np.asarray(observed), np.asarray(reference)
    if observed.shape != reference.shape:
        raise ValueError('matching oracle/output shapes required')
    # Large-system tensor contractions may have a very small common norm.
    # Rescale BEFORE squaring so an underflow cannot fake a zero discrepancy.
    scale = max(float(np.max(abs(observed), initial=0.)),
                float(np.max(abs(reference), initial=0.)), 1e-300)
    observed, reference = observed/scale, reference/scale
    return float(np.linalg.norm(observed-reference)/max(
        np.linalg.norm(observed), np.linalg.norm(reference), 1e-280))


def make_scalar_connectivity_reference(amplitude: Callable, params: object) -> Callable:
    """One-configuration Sz oracle with amplitude reuse between H shells.

    This oracle is not the timed production H kernel. Its cache only reuses
    identical immutable physical amplitudes; changing J2 changes every matrix
    element through the independent graph, without changing the state.
    """
    evaluate = jax.jit(lambda bits: jax.lax.map(
        lambda row: amplitude(params, row), bits, batch_size=4))
    cache: dict[tuple[int, ...], complex] = {}

    def reference(graph: SzHeisenbergGraph, row: np.ndarray) -> tuple[complex, complex]:
        physical = tuple(int(x) for x in row)
        connected, elements = graph.get_conn(physical)
        missing = tuple(dict.fromkeys(x for x in (physical, *connected) if x not in cache))
        if missing:
            values = np.asarray(evaluate(jnp.asarray(missing, dtype=jnp.int32)))
            if not np.isfinite(values).all():
                raise FloatingPointError('nonfinite independent amplitude oracle')
            cache.update(zip(missing, values, strict=True))
        return cache[physical], np.dot(elements, [cache[x] for x in connected])

    return reference


def collect_accounting(config_path: Path, job_id: str) -> dict:
    """Record even a native-kernel hard kill through an afterany CPU job."""
    if not job_id.isdigit():
        raise ValueError('numeric Slurm job id required')
    spec = json.loads(config_path.read_text())
    output = Path(spec['output'])
    raw = subprocess.check_output([
        '/usr/local/slurm/current/bin/sacct', '-j', job_id, '-X', '-n', '-P',
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'], text=True)
    row = next((x.split('|') for x in raw.splitlines() if x.startswith(job_id+'|')), None)
    if row is None:
        raise RuntimeError('completed-job Slurm accounting unavailable')
    final_path = output/'results.json'
    final = json.loads(final_path.read_text()) if final_path.exists() else {}
    complete = row[1] == 'COMPLETED' and row[2] == '0:0' and final.get('status') == 'completed'
    record = dict(job_id=job_id, Slurm_state=row[1], exit_code=row[2],
        elapsed_seconds=int(row[3]), allocated_resources=row[4],
        status='completed' if complete else 'censored_or_failed',
        completed_cases=len(final.get('cases', [])), last_operation=final.get('current_operation'),
        training_steps=0, energy_estimate=False, accounting_raw=raw)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'accounting.json', record)
    return record


def run(config_path: Path) -> dict:
    """Time unchanged full-H batches; never optimize or infer a Born energy."""
    jax.config.update('jax_enable_x64', True)
    spec = json.loads(config_path.read_text())
    if spec.get('require_gpu') and jax.default_backend() != 'gpu':
        raise RuntimeError('registered batch cost requires an actual GPU')
    if spec['tolerance'] > 2e-11 or spec['tolerance'] <= 0:
        raise ValueError('positive oracle tolerance at most2e-11 required')
    output = Path(spec['output'])
    if output.exists():
        raise FileExistsError('batch cost output is immutable; use a new directory')
    output.mkdir(parents=True)
    start = perf_counter()
    result = dict(status='running', stage='batch_cost', backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()], cases=[], skipped_optional_cases=[],
        new_training_steps=0, energy_estimate=False, equilibrium_claim=False,
        config_sha256=file_sha256(config_path), created_utc=datetime.now(timezone.utc).isoformat())

    def save() -> None:
        result['elapsed_seconds'] = perf_counter()-start
        atomic_json(output/'results.json', result)

    def deadline(signum, frame) -> None:
        result.update(status='censored', censor_reason='termination '+signal.Signals(signum).name)
        save()
        raise SystemExit(2)

    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, deadline)

    def phase(name: str) -> None:
        result['current_operation'] = name
        save()
        if perf_counter()-start >= spec['seconds_cap']:
            result.update(status='censored', censor_reason='registered cap before '+name)
            save()
            raise SystemExit(2)

    sources = transitive_local_source_manifest(
        [Path(__file__), ROOT/'su2_nqs/vmc/local_energy.py'], root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    atomic_json(output/'registration.json', dict(config=spec, sources=sources))
    phase('load_sealed_target_cost_state')
    pair, old_spec = checked_cost_parent(spec['cost_parent'])
    cfg = MixedClebschConfig(**old_spec['model'])
    bank = arity_clebsch_bank(pair['ternary_joint'], cfg, 'ternary_joint', old_spec['binary_rank'])
    other = arity_clebsch_bank(pair['binary_truncated'], cfg, 'binary_truncated', old_spec['binary_rank'])
    result['common_arity_bank_max_error'] = max(float(np.max(abs(np.asarray(a)-np.asarray(b))))
        for a, b in zip(jax.tree.leaves(bank), jax.tree.leaves(other), strict=True))
    if result['common_arity_bank_max_error'] > 2e-11:
        raise ArithmeticError('different arity parents cannot share a batch H timing')
    images = np.asarray(triangular_hnf_gamma_a1_group(((cfg.length, 0), (0, cfg.length))), np.int32)
    projected = make_mixed_projected_amplitude(cfg, images, chunk=old_spec['projection_chunk'])
    states = balanced_block_probe_states(cfg, count=max(c['batch'] for c in spec['cases']), seed=spec['seed'])
    reference = make_scalar_connectivity_reference(projected, bank)
    result.update(model=old_spec['model'], full_A1_images=len(images),
        parameter_counts={a: mixed_parameter_count(p) for a, p in pair.items()},
        initial_state_preserved_from_completed_cost=True,
        probe_distribution='synthetic nine-site block balanced; not Born equilibrium')
    for case in spec['cases']:
        # Optional batch8 never takes time reserved for mandatory J0 batch4.
        if case.get('optional') and spec['seconds_cap']-(perf_counter()-start) < case['minimum_remaining_seconds']:
            result['skipped_optional_cases'].append(dict(case, reason='registered remaining-time rule'))
            save()
            continue
        j2, batch = float(case['j2_over_j1']), int(case['batch'])
        if batch < 1 or not np.isfinite(j2):
            raise ValueError('positive batch and finite coupling required')
        bonds = tuple(triangular_j1j2_bonds(cfg.length, cfg.length, j2=j2, periodic=True))
        graph = SzHeisenbergGraph(cfg.length**2, bonds, n_up=cfg.length**2//2)
        oracle = []
        oracle_start = perf_counter()
        for index, row in enumerate(states[:batch]):
            phase(f'J2={j2}_batch={batch}_scalar_oracle_row={index}')
            oracle.append(reference(graph, row))
        oracle = np.asarray(oracle)
        oracle_seconds = perf_counter()-oracle_start
        action = make_amplitude_heisenberg_action(projected, bonds, n_sites=cfg.length**2)
        inputs = jnp.asarray(states[:batch])
        phase(f'J2={j2}_batch={batch}_full_H_cold')
        tick = perf_counter()
        cold_values = jax.block_until_ready(action(bank, inputs))
        cold = perf_counter()-tick
        phase(f'J2={j2}_batch={batch}_full_H_warm')
        tick = perf_counter()
        values = jax.block_until_ready(action(bank, inputs))
        warm = perf_counter()-tick
        errors = dict(amplitude=relative_vector_error(np.asarray(values[0]), oracle[:, 0]),
            H_action=relative_vector_error(np.asarray(values[1]), oracle[:, 1]),
            cold_warm=relative_vector_error(np.asarray(cold_values), np.asarray(values)))
        if max(errors.values()) > spec['tolerance'] or not np.isfinite(np.asarray(values)).all():
            result.update(status='failed_oracle', failed_case=case, failed_errors=errors)
            save()
            raise ArithmeticError('batch full-H/scalar oracle discrepancy')
        memory = [d.memory_stats() or {} for d in jax.local_devices()]
        result['cases'].append(dict(case, Hamiltonian_bonds=len(bonds),
            first_seconds=cold, warm_seconds=warm, warm_seconds_per_center=warm/batch,
            centers_per_second=batch/warm, independent_scalar_oracle_seconds=oracle_seconds,
            relative_errors=errors, full_H_calls_in_timed_kernel=2*batch,
            cumulative_peak_device_bytes=max(int(m.get('peak_bytes_in_use', 0)) for m in memory),
            allocated_device_bytes=max(int(m.get('bytes_reserved', 0)) for m in memory)))
        save()
        print(json.dumps(result['cases'][-1]), flush=True)
    result.update(status='completed', censored=False)
    save()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--collect-job')
    args = parser.parse_args()
    report = collect_accounting(args.config, args.collect_job) if args.collect_job else run(args.config)
    print(json.dumps(report, indent=2))
