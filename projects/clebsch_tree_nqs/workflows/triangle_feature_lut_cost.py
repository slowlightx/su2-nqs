"""One bounded, same-state legacy/LUT cost comparison; no training or energy.

Reuse the completed N144 cost checkpoint and the independent legacy kernel.
Fresh LUT preparation is included in every timed H call and in parameter AD.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.triangle_scale_batch_cost import (
    checked_cost_parent, collect_accounting, relative_vector_error,
)
from projects.clebsch_tree_nqs.workflows.triangle_scale_cost import balanced_block_probe_states
from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import (
    prepare_feature_lut_bank, make_feature_lut_projected_amplitude,
)
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, make_mixed_projected_amplitude
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action

ROOT = Path(__file__).resolve().parents[3]


def run(config_path: Path) -> dict:
    """Keep all1728 A1 images/all864 H bonds even if the time cap censors3 pairs."""
    jax.config.update('jax_enable_x64', True)
    spec = json.loads(config_path.read_text())
    if spec['require_gpu'] and jax.default_backend() != 'gpu':
        raise RuntimeError('actual GPU required by frozen cost comparison')
    if not 0 < spec['tolerance'] <= 2e-11 or spec['warm_repeats'] != 3:
        raise ValueError('registered2e-11 oracle and three warm pairs required')
    output = Path(spec['output'])
    if output.exists():
        raise FileExistsError('cost outputs are immutable')
    output.mkdir(parents=True)
    start = perf_counter()
    result = dict(status='running', cases=[], measurements={}, relative_errors={},
        new_training_steps=0, energy_estimate=False, equilibrium_claim=False,
        backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
        config_sha256=file_sha256(config_path), preparation_included_in_new_H_and_AD=True)

    def save() -> None:
        result['elapsed_seconds'] = perf_counter()-start
        result['device_memory_cumulative_highwater'] = [dict(
            (k, int(v)) for k, v in (d.memory_stats() or {}).items()
            if k in ('bytes_in_use', 'peak_bytes_in_use', 'bytes_limit', 'bytes_reserved'))
            for d in jax.local_devices()]
        atomic_json(output/'results.json', result)

    def censor(reason: str) -> None:
        signal.alarm(0)
        result.update(status='censored', censored=True, censor_reason=reason)
        save()

    def deadline(signum, frame) -> None:
        censor('signal '+signal.Signals(signum).name)
        raise SystemExit(2)

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, deadline)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(int(spec['seconds_cap']))

    def phase(name: str) -> None:
        result['current_operation'] = name
        save()
        if result['elapsed_seconds'] >= spec['seconds_cap']:
            censor('registered13-minute cap before '+name)
            raise SystemExit(2)

    def cold(name, fn, *args):
        """Separate lowering, compilation, and synchronized first execution."""
        phase(name+'_lower_compile_first')
        tick = perf_counter(); lowered = jax.jit(fn).lower(*args); lower = perf_counter()-tick
        tick = perf_counter(); executable = lowered.compile(); compile_seconds = perf_counter()-tick
        tick = perf_counter(); value = jax.block_until_ready(executable(*args)); first = perf_counter()-tick
        memory = executable.memory_analysis()
        result['measurements'][name] = dict(lowering_seconds=lower, compilation_seconds=compile_seconds,
            first_execution_seconds=first, compiler_memory_estimates_not_device_highwater={
                k: int(getattr(memory, k)) for k in (
                    'argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes')})
        save()
        return executable, value

    sources = transitive_local_source_manifest([Path(__file__), ROOT/'su2_nqs/vmc/local_energy.py'],
        root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    atomic_json(output/'registration.json', dict(config=spec, sources=sources))
    phase('sealed_parent_and_exact_geometry')
    pair, old_spec = checked_cost_parent(spec['cost_parent'])
    cfg = MixedClebschConfig(**old_spec['model'])
    if cfg.length != 12 or old_spec['j2_over_j1'] != .125:
        raise ValueError('this bounded counter is N144 J2=.125 only')
    arm = spec['arm']; params = pair[arm]
    bank_fn = lambda p: arity_clebsch_bank(p, cfg, arm, old_spec['binary_rank'])
    bank = bank_fn(params)
    # Exactly the original360469 synthetic physical input, not a new Born draw.
    rows = jnp.asarray(balanced_block_probe_states(cfg, count=1, seed=old_spec['seed']+1))
    group = np.asarray(triangular_hnf_gamma_a1_group(((12, 0), (0, 12))), np.int32)
    bonds = tuple(triangular_j1j2_bonds(12, 12, j2=.125, periodic=True))
    old = make_mixed_projected_amplitude(cfg, group, chunk=old_spec['projection_chunk'])
    new = make_feature_lut_projected_amplitude(cfg, group, chunk=old_spec['projection_chunk'])
    old_h = make_amplitude_heisenberg_action(old, bonds, n_sites=144)
    new_h = make_amplitude_heisenberg_action(new, bonds, n_sites=144)
    result.update(model=old_spec['model'], arm=arm, full_A1_images=len(group), Hamiltonian_bonds=len(bonds),
        original_arity_params_sha256=pytree_sha256(params), original_physical_bank_sha256=pytree_sha256(bank),
        probe_bits_sha256=pytree_sha256(rows), parameter_count=sum(x.size for x in jax.tree.leaves(params)),
        probe_distribution='original synthetic block-balanced singleton; no Born energy estimate')
    prepare, prepared = cold('LUT_preparation', lambda b: prepare_feature_lut_bank(b, cfg), bank)
    execute_old_h, values_old = cold('old_complete_H', old_h, bank, rows)
    execute_new_h, values_new = cold('prepared_complete_H', new_h, prepared, rows)
    result['relative_errors'].update(amplitude=relative_vector_error(values_new[0], values_old[0]),
        full_H=relative_vector_error(values_new[1], values_old[1]))
    scalar_old = lambda p: old(bank_fn(p), rows[0])
    scalar_new = lambda p: new(prepare_feature_lut_bank(bank_fn(p), cfg), rows[0])
    direction = jax.tree.map(lambda w: 1e-4*jnp.sin(jnp.arange(w.size).reshape(w.shape)+.2), params)
    _, tangent_old = cold('old_parameter_JVP', lambda p, v: jax.jvp(scalar_old, (p,), (v,))[1], params, direction)
    _, tangent_new = cold('prepare_inclusive_parameter_JVP', lambda p, v: jax.jvp(scalar_new, (p,), (v,))[1], params, direction)
    execute_g0, gradient_old = cold('old_full_parameter_gradient', jax.grad(scalar_old), params)
    execute_g1, gradient_new = cold('prepare_inclusive_full_parameter_gradient', jax.grad(scalar_new), params)
    result['relative_errors']['parameter_JVP'] = relative_vector_error(tangent_new, tangent_old)
    result['relative_errors']['full_parameter_gradient'] = relative_vector_error(
        np.concatenate([np.asarray(a).ravel() for a in jax.tree.leaves(gradient_new)]),
        np.concatenate([np.asarray(a).ravel() for a in jax.tree.leaves(gradient_old)]))
    if (not np.isfinite(list(result['relative_errors'].values())).all()
            or max(result['relative_errors'].values()) > spec['tolerance']):
        result.update(status='failed_oracle', censored=False); save()
        raise ArithmeticError('full-H/parameter AD replay failed')
    phase('all_state_and_derivative_replays_passed')
    for repeat in range(3):
        # A completed pair is never advertised if its mate was not measured.
        # Use synchronized actual first/warm H times, not a reduced projector.
        estimated_pair_seconds = (sum(result['cases'][-1][k] for k in (
            'old_H_seconds', 'new_prepare_plus_H_seconds', 'old_full_gradient_seconds',
            'new_prepare_inclusive_full_gradient_seconds'))
            if result['cases'] else result['measurements']['old_complete_H']['first_execution_seconds']
                +result['measurements']['prepared_complete_H']['first_execution_seconds']
                +result['measurements']['old_full_parameter_gradient']['first_execution_seconds']
                +result['measurements']['prepare_inclusive_full_parameter_gradient']['first_execution_seconds'])
        if spec['seconds_cap']-(perf_counter()-start) < 1.05*estimated_pair_seconds+5:
            censor('insufficient registered time for another complete alternating old/new pair')
            return result
        phase('warm_pair_'+str(repeat))
        tick = perf_counter(); jax.block_until_ready(execute_old_h(bank, rows)); old_seconds = perf_counter()-tick
        tick = perf_counter(); fresh = jax.block_until_ready(prepare(bank))
        jax.block_until_ready(execute_new_h(fresh, rows)); new_seconds = perf_counter()-tick
        tick = perf_counter(); jax.block_until_ready(execute_g0(params)); grad_old_seconds = perf_counter()-tick
        tick = perf_counter(); jax.block_until_ready(execute_g1(params)); grad_new_seconds = perf_counter()-tick
        result['cases'].append(dict(repeat=repeat, old_H_seconds=old_seconds,
            new_prepare_plus_H_seconds=new_seconds, old_full_gradient_seconds=grad_old_seconds,
            new_prepare_inclusive_full_gradient_seconds=grad_new_seconds))
        save(); print(json.dumps(result['cases'][-1]), flush=True)
    result.update(status='completed', censored=False)
    signal.alarm(0)
    save()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--collect-job')
    args = parser.parse_args()
    report = collect_accounting(args.config, args.collect_job) if args.collect_job else run(args.config)
    print(json.dumps(report, indent=2))
