"""Exact N36 full-H cost, streamed at fixed centers; no training or energy estimate.

Both overlapping gates and all attention parameters are live. Only the physical
carrier is fixed. Full SU2 and one final full A1 projection are unchanged by
chunking. All repeated carrier queries across chunks count in measured cost.
"""
from __future__ import annotations

import argparse
import gc
import io
import json
from pathlib import Path
import platform
import resource
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.overlapping_carrier_queries import (
    compile_overlapping_carrier_queries, evaluate_overlapping_carrier_queries,
    overlapping_carrier_amplitudes,
)
from su2_nqs.benchmarks.triangular_induced_parton import build_n36_pi_zero_flux_orbit
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig, projected_singlet_agp_amplitudes
from su2_nqs.models.shared_resonance_attention import (
    ResonanceAttentionConfig, init_shared_resonance_attention,
    make_resonance_attention_reference, shared_resonance_generators,
    shared_resonance_parameter_count,
)
from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries, gather_physical_h_action

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_overlap_attention_cost.json'
ARMS = ('cascade', 'no_cascade', 'diagonal_conditioned', 'factorized')


def validate_cost_config(config: dict) -> None:
    """Reject protocol changes before any numerical setup or output creation."""
    if (config['task'] != 'spin_overlap_attention_cost' or config['geometry'] != 'triangular'
        or config['length'] != 6 or config['j2_over_j1'] != .125
        or config['training_steps'] != 0 or config['remote_allowed']
        or tuple(config['arms']) != ARMS or len(config['center_seeds']) != 2
        or len(set(config['center_seeds'])) != 2):
        raise ValueError('configuration must describe this fixed local N36 cost protocol')
    for key in ('identity_relative_tolerance', 'jvp_relative_tolerance', 'checkpoint_relative_tolerance'):
        if not np.isfinite(config[key]) or not 0 < config[key] <= 1e-4:
            raise ValueError('finite tight validation tolerances required')
    for key in ('H_output_chunk_size', 'warm_repeats', 'carrier_batch_size',
                'max_unique_queries', 'max_terminal_requests', 'max_plan_bytes', 'wall_seconds_cap'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError('positive integer resource settings required')


def peak_rss_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value)*(1 if platform.system() == 'Darwin' else 1024)/2**30


def _timed(function, *arguments):
    tick = perf_counter()
    result = function(*arguments)
    jax.block_until_ready(result)
    return result, perf_counter()-tick


def _complex(values) -> dict:
    a = np.asarray(values)
    return dict(real=a.real.tolist(), imag=a.imag.tolist())


def _relative(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b)))/max(float(np.linalg.norm(b)), 1e-300)


def _save_npz(path: Path, **arrays) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def profile_chunk(states, group, cells, carrier, config, cfg, offsets, reference,
                  initial, params, direction, restored, restored_reference, scalar_weights):
    """One complete amplitude chunk, with live forward, reverse AD and JVP.

    scalar_weights are the exact global linear weights for a normalized
    sum(psi)+sum(Hpsi) cost probe, not an energy gradient. Summing the returned
    gradients is the reverse derivative of that complete full-H scalar.
    """
    tick = perf_counter()
    plan = compile_overlapping_carrier_queries(states, group, cells=cells,
        max_unique_queries=config['max_unique_queries'],
        max_terminal_requests=config['max_terminal_requests'], max_plan_bytes=config['max_plan_bytes'])
    profile = dict(plan=plan.diagnostics, planning_seconds=perf_counter()-tick, arms={})
    values, seconds = _timed(lambda: evaluate_overlapping_carrier_queries(
        plan, carrier, batch_size=config['carrier_batch_size']))
    values = jnp.asarray(values)
    profile.update(carrier_query_inclusive_seconds=seconds, determinants_evaluated=3*len(values))

    def amplitudes(p, v, mode='cascade', restriction='full', ref=reference):
        tables = shared_resonance_generators(p, ref, offsets, cfg, restriction=restriction)
        return overlapping_carrier_amplitudes(tables[0], tables[1], plan, v, mode=mode)

    forward = jax.jit(amplitudes)
    identity, first = _timed(forward, initial, values)
    profile['forward_compile_inclusive_seconds'] = first
    endpoints = {}
    for arm in ARMS:
        mode = 'no_cascade' if arm == 'no_cascade' else 'cascade'
        restriction = arm if arm in ('diagonal_conditioned', 'factorized') else 'full'
        fn = forward if arm == 'cascade' else jax.jit(
            lambda p, v, m=mode, r=restriction: amplitudes(p, v, mode=m, restriction=r))
        endpoint, seconds = _timed(fn, params, values)
        warm = [_timed(fn, params, values)[1] for _ in range(config['warm_repeats'])]
        endpoints[arm] = np.asarray(endpoint)
        profile['arms'][arm] = dict(first_seconds=seconds, warm_seconds=warm)
    weights = jnp.asarray(scalar_weights)
    gradient = jax.jit(jax.grad(lambda p, v: jnp.real(jnp.sum(weights*amplitudes(p, v)))))
    grad, seconds = _timed(gradient, params, values)
    profile.update(gradient_compile_inclusive_seconds=seconds,
                   gradient_warm_seconds=[_timed(gradient, params, values)[1] for _ in range(config['warm_repeats'])])
    jvp = jax.jit(lambda p, v: jax.jvp(lambda x: amplitudes(x, v), (p,), (direction,))[1])
    derivative, seconds = _timed(jvp, params, values)
    profile.update(jvp_compile_inclusive_seconds=seconds,
                   jvp_warm_seconds=[_timed(jvp, params, values)[1] for _ in range(config['warm_repeats'])])
    eps = 1e-5
    plus = {k: v+eps*direction[k] for k, v in params.items()}
    minus = {k: v-eps*direction[k] for k, v in params.items()}
    fd = np.asarray((forward(plus, values)-forward(minus, values))/(2*eps))
    replay = amplitudes(restored, values, ref=restored_reference)
    profile['checkpoint_relative_error'] = _relative(replay, endpoints['cascade'])
    profile['peak_rss_gib'] = peak_rss_gib()
    # A new plan creates a new closure/JIT. Its compilation is compulsory in
    # this implementation, unlike replay on a previously evaluated fixed plan.
    profile['fresh_forward_inclusive_seconds'] = (profile['planning_seconds']+
        profile['carrier_query_inclusive_seconds']+profile['forward_compile_inclusive_seconds'])
    profile['fresh_forward_plus_reverse_AD_inclusive_seconds'] = (
        profile['fresh_forward_inclusive_seconds']+profile['gradient_compile_inclusive_seconds'])
    arrays = dict(identity=np.asarray(identity), derivative=np.asarray(derivative), finite_difference=fd, **endpoints)
    if any(not np.isfinite(a).all() for a in (*arrays.values(), *jax.tree.leaves(grad))):
        raise FloatingPointError('nonfinite amplitude or gradient')
    if profile['checkpoint_relative_error'] > config['checkpoint_relative_tolerance']:
        raise FloatingPointError('checkpoint amplitude replay failed')
    return profile, arrays, {k: np.asarray(v) for k, v in grad.items()}


def run(output: Path, config_path: Path = CONFIG) -> dict:
    config = json.loads(config_path.read_text())
    validate_cost_config(config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('refusing to overwrite a registered experiment')
    output.mkdir(parents=True, exist_ok=True)
    registration = dict(config=config, config_sha256=file_sha256(config_path),
        source_manifest=transitive_local_source_manifest([Path(__file__)], root=ROOT),
        runtime=numerical_runtime_identity())
    atomic_json(output/'registration.json', registration)
    start = perf_counter()
    result = dict(status='running', registration_sha256=file_sha256(output/'registration.json'),
        config=config, geometries={}, training_steps=0, energy_estimate=None,
        variance_estimate=None, remote_released=False)

    def budget(stage):
        atomic_json(output/'progress.json', dict(stage=stage, seconds=perf_counter()-start, peak_rss_gib=peak_rss_gib()))
        print(stage, flush=True)
        if peak_rss_gib() > config['peak_rss_gib_cap']:
            raise MemoryError('registered peak-RSS cap exceeded at a chunk boundary')
        if perf_counter()-start > config['wall_seconds_cap']:
            raise TimeoutError('registered wall cap exceeded')

    def alarm_handler(*_):
        raise TimeoutError('registered wall cap exceeded')

    previous_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(config['wall_seconds_cap'])
    try:
        cfg = ResonanceAttentionConfig(config['width'], config['hidden_width'])
        offsets = jnp.asarray(config['displacements'])
        initial = init_shared_resonance_attention(jax.random.PRNGKey(config['initialization_seed']), cfg)
        reference = make_resonance_attention_reference(initial, offsets, cfg)
        rng = np.random.default_rng(config['perturbation_seed'])
        noise = {k: rng.normal(size=v.shape) for k, v in initial.items()}
        norm = np.sqrt(sum(np.sum(v*v) for v in noise.values()))
        direction = {k: jnp.asarray(v/norm) for k, v in noise.items()}
        params = {k: v+config['parameter_perturbation_norm']*direction[k] for k, v in initial.items()}
        result.update(parameter_count=shared_resonance_parameter_count(cfg), shared_table_maximum_coordinates=60)
        centers = []
        for seed in config['center_seeds']:
            row = np.zeros(36, dtype=np.int8)
            row[np.random.default_rng(seed).choice(36, 18, replace=False)] = 1
            centers.append(row)
        centers = np.asarray(centers)
        group = triangular_hnf_gamma_a1_group(((6, 0), (0, 6)))
        bonds = triangular_hnf_j1j2_bonds(((6, 0), (0, 6)), j2=config['j2_over_j1'])
        hplan, seconds = _timed(compile_physical_h_queries, centers, bonds)
        result['h_stencil'] = dict(hplan.diagnostics, seconds=seconds)
        orbit = build_n36_pi_zero_flux_orbit()
        agp = ProjectedSingletAGPConfig(36, (tuple(range(36)),))
        pairings = [jnp.asarray(p) for p in orbit.pairings]

        @jax.jit
        def carrier_kernel(bits):
            columns = jnp.stack([projected_singlet_agp_amplitudes(
                {'pairing_real': p}, bits, agp).amplitudes for p in pairings], axis=1)
            return columns @ jnp.asarray([1., -1., -1.])/3

        batch = config['carrier_batch_size']
        def carrier(bits):
            # Fixed shape includes padded evaluations in the determinant ledger.
            length = len(bits)
            padded = np.pad(np.asarray(bits), ((0, batch-length), (0, 0)), mode='edge')
            return carrier_kernel(jnp.asarray(padded))[:length]

        _, seconds = _timed(carrier, centers[:1])
        result['carrier_initial_compile_seconds'] = seconds
        carrier_values = np.asarray(carrier(hplan.configurations))
        cpsi, chpsi = map(np.asarray, gather_physical_h_action(hplan, carrier_values))
        scale = max(float(np.linalg.norm(cpsi)), 1e-300)
        hscale = max(float(np.linalg.norm(chpsi)), 1e-300)
        weights = np.zeros(len(hplan.configurations))
        np.add.at(weights, hplan.center_indices, 1/scale)
        np.add.at(weights, hplan.connected_indices[hplan.valid], hplan.elements[hplan.valid]/hscale)
        _save_npz(output/'checkpoint.npz', reference=reference, offsets=offsets, centers=centers,
                  **{'parameter_'+k: v for k, v in params.items()}, **{'initial_'+k: v for k, v in initial.items()},
                  **{'direction_'+k: v for k, v in direction.items()})
        with np.load(output/'checkpoint.npz') as checkpoint:
            restored = {k: jnp.asarray(checkpoint['parameter_'+k]) for k in params}
            restored_reference = checkpoint['reference'].copy()
        result['checkpoint_sha256'] = file_sha256(output/'checkpoint.npz')
        result['checkpoint_parameter_max_error'] = max(float(np.max(np.abs(
            np.asarray(params[k])-np.asarray(restored[k])))) for k in params)
        stored = dict(carrier_all_H_outputs=carrier_values)
        for geometry in ('physical', 'scrambled'):
            profile = dict(chunks=[], arms={})
            result['geometries'][geometry] = profile
            collected = {k: [] for k in (*ARMS, 'identity', 'derivative', 'finite_difference')}
            total_gradient = {k: np.zeros_like(v) for k, v in params.items()}
            chunk_size = config['H_output_chunk_size']
            for index in range(0, len(hplan.configurations), chunk_size):
                budget(f'{geometry}: full-H outputs {index}:{min(index+chunk_size,len(hplan.configurations))}')
                chunk, arrays, grad = profile_chunk(hplan.configurations[index:index+chunk_size], group,
                    config[geometry+'_cells'], carrier, config, cfg, offsets, reference, initial,
                    params, direction, restored, restored_reference, weights[index:index+chunk_size])
                count = chunk['plan']['unique_carrier_queries']
                chunk['determinants_including_padding'] = 3*batch*((count+batch-1)//batch)
                profile['chunks'].append(chunk)
                for key in collected:
                    collected[key].append(arrays[key])
                for key in total_gradient:
                    total_gradient[key] += grad[key]
                atomic_json(output/'partial_results.json', result)
                jax.clear_caches()
                gc.collect()
            action_results = {}
            for key, pieces in collected.items():
                all_values = np.concatenate(pieces)
                stored[geometry+'_'+key+'_all_H_outputs'] = all_values
                action_results[key] = tuple(map(np.asarray, gather_physical_h_action(hplan, jnp.asarray(all_values))))
            errors = [_relative(a,b) for a,b in zip(action_results['identity'],(cpsi,chpsi),strict=True)]
            jvp_errors = [_relative(a,b) for a,b in zip(action_results['derivative'],action_results['finite_difference'],strict=True)]
            profile.update(identity_relative_errors=errors, jvp_finite_difference_relative_errors=jvp_errors,
                gradient_leaf_norms={k: float(np.linalg.norm(v)) for k,v in total_gradient.items()},
                carrier_queries_summed_over_chunks=sum(c['plan']['unique_carrier_queries'] for c in profile['chunks']),
                determinants_including_padding=sum(c['determinants_including_padding'] for c in profile['chunks']),
                planning_seconds=sum(c['planning_seconds'] for c in profile['chunks']),
                carrier_query_inclusive_seconds=sum(c['carrier_query_inclusive_seconds'] for c in profile['chunks']),
                fresh_full_H_inclusive_seconds=sum(c['fresh_forward_inclusive_seconds'] for c in profile['chunks']),
                fresh_full_H_plus_reverse_AD_inclusive_seconds=sum(c['fresh_forward_plus_reverse_AD_inclusive_seconds'] for c in profile['chunks']))
            for arm in ARMS:
                psi, hp = action_results[arm]
                profile['arms'][arm] = dict(psi=_complex(psi), hpsi=_complex(hp),
                    cached_full_H_warm_seconds=sum(float(np.median(c['arms'][arm]['warm_seconds'])) for c in profile['chunks']))
            profile['mixed_point_amplitude_relative_norm'] = _relative(action_results['cascade'][0],action_results['no_cascade'][0])
            profile['mixed_point_H_relative_norm'] = _relative(action_results['cascade'][1],action_results['no_cascade'][1])
            profile['linear_512_centers_1000_steps_cpu_hours_not_forecast'] = profile[
                'fresh_full_H_plus_reverse_AD_inclusive_seconds']/len(centers)*512*1000/3600
            stored.update({geometry+'_gradient_'+k: v for k, v in total_gradient.items()})
            if max(errors) > config['identity_relative_tolerance'] or max(jvp_errors) > config['jvp_relative_tolerance']:
                raise FloatingPointError('full-H identity or JVP validation failed')
            if any(not np.isfinite(v) or v == 0 for v in profile['gradient_leaf_norms'].values()):
                raise FloatingPointError('a shared-backbone parameter block has zero/nonfinite full-H gradient')
            budget(geometry+': completed full-H and reverse AD')
        _save_npz(output/'full_h_replay.npz', centers=centers, h_configurations=hplan.configurations,
                  h_centers=hplan.center_indices, h_indices=hplan.connected_indices,
                  h_elements=hplan.elements, h_valid=hplan.valid, **stored)
        result.update(status='completed', replay_sha256=file_sha256(output/'full_h_replay.npz'),
            interpretation='Cost/algebra only. Uniform fixed-M points are not wavefunction samples. No energy/variance estimate, optimized null, full ViT advantage, GPU timing or scaling law. One shared60-table; multiplicity propagation, not fixed-partition q exchange.')
    except (MemoryError, TimeoutError) as error:
        result.update(status='resource_cap_reached', stop_reason=str(error))
    except Exception as error:
        result.update(status='failed_validation_or_runtime', stop_reason=repr(error))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        result.update(total_seconds=perf_counter()-start, peak_rss_gib=peak_rss_gib())
        atomic_json(output/'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    result = run(args.output, args.config)
    print(json.dumps(dict(status=result['status'], seconds=result['total_seconds'], peak_rss_gib=result['peak_rss_gib']), indent=2))


if __name__ == '__main__':
    main()
