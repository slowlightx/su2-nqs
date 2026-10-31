"""Bounded six-branch Schur-conditioned attention amplitude/cost preflight.

This workflow never trains a state or estimates its variational energy. The
triangle uses an unchanged, independently benchmarked exact-A1 parton carrier;
the square is explicitly a residual-only cost check until such a carrier exists.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit,
    projected_pi_zero_flux_a1_amplitudes_jax,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact, save_parameter_artifact
from su2_nqs.frontier.provenance import (
    file_sha256, numerical_runtime_identity, transitive_local_source_manifest,
)
from su2_nqs.models.schur_cutset_attention import (
    CutsetAttentionConfig, conditional_coefficients, cutset_raw_amplitude,
    init_cutset_attention, make_branch_bank, make_free_branch_bank, make_free_parameters,
)
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.su2.fusion_tree import Leaf
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.projected_spin_work import make_chunked_a1_amplitudes
from .spin import _bonds, _groups
from .spin_decoder_cost import ROOT, peak_rss_bytes, physical_config
from .spin_decoder_reuse import measure
from .spin_vap_decoder import init_decoder

jax.config.update('jax_enable_x64', True)
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_schur_cutset.json'


def registration(spec: dict) -> dict:
    """Seal the finite protocol, unchanged parent/probes and numeric sources."""
    if (spec['amplitude_probes'], spec['H_probes'], spec['JVP_probes'], spec['training_steps']) != (8, 2, 1, 0):
        raise ValueError('registered preflight requires 8 amplitude, 2 H, 1 JVP probes and no training')
    if spec['arms'] != ['attention', 'scrambled', 'rank_free', 'full_bank']:
        raise ValueError('all four preregistered controls are required')
    paths = [ROOT/spec['parent']/case['geometry']/name for case in spec['cases']
             for name in ('parameters.json', 'parameters.npz', 'values.npz')]
    paths.append(ROOT/spec['triangular_carrier_reference'])
    result = dict(config=spec,
        inputs={str(path.relative_to(ROOT)): file_sha256(path) for path in paths},
        runtime=numerical_runtime_identity(),
        sources=transitive_local_source_manifest((Path(__file__),), root=ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    result['registration_sha256'] = canonical_sha256(result)
    return result


def _cell_indices(node) -> tuple[int, ...]:
    if isinstance(node, Leaf):
        return (node.index,)
    return _cell_indices(node.left) + _cell_indices(node.right)


def run(path: Path, geometry: str) -> dict:
    """One bounded geometry, four fresh compiled controls, no optimizer steps."""
    spec = json.loads(path.read_text()); output = ROOT/spec['output']; destination = output/geometry
    registered = json.loads((output/'registration.json').read_text())
    if registered != registration(spec):
        raise ValueError('registration mismatch')
    if (destination/'progress.json').exists():
        raise FileExistsError('do not overwrite a started dry run')
    case = next(row for row in spec['cases'] if row['geometry'] == geometry)
    start = perf_counter(); measurements = {}; validation = {}; results = {}; raw = {}
    full_H_centers = 0

    def progress(event: str) -> None:
        atomic_json(destination/'progress.json', dict(event=event, geometry=geometry,
            active_seconds=perf_counter()-start, peak_rss_bytes=peak_rss_bytes(),
            full_H_centers=full_H_centers, measurements=measurements, validation=validation))
        print(geometry, event, round(perf_counter()-start, 2), 's',
              round(peak_rss_bytes()/2**30, 3), 'GiB', flush=True)
        if peak_rss_bytes() > spec['peak_rss_limit_gib']*2**30:
            raise MemoryError('registered process RSS cap')

    def alarm(_signal, _frame):
        raise TimeoutError('registered wall limit; no automatic budget extension')

    previous_signal = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(spec['wall_limit_seconds_per_process'])
    try:
        progress('setup')
        tree = physical_config(6, 6, spec['width']); parent = ROOT/spec['parent']/geometry
        base = load_parameter_artifact(parent/'parameters.json', init_decoder(2026093003, tree))
        parent_hash = pytree_sha256(base)
        coordinates = tuple((float(np.mean(np.asarray(cell) % 6)),
                             float(np.mean(np.asarray(cell) // 6))) for cell in tree.cells)
        config = CutsetAttentionConfig(tree, coordinates, cutset_cell=spec['selected_cell'],
            embedding_dim=spec['attention_width'], rank=spec['condition_rank'],
            scramble_seed=spec['scramble_seed'])
        attention = init_cutset_attention(jax.random.PRNGKey(spec['seed']), base, config)
        with np.load(parent/'values.npz') as archive:
            states = jnp.asarray(archive['states'], dtype=jnp.int32)
        if states.shape != (8, 36) or not np.all(np.asarray(states).sum(axis=1) == 18):
            raise ValueError('saved probe contract changed')
        raw['states'] = np.asarray(states)
        group = _groups(geometry, 6, 6)['gamma_a1']; projector = PhysicalSpatialProjector(group)
        bonds = _bonds(geometry, 6, 6, case['j2_over_j1'])
        residual_batch = make_chunked_a1_amplitudes(
            lambda bank, bits: cutset_raw_amplitude(bank, bits, config), projector,
            chunk_size=spec['projection_chunk_size'])
        orbit = build_n36_pi_zero_flux_orbit() if geometry == 'triangular' else None

        carrier_ready = geometry == 'triangular'
        if carrier_ready:
            carrier_values = jax.jit(lambda x: projected_pi_zero_flux_a1_amplitudes_jax(x, orbit=orbit))(states)
            raw['carrier_amplitudes'] = np.asarray(carrier_values)
        else:
            carrier_values = None

        arm_parameters = {
            'attention': attention, 'scrambled': attention,
            'rank_free': make_free_parameters(attention, config),
        }
        arm_parameters['full_bank'] = make_branch_bank(attention, config)
        generators = {
            'attention': lambda p: make_branch_bank(p, config),
            'scrambled': lambda p: make_branch_bank(p, config, scrambled=True),
            'rank_free': lambda p: make_free_branch_bank(p, config),
            'full_bank': lambda p: p,
        }
        remote_nodes = [index for index, node in enumerate(tree.cell_tree.internal_nodes)
                        if config.cutset_cell not in _cell_indices(node)]
        for scrambled, label in ((False, 'physical'), (True, 'scrambled')):
            coefficients = np.asarray(conditional_coefficients(attention['attention'], config, scrambled=scrambled))
            raw[label+'_conditional_coefficients'] = coefficients
            validation[label+'_remote_node_ids'] = remote_nodes
            validation[label+'_remote_condition_difference_norm'] = float(
                np.linalg.norm(coefficients[:, remote_nodes]-coefficients[:1, remote_nodes]))

        attention_bank_leaves = None
        for arm in spec['arms']:
            progress(arm+'_begin')
            parameters = arm_parameters[arm]; generate = generators[arm]
            identity = dict(registration_sha256=registered['registration_sha256'], geometry=geometry, arm=arm)
            parameter_path = destination/arm/'parameters.json'
            presave_hash = pytree_sha256(parameters)
            save_parameter_artifact(parameter_path, parameters, identity=identity)
            parameters = load_parameter_artifact(parameter_path, parameters, expected_identity=identity)
            parameter_hash = pytree_sha256(parameters)
            record, executable, reference_bank = measure(generate, (parameters,), spec['timing_repeats'])
            measurements[arm+'_tensor_generation'] = record; del executable
            reference_bank = jax.tree.map(jax.lax.stop_gradient, reference_bank)
            bank_hash = pytree_sha256(reference_bank)
            bank_leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(reference_bank)]
            if arm == 'attention':
                attention_bank_leaves = bank_leaves
            bank_max_difference = max(float(np.max(abs(a-b))) for a, b in
                zip(bank_leaves, attention_bank_leaves, strict=True))

            def amplitudes(bank, bits):
                current = residual_batch(bank, bits)
                if not carrier_ready:
                    return current
                reference = residual_batch(reference_bank, bits)
                parton = projected_pi_zero_flux_a1_amplitudes_jax(bits, orbit=orbit)
                return parton + spec['residual_scale']*(current-reference)

            record, amplitude_executable, values = measure(amplitudes, (reference_bank, states), spec['timing_repeats'])
            measurements[arm+'_amplitude'] = record; raw[arm+'_amplitudes'] = np.asarray(values)
            action = make_amplitude_heisenberg_action(
                lambda bank, bits: amplitudes(bank, bits[None, :])[0], bonds, n_sites=36)
            record, executable, action_values = measure(action, (reference_bank, states[:spec['H_probes']]), spec['timing_repeats'])
            measurements[arm+'_H'] = record; del executable
            arm_H_centers = spec['H_probes']*(1+spec['timing_repeats'])
            full_H_centers += arm_H_centers
            psi, hpsi = map(np.asarray, action_values)
            raw[arm+'_H_psi'] = psi; raw[arm+'_H_action'] = hpsi
            raw[arm+'_H_ratios'] = np.divide(hpsi, psi,
                out=np.full(hpsi.shape, complex(np.nan, np.nan)), where=psi != 0)

            flat, unravel = ravel_pytree(parameters)
            direction = jnp.sin(jnp.arange(flat.size, dtype=jnp.float64)+.317)
            direction /= jnp.linalg.norm(direction)
            def theta_amplitudes(theta, bits):
                return amplitudes(generate(unravel(theta)), bits)
            def theta_jvp(theta, bits):
                return jax.jvp(lambda x: theta_amplitudes(x, bits), (theta,), (direction,))
            record, executable, tangent = measure(theta_jvp, (flat, states[:spec['JVP_probes']]), spec['timing_repeats'])
            measurements[arm+'_end_to_end_JVP'] = record; del executable
            _, tangent_value = map(np.asarray, tangent)
            raw[arm+'_JVP'] = tangent_value; raw[arm+'_direction'] = np.asarray(direction)
            epsilon = spec['finite_difference_epsilon']
            finite = jax.jit(theta_amplitudes)
            plus = np.asarray(finite(flat+epsilon*direction, states[:spec['JVP_probes']]))
            minus = np.asarray(finite(flat-epsilon*direction, states[:spec['JVP_probes']]))
            difference = (plus-minus)/(2*epsilon)
            raw[arm+'_finite_difference'] = difference
            scaled_error = float(np.max(abs(difference-tangent_value)/(1+abs(difference))))
            relative_error = float(np.linalg.norm(difference-tangent_value)/max(np.linalg.norm(tangent_value), 1e-300))
            del finite
            if carrier_ready:
                replay = float(np.linalg.norm(np.asarray(values)-np.asarray(carrier_values))/max(np.linalg.norm(carrier_values), 1e-300))
                relative_tangent = float(np.linalg.norm(tangent_value)/max(np.linalg.norm(np.asarray(carrier_values)[:1]), 1e-300))
            else:
                replay = None; relative_tangent = None
            # A finite nonzero candidate is checked, not merely the known carrier.
            candidate = generate(unravel(flat+epsilon*direction))
            image_ids = np.asarray([0, len(group)//3, len(group)//2+1, len(group)-1])
            image_states = states[:1, np.asarray(group)[image_ids]].reshape(4, 36)
            candidate_values = np.asarray(amplitude_executable(candidate, states))
            raw[arm+'_candidate_amplitudes'] = candidate_values
            base_candidate = candidate_values[0]
            projected_images = np.asarray(jax.jit(amplitudes)(candidate, image_states))
            a1_error = float(np.max(abs(projected_images-base_candidate))/max(abs(base_candidate), 1e-300))
            raw[arm+'_candidate_A1_images'] = projected_images
            results[arm] = dict(parameter_count=int(flat.size), parameter_sha256=parameter_hash,
                generated_bank_parameter_count=sum(int(x.size) for x in jax.tree.leaves(reference_bank)),
                generated_bank_sha256=bank_hash,
                generated_bank_max_abs_difference_from_attention=bank_max_difference,
                presave_parameter_sha256=presave_hash,
                checkpoint_exact=presave_hash == parameter_hash,
                strong_carrier_ready=carrier_ready, centered_carrier_replay_relative_error=replay,
                tangent_norm=float(np.linalg.norm(tangent_value)),
                probe_tangent_relative_to_carrier=relative_tangent,
                probe_tangent_is_full_physical_state_distance=False,
                JVP_finite_difference_scaled_error=scaled_error,
                JVP_finite_difference_relative_error=relative_error,
                candidate_A1_probe_relative_error=a1_error,
                full_H_centers=arm_H_centers,
                H_target_amplitude_requests=arm_H_centers*(len(bonds)+1),
                H_semantic_raw_branch_contractions=arm_H_centers*(len(bonds)+1)*len(group)*6*(2 if carrier_ready else 1),
                H_carrier_determinants=arm_H_centers*(len(bonds)+1)*3 if carrier_ready else 0,
                exact_nodes_in_H_probes=int(np.sum(psi == 0)),
                variational_energy_estimated=False)
            if pytree_sha256(base) != parent_hash:
                raise ValueError('residual parent was mutated')
            del amplitude_executable, action, reference_bank
            progress(arm+'_done')
            # Controls intentionally compile independently; avoid keeping four
            # large unrelated executables alive merely through JAX's cache.
            jax.clear_caches(); gc.collect()

        np.savez_compressed(destination/'values.npz', **raw)
        for arm in ('rank_free', 'full_bank'):
            validation[arm+'_initial_bank_exact'] = results[arm]['generated_bank_sha256'] == results['attention']['generated_bank_sha256']
            validation[arm+'_initial_amplitude_relative_error'] = float(
                np.linalg.norm(raw[arm+'_amplitudes']-raw['attention_amplitudes'])/
                max(np.linalg.norm(raw['attention_amplitudes']), 1e-300))
        validation['spatial_control_same_parameter_direction'] = bool(np.array_equal(
            raw['attention_direction'], raw['scrambled_direction']))
        validation['spatial_control_JVP_probe_absolute_difference'] = float(np.linalg.norm(
            raw['attention_JVP']-raw['scrambled_JVP']))
        validation['spatial_control_JVP_probe_relative_difference'] = float(
            validation['spatial_control_JVP_probe_absolute_difference']/max(
                np.linalg.norm(raw['attention_JVP']), np.linalg.norm(raw['scrambled_JVP']), 1e-300))
        validation['spatial_control_candidate_probe_relative_difference'] = float(np.linalg.norm(
            raw['attention_candidate_amplitudes']-raw['scrambled_candidate_amplitudes'])/max(
                np.linalg.norm(raw['attention_candidate_amplitudes']),
                np.linalg.norm(raw['scrambled_candidate_amplitudes']), 1e-300))
        validation['spatial_control_differences_are_probe_only_not_global_FS'] = True
        passed = all(row['finite'] for row in measurements.values()) and all(
            row['checkpoint_exact'] and row['JVP_finite_difference_scaled_error'] < 1e-7
            and row['JVP_finite_difference_relative_error'] < 1e-3
            and np.isfinite(row['tangent_norm']) and row['tangent_norm'] > 0
            and (not carrier_ready or row['centered_carrier_replay_relative_error'] < 1e-10)
            and row['candidate_A1_probe_relative_error'] < 1e-8 for row in results.values())
        passed = passed and all(validation[label+'_remote_condition_difference_norm'] > 1e-10
                                for label in ('physical', 'scrambled'))
        passed = passed and all(validation[arm+'_initial_amplitude_relative_error'] < 1e-10
            and results[arm]['generated_bank_max_abs_difference_from_attention'] < 1e-12
            for arm in ('rank_free', 'full_bank'))
        passed = passed and validation['spatial_control_same_parameter_direction'] and (
            validation['spatial_control_JVP_probe_relative_difference'] > 1e-10)
        result = dict(task=spec['task'], geometry=geometry, case=case,
            registration_sha256=registered['registration_sha256'], parent_parameters_sha256=parent_hash,
            initial_parent_parameters_sha256=parent_hash, final_parent_parameters_sha256=pytree_sha256(base),
            measurements=measurements, validation=validation, arms=results,
            branch_count=6, spatial_images=len(group), bond_count=len(bonds),
            full_H_centers=full_H_centers, frozen_amplitude_probes=spec['amplitude_probes'],
            strong_carrier_ready=carrier_ready, square_residual_only=not carrier_ready,
            triangular_carrier_reference=spec['triangular_carrier_reference'] if carrier_ready else None,
            state_representation='physical_amplitudes_with_complete_local_Schur_cutset',
            training_steps=0, new_ED=False, new_samples=0, remote_submitted=False,
            no_variational_win_claim=True, passed=bool(passed),
            wall_seconds=perf_counter()-start, peak_rss_bytes=peak_rss_bytes(),
            raw_sha256=file_sha256(destination/'values.npz'))
        result['result_sha256'] = canonical_sha256(result)
        atomic_json(destination/'results.json', result); progress('complete')
        return result
    except Exception as error:
        atomic_json(destination/'failure.json', dict(type=type(error).__name__, message=str(error),
            active_seconds=perf_counter()-start, full_H_centers=full_H_centers))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous_signal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--geometry', choices=('triangular', 'square'))
    args = parser.parse_args(); spec = json.loads(args.config.read_text())
    if args.register:
        path = ROOT/spec['output']/'registration.json'
        if path.exists():
            raise FileExistsError(path)
        record = registration(spec); atomic_json(path, record); print(record['registration_sha256'])
    else:
        if args.geometry is None:
            parser.error('--geometry is required unless registering')
        run(args.config, args.geometry)


if __name__ == '__main__':
    main()
