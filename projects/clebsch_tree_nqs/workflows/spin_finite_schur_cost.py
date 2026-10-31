"""Bounded finite-Schur CNN/attention cost gate with untied free tensors.

One reusable runner for both geometries. No optimizer, ED or new samples.
The exact projected H work kernel and checkpoint/timing utilities are reused,
not reimplemented. See the sealed registration for the state-manifold boundary.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.parameter_artifact import save_parameter_artifact, load_parameter_artifact
from su2_nqs.frontier.provenance import file_sha256, numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.models.finite_schur_decoder import (
    FiniteSchurDecoderConfig, init_finite_schur_network, neural_reduced_entries,
    neural_reduced_bank, free_reduced_bank, make_finite_schur_amplitude,
    reduced_entry_metadata, centered_additive_amplitudes,
    init_layer_standardized_free_tensors,
)
from su2_nqs.models.schur_cutset_compact import make_full_a1_batch
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.projected_spin_work import make_deduplicated_a1_heisenberg, make_frozen_vmc_force
from .spin import _groups, _bonds
from .spin_decoder_cost import ROOT, benchmark, physical_config, peak_rss_bytes

jax.config.update('jax_enable_x64', True)
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_finite_schur_cost.json'


def registration(spec: dict) -> dict:
    """Seal config, frozen probes, numerical sources and runtime before timing."""
    record = dict(config=spec, probe_sha256=file_sha256(ROOT/spec['probe_file']),
        runtime=numerical_runtime_identity(), sources=transitive_local_source_manifest((Path(__file__),),
            root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')))
    record['registration_sha256'] = canonical_sha256(record)
    return record


def run(config_path: Path, geometry: str) -> dict:
    """Measure one geometry; all H terms and A1 images retained, no training."""
    spec = json.loads(config_path.read_text()); output = ROOT/spec['output']
    reg = json.loads((output/'registration.json').read_text())
    if reg != registration(spec):
        raise ValueError('registration mismatch')
    case = next(c for c in spec['cases'] if c['geometry'] == geometry)
    destination = output/geometry
    if (destination/'progress.json').exists():
        raise FileExistsError(destination)
    measurements, checks, values = {}, {}, {}
    begin = perf_counter()
    def persist(status):
        atomic_json(destination/'progress.json', dict(status=status, geometry=geometry,
            elapsed_seconds=perf_counter()-begin, measurements=measurements, checks=checks,
            registration_sha256=reg['registration_sha256'], peak_rss_bytes=peak_rss_bytes()))
        print(geometry, status, round(perf_counter()-begin, 3), 's',
              round(peak_rss_bytes()/2**30, 3), 'GiB', flush=True)
        if peak_rss_bytes() > spec['peak_rss_limit_gib']*2**30:
            raise MemoryError('registered RSS cap reached')
    def alarm(_signal, _frame):
        raise TimeoutError('registered wall budget reached')
    previous = signal.signal(signal.SIGALRM, alarm); signal.alarm(spec['wall_limit_seconds_per_process'])
    persist('started')
    try:
        rows, columns = spec['rows'], spec['columns']
        cfg = FiniteSchurDecoderConfig(replace(physical_config(rows, columns, spec['multiplicity_width']),
            maximum_intermediate_q=spec['maximum_q']), rows//2, columns//2,
            spec['embedding_dim'], spec['depth'])
        groups = _groups(geometry, rows, columns)
        projector = PhysicalSpatialProjector(groups['gamma_a1'])
        bonds = _bonds(geometry, rows, columns, case['j2_over_j1'])
        with np.load(ROOT/spec['probe_file']) as data:
            states = data[geometry+'_states'][:max(spec['batches'])].astype(np.int32)
        params = init_finite_schur_network(jax.random.PRNGKey(spec['seed']), cfg)
        identity = dict(registration_sha256=reg['registration_sha256'], geometry=geometry)
        save_parameter_artifact(destination/'network.json', params, identity=identity)
        loaded = load_parameter_artifact(destination/'network.json', params, expected_identity=identity)
        checks['checkpoint_identical'] = pytree_sha256(loaded) == pytree_sha256(params)
        params = loaded
        entries = neural_reduced_entries(params, cfg)
        free = {'entries': jnp.stack((entries.real, entries.imag), axis=1)}
        save_parameter_artifact(destination/'free_tensors.json', free, identity=identity)
        make_bank = lambda p: neural_reduced_bank(p, cfg)
        make_free = lambda p: free_reduced_bank(p, cfg)
        bank = make_bank(params)
        if spec.get('initialization_calibration_rows') is not None:
            from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
            key=jax.random.PRNGKey(spec['initialization_seed'])
            probes=initialize_spin_exchange_chains(jax.random.fold_in(key,1),
                n_chains=spec['initialization_calibration_rows'],n_sites=rows*columns,n_up=rows*columns//2)
            independent_free,initialization_report=init_layer_standardized_free_tensors(key,cfg,np.asarray(probes))
            # Exact DOUBLE-layer readout of the RAW trees only. This is not
            # the Schmidt spectrum of their Reynolds sum or AGP+residual.
            from su2_nqs.models.finite_schur_capacity import finite_bank_reduced_tree
            from su2_nqs.su2.reduced_tree_environment import reduced_tree_environments, canonical_edge_spectrum
            spectra={}
            for name,selected_bank in (('neural_generated',bank),
                                      ('independent_uniform',make_free(independent_free))):
                state=finite_bank_reduced_tree(selected_bank,cfg)
                env=reduced_tree_environments(state)
                edges=[]
                for index,node in enumerate(state.tree.internal_nodes[:-1]):
                    spectrum=canonical_edge_spectrum(env,node)
                    blocks={}
                    for q,b in spectrum.blocks.items():
                        w=np.asarray(b.multiplicity_weights); total=float(w.sum())
                        blocks[str(q)]=dict(weight=total,metric_rank=b.metric_rank,
                            occupied_multiplicities=int(np.count_nonzero(w > 1e-10)),
                            conditional_participation=(float(total**2/np.sum(w*w)) if total else 0.))
                    edges.append(dict(index=index,entropy=spectrum.entropy,blocks=blocks))
                spectra[name]=dict(raw_norm_squared=env.norm_squared,
                    edge_norm_discrepancy=env.maximum_relative_edge_norm_error,edges=edges)
            initialization_report['raw_tree_capacity']=spectra
            initialization_report['projected_or_additive_spectrum']=False
            checks['layer_standardization_scaled_error']=float(max(
                abs(v-1) for v in initialization_report['activation_std_after']))
            checks['raw_edge_norm_scaled_error']=float(max(
                v['edge_norm_discrepancy'] for v in spectra.values()))
            atomic_json(destination/'independent_initialization.json',initialization_report)
        raw = make_finite_schur_amplitude(cfg)
        projected = make_full_a1_batch(raw, projector, chunk_size=spec['projection_chunk'])
        plan, evaluate, _ = make_deduplicated_a1_heisenberg(raw, projector, bonds, chunk_size=spec['H_chunk'])
        reports, executables = {}, {}
        def measure(name, fn, p, bits):
            persist(name+'_begin')
            report, executable, value = benchmark(fn, p, bits, spec['timing_repeats'])
            measurements[name] = report; values[name] = value; executables[name] = executable
            persist(name+'_done')
            return value
        measure('bank_generation', lambda p, _: neural_reduced_entries(p, cfg), params, states[:1])
        # The dynamical H plan is timed independently and again inside each H call.
        for batch in spec['batches']:
            bits = states[:batch]; prepared = jax.jit(plan)(jnp.asarray(bits))
            reports[str(batch)] = dict(unique_queries=int(prepared.count),
                unmerged_image_queries=int(prepared.inverse.size), full_bonds=len(bonds),
                full_images=projector.group_order)
            measure(f'plan_b{batch}', lambda p, x: jnp.asarray(plan(x).count, dtype=jnp.float64).reshape(1), {}, bits)
            for arm, p, generator in (('network', params, make_bank), ('free', free, make_free)):
                measure(f'{arm}_raw_b{batch}', lambda p, x: jax.vmap(raw, in_axes=(None, 0))(generator(p), x), p, bits)
                measure(f'{arm}_A1_b{batch}', lambda p, x: projected(generator(p), x), p, bits)
                measure(f'{arm}_full_H_b{batch}', lambda p, x: evaluate(generator(p), plan(x))[0], p, bits)
            fixed_energies = values[f'network_full_H_b{batch}']
            for arm, p, generator in (('network', params, make_bank), ('free', free, make_free)):
                amplitude = lambda p, x: projected(generator(p), x)
                force = make_frozen_vmc_force(amplitude, jnp.asarray(fixed_energies), rows*columns)
                measure(f'{arm}_score_force_b{batch}', lambda p, x: ravel_pytree(force(p, x))[0], p, bits)
        for chunk in spec.get('compare_H_chunks',[]):
            _,alternative,_=make_deduplicated_a1_heisenberg(raw,projector,bonds,chunk_size=chunk)
            batch=max(spec['batches'])
            name=f'network_full_H_chunk{chunk}_b{batch}'
            value=measure(name,lambda p,x:alternative(make_bank(p),plan(x))[0],params,states[:batch])
            expected=values[f'network_full_H_b{batch}']
            checks[f'H_chunk{chunk}_replay_relative_error']=float(np.linalg.norm(value-expected)/
                                                                 max(np.linalg.norm(expected),1e-300))
        persist('physical_validation_begin')
        batch = max(spec['batches']); raw_values = values[f'network_raw_b{batch}']
        for mode in ('raw', 'A1', 'full_H'):
            a, b = values[f'network_{mode}_b{batch}'], values[f'free_{mode}_b{batch}']
            checks[mode+'_control_relative_error'] = float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-300))
        group = np.asarray(groups['gamma_a1']); selected = group[[0, len(group)//3, len(group)//2+1, -1]]
        image_values = np.asarray(jax.jit(projected)(bank, states[:1, selected].reshape(-1, rows*columns)))
        center_value = values[f'network_A1_b{batch}'][0]
        checks['A1_covariance_relative_error'] = float(np.linalg.norm(image_values-center_value)/max(abs(center_value), 1e-300))
        center = states[0]; up, down = np.flatnonzero(center), np.flatnonzero(1-center)
        swaps = []
        for a in up:
            for b in down:
                row = center.copy(); row[a], row[b] = row[b], row[a]; swaps.append(row)
        raw_function = jax.jit(jax.vmap(raw, in_axes=(None, 0)))
        swapped = np.asarray(raw_function(bank, jnp.asarray(swaps)))
        checks['S2_scaled_residual'] = float(abs((rows*columns/2)*raw_values[0]+sum(swapped))/(
            rows*columns/2*abs(raw_values[0])+sum(abs(swapped))))
        graph = SzHeisenbergGraph(rows*columns, tuple(bonds), n_up=rows*columns//2)
        connected, elements = graph.get_conn(tuple(map(int, center)))
        # Independent graph columns, separately projected amplitudes, no division shortcut.
        conn_values = np.asarray(jax.jit(projected)(bank, jnp.asarray(connected, dtype=jnp.int32)))
        h_expected = np.dot(elements, conn_values)
        h_actual = values[f'network_full_H_b{batch}'][0]*center_value
        checks['full_H_action_scaled_error'] = float(abs(h_expected-h_actual)/max(sum(abs(np.asarray(elements)*conn_values)), 1e-300))
        flat, unravel = ravel_pytree(params)
        direction = jnp.sin(jnp.arange(flat.size)+.7); direction /= jnp.linalg.norm(direction)
        fn = jax.jit(lambda z: projected(make_bank(unravel(z)), jnp.asarray(states)))
        _, tangent = jax.jvp(fn, (flat,), (direction,))
        epsilon = 1e-5
        fd = (fn(flat+epsilon*direction)-fn(flat-epsilon*direction))/(2*epsilon)
        checks['gradient_direction_relative_error'] = float(np.linalg.norm(fd-tangent)/max(np.linalg.norm(tangent), 1e-300))
        if geometry == 'triangular':
            from su2_nqs.benchmarks.triangular_induced_parton import (
                build_n36_pi_zero_flux_orbit, projected_pi_zero_flux_a1_amplitudes_jax)
            orbit = build_n36_pi_zero_flux_orbit()
            carrier = lambda x: projected_pi_zero_flux_a1_amplitudes_jax(x, orbit=orbit)
            # Keep comparable pointwise scale without energy fitting or query filtering.
            c = np.asarray(jax.jit(carrier)(states))
            scale = float(np.linalg.norm(c)/np.linalg.norm(values[f'network_A1_b{batch}']))
            additive = centered_additive_amplitudes(projected, carrier, bank, scale=scale)
            value = measure('carrier_centered_A1_b4', lambda p, x: additive(make_bank(p), x), params, states)
            checks['carrier_initial_replay_relative_error'] = float(np.linalg.norm(value-c)/max(np.linalg.norm(c), 1e-300))
            _, add_tangent = jax.jvp(lambda z: additive(make_bank(unravel(z)), states), (flat,), (direction,))
            checks['additive_tangent_relative_error'] = float(np.linalg.norm(add_tangent-scale*tangent)/max(np.linalg.norm(scale*tangent), 1e-300))
            # H of the same centered additive physical state, including BOTH residuals.
            def additive_H(p, x):
                prepared = plan(x)
                ratios, psi, _, _ = evaluate(make_bank(p), prepared)
                reference_ratios, reference_psi, _, _ = evaluate(bank, prepared)
                connected = x[:, None, :].repeat(len(bonds), axis=1)
                edges = jnp.asarray([(a, b) for a, b, _ in bonds]); weights = jnp.asarray([v for _, _, v in bonds])
                a, b = edges.T
                flipped = connected.at[:, jnp.arange(len(bonds)), a].set(x[:, b])
                flipped = flipped.at[:, jnp.arange(len(bonds)), b].set(x[:, a])
                unlike = x[:, a] != x[:, b]
                c0 = carrier(x); c1 = carrier(flipped.reshape(-1, rows*columns)).reshape(x.shape[0], -1)
                hc = jnp.sum(weights[None]*(jnp.where(unlike, -.25, .25)*c0[:, None]+.5*unlike*c1), axis=1)
                value = c0+scale*(psi-reference_psi)
                return (hc+scale*(ratios*psi-reference_ratios*reference_psi))/value
            additive_ratios = measure('carrier_centered_full_H_b4', additive_H, params, states)
            carrier_connections = np.asarray(jax.jit(carrier)(jnp.asarray(connected, dtype=jnp.int32)))
            carrier_expected = np.dot(elements, carrier_connections)
            checks['carrier_full_H_scaled_error'] = float(abs(additive_ratios[0]*c[0]-carrier_expected)/max(
                sum(abs(np.asarray(elements)*carrier_connections)), 1e-300))
            values['carrier_values'] = c
            checks['centered_additive_scale'] = scale
            # This is only a throughput/definition gate, NOT carrier ground-state energy.
            checks['carrier_full_H_finite'] = bool(np.isfinite(additive_ratios).all())
        passed = all(row['finite'] for row in measurements.values()) and checks['checkpoint_identical']
        for name, value in checks.items():
            if name.endswith(('relative_error', 'scaled_residual', 'scaled_error')):
                tolerance = 2e-6 if name == 'gradient_direction_relative_error' else 1e-9
                passed = passed and np.isfinite(value) and value < tolerance
        np.savez_compressed(destination/'values.npz', states=states, **values)
        meta = reduced_entry_metadata(cfg)
        result = dict(registration_sha256=reg['registration_sha256'], geometry=geometry,
            case=case, passed=bool(passed), measurements=measurements, checks=checks,
            query_counts=reports, maximum_q=cfg.tree.maximum_intermediate_q,
            multiplicity_width=cfg.tree.feature_dim,
            internal_virtual_dimension=sum(q+1 for q in range(0, cfg.tree.maximum_intermediate_q+1, 2))*cfg.tree.feature_dim,
            free_real_parameters=meta['free_real_parameters'], neural_real_parameters=int(flat.size),
            neural_is_subset_of_free_manifold=True, complete_local_schur=True,
            backend=jax.default_backend(), wall_seconds=perf_counter()-begin, peak_rss_bytes=peak_rss_bytes(),
            values_sha256=file_sha256(destination/'values.npz'), training_steps=0, new_samples=0,
            new_ED=False, remote_submitted=False, arbitrary_full_configuration_network=False)
        result['result_sha256'] = canonical_sha256(result)
        atomic_json(destination/'results.json', result); persist('complete' if passed else 'validation_failed')
        return result
    except Exception as error:
        atomic_json(destination/'failure.json', dict(type=type(error).__name__, message=str(error),
            wall_seconds=perf_counter()-begin, measurements=measurements, checks=checks))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--geometry', choices=('triangular', 'square'))
    args = parser.parse_args(); spec = json.loads(args.config.read_text())
    if args.register:
        path = ROOT/spec['output']/'registration.json'
        if path.exists(): raise FileExistsError(path)
        record = registration(spec); atomic_json(path, record); print(record['registration_sha256'])
    else:
        if args.geometry is None: parser.error('--geometry required for an isolated run')
        run(args.config, args.geometry)


if __name__ == '__main__': main()
