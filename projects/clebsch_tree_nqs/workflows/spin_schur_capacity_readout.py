"""Saved-state canonical spin spectra and fixed-physical-D capacity readout.

Reuses the historical Schur metric environment as an independent oracle, the
unchanged raw decoder, trusted coherent VAP data and existing physical H
action. No optimizer, head, new sampler, eigensolve for a ground state or GPU
job. Effective compression is not an improvement of the padded runtime.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import json
import shutil
import signal
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest, numerical_runtime_identity
from su2_nqs.models.finite_schur_decoder import (
    FiniteSchurDecoderConfig, neural_reduced_bank, free_reduced_bank, make_finite_schur_amplitude,
)
from su2_nqs.models.finite_schur_capacity import finite_bank_reduced_tree, reduced_tree_finite_bank
from su2_nqs.models.schur_metric_retraction import schur_norm_environments
from su2_nqs.su2.fusion_tree import Leaf
from su2_nqs.su2.reduced_tree_environment import (
    MultipletAllocation, reduced_tree_environments, canonical_edge_spectrum,
    gauge_transform_edge, allocate_multiplets, allocate_multiplet_weights, truncate_internal_edge,
)
from su2_nqs.benchmarks.su2_physics_protocol import saved_amplitude_region_density
from su2_nqs.benchmarks.regional_su2_readout import regional_schur_columns, su2_multiplicity_readout
from su2_nqs.benchmarks.frustrated_error_direction import full_heisenberg_graph_action
from su2_nqs.vmc.resumable_checkpoint import load_resumable_checkpoint
from .spin_decoder_cost import ROOT, physical_config, peak_rss_bytes
from .spin import _bonds

jax.config.update('jax_enable_x64', True)
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_schur_capacity_readout.json'


def _allocation(copies: dict, weights: dict[int, np.ndarray]) -> MultipletAllocation:
    checked = {int(q): int(number) for q, number in copies.items()}
    if set(checked) != set(weights) or any(not 0 <= checked[q] <= len(weights[q]) for q in weights):
        raise ValueError('one legal retained-copy count per q required')
    retained = sum(float(weights[q][:checked[q]].sum()) for q in weights)
    return MultipletAllocation(checked, sum((q+1)*number for q, number in checked.items()), retained, 1-retained)


def _allocation_json(value: MultipletAllocation) -> dict:
    return dict(copies_by_q={str(q): int(number) for q, number in value.copies_by_q.items()},
                physical_D=value.physical_bond_dimension, retained_weight=value.retained_weight,
                discarded_weight=value.discarded_weight)


def _spectrum_json(value) -> dict:
    return dict(physical_D=value.physical_bond_dimension, entropy_nats=value.entropy,
        normalized_weight_error=value.normalized_weight_error,
        sectors={str(q):dict(p=float(block.multiplicity_weights.sum()),
            sigma_eigenvalues=block.multiplicity_weights.tolist(),
            physical_eigenvalues_per_magnetic_copy=block.magnetic_eigenvalues.tolist(),
            metric_rank=block.metric_rank, stored_copies=len(block.multiplicity_weights))
                 for q, block in value.blocks.items()})


def _physical_subtree(node, cells) -> tuple[int, ...]:
    if isinstance(node, Leaf):
        return tuple(cells[node.index])
    return _physical_subtree(node.left, cells)+_physical_subtree(node.right, cells)


def _physical_projector(rho: np.ndarray, copies: dict[int, int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Whole-multiplet projector from an ACTUALLY invariant coherent RDM."""
    n = len(rho).bit_length()-1
    unitary, labels = regional_schur_columns(n)
    x = unitary.conj().T @ rho @ unitary
    projector = np.zeros_like(rho)
    sigmas = {}
    for q in range(n % 2, n+1, 2):
        indices = [[i for i, label in enumerate(labels) if label[:2] == (q, m)] for m in range(-q, q+1, 2)]
        sigma = sum(x[np.ix_(ids, ids)] for ids in indices)
        sigmas[q] = sigma
        _, vectors = np.linalg.eigh(sigma)
        keep = copies[q]
        if keep:
            selected = vectors[:, -keep:]
            for ids in indices:
                columns = unitary[:, ids] @ selected
                projector += columns @ columns.conj().T
    return projector, sigmas


def _project_state(state: np.ndarray, n: int, sites: tuple[int, ...], projector: np.ndarray) -> np.ndarray:
    rest = tuple(site for site in range(n) if site not in sites)
    permutation = sites+rest
    matrix = state.reshape((2,)*n).transpose(permutation).reshape(2**len(sites), -1)
    projected = projector @ matrix
    return projected.reshape((2,)*n).transpose(np.argsort(permutation)).ravel()


def run(config_path: Path) -> dict:
    """One <=30-minute offline readout, preserving every source checkpoint."""
    settings = json.loads(config_path.read_text())
    if (settings['task'] != 'saved_raw_schur_canonical_capacity_readout'
            or settings['wall_seconds'] > 1800 or settings['raw_physical_budget'] != 12
            or settings['target_physical_budget'] != 18 or settings['timing_repeats'] != 3):
        raise ValueError('fixed bounded canonical-capacity protocol required')
    destination = ROOT/settings['output']
    if destination.exists():
        raise FileExistsError('fresh immutable capacity readout required')
    parent = ROOT/settings['decoder_source']
    source_config = json.loads((ROOT/settings['decoder_config']).read_text())
    source_registration = json.loads((parent/'registration.json').read_text())
    if source_registration['config'] != source_config:
        raise ValueError('saved decoder geometry/config changed')
    paths = [config_path, ROOT/settings['decoder_config'], parent/'registration.json', parent/'control.npz',
             ROOT/settings['target_state'], ROOT/settings['target_results']]
    paths.extend(parent/arm/'checkpoint.pkl' for arm in ('network', 'free'))
    inputs = {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}
    sources = transitive_local_source_manifest((Path(__file__),), root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    destination.mkdir(parents=True)
    for relative in sources:
        target = destination/'source_snapshots'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
    registration = dict(config=settings, input_sha256=inputs, source_manifest=sources,
                        runtime=numerical_runtime_identity(), new_training=0, new_samples=0, new_ED=0, new_head=False)
    registration['registration_sha256'] = canonical_sha256(registration)
    atomic_json(destination/'registration.json', registration)
    start = perf_counter()
    def alarm(_signal, _frame):
        raise TimeoutError('registered thirty-minute compute cap reached')
    previous = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(settings['wall_seconds'])
    try:
        cfg = FiniteSchurDecoderConfig(replace(physical_config(source_config['rows'], source_config['columns'],
            source_config['multiplicity_width']), maximum_intermediate_q=source_config['maximum_q']),
            source_config['rows']//2, source_config['columns']//2, source_config['embedding_dim'], source_config['depth'])
        with np.load(parent/'control.npz') as arrays:
            probes = arrays['states'][:settings['cached_probe_count']]
        raw = jax.jit(jax.vmap(make_finite_schur_amplitude(cfg), in_axes=(None, 0)))
        report = {}
        arrays = {}
        banks = {}
        for arm in ('network', 'free'):
            saved = load_resumable_checkpoint(parent/arm/'checkpoint.pkl')
            if saved['run_identity']['registration_sha256'] != source_registration['registration_sha256']:
                raise ValueError('saved checkpoint identity changed')
            tick = perf_counter()
            bank = neural_reduced_bank(saved['params'], cfg) if arm == 'network' else free_reduced_bank(saved['params'], cfg)
            bank = jax.tree.map(lambda value: np.asarray(value).copy(), bank)
            generation_seconds = perf_counter()-tick
            banks[arm] = bank
            state = finite_bank_reduced_tree(bank, cfg)
            times = []
            for _ in range(settings['timing_repeats']):
                tick = perf_counter(); environment = reduced_tree_environments(state); times.append(perf_counter()-tick)
            old_grams, old_environments = schur_norm_environments(bank, cfg)
            relative_gram = relative_outside = 0.
            for node in state.tree.internal_nodes:
                for q in environment.upward[node]:
                    g, e = environment.upward[node][q], environment.downward[node][q]
                    relative_gram = max(relative_gram, float(np.linalg.norm(g-old_grams[node][q])/max(np.linalg.norm(g), 1e-300)))
                    relative_outside = max(relative_outside, float(np.linalg.norm(e-old_environments[node][q].T/(q+1))/max(np.linalg.norm(e), 1e-300)))
            edge_index = settings['raw_edge_index']; node = state.tree.internal_nodes[edge_index]
            tick = perf_counter(); spectrum = canonical_edge_spectrum(environment, node); spectrum_seconds = perf_counter()-tick
            weights = {q: block.multiplicity_weights for q, block in spectrum.blocks.items()}
            adaptive = allocate_multiplets(spectrum, settings['raw_physical_budget'])
            baseline = _allocation(settings['raw_baseline_copies'], weights)
            if adaptive.physical_bond_dimension != baseline.physical_bond_dimension:
                raise ValueError('capacity comparison must have identical physical D')
            canonical = gauge_transform_edge(state, edge_index, {q: block.gauge for q, block in spectrum.blocks.items()})
            rebuilt = gauge_transform_edge(canonical, edge_index, {q: block.gauge_inverse for q, block in spectrum.blocks.items()})
            tick = perf_counter(); old_values = np.asarray(raw(bank, jnp.asarray(probes))); first_query_seconds = perf_counter()-tick
            replay = np.asarray(raw(reduced_tree_finite_bank(rebuilt, cfg), jnp.asarray(probes)))
            reconstruction = float(np.linalg.norm(replay-old_values)/max(np.linalg.norm(old_values), 1e-300))
            compressed = {}
            for name, allocation in (('simple_same_D', baseline), ('spectrum_same_D', adaptive)):
                projected = truncate_internal_edge(state, edge_index, spectrum, allocation)
                actual = reduced_tree_environments(projected).norm_squared/environment.norm_squared
                if abs(actual-allocation.retained_weight) > 2e-9:
                    raise FloatingPointError('physical projection norm does not replay canonical weights')
                compressed[name] = dict(_allocation_json(allocation), actual_norm_ratio=actual,
                    norm_retention_error=abs(actual-allocation.retained_weight),
                    padded_bank_storage_unchanged=True, new_energy_optimization=False)
            if max(relative_gram, relative_outside, reconstruction, environment.maximum_relative_edge_norm_error,
                   spectrum.normalized_weight_error) > 2e-9:
                raise FloatingPointError('canonical environment or physical reconstruction fails')
            arrays[arm+'_probe_amplitudes'] = old_values
            arrays[arm+'_reconstructed_probe_amplitudes'] = replay
            for q, block in spectrum.blocks.items():
                arrays[f'{arm}_sigma_eigenvalues_q{q}'] = block.multiplicity_weights
                arrays[f'{arm}_canonical_gauge_q{q}'] = block.gauge
            report[arm] = dict(checkpoint_sha256=inputs[str((parent/arm/'checkpoint.pkl').relative_to(ROOT))],
                parameter_sha256=pytree_sha256(saved['params']), raw_bank_sha256=pytree_sha256(bank),
                saved_attempts=saved['completed_steps'], saved_accepted_updates=sum(bool(row.get('accepted', False)) for row in saved['history']),
                source_state='RAW decoder component of saved additive/full-A1 experiment, not its complete physical state',
                physical_cut=list(_physical_subtree(node, cfg.tree.cells)), doubled_q_cutoff=cfg.tree.maximum_intermediate_q,
                stored_chi=cfg.tree.feature_dim, spectrum=_spectrum_json(spectrum), comparisons=compressed,
                raw_tree_norm_squared=environment.norm_squared,
                environment_median_seconds=float(np.median(times)), environment_repeat_seconds=times,
                spectrum_seconds=spectrum_seconds, bank_generation_seconds=generation_seconds,
                raw_first_compile_plus_query_seconds=first_query_seconds,
                known_probe_count=len(probes), reconstructed_amplitude_relative_error=reconstruction,
                existing_metric_Gram_relative_error=relative_gram, existing_summed_m_environment_relative_error=relative_outside,
                all_edge_norm_relative_error=environment.maximum_relative_edge_norm_error,
                no_inference_of_missing_multiplicities=True, speedup_claim=False)
            print(arm, 'environment sec', round(float(np.median(times)), 4), 'p(q)',
                  {q:round(float(v.sum()), 5) for q,v in weights.items()}, 'D12', _allocation_json(adaptive), flush=True)
        same_initial_or_terminal_bank = pytree_sha256(banks['network']) == pytree_sha256(banks['free'])

        with np.load(ROOT/settings['target_state']) as source:
            basis = source['basis']; psi = source[settings['target_vector']]
        region = tuple(settings['target_region'])
        rho = saved_amplitude_region_density(psi, basis, region)
        tick = perf_counter(); readout = su2_multiplicity_readout(rho); readout_seconds = perf_counter()-tick
        weights = {int(q):np.asarray(row['unnormalized_multiplicity_spectrum']) for q,row in readout['sectors'].items()}
        adaptive = allocate_multiplet_weights(weights, settings['target_physical_budget'])
        baseline = _allocation(settings['target_baseline_copies'], weights)
        if adaptive.physical_bond_dimension != baseline.physical_bond_dimension:
            raise ValueError('trusted target capacity comparison must match physical D')
        n = basis.shape[1]; integers = basis.astype(np.int64) @ (1 << np.arange(n-1, -1, -1))
        full = np.zeros(1 << n, complex); full[integers] = psi
        geometry = json.loads((ROOT/settings['target_results']).read_text())['case']
        action = full_heisenberg_graph_action(n, _bonds(geometry['geometry'], geometry['rows'], geometry['columns'], geometry['j2_over_j1']))
        def moments(vector):
            norm = np.vdot(vector, vector).real
            hpsi = action.diagonal*vector + np.sum(action.offdiagonal_weights*vector[action.swapped_indices], axis=0)
            energy = np.vdot(vector, hpsi)/norm
            return dict(energy_per_site=float(energy.real/n), variance_per_site=float(np.vdot(hpsi-energy*vector, hpsi-energy*vector).real/norm/n))
        before = moments(full)
        target = {}
        for name, allocation in (('simple_same_D', baseline), ('spectrum_same_D', adaptive)):
            projector, sigmas = _physical_projector(rho, dict(allocation.copies_by_q))
            projected = _project_state(full, n, region, projector)
            actual_norm = float(np.vdot(projected, projected).real)
            actual_overlap = np.vdot(full, projected)
            if max(abs(actual_norm-allocation.retained_weight), abs(actual_overlap-allocation.retained_weight),
                   np.linalg.norm(projector@projector-projector)) > 2e-9:
                raise FloatingPointError('actual regional physical projector disagrees with sigma weights')
            arrays[name+'_target_projector'] = projector
            target[name] = dict(_allocation_json(allocation), actual_norm_ratio=actual_norm,
                overlap_real=float(actual_overlap.real), overlap_imaginary=float(actual_overlap.imag),
                projector_idempotency_frobenius=float(np.linalg.norm(projector@projector-projector)),
                physical_before=before, physical_after= moments(projected),
                preserves_SU2=True, full_A1_not_preserved_by_regional_compression=True,
                no_variational_training_or_energy_win_claim=True)
        arrays['target_actual_rho'] = rho
        for q, sigma in sigmas.items():
            arrays[f'target_sigma_q{q}'] = sigma
        arrays['cached_probes'] = probes
        data_path = destination/'readouts.npz'; np.savez_compressed(data_path, **arrays)
        if any(file_sha256(ROOT/relative) != digest for relative,digest in inputs.items()):
            raise AssertionError('saved input changed during readout')
        result = dict(registration_sha256=registration['registration_sha256'],
            raw_decoder_comparisons=report, raw_neural_free_banks_identical=same_initial_or_terminal_bank,
            trusted_N16_target=dict(source=settings['target_state'], vector=settings['target_vector'], sites=list(region),
                exact_physical_su2_diagnostics=readout['diagnostics'], multiplicity_spectrum=weights_to_json(weights),
                physical_readout_seconds=readout_seconds, comparisons=target,
                scope='Eight-site physical RDM of saved exact-projected N16 ViT. Not the N36 raw-tree edge or missing global virtual modes.'),
            readouts_sha256=file_sha256(data_path), new_training=0,new_samples=0,new_ED=0,new_head=False,
            full_H_action_N16_vectors=3, no_N36_H_or_new_sampling=True,
            stationarity_or_variational_win_claim=False, padded_speedup_claim=False,
            wall_seconds=perf_counter()-start, peak_rss_bytes=peak_rss_bytes(),
            capacity_choice_scope='Exact whole-multiplet compression within measured support. No automatic choice of a new trainable representation.')
        result['result_sha256'] = canonical_sha256(result)
        atomic_json(destination/'results.json', result)
        atomic_json(destination/'completion.json', dict(complete=True,results_sha256=file_sha256(destination/'results.json'),
                    registration_sha256=file_sha256(destination/'registration.json'),readouts_sha256=file_sha256(data_path)))
        print('complete', round(result['wall_seconds'], 3), 's; target D18', _allocation_json(adaptive), flush=True)
        return result
    except BaseException as error:
        atomic_json(destination/'incomplete.json', dict(type=type(error).__name__,message=str(error),wall_seconds=perf_counter()-start))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)


def weights_to_json(weights: dict[int, np.ndarray]) -> dict:
    """Plain doubled-q scalar spectra for durable results."""
    return {str(q):values.tolist() for q,values in weights.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    run(parser.parse_args().config)


if __name__ == '__main__':
    main()
