"""Reusable zero-query SU(2) protocol on sealed saved physical regional RDMs.

Reads finite-MPS quality sequences and explicitly separate infinite-iPEPS
controls. No optimization, many-body ED, MPS contraction, sampling or twirl.
The output can be regenerated in a fresh directory as new RDMs arrive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.readout_quality import physical_problem_signature
from su2_nqs.benchmarks.su2_physics_protocol import (
    multiplicity_operator_distance, regional_su2_protocol, saved_amplitude_region_density,
    rotational_symmetry_defect, rotational_defect_chain_audit, regional_reference_chain_audit,
)
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from .spin_tensor_network_reference import ROOT, _json, file_sha256, transitive_local_source_manifest

DEFAULT = ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_su2_analysis.json'


def _scalars(readout: dict) -> dict[str, float]:
    """Predeclared physical diagnostics, not a search for maximally different levels."""
    result = {}
    mult = readout['multiplicity']
    for q, sector in mult['sectors'].items():
        result[f'p_q{q}'] = sector['p']
        if sector['conditional_operator_purity'] is not None:
            result[f'operator_purity_q{q}'] = sector['conditional_operator_purity']
    result['spin_squared'] = mult['physical']['spin_squared_mean']
    for q, p in readout['partition']['singlet_pair_channels'].items():
        result[f'p_qL{q}_qR{q}_qT0'] = p
    if 'six_body' in readout:
        six = readout['six_body']
        for q, split in six['projector_splits'].items():
            for kind, value in split.items():
                result[f'projector_q{q}_{kind}'] = value
        result['connected_three_bridge_sum'] = six['three_bond']['orbits']['Y']['connected']
    for name in ('E2_intertriangle_connected_dot', 'chi_intertriangle_connected',
                 'dimer_parallel_connected_average'):
        if name in readout.get('local_spatial_channels', {}):
            result[name] = readout['local_spatial_channels'][name]
    if 'triangle_resonance' in readout:
        resonance = readout['triangle_resonance']
        result['triangle_p_doublet_doublet'] = resonance['p_doublet_doublet']
        if resonance['conditional'] is not None:
            for name in ('E2_axis_average', 'E2_axis_average_connected',
                         'A2_axis_correlation', 'A2_axis_correlation_connected'):
                result['triangle_conditional_'+name] = resonance['conditional'][name]
    return result


def _regional_scalar_data(rho: np.ndarray, cut: dict) -> tuple[dict, dict, dict]:
    """Common deterministic operator bank for NQS AND tensor-network RDMs."""
    report, data = regional_su2_protocol(rho, cut['left'],
        cyclic=cut.get('cyclic', False),
        compact_triangular_2x3=cut.get('compact_triangular_2x3', False))
    values = _scalars(report)
    for row in report['partition']['weights']:
        name = 'joint_qL{}_qR{}_qT{}'.format(row['q_left'], row['q_right'], row['q_total'])
        values[name] = row['probability']
    for q, sigma in data.items():
        for i in range(len(sigma)):
            for j in range(i, len(sigma)):
                values[f'sigma_q{q}_{i}_{j}_real'] = float(sigma[i, j].real)
                if i != j:
                    values[f'sigma_q{q}_{i}_{j}_imag'] = float(sigma[i, j].imag)
    for row in report['conventional']['spin_pairs']:
        values['spin_pair_'+ '_'.join(map(str, row['sites']))] = row['value']
    for row in report['conventional']['disjoint_connected_dimers']:
        name = '_'.join(str(i) for bond in row['bonds'] for i in bond)
        values['connected_dimer_'+name] = row['value']
    for q, weights in report['conventional'].get('cyclic', {}).get('p_q_k', {}).items():
        for k, value in enumerate(weights):
            values[f'cyclic_q{q}_k{k}'] = value
    # No entanglement spectrum is claimed from this measurement, including
    # when a particular empirical matrix passes the defect threshold.
    report['multiplicity']['physical'].pop('invariant_readout', None)
    return report, data, values


def cached_regional_readout(matrices: np.ndarray, chain_ids: np.ndarray,
                           cut: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """Apply the SAME saved-RDM protocol to coherent Born records.

    Delete whole chains, not individual correlated snapshots. Joint labels
    are doubled q=2S. Multiplicity entries use the fixed regional CG basis;
    they are invariant-operator expectations, not strict entanglement levels.
    """
    from su2_nqs.vmc.regional_density import regional_density_summary
    summary = regional_density_summary(matrices, chain_ids)

    def evaluate(rho: np.ndarray) -> tuple[dict, dict, dict]:
        return _regional_scalar_data(rho, cut)

    report, data, values = evaluate(summary['rho'])
    deleted = [evaluate(rho)[2] for rho in summary['jackknife_rhos']]
    undefined = {name: value for name, value in values.items()
                 if any(name not in row for row in deleted)}
    # Do not assign a zero/finite SE to a conditional ratio whose sector is
    # empty in any delete-chain replicate. Preserve its point value separately.
    values = {name: value for name, value in values.items() if name not in undefined}
    names = list(values)
    jack = np.array([[row[name] for name in names] for row in deleted])
    centered = jack-jack.mean(axis=0)
    covariance = (len(jack)-1)/len(jack)*(centered.T@centered)
    errors = dict(zip(names, map(float, np.sqrt(np.diag(covariance)))))
    arrays = dict(rho=summary['rho'], jackknife_rhos=summary['jackknife_rhos'],
                  per_record_matrices=matrices, chain_ids=chain_ids,
                  scalar_jackknife_values=jack, scalar_covariance=covariance,
                  **{f'sigma_q{q}': matrix for q, matrix in data.items()})
    return dict(sites=cut['sites'], left=cut['left'], protocol=report,
                scalar_names=names, scalars=values, scalar_whole_chain_SE=errors,
                diagnostics={k:v for k,v in summary.items() if k not in ('rho','jackknife_rhos')},
                conditional_coherent_Born_estimator=True,
                conditional_ratios_use_whole_chain_jackknife=True,
                undefined_jackknife_scalars=undefined,
                strict_multiplicity_spectrum_reported=False, input_repaired=False), arrays


def analyze(config: Path, output: Path | None = None) -> dict:
    """Read immutable caches and compare SAME-H/cut quality trajectories.

    MPS contraction has no Monte-Carlo SE. Successive-chi changes are reported
    as empirical drifts, never certified error bars. Initialization comparisons
    are quality diagnostics, not competing-phase assignments.
    """
    spec = json.loads(config.read_text())
    if spec.get('mode') == 'unified_quality_readout':
        return analyze_unified_quality_readout(config, output)
    if spec.get('mode') == 'readout_controls':
        return analyze_readout_controls(config, output)
    if spec.get('mode') == 'multipartite_witness_replay':
        return analyze_multipartite_witness_replay(config, output)
    if spec.get('mode') == 'multipartite_witness':
        return analyze_multipartite_witness(config, output)
    if spec.get('mode') == 'invariant_maximum_entropy':
        return analyze_invariant_maximum_entropy(config, output)
    if spec.get('mode') == 'cached_quality_sequences':
        return analyze_cached_quality_sequences(config, output)
    if spec.get('mode') == 'saved_symmetry_defect':
        return analyze_symmetry_defect(config, output)
    if spec.get('mode') == 'saved_sequences':
        return analyze_saved_sequences(config, output)
    if spec.get('mode') == 'cached_candidate_reference':
        return analyze_cached_candidate_reference(config, output)
    output = output or ROOT/spec['output']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use a fresh output directory; completed audits are immutable')
    inputs, sources = {}, []
    for source in spec['artifacts']:
        directory = ROOT/source['path']
        record = json.loads((directory/'results.json').read_text())
        for name in ('results.json', 'rdms.npz'):
            path = directory/name
            inputs[str(path.relative_to(ROOT))] = file_sha256(path)
        if file_sha256(directory/'rdms.npz') != record['rdms_sha256']:
            raise ValueError(f'raw RDM seal mismatch: {directory}')
        if source['kind'] == 'finite_dmrg':
            cfg = record['registration']['config']
            if cfg['regions'] != spec['regions']:
                raise ValueError('finite physical cut/site ordering mismatch')
            if (cfg['rows'], cfg['columns'], cfg['j2']) != (6, 6, .125):
                raise ValueError('this registration is triangular6x6/PBC,j=.125 only')
            signature = physical_problem_signature(36, record['registration']['bonds'],
                geometry='triangular120_6x6_row_major', boundary='PBC_both_physical')
        elif source['kind'] == 'infinite_ipeps_control':
            signature = None
        else:
            raise ValueError('unknown physical artifact kind')
        sources.append((source, directory, record, signature))
    finite_signatures = {sig for _, _, _, sig in sources if sig is not None}
    if len(finite_signatures) != 1:
        raise ValueError('finite artifacts do not share the same labelled physical Hamiltonian')
    globals_by_stage = {}
    for relative in spec.get('global_audits', []):
        path = ROOT/relative
        audit = json.loads(path.read_text())
        inputs[relative] = file_sha256(path)
        if isinstance(audit['records'], list):
            for item in audit['records']:
                if item.get('kind') == 'finite6x6_PBC_MPS':
                    family = Path(item['artifact']).name
                    globals_by_stage[family, item['final_chi']] = item['global_spin']
        else:
            for family, item in audit['records'].items():
                globals_by_stage[family, 1024] = item['global_spin']
    registry = dict(
        config=spec, config_sha256=file_sha256(config), inputs=inputs,
        symmetry_tolerance=1e-9, physical_problem_signature=next(iter(finite_signatures)),
        sources=transitive_local_source_manifest((Path(__file__),), root=ROOT,
            package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs')),
        operator_protocol='actual coherent rho; no symmetry repair; sigma_q is invariant-operator data, not a strict spectrum for noninvariant rho',
        statistical_protocol='no new sampling; DMRG refinements are deterministic empirical drift, NOT SE or certified bias bounds',
    )
    registry['sources']['su2_nqs/frontier/provenance.py'] = file_sha256(ROOT/'su2_nqs/frontier/provenance.py')
    output.mkdir(parents=True, exist_ok=True)
    for relative, digest in registry['sources'].items():
        # Seal old/live dependencies without editing historical registrations.
        target = output/'source_snapshots'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
        if file_sha256(target) != digest:
            raise RuntimeError('analysis source changed during registration')
    _json(output/'registration.json', registry)
    start, records, matrices, raw = perf_counter(), [], {}, {}
    for source, directory, record, signature in sources:
        finite = source['kind'] == 'finite_dmrg'
        stages = record.get('history', record.get('records'))
        if stages is None:
            stages = [dict(chi_max=record['actual_chi'], energy_per_site=record['energy_per_site'],
                           max_discarded_weight=record['maximum_discarded_weight'],
                           sweeps=record['completed_sweeps'])]
        with np.load(directory/'rdms.npz') as archive:
            for stage in stages:
                chi = int(stage['chi_max'] if finite else stage['chi'])
                item = dict(family=source['family'], kind=source['kind'], chi=chi,
                    artifact=source['path'], energy_per_site=stage['energy_per_site'],
                    variance_measured=False, physical_problem_signature=signature,
                    global_spin=globals_by_stage.get((source['family'], chi)),
                    regions={}, scalars={},
                    quality=dict(sweeps=stage.get('sweeps'), maximum_discarded_weight=stage.get('max_discarded_weight')))
                cuts = spec['regions'] if finite else {'compact4': spec['regions']['compact4']}
                for name in cuts:
                    key = f'chi{chi}_{name}' if finite else f'chi{chi}_physical_{name}'
                    rho = np.array(archive[key])
                    left = spec['left_sites'][name]
                    measured, sigma = regional_su2_protocol(rho, left,
                        cyclic=name == 'winding6', compact_triangular_2x3=name == 'compact6')
                    item['regions'][name] = measured
                    item['scalars'][name] = _scalars(measured)
                    tag = f'{source["family"]}_chi{chi}_{name}'
                    raw[source['family'], chi, name] = rho
                    matrices[tag+'_raw_rho'] = rho
                    for q, matrix in sigma.items():
                        matrices[tag+f'_sigma_q{q}'] = matrix
                records.append(item)
        print('measured cached', source['family'], directory.name, flush=True)
    records.sort(key=lambda item: (item['family'], item['chi']))
    sequences = {}
    for family in sorted({item['family'] for item in records}):
        stages = [item for item in records if item['family'] == family]
        changes = []
        for a, b in zip(stages[:-1], stages[1:]):
            region_changes = {}
            for name in set(a['regions']) & set(b['regions']):
                region_changes[name] = dict(
                    distance=multiplicity_operator_distance(raw[family, a['chi'], name], raw[family, b['chi'], name]),
                    scalar_changes={key: b['scalars'][name][key]-value
                                    for key, value in a['scalars'][name].items() if key in b['scalars'][name]},
                )
            changes.append(dict(chi_a=a['chi'], chi_b=b['chi'],
                energy_change_per_site=b['energy_per_site']-a['energy_per_site'], regions=region_changes))
        sequences[family] = changes
    comparisons = []
    for chi in spec.get('initialization_comparison_chis', []):
        families = ['dimer', 'stripe']
        endpoints, previous = [], []
        for family in families:
            series = [item for item in records if item['family'] == family and item['chi'] <= chi]
            if len(series) < 2 or series[-1]['chi'] != chi:
                raise ValueError('both initialization endpoints need declared own refinement history')
            endpoints.append(series[-1]); previous.append(series[-2])
        regions = {}
        for name in spec['regions']:
            changes = {}
            for key, value in endpoints[0]['scalars'][name].items():
                difference = abs(endpoints[1]['scalars'][name][key]-value)
                drifts = [abs(endpoints[i]['scalars'][name][key]-previous[i]['scalars'][name][key])
                          for i in range(2)]
                changes[key] = dict(difference=difference, own_refinement_drifts=drifts,
                    drift_sum=sum(drifts), exceeds_drift_sum=difference > sum(drifts)+1e-12)
            regions[name] = dict(
                distance=multiplicity_operator_distance(raw['dimer', chi, name], raw['stripe', chi, name]),
                scalar_comparisons=changes,
            )
        comparisons.append(dict(chi=chi, regions=regions, physically_distinct_families=False,
            warning='Two initializations of ONE DMRG method, not two phases; unequal energies and unresolved convergence prevent competition claims.'))
    np.savez_compressed(output/'readouts.npz', **matrices)
    result = dict(registration=registry, records=records, quality_sequences=sequences,
        initialization_diagnostics=comparisons, seconds=perf_counter()-start,
        readouts_sha256=file_sha256(output/'readouts.npz'),
        new_training=0, new_ED=0, new_samples=0, new_MPS_contractions=0, new_H_calls=0,
        phase_claim=False, globally_strict_SU2_claim=False,
        disposition='Calibrate invariant operator data and own-quality sensitivity. No late-sweep RDM convergence or competing phases established. Chi2048 global K/M/S2 and H2 variance are not available locally; do not infer them from regional RDMs.')
    _json(output/'results.json', result)
    _json(output/'completion.json', dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'), readouts_sha256=result['readouts_sha256']))
    return result


def _problem_signature(problem: dict, *, maximum_region_sites: int = 6) -> tuple[str, list]:
    """Register labelled finite Hamiltonian/cuts; never compare across problems."""
    if maximum_region_sites not in (6,8):
        raise ValueError('six-site protocol or explicit eight-site raw acquisition only')
    rows, columns = problem['rows'], problem['columns']
    if (problem['boundary'] != 'PBC_both_physical'
            or problem['geometry'] not in ('triangular', 'square')):
        raise ValueError('saved-sequence analysis requires a declared finite PBC geometry')
    builder = triangular_j1j2_bonds if problem['geometry'] == 'triangular' else square_j1j2_bonds
    bonds = builder(rows, columns, j2=problem['j2'], periodic=True)
    signature = physical_problem_signature(rows*columns, bonds,
        geometry=f'{problem["geometry"]}_{rows}x{columns}_row_major',
        boundary=problem['boundary'])
    if set(problem['regions']) != set(problem['left_sites']):
        raise ValueError('each physical cut needs its own explicit bipartition')
    for name, cut in problem['regions'].items():
        if (len(set(cut)) != len(cut) or not 2 <= len(cut) <= maximum_region_sites
                or any(isinstance(i, bool) or not isinstance(i, int)
                       or not 0 <= i < rows*columns for i in cut)):
            raise ValueError('invalid ordered physical cut')
        if name in problem.get('compact_triangular_2x3', []):
            row, col = divmod(cut[0], columns)
            expected = [((row+dr) % rows)*columns+(col+dc) % columns
                        for dr in range(2) for dc in range(3)]
            if problem['geometry'] != 'triangular' or cut != expected:
                raise ValueError('local triangular channels require the actual ordered 2x3 patch')
        if name in problem.get('cyclic_regions', []):
            row, col = divmod(cut[0], columns)
            if len(cut) != columns or cut != [row*columns+(col+i) % columns for i in range(columns)]:
                raise ValueError('cyclic projector requires a complete physical row in cyclic order')
    return signature, [list(b) for b in bonds]


def _saved_sequence_inputs(spec: dict, root: Path) -> tuple[list[dict], dict[str, str]]:
    """Load sealed completed sweeps or exact neural states with UNIQUE state IDs.

    Multiple sweeps at one chi are never overwritten or mistaken for distinct
    candidate phases. Partial sweeps / unchecked NPZ files fail closed.
    """
    loaded, inputs = [], {}
    problems = {name: _problem_signature(p) for name, p in spec['problems'].items()}

    def read(path: Path) -> dict:
        inputs[str(path.relative_to(root))] = file_sha256(path)
        return json.loads(path.read_text())

    for source in spec['artifacts']:
        directory = root/source['path']
        problem = spec['problems'][source['problem']]
        signature, bonds = problems[source['problem']]
        common = dict(family=source['family'], problem=source['problem'],
                      physical_problem_signature=signature, artifact=source['path'], kind=source['kind'])
        if source['kind'] == 'saved_vit':
            record = read(directory/'results.json')
            if record['arm'] != 'vap' or not record['all_parameters_trained']:
                raise ValueError('this saved-amplitude adapter requires full-network VAP trials')
            array_path = directory/'states.npz'
            inputs[str(array_path.relative_to(root))] = file_sha256(array_path)
            if inputs[str(array_path.relative_to(root))] != record['physical_states_sha256']:
                raise ValueError('saved neural amplitude seal mismatch')
            case = record['case']
            if (case['geometry'], case['rows'], case['columns'], case['j2_over_j1']) != (
                    problem['geometry'], problem['rows'], problem['columns'], problem['j2']):
                raise ValueError('neural Hamiltonian/geometry registration mismatch')
            with np.load(array_path) as archive:
                basis = np.array(archive['basis'])
                if basis.shape[1] != problem['rows']*problem['columns']:
                    raise ValueError('saved basis dimension differs from physical registration')
                for stage in spec['saved_vit_stages']:
                    metrics = record[stage['metrics']]
                    psi = np.array(archive[stage['state']])
                    rdms = {name: saved_amplitude_region_density(psi, basis, sites)
                            for name, sites in problem['regions'].items()}
                    # Independent old four-site readouts must replay; six-site
                    # coherent partial traces are NEW analysis, not new states.
                    for cut in metrics.get('cuts', {}).values():
                        match = next((name for name, sites in problem['regions'].items()
                                      if sites == cut['sites']), None)
                        if match is not None:
                            report, _ = regional_su2_protocol(rdms[match], problem['left_sites'][match])
                            for q, p in cut['q_probabilities'].items():
                                if abs(report['multiplicity']['sectors'][q]['p']-p) > 1e-11:
                                    raise ValueError('saved physical four-site probabilities did not replay')
                    loaded.append(dict(**common, state_id=stage['state'], order=stage['order'],
                        energy_per_site=metrics['energy_per_site'], variance_per_site=metrics['variance_per_site'],
                        variance_measured=True, quality=dict(stage=stage['description'],
                            energy_error_per_site=metrics['energy_error_per_site'],
                            global_spin_squared=metrics['spin_squared'],
                            global_singlet_residual=metrics['singlet_residual_norm'],
                            parameter_count=record['parameter_count'], all_parameters_trained=record['all_parameters_trained']),
                        rdms=rdms))
        elif source['kind'] in ('finite_dmrg_anchor', 'dmrg_completed_sweeps'):
            registration = read(directory/'registration.json')
            cfg = registration['config']
            if ((cfg['rows'], cfg['columns'], cfg['j2']) != (problem['rows'], problem['columns'], problem['j2'])
                    or cfg['regions'] != problem['regions'] or registration['bonds'] != bonds):
                raise ValueError('DMRG physical Hamiltonian/cut registration mismatch')
            if source['kind'] == 'finite_dmrg_anchor':
                record = read(directory/'results.json')
                if record['status'] != 'complete':
                    raise ValueError('anchor must be a completed DMRG artifact')
                pairs = [(directory/'results.json', directory/'rdms.npz', record,
                          record['actual_chi'], record['completed_sweeps'])]
            else:
                paths = sorted((directory/'sweep_readouts').glob('sweep*.json'))
                if not paths:
                    raise FileNotFoundError(f'no completed-sweep readouts: {directory}')
                pairs = []
                for path in paths:
                    record = read(path)
                    sweep = record['global_sweep']
                    if (path.name != f'sweep{sweep:04d}.json' or record['chi'] != cfg['chi']
                            or record.get('symmetry_repair') is not False or sweep < 1):
                        raise ValueError('invalid completed-sweep boundary metadata')
                    pairs.append((path, path.with_suffix('.npz'), record, record['chi'], sweep))
            for _, array_path, record, chi, sweep in pairs:
                if not np.isfinite(record['energy_per_site']):
                    raise ValueError('completed sweep requires a finite measured energy')
                inputs[str(array_path.relative_to(root))] = file_sha256(array_path)
                if inputs[str(array_path.relative_to(root))] != record['rdms_sha256']:
                    raise ValueError('completed-sweep raw RDM seal mismatch')
                with np.load(array_path) as archive:
                    rdms = {name: np.array(archive[f'chi{chi}_{name}']) for name in problem['regions']}
                loaded.append(dict(**common, state_id=f'chi{chi}_sweep{sweep:04d}', order=sweep,
                    chi=chi, global_sweep=sweep, energy_per_site=record['energy_per_site'],
                    variance_per_site=None, variance_measured=False, quality=dict(
                        maximum_discarded_weight=record.get('final_sweep_max_discarded_weight', record.get('maximum_discarded_weight')),
                        canonical_norm_error=record.get('canonical_norm_error'),
                        sweep_energy_closure_error=record.get('sweep_energy_closure_error')), rdms=rdms))
        else:
            raise ValueError('unsupported saved-sequence kind')
    keys = [(item['problem'], item['family'], item['state_id']) for item in loaded]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate state IDs; repeated chi needs unique global-sweep IDs')
    return loaded, inputs


def analyze_saved_sequences(config: Path, output: Path | None = None) -> dict:
    """Apply the SAME raw-rho protocol to late DMRG sweeps and credible ViTs.

    No statistical SE is manufactured for exact-support states. Independent
    seeds diagnose quality reproducibility, never different candidate phases.
    Different finite problems/cuts are registered separately and NOT compared.
    """
    spec = json.loads(config.read_text())
    output = output or ROOT/spec['output']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use a fresh output directory; completed audits are immutable')
    loaded, inputs = _saved_sequence_inputs(spec, ROOT)
    sources = transitive_local_source_manifest((Path(__file__),), root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    sources['su2_nqs/frontier/provenance.py'] = file_sha256(ROOT/'su2_nqs/frontier/provenance.py')
    registry = dict(config=spec, config_sha256=file_sha256(config), inputs=inputs,
        sources=sources, symmetry_tolerance=1e-9,
        problem_signatures={key: _problem_signature(p)[0] for key, p in spec['problems'].items()},
        protocol='Actual coherent rho, no repair. Same regional_su2_protocol for MPS and ViT; strict spectra only after actual SU2 checks. Refinement drift is NOT a certified error bar.')
    output.mkdir(parents=True, exist_ok=True)
    for relative, digest in sources.items():
        target = output/'source_snapshots'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
        if file_sha256(target) != digest:
            raise RuntimeError('analysis source changed during registration')
    _json(output/'registration.json', registry)
    start, records, matrices, raw = perf_counter(), [], {}, {}
    for loaded_item in loaded:
        item = {key: value for key, value in loaded_item.items() if key != 'rdms'}
        item.update(regions={}, scalars={})
        problem = spec['problems'][item['problem']]
        for name, rho in loaded_item['rdms'].items():
            report, sigma = regional_su2_protocol(rho, problem['left_sites'][name],
                cyclic=name in problem.get('cyclic_regions', []),
                compact_triangular_2x3=name in problem.get('compact_triangular_2x3', []))
            item['regions'][name], item['scalars'][name] = report, _scalars(report)
            key = (item['problem'], item['family'], item['state_id'], name)
            tag = '__'.join(key)
            raw[key], matrices[tag+'_raw_rho'] = rho, rho
            for q, reduced in sigma.items():
                matrices[tag+f'_sigma_q{q}'] = reduced
        records.append(item)
    records.sort(key=lambda r: (r['problem'], r['family'], r['order'], r['state_id']))

    def compare(a: dict, b: dict) -> dict:
        if (a['physical_problem_signature'] != b['physical_problem_signature'] or a['problem'] != b['problem']):
            raise ValueError('different physical problems/cut registrations cannot be compared')
        return dict(state_a=a['state_id'], state_b=b['state_id'],
            energy_change_per_site=b['energy_per_site']-a['energy_per_site'],
            regions={name: dict(distance=multiplicity_operator_distance(
                raw[a['problem'], a['family'], a['state_id'], name],
                raw[b['problem'], b['family'], b['state_id'], name]),
                scalar_changes={key: b['scalars'][name][key]-value
                                for key, value in a['scalars'][name].items() if key in b['scalars'][name]})
                for name in a['regions']})

    sequences, replicas = {}, []
    for problem_id in spec['problems']:
        endpoints = []
        for family in sorted({r['family'] for r in records if r['problem'] == problem_id}):
            stages = [r for r in records if r['family'] == family and r['problem'] == problem_id]
            sequences[f'{problem_id}/{family}'] = [compare(a, b) for a, b in zip(stages[:-1], stages[1:])]
            endpoints.append(stages[-1])
        # All saved seeds, not cherry-picked spectra or maximally different cuts.
        for i, a in enumerate(endpoints):
            for b in endpoints[i+1:]:
                replicas.append(dict(problem=problem_id, family_a=a['family'], family_b=b['family'],
                    **compare(a, b), physically_distinct_families=False,
                    warning='Independent initializations of ONE neural family, not two phases.'))
    np.savez_compressed(output/'readouts.npz', **matrices)
    result = dict(registration=registry, records=records, quality_sequences=sequences,
        replica_diagnostics=replicas, readouts_sha256=file_sha256(output/'readouts.npz'),
        seconds=perf_counter()-start, new_training=0, new_ED=0, new_samples=0,
        new_MPS_contractions=0, new_H_calls=0, phase_claim=False,
        disposition='Quality/readout calibration only. Completed sweeps retain unique IDs at fixed chi. No inference across sizes/regions; no competitive N36 neural or competing-phase claim.')
    _json(output/'results.json', result)
    _json(output/'completion.json', dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'), readouts_sha256=result['readouts_sha256']))
    return result


def analyze_cached_quality_sequences(config: Path, output: Path | None = None) -> dict:
    """Compare independent NQS refinements and deterministic DMRG refinements.

    Uses saved coherent RDMs only. Scalar contrasts preserve signed values;
    quality-drift screens are empirical, NOT bias bounds or simultaneous
    significance/phase tests. Whole-chain quadratic noise correction is
    reported separately from noisy plug-in matrix distances.
    """
    from .spin_spatial_vit_quality import read_sealed
    from su2_nqs.benchmarks.readout_quality import compare_with_refinement
    spec = json.loads(config.read_text())
    output = output or ROOT/spec['output']
    if output.exists():
        raise FileExistsError('fresh immutable quality comparison required')
    start, inputs = perf_counter(), {}

    def checked(path: Path, expected: str | None = None) -> str:
        path = path.resolve()
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError('changed sealed input: '+str(path))
        inputs[str(path.relative_to(ROOT))] = digest
        return digest

    signature, _ = _problem_signature(spec['problem'])
    candidates = []
    for relative in spec['candidates']:
        directory = ROOT/relative
        registration = read_sealed(directory/'registration.json')
        result = read_sealed(directory/'results.json')
        if (result['registration_sha256'] != registration['result_sha256']
                or result['physical_problem_signature'] != signature):
            raise ValueError('candidate labelled Hamiltonian/registration differs')
        for name in ('registration.json', 'results.json'):
            checked(directory/name)
        checked(directory/'physics.npz', result['raw_sha256'])
        for name, digest in registration['inputs_sha256'].items():
            checked(Path(name), digest)
        for name, digest in registration['source_manifest'].items():
            checked(directory/'source_snapshots'/name, digest)
        for cut, region in result['regions'].items():
            if (region['sites'] != spec['problem']['regions'][cut]
                    or region['left'] != spec['problem']['left_sites'][cut]):
                raise ValueError('candidate physical region/site order differs')
        with np.load(directory/'physics.npz') as data:
            arrays = {k:np.array(data[k]) for k in data.files}
        candidates.append((result, arrays))
    if len(candidates) != 2 or candidates[0][0]['step'] >= candidates[1][0]['step']:
        raise ValueError('declare exactly two increasing independent quality endpoints')
    directory = ROOT/spec['reference_protocol']
    checked(directory/'completion.json')
    seal = json.loads((directory/'completion.json').read_text())
    for name, key in [('results.json','results_sha256'), ('registration.json','registration_sha256'),
                      ('readouts.npz','readouts_sha256')]:
        checked(directory/name, seal[key])
    reference = json.loads((directory/'results.json').read_text())
    reference_problem = reference['registration']['config']['problems'][spec['reference_problem']]
    if reference_problem != spec['problem']:
        # Extra calibrated cuts are permitted; every compared cut/partition
        # and channel flag must agree, as must the labelled Hamiltonian.
        if _problem_signature(reference_problem)[0] != signature:
            raise ValueError('reference physical Hamiltonian differs')
        for cut in spec['problem']['regions']:
            for field in ('regions','left_sites'):
                if reference_problem[field][cut] != spec['problem'][field][cut]:
                    raise ValueError('reference physical cut/partition differs')
            for field in ('compact_triangular_2x3','cyclic_regions'):
                if (cut in reference_problem.get(field, [])) != (cut in spec['problem'].get(field, [])):
                    raise ValueError('reference geometric channel routing differs')
    rows = sorted((r for r in reference['records'] if r['problem']==spec['reference_problem']
                   and r['family']==spec['reference_family']), key=lambda r:r['order'])
    ids = [r['state_id'] for r in rows]
    if (spec['reference_endpoint'] != ids[-1]
            or not set(spec['reference_refinements']).issubset(ids)):
        raise ValueError('declared reference endpoint/refinements not available')
    results = {}
    with np.load(directory/'readouts.npz') as archive:
        for cut in spec['problem']['regions']:
            cut_spec = dict(sites=spec['problem']['regions'][cut],left=spec['problem']['left_sites'][cut],
                cyclic=cut in spec['problem'].get('cyclic_regions', []),
                compact_triangular_2x3=cut in spec['problem'].get('compact_triangular_2x3', []))
            prefix = '__'.join((spec['reference_problem'],spec['reference_family']))+'__'
            rhos, deterministic = {}, []
            for row in rows:
                rho = np.array(archive[prefix+row['state_id']+'__'+cut+'_raw_rho'])
                report, _, scalars = _regional_scalar_data(rho,cut_spec)
                rhos[row['state_id']] = rho
                deterministic.append(dict(state_id=row['state_id'],chi=row['chi'],
                    global_sweep=row['global_sweep'],energy_per_site=row['energy_per_site'],
                    scalars=scalars,strict_su2_eligible=report['multiplicity']['physical']['strict_su2_eligible']))
            old, new = [r['regions'][cut] for r,_ in candidates]
            ref = deterministic[-1]['scalars']
            refinements = [r['scalars'] for r in deterministic if r['state_id'] in spec['reference_refinements']]
            comparison = {}
            for name in new['scalars']:
                if name not in old['scalars'] or name not in ref:
                    continue
                v0, v1 = old['scalars'][name], new['scalars'][name]
                s0, s1 = old['scalar_whole_chain_SE'][name], new['scalar_whole_chain_SE'][name]
                ds = float(np.hypot(s0,s1))
                drift = v1-v0
                dref = max(abs(ref[name]-r[name]) for r in refinements)
                contrast = v1-ref[name]
                excess = abs(contrast)-abs(drift)-dref
                comparison[name] = dict(vit_before=v0,vit_after=v1,vit_before_SE=s0,vit_after_SE=s1,
                    reference=ref[name],between_family_signed=contrast,between_family_SE=s1,
                    vit_quality_signed_change=drift,vit_quality_independent_SE=ds,
                    dmrg_quality_envelope=dref,point_excess_over_quality_drifts=excess,
                    conservative_two_SE_screen=bool(excess > 2*(s1+ds)),
                    paired_quality_endpoints=False)
            rho = candidates[1][1][cut+'__rho']
            fixed = rhos[spec['reference_endpoint']]
            results[cut] = dict(scalars=comparison,dmrg_quality_sequence=deterministic,
                primary_observables=spec['primary_observables'][cut],
                primary_passing_screens=[n for n in spec['primary_observables'][cut]
                    if comparison[n]['conservative_two_SE_screen']],
                raw_and_invariant_noise_audit=regional_reference_chain_audit(
                    candidates[1][1][cut+'__per_record_matrices'],candidates[1][1]['chain_ids'],fixed),
                empirical_matrix_refinement=compare_with_refinement(rho,fixed,
                    refinements_a=[candidates[0][1][cut+'__rho']],
                    refinements_b=[rhos[s] for s in spec['reference_refinements']]))
    sources = transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    registry = dict(config=spec,config_sha256=checked(config),inputs=inputs,sources=sources,
        physical_problem_signature=signature,
        statistical_protocol='Independent frozen endpoints, whole-chain jackknife. Deterministic DMRG drift is empirical sensitivity, not SE or bias bound. Pointwise conservative two-SE screen is NOT a simultaneous significance/phase test; covariance of scalar features is saved with each endpoint.')
    output.mkdir(parents=True)
    for name in sources:
        target = output/'source_snapshots'/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    _json(output/'registration.json',registry)
    result = dict(registration=registry,regions=results,seconds=perf_counter()-start,
        new_training=0,new_samples=0,new_H_calls=0,new_amplitude_queries=0,new_MPS_contractions=0,
        strict_spectrum_reported=False,phase_claim=False,
        disposition='Quality-controlled scalar/resonance comparison, not competing-phase identification. A passing scalar screen alone would not distinguish lower-order explanations, nonsinglet contamination or ansatz bias.')
    _json(output/'results.json',result)
    _json(output/'completion.json',dict(results_sha256=file_sha256(output/'results.json'),
                                      registration_sha256=file_sha256(output/'registration.json')))
    return result


def analyze_cached_candidate_reference(config: Path, output: Path | None = None) -> dict:
    """Compare a frozen coherent Born cache against SAME-H/cut MPS refinements.

    Distinct whole-chain pairs remove squared-distance diagonal noise bias.
    This is not a phase test: a single neural endpoint has no own refinement
    series, and deterministic MPS drift is not a certified reference error.
    No amplitudes, samples, Hamiltonian queries or MPS contractions are made.
    """
    config = config.resolve()
    spec = json.loads(config.read_text())
    output = output or ROOT/spec['output']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use a fresh output directory; completed audits are immutable')
    start, inputs = perf_counter(), {}

    def checked(path: Path, expected: str | None = None) -> str:
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'sealed input changed: {path}')
        inputs[str(path.relative_to(ROOT))] = digest
        return digest

    def canonical(path: Path) -> dict:
        checked(path)
        data = json.loads(path.read_text())
        if data.get('result_sha256') != canonical_sha256({k:v for k,v in data.items() if k!='result_sha256'}):
            raise ValueError(f'canonical record seal mismatch: {path}')
        return data

    candidate_dir = ROOT/spec['candidate']
    candidate_registration = canonical(candidate_dir/'registration.json')
    candidate = canonical(candidate_dir/'results.json')
    canonical(candidate_dir/'timing.json')
    if (candidate['registration_sha256'] != candidate_registration['result_sha256']
            or candidate['params_sha256'] != spec['params_sha256']
            or candidate['step'] != spec['step']
            or not candidate['region']['conditional_coherent_Born_estimator']):
        raise ValueError('candidate must be the declared frozen coherent Born audit')
    checked(candidate_dir/'physics.npz', candidate['raw_sha256'])
    for relative, digest in candidate_registration['source_manifest'].items():
        checked(candidate_dir/'source_snapshots'/relative, digest)
    parent_dir = ROOT/spec['candidate_parent']
    parent = canonical(parent_dir/'registration.json')
    for relative, digest in candidate_registration['frozen_parent_source_manifest'].items():
        checked(parent_dir/'source_snapshots'/relative, digest)
    for path, digest in candidate_registration['inputs_sha256'].items():
        remote = candidate_registration['config']['parent']+'/'
        local = parent_dir/path[len(remote):] if path.startswith(remote) else ROOT/path
        checked(local, digest)
    model = parent['config']['model']
    problem = spec['problem']
    if (parent['config']['coordinate_convention'] != 'physical120'
            or (model['geometry'], model['rows'], model['columns'], parent['config']['j2_over_j1'])
              != (problem['geometry'], problem['rows'], problem['columns'], problem['j2'])):
        raise ValueError('candidate labelled Hamiltonian differs from reference registration')
    cut = spec['cut']
    if candidate['region']['sites'] != problem['regions'][cut]:
        raise ValueError('candidate/reference physical cut or site ordering differs')
    signature, _ = _problem_signature(problem)
    reference_dir = ROOT/spec['reference_protocol']
    checked(reference_dir/'completion.json')
    seal = json.loads((reference_dir/'completion.json').read_text())
    for name, key in [('results.json','results_sha256'),('registration.json','registration_sha256'),
                      ('readouts.npz','readouts_sha256')]:
        checked(reference_dir/name,seal[key])
    reference = json.loads((reference_dir/'results.json').read_text())
    for relative, digest in reference['registration']['inputs'].items():
        checked(ROOT/relative,digest)
    for relative, digest in reference['registration']['sources'].items():
        checked(reference_dir/'source_snapshots'/relative,digest)
    reference_problem = reference['registration']['config']['problems'][spec['reference_problem']]
    if (_problem_signature(reference_problem)[0] != signature
            or reference_problem['regions'][cut] != problem['regions'][cut]
            or reference_problem['left_sites'][cut] != problem['left_sites'][cut]):
        raise ValueError('reference physical Hamiltonian/cut/partition differs')
    with np.load(candidate_dir/'physics.npz') as cache:
        matrices, chains = np.array(cache['per_record_matrices']), np.array(cache['chain_ids'])
        rho, jackknife = np.array(cache['rho']), np.array(cache['jackknife_rhos'])
        scalar, zz = np.array(cache['scalar']), np.array(cache['zz'])
        if (list(cache['region_sites']) != problem['regions'][cut]
                or matrices.shape[0] != candidate['records']
                or len(np.unique(chains)) != candidate['independent_chains']
                or not np.allclose(matrices.mean(axis=0),rho,atol=1e-13,rtol=0)):
            raise ValueError('candidate coherent records do not replay their declared RDM')
    left = problem['left_sites'][cut]
    report, _ = regional_su2_protocol(rho,left)
    candidate_scalars = _scalars(report)
    deleted = [_scalars(regional_su2_protocol(matrix,left)[0]) for matrix in jackknife]
    scalar_errors = {key:float(np.sqrt((len(deleted)-1)*np.var([item[key] for item in deleted])))
                     for key in candidate_scalars}
    # Use the predeclared physical M orbit and scalar-vs-3zz contrasts only;
    # the same chain carries both estimates, so covariance is retained.
    names = list(parent['config']['momenta'])
    ids = np.unique(chains)
    scalar_means = np.stack([scalar[chains==c].mean(axis=0) for c in ids])
    zz_means = np.stack([zz[chains==c].mean(axis=0) for c in ids])
    def linear_record(values: np.ndarray) -> dict:
        return dict(value=float(values.real.mean()),
                    whole_chain_SE=float(values.real.std(ddof=1)/np.sqrt(len(values))))
    structure = dict(M_scalar_orbit_mean=linear_record(scalar_means[:,[names.index(k) for k in ('M1','M2','M3')]].mean(axis=1)),
        scalar_minus_three_zz={name:linear_record(scalar_means[:,i]-3*zz_means[:,i])
                              for i,name in enumerate(names)})
    rows = [r for r in reference['records'] if r['problem']==spec['reference_problem']
            and r['family']==spec['reference_family'] and r['state_id'] in spec['reference_states']]
    rows.sort(key=lambda r:r['order'])
    if len(rows)!=len(spec['reference_states']):
        raise ValueError('all declared deterministic quality nodes must exist exactly once')
    comparisons, quality, previous_rho, previous = [], [], None, None
    with np.load(reference_dir/'readouts.npz') as archive:
        for row in rows:
            key='__'.join((row['problem'],row['family'],row['state_id'],cut))+'_raw_rho'
            fixed=np.array(archive[key])
            fixed_scalars=_scalars(regional_su2_protocol(fixed,left)[0])
            contrasts={name:dict(value=value-fixed_scalars[name],whole_chain_SE=scalar_errors[name],
                                  signal_over_SE=(value-fixed_scalars[name])/scalar_errors[name] if scalar_errors[name]>0 else None)
                       for name,value in candidate_scalars.items()}
            comparisons.append(dict(state_id=row['state_id'],chi=row['chi'],global_sweep=row['global_sweep'],
                energy_per_site=row['energy_per_site'],reference_scalars=fixed_scalars,scalar_contrasts=contrasts,
                squared_distance=regional_reference_chain_audit(matrices,chains,fixed)))
            if previous_rho is not None:
                quality.append(dict(state_a=previous['state_id'],state_b=row['state_id'],
                    energy_change_per_site=row['energy_per_site']-previous['energy_per_site'],
                    distance=multiplicity_operator_distance(previous_rho,fixed),
                    raw_squared_frobenius_drift=float(np.linalg.norm(fixed-previous_rho)**2),
                    scalar_changes={name:value-previous['scalars'][cut][name] for name,value in fixed_scalars.items()}))
            previous_rho, previous=fixed,row
    sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    registry=dict(config=spec,config_sha256=checked(config),inputs=inputs,sources=sources,
        physical_problem_signature=signature,whole_chains=candidate['independent_chains'],
        candidate_params_sha256=candidate['params_sha256'],candidate_checkpoint_sha256=candidate['checkpoint_sha256'],
        statistical_protocol='Distinct whole-chain pairs for quadratic discrepancies; delete-whole-chain jackknife SE. Deterministic MPS drift is empirical quality sensitivity, not a reference SE. No phase inference or multiple-comparison significance claim.')
    output.mkdir(parents=True,exist_ok=True)
    for relative,digest in sources.items():
        target=output/'source_snapshots'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
        if file_sha256(target)!=digest:
            raise RuntimeError('analysis source changed during registration')
    _json(output/'registration.json',registry)
    result=dict(registration=registry,records=comparisons,reference_quality_sequence=quality,
        candidate_scalars=candidate_scalars,candidate_scalar_whole_chain_SE=scalar_errors,
        structure_factor_paired_diagnostics=structure,seconds=perf_counter()-start,
        new_training=0,new_ED=0,new_samples=0,new_MPS_contractions=0,new_H_calls=0,new_amplitude_queries=0,
        input_repaired=False,physical_state_replaced=False,strict_entanglement_spectrum_reported=False,
        phase_claim=False,mixing_certified=False,candidate_own_quality_sequence_available=False,
        disposition='Only same-H/cut finite-quality calibration. Current ordinary ViT is not exactly SU2; invariant operator data are legal, not strict entanglement levels. A single neural endpoint and a deterministic MPS refinement sequence cannot establish competing phases; quadratic noise and reference quality remain explicit.')
    _json(output/'results.json',result)
    _json(output/'completion.json',dict(results_sha256=file_sha256(output/'results.json'),
                                      registration_sha256=file_sha256(output/'registration.json')))
    return result


def analyze_symmetry_defect(config: Path, output: Path | None = None) -> dict:
    """Bounded zero-query audit of rotational defect versus own-quality drift.

    Complete source protocols are sealed before reuse. Sampled ordinary-ViT
    states retain their exact parent SHA, physical sites, per-stream seeds and
    whole-chain IDs. No cross-size/cut comparison, physical-state twirl,
    positivity repair, spectrum fabrication or new sampling is performed.
    """
    spec = json.loads(config.read_text())
    output = output or ROOT/spec['output']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use a fresh output directory; completed audits are immutable')
    start, inputs, loaded = perf_counter(), {}, []

    def input_file(path: Path) -> str:
        digest = file_sha256(path)
        inputs[str(path.relative_to(ROOT))] = digest
        return digest

    def canonical_record(path: Path) -> dict:
        input_file(path)
        record = json.loads(path.read_text())
        if record.get('result_sha256') != canonical_sha256({k: v for k, v in record.items() if k != 'result_sha256'}):
            raise ValueError(f'canonical record seal mismatch: {path}')
        return record

    for source in spec['sealed_protocols']:
        directory = ROOT/source['path']
        input_file(directory/'completion.json')
        seal = json.loads((directory/'completion.json').read_text())
        for name, key in [('results.json', 'results_sha256'), ('registration.json', 'registration_sha256'),
                          ('readouts.npz', 'readouts_sha256')]:
            if input_file(directory/name) != seal[key]:
                raise ValueError('sealed source protocol changed')
        record = json.loads((directory/'results.json').read_text())
        for relative, digest in record['registration']['inputs'].items():
            if input_file(ROOT/relative) != digest:
                raise ValueError('source protocol physical input changed')
        for relative, digest in record['registration']['sources'].items():
            if input_file(directory/'source_snapshots'/relative) != digest:
                raise ValueError('source protocol snapshot changed')
        with np.load(directory/'readouts.npz') as arrays:
            for row in record['records']:
                problem = record['registration']['config']['problems'][row['problem']]
                common = {key: row[key] for key in ('problem', 'family', 'state_id', 'order', 'energy_per_site',
                    'variance_per_site', 'variance_measured', 'physical_problem_signature', 'quality')}
                common.update(method=source['method'], source_protocol=source['path'],
                    source_kind='deterministic_saved_physical_rdm', globally_strict_SU2_claim=False)
                for cut, sites in problem['regions'].items():
                    key = '__'.join((row['problem'], row['family'], row['state_id'], cut))+'_raw_rho'
                    loaded.append(dict(**common, cut=cut, sites=sites,
                        compact_triangular_2x3=cut in problem.get('compact_triangular_2x3', []),
                        rho=np.array(arrays[key]), cached_original_scalars=row['scalars'][cut]))

    for source in spec.get('ordinary_vit_audits', []):
        directory = ROOT/source['path']
        registry = canonical_record(directory/'registration.json')
        cfg = registry['config']
        if (cfg['model']['geometry'], cfg['model']['rows'], cfg['model']['columns'], cfg['j2_over_j1']) != ('triangular', 6, 6, .125):
            raise ValueError('ordinary cached sequence is triangular6x6/PBC,J2=.125 only')
        problem = dict(geometry='triangular', rows=6, columns=6, j2=.125,
            boundary='PBC_both_physical', regions={key: value['sites'] for key, value in cfg['regions'].items()},
            left_sites={key: value['left'] for key, value in cfg['regions'].items()},
            compact_triangular_2x3=['compact6'], cyclic_regions=['winding6'])
        signature, _ = _problem_signature(problem)
        for step in source['steps']:
            audit_dir = directory/'audits'/f'step{step:05d}'
            audit = canonical_record(audit_dir/'results.json')
            if audit['step'] != step or audit['registration_sha256'] != registry['result_sha256']:
                raise ValueError('cached independent audit parent registration mismatch')
            # The audit is independent of training, not independent of each
            # other observable. Distinct streams have explicit disjoint IDs.
            stream_reports = []
            for path in sorted(audit_dir.glob('stream*.json')):
                stream = canonical_record(path)
                if stream['params_sha256'] != audit['params_sha256'] or input_file(path.with_suffix('.npz')) != stream['raw_sha256']:
                    raise ValueError('sample stream seal or neural parent mismatch')
                stream_reports.append(stream)
            if len(stream_reports) != 2 or len({s['seed'] for s in stream_reports}) != 2:
                raise ValueError('exactly two separately seeded cached audit streams required')
            for cut, sites in problem['regions'].items():
                metadata = canonical_record(audit_dir/(cut+'.json'))
                if metadata['sites'] != sites or input_file(audit_dir/(cut+'.npz')) != metadata['raw_sha256']:
                    raise ValueError('raw neural physical RDM/site seal mismatch')
                with np.load(audit_dir/(cut+'.npz')) as arrays:
                    rho, raw, groups = (np.array(arrays[key]) for key in ('rho', 'per_record_matrices', 'chain_ids'))
                    if arrays['sites'].tolist() != sites or len(np.unique(groups)) != 64:
                        raise ValueError('cached physical site order / whole-chain count changed')
                    replay = raw.sum(axis=0)/np.trace(raw.sum(axis=0)).real
                    if np.max(abs(replay-rho)) > 2e-14:
                        raise ValueError('raw neural RDM does not replay from cached records')
                loaded.append(dict(method='ordinary_spatial_vit_4032', problem='triangular_n36', family=source['family'],
                    state_id=f'step{step:05d}', order=step, cut=cut, sites=sites,
                    physical_problem_signature=signature, rho=rho, per_record_matrices=raw, chain_ids=groups,
                    compact_triangular_2x3=cut == 'compact6', source_kind='independent_cached_born_audit',
                    source_audit=str(audit_dir.relative_to(ROOT)), params_sha256=audit['params_sha256'],
                    checkpoint_sha256=audit['checkpoint_sha256'], globally_strict_SU2_claim=audit['strict_su2'],
                    audit_projection=audit['audit_projection'], sampled_A1_relative_error=audit['sampled_A1_relative_error'],
                    energy_per_site=audit['energy']['energy_per_site'], variance_per_site=audit['energy']['variance_per_site'],
                    variance_measured=True, global_spin_squared=audit['total_spin_squared'],
                    energy_quality=audit['energy'], independent_stream_seeds=[s['seed'] for s in stream_reports],
                    quality=dict(parameter_count=4032, independent_chains=64, mixing_certified=False)))
    sources = transitive_local_source_manifest((Path(__file__),), root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    sources['su2_nqs/frontier/provenance.py'] = file_sha256(ROOT/'su2_nqs/frontier/provenance.py')
    registry = dict(config=spec, config_sha256=file_sha256(config), inputs=inputs, sources=sources,
        comparator_protocol='The SU2 conditional expectation is only a mathematical distance comparator; actual rho and all physical readouts remain unchanged. No manufactured spectrum.',
        chain_protocol='Independent cached audit RNG streams; equal-size WHOLE chains. The quadratic cross-chain numerator is unbiased under IID chain means; self-normalized ratio bias and mixing bias remain. Negative estimates retained.')
    output.mkdir(parents=True, exist_ok=True)
    for relative, digest in sources.items():
        target = output/'source_snapshots'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
        if file_sha256(target) != digest:
            raise RuntimeError('analysis source changed during registration')
    _json(output/'registration.json', registry)
    records, raw_matrices, raw_by_key = [], {}, {}
    for item in loaded:
        rho = item['rho']
        common = {k: v for k, v in item.items() if k not in ('rho', 'per_record_matrices', 'chain_ids')}
        common['rotational_audit'] = rotational_symmetry_defect(rho,
            compact_triangular_2x3=item['compact_triangular_2x3'])
        if 'per_record_matrices' in item:
            common['chain_noise_audit'] = rotational_defect_chain_audit(item['per_record_matrices'], item['chain_ids'],
                compact_triangular_2x3=item['compact_triangular_2x3'])
        key = '__'.join(str(item[k]) for k in ('method', 'problem', 'family', 'state_id', 'cut'))
        raw_matrices[key+'_actual_rho'] = rho
        raw_by_key[key] = rho
        records.append(common)
    records.sort(key=lambda r: tuple(r[k] for k in ('method', 'problem', 'family', 'cut', 'order')))
    sequences = []
    groups = sorted({tuple(r[k] for k in ('method', 'problem', 'family', 'cut')) for r in records})
    for group in groups:
        series = [r for r in records if tuple(r[k] for k in ('method', 'problem', 'family', 'cut')) == group]
        for before, after in zip(series[:-1], series[1:]):
            if before['sites'] != after['sites'] or before['physical_problem_signature'] != after['physical_problem_signature']:
                raise ValueError('quality drift requires exactly the same labelled H and physical cut')
            key = lambda r: '__'.join(str(r[k]) for k in ('method', 'problem', 'family', 'state_id', 'cut'))
            a, b = before['rotational_audit'], after['rotational_audit']
            distances = multiplicity_operator_distance(raw_by_key[key(before)], raw_by_key[key(after)])
            scalar_change = {name: row['actual']-a['invariant_operator_checks'][name]['actual']
                             for name, row in b['invariant_operator_checks'].items()}
            sequences.append(dict(method=group[0], problem=group[1], family=group[2], cut=group[3],
                sites=after['sites'], state_a=before['state_id'], state_b=after['state_id'],
                energy_change_per_site=after['energy_per_site']-before['energy_per_site'],
                distance=distances, rotational_defect_sum=a['rotational_defect_trace_distance']+b['rotational_defect_trace_distance'],
                raw_minus_invariant_distance_triangle_bound=max(0., distances['physical_trace_distance']-distances['invariant_operator_trace_distance']),
                invariant_scalar_changes=scalar_change,
                warning='Own-family quality drift, not a phase comparison. Invariant scalar values cannot change just because diagnostic rotational averaging removes magnetic-index anisotropy.'))
    np.savez_compressed(output/'actual_rdms.npz', **raw_matrices)
    result = dict(registration=registry, records=records, quality_sequences=sequences,
        actual_rdms_sha256=file_sha256(output/'actual_rdms.npz'), seconds=perf_counter()-start,
        new_training=0, new_ED=0, new_samples=0, new_MPS_contractions=0, new_H_calls=0,
        new_amplitude_queries=0, input_repaired=False, strict_entanglement_spectrum_reported=False,
        phase_claim=False, globally_strict_SU2_claim=False,
        disposition='Invariant physical readouts remain legal on ordinary ViTs. Strict spectra require actual symmetry, not a noisy sample threshold. Small DMRG rotational defect does not remove finite-chi quality drift; no competitive or candidate-phase inference.')
    _json(output/'results.json', result)
    _json(output/'completion.json', dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'), actual_rdms_sha256=result['actual_rdms_sha256']))
    return result


def analyze_invariant_maximum_entropy(config: Path, output: Path | None = None) -> dict:
    """Bounded zero-query maximum entropy and cross-stream predictive audit.

    Reuses the already sealed same-region comparison, with fixed feature span
    and whole-chain folds. No neural parameters, physical RDMs or old results
    are modified. All fits, including delete-chain refits, must close moments.
    """
    from su2_nqs.benchmarks.su2_maximum_entropy import (
        audit_invariant_chains, fit_invariant_maximum_entropy, invariant_constraint_space,
    )
    from .spin_spatial_vit_quality import read_sealed
    spec=json.loads(config.read_text())
    output=output or ROOT/spec['output']
    if output.exists():
        raise FileExistsError('maximum-entropy audit needs a fresh immutable output')
    start=perf_counter();inputs={};records=[];arrays={}
    def checked(path,expected=None):
        path=Path(path).resolve();digest=file_sha256(path)
        if expected is not None and digest!=expected:
            raise ValueError('sealed input changed: '+str(path))
        inputs[str(path.relative_to(ROOT))]=digest
        return digest
    parent=ROOT/spec['same_region_comparison']
    completion=json.loads((parent/'completion.json').read_text())
    checked(parent/'completion.json')
    checked(parent/'results.json',completion['results_sha256'])
    previous=json.loads((parent/'results.json').read_text())
    for name,digest in previous['registration']['inputs'].items():
        checked(ROOT/name,digest)
    prior=previous['registration']['config']
    if list(prior['problem']['regions'])!=['compact6','winding6']:
        raise ValueError('this bounded audit expects the registered two six-site cuts')
    space=invariant_constraint_space(6)
    source=ROOT/spec['trusted_small_archive']
    trusted=read_sealed(source/'results.json')
    checked(source/'results.json')
    checked(source/'readout.npz',trusted['readout_sha256'])
    checked(config)
    sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    registry=dict(config=spec,inputs=inputs,sources=sources,
        physical_problem=prior['problem'],constraint_space=space.diagnostics,
        statistical_plan='Sorted whole chains: first64=stream0, last64=stream1; equal-stream estimator; swap train/test; full stratified delete-chain refits. Both signed scores and nonnegative-in-population plug-in information retained. No phase discovery rule.')
    output.mkdir(parents=True)
    for name in sources:
        target=output/'source_snapshots'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    _json(output/'registration.json',registry)
    def fit(rho,n):
        local=invariant_constraint_space(n);omega=local.data(rho)
        report,model,log_model=fit_invariant_maximum_entropy(omega,local,
            max_iterations=spec['max_fit_iterations'])
        if not report['certified_moment_fit']:
            raise RuntimeError('failed moment fit: '+str(report))
        return report,dict(omega=omega,maxent=model,log_maxent=log_model)
    def add_arrays(key,data):
        arrays.update({key+'__'+k:v for k,v in data.items()})
    def budget():
        if perf_counter()-start>spec['wall_seconds_cap']:
            raise RuntimeError('fixed local analysis budget exhausted; partial output is not complete')
    # Exact saved N16 states at different couplings: estimator/interpretation
    # calibration, not new diagonalization or neural accuracy-tail optimization.
    with np.load(source/'readout.npz') as saved:
        for coupling in spec['small_coupling_labels']:
            for n in (4,6):
                key=f'triangular_j{coupling}_strip_{n}_rho'
                report,data=fit(saved[key],n)
                records.append(dict(method='saved_N16',state_id='j'+coupling,
                    cut='strip_'+str(n),fit=report))
                add_arrays('N16__'+key,data)
    refdir=ROOT/prior['reference_protocol']
    ref=json.loads((refdir/'results.json').read_text())
    refrows=sorted((r for r in ref['records'] if r['problem']==prior['reference_problem']
        and r['family']==prior['reference_family']),key=lambda r:r['order'])
    with np.load(refdir/'readouts.npz') as saved:
        for row in refrows:
            for cut in prior['problem']['regions']:
                budget()
                key='__'.join((prior['reference_problem'],prior['reference_family'],row['state_id'],cut))
                report,data=fit(saved[key+'_raw_rho'],6)
                records.append(dict(method='DMRG',state_id=row['state_id'],cut=cut,
                    chi=row['chi'],global_sweep=row['global_sweep'],energy_per_site=row['energy_per_site'],fit=report))
                add_arrays('DMRG__'+row['state_id']+'__'+cut,data)
    for relative in prior['candidates']:
        directory=ROOT/relative
        r=read_sealed(directory/'results.json')
        with np.load(directory/'physics.npz') as saved:
            ids=saved['chain_ids'];unique,counts=np.unique(ids,return_counts=True)
            if len(unique)!=128 or not np.all(counts==4) or not np.array_equal(unique,np.arange(128)):
                raise ValueError('predeclared128 whole chains/four records and two streams required')
            for cut in prior['problem']['regions']:
                budget()
                chain_rhos=np.stack([saved[cut+'__per_record_matrices'][ids==g].mean(0) for g in unique])
                omegas=np.stack([space.data(rho) for rho in chain_rhos])
                report,data=audit_invariant_chains(omegas,(unique>=64).astype(int),space,
                    max_iterations=spec['max_fit_iterations'])
                records.append(dict(method='ViT',state_id=str(r['step']),cut=cut,
                    audit=report))
                add_arrays('ViT__'+str(r['step'])+'__'+cut,data)
                print(json.dumps(dict(completed='ViT'+str(r['step'])+'/'+cut,
                    point=report['point'],refit_SE=report['stratified_whole_chain_refit_SE'])),flush=True)
    comparisons={}
    for cut in prior['problem']['regions']:
        vi=sorted((r for r in records if r['method']=='ViT' and r['cut']==cut),key=lambda r:int(r['state_id']))
        dm=[r for r in records if r['method']=='DMRG' and r['cut']==cut]
        endpoint=next(r for r in dm if r['state_id']==prior['reference_endpoint'])
        refvalue=endpoint['fit']['information_TR_even']
        drift=max(abs(refvalue-r['fit']['information_TR_even']) for r in dm
            if r['state_id'] in prior['reference_refinements'])
        old,new=vi[0]['audit'],vi[1]['audit']
        key='information_TR_even'
        own=new['point'][key]-old['point'][key]
        own_se=float(np.hypot(old['stratified_whole_chain_refit_SE'][key],new['stratified_whole_chain_refit_SE'][key]))
        contrast=new['point'][key]-refvalue;se=new['stratified_whole_chain_refit_SE'][key]
        comparisons[cut]=dict(reference=refvalue,reference_refinement_drift=drift,
            neural_quality_change=own,neural_quality_independent_SE=own_se,
            plugin_contrast=contrast,plugin_contrast_SE=se,
            exploratory_quality_screen=bool(abs(contrast)>abs(own)+drift+2*(se+own_se)),
            latest_signed_predictive_score=new['point']['heldout_TR_even'],
            latest_predictive_full_refit_SE=new['stratified_whole_chain_refit_SE']['heldout_TR_even'],
            no_phase_claim=True,screen_not_preregistered_discovery=True)
    np.savez_compressed(output/'readouts.npz',**arrays)
    result=dict(registration=registry,records=records,comparisons=comparisons,
        seconds=perf_counter()-start,new_training=0,new_ED=0,new_samples=0,
        new_H_calls=0,new_amplitude_queries=0,phase_claim=False,
        strict_entanglement_spectrum=False,is_entanglement_witness=False,
        readouts_sha256=file_sha256(output/'readouts.npz'))
    _json(output/'results.json',result)
    _json(output/'completion.json',dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'),readouts_sha256=result['readouts_sha256']))
    return result


def _singlet_channel_scalars(rho: np.ndarray, cut: dict) -> tuple[dict, dict]:
    """Occupation, internal purity and coherence in declared regional singlets.

    A row cut and the elementary-triangle cut are different observables.
    Keep both names; never transfer E2/A2 triangle labels onto a winding row.
    """
    from su2_nqs.benchmarks.spin_partition_readout import singlet_resonance_readout
    from su2_nqs.benchmarks.spin_operator_information import joint_spin_operator_split
    partitions = {'rows': cut['left']}
    if cut.get('compact_triangular_2x3', False):
        partitions['triangles'] = [0, 3, 4]
    scalars, arrays = {}, {}
    for name, left in partitions.items():
        report, block = singlet_resonance_readout(rho, left)
        arrays[name+'_singlet_block'] = block
        for field in ('p11_singlet', 'p33_singlet', 'doublet_conditional_operator_purity',
                      'cross_q1_q3_coherence_norm', 'cross_q1_q3_normalized_coherence'):
            if report[field] is not None:
                scalars[name+'_'+field] = report[field]
        # Signed linear high-order pieces and actual coherences accompany the
        # positively biased plug-in norm. Their signs depend on a fixed CG basis.
        for q in (1, 3):
            split = joint_spin_operator_split(6, left, (q, q, 0)).measure(rho)
            scalars[name+f'_p{q}{q}_high'] = split['high']
        for a in range(4):
            scalars[name+f'_coherence_{a}_real'] = float(block[a, 4].real)
            scalars[name+f'_coherence_{a}_imag'] = float(block[a, 4].imag)
    return scalars, arrays


def analyze_unified_quality_readout(config: Path, output: Path | None = None) -> dict:
    """Zero-query regional and singlet-resonance analysis of quality sequences.

    Every cache is hash checked against its immutable registration. Missing
    regions remain missing; global structure factors cannot be inferred from
    local RDMs. Independent endpoints use unpaired errors, while operator
    contrasts on one endpoint retain the joint delete-chain covariance.
    """
    from .spin_spatial_vit_quality import read_sealed
    from su2_nqs.vmc.regional_density import regional_density_summary
    from su2_nqs.benchmarks.su2_maximum_entropy import (
        invariant_constraint_space, fit_invariant_maximum_entropy, audit_invariant_chains,
    )
    spec = json.loads(config.read_text())
    output = output or ROOT/spec['output']
    if output.exists():
        raise FileExistsError('unified readout requires a fresh immutable output')
    start, inputs, records, arrays, missing = perf_counter(), {}, [], {}, []
    signature, _ = _problem_signature(spec['problem'])
    space = invariant_constraint_space(6)

    def checked(path: Path, expected: str | None = None) -> None:
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError('changed sealed input: '+str(path))
        inputs[str(path.resolve().relative_to(ROOT))] = digest

    def budget() -> None:
        if perf_counter()-start > spec['wall_seconds_cap']:
            raise RuntimeError('fixed analysis budget exhausted; no completion seal')

    def cut_definition(name: str) -> dict:
        p = spec['problem']
        return dict(sites=p['regions'][name], left=p['left_sites'][name],
                    cyclic=name in p.get('cyclic_regions', []),
                    compact_triangular_2x3=name in p.get('compact_triangular_2x3', []))

    def fit(rho: np.ndarray) -> tuple[dict, dict]:
        omega = space.data(rho)
        report, model, log_model = fit_invariant_maximum_entropy(omega, space,
            max_iterations=spec['max_fit_iterations'])
        if not report['certified_moment_fit']:
            raise RuntimeError('maximum entropy fit failed its unchanged moment tolerances')
        return report, dict(omega=omega, maxent=model, log_maxent=log_model)

    refdir = ROOT/spec['reference_protocol']
    done = json.loads((refdir/'completion.json').read_text())
    checked(refdir/'completion.json')
    checked(refdir/'results.json', done['results_sha256'])
    checked(refdir/'readouts.npz', done['readouts_sha256'])
    reference = json.loads((refdir/'results.json').read_text())
    rp = reference['registration']['config']['problems'][spec['reference_problem']]
    if _problem_signature(rp)[0] != signature:
        raise ValueError('reference Hamiltonian/physical boundary differs')
    with np.load(refdir/'readouts.npz') as data:
        for row in sorted(reference['records'], key=lambda r:r['order']):
            if row['problem'] != spec['reference_problem'] or row['family'] != spec['reference_family']:
                continue
            for name in spec['problem']['regions']:
                budget()
                if name not in rp['regions']:
                    missing.append(dict(method='DMRG', state_id=row['state_id'], cut=name,
                                        reason='raw regional RDM absent from local sealed archive'))
                    continue
                if (rp['regions'][name] != spec['problem']['regions'][name]
                        or rp['left_sites'][name] != spec['problem']['left_sites'][name]):
                    raise ValueError('reference physical region order/partition differs')
                key = '__'.join((row['problem'], row['family'], row['state_id'], name))
                rho = np.array(data[key+'_raw_rho'])
                report, _, scalars = _regional_scalar_data(rho, cut_definition(name))
                extra, blocks = _singlet_channel_scalars(rho, cut_definition(name))
                scalars.update(extra)
                me, features = fit(rho)
                records.append(dict(method='DMRG', state_id=row['state_id'], cut=name,
                    chi=row['chi'], sweep=row['global_sweep'], energy_per_site=row['energy_per_site'],
                    energy_SE=None, variance_per_site=row['variance_per_site'], scalars=scalars,
                    scalar_SE=None, strict_su2_eligible=report['multiplicity']['physical']['strict_su2_eligible'],
                    symmetry_defect=report['multiplicity']['symmetry_defect_trace_distance'], maxent=me))
                arrays.update({key+'__'+k:v for k,v in dict(rho=rho, **blocks, **features).items()})

    for candidate in spec['candidates']:
        step, seen, identity = candidate['step'], set(), None
        audit_path = ROOT/candidate['energy_audit']
        energy_record = read_sealed(audit_path)
        checked(audit_path)
        if energy_record['step'] != step:
            raise ValueError('energy audit belongs to another endpoint')
        energy = energy_record['energy']
        for relative in candidate['region_caches']:
            directory = ROOT/relative
            reg = read_sealed(directory/'registration.json')
            saved = read_sealed(directory/'results.json')
            for name in ('registration.json', 'results.json'):
                checked(directory/name)
            checked(directory/'physics.npz', saved['raw_sha256'])
            for path, digest in reg['inputs_sha256'].items():
                checked(ROOT/path, digest)
            for path, digest in reg['source_manifest'].items():
                checked(directory/'source_snapshots'/path, digest)
            if (saved['physical_problem_signature'] != signature or saved['step'] != step
                    or saved['registration_sha256'] != reg['result_sha256']
                    or saved['params_sha256'] != energy_record['params_sha256']
                    or saved['checkpoint_sha256'] != energy_record['checkpoint_sha256']):
                raise ValueError('regional cache and energy parent differ')
            with np.load(directory/'physics.npz') as data:
                ids = np.array(data['chain_ids'])
                if not np.array_equal(np.unique(ids), np.arange(128)):
                    raise ValueError('two registered 64-chain streams required')
                current = (saved['params_sha256'], ids.tobytes(), data['states'].tobytes())
                if identity is not None and current != identity:
                    raise ValueError('regional caches do not share the same parent/records')
                identity = current
                for name, cached in saved['regions'].items():
                    budget()
                    if name not in spec['problem']['regions']:
                        continue
                    if name in seen:
                        raise ValueError('duplicate regional cache for one endpoint')
                    seen.add(name)
                    cut = cut_definition(name)
                    if cached['sites'] != cut['sites'] or cached['left'] != cut['left']:
                        raise ValueError('candidate physical region order/partition differs')
                    summary = regional_density_summary(np.array(data[name+'__per_record_matrices']), ids)
                    rho, jack = summary['rho'], summary['jackknife_rhos']
                    np.testing.assert_allclose(rho, data[name+'__rho'], atol=1e-12, rtol=0)
                    extra, blocks = _singlet_channel_scalars(rho, cut)
                    deleted = [_singlet_channel_scalars(r, cut)[0] for r in jack]
                    names = [n for n in extra if all(n in d for d in deleted)]
                    values = np.array([[d[n] for n in names] for d in deleted])
                    combined_names = cached['scalar_names']+names
                    combined_jack = np.column_stack((data[name+'__scalar_jackknife_values'], values))
                    center = combined_jack-combined_jack.mean(0)
                    covariance = (len(jack)-1)/len(jack)*(center.T@center)
                    errors = dict(zip(combined_names, map(float, np.sqrt(np.diag(covariance)))))
                    scalars = dict(cached['scalars'], **{n:extra[n] for n in names})
                    chain_rhos = np.stack([data[name+'__per_record_matrices'][ids==g].mean(0)
                                           for g in range(128)])
                    omega = np.stack([space.data(r) for r in chain_rhos])
                    me, features = audit_invariant_chains(omega, (np.arange(128)>=64).astype(int),
                        space, max_iterations=spec['max_fit_iterations'])
                    records.append(dict(method='ViT', state_id=str(step), step=step, cut=name,
                        energy_per_site=energy['energy_per_site'], energy_SE=energy['chain_mean_standard_error'],
                        variance_per_site=energy['variance_per_site'], variance_SE=energy['variance_chain_standard_error'],
                        scalars=scalars, scalar_SE=errors, scalar_names=combined_names,
                        strict_su2_eligible=cached['protocol']['multiplicity']['physical']['strict_su2_eligible'],
                        symmetry_defect=cached['protocol']['multiplicity']['symmetry_defect_trace_distance'],
                        maxent=me, independent_chains=128, mixing_certified=False))
                    key = 'ViT__'+str(step)+'__'+name
                    arrays.update({key+'__'+k:v for k,v in dict(rho=rho, **blocks,
                        **features, combined_scalar_jackknife=combined_jack, combined_covariance=covariance).items()})
                    print(json.dumps(dict(completed=key, heldout=me['point']['heldout_TR_even'],
                        SE=me['stratified_whole_chain_refit_SE']['heldout_TR_even'])), flush=True)
        for name in set(spec['problem']['regions'])-seen:
            missing.append(dict(method='ViT', state_id=str(step), cut=name, reason='regional cache absent'))

    comparisons = {}
    for name in spec['problem']['regions']:
        vit = sorted((r for r in records if r['method']=='ViT' and r['cut']==name), key=lambda r:r['step'])
        dmrg = sorted((r for r in records if r['method']=='DMRG' and r['cut']==name), key=lambda r:r['sweep'])
        if len(vit)<2 or len(dmrg)<2:
            comparisons[name] = dict(quality_comparison_complete=False, missing_reference=not dmrg)
            continue
        old, new, ref = vit[-2], vit[-1], dmrg[-1]
        refinements = [r for r in dmrg if r['state_id'] in spec['reference_refinements']]
        if len(refinements) != len(spec['reference_refinements']):
            raise ValueError('declared DMRG last-sweep/last-chi refinements missing')
        contrasts = {}
        for observable in spec['primary_observables'][name]:
            before, after, fixed = old['scalars'][observable], new['scalars'][observable], ref['scalars'][observable]
            se, ds = new['scalar_SE'][observable], float(np.hypot(new['scalar_SE'][observable], old['scalar_SE'][observable]))
            own = after-before
            drift = max(abs(fixed-r['scalars'][observable]) for r in refinements)
            delta = after-fixed
            contrasts[observable] = dict(vit=after, vit_SE=se, dmrg=fixed,
                between_family=delta, between_family_SE=se, neural_quality_change=own,
                neural_quality_SE=ds, dmrg_quality_envelope=drift,
                descriptive_two_SE_screen=bool(abs(delta)>abs(own)+drift+2*(se+ds)))
        comparisons[name] = dict(quality_comparison_complete=True, primary=contrasts,
            primary_passing_screens=[n for n,v in contrasts.items() if v['descriptive_two_SE_screen']],
            matrix_distance=multiplicity_operator_distance(arrays['ViT__'+new['state_id']+'__'+name+'__rho'],
                arrays['__'.join((spec['reference_problem'],spec['reference_family'],ref['state_id'],name))+'__rho']))
    checked(config)
    sources = transitive_local_source_manifest((Path(__file__),), root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    registration = dict(config=spec, inputs=inputs, sources=sources, physical_problem_signature=signature,
        protocol='Fixed cached states; three regional geometries; row and triangle cuts separate; independent endpoint errors, complete within-endpoint scalar covariance. Cross-fitted maximum entropy with full chain refits. Pointwise exploratory screens are not multiplicity-corrected tests or bias certificates.')
    output.mkdir(parents=True)
    for name in sources:
        target = output/'source_snapshots'/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    _json(output/'registration.json', registration)
    np.savez_compressed(output/'readouts.npz', **arrays)
    result = dict(registration=registration, records=records, comparisons=comparisons,
        missing=missing, seconds=perf_counter()-start, new_training=0, new_samples=0,
        new_H_calls=0, new_amplitude_queries=0, new_MPS_contractions=0,
        strict_entanglement_spectrum=False, phase_claim=False,
        coherence_norm_bias_removed=False, readouts_sha256=file_sha256(output/'readouts.npz'))
    _json(output/'results.json', result)
    _json(output/'completion.json', dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'), readouts_sha256=result['readouts_sha256']))
    return result


def product_dimer_region_density(sites: list[int], pairs: list[list[int]]) -> np.ndarray:
    """Exact RDM of a full product of singlets, including cut dimers.

    Physical down/up bits, MSB-first. A dimer crossing the regional boundary
    leaves I/2, not a fabricated pure regional singlet. No many-body vector.
    """
    n = len(sites)
    if (not 1 <= n <= 8 or len(set(sites)) != n or not pairs
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in sites)):
        raise ValueError('distinct ordered sites and a complete dimer covering required')
    endpoints = [i for pair in pairs for i in pair]
    if (any(len(pair)!=2 or pair[0]==pair[1] for pair in pairs)
            or len(set(endpoints))!=len(endpoints) or not set(sites).issubset(endpoints)):
        raise ValueError('dimer covering has repeated or missing sites')
    bits = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    positions = {site:i for i,site in enumerate(sites)}
    v, paired = np.ones(2**n), set()
    for i,j in pairs:
        if i in positions and j in positions:
            a,b = positions[i], positions[j]
            v *= (bits[:,a]-bits[:,b])/np.sqrt(2)
            paired.update((a,b))
    rho = np.outer(v,v)
    for i in set(range(n))-paired:
        rho *= (bits[:,i,None]==bits[None,:,i])/2
    if abs(np.trace(rho)-1)>1e-12:
        raise RuntimeError('product-dimer partial trace is not normalized')
    return rho


def analyze_readout_controls(config: Path, output: Path | None = None) -> dict:
    """Exact existing-state/synthetic calibration, no new quantum AFM state."""
    from .spin_spatial_vit_quality import read_sealed
    spec=json.loads(config.read_text()); output=output or ROOT/spec['output']
    if output.exists():
        raise FileExistsError('fresh calibration output required')
    records,arrays,inputs=[],{},{}
    def measure(label,rho,cut,kind,**metadata):
        report,_,scalars=_regional_scalar_data(rho,cut)
        extra,blocks=_singlet_channel_scalars(rho,cut);scalars.update(extra)
        records.append(dict(label=label,kind=kind,sites=cut['sites'],left=cut['left'],
            scalars=scalars,strict_su2_eligible=report['multiplicity']['physical']['strict_su2_eligible'],
            symmetry_defect=report['multiplicity']['symmetry_defect_trace_distance'],**metadata))
        arrays[label+'__rho']=rho
        arrays.update({label+'__'+k:v for k,v in blocks.items()})
    for name,sites in spec['problem']['regions'].items():
        cut=dict(sites=sites,left=spec['problem']['left_sites'][name],
            cyclic=name in spec['problem'].get('cyclic_regions',[]),
            compact_triangular_2x3=name in spec['problem'].get('compact_triangular_2x3',[]))
        rows,columns=spec['problem']['rows'],spec['problem']['columns']
        if columns%2:
            raise ValueError('horizontal dimer calibration requires even columns')
        for shift in (0,1):
            pairs=[[columns*r+(2*c+shift)%columns,columns*r+(2*c+1+shift)%columns]
                   for r in range(rows) for c in range(columns//2)]
            measure(name+f'__horizontal_dimers_shift{shift}',product_dimer_region_density(sites,pairs),
                    cut,'synthetic_dimer',quantum_AF_M_reference=False,global_pairs=pairs)
        waves=([('classical_K_plus',[1/3,1/3]),('classical_K_minus',[-1/3,-1/3]),
                ('classical_M1',[.5,0]),('classical_M2',[0,.5]),('classical_M3',[.5,.5])]
               if spec['problem']['geometry']=='triangular' else
               [('classical_Neel',[.5,.5]),('classical_stripe_x',[.5,0]),('classical_stripe_y',[0,.5])])
        for label,k in waves:
            coordinates=np.array([divmod(i,columns) for i in sites]); phases=2*np.pi*(coordinates@k)
            v=np.array([1.],complex)
            for phase in phases:v=np.kron(v,np.array([np.exp(1j*phase),1.])/np.sqrt(2))
            measure(name+'__'+label,np.outer(v,v.conj()),cut,'separable_magnetic_product',
                    quantum_AF_M_reference=False,wavevector_reciprocal_cycles=k)
    directory=ROOT/spec['trusted_small_archive'];saved=read_sealed(directory/'results.json')
    inputs[str((directory/'results.json').relative_to(ROOT))]=file_sha256(directory/'results.json')
    inputs[str((directory/'readout.npz').relative_to(ROOT))]=file_sha256(directory/'readout.npz')
    if inputs[str((directory/'readout.npz').relative_to(ROOT))]!=saved['readout_sha256']:
        raise ValueError('existing small-state RDM seal changed')
    cut=dict(sites=[0,1,4,5,8,9],left=[0,2,4])
    with np.load(directory/'readout.npz') as data:
        labels=spec.get('trusted_small_labels',
            ['triangular_j0','triangular_j0.075','triangular_j0.125','triangular_j0.175',
             'dimer_internal','dimer_four_crossing'])
        for label in labels:
            measure('N16__'+label,data[label+'_strip_6_rho'],cut,'saved_N16_calibration',
                    n_sites=16,comparable_to_N36=False)
    inputs[str(config.resolve().relative_to(ROOT))]=file_sha256(config)
    sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    output.mkdir(parents=True)
    for name in sources:
        p=output/'source_snapshots'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    _json(output/'registration.json',dict(config=spec,inputs=inputs,sources=sources))
    np.savez_compressed(output/'readouts.npz',**arrays)
    result=dict(records=records,new_ED=0,new_training=0,new_samples=0,new_H_calls=0,
        new_amplitude_queries=0,phase_claim=False,
        interpretation='Exact dimer boundary partial traces and unaveraged separable K/M controls. These are calibration, not optimized quantum AFM/VBS candidate states. Saved N16 cuts are not N36 size-scaling pairs.')
    _json(output/'results.json',result)
    _json(output/'completion.json',dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'),readouts_sha256=file_sha256(output/'readouts.npz')))
    return result


def render_unified_quality(result_path: Path, destination: Path,
                           global_readouts: list[Path] | None = None) -> list[str]:
    """Line/errorbar quality plots; no bar charts or physical phase assignment."""
    import matplotlib.pyplot as plt
    result=json.loads(result_path.read_text());rows=result['records']
    destination.mkdir(parents=True,exist_ok=True)
    colors={'ViT':'#2166ac','DMRG':'#b2182b'};paths=[]
    def series(ax,cut,field,label):
        for method in colors:
            group=sorted((r for r in rows if r['method']==method and r['cut']==cut),
                         key=lambda r:r.get('step',r.get('sweep',0)))
            if not group:continue
            x=[r['energy_per_site'] for r in group];y=[r['scalars'][field] for r in group]
            err=None if method=='DMRG' else [r['scalar_SE'][field] for r in group]
            ax.errorbar(x,y,yerr=err,fmt='o-',ms=4,lw=1,color=colors[method],label=method)
            for r,a,b in zip(group,x,y):
                if method=='ViT':ax.annotate(str(r['step']),(a,b),xytext=(4,4),textcoords='offset points',fontsize=7)
        ax.set_xlabel('Energy per site');ax.set_ylabel(label);ax.grid(alpha=.2)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            path=destination/(name+'.'+ext);fig.savefig(path,dpi=180,bbox_inches='tight');paths.append(str(path))
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    cuts=['compact6','local_hexagon6','winding6']
    for j,cut in enumerate(cuts):
        series(axes[0,j],cut,'rows_p33_singlet',r'$p(q_L=q_R=3,q_T=0)$')
        series(axes[1,j],cut,'rows_p33_high',r'Linear high-order part of $P_{330}$')
        axes[0,j].set_title(cut+(' (DMRG RDM missing)' if cut=='local_hexagon6' else ''))
    axes[0,0].legend();fig.suptitle('Same regional 3+3 partitions; quality trajectories, not phase contrasts',fontsize=12)
    save(fig,'su2_uniform_quality_channels')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    metrics=[('triangles_p33_singlet',r'Triangle-pair $p_{330}$'),
             ('triangles_doublet_conditional_operator_purity',r'Doublet-pair conditional purity'),
             ('triangles_cross_q1_q3_coherence_norm',r'$\|B_{11,33}\|$ (plug-in bias retained)'),
             ('triangles_p33_high',r'Triangle $P_{330}$ high-order part'),
             ('triangle_conditional_E2_axis_average_connected',r'Conditional $E_2$ correlation'),
             ('triangle_conditional_A2_axis_correlation_connected',r'Conditional $A_2$ correlation')]
    for ax,(field,label) in zip(axes.flat,metrics):series(ax,'compact6',field,label)
    axes[0,0].legend();fig.suptitle('Elementary triangles [0,3,4] | [1,2,5]: occupation and internal resonance',fontsize=12)
    save(fig,'su2_uniform_quality_resonance')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for j,cut in enumerate(cuts):
        for method in colors:
            group=sorted((r for r in rows if r['method']==method and r['cut']==cut),key=lambda r:r.get('step',r.get('sweep',0)))
            if not group:continue
            y=[r['maxent']['information_TR_even'] if method=='DMRG' else r['maxent']['point']['information_TR_even'] for r in group]
            err=None if method=='DMRG' else [r['maxent']['stratified_whole_chain_refit_SE']['information_TR_even'] for r in group]
            axes[0,j].errorbar([r['energy_per_site'] for r in group],y,yerr=err,fmt='o-',ms=4,lw=1,color=colors[method],label=method)
            if method=='ViT':
                axes[1,j].errorbar([r['step'] for r in group],[r['maxent']['point']['heldout_TR_even'] for r in group],
                    yerr=[r['maxent']['stratified_whole_chain_refit_SE']['heldout_TR_even'] for r in group],fmt='o-',color=colors[method])
        axes[0,j].set_title(cut);axes[0,j].set_xlabel('Energy per site');axes[0,j].set_ylabel('Plug-in remainder (nats)')
        axes[0,j].ticklabel_format(axis='x',style='plain',useOffset=False)
        axes[1,j].axhline(0,color='gray',lw=.8);axes[1,j].set_xlabel('Training endpoint');axes[1,j].set_ylabel('Signed cross-fitted gain (nats)')
        for ax in axes[:,j]:ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Information beyond fixed p(q), all pairs and symmetrized two-bond scalars; not topology',fontsize=12)
    save(fig,'su2_uniform_quality_information')
    if global_readouts:
        from .spin_spatial_vit_quality import read_sealed
        global_rows=sorted([read_sealed(path) for path in global_readouts],key=lambda r:r['step'])
        fig,axes=plt.subplots(1,3,figsize=(13,3.7))
        for label in ('K','minus_K'):
            axes[0].errorbar([r['step'] for r in global_rows],
                [r['structure_factors'][label]['scalar']['value'] for r in global_rows],
                yerr=[r['structure_factors'][label]['scalar']['whole_chain_SE'] for r in global_rows],
                fmt='o-',label=label)
        for label in ('M1','M2','M3'):
            axes[1].errorbar([r['step'] for r in global_rows],
                [r['structure_factors'][label]['scalar']['value'] for r in global_rows],
                yerr=[r['structure_factors'][label]['scalar']['whole_chain_SE'] for r in global_rows],
                fmt='o-',label=label)
        axes[2].errorbar([r['step'] for r in global_rows],
            [r['total_spin_squared']['value'] for r in global_rows],
            yerr=[r['total_spin_squared']['whole_chain_SE'] for r in global_rows],fmt='o-')
        for ax,label in zip(axes,[r'$S(K),S(-K)$',r'$S(M)$',r'$\langle S_{\mathrm{tot}}^2\rangle$']):
            ax.set_xlabel('Training endpoint');ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[0].legend();axes[1].legend();axes[2].axhline(0,color='gray',lw=.8)
        fig.suptitle('Full S.S estimators on independent endpoint chains; not strict SU(2)',fontsize=12)
        save(fig,'su2_uniform_quality_magnetic')
    return paths


def analyze_multipartite_witness(config: Path, output: Path | None = None) -> dict:
    """Fit on sealed reference/training RDMs; audit frozen witnesses linearly.

    No new states or Born records. Full and low-order-plus-spin-weight
    witnesses use identical predeclared parties and finite solver budgets.
    Open5 is a partial trace, not a same-size control. Input RDMs are never
    repaired. Cached audit streams are disjoint from THIS witness fit, but
    have already been used for other observables and are not a new experiment.
    """
    import cvxpy
    import scs
    from su2_nqs.benchmarks.su2_multipartite_witness import (
        solve_su2_multipartite_witness, evaluate_frozen_witness,
        frozen_witness_chain_audit,
    )
    from .spin_spatial_vit_quality import read_sealed
    spec=json.loads(config.read_text());output=output or ROOT/spec['output']
    if output.exists():
        raise FileExistsError('use a fresh immutable multipartite witness output')
    start=perf_counter();inputs={};arrays={};records=[]
    def checked(path,expected=None):
        path=Path(path).resolve();digest=file_sha256(path)
        if expected is not None and digest!=expected:
            raise ValueError('sealed input changed: '+str(path))
        inputs[str(path.relative_to(ROOT))]=digest
    checked(config)
    refdir=ROOT/spec['reference_protocol'];ref=json.loads((refdir/'results.json').read_text())
    seal=json.loads((refdir/'completion.json').read_text())
    checked(refdir/'completion.json')
    checked(refdir/'results.json',seal['results_sha256'])
    checked(refdir/'registration.json',seal['registration_sha256'])
    checked(refdir/'results.json');checked(refdir/'readouts.npz',ref['readouts_sha256'])
    for name,digest in ref['registration']['inputs'].items():
        checked(ROOT/name,digest)
    refrows=[r for r in ref['records'] if r['problem']==spec['reference_problem']
             and r['family']==spec['reference_family']]
    refrows.sort(key=lambda r:r['order'])
    with np.load(refdir/'readouts.npz') as saved:
        reference={r['state_id']:{cut:saved['__'.join((r['problem'],r['family'],r['state_id'],cut))+'_raw_rho']
            for cut in set(spec['regions'])|{'winding6'}
            if '__'.join((r['problem'],r['family'],r['state_id'],cut))+'_raw_rho' in saved.files} for r in refrows}
    candidates={}
    for relative in spec['candidates']:
        directory=ROOT/relative;row=read_sealed(directory/'results.json')
        reg=read_sealed(directory/'registration.json')
        checked(directory/'results.json');checked(directory/'registration.json')
        checked(directory/'physics.npz',row['raw_sha256'])
        for name,digest in reg['inputs_sha256'].items():
            checked(ROOT/name,digest)
        if row['physical_problem_signature']!=refrows[-1]['physical_problem_signature']:
            raise ValueError('candidate/reference physical Hamiltonian mismatch')
        with np.load(directory/'physics.npz') as saved:
            ids=saved['chain_ids'];unique,counts=np.unique(ids,return_counts=True)
            if not np.array_equal(unique,np.arange(128)) or not np.all(counts==4):
                raise ValueError('predeclared two-stream 128-chain/four-record cache required')
            data=dict(ids=ids)
            for cut in row['regions']:
                if not np.array_equal(saved[cut+'__chain_ids'],ids):
                    raise ValueError('regional/global chain order differs')
                data[cut]=saved[cut+'__per_record_matrices']
            candidates[str(row['step'])]=data
        for cut in set(spec['regions'])&set(row['regions']):
            if row['regions'][cut]['sites']!=spec['regions'][cut]['sites']:
                raise ValueError('physical region ordering mismatch')
    def region(data,cut):
        if cut=='open5':
            value=data['winding6']
            return np.trace(value.reshape(*value.shape[:-2],32,2,32,2),axis1=-3,axis2=-1)
        return data[cut]
    sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    registration=dict(config=spec,inputs=inputs,sources=sources,
        packages=dict(numpy=np.__version__,cvxpy=cvxpy.__version__,scs=scs.__version__),
        statistical_plan='Reference-trained W uses all independent candidate chains. Neural W fits6600 chains0..63 and audits64..127; never refit on test. Two-SE summaries are exploratory pointwise screens, not multiple-testing certificates. Old audit caches have been used for other observables.',
        TR_even_witness_restriction=True,physical_RDM_twirl=False)
    output.mkdir(parents=True)
    for name in sources:
        target=output/'source_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    _json(output/'registration.json',registration)
    for cut,region_spec in spec['regions'].items():
        for trainer in spec['trainers']:
            if trainer=='reference':
                rho=region(reference[spec['reference_endpoint']],cut)
                fitting=dict(kind='DMRG',state=spec['reference_endpoint'],chain_ids=[])
            elif trainer=='ViT6600_stream0':
                source=candidates['6600'];rho=region(source,cut)[source['ids']<64].mean(0)
                fitting=dict(kind='ViT',state='6600',chain_ids=list(range(64)))
            else:
                raise ValueError('unknown preregistered trainer')
            for span in spec['operator_spans']:
                if perf_counter()-start>spec['wall_seconds_cap']:
                    raise RuntimeError('bounded witness budget exhausted; partial results not complete')
                fit,certificate=solve_su2_multipartite_witness(rho,region_spec['parties'],
                    time_reversal_even=True,operator_span=span,solver_options=spec['solver_options'])
                key='__'.join((cut,trainer,span))
                arrays[key+'__witness']=certificate.witness
                arrays[key+'__Q']=np.stack(certificate.q_operators)
                arrays[key+'__training_rho']=rho
                _json(output/(key+'.json'),dict(fit=fit,fitting=fitting,cuts=certificate.cuts))
                evaluations=[]
                for old in refrows:
                    if (cut if cut!='open5' else 'winding6') not in reference[old['state_id']]:
                        continue  # Geometry RDM acquisition may still be pending.
                    matrix=region(reference[old['state_id']],cut)
                    evaluations.append(dict(method='DMRG',state_id=old['state_id'],
                        energy_per_site=old['energy_per_site'],
                        audit=evaluate_frozen_witness(certificate,matrix)))
                for step,data in candidates.items():
                    mask=(data['ids']>=64) if trainer=='ViT6600_stream0' and step=='6600' else np.ones(len(data['ids']),bool)
                    audit=frozen_witness_chain_audit(certificate,region(data,cut)[mask],data['ids'][mask])
                    audit['fit_audit_chain_overlap']=0
                    evaluations.append(dict(method='ViT',state_id=step,audit=audit))
                # Separable magnetic product and ordinary dimer controls. The
                # latter can be entangled as COARSE parties; not a QSL tag.
                n=int(np.log2(len(rho)));angles=2*np.pi*np.arange(n)/3
                product=np.array([1.],complex)
                for theta in angles:
                    product=np.kron(product,np.array([1.,np.exp(1j*theta)])/np.sqrt(2))
                singlet=np.array([0.,1.,-1.,0.])/np.sqrt(2)
                dimers=np.array([[1.]])
                for _ in range(n//2):dimers=np.kron(dimers,np.outer(singlet,singlet))
                if n%2:dimers=np.kron(dimers,np.eye(2)/2)
                controls=dict(classical_120_product=evaluate_frozen_witness(certificate,np.outer(product,product.conj())),
                    independent_dimers=evaluate_frozen_witness(certificate,dimers))
                if controls['classical_120_product']['expectation'] < -1e-9:
                    raise RuntimeError('separable control violates witness')
                record=dict(key=key,cut=cut,trainer=trainer,operator_span=span,
                    fit=fit,fitting=fitting,evaluations=evaluations,controls=controls)
                records.append(record)
                _json(output/'progress.json',dict(records=records,seconds=perf_counter()-start,complete=False))
                print(json.dumps(dict(key=key,fit_bound=fit['entanglement_lower_bound'],
                    identity_shift=certificate.diagnostics['identity_shift'],seconds=perf_counter()-start)),flush=True)
    np.savez_compressed(output/'witnesses.npz',**arrays)
    result=dict(registration=registration,records=records,seconds=perf_counter()-start,
        witnesses_sha256=file_sha256(output/'witnesses.npz'),new_training=0,new_samples=0,
        new_H_calls=0,new_amplitude_queries=0,phase_claim=False,
        caveat='Feasible lower bounds for DECLARED parties, not certified optimal GMN. Positive coarse-party values can occur in product-dimer states; zero does not exclude entanglement. Full versus low-order difference is not a proof of high-order necessity without an upper certificate for the restricted optimum.')
    _json(output/'results.json',result)
    _json(output/'completion.json',dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json'),witnesses_sha256=result['witnesses_sha256']))
    return result


def analyze_multipartite_witness_replay(config: Path, output: Path | None = None) -> dict:
    """Apply ALL sealed witnesses unchanged to a new cached frozen endpoint.

    This path does not import an SDP solver, fit a witness, query amplitudes,
    sample, or modify a state. Historical sources are checked against their
    archived snapshots, not against legitimately evolved live source files.
    Same-training-step reuse is rejected unless the caller explicitly asks
    for the ORIGINAL held-out chains; a new endpoint uses all 128 chains.
    """
    from su2_nqs.benchmarks.su2_multipartite_witness import (
        WitnessCertificate, canonical_bipartitions,
        validate_witness_certificate, frozen_witness_chain_audit,
    )
    from .spin_spatial_vit_quality import read_sealed
    start=perf_counter();spec=json.loads(config.read_text())
    output=output or ROOT/spec['output']
    if output.exists():
        raise FileExistsError('use a fresh immutable frozen-witness replay output')
    policy=spec.get('same_training_step_policy','reject')
    if policy not in ('reject','original_heldout'):
        raise ValueError('same_training_step_policy must be reject or original_heldout')
    expected_step=int(spec['expected_step']);inputs={};verified={}
    def checked(path,expected=None):
        path=Path(path).resolve()
        digest=verified.get(path)
        if digest is None:
            digest=file_sha256(path);verified[path]=digest
        if expected is not None and digest!=expected:
            raise ValueError('sealed input hash mismatch: '+str(path))
        try: key=str(path.relative_to(ROOT))
        except ValueError: key=str(path)
        inputs[key]=digest
        return digest
    def archived_sources(directory,manifest):
        for name,digest in manifest.items():
            checked(directory/'source_snapshots'/name,digest)
    def checked_readout(relative):
        directory=ROOT/relative
        checked(directory/'results.json');checked(directory/'registration.json')
        row=read_sealed(directory/'results.json');reg=read_sealed(directory/'registration.json')
        if row['registration_sha256']!=reg['result_sha256']:
            raise ValueError('cached RDM registration identity mismatch')
        checked(directory/'physics.npz',row['raw_sha256'])
        for name,digest in reg['inputs_sha256'].items():checked(ROOT/name,digest)
        archived_sources(directory,reg['source_manifest'])
        parent=ROOT/reg['config']['parent']
        archived_sources(parent,reg.get('frozen_parent_source_manifest',{}))
        for key in ('step','params_sha256','checkpoint_sha256'):
            if row[key]!=reg['config'][key]:
                raise ValueError('cached frozen state identity mismatch')
        return directory,row,reg
    checked(config)
    prereg=None
    if spec.get('pre_registration'):
        path=ROOT/spec['pre_registration'];checked(path);prereg=json.loads(path.read_text())
        if (int(prereg['step'])!=expected_step
                or prereg['witness_sources']!=spec['witness_sources']
                or not prereg['all_saved_witnesses_included']
                or prereg['refit_or_select_witness_after_audit']):
            raise ValueError('replay differs from preregistered witness/endpoint selection')
    identity=None;regions={};global_ids=None;known_problem=None;known_bonds=None
    for relative in spec['candidate_readouts']:
        directory,row,reg=checked_readout(relative)
        if row['step']!=expected_step:
            raise ValueError('cached endpoint differs from expected_step')
        current=(row['step'],row['params_sha256'],row['checkpoint_sha256'],
                 row['physical_problem_signature'],tuple(reg['audit_seeds']),
                 tuple(reg['fixed_snapshot_indices']))
        if identity is not None and current!=identity:
            raise ValueError('candidate caches do not share one frozen state/audit protocol')
        identity=current;known_problem=row['physical_problem_signature']
        if known_bonds is not None and known_bonds!=row['bonds']:
            raise ValueError('candidate Hamiltonian bonds differ')
        known_bonds=row['bonds']
        with np.load(directory/'physics.npz') as archive:
            ids=np.array(archive['chain_ids']);unique,counts=np.unique(ids,return_counts=True)
            if not np.array_equal(unique,np.arange(128)) or not np.all(counts==4):
                raise ValueError('replay requires the registered 128 whole chains/four records')
            if global_ids is not None and not np.array_equal(ids,global_ids):
                raise ValueError('candidate regional record ordering differs')
            global_ids=ids
            for cut,readout in row['regions'].items():
                if not np.array_equal(archive[cut+'__chain_ids'],ids):
                    raise ValueError('regional chain identities differ from source chains')
                data=np.array(archive[cut+'__per_record_matrices'])
                if data.shape!=(len(ids),2**len(readout['sites']),2**len(readout['sites'])):
                    raise ValueError('regional RDM dimensions/site map mismatch')
                if cut in regions:
                    raise ValueError('duplicate candidate regional cache')
                regions[cut]=dict(sites=readout['sites'],matrices=data)
    if identity is None:
        raise ValueError('at least one sealed cached endpoint required')
    if prereg is not None:
        if (tuple(prereg['snapshot_indices_per_chain'])!=identity[-1]
                or prereg['whole_chains']!=128 or prereg['records']!=512):
            raise ValueError('candidate sampling summary differs from preregistration')
    records=[];source_count=0
    for relative in spec['witness_sources']:
        directory=ROOT/relative;checked(directory/'completion.json')
        completion=json.loads((directory/'completion.json').read_text())
        for name,key in [('results.json','results_sha256'),('registration.json','registration_sha256'),
                         ('witnesses.npz','witnesses_sha256')]:checked(directory/name,completion[key])
        old=json.loads((directory/'results.json').read_text())
        reg=json.loads((directory/'registration.json').read_text())
        if old['registration']!=reg or old['witnesses_sha256']!=completion['witnesses_sha256']:
            raise ValueError('witness completion/registration inconsistent')
        for name,digest in reg['inputs'].items():checked(ROOT/name,digest)
        archived_sources(directory,reg['sources'])
        for prior in reg['config']['candidates']:
            _,prior_row,_=checked_readout(prior)
            if (prior_row['physical_problem_signature']!=known_problem
                    or prior_row['bonds']!=known_bonds):
                raise ValueError('frozen witness and new candidate Hamiltonian differ')
        with np.load(directory/'witnesses.npz') as archive:
            for old_record in old['records']:
                key=old_record['key'];cut=old_record['cut']
                definition=reg['config']['regions'][cut]
                if cut=='open5':
                    if 'winding6' not in regions or regions['winding6']['sites'][:-1]!=definition['sites']:
                        raise ValueError('open5 must be the declared partial trace of winding6')
                    values=regions['winding6']['matrices']
                    values=np.trace(values.reshape(len(values),32,2,32,2),axis1=2,axis2=4)
                else:
                    if cut not in regions or regions[cut]['sites']!=definition['sites']:
                        raise ValueError('frozen witness/new RDM physical site ordering mismatch')
                    values=regions[cut]['matrices']
                if prereg is not None and prereg['regions'].get(cut)!=definition['sites']:
                    raise ValueError('regional geometry differs from preregistration')
                cuts=tuple(tuple(part) for part in old_record['fit']['cuts'])
                if cuts!=canonical_bipartitions(definition['parties'],len(definition['sites'])):
                    raise ValueError('witness cuts differ from declared spatial parties')
                certificate=WitnessCertificate(np.array(archive[key+'__witness']),
                    tuple(np.array(q) for q in archive[key+'__Q']),cuts,old_record['fit']['certificate'])
                feasibility=validate_witness_certificate(certificate)
                fitting=old_record['fitting'];mask=np.ones(len(global_ids),bool)
                same=(fitting['kind']=='ViT' and str(fitting['state'])==str(expected_step))
                if same:
                    if policy!='original_heldout':
                        raise ValueError('same-training-step replay requires explicit original_heldout policy')
                    if fitting['chain_ids']!=list(range(64)):
                        raise ValueError('unrecognized original training-chain allocation')
                    mask=global_ids>=64
                selected=np.unique(global_ids[mask])
                audit=frozen_witness_chain_audit(certificate,values[mask],global_ids[mask])
                audit.update(fit_audit_chain_overlap=0,selected_chain_ids=selected.tolist(),
                    audit_state_differs_from_fit=not same)
                records.append(dict(source=relative,key=key,cut=cut,
                    trainer=old_record['trainer'],operator_span=old_record['operator_span'],
                    fitting=fitting,sites=definition['sites'],parties=definition['parties'],
                    state_id=str(expected_step),audit=audit,feasibility=feasibility))
                source_count+=1
    if not records:
        raise ValueError('no saved witnesses found')
    paired=[]
    for source,cut,trainer in dict.fromkeys((r['source'],r['cut'],r['trainer']) for r in records):
        group={r['operator_span']:r for r in records
               if (r['source'],r['cut'],r['trainer'])==(source,cut,trainer)}
        if set(group)!= {'full','low_order_and_spin_weights'}:
            raise ValueError('each frozen comparison must retain both preregistered operator spans')
        full,low=group['full']['audit'],group['low_order_and_spin_weights']['audit']
        if full['selected_chain_ids']!=low['selected_chain_ids']:
            raise ValueError('paired witness comparison needs the same audit chains')
        difference=np.array(low['chain_expectations'])-np.array(full['chain_expectations'])
        paired.append(dict(source=source,cut=cut,trainer=trainer,state_id=str(expected_step),
            signed_full_minus_restricted=float(difference.mean()),
            paired_whole_chain_SE=float(difference.std(ddof=1)/np.sqrt(len(difference))),
            chain_differences=difference.tolist(),independent_chains=len(difference),
            interpretation='Difference of signed frozen feasible witness bounds, NOT a certified gap between optimal GMNs.'))
    sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    registration=dict(config=spec,inputs=inputs,sources=sources,
        state=dict(step=identity[0],params_sha256=identity[1],checkpoint_sha256=identity[2],
                   physical_problem_signature=known_problem),
        all_saved_witnesses_included=True,source_witness_count=source_count,
        new_fits=0,new_training=0,new_samples=0,new_H_calls=0,new_amplitude_queries=0,
        solver_imported=False,physical_RDM_twirl=False,
        statistical_plan='Frozen witness linear estimates; all new endpoint chains, or explicitly original held-out chains for same-step reuse. Full-minus-restricted uses paired chain differences. No refitting/selection.')
    output.mkdir(parents=True)
    for name in sources:
        target=output/'source_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    _json(output/'registration.json',registration)
    result=dict(registration=registration,records=records,paired_comparisons=paired,
        seconds=perf_counter()-start,new_fits=0,new_training=0,new_samples=0,
        new_H_calls=0,new_amplitude_queries=0,phase_claim=False,
        caveat='Signed finite-sample witness estimates with pointwise approximate chain errors. No sampler-bias/multiple-testing guarantee. Coarse-party GME is not six-spin GME, QSL or topology; zero does not exclude entanglement.')
    _json(output/'results.json',result)
    _json(output/'completion.json',dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json')))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.config, args.output)
    print(json.dumps(dict(seconds=result.get('seconds'),
        records=len(result.get('records', result.get('regions', {}))), phase_claim=False)))
