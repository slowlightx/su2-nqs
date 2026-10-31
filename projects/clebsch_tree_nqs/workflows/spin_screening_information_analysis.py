"""Bounded screening-resonance analysis of saved RDM/whole-chain quality series."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copy2
from time import perf_counter

import numpy as np
from scipy.stats import norm

from su2_nqs.benchmarks.spin_rotation_calibration import lieb_mattis_singlet, singlet_covering_state
from su2_nqs.benchmarks.spin_screening_information import screening_information_readout
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from .triangle_symmetry_contrast import jackknife_se, quality_contrast

ROOT = Path(__file__).resolve().parents[3]
SCALARS = ('p11', 'holevo_nats', 'weighted_holevo_nats', 'conditional_trace_distance',
           'pseudoscalar_product', 'pseudoscalar_low', 'pseudoscalar_high',
           'pseudoscalar_connected', 'pseudoscalar_left', 'pseudoscalar_right')


def analyze(config_path: Path) -> dict:
    """Frozen geometry/capacity-independent physical readout; no new state work."""
    cfg = json.loads(config_path.read_text())
    output = ROOT/cfg['output']
    if output.exists():
        raise FileExistsError('completed study immutable; choose another output version')
    inputs = {}; loaded = []
    def check(path: Path, expected: str | None = None) -> None:
        digest = file_sha256(path)
        if expected is not None and expected != digest:
            raise ValueError('input changed: '+str(path))
        inputs[str(path.resolve().relative_to(ROOT))] = digest
    check(config_path)
    base_path = ROOT/cfg['nqs_input_config']; check(base_path, cfg['nqs_input_config_sha256'])
    base = json.loads(base_path.read_text())
    for entry in base['inputs']:
        folder = ROOT/entry['directory']; payload = {}
        for name, key in [('results.json', 'results_sha256'), ('registration.json', 'registration_sha256'),
                          ('physics.npz', 'arrays_sha256')]:
            check(folder/name, entry[key])
            if name.endswith('.json'):
                obj = json.loads((folder/name).read_text())
                if canonical_sha256({k:v for k,v in obj.items() if k != 'result_sha256'}) != obj['result_sha256']:
                    raise ValueError('producer canonical seal failed')
                payload[name] = obj
        producer, reg = payload['results.json'], payload['registration.json']
        problem = reg['config']['problem']
        if (problem['geometry'] != 'triangular' or problem['j2'] != entry['j2'] or
                (problem['rows'], problem['columns']) != (6, 6) or
                problem['boundary'] != 'PBC_both_physical' or not producer['full_A1'] or
                producer['step'] != entry['step'] or producer['registration_sha256'] != reg['result_sha256'] or
                producer['raw_sha256'] != entry['arrays_sha256'] or
                producer['params_sha256'] != reg['config']['params_sha256']):
            raise ValueError('producer/physical state mismatch')
        with np.load(folder/'physics.npz') as arrays:
            for cut, sites in base['regions'].items():
                if problem['regions'][cut] != sites or problem['left_sites'][cut] != cfg['left_sites']:
                    raise ValueError('geometry/site ordering mismatch')
                jack = np.array(arrays[cut+'__jackknife_rhos'])
                if len(jack) != producer['independent_chains']:
                    raise ValueError('whole-chain group count changed')
                loaded.append((dict(label=entry['label'], method='ViT', j2=entry['j2'],
                    step=entry['step'], cut=cut, sites=sites, chi=None,
                    checkpoint_sha256=producer['checkpoint_sha256']), np.array(arrays[cut+'__rho']), jack))
    for entry in cfg['dmrg_series']:
        folder = ROOT/entry['directory']; check(folder/'results.json', entry['results_sha256'])
        check(folder/'readouts.npz', entry['arrays_sha256']); series = json.loads((folder/'results.json').read_text())
        with np.load(folder/'readouts.npz') as arrays:
            for row in series['records']:
                if row['label'] not in entry['labels'] or row['cut'] not in base['regions']:
                    continue
                cut = row['cut']; sites = row.get('sites', base['regions'][cut])
                if sites != base['regions'][cut] or row.get('left', cfg['left_sites']) != cfg['left_sites']:
                    raise ValueError('DMRG cut mismatch')
                sweep = row.get('sweep') or int(row['label'].split('_')[2])
                loaded.append((dict(label=row['label'], method='DMRG', j2=.125, step=sweep,
                    cut=cut, sites=sites, chi=row['chi'], energy=row.get('energy', row.get('energy_per_site'))),
                    np.array(arrays[row['label']+'__'+cut+'__rho']), None))
    sources = transitive_local_source_manifest([Path(__file__)], root=ROOT,
        package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    output.mkdir(parents=True)
    for name in sources:
        target = output/'source_snapshots'/name; target.parent.mkdir(parents=True, exist_ok=True)
        copy2(ROOT/name, target)
    def write(name: str, payload: dict) -> None:
        atomic_json(output/name, {**payload, 'result_sha256':canonical_sha256(payload)})
    write('registration.json', dict(config=cfg, inputs_sha256=inputs, sources=sources,
        prospective=False, numerical_comparisons_frozen_before_execution=True,
        scope='Actual invariant channel readouts, not repaired input states or strict SU2 entanglement levels.'))
    started = perf_counter(); rows = []; saved = {}
    for meta, rho, jack in loaded:
        if perf_counter()-started > cfg['seconds_cap']:
            raise TimeoutError('frozen offline budget reached')
        report = screening_information_readout(rho, cfg['left_sites'])
        values = {key: report[key] for key in SCALARS}; se = {}; prefix = 'r'+str(len(rows))
        saved[prefix+'__rho'] = rho
        for qt, matrix in zip((0, 2), report['reduced_multiplicity']):
            saved[prefix+'__reduced_qT'+str(qt)] = matrix
        if jack is not None:
            deleted = [screening_information_readout(r, cfg['left_sites']) for r in jack]
            for key in SCALARS:
                if values[key] is not None and all(r[key] is not None for r in deleted):
                    a = np.array([r[key] for r in deleted]); saved[prefix+'__'+key+'__jackknife'] = a
                    se[key] = float(jackknife_se(a))
        rows.append(dict(**meta, values=values, whole_chain_SE=se, chains=0 if jack is None else len(jack),
            array_prefix=prefix, screening_probabilities=report['screening_probabilities'],
            operator_diagnostics=report['operator_diagnostics'], strict_su2_spectrum=False,
            nonlinear_bias_removed=False, DMRG_statistical_SE_measured=False if jack is None else None))
    critical = float(norm.ppf(1-cfg['family_alpha']/(2*cfg['comparison_family_size'])))
    comparisons = []
    for cut in base['regions']:
        zero = sorted((r for r in rows if r['method']=='ViT' and r['j2']==0 and r['cut']==cut), key=lambda r:r['step'])
        spinliquid = sorted((r for r in rows if r['method']=='ViT' and r['j2']==.125 and r['cut']==cut), key=lambda r:r['step'])
        dmrg = sorted((r for r in rows if r['method']=='DMRG' and r['cut']==cut), key=lambda r:(r['chi'],r['step']))
        for key in cfg['primary_readouts']:
            for label, a, b in [('coupling_0125_minus_0', zero, spinliquid), ('DMRG_minus_ViT_0125', spinliquid, dmrg)]:
                if not a or not b:
                    continue
                errors = [x[-1]['whole_chain_SE'].get(key, 0.) for x in (a,b)]
                contrast = quality_contrast([r['values'][key] for r in a], [r['values'][key] for r in b],
                    *errors, multiplier=critical)
                contrast['pointwise_multiple_testing_corrected'] = True
                comparisons.append(dict(type=label, cut=cut, readout=key, endpoint_labels=[a[-1]['label'], b[-1]['label']],
                    family_size=cfg['comparison_family_size'], critical_SE=critical,
                    missing_DMRG_SE_is_not_zero_uncertainty=label.startswith('DMRG'), **contrast))
    if len(comparisons) != cfg['comparison_family_size']:
        raise ValueError('comparison family differs from frozen size')
    controls = []
    vectors = {
        'three_crossing_dimers': singlet_covering_state(6, [(0,3),(1,4),(2,5)]),
        'one_crossing_dimer': singlet_covering_state(6, [(0,1),(2,3),(4,5)]),
        'SU2_restored_Lieb_Mattis': lieb_mattis_singlet(6, (0,2,4)),
    }
    v = np.array([1.], complex)
    for i in range(6): v = np.kron(v, np.array([np.exp(2j*np.pi*(i%3)/3),1])/np.sqrt(2))
    vectors['classical_120_product'] = v
    for name, v in vectors.items():
        rho = np.outer(v,v.conj()); report = screening_information_readout(rho, cfg['left_sites'])
        controls.append(dict(label=name, values={key:report[key] for key in SCALARS}, QSL=False))
        saved['control__'+name+'__rho'] = rho
    np.savez_compressed(output/'readouts.npz', **saved)
    result = dict(records=rows, comparisons=comparisons, controls=controls, seconds=perf_counter()-started,
        raw_sha256=file_sha256(output/'readouts.npz'), new_samples=0, new_H_calls=0,
        new_amplitude_queries=0, new_MPS_contractions=0, new_training=0, phase_claim=False,
        statistical_scope='Whole-chain delete-one errors preserve endpoint correlations. Coupling endpoints independent. Normal Bonferroni sensitivity, not exact finite-chain confidence or mixing certificate. Entropy/conditioning plug-in bias is not removed. DMRG drift deterministic, not zero error.',
        interpretation='Holevo relates singlet/triplet screening to internal reduced resonance states; high pseudoscalar remainder is independent only of explicitly registered linear operator span, not a maximum-entropy prediction or QSL signature.')
    write('results.json', result)
    write('completion.json', dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json')))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    r = analyze(parser.parse_args().config)
    print(json.dumps(dict(records=len(r['records']), seconds=r['seconds'], phase_claim=False)))
