"""Quality-controlled physical strip magnetization from frozen Born streams.

No state, sample, amplitude, or H action is generated. m=2*S_A^z is a
magnetic label, not q=2*S_A. Finite FCS is not an anyon/phase classifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copy2
from statistics import NormalDist
from time import perf_counter
from typing import Sequence

import numpy as np

from su2_nqs.benchmarks.mps_spin_rotation import physical_spin_rotation_summary
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest

ROOT = Path(__file__).resolve().parents[3]
METRICS = ('variance_sz', 'fourth_cumulant_sz', 'magnetic_number_entropy_nats', 'pi_rotation_real')


def prefix_chain_histograms(streams: Sequence[np.ndarray], cut: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep each (stream, chain) intact; axes are [snapshot, chain, site]."""
    if not streams or isinstance(cut, bool) or not isinstance(cut, int) or cut < 1:
        raise ValueError('nonempty Born streams and positive integer cut required')
    histograms = []
    n_sites = None
    for stream in streams:
        bits = np.asarray(stream)
        if (bits.ndim != 3 or bits.dtype.kind not in 'biu' or min(bits.shape) < 1
                or cut > bits.shape[-1] or np.any((bits != 0) & (bits != 1))):
            raise ValueError('valid snapshot/chain/site spin-half bits required')
        if n_sites is None:
            n_sites = bits.shape[-1]
        if bits.shape[-1] != n_sites or np.any(2*bits.sum(-1)-n_sites):
            raise ValueError('all inputs must be the same fixed-Sz=0 Hilbert space')
        counts = bits[..., :cut].sum(-1)
        for chain in range(bits.shape[1]):
            histograms.append(np.bincount(counts[:, chain], minlength=cut+1)/len(counts))
    return np.arange(-cut, cut+1, 2, dtype=np.int64), np.asarray(histograms)


def fcs_metrics(m: np.ndarray, p: np.ndarray, cut: int) -> tuple[np.ndarray, dict]:
    """Four fixed physical observables; full curve remains descriptive."""
    report = physical_spin_rotation_summary(m, p, n_region=cut,
        angles=np.linspace(0., 2*np.pi, 129))
    report['pi_rotation_real'] = report['rotation_real'][64]
    return np.asarray([report[k] for k in METRICS]), report


def analyze(config_path: Path) -> dict:
    """Execute one frozen retrospective comparison with chain uncertainty."""
    spec = json.loads(config_path.read_text())
    output = ROOT/spec['output']
    if output.exists():
        raise FileExistsError('completed results are immutable; choose a fresh output')
    if spec['fixed_metrics'] != list(METRICS) or spec['cut'] != 18:
        raise ValueError('this preregistration fixes cut18 and all four metrics')
    started = perf_counter()
    inputs = {str(config_path.relative_to(ROOT)): file_sha256(config_path)}
    loaded = []
    for item in spec['inputs']:
        reg_path = ROOT/item['registration']; audit_path = ROOT/item['audit_report']
        registration = json.loads(reg_path.read_text()); audit = json.loads(audit_path.read_text())
        for data, path, expected in ((registration,reg_path,item['registration_file_sha256']),
                                    (audit,audit_path,item['audit_file_sha256'])):
            if (file_sha256(path) != expected or data['result_sha256'] != canonical_sha256(
                    {k:v for k,v in data.items() if k != 'result_sha256'})):
                raise ValueError('physical registration or audit seal mismatch')
            inputs[str(path.relative_to(ROOT))] = expected
        config = registration['config']; model = config['model']
        if (config['j2_over_j1'] != item['j2'] or model['geometry'] != 'triangular'
                or model['rows'] != 6 or model['columns'] != 6 or model['target_m2'] != 0
                or audit['step'] != item['step']
                or audit['registration_sha256'] != registration['result_sha256']
                or audit['energy']['energy_per_site'] != item['energy_per_site']):
            raise ValueError('same physical torus, coupling, step or audit identity changed')
        streams, parents = [], []
        for entry in item['streams']:
            path = ROOT/entry['arrays']; manifest_path = ROOT/entry['manifest']
            manifest = json.loads(manifest_path.read_text())
            if (file_sha256(path) != entry['arrays_sha256']
                    or file_sha256(manifest_path) != entry['manifest_sha256']
                    or manifest['raw_sha256'] != entry['arrays_sha256']
                    or manifest['result_sha256'] != canonical_sha256(
                        {k:v for k,v in manifest.items() if k != 'result_sha256'})):
                raise ValueError('frozen Born input seal mismatch')
            with np.load(path) as archive:
                streams.append(np.array(archive['states']))
            parents.append((manifest['params_sha256'], manifest['registration_sha256']))
            if streams[-1].shape != (128, 64, 36):
                raise ValueError('fixed physical 6x6 audit dimensions changed')
            for selected in (path, manifest_path):
                inputs[str(selected.relative_to(ROOT))] = file_sha256(selected)
        if len(set(parents)) != 1 or len(streams) != 2:
            raise ValueError('independent streams must measure one unchanged physical parent')
        if parents[0] != (audit['params_sha256'],registration['result_sha256']):
            raise ValueError('Born streams do not belong to the sealed physical audit parent')
        loaded.append((item, streams, parents[0]))
    sources = transitive_local_source_manifest([Path(__file__)], root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    output.mkdir(parents=True)
    for path in sources:
        destination = output/'source_snapshots'/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(ROOT/path, destination)
    def write(name: str, payload: dict) -> None:
        atomic_json(output/name, dict(**payload, result_sha256=canonical_sha256(payload)))
    write('registration.json', dict(config=spec, inputs_sha256=inputs, sources=sources,
        retrospective=True, cut_and_metrics_frozen_before_execution=True))
    records, arrays = [], {}
    for i, (item, streams, parent) in enumerate(loaded):
        if perf_counter()-started > spec['seconds_cap']:
            raise TimeoutError('offline fixed budget exceeded')
        m, hist = prefix_chain_histograms(streams, spec['cut'])
        p = hist.mean(0)
        estimates, summary = fcs_metrics(m, p, spec['cut'])
        jack = (hist.sum(0)-hist)/(len(hist)-1)
        jack_values = np.array([fcs_metrics(m, row, spec['cut'])[0] for row in jack])
        se = np.sqrt((len(jack)-1)/len(jack)*((jack_values-jack_values.mean(0))**2).sum(0))
        arrays['r'+str(i)+'__probabilities'] = p
        arrays['r'+str(i)+'__chain_histograms'] = hist
        arrays['r'+str(i)+'__jackknife_metrics'] = jack_values
        records.append(dict(label=item['label'],j2=item['j2'],step=item['step'],
            params_sha256=parent[0],registration_sha256=parent[1],
            samples=sum(x.shape[0]*x.shape[1] for x in streams),independent_chains=len(hist),
            energy_per_site=item['energy_per_site'],
            metrics={k:dict(value=float(v),whole_chain_SE=float(e)) for k,v,e in zip(METRICS,estimates,se)},
            summary=summary,mixing_certified=False,strict_su2=False))
    comparisons = []
    z = NormalDist().inv_cdf(1.-.05/(2*len(METRICS)))
    for metric in METRICS:
        group0 = [r for r in records if r['j2'] == 0.]
        group1 = [r for r in records if r['j2'] == .125]
        if not group0 or not group1:
            raise ValueError('both preregistered coupling sequences are required')
        group0.sort(key=lambda r:r['step']); group1.sort(key=lambda r:r['step'])
        values = [[r['metrics'][metric]['value'] for r in group] for group in (group0,group1)]
        drift = sum(float(np.ptp(v)) for v in values)
        difference = values[1][-1]-values[0][-1]
        se = float(np.hypot(group0[-1]['metrics'][metric]['whole_chain_SE'],
                            group1[-1]['metrics'][metric]['whole_chain_SE']))
        comparisons.append(dict(metric=metric,difference=difference,whole_chain_SE=se,
            sum_empirical_quality_ranges=drift,normal_familywise_multiplier=z,
            threshold=drift+z*se,exceeds_empirical_threshold=bool(abs(difference)>drift+z*se),
            quality_ranges_are_bias_certificates=False))
    arrays['magnetic_labels'] = m
    np.savez_compressed(output/'readouts.npz', **arrays)
    result = dict(records=records,contrasts=comparisons,physical_region=list(range(18)),
        geometry='first three complete rows of6x6 torus; two physical boundaries',
        new_samples=0,new_H_calls=0,new_amplitude_queries=0,new_training=0,
        statistics='Whole independent stream/chain delete-one jackknife; four fixed normal/Bonferroni contrasts plus own endpoint ranges; neither mixing nor systematic bias certified.',
        warnings=['Magnetic FCS is not SU2 irrep or entanglement spectrum.',
                  'Variance is integrated two-body information, kappa4 only up to four-body. Larger support alone does not establish new information beyond all low-order correlations.',
                  'Number entropy is not entanglement entropy; no SET, DSL, cusp or phase classification.',
                  'Different couplings are not competing branches of the same Hamiltonian.'],
        raw_sha256=file_sha256(output/'readouts.npz'),seconds=perf_counter()-started)
    write('results.json',result)
    write('completion.json',dict(status='complete',results_sha256=file_sha256(output/'results.json'),
        raw_sha256=result['raw_sha256'],original_inputs_unchanged=all(file_sha256(ROOT/p)==h for p,h in inputs.items())))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args = parser.parse_args()
    report = analyze(args.config.resolve())
    print(json.dumps(dict(records=len(report['records']),contrasts=report['contrasts'],new_samples=0)))
