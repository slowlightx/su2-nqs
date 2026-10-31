"""One reusable, zero-query block-entanglement analysis for saved quality series."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.spin_sector_entanglement import spin_sector_entanglement
from su2_nqs.frontier.provenance import file_sha256
from .spin_tensor_network_reference import ROOT, _json, transitive_local_source_manifest


def scalar_values(report: dict) -> dict[str, float]:
    """Fixed scalar bank; do not search spectra for maximal state differences."""
    values = {k: report[k] for k in ('raw_negativity', 'weighted_local_charge_negativity',
                                    'weighted_multiplicity_negativity')}
    for key, row in report['sectors'].items():
        for field in ('screening_witness', 'multiplicity_negativity'):
            if row.get(field) is not None:
                values[key+'__'+field] = row[field]
    return values


def audit_spin_isotropy(directory: Path, *, n_sites: int) -> dict:
    """Paired scalar-minus-3zz diagnostics on existing independent full-H chains.

    For even spin-1/2 systems in Sz=0, non-singlet weight <= <S_tot^2>/2.
    This is an expectation bound; its Monte-Carlo point estimate/SE is not
    a certified probability bound. The rank-two correlation contrast is zero
    for an actual singlet. Preserve cross-observable chain covariance.
    """
    from .spin_spatial_vit_quality import read_sealed
    if isinstance(n_sites, bool) or not isinstance(n_sites, int) or n_sites < 2 or n_sites % 2:
        raise ValueError('even positive integer spin-1/2 site count required')
    saved = read_sealed(directory/'results.json')
    structure = read_sealed(directory/'structure.json')
    if file_sha256(directory/'structure.npz') != structure['raw_sha256']:
        raise ValueError('saved structure-factor input seal changed')
    with np.load(directory/'structure.npz') as arrays:
        ids, scalar, zz = arrays['chain_ids'], arrays['scalar'], arrays['zz']
        chains = np.unique(ids)
        if (len(chains) < 2 or len(ids) != saved['energy']['samples'] or
                len(chains) != saved['energy']['independent_chains'] or scalar.shape != zz.shape or
                scalar.shape != (len(ids),len(saved['structure_factors']))):
            raise ValueError('registered structure-factor chain/record layout differs')
        readout = {}
        for i, momentum in enumerate(saved['structure_factors']):
            contrast = (scalar[:,i]-3*zz[:,i]).real
            means = np.array([contrast[ids==c].mean() for c in chains])
            value, error = float(means.mean()), float(means.std(ddof=1)/np.sqrt(len(chains)))
            readout[momentum] = dict(scalar_minus_3zz=value, paired_chain_SE=error)
    spin = saved['total_spin_squared']
    return dict(rank_two_correlations=readout,
        nonsinglet_weight_upper_bound_point_estimate=spin['value']/2,
        point_estimate_SE=spin['chain_SE']/2, strict_su2=False,
        inputs_sha256={name:file_sha256(directory/name) for name in
                       ('results.json','structure.json','structure.npz')},
        warning='Energy/space-group quality is not strict SU2. Estimates are not '
                'certified symmetry weights or mixing guarantees. No input twirl/repair.')


def _conservative_witness_errors(row: dict) -> dict[str, float]:
    """Triangle-inequality SE upper bounds, not fake independent joint weights."""
    old = row.get('whole_chain_SE')
    if not old:
        return {}
    result = {}
    for q in (1, 3):
        names = [f'joint_qL{q}_qR{q}_qT{qt}' for qt in range(0, 2*q+1, 2)]
        if all(name in old for name in names):
            result[f'{q},{q}__screening_witness'] = ((1-1/(q+1))*old[names[0]] +
                sum(old[name]/(q+1) for name in names[1:]))
    return result


def analyze(config: Path) -> dict:
    """Actual coherent RDMs; local-charge postselection only; quality controls."""
    spec = json.loads(config.read_text())
    output = ROOT/spec['output']
    if output.exists():
        raise FileExistsError('use a new output: completed analysis is immutable')
    start, inputs, records = perf_counter(), {}, []
    def checked(path: Path, expected: str | None = None) -> None:
        digest = file_sha256(path)
        if expected is not None and expected != digest:
            raise ValueError('sealed raw input changed: '+str(path))
        inputs[str(path.relative_to(ROOT))] = digest
    directory = ROOT/spec['series']
    series_path, arrays_path = directory/'results.json', directory/'readouts.npz'
    checked(series_path, spec['series_results_sha256'])
    checked(arrays_path, spec['series_readouts_sha256'])
    source = json.loads(series_path.read_text())
    cached = {}
    for entry in spec.get('jackknife_caches', []):
        path = ROOT/entry['path']; checked(path, entry['sha256'])
        with np.load(path) as archive:
            cached[entry['label']] = {k: np.array(archive[k]) for k in archive.files
                                     if k.endswith('__rho') or k.endswith('__jackknife_rhos')}
    def measure(row: dict, rho: np.ndarray, jackknife: np.ndarray | None = None) -> None:
        if perf_counter()-start > spec['seconds_cap']:
            raise RuntimeError('analysis budget exhausted; no completion seal')
        report = spin_sector_entanglement(rho, row['left'])
        scalars = scalar_values(report)
        errors, bounds = {}, _conservative_witness_errors(row)
        if jackknife is not None:
            deleted = [scalar_values(spin_sector_entanglement(r, row['left'])) for r in jackknife]
            for name in scalars:
                if all(name in r for r in deleted):
                    a = np.array([r[name] for r in deleted])
                    errors[name] = float(np.sqrt((len(a)-1)/len(a)*sum((a-a.mean())**2)))
        records.append(dict(label=row['label'], method=row['method'],
            geometry=row['problem']['geometry'], j2=row['problem']['j2'],
            boundary=row['problem']['boundary'], physical_problem_signature=row['physical_problem_signature'],
            cut=row['cut'], sites=row['sites'], left=row['left'],
            step=row.get('step'), chi=row.get('chi'), energy=row.get('energy'),
            scalars=scalars, whole_chain_SE=errors,
            witness_SE_upper_bounds=bounds, report=report,
            nonlinear_bias_removed=False, new_samples=0))
    with np.load(arrays_path) as archive:
        for row in source['records']:
            if len(row['sites']) != 6 or len(row['left']) != 3:
                continue
            key = row['label']+'__'+row['cut']+'__rho'
            rho = archive[key]
            cache = cached.get(row['label'], {})
            mean_key, jack_key = row['cut']+'__rho', row['cut']+'__jackknife_rhos'
            if jack_key in cache and np.linalg.norm(cache[mean_key]-rho) > 1e-12:
                raise ValueError('jackknife belongs to another physical RDM')
            measure(row, rho, cache.get(jack_key))
    # New immutable milestones use the same raw mean/whole-chain protocol.
    from .spin_spatial_vit_quality import read_sealed
    from .spin_su2_analysis import _problem_signature
    for entry in spec.get('additional_cached', []):
        directory, audit_path = ROOT/entry['directory'], ROOT/entry['audit']
        for name in ('results.json', 'registration.json'):
            checked(directory/name, entry[name+'_sha256'])
        checked(audit_path, entry['audit_sha256'])
        saved, reg, audit = [read_sealed(p) for p in
                            (directory/'results.json', directory/'registration.json', audit_path)]
        problem = reg['config']['problem']
        signature, _ = _problem_signature(problem)
        if (saved['registration_sha256'] != reg['result_sha256'] or
                signature != saved['physical_problem_signature'] or
                saved['step'] != audit['step'] or
                any(saved[k] != audit[k] for k in ('params_sha256', 'checkpoint_sha256'))):
            raise ValueError('new regional readout and independent energy audit differ')
        checked(directory/'physics.npz', saved['raw_sha256'])
        with np.load(directory/'physics.npz') as archive:
            for cut, region in saved['regions'].items():
                row = dict(label=entry['label'], method='NQS', problem=problem,
                    physical_problem_signature=signature, cut=cut, sites=region['sites'],
                    left=region['left'], step=saved['step'], energy=audit['energy'],
                    whole_chain_SE=region['scalar_whole_chain_SE'])
                measure(row, archive[cut+'__rho'], archive[cut+'__jackknife_rhos'])
    # A second, analytically singlet family uses the same physical-RDM readout.
    # Keep its two independent audit streams separate: averaging normalized
    # RDMs is NOT the same as pooling the underlying importance ratios.
    for entry in spec.get('agp_cached', []):
        directory = ROOT/entry['directory']
        checked(directory/'registration.json', entry['registration_sha256'])
        registration = read_sealed(directory/'registration.json')
        suite = registration['config']
        for case in suite['cases']:
            folder = directory/case['name']
            checked(folder/'completion.json')
            complete = read_sealed(folder/'completion.json')
            if complete['status'] != 'complete':
                raise ValueError('AGP independent audit is not complete')
            for name in ('registration.json', 'results.json', 'frozen_endpoint.json'):
                checked(folder/name, complete['outputs'][name])
            result = read_sealed(folder/'results.json')
            frozen = read_sealed(folder/'frozen_endpoint.json')
            local = read_sealed(folder/'registration.json')
            if (result['case'] != case or not result['complete_bounded_audit'] or
                    local['case'] != case or local['suite_registration_sha256'] !=
                    registration['result_sha256'] or result['frozen_checkpoints'] !=
                    frozen['checkpoints']):
                raise ValueError('AGP case/registration/checkpoints differ')
            problem = dict(geometry=case['lattice'], rows=6, columns=6,
                j2=case['j2_over_j1'], boundary='PBC_both_physical',
                regions={k:v['sites'] for k,v in suite['regions'].items()},
                left_sites={k:v['left'] for k,v in suite['regions'].items()})
            signature, _ = _problem_signature(problem)
            for stream in (0, 1):
                audit_path = folder/f'audit_{stream}.json'
                raw_path = audit_path.with_suffix('.npz')
                checked(audit_path, complete['outputs'][audit_path.name])
                audit = read_sealed(audit_path)
                checked(raw_path, audit['raw_file_sha256'])
                if (complete['outputs'][raw_path.name] != audit['raw_file_sha256'] or
                        audit['checkpoints'] != frozen['checkpoints']):
                    raise ValueError('AGP raw RDM/checkpoint seal differs')
                with np.load(raw_path) as archive:
                    for quality, moments in audit['states'].items():
                        for cut, region in suite['regions'].items():
                            if len(region['sites']) != 6:
                                continue
                            energy = dict(energy_per_site=moments['energy_per_site'],
                                chain_mean_standard_error=moments['energy_chain_SE'],
                                variance_per_site=moments['variance_per_site'],
                                variance_chain_standard_error=moments['variance_chain_SE'])
                            row = dict(label=f'agp_{case["name"]}_audit{stream}_{quality}',
                                method='AGP', problem=problem,
                                physical_problem_signature=signature, cut=cut,
                                sites=region['sites'], left=region['left'],
                                step=int(quality.removeprefix('step_')), energy=energy)
                            measure(row, archive[f'{cut}_{quality}_rho'],
                                    archive[f'{cut}_{quality}_jackknife_rhos'])
    controls = []
    def control(label: str, v: np.ndarray) -> None:
        controls.append(dict(label=label, report=spin_sector_entanglement(np.outer(v,v.conj()), (0,1,2))))
    # Independent bit construction, not CG-generated controls.
    for label, pairs in [('three_crossing_dimers', [(0,3),(1,4),(2,5)]),
                         ('one_crossing_dimer', [(0,1),(2,3),(4,5)])]:
        v = np.zeros(64, complex)
        for packed in range(64):
            bits = [(packed >> (5-i)) & 1 for i in range(6)]
            if all(bits[a] != bits[b] for a,b in pairs):
                v[packed] = np.prod([1 if bits[a] else -1 for a,b in pairs])/np.sqrt(8)
        control(label, v)
    v = np.array([1.], complex)
    for i in range(6):
        v = np.kron(v, np.array([np.exp(2j*np.pi*(i%3)/3),1])/np.sqrt(2))
    control('classical_120_product_NOT_quantum_AFM', v)
    sources = transitive_local_source_manifest((Path(__file__),), root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    checked(config.resolve())
    output.mkdir(parents=True)
    for path in sources:
        target = output/'source_snapshots'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/path).read_bytes())
    _json(output/'registration.json', dict(config=spec, inputs_sha256=inputs, sources=sources))
    result = dict(records=records, controls=controls, seconds=perf_counter()-start,
        new_samples=0, new_H_calls=0, new_amplitude_queries=0, new_training=0,
        strict_entanglement_spectrum=False, phase_claim=False,
        warning='Exploratory frozen operator bank, not held-out phase evidence. W>0 certifies '
                '3+3 block entanglement, not a QSL. Multiplicity negativity is obtained after '
                'LOCAL charge measurement and tracing magnetic indices; no nonlocal singlet '
                'postselection. Nonlinear sampling bias remains. Missing SEs stay missing. '
                'Do not compare lattices or translate differences into Z2/U1 labels.')
    _json(output/'results.json', result)
    _json(output/'completion.json', dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json')))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.config)
    print(json.dumps(dict(records=len(result['records']), seconds=result['seconds'],
                          new_H_calls=0, phase_claim=False)))


def render_quality(result_path: Path, destination: Path) -> list[Path]:
    """Error-point/quality lines only; never promote noisy negativity to phases."""
    import matplotlib.pyplot as plt
    result = json.loads(result_path.read_text())
    rows = result['records']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    styles = dict(compact6=('tab:blue', 'o'), local_hexagon6=('tab:orange', 's'),
                  winding6=('tab:green', '^'))
    for cut, (color, marker) in styles.items():
        selected = [r for r in rows if r['method']=='NQS' and r['geometry']=='triangular'
                    and r['j2']==.125 and r['cut']==cut
                    and '3,3__screening_witness' in r['whole_chain_SE']]
        for index, row in enumerate(selected):
            e=row['energy']['energy_per_site']; y=row['scalars']['3,3__screening_witness']
            axes[0].errorbar(e,y,xerr=row['energy']['chain_mean_standard_error'],
                yerr=row['whole_chain_SE']['3,3__screening_witness'],fmt=marker,
                color=color,alpha=.75,label=cut if index==0 else None)
        ref = [r for r in rows if r['method']=='DMRG' and r['cut']==cut]
        if ref:
            last=ref[-1]
            axes[0].axhline(last['scalars']['3,3__screening_witness'],color=color,
                           linestyle=':',lw=1,label=cut+' DMRG')
    axes[0].axhline(0,color='0.5',lw=.8)
    axes[0].set(xlabel='ViT E/N (independent audit)', ylabel=r'$W_{q=3}$',
                title=r'Triangle $J_2=0.125$: quartet screening')
    axes[0].legend(fontsize=7,loc='best');axes[0].ticklabel_format(axis='x',useOffset=False)
    for j, color in ((0.,'tab:blue'),(.125,'tab:orange')):
        selected=[r for r in rows if r['geometry']=='triangular' and r['method']=='NQS'
                  and r['j2']==j and r['cut']=='local_hexagon6'
                  and '1,1__multiplicity_negativity' in r['whole_chain_SE']]
        for index,row in enumerate(selected):
            # Newly restarted condition counters differ from inherited steps.
            step=row['step']+(10600 if row['label'].startswith('tri125_') else 0)
            axes[1].errorbar(step,row['scalars']['1,1__multiplicity_negativity'],
                yerr=row['whole_chain_SE']['1,1__multiplicity_negativity'],fmt='o',
                color=color,alpha=.8,label=f'ViT J2={j}' if index==0 else None)
    axes[1].axhline(0,color='0.5',lw=.8,linestyle=':')
    axes[1].set(xlabel='Checkpoint step (condition arms shifted by 10600)',
                ylabel='Conditional multiplicity negativity (qL=qR=1)',
                title='Local hexagon: nonlinear bias remains')
    axes[1].legend(fontsize=8)
    for j,color in ((0.,'tab:blue'),(.125,'tab:orange')):
        chosen=[r for r in rows if r['method']=='NQS' and r['geometry']=='triangular'
                and r['j2']==j and 'raw_negativity' in r['whole_chain_SE']]
        for index,row in enumerate(chosen):
            position=list(styles).index(row['cut'])+(j-.0625)*.8
            axes[2].errorbar(position,row['scalars']['raw_negativity'],
                yerr=row['whole_chain_SE']['raw_negativity'],fmt='o',color=color,
                alpha=.7,label=f'ViT J2={j}' if index==0 else None)
    for index,cut in enumerate(styles):
        ref=[r for r in rows if r['method']=='DMRG' and r['cut']==cut]
        if ref: axes[2].plot(index,ref[-1]['scalars']['raw_negativity'],'kx',
                            label='DMRG J2=.125' if index==0 else None)
    axes[2].set(xticks=range(3),xticklabels=['compact6','hexagon6','winding6'],
                ylabel='Raw negativity across the chosen 3+3 cut',
                title='Geometry and quality, not a QSL classifier')
    axes[2].legend(fontsize=8)
    destination.mkdir(parents=True,exist_ok=True)
    paths=[destination/f'su2_screening_entanglement_quality_20261012.{suffix}'
           for suffix in ('png','pdf')]
    for path in paths:fig.savefig(path,dpi=170)
    plt.close(fig)
    return paths


if __name__ == '__main__':
    main()
