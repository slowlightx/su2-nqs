"""Frozen same-geometry, quality-controlled spin-symmetry contrasts.

This is offline analysis of saved physical RDMs. No state is repaired or
trained, and no new Born draw, amplitude query or Hamiltonian action occurs.
q=2S and m=2M are doubled labels; bit zero is down, site zero is the MSB.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copy2
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.regional_su2_readout import regional_schur_columns
from su2_nqs.benchmarks.spin_operator_information import joint_spin_operator_split
from su2_nqs.benchmarks.spin_rotation_disorder import regional_spin_rotation
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest

ROOT = Path(__file__).resolve().parents[3]


def rotation_operators(n: int, angles: np.ndarray) -> dict[str, np.ndarray]:
    """Physical-axis and scalar rotation operators, no twirled state.

    Scalar is the SU2 character average. It is already determined by p(q),
    not extra information about multiplicity coherences or spinons.
    """
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 8:
        raise ValueError('bounded integer region size required')
    angles = np.asarray(angles)
    if (angles.ndim != 1 or not len(angles) or angles.dtype.kind not in 'fiu'
            or not np.isfinite(angles).all()):
        raise ValueError('nonempty finite real angles required')
    bits = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    m = 2*bits.sum(1)-n
    z = np.array([np.diag(np.exp(.5j*m*t)) for t in angles])
    x = np.array([[1.]])
    for _ in range(n):
        x = np.kron(x, np.array([[1., 1.], [-1., 1.]])/np.sqrt(2))
    xrot = np.einsum('ij,tjk,lk->til', x, z, x, optimize=True)
    u, labels = regional_schur_columns(n)
    chars = np.array([[np.exp(.5j*np.arange(-q, q+1, 2)*t).mean()
                       for q, _, _ in labels] for t in angles])
    scalar = np.einsum('ij,tj,kj->tik', u, chars, u.conj(), optimize=True)
    return dict(z=z, x=xrot, scalar=scalar)


def linear_means(rhos: np.ndarray, operators: np.ndarray) -> np.ndarray:
    """Coherent linear expectations preserving each whole-chain jackknife row."""
    return np.einsum('nij,kji->nk', rhos, operators, optimize=True)


def jackknife_se(values: np.ndarray) -> np.ndarray:
    """Real whole-chain leave-one-out SE, with covariance retained in rows."""
    if len(values) < 2:
        raise ValueError('at least two leave-one-chain-out estimates required')
    centered = np.asarray(values).real-np.asarray(values).real.mean(0)
    return np.sqrt((len(centered)-1)/len(centered)*np.sum(centered**2, axis=0))


def quality_contrast(values0: list[float], values1: list[float],
                     error0: float, error1: float, *, multiplier: float = 2.) -> dict:
    """Fixed-endpoint difference versus own-state ranges plus descriptive SE.

    Quality ranges are empirical drifts, NOT certified bias bounds. This
    pointwise gate does not certify equilibrium or correct for many tests.
    """
    if (not values0 or not values1 or min(error0, error1, multiplier) < 0
            or not np.isfinite([*values0, *values1, error0, error1, multiplier]).all()):
        raise ValueError('finite series and nonnegative errors required')
    difference = values1[-1]-values0[-1]
    drift0, drift1 = np.ptp(values0), np.ptp(values1)
    se = np.hypot(error0, error1)
    threshold = drift0+drift1+multiplier*se
    return dict(difference=float(difference), independent_whole_chain_SE=float(se),
        own_quality_range_0=float(drift0), own_quality_range_0125=float(drift1),
        empirical_threshold=float(threshold), exceeds_empirical_threshold=bool(abs(difference)>threshold),
        quality_ranges_are_certified_bias_bounds=False, pointwise_multiple_testing_corrected=False)


def analyze(config_path: Path) -> dict:
    """Execute the frozen bounded retrospective study; output never overwritten."""
    config = json.loads(config_path.read_text())
    output = ROOT/config['output']
    if output.exists():
        raise FileExistsError('use a new output; completed study is immutable')
    started = perf_counter()
    config_sha = file_sha256(config_path)
    inputs: dict[str, str] = {str(config_path.resolve().relative_to(ROOT)): config_sha}
    loaded = []
    for entry in config['inputs']:
        directory = ROOT/entry['directory']
        objects = {}
        for name, key in [('results.json','results_sha256'),
                          ('registration.json','registration_sha256'),
                          ('physics.npz','arrays_sha256')]:
            path = directory/name
            digest = file_sha256(path)
            if digest != entry[key]:
                raise ValueError('input seal changed: '+str(path))
            inputs[str(path.relative_to(ROOT))] = digest
            if path.suffix == '.json':
                data = json.loads(path.read_text())
                if data['result_sha256'] != canonical_sha256({k:v for k,v in data.items() if k!='result_sha256'}):
                    raise ValueError('canonical input seal changed')
                objects[name] = data
        registration, producer = objects['registration.json'], objects['results.json']
        problem = registration['config']['problem']
        if (problem['geometry'] != 'triangular' or problem['j2'] != entry['j2']
                or problem['boundary'] != 'PBC_both_physical'
                or problem['rows'] != 6 or problem['columns'] != 6
                or not producer['full_A1'] or producer['step'] != entry['step']
                or producer['registration_sha256'] != registration['result_sha256']
                or producer['raw_sha256'] != entry['arrays_sha256']
                or producer['params_sha256'] != registration['config']['params_sha256']):
            raise ValueError('physical state or producer provenance mismatch')
        for cut, sites in config['regions'].items():
            if (problem['regions'][cut] != sites or problem['left_sites'][cut] != config['left_sites']):
                raise ValueError('physical region or subblock ordering mismatch')
        with np.load(directory/'physics.npz') as archive:
            arrays = {k: np.array(archive[k]) for k in archive.files
                      if k.endswith('__rho') or k.endswith('__jackknife_rhos')}
        loaded.append((entry, producer, arrays))
    # Seal all selected inputs BEFORE numerical analysis. A completed study
    # also keeps the exact local source closure, not just current filenames.
    sources = transitive_local_source_manifest([Path(__file__)], root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    output.mkdir(parents=True)
    for name in sources:
        destination = output/'source_snapshots'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(ROOT/name, destination)
    def write(name: str, payload: dict) -> None:
        atomic_json(output/name, {**payload, 'result_sha256':canonical_sha256(payload)})
    write('registration.json',dict(config=config, config_sha256=config_sha,
        inputs_sha256=inputs, sources=sources, prospective=False,
        numerical_comparisons_frozen_before_execution=True))
    angles = np.linspace(0, 2*np.pi, config['angle_points'])
    if len(angles)%2 != 1:
        raise ValueError('odd common angle grid must contain pi')
    pi = (len(angles)-1)//2
    ops = rotation_operators(6, angles)
    split = joint_spin_operator_split(6, config['left_sites'], (3,3,0))
    scalar_ops = np.stack([split.full, split.high])
    records, saved = [], {'angles': angles}
    maxima = dict(core_reference_error=0., producer_scalar_error=0., center_identity_error=0.)
    for entry, producer, arrays in loaded:
        for cut in config['regions']:
            if perf_counter()-started > config['seconds_cap']:
                raise TimeoutError('registered offline analysis cap reached')
            rho = arrays[cut+'__rho']
            jack = arrays[cut+'__jackknife_rhos']
            if len(jack) != producer['independent_chains']:
                raise ValueError('jackknife group count mismatch')
            reference = regional_spin_rotation(rho, angles)
            prefix = 'r'+str(len(records))
            summary = {}
            for axis, operator in ops.items():
                mean = linear_means(rho[None],operator)[0]
                jk = linear_means(jack,operator)
                error = jackknife_se(jk)
                maxima['core_reference_error'] = max(maxima['core_reference_error'],float(np.max(abs(mean-reference[axis]))))
                maxima['center_identity_error'] = max(maxima['center_identity_error'],float(abs(mean[-1]-1)))
                saved[prefix+'__'+axis] = mean
                saved[prefix+'__'+axis+'__jackknife'] = jk
                saved[prefix+'__'+axis+'__SE'] = error
                summary[axis] = dict(pi_real=float(mean[pi].real),pi_SE=float(error[pi]))
            means = linear_means(rho[None],scalar_ops)[0].real
            jk_scalar = linear_means(jack,scalar_ops).real
            scalar_errors = jackknife_se(jk_scalar)
            for i,name in enumerate(config['fixed_scalar_readouts']):
                saved[prefix+'__'+name+'__jackknife'] = jk_scalar[:,i]
                expected = producer['regions'][cut]['scalars'][name]
                maxima['producer_scalar_error'] = max(maxima['producer_scalar_error'],abs(float(means[i])-expected))
                summary[name] = dict(value=float(means[i]),whole_chain_SE=float(scalar_errors[i]))
            # Raw x-z difference retains same-chain covariance.
            difference = saved[prefix+'__z__jackknife']-saved[prefix+'__x__jackknife']
            summary['z_minus_x'] = dict(pi_real=summary['z']['pi_real']-summary['x']['pi_real'],
                                      pi_SE=float(jackknife_se(difference)[pi]))
            records.append(dict(label=entry['label'],j2=entry['j2'],step=entry['step'],cut=cut,
                sites=config['regions'][cut],left_sites=config['left_sites'],array_prefix=prefix,
                values=summary,records=producer['records'],chains=len(jack),
                strict_SU2=False,full_A1=True,input_repaired=False,
                factorization_frobenius=reference['factorization_frobenius']))
    contrasts, curves = [], []
    rng = np.random.default_rng(config['multiplier_seed'])
    # The same endpoint chain groups enter every angle, axis and region.
    # Reuse multipliers globally to retain cross-region/axis covariance too.
    endpoint_counts = [len(saved[r['array_prefix']+'__z__jackknife'])
        for j in (0,.125) for r in [max((x for x in records if x['j2']==j),key=lambda x:x['step'])]]
    multipliers = [rng.normal(size=(config['multiplier_replicates'],n)) for n in endpoint_counts]
    for cut in config['regions']:
        sequences = [sorted([r for r in records if r['cut']==cut and r['j2']==j],
                            key=lambda r:r['step']) for j in (0,.125)]
        endpoints = [s[-1] for s in sequences]
        for name in ('z','x','scalar',*config['fixed_scalar_readouts']):
            value, se = ('pi_real','pi_SE') if name in ops else ('value','whole_chain_SE')
            report = quality_contrast([r['values'][name][value] for r in sequences[0]],
                [r['values'][name][value] for r in sequences[1]],
                endpoints[0]['values'][name][se],endpoints[1]['values'][name][se],
                multiplier=config['point_threshold_SE'])
            contrasts.append(dict(cut=cut,readout=name,endpoints=[r['label'] for r in endpoints],**report))
        for axis in ops:
            a,b = [saved[r['array_prefix']+'__'+axis] for r in endpoints]
            ja,jb = [saved[r['array_prefix']+'__'+axis+'__jackknife'].real for r in endpoints]
            centered = [j-j.mean(0) for j in (ja,jb)]
            se = np.hypot(jackknife_se(ja),jackknife_se(jb))
            if [len(c) for c in centered] != endpoint_counts:
                raise ValueError('common endpoint group count differs between geometries')
            noises = sum(g@c*np.sqrt((len(c)-1)/len(c)) for g,c in zip(multipliers,centered))
            p='c'+str(len(curves))
            saved[p+'__difference'] = b-a
            saved[p+'__SE'] = se
            saved[p+'__multiplier_noises'] = noises
            saved[p+'__quality_range_0'] = np.ptp(np.stack([saved[r['array_prefix']+'__'+axis].real for r in sequences[0]]),axis=0)
            saved[p+'__quality_range_0125'] = np.ptp(np.stack([saved[r['array_prefix']+'__'+axis].real for r in sequences[1]]),axis=0)
            curves.append(dict(cut=cut,axis=axis,array_prefix=p))
    # ONE envelope over the entire declared curve family, not a selected peak.
    standard = []
    for row in curves:
        p=row['array_prefix']; se=saved[p+'__SE']; active=se>1e-11
        noise=saved.pop(p+'__multiplier_noises')
        standard.append(np.max(abs(noise[:,active]/se[active]),axis=1) if np.any(active)
                        else np.zeros(config['multiplier_replicates']))
    critical=float(np.quantile(np.max(standard,axis=0),.95))
    for row in curves:
        p=row['array_prefix']
        threshold=saved[p+'__quality_range_0']+saved[p+'__quality_range_0125']+critical*saved[p+'__SE']
        mask=abs(saved[p+'__difference'].real)>threshold
        saved[p+'__exceeds_simultaneous_empirical_envelope']=mask
        row.update(simultaneous_95_critical=critical,grid_points_exceeding=int(mask.sum()),
            includes_all_axes_angles_geometries=True,equilibrium_certified=False,
            multiplier_approximation=True,quality_ranges_are_certified_bias_bounds=False)
    np.savez_compressed(output/'contrasts.npz',**saved)
    result=dict(records=records,contrasts=contrasts,curve_family=curves,
        diagnostics=maxima,operator_split_diagnostics=split.diagnostics,
        seconds=perf_counter()-started,raw_sha256=file_sha256(output/'contrasts.npz'),
        new_training=0,new_samples=0,new_amplitude_queries=0,new_H_calls=0,
        phase_claim=False,QSL_specific_signal=False,cusp_fit=False,log_floor=False,
        warning='Retrospective fixed-geometry comparison. Scalar rotation is p(q) information; '
                'full/high joint projector comparison is not a QSL witness. Quality ranges are '
                'empirical, not bias bounds. Whole-chain jackknife and normal multipliers do not '
                'certify mixing. Different coupling states are not different topological sectors.')
    write('results.json',result)
    write('completion.json',dict(registration_sha256=file_sha256(output/'registration.json'),
        results_sha256=file_sha256(output/'results.json')))
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    args=parser.parse_args()
    report=analyze(args.config)
    print(json.dumps(dict(seconds=report['seconds'],diagnostics=report['diagnostics'],
        contrasts=report['contrasts'],phase_claim=False),indent=2))


if __name__=='__main__':
    main()
