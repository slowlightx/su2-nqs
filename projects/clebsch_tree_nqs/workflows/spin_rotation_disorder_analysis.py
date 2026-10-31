"""Zero-query, frozen-input partial-rotation falsification and quality checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.spin_rotation_disorder import (
    regional_spin_rotation, sampled_spin_rotation, sampled_translation_rotation,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.lattices.triangular import triangular_bonds

ROOT = Path(__file__).resolve().parents[3]


def analyze(config_path: Path) -> dict:
    """No training, H, amplitude queries, new samples, twirl, or cusp fitting."""
    settings = json.loads(config_path.read_text())
    inherited = {}
    if settings.get('base_config'):
        base = ROOT/settings['base_config']
        digest = file_sha256(base)
        if digest != settings['base_config_sha256']:
            raise ValueError('base configuration seal changed')
        inherited[str(base.relative_to(ROOT))] = digest
        settings = {**json.loads(base.read_text()), **settings}
    output = ROOT/settings['output']
    if output.exists():
        raise FileExistsError('completed analysis is immutable; use a new output')
    started = perf_counter()
    inputs, records, saved = inherited, [], {}
    angles = np.linspace(0, 2*np.pi, settings['angle_points'])
    pi = (len(angles)-1)//2
    if len(angles) % 2 != 1:
        raise ValueError('odd angle grid required to include pi exactly')
    def checked(name, expected=None):
        path = ROOT/name
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError('input seal changed: '+str(path))
        inputs[str(path.relative_to(ROOT))] = digest
        return path
    def sealed(name, expected=None):
        path = checked(name, expected)
        record = json.loads(path.read_text())
        if record.get('result_sha256') != canonical_sha256({k:v for k,v in record.items() if k != 'result_sha256'}):
            raise ValueError('JSON content seal changed: '+str(path))
        return record
    def measure(row, rho, jackknife=None):
        if perf_counter()-started > settings['seconds_cap']:
            raise TimeoutError('registered offline analysis cap')
        index = len(records)
        report = regional_spin_rotation(rho, angles)
        details = {}
        for axis in ('z', 'x', 'scalar'):
            curve = report[axis]
            saved[f'r{index}__{axis}'] = curve
            details[axis] = dict(pi_real=float(curve[pi].real), pi_imaginary=float(curve[pi].imag))
        if jackknife is not None:
            estimates = [regional_spin_rotation(r, angles) for r in jackknife]
            for axis in ('z', 'x', 'scalar'):
                a = np.array([r[axis] for r in estimates])
                centered = a-a.mean(0)
                scale = (len(a)-1)/len(a)
                re = np.sqrt(scale*np.sum(centered.real**2, 0))
                im = np.sqrt(scale*np.sum(centered.imag**2, 0))
                saved[f'r{index}__{axis}_real_SE'], saved[f'r{index}__{axis}_imag_SE'] = re, im
                details[axis].update(pi_real_whole_chain_SE=float(re[pi]),
                                    pi_imag_whole_chain_SE=float(im[pi]))
        records.append(dict(**row, array_prefix=f'r{index}', axes=details, p_q=report['p_q'],
            factorization_frobenius=report['factorization_frobenius'],
            axis_scalar_max_difference=report['axis_scalar_max_difference'],
            center_identity_error=float(max(abs(report[k][-1]-(-1)**int(np.log2(len(rho)))) for k in ('x','z','scalar'))),
            strict_SU2_spectrum=False, scalar_is_rotation_averaged_observable=True,
            QSL_claim=False))
    # This older producer has a byte-level seal but no internal canonical seal.
    # Do not rewrite it to make it look like a newer producer.
    series = json.loads(checked(settings['series_results'], settings['series_results_sha256']).read_text())
    arrays = checked(settings['series_arrays'], settings['series_arrays_sha256'])
    cache = {}
    for entry in settings['jackknife_caches']:
        path = checked(entry['path'], entry['sha256'])
        with np.load(path) as archive:
            cache[entry['label']] = {k: np.array(archive[k]) for k in archive.files
                if k.endswith('__rho') or k.endswith('__jackknife_rhos')}
    with np.load(arrays) as archive:
        for row in series['records']:
            if row['problem']['geometry'] != 'triangular':
                continue
            prefix = row['label']+'__'+row['cut']
            rho = archive[prefix+'__rho']
            old = cache.get(row['label'], {})
            jack = old.get(row['cut']+'__jackknife_rhos')
            if jack is not None and np.linalg.norm(old[row['cut']+'__rho']-rho) > 1e-12:
                raise ValueError('jackknife belongs to another mean RDM')
            measure({k:row[k] for k in ('label','method','cut','sites','energy')}
                | dict(j2=row['problem']['j2'], chi=row.get('chi'), step=row.get('step'),
                       sweep=row.get('sweep')), rho, jack)
    # Diagonal nonlocal observable on existing full-H audit records.
    sampled, averaged = [], []
    for entry in settings['born_audits']:
        audit = sealed(entry['directory']+'/results.json', entry['audit_sha256'])
        if settings.get('translation_average', False):
            proof = settings['physical_configs'][entry['label']]
            config = json.loads(checked(proof['path'],proof['sha256']).read_text())
            if (audit['audit_projection'] != 'gamma_a1' or not config['spin_inversion_even']
                    or config['model']['rows'] != 6 or config['model']['columns'] != 6
                    or audit['sampled_A1_relative_error'] > 1e-10):
                raise ValueError('translation/spin-inversion state-symmetry prerequisite failed')
        all_states, all_chains = [], []
        chain_offset = 0
        for stream in audit['streams']:
            seed = stream['seed']
            raw = checked(entry['directory']+f'/stream{seed}.npz', stream['raw_sha256'])
            proof = sealed(entry['directory']+f'/stream{seed}.json')
            if (proof['params_sha256'] != audit['params_sha256'] or
                    proof['registration_sha256'] != audit['registration_sha256'] or
                    proof['raw_sha256'] != stream['raw_sha256']):
                raise ValueError('Born record provenance differs from frozen audit')
            with np.load(raw) as archive:
                x = archive['states']
                if x.ndim != 3 or x.shape[2] != 36 or not np.all(x.sum(2) == 18):
                    raise ValueError('not the registered N36 Sz0 records')
                all_states.append(x.reshape(-1,36))
                all_chains.append(np.tile(np.arange(x.shape[1])+chain_offset, x.shape[0]))
                chain_offset += x.shape[1]
        bits, chains = np.concatenate(all_states), np.concatenate(all_chains)
        if len(bits) != audit['energy']['samples'] or len(np.unique(chains)) != audit['energy']['independent_chains']:
            raise ValueError('audit group counts differ')
        for region in settings['regions']:
            report = sampled_spin_rotation(bits, chains, region['sites'], angles)
            complement = [i for i in range(36) if i not in region['sites']]
            other = sampled_spin_rotation(bits, chains, complement, -angles)
            error = float(np.max(abs(report['chain_means']-other['chain_means'])))
            if error > 1e-12:
                raise AssertionError('fixed total M=0 complement identity failed')
            prefix = 'b'+str(len(sampled))
            for k in ('value','chain_means','real_chain_SE','imaginary_chain_SE'):
                saved[prefix+'__'+k] = report[k]
            # A pointwise descriptive mask, not a certified confidence band.
            unresolved = abs(report['value']) <= 3*np.hypot(report['real_chain_SE'],report['imaginary_chain_SE'])
            saved[prefix+'__unresolved_near_zero'] = unresolved
            members = set(region['sites'])
            geometry = dict(region, coordinates_row_column=[[i//6,i%6] for i in region['sites']],
                nn_cut_bonds=sum((a in members) != (b in members)
                                 for a,b in triangular_bonds(6,6,periodic=True)),
                boundary_definition='number of crossing J1 bonds; not a fit variable',
                system_rows=6,system_columns=6,aspect_and_cut_bonds_not_matched=True)
            sampled.append(dict(label=entry['label'], region=geometry, array_prefix=prefix,
                energy=audit['energy'], pi_real=float(report['value'][pi].real),
                pi_imaginary=float(report['value'][pi].imag),
                pi_real_chain_SE=float(report['real_chain_SE'][pi]),
                pi_imaginary_chain_SE=float(report['imaginary_chain_SE'][pi]),
                pi_unresolved_near_zero=bool(unresolved[pi]),
                complement_identity_max_error=error, samples=len(bits), chains=len(np.unique(chains)),
                phase_information_accessed=False, new_H_calls=0, new_samples=0, QSL_claim=False))
            if settings.get('translation_average', False):
                orbit = sampled_translation_rotation(bits,chains,region['sites'],angles,
                    rows=6,columns=6,spin_inversion_average=True)
                orbit_prefix = 'o'+str(len(averaged))
                for k in ('value','chain_means','real_chain_SE','imaginary_chain_SE',
                          'paired_difference','paired_real_chain_SE','paired_imaginary_chain_SE'):
                    saved[orbit_prefix+'__'+k] = orbit[k]
                orbit_zero = abs(orbit['value']) <= 3*np.hypot(orbit['real_chain_SE'],orbit['imaginary_chain_SE'])
                saved[orbit_prefix+'__unresolved_near_zero'] = orbit_zero
                averaged.append(dict(label=entry['label'],region=geometry,array_prefix=orbit_prefix,
                    direct_array_prefix=prefix,pi_real=float(orbit['value'][pi].real),
                    pi_real_chain_SE=float(orbit['real_chain_SE'][pi]),
                    pi_unresolved_near_zero=bool(orbit_zero[pi]),
                    pi_paired_difference=float(orbit['paired_difference'][pi].real),
                    pi_paired_difference_chain_SE=float(orbit['paired_real_chain_SE'][pi]),
                    records=orbit['records'],chains=orbit['chains'],translation_images=36,
                    images_are_independent=False,spin_inversion_average=True,
                    same_physical_expectation_by_exact_state_symmetry=True,
                    phase_information_accessed=False,new_H_calls=0,new_samples=0,QSL_claim=False))
    controls = []
    for n_cross in (0,2,4,6):
        curve = np.cos(angles/2)**n_cross
        saved[f'control_dimer_{n_cross}'] = curve
        controls.append(dict(label=f'trivial_product_dimers_{n_cross}_cut_bonds',
            pi_value=float(curve[pi]), formula='cos(theta/2)^n_cross', topological=False,
            perimeter_law_is_not_QSL_proof=True))
    # Existing exact dimer, short-loop and collective-order controls, not new
    # optimized states, calibrate rotational spectra without altering them.
    if settings.get('control_arrays'):
        path = checked(settings['control_arrays'], settings['control_arrays_sha256'])
        with np.load(path) as archive:
            for entry in settings['control_keys']:
                measure(dict(label=entry['label'],method='exact_non_QSL_control',cut=entry['key'],
                    sites=list(range(6)),energy=None,j2=None,chi=None,step=None,sweep=None),archive[entry['key']])
    sources = transitive_local_source_manifest([Path(__file__)], root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    checked(config_path.resolve())
    output.mkdir(parents=True)
    saved['angles'] = angles
    np.savez_compressed(output/'curves.npz', **saved)
    def write(name, payload):
        atomic_json(output/name, {**payload,'result_sha256':canonical_sha256(payload)})
    write('registration.json',dict(config=settings,inputs_sha256=inputs,sources=sources))
    result = dict(records=records,born_records=sampled,translation_averaged_records=averaged,controls=controls,
        seconds=perf_counter()-started,raw_sha256=file_sha256(output/'curves.npz'),
        new_H_calls=0,new_samples=0,new_amplitude_queries=0,new_training=0,
        angle_points=len(angles),log_floor=False,cusp_fit=False,phase_claim=False,
        warning='Finite-region spin characteristic, not QSL certification. Scalar character '
                'does not access multiplicity coherence. No large-boundary/system-size '
                'limit or emergent boundary algebra is measured. A4..16 on one torus '
                'is a region-size feasibility probe, not thermodynamic scaling. '
                'Whole-chain uncertainty is descriptive; mixing and nonlinear log bias '
                'are not certified. Signed/complex Z retained; no -log abs(Z) fitted.')
    write('results.json',result)
    write('completion.json',dict(results_sha256=file_sha256(output/'results.json'),
        registration_sha256=file_sha256(output/'registration.json')))
    if settings.get('render_figure', False):
        render_figure(result, saved, output)
        if averaged:
            render_translation_figure(result,saved,output)
    return result


def render_figure(result: dict, saved: dict, output: Path) -> None:
    """Exploration only: signed moments and raw errors, never a fitted cusp."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    angles = saved['angles']/np.pi
    fig, axes = plt.subplots(2,2,figsize=(11.5,8),layout='constrained')
    ax = axes[0,0]
    for label, axis, style, color in [
        ('triangular_0.125_10600','z','-','#0072B2'),
        ('triangular_0.125_10600','x','--','#009E73'),
        ('triangular_0.125_10600','scalar',':','#D55E00'),
        ('DMRG_10240_32','z','-','black')]:
        row = next(r for r in result['records'] if r['label']==label and r['cut']=='compact6')
        prefix = row['array_prefix']+'__'+axis
        title = 'ViT '+axis if label.startswith('triangular') else 'DMRG z'
        y = saved[prefix].real
        ax.plot(angles,y,style,color=color,label=title,lw=1.6)
        if prefix+'_real_SE' in saved:
            error = saved[prefix+'_real_SE']
            ax.fill_between(angles,y-error,y+error,color=color,alpha=.08)
    ax.set(title='(a) Actual axes vs scalar character: compact6',ylabel=r'Re $Z_A(\theta)$')
    ax.legend(fontsize=8,ncol=2)
    ax = axes[0,1]
    for row in result['born_records']:
        if row['label']!='tri125_d2e3_100':
            continue
        ax.plot(angles,saved[row['array_prefix']+'__value'].real,label=row['region']['name'])
    ax.set(title='(b) Existing full-A1 ViT Born records',ylabel=r'Re $Z_A^z(\theta)$')
    ax.legend(fontsize=8,ncol=2)
    ax = axes[1,0]
    for label,color,offset in [('tri125_d2e3_100','#0072B2',-.13),
                               ('tri125_d2e4_100','#D55E00',.13)]:
        rows = [r for r in result['born_records'] if r['label']==label]
        ax.errorbar([len(r['region']['sites'])+offset for r in rows],
                    [r['pi_real'] for r in rows],yerr=[r['pi_real_chain_SE'] for r in rows],
                    fmt='o-',capsize=3,color=color,label=label.replace('tri125_',''))
    ax.axhline(0,color='grey',lw=.8)
    ax.set(title='(c) Finite-region feasibility, not size scaling',xlabel=r'Region spins $|A|$',
           ylabel=r'Re $Z_A^z(\pi)$, whole-chain SE')
    ax.legend(fontsize=8)
    ax = axes[1,1]
    for n in (2,4,6):
        ax.plot(angles,saved[f'control_dimer_{n}'],label=f'{n} crossing dimers')
    ax.set(title='(d) Trivial singlet dimers also have pi zeros',ylabel=r'$\cos^{n_{cut}}(\theta/2)$')
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        if ax is not axes[1,0]:
            ax.set_xlabel(r'$\theta/\pi$')
            ax.set_xlim(0,2)
    fig.suptitle('Triangular 6 x 6, J2/J1=0.125: partial spin rotation checks\n'
                 'No new samples or H calls; no cusp fit or QSL identification',fontsize=12)
    for extension in ('png','pdf'):
        fig.savefig(output/f'partial_rotation_checks.{extension}',dpi=170)
    plt.close(fig)


def render_translation_figure(result: dict, saved: dict, output: Path) -> None:
    """Compare equal-expectation estimators without treating images as samples."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
    for label,color in [('tri125_d2e3_100','#0072B2'),('tri125_d2e4_100','#D55E00')]:
        direct=[r for r in result['born_records'] if r['label']==label]
        orbit=[r for r in result['translation_averaged_records'] if r['label']==label]
        x=[len(r['region']['sites']) for r in orbit]
        axes[0].errorbar(x,[r['pi_real'] for r in direct],
            yerr=[r['pi_real_chain_SE'] for r in direct],fmt='o--',color=color,alpha=.45,capsize=2)
        axes[0].errorbar(x,[r['pi_real'] for r in orbit],
            yerr=[r['pi_real_chain_SE'] for r in orbit],fmt='s-',color=color,capsize=3,
            label=label.replace('tri125_','')+' translated')
        axes[1].plot(x,[a['pi_real_chain_SE']/b['pi_real_chain_SE']
                       for a,b in zip(direct,orbit)],'s-',color=color,label=label.replace('tri125_',''))
    axes[0].axhline(0,color='grey',lw=.8)
    axes[0].set(ylabel=r'Re $Z_A^z(\pi)$, whole-chain SE',title='Same state, direct vs symmetry average')
    axes[1].axhline(1,color='grey',lw=.8)
    axes[1].set(ylabel='Direct SE / translation-averaged SE',title='Empirical precision gain, not new samples')
    for ax in axes:
        ax.set_xlabel(r'Region spins $|A|$');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('36 correlated translations per old record; still128 independent chains\n'
                 'Zero new H/amplitude/sample queries; no logarithm, cusp or QSL claim',fontsize=11)
    for extension in ('png','pdf'):
        fig.savefig(output/f'translation_precision_checks.{extension}',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True)
    result = analyze(parser.parse_args().config)
    print(json.dumps(dict(records=len(result['records']),born_records=len(result['born_records']),
                         seconds=result['seconds'],phase_claim=False,new_H_calls=0)))
