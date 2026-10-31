"""Frozen Schmidt spin textures and bounded higher-rotor screening.

Neither boundary polarization nor a many-candidate rotor fit assigns a phase.
Use the registered row-major physical geometry, verified spin content and
the same unchanged MPS parent. This workflow consumes cached results only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from su2_nqs.benchmarks.rotor_family_scan import scan_quadratic_rotor_families
from su2_nqs.benchmarks.mps_rotation_covariance import fixed_spin_casimir_difference
from .dmrg_archive import file_sha256
from .dmrg_region_probe import _atomic_json as atomic_json, _canonical


def boundary_triplet_control() -> dict:
    """Physical 18-spin S=1,M=1 vector: two boundary spins plus eight dimers.

    Bits use down=0/up=1, site0 MSB. S+ annihilation at M=1 proves total
    doubled q=2. This is a trivial boundary control, not a QSL wavefunction.
    """
    n = 18
    boundaries = (0, 12)
    remaining = [i for i in range(n) if i not in boundaries]
    pairs = list(zip(remaining[::2], remaining[1::2]))
    vector = np.zeros(2**n, dtype=float)
    for choices in range(2**len(pairs)):
        packed = sum(1 << (n-1-i) for i in boundaries)
        sign = 1
        for k, (i, j) in enumerate(pairs):
            bit = (choices >> k) & 1
            packed |= 1 << (n-1-(i if bit else j))
            sign *= 1 if bit else -1
        vector[packed] = sign/np.sqrt(2**len(pairs))
    positions = np.arange(2**n)
    probability = vector**2
    magnetizations = np.array([probability @ (((positions >> (n-1-i)) & 1)-.5)
                              for i in range(n)])
    raised = np.zeros_like(vector)
    for i in range(n):
        mask = 1 << (n-1-i)
        sources = positions[(positions & mask) == 0]
        raised[sources | mask] += vector[sources]
    return dict(n_region=n, q=2, m=2, row_spin_z=magnetizations.reshape(3, 6).sum(axis=1).tolist(),
        norm_error=float(abs(probability.sum()-1)), raising_norm=float(np.linalg.norm(raised)),
        ordinary_boundary_dimers=True, QSL=False,
        note='Unique m=+1 vector of two boundary spin halves; internal dimers may cross rows but have zero one-point moments.')


def run(inputs: list[Path], output: Path, reference: Path | None = None) -> dict:
    """Analyze all stated parents; full raw level windows remain available."""
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in inputs:
        worker = json.loads(path.read_text())
        if worker.get('record_sha256') != _canonical({k: v for k, v in worker.items() if k != 'record_sha256'}):
            raise ValueError('changed physical worker record')
        if worker['status'] != 'complete' or worker['parent_fingerprint_before'] != worker['parent_fingerprint_after']:
            raise ValueError('complete unchanged physical parent required')
        data = worker['readout']
        groups = ('boundary_row0', 'interior_row1', 'boundary_row2')
        if (set(data['spatial_spin_groups']) != set(groups) or not data['groups_partition_prefix']
                or data['cut'] != 18 or data['n_sites'] != 36
                or any(data['spatial_spin_groups'][name] != list(range(6*i, 6*(i+1)))
                       for i, name in enumerate(groups))):
            raise ValueError('registered three-row physical prefix required')
        sectors = data['leading_spin_content']['sectors']
        profiles = {}
        spin_energies = {}
        excluded = {}
        for m in ('0', '2', '4', '6'):
            levels = sectors[m]['leading_levels']
            # For m=q a near-minimal Casimir bounds *all* larger-spin admixture.
            # No generic q is assigned from a mean between two Casimir values.
            valid = [a for a in levels if a['higher_spin_weight_upper_bound_nominal'] < 1e-5]
            spin_energies[int(m)] = [a['entanglement_energy'] for a in valid]
            excluded[m] = len(levels)-len(valid)
            profile = np.asarray([
                [a['physical_observable_expectations'][name] for name in groups] for a in levels])
            closure = float(np.max(abs(profile.sum(axis=1)-int(m)/2)))
            if closure > 1e-8:
                raise ValueError('spatial spin profile failed total physical Sz closure')
            profiles[m] = dict(leading_row_spin_z=profile[0].tolist(),
                total_row_spin_z=float(profile[0].sum()), maximum_total_Sz_closure_error=closure,
                lowest_entanglement_energy=levels[0]['entanglement_energy'],
                lowest_Casimir=levels[0]['physical_casimir_expectation'],
                interior_antipolarization=float(-profile[0, 1]),
                boundary_spin_z_sum=float(profile[0, 0]+profile[0, 2]),
                all_levels=levels)
            if int(m) > 0 and levels[0]['higher_spin_weight_upper_bound_nominal'] < 1e-5:
                profiles[m]['angular_screening_interpretation'] = dict(
                    outer_minus_interior_Casimir_in_fixed_spin_limit=fixed_spin_casimir_difference(
                        int(m), int(m), float(profile[0, 0]+profile[0, 2])),
                    inferred_from_same_polarization_not_an_independent_measurement=True,
                    assumes_exact_dominant_spin=True,
                    nominal_higher_spin_weight=levels[0]['higher_spin_weight_upper_bound_nominal'])
        ground_anchor = {q: values.copy() for q, values in spin_energies.items()}
        ground_anchor[0] = ground_anchor[0][:1]
        records.append(dict(chi=worker['actual_chi'], input=str(path), input_sha256=file_sha256(path),
            checkpoint_sha256=worker['checkpoint_sha256'], profiles=profiles,
            dominant_spin_tolerance=1e-5, excluded_uncertain_levels=excluded,
            retained_spin_level_counts={q: len(a) for q, a in spin_energies.items()},
            ground_anchored_rotor_scan=scan_quadratic_rotor_families(ground_anchor),
            all_singlet_anchor_rotor_scan=scan_quadratic_rotor_families(spin_energies),
            load_seconds=worker['load_and_input_seal_seconds'],
            operator_seconds=worker['full_environment_and_operator_seconds'],
            worker_peak_RSS_bytes=worker['worker_peak_rss_bytes'],
            phase_claim=False))
    controls = dict(ideal_collective_120_singlet=dict(row_spin_z=[1/3]*3,
        explanation='Each row contains equal fractions of each permutation-symmetric maximal-spin sublattice; restricted row Sz equals total Sz/3.',
        uniform_not_a_theorem_for_all_magnetic_states=True),
        boundary_dimers=boundary_triplet_control())
    if reference is not None:
        magnetic = json.loads(reference.read_text())['reference']
        if not magnetic['strict_su2_verified']:
            raise ValueError('validated physical magnetic reference required')
        energies = {q: sorted(magnetic['q_sectors'][str(q)]['entanglement_energies'])
                    for q in (0, 2, 4, 6)}
        controls['magnetic_reference_same_rotor_screen'] = scan_quadratic_rotor_families(energies)
        controls['magnetic_reference_input'] = str(reference)
        controls['magnetic_reference_sha256'] = file_sha256(reference)
    result = dict(records=records, controls=controls, region='three six-site rows, 18-site half torus',
        analysis_source_sha256=file_sha256(Path(__file__)),
        rotor_scan_source_sha256=file_sha256(Path(__file__).resolve().parents[3]/'su2_nqs/benchmarks/rotor_family_scan.py'),
        no_new_states=True, no_training=True, no_new_samples=True,
        spatial_profile_not_qsl_specific=True, thermodynamic_phase_assigned=False,
        interpretation=('Boundary-enhanced polarization with an antipolarized interior is an extended screening texture, not a uniform ideal collective mode or two bare free boundary spins.'
            if all(r['profiles']['2']['leading_row_spin_z'][1] < 0 and
                   min(r['profiles']['2']['leading_row_spin_z'][0], r['profiles']['2']['leading_row_spin_z'][2]) > 0
                   for r in records) else 'Measured spatial spin texture; no automatic collective-versus-boundary classifier.'),
        rotor_warning='Dense weak-spectrum matches after many fits are not an isolated TOS. Inspect S3 prediction, level separation, symmetry action and cross-geometry controls.',
        rotor_prediction_error_not_a_phase_threshold=True,
        finite_size_scaling_not_numerical_precision=True)
    atomic_json(output/'spatial_analysis.json', result)
    _plot(result, output)
    return result


def _plot(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
    rows = np.arange(3)
    for record in result['records']:
        p = record['profiles']['2']['leading_row_spin_z']
        axes[0].plot(rows, p, 'o-', label=f"Actual chi={record['chi']}")
    axes[0].plot(rows, [1/3]*3, 's--', color='#777777', label='Ideal collective singlet')
    axes[0].plot(rows, [.5, 0, .5], 'D:', color='#c25b27', label='Ordinary boundary-dimer control')
    axes[0].axhline(0, color='#555555', lw=.7)
    axes[0].set(xticks=rows, xticklabels=['Boundary row 0', 'Interior row 1', 'Boundary row 2'],
        ylabel=r'$\langle S^z_{\rm row}\rangle$ in lowest $S_A\simeq1,m_A=2$ vector',
        title='Actual soft triplet spin texture')
    axes[0].legend(fontsize=8)
    record = result['records'][-1]
    levels = record['profiles']
    baseline = levels['0']['lowest_entanglement_energy']
    for m in ('0', '2', '4', '6'):
        e = [a['entanglement_energy']-baseline for a in levels[m]['all_levels']
             if a['higher_spin_weight_upper_bound_nominal'] < 1e-5]
        axes[1].plot([int(m)/2]*len(e), e, '.', color='#2878b5', ms=4)
    best = record['ground_anchored_rotor_scan'].get('candidates', [])
    if best:
        candidate = best[0]
        for q, key in ((2, 'spin1_observed'), (4, 'spin2_prediction'), (6, 'spin3_prediction')):
            values = np.asarray(candidate[key])-baseline
            axes[1].plot([q/2+.1]*len(values), values, 'x', color='#c25b27', ms=6,
                label='Best ground-anchor prediction (not proof)' if q == 2 else None)
    axes[1].set(xlabel=r'Dominant physical $S_A$ from minimal Casimir', ylabel=r'$\xi-\xi_0$',
        title=f"Higher-family screen, chi={record['chi']}", ylim=(-.15, 12))
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.15)
    for suffix in ('png', 'pdf'): fig.savefig(output/f'schmidt_spin_texture.{suffix}', dpi=180)
    plt.close(fig)


def screening_summary(spatial_input: Path, output: Path) -> dict:
    """Explain cached low/high modes with SU2 block-coupling bounds.

    No new states/operators/samples. Revalidate original worker records rather
    than treating edited derived figures as numerical inputs. Bounds do NOT
    measure the full joint-spin distribution or identify a QSL.
    """
    from su2_nqs.benchmarks.spin_screening import screening_weight_bound
    saved = json.loads(spatial_input.read_text())
    records = []
    for item in saved['records']:
        path = Path(item['input'])
        if file_sha256(path) != item['input_sha256']:
            raise ValueError('cached physical input changed')
        worker = json.loads(path.read_text())
        if worker['record_sha256'] != _canonical({k:v for k,v in worker.items() if k != 'record_sha256'}):
            raise ValueError('physical worker seal changed')
        if worker['status'] != 'complete' or worker['parent_fingerprint_before'] != worker['parent_fingerprint_after']:
            raise ValueError('unchanged complete physical parent required')
        data = worker['readout']
        groups = ('boundary_row0','interior_row1','boundary_row2')
        if (data['cut'] != 18 or data['n_sites'] != 36 or not data['groups_partition_prefix']
                or set(data['spatial_spin_groups']) != set(groups)
                or any(data['spatial_spin_groups'][name] != list(range(6*i,6*(i+1)))
                       for i,name in enumerate(groups))):
            raise ValueError('same18-site prefix geometry required')
        baseline = item['profiles']['0']['lowest_entanglement_energy']
        levels = []
        for charge in ('2','4','6'):
            raw = data['leading_spin_content']['sectors'][charge]['leading_levels']
            if raw != item['profiles'][charge]['all_levels']:
                raise ValueError('derived levels disagree with physical worker')
            for index, level in enumerate(raw):
                if level['higher_spin_weight_upper_bound_nominal'] >= 1e-5: continue
                mid = level['physical_observable_expectations']['interior_row1']
                bound = screening_weight_bound(n_left=12,n_right=6,
                    q_total=int(charge),m2=int(charge),right_spin_z=mid,
                    casimir=level['physical_casimir_expectation'])
                levels.append(dict(q=int(charge),m2=int(charge),index_in_charge_block=index,
                    entanglement_energy=level['entanglement_energy'],
                    energy_above_first_singlet=level['entanglement_energy']-baseline,
                    interior_spin_z=mid, interior_fraction_of_total_M=mid/(int(charge)/2),
                    bound=bound))
        records.append(dict(chi=item['chi'],worker_sha256=item['input_sha256'],levels=levels,
            lowest_triplet=next(v for v in levels if v['q']==2 and v['index_in_charge_block']==0)))
    result = dict(records=records,source_spatial_sha256=file_sha256(spatial_input),
        explanation='Negative interior spin in the totalS1 mode requires So=Si+1 with Si>0, not an inert middle-spin block.',
        no_new_states=True,no_new_samples=True,no_new_H_calls=True,
        independent_new_observable=False,qsl_specific=False,
        ordinary_boundary_control=dict(interior_spin_z=0.,negative_channel_weight=5/12,
            active_block_spin_weights={'So1_Si0':1/4,'So0_Si1':1/12,'So1_Si1':1/4,'So2_Si1':5/12},
            note='Two boundary ups + two crossed dimers. Negative channel weight is ALREADY nonzero; positive/negative contributions cancel. Other dimers internal to groups are spin0.'),
        caveats=['Input Casimir precision is declared, not a certified floating-point interval.',
            'Higher near-degenerate Schmidt vectors can have basis-dependent individual profiles; use subspace averages before family claims.',
            'Same-cut actualJ0 and full joint-spin weights remain needed. A bound is not the measured distribution.',
            'These are entanglement modes, not physical excitation energies.'])
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'screening_channel_bounds.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot([r['chi'] for r in records],
        [r['lowest_triplet']['bound']['precision_adjusted_negative_channel_weight_lower_bound'] for r in records],'o-')
    axes[0].set(xlabel='DMRG chi',ylabel='Derived negative-channel weight lower bound',
        title='Lowest S=1: spinful interior required',ylim=(0,.25))
    latest=records[-1]
    for q in (2,4,6):
        group=[v for v in latest['levels'] if v['q']==q]
        axes[1].plot([v['energy_above_first_singlet'] for v in group],
            [v['interior_fraction_of_total_M'] for v in group],'.',label=f'S={q//2}')
    axes[1].axhline(0,color='#777777',lw=.8)
    axes[1].axhline(1/3,color='#777777',lw=.8,ls='--',label='Ideal collective control')
    axes[1].set(xlabel='Entanglement energy above first singlet',ylabel='Interior Sz / total M',
        title='Higher pure-spin textures\n(not certified families)')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
        ax.tick_params(labelsize=9)
        ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_size(10)
        ax.title.set_size(11)
    for ext in ('png','pdf'):
        fig.savefig(output/f'screening_channel_bounds.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+')
    parser.add_argument('--screening-input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    if args.screening_input is not None:
        if args.input is not None or args.reference is not None:
            parser.error('cached screening summary is separate from the original worker analysis')
        report=screening_summary(args.screening_input,args.output)
        print(json.dumps([dict(chi=r['chi'],negative_channel_lower_bound=
            r['lowest_triplet']['bound']['precision_adjusted_negative_channel_weight_lower_bound']) for r in report['records']]))
    elif args.input is not None:
        report = run(args.input, args.output, args.reference)
        print(json.dumps([dict(chi=r['chi'], triplet_rows=r['profiles']['2']['leading_row_spin_z'])
                          for r in report['records']]))
    else:
        parser.error('physical worker --input or cached --screening-input required')
