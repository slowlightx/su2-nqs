"""Replay and plot frozen scalar-channel readouts and their quality drift.

No detector fitting, state optimization, RDM repair, or additional sampling.
Spatial points are oriented translations on one finite torus, not a length fit.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from .frozen_operator_readout import summarize
from .spin_spatial_vit_quality import ROOT, read_sealed

BASE = ROOT / 'projects/clebsch_tree_nqs/results/remote_readout_review_20261013'
PHYSICS = ROOT / 'projects/clebsch_tree_nqs/results/su2_structure_mechanism_20261013'
FIGURES = ROOT / 'projects/clebsch_tree_nqs/figures/exploration'
KEYS = ('j0_s750', 'j0_s1000', 'j0125_s100', 'j0125_s200')
OPERATORS = ('gme', 'transfer', 'single_change', 'center_transfer')
SHIFTS = ((2, 0), (0, 3), (3, 0), (3, 3))
LABELS = dict(gme='Frozen GME score (2+1+2 parties)', transfer='Singlet/triplet transfer',
              single_change='Single-pair charge change', center_transfer='Center-correlated transfer R')


def checked_results(directory: Path) -> tuple[dict, float]:
    """Check seals and recompute every chain mean and jackknife from raw moments."""
    result = read_sealed(directory / 'results.json')
    if result['status'] != 'completed':
        raise ValueError('only completed immutable outputs can be analyzed')
    discrepancy = 0.0
    for phase in ('local', *result['distances']):
        record = result['local'] if phase == 'local' else result['distances'][phase]
        p = directory / phase / 'moments.npz'
        if file_sha256(p) != record['raw_sha256']:
            raise ValueError('raw moments changed')
        with np.load(p) as z:
            replay = summarize({k: z[k] for k in z.files if k != 'chain_ids'}, z['chain_ids'])
        for k, value in replay.items():
            for field in ('value', 'chain_SE'):
                discrepancy = max(discrepancy, abs(value[field]-record['moments'][k][field]))
    if discrepancy > 1e-12:
        raise ValueError('saved moment summary disagrees with independent replay')
    return result, discrepancy


def contrast(values: list[dict], family_size: int) -> dict:
    """Descriptive coupling contrast against two observed endpoint drifts.

    Chains from different endpoint audits are independent. Observed optimization
    drift is not a bound on variational bias. Normal errors are descriptive.
    """
    early0, late0, early1, late1 = values
    delta = late1['value']-late0['value']
    se = np.hypot(late0['chain_SE'], late1['chain_SE'])
    drift = abs(late0['value']-early0['value'])+abs(late1['value']-early1['value'])
    z = float(norm.ppf(1-0.05/(2*family_size)))
    return dict(delta_J0125_minus_J0=delta, independent_SE=float(se),
                own_endpoint_drift_sum=drift, family_size=family_size, critical_z=z,
                exceeds_drift_plus_family_error=bool(abs(delta)>drift+z*se),
                formal_phase_test=False)


def support_split_analysis() -> dict:
    """Resolve the same fixed scalar correlator by Pauli support, with paired errors.

    The displacement was selected in the first exploratory scan. This follow-up
    diagnoses its mechanism; it is not an independent discovery dataset.
    """
    records = {}; replays = {}; inputs = {}
    for key in KEYS:
        directory = BASE / (key+'_single_support_v1')
        result, replays[key] = checked_results(directory)
        inputs[str(directory.relative_to(ROOT) / 'results.json')] = file_sha256(directory / 'results.json')
        with np.load(directory/'dr2_dc0/moments.npz') as z:
            ids = np.unique(z['chain_ids'])
            means = {k: np.array([z[k][z['chain_ids']==c].mean() for c in ids])
                     for k in z.files if k != 'chain_ids'}
        n = len(ids); jack = {}; plugin = {}; components = {}
        for short, op in (('total','single_total'), ('two_spin','single_two_spin'), ('four_spin','single_four_spin')):
            a, b, ab = (means[op+'__'+s] for s in ('A', 'B', 'AB'))
            ja, jb, jab = ((n*v.mean()-v)/(n-1) for v in (a, b, ab))
            jack[short] = jab-ja*jb
            plugin[short] = ab.mean()-a.mean()*b.mean()
        jack['mixed'] = jack['total']-jack['two_spin']-jack['four_spin']
        plugin['mixed'] = plugin['total']-plugin['two_spin']-plugin['four_spin']
        for short in jack:
            components[short] = dict(value=float(n*plugin[short]-(n-1)*jack[short].mean()),
                chain_SE=float(np.sqrt((n-1)*np.mean((jack[short]-jack[short].mean())**2))))
        original = read_sealed(BASE/(key+'_distance_quality_v1')/'results.json')
        expected = original['distances']['dr2_dc0']['moments']['single_change__connected']['value']
        if abs(components['total']['value']-expected) > 1e-12:
            raise ValueError('support split changed the original frozen correlator')
        records[key] = dict(components=components, original_total_error=abs(components['total']['value']-expected),
                            seconds=result['seconds'])
    # Original 16 spatial contrasts plus three algebraic contributions;
    # conservative descriptive errors do not undo retrospective selection.
    comparisons = {name: contrast([records[k]['components'][name] for k in KEYS], 19)
                   for name in ('two_spin','four_spin','mixed')}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), constrained_layout=True)
    for ax, name, title in zip(axes, ('two_spin','four_spin','mixed'),
            ('Two-spin x two-spin', 'Four-spin x four-spin', 'Mixed two-spin x four-spin')):
        for i, pair in enumerate((KEYS[:2], KEYS[2:])):
            vals = [records[k]['components'][name] for k in pair]
            x = np.array([i-.06, i+.06]); color = ('#1677a3','#c34d29')[i]
            ax.plot(x, [v['value'] for v in vals], color=color, lw=.8)
            for j, v in enumerate(vals):
                ax.errorbar(x[j], v['value'], yerr=v['chain_SE'], fmt='o', color=color,
                            mfc='white' if j == 0 else color, capsize=3,
                            label=('Earlier' if j == 0 else 'Latest') if i == 0 else None)
        ax.axhline(0, color='gray', lw=.7)
        ax.set(xticks=[0,1], xticklabels=['J2 = 0','J2 = 0.125'], title=title, xlim=(-.3,1.3))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axes[0].legend(fontsize=8)
    fig.suptitle('Algebraic origin of the fixed-channel correlation at (2,0)\nSame parent states and records; mixed term includes both orderings')
    for ext in ('png','pdf'):
        fig.savefig(FIGURES / ('frozen_su2_support_split_20261013.'+ext), dpi=180)
    plt.close(fig)
    output = dict(records=records, comparisons=comparisons, input_sha256=inputs,
                  raw_moment_replay_max=replays, same_records_as_initial_scan=True,
                  exploratory_selected_displacement=True, independent_confirmation=False)
    atomic_json(BASE/'support_split_analysis.json', output)
    return output


def run() -> dict:
    results = {}; checks = {}; inputs = {}
    for key in KEYS:
        path = BASE / (key+'_distance_quality_v1')
        results[key], checks[key] = checked_results(path)
        inputs[str(path.relative_to(ROOT) / 'results.json')] = file_sha256(path / 'results.json')
    control_path = PHYSICS / 'dimer_distance_controls.json'
    controls = json.loads(control_path.read_text())
    inputs[str(control_path.relative_to(ROOT))] = file_sha256(control_path)
    dmrg = json.loads((BASE / 'optimization_review.json').read_text())['dmrg_latest_complete_sweep']
    local = {op: contrast([results[k]['local']['moments'][op+'__A'] for k in KEYS], 4)
             for op in OPERATORS}
    spatial = []
    for op in OPERATORS:
        for shift in SHIFTS:
            tag = 'dr%d_dc%d' % shift
            values = [results[k]['distances'][tag]['moments'][op+'__connected'] for k in KEYS]
            spatial.append(dict(operator=op, shift=list(shift), endpoint_values=dict(zip(KEYS, values)),
                                contrast=contrast(values, len(OPERATORS)*len(SHIFTS))))
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), constrained_layout=True)
    colors = ('#1677a3', '#c34d29')
    for ax, op in zip(axes.flat, OPERATORS):
        for i, pair in enumerate((KEYS[:2], KEYS[2:])):
            vals = [results[k]['local']['moments'][op+'__A'] for k in pair]
            x = np.array([i-.07, i+.07])
            ax.plot(x, [v['value'] for v in vals], color=colors[i], alpha=.65)
            for j, v in enumerate(vals):
                ax.errorbar(x[j], v['value'], yerr=v['chain_SE'], fmt='o', color=colors[i],
                            mfc='white' if j == 0 else colors[i], capsize=3,
                            label=('Earlier endpoint' if j == 0 else 'Latest endpoint') if i == 0 else None)
        ax.plot(1.22, dmrg['frozen_upper_scores'][op], '*', color='black', ms=10,
                label='DMRG 10240 / sweep 33')
        ax.set(xticks=[0, 1], xticklabels=['J2 = 0', 'J2 = 0.125'], title=LABELS[op], xlim=(-.4, 1.45))
        ax.grid(axis='y', alpha=.15)
        if op == 'gme':
            ax.axhline(0, color='gray', lw=.7); ax.legend(fontsize=8)
    fig.suptitle('Frozen operators: local SU(2) charge processes\n128 chains per endpoint; bars = whole-chain SE; lines = observed quality drift')
    for ext in ('png', 'pdf'):
        fig.savefig(FIGURES / ('frozen_su2_local_quality_20261013.'+ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    for ax, op in zip(axes, ('center_transfer', 'transfer', 'single_change')):
        for i, pair in enumerate((KEYS[:2], KEYS[2:])):
            for j, key in enumerate(pair):
                vals = [results[key]['distances']['dr%d_dc%d'%s]['moments'][op+'__connected'] for s in SHIFTS]
                ax.errorbar(np.arange(4)+(i-.5)*.08, [v['value'] for v in vals],
                            yerr=[v['chain_SE'] for v in vals], color=colors[i],
                            fmt='o--' if j == 0 else 'o-', mfc='white' if j == 0 else colors[i],
                            lw=.8 if j == 0 else 1.3, alpha=.45 if j == 0 else 1,
                            label=('J2 = '+('0' if i == 0 else '0.125')+(' earlier' if j == 0 else ' latest')))
        dimer = [next(r for r in controls['records'] if r['operator'] == op and r['shift'] == list(s))
                 ['space_group_mixture']['connected'] for s in SHIFTS]
        ax.plot(np.arange(4), dimer, 's:', color='gray', ms=4, label='Symmetric dimer mixture')
        ax.axhline(0, color='black', lw=.6)
        ax.set(xticks=range(4), xticklabels=[str(s) for s in SHIFTS], xlabel='Oriented translation (dr, dc)',
               title=LABELS[op], ylabel='Connected correlation')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0)); ax.grid(axis='y', alpha=.15)
    axes[0].legend(fontsize=7)
    fig.suptitle('Spatial organization on the 6 x 6 torus\n16384 cached records per point; genuine joint estimator; no correlation-length fit')
    for ext in ('png', 'pdf'):
        fig.savefig(FIGURES / ('frozen_su2_distance_quality_20261013.'+ext), dpi=180)
    plt.close(fig)
    out = dict(local_contrasts=local, spatial_contrasts=spatial, moment_replay_max=checks,
        input_sha256=inputs, source_sha256=file_sha256(Path(__file__)),
        limitations=['Retrospective operator selection on earlier states; exploratory comparisons.',
                     'Observed quality drift is not a rigorous state-error bound.',
                     'The three shifts of length three differ in orientation, not three radii.',
                     'Dimer-mixture covariance is classical; nonzero C alone is not a QSL witness.',
                     'Local center-transfer is in the two-bond operator span; its separated product is eight-spin.',
                     'Ordinary ViT is full A1 but not strict SU2; scalar expectations remain valid.'],
        seconds={k: results[k]['seconds'] for k in KEYS},
        projected_queries={k: results[k]['projected_amplitude_queries'] for k in KEYS},
        new_samples=0, new_training=0, new_H_calls=0)
    atomic_json(BASE / 'spatial_quality_analysis.json', out)
    if all((BASE/(k+'_single_support_v1')/'results.json').exists() for k in KEYS):
        out['support_split'] = support_split_analysis()
    print(json.dumps(dict(local=local, spatial_passes=[r for r in spatial if r['contrast']['exceeds_drift_plus_family_error']],
                         replay_max=max(checks.values())), indent=2))
    return out


if __name__ == '__main__':
    run()
