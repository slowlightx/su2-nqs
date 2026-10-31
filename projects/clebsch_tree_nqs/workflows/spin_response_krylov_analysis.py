"""Independent imaginary-moment and chain stability audit of saved Ritz data.

No amplitude calls, new sampling, or re-selection of sources occurs here.
Bootstrap draws resample existing independent chain means only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from .spin_spatial_vit_quality import read_sealed


def _metrics(rows: np.ndarray) -> np.ndarray:
    """Rows are m0,m1,m2,m3,parent m0,parent m1; allow trailing replicas."""
    mean = rows[1]/rows[0]
    variance = rows[2]/rows[0]-mean**2
    diagonal = (rows[3]/rows[0]-2*mean*rows[2]/rows[0]+mean**3)/variance
    gap = np.sqrt((mean-diagonal)**2+4*variance)
    low = .5*(mean+diagonal-gap)
    high = .5*(mean+diagonal+gap)
    parent = rows[5]/rows[4]
    return np.array([mean-parent, variance, low-parent, high-parent,
                     (diagonal-low)/gap, (high-diagonal)/gap])


def run(root: Path, output: Path) -> dict:
    keys = ('j0_s1000', 'j0125_s200')
    field_names = ('centroid_from_parent', 'source_width_squared', 'lowest_ritz_from_parent',
                   'upper_ritz_from_parent', 'lowest_source_weight', 'upper_source_weight')
    rng = np.random.default_rng(141014)
    result = dict(scope='Existing frozen 512-record m3 data only; no new quantum samples or amplitudes',
        uncertainty='whole-chain jackknife and 2000 chain-bootstrap replicas, fixed seed 141014', rows={})
    for key in keys:
        directory = root/key
        sealed = read_sealed(directory/'results.json')
        path = directory/'moments.npz'
        if file_sha256(path) != sealed['raw_sha256']:
            raise ValueError('raw response moment data changed')
        with np.load(path) as archive:
            saved = {name: np.array(archive[name]) for name in archive.files}
        groups = saved['chain_ids']; ids = np.unique(groups); n = len(ids)
        means = {name: np.array([saved[name][groups == group].mean() for group in ids])
                 for name in saved if name not in ('chain_ids', 'selected_record_indices')}
        banks = dict(parent=('parent',), K=('K',), minusK=('minusK',), M1=('M1',),
                     M2=('M2',), M3=('M3',), Kstar=('K', 'minusK'), Mstar=('M1', 'M2', 'M3'))
        imaginary = {}
        for bank, names in banks.items():
            norm = np.mean([means[name+'__m0'] for name in names], axis=0)
            imaginary[bank] = {}
            for degree in (1, 3):
                row = np.mean([means[name+'__m'+str(degree)+'_imag'] for name in names], axis=0)
                value = float(row.mean()); se = float(row.std(ddof=1)/np.sqrt(n))
                ratio = value/norm.mean()
                deleted = ((n*row.mean()-row)/(n-1))/((n*norm.mean()-norm)/(n-1))
                ratio_se = float(np.sqrt((n-1)*np.mean((deleted-deleted.mean())**2)))
                imaginary[bank]['m'+str(degree)] = dict(unnormalized_value=value,chain_SE=se,
                    z_score=value/se if se else None, normalized_value=float(ratio),
                    normalized_chain_SE=ratio_se, normalized_z_score=float(ratio/ratio_se) if ratio_se else None)
        stars = {}; star_replicas = {}
        for bank, names in (('Kstar', ('K', 'minusK')), ('Mstar', ('M1', 'M2', 'M3'))):
            rows = np.array([np.mean([means[name+'__m'+str(r)] for name in names], axis=0)
                             for r in range(4)] + [means['parent__m0'], means['parent__m1']])
            average = rows.mean(1)
            full = _metrics(average)
            deleted = _metrics((n*average[:, None]-rows)/(n-1))
            star_replicas[bank] = (full, deleted)
            jack_se = np.sqrt((n-1)*np.mean((deleted-deleted.mean(1)[:, None])**2, axis=1))
            bootstrap_rows = rows[:, rng.integers(n, size=(2000,n))].mean(2)
            bootstrap = _metrics(bootstrap_rows)
            record = dict(metrics={},minimum_leave_one_chain_source_variance=float(deleted[1].min()),
                bootstrap_nonfinite_replicas=int(np.count_nonzero(~np.isfinite(bootstrap).all(0))),
                independent_seed_partitions={})
            for i, field in enumerate(field_names):
                reported = sealed['response_stars'][bank]['metrics'][field]
                if abs(full[i]-reported['value']) > 1e-12 or abs(jack_se[i]-reported['chain_SE']) > 1e-12:
                    raise ValueError('independent jackknife disagrees with saved result')
                record['metrics'][field] = dict(value=float(full[i]), chain_SE=float(jack_se[i]),
                    jackknife_bias_over_SE=float((n-1)*(deleted[i].mean()-full[i])/jack_se[i]),
                    largest_single_chain_removal_over_SE=float(np.max(abs(deleted[i]-full[i]))/jack_se[i]),
                    chain_bootstrap_SE=float(np.nanstd(bootstrap[i],ddof=1)),
                    chain_bootstrap_95percent=np.nanpercentile(bootstrap[i],[2.5,97.5]).tolist())
            for label,mask in (('original_stream0',ids<64),('original_stream1',ids>=64)):
                record['independent_seed_partitions'][label] = dict(zip(field_names,_metrics(rows[:,mask].mean(1)).tolist()))
            stars[bank] = record
        separation = star_replicas['Mstar'][0][2]-star_replicas['Kstar'][0][2]
        separation_deleted = star_replicas['Mstar'][1][2]-star_replicas['Kstar'][1][2]
        separation_se = np.sqrt((n-1)*np.mean((separation_deleted-separation_deleted.mean())**2))
        result['rows'][key] = dict(raw_sha256=file_sha256(path),imaginary_moment_checks=imaginary,
            stars=stars, records=len(groups), independent_chains=n,
            paired_M_minus_K_ritz_separation=dict(value=float(separation),chain_SE=float(separation_se),
                jackknife_bias_estimate=float((n-1)*(separation_deleted.mean()-separation))))
    contrasts = {}
    for star in ('Kstar','Mstar'):
        contrasts[star] = {}
        for field in field_names:
            zero = result['rows'][keys[0]]['stars'][star]['metrics'][field]
            frustrated = result['rows'][keys[1]]['stars'][star]['metrics'][field]
            difference = frustrated['value']-zero['value']
            se = np.hypot(zero['chain_SE'],frustrated['chain_SE'])
            contrasts[star][field] = dict(difference=float(difference),independent_coupling_SE=float(se),
                                          z_score=float(difference/se))
    result['coupling_contrasts'] = contrasts
    zero = result['rows'][keys[0]]['paired_M_minus_K_ritz_separation']
    frustrated = result['rows'][keys[1]]['paired_M_minus_K_ritz_separation']
    difference = frustrated['value']-zero['value']
    se = np.hypot(zero['chain_SE'],frustrated['chain_SE'])
    result['M_minus_K_separation_coupling_contrast'] = dict(difference=float(difference),
        independent_coupling_SE=float(se),z_score=float(difference/se))
    result['claim_boundary'] = ('K/M source-Ritz changes within the two-dimensional response trial space. '
        'Neither Ritz convergence/residual nor a gap is established; ordinary parent is not exactly SU2. '
        'A 95-percent source weight in a two-node quadrature is not a physical quasiparticle residue.')
    atomic_json(output,result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().input,parser.parse_args().output)
