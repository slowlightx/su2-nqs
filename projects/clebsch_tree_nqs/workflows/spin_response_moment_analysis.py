"""Quality drift and momentum-resolved source moments, never gap fitting."""
from __future__ import annotations
import argparse
from pathlib import Path
import json

import numpy as np
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from .spin_spatial_vit_quality import read_sealed


def run(root: Path) -> dict:
    """Make compact numerical tables and line/errorbar figures from sealed runs."""
    import matplotlib.pyplot as plt
    keys = ('j0_s750','j0_s1000','j0125_s100','j0125_s200')
    rows = {}
    for key in keys:
        directory = root/key
        record = read_sealed(directory/'results.json')
        registration=read_sealed(directory/'registration.json')
        parent_file=root/'inputs'/('j0125_parent_registration.json' if key.startswith('j0125')
                                  else 'j0_parent_registration.json')
        parent=read_sealed(parent_file);spec=parent['config']
        if (file_sha256(parent_file)!=registration['inputs'][registration['config']['parent']+'/registration.json']
                or not spec.get('spin_inversion_even',False) or spec['model']['target_m2']!=0
                or spec['model']['rows']*spec['model']['columns']!=36):
            raise ValueError('odd-spin response bound needs verified N=36, Sz=0, even spin-flip parent')
        if record['status'] != 'completed' or file_sha256(directory/'moments.npz') != record['raw_sha256']:
            raise ValueError('incomplete or changed source moments')
        with np.load(directory/'moments.npz') as data:
            chains = data['chain_ids']; ids = np.unique(chains); n = len(ids)
            for star, names in [('Kstar',('K','minusK')),('Mstar',('M1','M2','M3'))]:
                fields = ['norm','commutator_first','commutator_second']
                means = np.array([[np.mean([data[name+'__'+field][chains==c].mean() for name in names])
                                   for c in ids] for field in fields])
                def reduced(moment):
                    return moment[2]/moment[0]-(moment[1]/moment[0])**2
                full = reduced(means.mean(1)); jack = reduced((n*means.mean(1)[:,None]-means)/(n-1))
                record['metrics'][star+'__real_centered_commutator_norm_squared'] = dict(
                    value=float(full),chain_SE=float(np.sqrt((n-1)*np.mean((jack-jack.mean())**2))),
                    jackknife_bias_estimate=float((n-1)*(jack.mean()-full)),
                    definition='||[H,O]psi-Re(<Opsi|[H,O]psi>)/||Opsi||^2 Opsi||^2/||Opsi||^2; not an excitation variance')
                defect=record['metrics'][star+'__casimir_defect']
                record['metrics'][star+'__odd_sector_nontriplet_upper_estimate'] = dict(
                    value=defect['value']/100,chain_SE=defect['chain_SE']/100,
                    jackknife_bias_estimate=defect['jackknife_bias_estimate']/100,
                    definition='For odd total S only, P(S>=3)<=<(S^2-2)^2>/100. Statistical estimate of a bound, not a certified confidence interval.')
                energy_moments=np.array([[np.mean([data[name+'__'+field][chains==c].mean() for name in names])
                                         for c in ids] for field in ('norm','first','second')])
                parent_moments=np.array([[data['parent__'+field][chains==c].mean() for c in ids]
                                        for field in ('norm','first')])
                def relative_width(e,p):
                    center=e[1]/e[0];mu=center-p[1]/p[0]
                    return (e[2]/e[0]-center**2)/mu**2
                ef=energy_moments.mean(1);pf=parent_moments.mean(1)
                full=relative_width(ef,pf)
                jack=relative_width((n*ef[:,None]-energy_moments)/(n-1),
                                    (n*pf[:,None]-parent_moments)/(n-1))
                record['metrics'][star+'__relative_width_squared'] = dict(
                    value=float(full),chain_SE=float(np.sqrt((n-1)*np.mean((jack-jack.mean())**2))),
                    jackknife_bias_estimate=float((n-1)*(jack.mean()-full)),
                    definition='source width squared / centroid-from-parent squared; exploratory dimensionless comparison, not quasiparticle residue')
        rows[key] = record
    contrasts = {}
    for observable in ('Kstar__centroid_from_parent','Mstar__centroid_from_parent',
                       'M_minus_K_centroid','Kstar__width_squared','Mstar__width_squared',
                       'Kstar__spectral_weight_per_parent_norm','Mstar__spectral_weight_per_parent_norm',
                       'Kstar__real_centered_commutator_norm_squared','Mstar__real_centered_commutator_norm_squared'):
        v = [rows[k]['metrics'][observable] for k in keys]
        difference = v[3]['value']-v[1]['value']
        se = np.hypot(v[3]['chain_SE'],v[1]['chain_SE'])
        drift = abs(v[3]['value']-v[2]['value'])+abs(v[1]['value']-v[0]['value'])
        contrasts[observable] = dict(latest_difference=float(difference), independent_coupling_SE=float(se),
            observed_quality_drift_sum=float(drift), exceeds_drift_plus_3SE=bool(abs(difference)>drift+3*se),
            scope='descriptive quality sensitivity, not a rigorous variational-bias bound or phase classification')
    labels = [('Kstar', '#2c7fb8', 'K, -K'), ('Mstar','#d95f0e','three M')]
    fig, axes = plt.subplots(1,4,figsize=(14.5,3.4),layout='constrained')
    for ax, field, ylabel in zip(axes, ('spectral_weight_per_parent_norm','centroid_from_parent','width_squared','relative_width_squared'),
        (r'$\langle O_k^\dagger O_k\rangle$',r'Source centroid $\mu_k/J_1$',r'Source width squared $\sigma_k^2/J_1^2$',r'Relative width squared $\sigma_k^2/\mu_k^2$')):
        for name,color,label in labels:
            for indices,marker,alpha in [((0,2),'o',.35),((1,3),'s',1.)]:
                values=[rows[keys[i]]['metrics'][name+'__'+field] for i in indices]
                ax.errorbar([0,.125], [v['value'] for v in values], yerr=[v['chain_SE'] for v in values],
                    marker=marker, color=color, alpha=alpha, ls='--' if alpha<1 else '-', capsize=3,
                    label=label+(' earlier' if alpha<1 else ' latest'))
        ax.set(xlabel=r'$J_2/J_1$',ylabel=ylabel,xticks=[0,.125]); ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same 6x6 torus and saved ViT states; source moments, not excitation gaps',fontsize=11)
    for suffix in ('png','pdf'): fig.savefig(root/f'spin_source_moments.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.4),layout='constrained')
    for ax,field,ylabel in zip(axes,('total_spin_squared','casimir_defect'),
         (r'$\langle S_{\rm tot}^2\rangle$ in $O_k\psi$',r'$\|[S_{\rm tot}^2-2]O_k\psi\|^2/\|O_k\psi\|^2$')):
        for name,color,label in labels:
            values=[rows[keys[i]]['metrics'][name+'__'+field] for i in (1,3)]
            ax.errorbar([0,.125],[v['value'] for v in values],yerr=[v['chain_SE'] for v in values],
                        marker='o',color=color,label=label,capsize=3)
        ax.set(xlabel=r'$J_2/J_1$',ylabel=ylabel,xticks=[0,.125]);ax.grid(alpha=.15)
    axes[0].axhline(2,color='gray',ls=':',label='exact triplet');axes[0].legend(fontsize=8)
    fig.suptitle('Triplet character is measured, not imposed on ordinary ViT',fontsize=11)
    for suffix in ('png','pdf'):fig.savefig(root/f'spin_source_purity.{suffix}',dpi=180)
    plt.close(fig)
    result=dict(rows={k:r['metrics'] for k,r in rows.items()},coupling_contrasts=contrasts,
        source_results={k:file_sha256(root/k/'results.json') for k in keys},
        physical_claim='Momentum-dependent source energy distributions and measured spin purity only',
        parity_verified='N=36 and Sz=0: even global bit-flip parent -> O_z psi odd total S; next S after1 is3',
        no_gap_or_continuum_certificate=True, no_phase_assignment=True,
        width_identity_max_error=max(abs(r['metrics'][name+'__width_decomposition_error']['value'])
                                     for r in rows.values() for name in ('Kstar','Mstar')))
    atomic_json(root/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    run(p.parse_args().input)
