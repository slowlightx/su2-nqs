"""Predeclared free-exponent/window/peak analysis of frozen held-field data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from fnqs.artifacts import write_json
from fnqs.pilots.lineage import sha
from fnqs.scaling import collapse_fit, phase0_gate
from fnqs.diagnostics.frozen_stability import eta_fit, local_peak
from fnqs.scripts.run_frozen_confirmation import verified


def fit_arrays(sizes: list[int],fields: np.ndarray,chi: np.ndarray,window: float,
               degree: int,correction: bool) -> dict:
    """Select the frozen physical window without fixing the fitted hc or nu."""
    ns=np.repeat(sizes,len(fields));hs=np.tile(fields,len(sizes));ys=np.asarray(chi).ravel()
    mask=abs((hs-1.)*ns)<=window
    return collapse_fit(ns[mask],hs[mask],ys[mask],degree,correction)


def correlations_at(fields: np.ndarray,corr: list[np.ndarray],hc: float) -> list[np.ndarray]:
    """Interpolate independently measured full-C profiles at free fitted hc."""
    if not fields[0]<=hc<=fields[-1]:raise ValueError('free hc outside measured window')
    return [np.asarray([np.interp(hc,fields,c[:,r]) for r in range(c.shape[1])]) for c in corr]


def analyze(output: Path) -> dict:
    """Keep every initialization and all sensitivities, even when gates fail."""
    p=verified(output);a=p['analysis'];sizes=p['sizes'];seeds=p['seeds']
    confirmation=json.loads((output/'confirmation.json').read_text());count=len(p['reserved_points'])
    fields=np.asarray([g[3] for g in p['reserved_points']]);data=[];bindings={}
    for seed in seeds:
        group=[]
        for n in sizes:
            path=output/'runs'/f'L{n}_seed_{seed}'/'audit.json'
            rows=json.loads(path.read_text())['points'][:count];group.append(rows);bindings[str(path)]=sha(path)
            if any(r['split']!='reserved_confirmation' for r in rows):raise ValueError('analysis field leakage')
        data.append(group)
    chi=np.asarray([[[r['chi_total'] for r in rows] for rows in group] for group in data])
    corr=[[np.asarray([r['correlations'] for r in rows]) for rows in group] for group in data]
    primary=a['primary'];fits=[]
    for i,label in [*(enumerate(seeds)),(-1,'pooled')]:
        y=chi.mean(axis=0) if i==-1 else chi[i]
        c=[np.mean([cs[j] for cs in corr],axis=0) for j in range(len(sizes))] if i==-1 else corr[i]
        f=fit_arrays(sizes,fields,y,**primary)
        eta=eta_fit(sizes,correlations_at(fields,c,f['hc']),a['eta_primary_r_min'])
        sensitivity=[]
        for window in a['windows']:
            for degree in a['degrees']:
                for correction in a['corrections']:
                    alt=fit_arrays(sizes,fields,y,window,degree,correction)
                    sensitivity.append(dict(window=window,**alt))
        etas=[eta_fit(sizes,correlations_at(fields,c,f['hc']),r,correction)
              for r in a['eta_r_min'] for correction in a['corrections']]
        fits.append(dict(seed=label,primary=f,eta=eta,sensitivity=sensitivity,eta_sensitivity=etas,
            hc_spread=max(abs(s['hc']-f['hc']) for s in sensitivity),
            nu_spread=max(abs(s['nu']-f['nu']) for s in sensitivity),
            eta_spread=max(abs(s['eta']-eta['eta']) for s in etas),
            all_fits_interior=all(s['success'] and not s['at_bound'] for s in sensitivity)))
    # Initialization bootstrap preserves the same resampled seed indices at all L.
    # Gaussian conditional MC draws retain the measured chi/C cross covariance.
    rng=np.random.default_rng(a['bootstrap_seed']);replicates=[];failed=0
    roots=[]
    for group in data:
        blocks=[]
        for rows in group:
            factors=[]
            for r in rows:
                cov=np.asarray(r['joint_covariance']);values,vectors=np.linalg.eigh(cov)
                if values.min()<-1e-10:raise ValueError('joint estimator covariance is not PSD')
                factors.append(vectors*np.sqrt(np.maximum(0.,values)))
            blocks.append(factors)
        roots.append(blocks)
    for _ in range(a['bootstrap']):
        picked=rng.integers(len(seeds),size=len(seeds));ys=[];cs=[[] for _ in sizes]
        for i in picked:
            yy=[]
            for j,n in enumerate(sizes):
                values=[]
                for k in range(count):
                    base=np.r_[chi[i,j,k],corr[i][j][k]]
                    values.append(base+roots[i][j][k]@rng.normal(size=n+1))
                values=np.asarray(values);yy.append(values[:,0]);cs[j].append(values[:,1:])
            ys.append(yy)
        try:
            f=fit_arrays(sizes,fields,np.mean(ys,axis=0),**primary)
            e=eta_fit(sizes,correlations_at(fields,[np.mean(c,axis=0) for c in cs],f['hc']),a['eta_primary_r_min'])
            replicates.append([f['hc'],f['nu'],e['eta']])
        except ValueError:failed+=1
    intervals={name:np.quantile(np.asarray(replicates)[:,i],[.025,.5,.975]).tolist()
               for i,name in enumerate(('hc','nu','eta'))} if replicates else {}
    peaks=[]
    for i,seed in enumerate(seeds):
        for j,n in enumerate(sizes):
            alternatives=[local_peak(fields,chi[i,j],w) for w in a['peak_windows']]
            for offset in (0,1):alternatives.append({**local_peak(fields[offset::2],chi[i,j,offset::2],7),'thinning_offset':offset})
            nominal=local_peak(fields,chi[i,j],7);draws=[]
            for _ in range(a['bootstrap']):
                sample=chi[i,j]+rng.normal(size=count)*np.asarray([r['chi_se'] for r in data[i][j]])
                draws.append(local_peak(fields,np.maximum(sample,1e-12),7)['h_peak'])
            peaks.append(dict(seed=seed,n=n,primary=nominal,alternatives=alternatives,
                window_thinning_spread=max(abs(q['h_peak']-nominal['h_peak']) for q in alternatives),
                conditional_mc_interval=np.quantile(draws,[.025,.5,.975]).tolist(),
                finite_L_peak_is_not_hc=True))
    pooled=fits[-1];gate=phase0_gate(pooled['primary']['hc'],pooled['primary']['nu'],pooled['eta']['eta'],
        confirmation['confirmed_observable_accuracy'],sizes,len(seeds))
    stable=all(f['all_fits_interior'] and f['hc_spread']<=a['stability_limits']['hc'] and
        f['nu_spread']<=a['stability_limits']['nu'] and f['eta_spread']<=a['stability_limits']['eta'] for f in fits)
    complete_size_stability=len(sizes)>=4
    result=dict(fits=fits,bootstrap=dict(replicates=replicates,failed=failed,intervals=intervals,
        limitations=a['confidence_limits'],size_resampling=False),peaks=peaks,original_gate=gate,
        window_and_fit_form_stable=stable,minimum_size_drop_check_available=complete_size_stability,
        phase0_scientific_gate=gate['passed'] and stable and complete_size_stability,
        audit_sha256=bindings,confirmation_sha256=sha(output/'confirmation.json'),protocol_sha256=sha(output/'protocol.json'))
    write_json(output/'scaling_stability.json',result)
    plot(output,fields,sizes,seeds,chi,fits,peaks)
    return result


def plot(output: Path,fields: np.ndarray,sizes: list[int],seeds: list[int],chi: np.ndarray,fits: list,peaks: list) -> None:
    """Export a scientific figure preserving all three fresh initializations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from fnqs.exact import TFIMReference
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    for j,n in enumerate(sizes):
        for i,seed in enumerate(seeds):axes[j].plot(fields,chi[i,j],label=f'fresh seed {seed}',lw=1.2)
        axes[j].plot(fields,[TFIMReference(n,float(h)).fidelity_susceptibility() for h in fields],color='black',ls='--',label='independent JW')
        axes[j].set(title=f'Fixed L={n}',xlabel='physical h',ylabel='total susceptibility')
        axes[j].legend(fontsize=7)
    fig.savefig(output/'confirmation_profiles.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    analyze(parser.parse_args().output)
