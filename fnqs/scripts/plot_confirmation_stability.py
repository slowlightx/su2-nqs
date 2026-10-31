"""Display predeclared free-fit sensitivities and independent finite-L peaks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from fnqs.exact import TFIMReference
from fnqs.artifacts import write_json
from fnqs.pilots.lineage import sha
from fnqs.diagnostics.frozen_stability import local_peak
from fnqs.scripts.run_frozen_confirmation import verified


def run(output: Path) -> None:
    """Compare finite-size peak bias and numerical stability without refitting theta."""
    p=verified(output)
    if not (output/'endpoint_seal.json').exists():raise ValueError('all endpoints must be sealed before references')
    data=json.loads((output/'scaling_stability.json').read_text());fields=np.asarray([g[3] for g in p['reserved_points']])
    refs=[]
    for n in p['sizes']:
        y=np.array([TFIMReference(n,float(h)).fidelity_susceptibility() for h in fields])
        result=minimize_scalar(lambda h:-TFIMReference(n,float(h)).fidelity_susceptibility(),bounds=(.7,1.1),
            method='bounded',options={'xatol':1e-13})
        refs.append(dict(n=n,continuous_peak=float(result.x),optimization_success=bool(result.success),
            quadratic_peaks=[local_peak(fields,y,w) for w in (5,7,9)]))
    comparisons=[]
    for row in data['peaks']:
        ref=next(r for r in refs if r['n']==row['n']);nominal=row['primary']['h_peak']
        comparisons.append(dict(n=row['n'],seed=row['seed'],h_peak=nominal,reference_peak=ref['continuous_peak'],
            signed_bias=nominal-ref['continuous_peak'],window_thinning_spread=row['window_thinning_spread'],
            conditional_mc_interval=row['conditional_mc_interval'],bracketed=row['primary']['bracketed']))
    write_json(output/'peak_reference_check.json',dict(references=refs,comparisons=comparisons,
        finite_L_peaks_are_not_thermodynamic_hc=True,new_training=False,
        input_sha256=sha(output/'scaling_stability.json'),source_sha256=sha(Path(__file__))))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for index,fit in enumerate(data['fits']):
        for axis,name,true in [(axes[0,0],'hc',1.),(axes[0,1],'nu',1.)]:
            values=[f[name] for f in fit['sensitivity']]
            axis.plot(np.arange(len(values)),values,'o-',ms=3,lw=.7,label=str(fit['seed']))
            axis.axhline(true,color='black',ls='--',lw=.5)
            axis.set(xlabel='declared window / degree / correction combination',ylabel=f'free {name}')
        eta=[f['eta'] for f in fit['eta_sensitivity']]
        axes[1,0].plot(np.arange(len(eta)),eta,'o-',ms=3,lw=.7,label=str(fit['seed']))
    axes[1,0].axhline(.25,color='black',ls='--',lw=.5)
    axes[1,0].set(xlabel='r_min = 2, 3, 4; with / without 1/L correction',ylabel='free eta')
    for seed in p['seeds']:
        rows=sorted([r for r in comparisons if r['seed']==seed],key=lambda r:r['n'])
        axes[1,1].errorbar([r['n'] for r in rows],[r['h_peak'] for r in rows],
            yerr=[r['window_thinning_spread'] for r in rows],fmt='o-',lw=.7,capsize=3,label=str(seed))
    axes[1,1].plot([r['n'] for r in refs],[r['continuous_peak'] for r in refs],'k--',label='independent JW')
    axes[1,1].set(xlabel='fixed L',ylabel='finite-L peak h (bars: analysis spread)')
    for ax in axes.flat:ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.savefig(output/'scaling_and_peak_stability.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
