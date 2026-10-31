"""Explicit connected/offsite definitions on frozen, symmetry-projected audits.

For the verified fixed-Sz and even-spin-flip endpoints all local vector means
vanish exactly. Preserve original audit files and statistical errors. Do not
use noisy empirical magnetizations to subtract a spurious elastic term.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np

from su2_nqs.benchmarks.connected_spin_structure import (
    connected_spin_structure_factors,connected_longitudinal_structure_factors,
)
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from .spin_spatial_vit_quality import read_sealed


def run(root: Path, output: Path) -> dict:
    """Verify point-moment selection rules and relabel existing NQS readouts."""
    review=json.loads((root/'optimization_review.json').read_text());rows=[]
    for endpoint in review['nqs']:
        parent=root/'inputs/nqs'/endpoint['run']/'seed1'
        audit=parent/'audits'/f"step{endpoint['step']:05d}/results.json"
        if not audit.exists():continue
        registration=read_sealed(parent/'registration.json');record=read_sealed(audit)
        spec=registration['config'];model=spec['model'];n=model['rows']*model['columns']
        if (model['target_m2']!=0 or not spec.get('spin_inversion_even',False)
                or record['audit_projection']!='gamma_a1'):
            raise ValueError('cannot infer all point moments vanish for this endpoint')
        momenta=spec['momenta'];names=list(momenta)
        coordinates=np.array([divmod(i,model['columns']) for i in range(n)])
        m=np.zeros((n,3))
        scalar=np.array([record['structure_factors'][k]['scalar']['value'] for k in names])
        zz=np.array([record['structure_factors'][k]['zz']['value'] for k in names])
        s=connected_spin_structure_factors(scalar,m,coordinates,momenta)
        z=connected_longitudinal_structure_factors(zz,m,coordinates,momenta)
        readouts={}
        for i,name in enumerate(names):
            readouts[name]=dict(scalar_connected=float(s['connected'][i]),
                scalar_elastic=float(s['elastic'][i]),scalar_connected_offsite=float(s['connected_offsite'][i]),
                scalar_chain_SE=record['structure_factors'][name]['scalar']['chain_SE'],
                longitudinal_connected=float(z['connected'][i]),
                longitudinal_elastic=float(z['elastic'][i]),
                longitudinal_connected_offsite=float(z['connected_offsite'][i]),
                longitudinal_chain_SE=record['structure_factors'][name]['zz']['chain_SE'])
        rows.append(dict(run=endpoint['run'],step=endpoint['step'],geometry=model['geometry'],
            N=n,j2=spec['j2_over_j1'],energy=record['energy'],readouts=readouts,
            local_magnetizations_exactly_zero=True,
            reason='fixed total Sz gives <Sx>=<Sy>=0; even global bit flip gives each <Sz>=0',
            onsite_scalar_quantum_variance=s['onsite_quantum_variance'],
            onsite_longitudinal_quantum_variance=z['onsite_quantum_variance'],
            audit_sha256=file_sha256(audit),registration_sha256=file_sha256(parent/'registration.json')))
    result=dict(rows=rows,
        connected_definition='sum_ij exp(ik.(ri-rj)) [<Si.Sj>-<Si>.<Sj>]/N; includes i=j quantum variance',
        offsite_definition='connected minus sum_i [3/4-|<Si>|^2]/N; a separate signed report',
        old_NQS_values_equal_connected=True,source_response_elastic_subtraction_exactly_zero=True,
        magnetization_extrapolation=False,new_samples=0,new_training=0,new_amplitude_queries=0,
        review_sha256=file_sha256(root/'optimization_review.json'))
    output.mkdir(parents=True,exist_ok=True);atomic_json(output/'connected_structure.json',result)
    latest=[]
    for ratio in (0,.05,.10,.125,.175):
        candidates=[r for r in rows if r['geometry']=='triangular' and r['j2']==ratio
                    and 'd2e4' not in r['run']]
        if candidates:latest.append(max(candidates,key=lambda r:r['step']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for ax,field,title in zip(axes,('scalar_connected','scalar_connected_offsite'),
                              ('Connected, onsite fluctuations included','Connected, offsite contribution only')):
        for name,color,style in [('K','#2878b5','-'),('M1','#c25b27','-'),
                                 ('M2','#c25b27','--'),('M3','#c25b27',':')]:
            ax.errorbar([r['j2'] for r in latest],[r['readouts'][name][field] for r in latest],
                        yerr=[r['readouts'][name]['scalar_chain_SE'] for r in latest],
                        color=color,linestyle=style,marker='o',ms=4,capsize=2,label=name)
        ax.set(xlabel=r'$J_2/J_1$',ylabel=r'$S_c(k)$',title=title);ax.grid(alpha=.15)
    axes[0].legend(fontsize=8);fig.suptitle('Frozen 6x6 triangular ViT endpoints; explicit connected convention')
    for suffix in ('png','pdf'):fig.savefig(output/f'connected_structure_sequence.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps([dict(j2=r['j2'],S_K=r['readouts']['K']['scalar_connected'],
        S_K_offsite=r['readouts']['K']['scalar_connected_offsite'],
        S_M=[r['readouts'][name]['scalar_connected'] for name in ('M1','M2','M3')]) for r in latest],indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.input,args.output)
