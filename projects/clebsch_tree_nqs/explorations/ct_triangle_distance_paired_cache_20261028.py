"""Same-IID-prefix near/far frozen scalar comparison; absolutely zero queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from .ct_A1_frozen_cache_postprocess_20261028 import paired_statistics

ROOT=Path(__file__).resolve().parents[3]
RESULTS=ROOT/'projects/clebsch_tree_nqs/results'
CAMPAIGN=RESULTS/'su2_mechanism_local_20261028'
SPATIAL=RESULTS/'su2_spatial_frozen_channels_local_20261028'


def run(output: Path) -> dict:
    tick=perf_counter()
    if output.exists():raise FileExistsError('new immutable distance comparison required')
    paths=dict(six=CAMPAIGN/'interventions/frozen_six_body.npz',joint=CAMPAIGN/'interventions/frozen_operators.npz')
    with np.load(paths['six']) as f:ops={k:f[k] for k in f.files}
    with np.load(paths['joint']) as f:
        for key in ('P_1_1_0','P_1_1_2'):ops[key]=f[key]
    keys=tuple(ops);cases={};missing=[]
    for tag,near_dir,far_dir in [('ct6_j0',CAMPAIGN/'ct_A1_compact6',SPATIAL/'ct_far_J0_v1'),
                                ('ct6_j0125',CAMPAIGN/'ct_A1_compact6_j0125',SPATIAL/'ct_far_J125_v1')]:
        paths_pair=[p/(tag+'_physical_region.npz') for p in (near_dir,far_dir)]
        result_paths=[p/(tag+'_results.json') for p in (near_dir,far_dir)]
        if not all(p.exists() for p in paths_pair+result_paths):
            missing.append(tag);continue
        reports=[json.loads(p.read_text()) for p in result_paths]
        if reports[0]['parameter_sha256']!=reports[1]['parameter_sha256'] or reports[0]['source_products_sha256']!=reports[1]['source_products_sha256']:
            raise ValueError('two distances must share exact endpoint and originalIID panel')
        panels=[]
        for path in paths_pair:
            with np.load(path) as f:panels.append({k:f[k] for k in ('rows','sites','proposal','region_amplitudes')})
        near,far=panels
        if not np.array_equal(near['rows'],far['rows']) or not np.array_equal(near['proposal'],far['proposal']):
            raise ValueError('distance samples are not the same original IID records/Q')
        if not np.array_equal(near['sites'],[12,13,14,18,19,20]) or not np.array_equal(far['sites'],[12,14,15,18,19,21]):
            raise ValueError('frozen up/down near/far motif site order changed')
        products=RESULTS/'ct_status_20261028/verified_next_parents'/tag/'parent_snapshot/products.npz'
        if file_sha256(products)!=reports[0]['source_products_sha256']:raise ValueError('original IID source changed')
        with np.load(products) as f:parent=f['psi'][:512,1]
        q=near['proposal'];mass=abs(parent)**2/q
        values=[];center_errors=[]
        for panel in panels:
            packed=panel['rows'][:,panel['sites']]@(2**np.arange(5,-1,-1));phi=panel['region_amplitudes']
            err=float(np.max(abs(phi[np.arange(512),packed]-parent))/max(np.max(abs(parent)),1e-300))
            if err>2e-11:raise ValueError('common physical normalization/phase failed')
            center_errors.append(err)
            values.append(np.stack([parent.conj()*np.sum(O[packed]*phi,axis=1)/q for O in ops.values()],axis=1))
        joint=np.concatenate([values[0],values[1],values[1]-values[0]],axis=1)
        names=tuple(prefix+key for prefix in ('near::','far::','delta::') for key in keys)
        ratios={}
        for prefix in ('near::','far::'):
            ratios[prefix+'c0']=(prefix+'T_qtotal0',prefix+'P_1_1_0')
            ratios[prefix+'c1']=(prefix+'T_qtotal2',prefix+'P_1_1_2')
        report=paired_statistics(joint,mass,names,ratios)
        total=joint.sum(0);group=joint.reshape(32,16,len(names)).sum(1)
        leaves=(total-group)/(mass.sum()-mass.reshape(32,16).sum(1))[:,None]
        point=total/mass.sum();report['conditional_far_minus_near']={}
        for label,num,den in [('c0','T_qtotal0','P_1_1_0'),('c1','T_qtotal2','P_1_1_2')]:
            na,nb,fa,fb=[names.index(prefix+key) for prefix,key in [('near::',num),('near::',den),('far::',num),('far::',den)]]
            if min(point[nb].real,point[fb].real)<=0 or np.any(leaves[:,[nb,fb]].real<=0):
                report['conditional_far_minus_near'][label]=dict(withheld=True,reason='nonpositive estimated sector weight')
                continue
            delta=(leaves[:,fa]/leaves[:,fb]-leaves[:,na]/leaves[:,nb]).real
            report['conditional_far_minus_near'][label]=dict(value=float((point[fa]/point[fb]-point[na]/point[nb]).real),
                paired_group_jackknife_SE=float(np.sqrt(31/32*np.sum((delta-delta.mean())**2))))
        report.update(parameter_sha256=reports[0]['parameter_sha256'],global_step=reports[0]['global_step'],
            source_products_sha256=reports[0]['source_products_sha256'],
            input_caches=[dict(path=str(p.relative_to(ROOT)),sha256=file_sha256(p)) for p in paths_pair],
            cached_center_replay_errors=center_errors,records=512,independent_IID_groups=32,
            near_distance_squared=7/3,far_distance_squared=19/3,
            geometric_pair='Same CCW A=(12,18,19); nearB=(13,20,14);farB=(14,21,15)',
            TR_scope='Frozen CT banks and cached amplitudes are real; physical Hermitian scalar chirality onepoints vanish. No finite noisy imaginary local estimator is subtracted as if a real expectation.')
        cases[tag]=report
    if not cases:raise FileNotFoundError('no same-parent completed near/far pair available')
    output.mkdir(parents=True)
    result=dict(task='CT_frozen_same_prefix_two_distance_paired_readout_v1',cases=cases,not_available=missing,
        frozen_operator_source_sha256={k:file_sha256(p) for k,p in paths.items()},
        no_new_amplitude_queries=True,no_new_H_queries=True,no_new_samples=True,
        cached_rows_not_a_new_untouched_audit=True,
        interpretation_limits='Two torus distances only; no fitted lengthscale or phase claim. T0/T2 are invertible linear combinations of T andkappa, not two additional independent high-order observables.',
        source_script_sha256=file_sha256(Path(__file__)),wall_seconds=perf_counter()-tick)
    atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.output)
    print(json.dumps(dict(output=str(args.output),cases=list(result['cases']),not_available=result['not_available'],seconds=result['wall_seconds'])))
