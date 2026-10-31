"""One reusable quality/readout audit of frozen MPS and external iPEPS artifacts.

No optimization, ED, new samples or tensor changes. Broken-symmetry regional
projector weights are retained without manufacturing an entanglement spectrum.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import resource
import signal
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.tensor_network_readout import tenpy_graph_model
from su2_nqs.benchmarks.tensor_network_observables import regional_physical_observables,tenpy_spin_correlations
from .spin_tensor_network_reference import ROOT,_json,file_sha256,transitive_local_source_manifest


def analyze(artifacts: list[Path], output: Path, *, variance_max_chi: int = 64) -> dict:
    """Audit sealed inputs; exact H² is bounded to the declared low-chi stages.

    Final high-chi variance is explicitly UNMEASURED, not estimated from the
    truncation error or variance of a different checkpoint.
    """
    if (output/'results.json').exists():
        raise FileExistsError('completed analysis is immutable')
    import h5py
    from tenpy.tools import hdf5_io
    artifacts=[path.resolve() for path in artifacts]
    output.mkdir(parents=True,exist_ok=True)
    inputs={}; loaded=[]
    for directory in artifacts:
        result=json.loads((directory/'results.json').read_text())
        for p in [directory/'results.json',directory/'rdms.npz',*sorted(directory.glob('chi*.h5'))]:
            inputs[str(p.resolve())]=file_sha256(p)
        if file_sha256(directory/'rdms.npz')!=result['rdms_sha256']:
            raise ValueError('RDM seal mismatch')
        if 'checkpoint_sha256' in result and file_sha256(directory/'checkpoint.h5')!=result['checkpoint_sha256']:
            raise ValueError('MPS checkpoint seal mismatch')
        loaded.append((directory,result))
    registration=dict(inputs=inputs,variance_max_chi=variance_max_chi,variance_seconds_cap=120,
        sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')),
        protocol='Read frozen physical tensors/RDMs. No SU2 or spatial twirl, level clipping, optimization or phase labels. Infinite/finite energies are not ranked against one another.')
    registration['sources']['su2_nqs/frontier/provenance.py']=file_sha256(ROOT/'su2_nqs/frontier/provenance.py')
    _json(output/'registration.json',registration)
    start=perf_counter(); records=[]; arrays={}
    for directory,result in loaded:
        name=str(directory.relative_to(ROOT))
        with np.load(directory/'rdms.npz') as saved:
            regional={key:regional_physical_observables(saved[key],cyclic='winding6' in key)
                      for key in saved.files if 'native' not in key}
        record=dict(artifact=name,regional=regional)
        if 'history' in result:
            spec=result['registration']['config']; n=spec['rows']*spec['columns']
            model=tenpy_graph_model(n,[tuple(b) for b in result['registration']['bonds']])
            last=result['history'][-1]
            with h5py.File(directory/f'chi{last["chi_max"]}.h5','r') as handle:
                psi=hdf5_io.load_from_hdf5(handle)['psi']
            info,c=tenpy_spin_correlations(psi,spec['rows'],spec['columns'])
            replay=float(model.H_MPO.expectation_value(psi).real)/n
            bond_sum=sum(weight*c[i,j] for i,j,weight in result['registration']['bonds'])/n
            closure_error=float(abs(bond_sum-replay))
            if closure_error>1e-10:
                raise ValueError('physical pair sum disagrees with the MPO energy')
            record.update(kind='finite6x6_PBC_MPS',energy_per_site=replay,
                energy_replay_error=abs(replay-last['energy_per_site']),final_chi=last['actual_chi'],
                global_spin=info,bond_energy_closure_error=closure_error,final_variance_measured=False,
                planned_sweeps_completed=(len(result['history'])==len(spec['stages']) and
                    last.get('stage_sweeps_total',last['sweeps'])>=spec['stages'][-1]['sweeps']),
                last_energy_change_per_site=last['last_energy_change_per_site'])
            arrays[name.replace('/','_')+'_spin_correlations']=c
            eligible=[stage for stage in result['history'] if stage['actual_chi']<=variance_max_chi]
            if eligible:
                stage=eligible[-1]
                with h5py.File(directory/f'chi{stage["chi_max"]}.h5','r') as handle:
                    low=hdf5_io.load_from_hdf5(handle)['psi']
                old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('bounded full-H2 contraction')))
                signal.alarm(120); t=perf_counter()
                try:
                    value=float(model.H_MPO.variance(low).real)
                    record['variance_check']=dict(chi=stage['actual_chi'],variance_per_site=value/n,
                        seconds=perf_counter()-t,method='full uncompressed H² MPO contraction',
                        is_final_checkpoint=stage==last)
                    record['final_variance_measured']=stage==last
                except TimeoutError:
                    record['variance_check']=dict(chi=stage['actual_chi'],status='120s budget reached; variance NOT measured')
                finally:
                    signal.alarm(0); signal.signal(signal.SIGALRM,old)
            else:
                record['variance_check']=dict(status='No checkpoint below predeclared variance chi cap')
        else:
            record.update(kind='infinite_D5_spiral_iPEPS',ctm_records=result['records'],
                finite6x6_compatible=False,final_variance_measured=False)
        records.append(record)
        print('analyzed',directory,'kind',record['kind'],flush=True)
    np.savez_compressed(output/'correlations.npz',**arrays)
    result=dict(registration=registration,records=records,seconds=perf_counter()-start,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30,
        correlations_sha256=file_sha256(output/'correlations.npz'),new_ED=0,phase_claim=False,
        disposition='Reference-acquisition and quality diagnostics only. Require same-H competing states and stability under tensor/environment refinement before phase inference.')
    _json(output/'results.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variance-max-chi',type=int,default=64)
    args=p.parse_args(); analyze(args.artifacts,args.output,variance_max_chi=args.variance_max_chi)
