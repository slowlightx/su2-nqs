"""Bounded, immutable same-prefix18/same-row chirality C/M reference readout."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

from projects.clebsch_tree_nqs.workflows.dmrg_archive import file_sha256
from projects.clebsch_tree_nqs.workflows.dmrg_region_probe import (
    GIB,ROOT,_atomic_json,_canonical,bounded_subprocess,transitive_local_source_manifest,
)
from projects.clebsch_tree_nqs.workflows.mps_spin_rotation_probe import (
    _fingerprint,_own_peak_rss_bytes,_start_rss_watchdog,_validated_spec,
)


def _worker(request_path: Path) -> None:
    request=json.loads(request_path.read_text());spec=request['spec']
    memory=int(spec['memory_gib']*GIB)
    if sys.platform.startswith('linux'):
        _,hard=resource.getrlimit(resource.RLIMIT_AS)
        cap=memory if hard==resource.RLIM_INFINITY else min(memory,hard)
        resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    _start_rss_watchdog(memory);start=time.monotonic()
    import h5py
    import numpy as np
    import tenpy
    from tenpy.tools import hdf5_io
    from su2_nqs.benchmarks.mps_chirality_covariance import finite_mps_chirality_covariances
    checkpoint=Path(spec['checkpoint'])
    if file_sha256(checkpoint)!=spec['checkpoint_sha256']:raise ValueError('parent changed')
    with h5py.File(checkpoint,'r') as handle:
        if 'tenpy_version' in handle:
            version=handle['tenpy_version'][()]
            version=version.decode() if isinstance(version,bytes) else str(version)
            if version!=tenpy.__version__:raise ValueError('TeNPy version differs from checkpoint')
        psi=hdf5_io.Hdf5Loader(handle).load(spec.get('psi_hdf_group','psi'))
    if psi.L!=spec['n_sites'] or psi.bc!='finite' or max(psi.chi)>spec['chi_upper_bound']:
        raise ValueError('loaded parent differs from registered finite MPS')
    before=_fingerprint(psi);loaded=time.monotonic()
    edges=tuple(tuple((r*6+c,r*6+(c+1)%6) for c in range(6)) for r in range(3))
    progress_path=Path(request['worker_result']).parent/'progress.json'
    partial_path=Path(request['worker_result']).parent/'readout_partial.json'
    cache=None
    if partial_path.exists():
        partial=json.loads(partial_path.read_text())
        if (partial['record_sha256']!=_canonical({k:v for k,v in partial.items() if k!='record_sha256'})
                or partial['registration_sha256']!=request['registration_sha256']
                or partial['checkpoint_sha256']!=spec['checkpoint_sha256']):
            raise ValueError('partial physical-scalar cache identity mismatch')
        cache=partial['readout_cache']
    def progress(value):
        _atomic_json(progress_path,{k:v for k,v in value.items() if k!='readout_cache'})
        if 'readout_cache' in value:
            partial=dict(registration_sha256=request['registration_sha256'],
                checkpoint_sha256=spec['checkpoint_sha256'],readout_cache=value['readout_cache'],
                complete_matrix_claim=False,source_hashes_checked_by_registration=True)
            partial['record_sha256']=_canonical(partial);_atomic_json(partial_path,partial)
    readout=finite_mps_chirality_covariances(psi,18,edges,tolerance=spec['tolerance'],progress=progress,
        replica_method=spec.get('replica_method','legacy_six_probe'),cache=cache,include_C2=spec.get('include_C2',False))
    measured=time.monotonic();after=_fingerprint(psi)
    if before!=after or file_sha256(checkpoint)!=spec['checkpoint_sha256']:
        raise RuntimeError('readout changed physical parent/checkpoint')
    replay={key:abs(readout[key]-expected) for key,expected in spec['legacy_replay'].items()}
    if max(replay.values())>2e-9:raise ArithmeticError('same-parent old row1/full18 M replay failed')
    record=dict(status='complete',registration_sha256=request['registration_sha256'],spec=spec,
        checkpoint_sha256=spec['checkpoint_sha256'],actual_chi=max(psi.chi),readout=readout,
        parent_fingerprint_before=before,parent_fingerprint_after=after,
        load_and_input_seal_seconds=loaded-start,operator_seconds=measured-loaded,
        total_seconds=time.monotonic()-start,worker_peak_rss_bytes=_own_peak_rss_bytes(),
        legacy_replay_absolute_errors=replay,physical_edges=edges,
        runtime=dict(python=sys.version,tenpy=tenpy.__version__,numpy=np.__version__),
        new_training=0,new_samples=0,new_ED=0,strict_SU2_assumed=False,phase_claim=False)
    record['record_sha256']=_canonical(record)
    _atomic_json(Path(request['worker_result']),record)


def run_probe(config_path: Path,output: Path) -> dict:
    """Run one prefix18 reference under independent process wall/RSS caps."""
    spec=_validated_spec(json.loads(config_path.read_text()))
    if spec['n_sites']!=36 or spec['cut']!=18 or spec['columns']!=6:
        raise ValueError('this frozen same-operator comparison fixes N36/prefix18/6 columns')
    if spec['threads']>8 or spec['memory_gib']>64 or spec['wall_seconds']>1700:
        raise ValueError('approved per-parent 8CPU/64GiB/30min bound exceeded')
    if set(spec['legacy_replay'])!={'exact_old_row1_replay_value','exact_old_full18_replay_value'}:
        raise ValueError('both independently frozen old M values required')
    checkpoint=Path(spec['checkpoint']).resolve()
    if file_sha256(checkpoint)!=spec['checkpoint_sha256']:raise ValueError('sealed checkpoint mismatch')
    spec['checkpoint']=str(checkpoint)
    if output.exists() and any(output.iterdir()) and not (output/'registration.json').exists():
        raise FileExistsError('unregistered nonempty output cannot be resumed')
    output.mkdir(parents=True,exist_ok=True)
    sources=transitive_local_source_manifest([Path(__file__),ROOT/'su2_nqs/benchmarks/mps_chirality_covariance.py',
        ROOT/'su2_nqs/frontier/provenance.py'],
        root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs.workflows'))
    registration=dict(spec=spec,sources=sources,new_training=0,new_samples=0,new_ED=0,
        no_gpu=True,fixed_observable='Three oriented winding row kappa_z sums at the same prefix18 as NQS')
    digest=_canonical(registration);registration['registration_sha256']=digest
    if (output/'registration.json').exists():
        if json.loads((output/'registration.json').read_text())!=registration:
            raise ValueError('resumed readout source/config registration changed')
        if not spec.get('allow_partial_resume',False):raise FileExistsError('readout output immutable')
        if (output/'results.json').exists():
            existing=json.loads((output/'results.json').read_text())
            if existing['result_sha256']!=_canonical({k:v for k,v in existing.items() if k!='result_sha256'}) or existing['registration_sha256']!=digest:
                raise ValueError('resumed supervisor result identity mismatch')
            if existing['status']=='complete':return existing
            attempt=output/'failed_attempts'
            attempt.mkdir(exist_ok=True)
            archive=attempt/f"result{len(list(attempt.glob('result*.json'))):03d}.json"
            _atomic_json(archive,existing)
    else:_atomic_json(output/'registration.json',registration)
    request=dict(spec=spec,registration_sha256=digest,worker_result=str((output/'worker_result.json').resolve()))
    _atomic_json(output/'request.json',request)
    env=os.environ.copy()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[name]=str(spec['threads'])
    env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    supervisor=bounded_subprocess([sys.executable,'-m',
        'projects.clebsch_tree_nqs.workflows.mps_chirality_covariance_probe','--worker',str((output/'request.json').resolve())],
        wall_seconds=spec['wall_seconds'],memory_bytes=int(spec['memory_gib']*GIB),log_path=output/'worker.log',env=env)
    result=dict(status='failed',registration_sha256=digest,supervisor=supervisor)
    child=output/'worker_result.json'
    if supervisor['exit_code']==0 and child.exists():
        record=json.loads(child.read_text())
        if record['record_sha256']!=_canonical({k:v for k,v in record.items() if k!='record_sha256'}) or record['registration_sha256']!=digest:
            raise RuntimeError('worker output seal mismatch')
        result.update(status='complete',worker_result_sha256=file_sha256(child),readout=record)
    result['result_sha256']=_canonical(result);_atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker is not None:_worker(args.worker)
    elif args.config is not None and args.output is not None:
        result=run_probe(args.config,args.output);print(json.dumps(result),flush=True)
        if result['status']!='complete':raise SystemExit(1)
    else:parser.error('supply --config and --output, or internal --worker')
