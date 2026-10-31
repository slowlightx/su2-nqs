"""Bounded rotation covariance and leading Schmidt-vector spin content.

No new state, sampling, canonicalization, twirl, or optimizer is introduced.
Use the immutable finite MPS checkpoint and the existing resource supervisor.
"""
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
    GIB, ROOT, _atomic_json, _canonical, bounded_subprocess,
    transitive_local_source_manifest,
)
from projects.clebsch_tree_nqs.workflows.mps_spin_rotation_probe import (
    _fingerprint, _own_peak_rss_bytes, _start_rss_watchdog, _validated_spec,
)


def _worker(request_path: Path) -> None:
    request=json.loads(request_path.read_text())
    spec=request['spec']
    memory=int(spec['memory_gib']*GIB)
    if sys.platform.startswith('linux'):
        _,hard=resource.getrlimit(resource.RLIMIT_AS)
        cap=memory if hard==resource.RLIM_INFINITY else min(memory,hard)
        resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    _start_rss_watchdog(memory)
    start=time.monotonic()
    import h5py
    import numpy as np
    import tenpy
    from tenpy.tools import hdf5_io
    from su2_nqs.benchmarks.mps_rotation_covariance import finite_mps_prefix_rotation_covariance
    checkpoint=Path(spec['checkpoint'])
    if file_sha256(checkpoint)!=spec['checkpoint_sha256']:
        raise ValueError('checkpoint changed before loading')
    with h5py.File(checkpoint,'r') as handle:
        if 'tenpy_version' in handle:
            stored=handle['tenpy_version'][()]
            stored=stored.decode() if isinstance(stored,bytes) else str(stored)
            if stored!=tenpy.__version__:
                raise ValueError('checkpoint and worker TeNPy versions differ')
        psi=hdf5_io.Hdf5Loader(handle).load(spec.get('psi_hdf_group','psi'))
    if psi.L!=spec['n_sites'] or psi.bc!='finite' or max(psi.chi)>spec['chi_upper_bound']:
        raise ValueError('finite MPS differs from the registered size/bond cap')
    before=_fingerprint(psi)
    loaded=time.monotonic()
    readout=finite_mps_prefix_rotation_covariance(psi,spec['cut'],
        tolerance=spec['tolerance'],leading_levels=spec['leading_levels'],
        measured_charges=tuple(spec['measured_charges']),
        spin_groups={name: tuple(sites) for name, sites in spec['spin_groups'].items()},
        transition_sites=tuple(spec.get('transition_sites', [])),
        chirality_groups={name:tuple(tuple(edge) for edge in edges)
                          for name,edges in spec.get('chirality_groups',{}).items()},
        chirality_z_response=spec.get('chirality_z_response',False),
        spatial_strip_columns=spec.get('spatial_strip_columns'),
        spatial_levels=spec.get('spatial_levels',4),
        total_spin_ladder=spec.get('total_spin_ladder',False),
        bond_quadrupole_rows=tuple(spec.get('bond_quadrupole_rows',[])),
        fourier_spin_weights={name: tuple(np.exp(2j*np.pi*(
            np.arange(spec['cut']) % spec['columns']*k[0]
            +np.arange(spec['cut'])//spec['columns']*k[1])))
            for name, k in spec.get('spin_momenta', {}).items()})
    measured=time.monotonic()
    after=_fingerprint(psi)
    if before!=after or file_sha256(checkpoint)!=spec['checkpoint_sha256']:
        raise RuntimeError('measurement changed its physical parent or checkpoint')
    record=dict(status='complete',registration_sha256=request['registration_sha256'],
        checkpoint_sha256=spec['checkpoint_sha256'],spec=spec,actual_chi=max(psi.chi),
        readout=readout,parent_fingerprint_before=before,parent_fingerprint_after=after,
        load_and_input_seal_seconds=loaded-start,
        full_environment_and_operator_seconds=measured-loaded,
        postprocessing_and_verification_seconds=time.monotonic()-measured,
        worker_peak_rss_bytes=_own_peak_rss_bytes(),
        runtime=dict(python=sys.version,tenpy=tenpy.__version__,numpy=np.__version__),
        new_training=0,new_samples=0,new_ED=0,phase_claim=False,
        physical_rotation_covariance_measured=True,
        actual_leading_schmidt_vector_casimir_measured=True,
        strict_su2_irrep_spectrum_claimed=False)
    record['record_sha256']=_canonical(record)
    _atomic_json(Path(request['worker_result']),record)


def run_probe(config_path: Path, output: Path) -> dict:
    """Seal one fixed input and physical readout, under explicit process caps."""
    spec=_validated_spec(json.loads(config_path.read_text()))
    spec['renyi2']=False
    spatial_columns=spec.get('spatial_strip_columns')
    if spatial_columns is not None and (isinstance(spatial_columns,bool)
            or not isinstance(spatial_columns,int) or spatial_columns<2
            or spec['cut']%spatial_columns or spec['cut']>20):
        raise ValueError('bounded complete-row spatial strip required')
    spatial_levels=spec.get('spatial_levels',4)
    if (isinstance(spatial_levels,bool) or not isinstance(spatial_levels,int)
            or not 1<=spatial_levels<=16 or not isinstance(spec.get('total_spin_ladder',False),bool)):
        raise ValueError('spatial_levels must be1..16 and total_spin_ladder boolean')
    spec['leading_levels']=spec.get('leading_levels',8)
    if isinstance(spec['leading_levels'],bool) or not isinstance(spec['leading_levels'],int) or not 1<=spec['leading_levels']<=64:
        raise ValueError('leading_levels must be an integer in 1..64')
    spec['measured_charges'] = spec.get('measured_charges',
        list(range(spec['cut']%2, min(spec['cut'], 4)+1, 2)))
    if (not isinstance(spec['measured_charges'], list) or not spec['measured_charges']
            or len(set(spec['measured_charges'])) != len(spec['measured_charges'])
            or any(isinstance(m, bool) or not isinstance(m, int) or abs(m) > spec['cut']
                   or (m-spec['cut'])%2 for m in spec['measured_charges'])):
        raise ValueError('unique legal doubled prefix magnetic charges required')
    spec['spin_groups'] = spec.get('spin_groups', {})
    groups = spec['spin_groups']
    if (not isinstance(groups, dict) or len(groups) > 6
            or any(not isinstance(k, str) or not k for k in groups)
            or any(not isinstance(v, list) or not v
                   or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < spec['cut']
                          for i in v) for v in groups.values())):
        raise ValueError('up to six named nonempty prefix spin groups required')
    flattened = [i for sites in groups.values() for i in sites]
    if len(flattened) != len(set(flattened)):
        raise ValueError('spin groups must not overlap')
    transition_sites = spec.get('transition_sites', [])
    if (not isinstance(transition_sites, list) or len(set(transition_sites)) != len(transition_sites)
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < spec['cut']
                   for i in transition_sites)):
        raise ValueError('unique transition_sites inside prefix required')
    modes = spec.get('spin_momenta', {})
    if (not isinstance(modes, dict) or len(modes) > 8
            or any(not isinstance(k, str) or not k or not isinstance(v, list) or len(v) != 2
                   or any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in v)
                   for k, v in modes.items())
            or modes and (not isinstance(spec.get('columns'), int) or spec['columns'] < 1)):
        raise ValueError('named reciprocal coordinate momenta and positive columns required')
    chirality=spec.get('chirality_groups',{})
    if not isinstance(spec.get('chirality_z_response',False),bool):
        raise ValueError('chirality_z_response must be boolean')
    if (not isinstance(chirality,dict) or len(chirality)>6
            or any(not isinstance(k,str) or not k or not isinstance(edges,list) or not edges
                   or any(not isinstance(e,list) or len(e)!=2 or e[0]==e[1]
                       or any(not isinstance(i,int) or isinstance(i,bool) or not 0<=i<spec['cut'] for i in e)
                       for e in edges) for k,edges in chirality.items())):
        raise ValueError('legal oriented chirality bond groups required')
    checkpoint=Path(spec['checkpoint']).resolve()
    if file_sha256(checkpoint)!=spec['checkpoint_sha256']:
        raise ValueError('checkpoint does not match registered SHA256')
    spec['checkpoint']=str(checkpoint)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('completed probes are immutable; use a fresh output')
    output.mkdir(parents=True,exist_ok=True)
    from su2_nqs.benchmarks.mps_spin_rotation import estimate_prefix_rotation_memory
    estimate=estimate_prefix_rotation_memory(spec['chi_upper_bound'],checkpoint.stat().st_size)
    # Operator LPs and their dense closed-cut Grams coexist. Do not assume
    # sparsity, or count spatial observables as a free postprocessing step.
    # Include both charged and charge-zero completed LPs and every partial
    # LP, including the additional Sm partial for a requested kappa_z trace.
    with_z=spec.get('chirality_z_response',False)
    chiral_lps=(2 if with_z else 1)*len(chirality)+(3 if with_z else 2)*sum(map(len,chirality.values()))
    extra_bytes = 2*(len(groups)+len(transition_sites)+len(modes)+chiral_lps)*spec['chi_upper_bound']**2*16
    # Completed row Grams plus at most two open bonds while visiting a row.
    quadrupole_rows=spec.get('bond_quadrupole_rows',[])
    if quadrupole_rows:
        if (spatial_columns is None or spatial_columns < 4 or spatial_columns%2
                or not isinstance(quadrupole_rows,list) or len(quadrupole_rows)>4
                or len(set(quadrupole_rows))!=len(quadrupole_rows)
                or any(isinstance(r,bool) or not isinstance(r,int)
                    or not 0<=r<spec['cut']//spatial_columns for r in quadrupole_rows)
                or not {0,4}.issubset(spec['measured_charges'])):
            raise ValueError('quadrupole rows require even strip columns and m=0,4')
        extra_bytes += 2*(len(quadrupole_rows)+2)*spec['chi_upper_bound']**2*16
    if spatial_columns is not None:
        # Dense view/conversion of one physical MPS tensor plus the few
        # reconstructed vectors; no full prefix matrix or dense RDM.
        extra_bytes += 4*spec['chi_upper_bound']**2*16+2**spec['cut']*2*spatial_levels*16
    estimate['spatial_operator_extra_bytes'] = extra_bytes
    estimate['estimated_total_bytes'] += extra_bytes
    sources=transitive_local_source_manifest(
        [Path(__file__),ROOT/'su2_nqs/benchmarks/mps_rotation_covariance.py',
         ROOT/'projects/clebsch_tree_nqs/workflows/mps_spin_rotation_probe.py'],
        root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs.workflows'))
    for relative,sha in sources.items():
        target=output/'source_snapshots'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
        if file_sha256(target)!=sha:
            raise RuntimeError('source changed during registration')
    registration=dict(spec=spec,checkpoint_bytes=checkpoint.stat().st_size,
        memory_estimate=estimate,sources=sources,new_training=0,new_samples=0,new_ED=0,
        phase_claim=False,fixed_observable='physical prefix raising commutator, leading-vector Casimir and optional spatial Sz groups')
    digest=_canonical(registration)
    registration['registration_sha256']=digest
    _atomic_json(output/'registration.json',registration)
    if estimate['estimated_total_bytes']>int(spec['memory_gib']*GIB):
        result=dict(status='preflight_refused',reason='dense_memory_estimate_exceeds_cap',
                    registration_sha256=digest,memory_estimate=estimate)
    else:
        request=dict(spec=spec,registration_sha256=digest,
                     worker_result=str((output/'worker_result.json').resolve()))
        _atomic_json(output/'request.json',request)
        env=os.environ.copy()
        for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
            env[name]=str(spec['threads'])
        env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
        supervisor=bounded_subprocess(
            [sys.executable,'-m','projects.clebsch_tree_nqs.workflows.mps_rotation_covariance_probe',
             '--worker',str((output/'request.json').resolve())],
            wall_seconds=spec['wall_seconds'],memory_bytes=int(spec['memory_gib']*GIB),
            log_path=output/'worker.log',env=env)
        result=dict(status='failed',registration_sha256=digest,supervisor=supervisor)
        child=output/'worker_result.json'
        if supervisor['exit_code']==0 and child.exists():
            record=json.loads(child.read_text())
            if record.get('record_sha256')!=_canonical({k:v for k,v in record.items() if k!='record_sha256'}) or record['registration_sha256']!=digest:
                raise RuntimeError('worker seal differs from registration')
            result.update(status='complete',worker_result_sha256=file_sha256(child),readout=record)
    result['result_sha256']=_canonical(result)
    _atomic_json(output/'results.json',result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker is not None:
        _worker(args.worker)
    elif args.config is not None and args.output is not None:
        result=run_probe(args.config,args.output)
        print(json.dumps(dict(status=result['status'],result_sha256=result['result_sha256'])))
    else:
        parser.error('supply --config and --output, or internal --worker')


if __name__=='__main__':
    main()
