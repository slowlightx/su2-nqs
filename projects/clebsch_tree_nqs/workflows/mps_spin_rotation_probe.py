"""One bounded physical-prefix FCS contraction; unchanged finite MPS parent.

Run with the TeNPy Python: ``-m ...mps_spin_rotation_probe --config CONFIG
--output FRESH_DIRECTORY``. The supervisor sets wall/RSS limits; Linux also
sets an address-space limit. No optimizer, JAX, or canonicalization is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import threading
import time

from projects.clebsch_tree_nqs.workflows.dmrg_archive import file_sha256
from projects.clebsch_tree_nqs.workflows.dmrg_region_probe import (
    GIB, ROOT, _atomic_json, _canonical, _mps_fingerprint,
    bounded_subprocess, transitive_local_source_manifest,
)


def _validated_spec(raw: dict) -> dict:
    spec = dict(raw)
    for name in ('n_sites','chi_upper_bound','cut','threads'):
        value = spec[name]
        if isinstance(value,bool) or not isinstance(value,int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if spec['cut'] >= spec['n_sites']:
        raise ValueError('the measurement cut must be internal')
    for name in ('wall_seconds','memory_gib'):
        value = spec[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not 0 < value < float('inf'):
            raise ValueError(f'{name} must be positive and finite')
    if not re.fullmatch('[0-9a-f]{64}',spec['checkpoint_sha256']):
        raise ValueError('an explicit immutable checkpoint SHA256 is required')
    if not isinstance(spec.get('psi_hdf_group','psi'),str):
        raise ValueError('psi_hdf_group must be a string')
    # A fixed complete curve, not a data-selected angle. Its finite shape is
    # deliberately not a symmetry-fractionalization certificate.
    spec['angle_count'] = spec.get('angle_count',81)
    if (isinstance(spec['angle_count'],bool) or not isinstance(spec['angle_count'],int)
            or not 3 <= spec['angle_count'] <= 1001 or spec['angle_count'] % 2 != 1):
        raise ValueError('odd angle_count in 3..1001 ensures explicit pi and 2pi checks')
    spec['tolerance'] = spec.get('tolerance',1e-9)
    if not isinstance(spec['tolerance'],(int,float)) or not 0 < spec['tolerance'] < 1:
        raise ValueError('physicality tolerance must be between zero and one')
    spec['renyi2'] = spec.get('renyi2',False)
    if not isinstance(spec['renyi2'],bool):
        raise ValueError('renyi2 must be an explicit Boolean')
    return spec


def _fingerprint(psi) -> str:
    """Stored numeric blocks plus all gauge-charge metadata; no densification."""
    digest = hashlib.sha256(_mps_fingerprint(psi).encode())
    for b in psi._B:
        digest.update(repr(b.qtotal.tolist()).encode())
        for leg in b.legs:
            digest.update(repr((leg.qconj,leg.charges.tolist(),leg.slices.tolist())).encode())
    return digest.hexdigest()


def _own_peak_rss_bytes() -> int:
    """Platform-correct OS peak RSS; does not need a permitted ``ps`` call."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


def _start_rss_watchdog(memory_bytes: int) -> None:
    """Reject an unavailable metric; kill only this worker if cap is crossed.

    Complements the external RSS monitor, which a macOS sandbox may deny.
    Peak RSS is conservative (never smaller than current RSS). Linux retains
    RLIMIT_AS as an independent allocation guard. Checks occur every 10 ms;
    instantaneous allocations may overshoot before the OS-sampled kill.
    """
    if _own_peak_rss_bytes() <= 0:
        raise RuntimeError('worker cannot observe RSS; no unbounded fallback')
    def check() -> None:
        while True:
            if _own_peak_rss_bytes() > memory_bytes:
                os.kill(os.getpid(),signal.SIGKILL)
            time.sleep(.01)
    threading.Thread(target=check,name='prefix-fcs-rss-guard',daemon=True).start()


def _worker(request_path: Path) -> None:
    request = json.loads(request_path.read_text())
    spec = request['spec']
    memory = int(spec['memory_gib']*GIB)
    if sys.platform.startswith('linux'):
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = memory if hard == resource.RLIM_INFINITY else min(memory,hard)
        resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    _start_rss_watchdog(memory)
    start = time.monotonic()
    import h5py
    import numpy as np
    import tenpy
    from tenpy.tools import hdf5_io
    from su2_nqs.benchmarks.mps_spin_rotation import (
        finite_mps_prefix_charge_distribution, physical_spin_rotation_summary,
    )
    checkpoint = Path(spec['checkpoint'])
    if file_sha256(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('checkpoint changed before loading')
    with h5py.File(checkpoint,'r') as handle:
        if 'tenpy_version' in handle:
            stored = handle['tenpy_version'][()]
            stored = stored.decode() if isinstance(stored,bytes) else str(stored)
            if stored != tenpy.__version__:
                raise ValueError('checkpoint and worker TeNPy versions differ')
        psi = hdf5_io.Hdf5Loader(handle).load(spec.get('psi_hdf_group','psi'))
    if (psi.L != spec['n_sites'] or psi.bc != 'finite'
            or max(psi.chi) > spec['chi_upper_bound']):
        raise ValueError('loaded finite MPS differs from the registered size/bond cap')
    before = _fingerprint(psi)
    loaded = time.monotonic()
    include_renyi2 = bool(request.get('measure_renyi2',False))
    m,p,details = finite_mps_prefix_charge_distribution(
        psi,spec['cut'],tolerance=spec['tolerance'],renyi2=include_renyi2)
    measured = time.monotonic()
    angles = np.linspace(0,2*np.pi,spec['angle_count'])
    summary = physical_spin_rotation_summary(m,p,n_region=spec['cut'],angles=angles,
                                             tolerance=spec['tolerance'])
    after = _fingerprint(psi)
    if before != after or file_sha256(checkpoint) != spec['checkpoint_sha256']:
        raise RuntimeError('measurement changed its physical parent or input checkpoint')
    record = dict(status='complete',registration_sha256=request['registration_sha256'],
                  checkpoint_sha256=spec['checkpoint_sha256'],spec=spec,
                  actual_chi=max(psi.chi),actual_prefix=details,readout=summary,
                  parent_fingerprint_before=before,parent_fingerprint_after=after,
                  load_and_input_seal_seconds=loaded-start,
                  full_environment_and_charge_seconds=measured-loaded,
                  postprocessing_and_verification_seconds=time.monotonic()-measured,
                  worker_peak_rss_bytes=_own_peak_rss_bytes(),
                  worker_rss_watchdog_observed=True,worker_rss_watchdog_interval_seconds=.01,
                  address_space_guard=sys.platform.startswith('linux'),
                  runtime=dict(python=sys.version,tenpy=tenpy.__version__,numpy=np.__version__),
                  new_training=0,new_samples=0,new_ED=0,phase_claim=False,
                  renyi2_measured=include_renyi2,
                  physical_prefix_not_single_anyon_boundary=True,
                  strict_su2_schmidt_spectrum_measured=False)
    if include_renyi2:
        charges = np.asarray(details['renyi2']['magnetic_labels'])
        sector_purities = np.asarray(details['renyi2']['sector_purities'])
        charged = np.exp(.5j*np.outer(angles,charges)) @ sector_purities
        record['charged_renyi2'] = dict(
            angles=angles.tolist(),moment_real=charged.real.tolist(),
            moment_imaginary=charged.imag.tolist(),
            normalization=float(sector_purities.sum()),
            normalized_moment_real=(charged.real/sector_purities.sum()).tolist(),
            normalized_moment_imaginary=(charged.imag/sector_purities.sum()).tolist(),
            observable='Tr(rho_A**2 exp(i theta S_A^z))',
            contains_off_diagonal_coherence=True,qsl_specific=False,
            warning='Magnetic charge-resolved Renyi-2 moment, not a strict SU(2) irrep or anyon-sector spectrum.')
    record['record_sha256'] = _canonical(record)
    _atomic_json(Path(request['worker_result']),record)


def run_probe(config_path: Path, output: Path) -> dict:
    """Register fresh immutable input/sources and run at most one contraction."""
    spec = _validated_spec(json.loads(config_path.read_text()))
    checkpoint = Path(spec['checkpoint']).resolve()
    if file_sha256(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('checkpoint does not match registered byte SHA256')
    spec['checkpoint'] = str(checkpoint)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use a fresh output; completed probes are immutable')
    output.mkdir(parents=True,exist_ok=True)
    from su2_nqs.benchmarks.mps_spin_rotation import estimate_prefix_rotation_memory
    estimate = estimate_prefix_rotation_memory(spec['chi_upper_bound'],checkpoint.stat().st_size)
    sources = transitive_local_source_manifest(
        [Path(__file__),ROOT/'su2_nqs/benchmarks/mps_spin_rotation.py',
         ROOT/'projects/clebsch_tree_nqs/workflows/dmrg_region_probe.py',
         ROOT/'projects/clebsch_tree_nqs/workflows/dmrg_archive.py'],
        root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs.workflows'))
    for relative,sha in sources.items():
        target = output/'source_snapshots'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
        if file_sha256(target) != sha:
            raise RuntimeError('source changed during registration')
    registration = dict(spec=spec,checkpoint_bytes=checkpoint.stat().st_size,
                        memory_estimate=estimate,sources=sources,
                        new_training=0,new_samples=0,new_ED=0,phase_claim=False,
                        fixed_observable='actual physical prefix doubled-Sz FCS')
    digest = _canonical(registration)
    registration['registration_sha256'] = digest
    _atomic_json(output/'registration.json',registration)
    if estimate['estimated_total_bytes'] > int(spec['memory_gib']*GIB):
        result = dict(status='preflight_refused',reason='dense_memory_estimate_exceeds_cap',
                      registration_sha256=digest,memory_estimate=estimate)
    else:
        request = dict(spec=spec,registration_sha256=digest,measure_renyi2=False,
                       worker_result=str((output/'worker_result.json').resolve()))
        _atomic_json(output/'request.json',request)
        env = os.environ.copy()
        for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
            env[name] = str(spec['threads'])
        env['PYTHONPATH'] = str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
        supervisor = bounded_subprocess(
            [sys.executable,'-m','projects.clebsch_tree_nqs.workflows.mps_spin_rotation_probe',
             '--worker',str((output/'request.json').resolve())],
            wall_seconds=spec['wall_seconds'],memory_bytes=int(spec['memory_gib']*GIB),
            log_path=output/'worker.log',env=env)
        child = output/'worker_result.json'
        result = dict(status='failed',registration_sha256=digest,supervisor=supervisor)
        if supervisor['exit_code'] == 0 and child.exists():
            record = json.loads(child.read_text())
            if (record.get('record_sha256') != _canonical({k:v for k,v in record.items() if k!='record_sha256'})
                    or record['registration_sha256'] != digest):
                raise RuntimeError('worker result seal differs from immutable registration')
            result.update(status='complete',worker_result_sha256=file_sha256(child),readout=record)
        if result['status'] == 'complete' and spec['renyi2']:
            # A separate child guarantees an optional cubic Renyi calculation
            # cannot erase the completed n=1 FCS if it reaches a resource cap.
            remaining = spec['wall_seconds']-supervisor['supervisor_seconds']
            result['renyi2_attempt'] = dict(status='budget_exhausted_before_start')
            if remaining > 0:
                request2 = dict(spec=spec,registration_sha256=digest,measure_renyi2=True,
                    worker_result=str((output/'worker_renyi2_result.json').resolve()))
                _atomic_json(output/'request_renyi2.json',request2)
                supervision2 = bounded_subprocess(
                    [sys.executable,'-m','projects.clebsch_tree_nqs.workflows.mps_spin_rotation_probe',
                     '--worker',str((output/'request_renyi2.json').resolve())],
                    wall_seconds=remaining,memory_bytes=int(spec['memory_gib']*GIB),
                    log_path=output/'worker_renyi2.log',env=env)
                child2 = output/'worker_renyi2_result.json'
                result['renyi2_attempt'] = dict(status='failed',supervisor=supervision2)
                if supervision2['exit_code'] == 0 and child2.exists():
                    record2 = json.loads(child2.read_text())
                    if (record2.get('record_sha256') != _canonical({k:v for k,v in record2.items() if k!='record_sha256'})
                            or record2['registration_sha256'] != digest
                            or not record2['renyi2_measured']):
                        raise RuntimeError('optional Renyi worker seal differs from registered input')
                    result['renyi2_attempt'].update(status='complete',
                        worker_result_sha256=file_sha256(child2),readout=record2)
    result['result_sha256'] = _canonical(result)
    _atomic_json(output/'results.json',result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        _worker(args.worker)
    elif args.config is not None and args.output is not None:
        result = run_probe(args.config,args.output)
        print(json.dumps(dict(status=result['status'],result_sha256=result['result_sha256'])))
    else:
        parser.error('supply --config and --output, or internal --worker')


if __name__ == '__main__':
    main()
