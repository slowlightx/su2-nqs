"""Bounded physical magnetic-charge Schmidt spectrum of a sealed finite MPS.

Reuse the established TeNPy environment and resource supervisor. The result
contains actual eigenvalues, without modifying/canonicalizing the parent.
Run ``python -m ...mps_schmidt_probe --config CONFIG --output FRESH_OUTPUT``.
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
    request = json.loads(request_path.read_text())
    spec = request['spec']
    memory = int(spec['memory_gib']*GIB)
    if sys.platform.startswith('linux'):
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = memory if hard == resource.RLIM_INFINITY else min(memory, hard)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    _start_rss_watchdog(memory)
    start = time.monotonic()
    import h5py
    import numpy as np
    import tenpy
    from tenpy.tools import hdf5_io
    from su2_nqs.benchmarks.mps_schmidt_spectrum import finite_mps_prefix_schmidt_spectrum
    checkpoint = Path(spec['checkpoint'])
    if file_sha256(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('checkpoint changed before loading')
    with h5py.File(checkpoint, 'r') as handle:
        if 'tenpy_version' in handle:
            stored = handle['tenpy_version'][()]
            stored = stored.decode() if isinstance(stored, bytes) else str(stored)
            if stored != tenpy.__version__:
                raise ValueError('checkpoint and worker TeNPy versions differ')
        psi = hdf5_io.Hdf5Loader(handle).load(spec.get('psi_hdf_group', 'psi'))
    if (psi.L != spec['n_sites'] or psi.bc != 'finite'
            or max(psi.chi) > spec['chi_upper_bound']):
        raise ValueError('finite MPS differs from the registered size/bond cap')
    before = _fingerprint(psi)
    loaded = time.monotonic()
    spectrum = finite_mps_prefix_schmidt_spectrum(psi, spec['cut'],
                                               tolerance=spec['tolerance'])
    measured = time.monotonic()
    after = _fingerprint(psi)
    if before != after or file_sha256(checkpoint) != spec['checkpoint_sha256']:
        raise RuntimeError('measurement changed its physical parent or checkpoint')
    record = dict(status='complete', registration_sha256=request['registration_sha256'],
        checkpoint_sha256=spec['checkpoint_sha256'], spec=spec, actual_chi=max(psi.chi),
        readout=spectrum, parent_fingerprint_before=before, parent_fingerprint_after=after,
        load_and_input_seal_seconds=loaded-start,
        full_environment_and_eigensolve_seconds=measured-loaded,
        postprocessing_and_verification_seconds=time.monotonic()-measured,
        worker_peak_rss_bytes=_own_peak_rss_bytes(),
        runtime=dict(python=sys.version, tenpy=tenpy.__version__, numpy=np.__version__),
        new_training=0, new_samples=0, new_ED=0, phase_claim=False,
        actual_magnetic_charge_schmidt_spectrum_measured=True,
        strict_su2_irrep_spectrum_measured=False)
    record['record_sha256'] = _canonical(record)
    _atomic_json(Path(request['worker_result']), record)


def run_probe(config_path: Path, output: Path) -> dict:
    """Seal input/source; run one eigenspectrum child inside fixed limits."""
    spec = _validated_spec(json.loads(config_path.read_text()))
    spec['renyi2'] = False
    checkpoint = Path(spec['checkpoint']).resolve()
    if file_sha256(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('checkpoint does not match registered SHA256')
    spec['checkpoint'] = str(checkpoint)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('completed spectra are immutable; use a fresh output')
    output.mkdir(parents=True, exist_ok=True)
    from su2_nqs.benchmarks.mps_spin_rotation import estimate_prefix_rotation_memory
    estimate = estimate_prefix_rotation_memory(spec['chi_upper_bound'], checkpoint.stat().st_size)
    sources = transitive_local_source_manifest(
        [Path(__file__), ROOT/'su2_nqs/benchmarks/mps_schmidt_spectrum.py',
         ROOT/'projects/clebsch_tree_nqs/workflows/mps_spin_rotation_probe.py'],
        root=ROOT, package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs.workflows'))
    for relative, sha in sources.items():
        target = output/'source_snapshots'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
        if file_sha256(target) != sha:
            raise RuntimeError('source changed during registration')
    registration = dict(spec=spec, checkpoint_bytes=checkpoint.stat().st_size,
        memory_estimate=estimate, sources=sources, new_training=0, new_samples=0,
        new_ED=0, phase_claim=False, fixed_observable='physical prefix m=2Sz Schmidt eigenvalues')
    digest = _canonical(registration)
    registration['registration_sha256'] = digest
    _atomic_json(output/'registration.json', registration)
    if estimate['estimated_total_bytes'] > int(spec['memory_gib']*GIB):
        result = dict(status='preflight_refused', reason='dense_memory_estimate_exceeds_cap',
                      registration_sha256=digest, memory_estimate=estimate)
    else:
        request = dict(spec=spec, registration_sha256=digest,
                       worker_result=str((output/'worker_result.json').resolve()))
        _atomic_json(output/'request.json', request)
        env = os.environ.copy()
        for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[name] = str(spec['threads'])
        env['PYTHONPATH'] = str(ROOT)+os.pathsep+env.get('PYTHONPATH', '')
        supervisor = bounded_subprocess(
            [sys.executable, '-m', 'projects.clebsch_tree_nqs.workflows.mps_schmidt_probe',
             '--worker', str((output/'request.json').resolve())],
            wall_seconds=spec['wall_seconds'], memory_bytes=int(spec['memory_gib']*GIB),
            log_path=output/'worker.log', env=env)
        result = dict(status='failed', registration_sha256=digest, supervisor=supervisor)
        child = output/'worker_result.json'
        if supervisor['exit_code'] == 0 and child.exists():
            record = json.loads(child.read_text())
            if (record.get('record_sha256') != _canonical({k:v for k,v in record.items() if k!='record_sha256'})
                    or record['registration_sha256'] != digest):
                raise RuntimeError('worker seal differs from registration')
            result.update(status='complete', worker_result_sha256=file_sha256(child), readout=record)
    result['result_sha256'] = _canonical(result)
    _atomic_json(output/'results.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        _worker(args.worker)
    elif args.config is not None and args.output is not None:
        result = run_probe(args.config, args.output)
        print(json.dumps(dict(status=result['status'], result_sha256=result['result_sha256'])))
    else:
        parser.error('supply --config and --output, or internal --worker')


if __name__ == '__main__':
    main()
