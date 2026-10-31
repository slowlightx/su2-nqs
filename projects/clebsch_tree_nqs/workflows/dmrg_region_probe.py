"""Bounded, restartable fixed-MPS RDM and observable acquisition.

No optimizer, JAX, or symmetry repair is used. Optional energy/variance tasks
use the declared Hamiltonian MPO without changing the state. Each task runs
in its own TeNPy subprocess with wallclock/address-space/RSS limits. Completed
records are immutable and hash-verified on resume; a failed task cannot erase
a previous successful one.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence
from numbers import Integral

from projects.clebsch_tree_nqs.workflows.dmrg_archive import file_sha256

ROOT = Path(__file__).resolve().parents[3]
# Import the existing stdlib-only provenance utility without frontier/__init__,
# which eagerly imports optional JAX training integrations.
_PROVENANCE_PATH = ROOT/'su2_nqs/frontier/provenance.py'
_LOADER = importlib.util.spec_from_file_location('dmrg_region_provenance', _PROVENANCE_PATH)
_PROVENANCE = importlib.util.module_from_spec(_LOADER)
_LOADER.loader.exec_module(_PROVENANCE)
transitive_local_source_manifest = _PROVENANCE.transitive_local_source_manifest
GIB = 1024**3
TRIANGULAR_REGIONS = (
    ('local_hexagon6', (20, 21, 15, 8, 7, 13)),
    ('local_hexagon7', (20, 21, 15, 8, 7, 13, 14)),
)


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    """Replace only our small control records using a private fsynced file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.region-', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _positive(value: object, name: str, *, integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    if integer and int(value) != value:
        raise ValueError(f'{name} must be an integer')
    return int(value) if integer else float(value)


def estimate_region_memory(sites: Sequence[int], chi: int, *, itemsize: int,
                           checkpoint_bytes: int = 0, transient_factor: float = 3.,
                           reserve_bytes: int = 512*1024**2) -> dict[str, Any]:
    """Conservative DENSE estimate for the unchanged TeNPy get_rho_segment.

    Noncontiguous cuts keep up to k-1 open bra/ket sites, giving 4**(k-1)
    virtual matrices. Consecutive cuts instead form a 2**k-site theta.
    Charge sparsity can reduce storage; this estimate is NOT a guarantee of
    measured usage or throughput. A registered explicit override may bypass
    its refusal, never the actual process limits.
    """
    ordered = sorted(sites)
    k = len(ordered)
    if (not 1 <= k <= 8 or len(set(ordered)) != k or min(ordered) < 0
            or any(isinstance(i, bool) or not isinstance(i, Integral) for i in ordered)):
        raise ValueError('estimate requires 1..8 unique nonnegative sites')
    chi = int(_positive(chi, 'chi', integer=True))
    if itemsize not in (8, 16) or checkpoint_bytes < 0 or reserve_bytes < 0:
        raise ValueError('only float64/complex128 estimates and nonnegative overhead are supported')
    factor = float(_positive(transient_factor, 'transient_factor'))
    if factor < 2:
        raise ValueError('transient_factor must be at least two')
    consecutive = ordered == list(range(ordered[0], ordered[-1]+1))
    principal = itemsize*chi**2*(2**k if consecutive else 4**(k-1))
    output = itemsize*4**k
    total = math.ceil(factor*max(principal, output)+2*checkpoint_bytes+reserve_bytes)
    return dict(consecutive=consecutive, region_size=k, sorted_span=ordered[-1]-ordered[0]+1,
                chi_upper_bound=chi, itemsize=itemsize, principal_dense_bytes=principal,
                estimated_total_bytes=total, transient_factor=factor,
                sparsity_assumed=False, estimate_is_hard_guarantee=False)


def _settings(spec: dict) -> list[dict]:
    n = int(_positive(spec['n_sites'], 'n_sites', integer=True))
    int(_positive(spec['chi_upper_bound'], 'chi_upper_bound', integer=True))
    if spec.get('dtype', 'complex128') not in ('float64', 'complex128'):
        raise ValueError('dtype must conservatively declare float64 or complex128')
    if not re.fullmatch(r'[0-9a-f]{64}', spec['checkpoint_sha256']):
        raise ValueError('an explicit sealed checkpoint SHA256 is required')
    defaults = spec.get('limits', {})
    rows = spec.get('regions')
    if rows is None:
        if n != 36:
            raise ValueError('default triangular hexagon sites require N=36')
        rows = [dict(name=name, sites=list(sites)) for name, sites in TRIANGULAR_REGIONS]
    if not isinstance(rows, list) or not rows:
        raise ValueError('regions must be a nonempty ordered list')
    result, names = [], set()
    for row in rows:
        name, sites = row['name'], row['sites']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in names:
            raise ValueError('unique safe region names required')
        if (not isinstance(sites, list) or not 1 <= len(sites) <= 8
                or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n for i in sites)
                or len(set(sites)) != len(sites)):
            raise ValueError('each region needs 1..8 unique in-range integer sites in physical order')
        names.add(name)
        limits = {**defaults, **row.get('limits', {})}
        wall = float(_positive(limits['wall_seconds'], 'wall_seconds'))
        memory = float(_positive(limits['memory_gib'], 'memory_gib'))
        threads = int(_positive(limits.get('threads', 1), 'threads', integer=True))
        tolerance = float(_positive(row.get('tolerance', 1e-9), 'tolerance'))
        if not isinstance(row.get('dense_estimate_override', False), bool):
            raise ValueError('dense_estimate_override must be an explicit Boolean')
        result.append(dict(name=name, sites=sites, limits=dict(wall_seconds=wall,
                           memory_gib=memory, threads=threads), tolerance=tolerance,
                           dense_estimate_override=bool(row.get('dense_estimate_override', False))))
        if 'observable' in row:
            observable = row['observable']
            if not isinstance(observable, dict) or observable.get('kind') not in {
                    'spin_correlations', 'dimer_correlations', 'hamiltonian_variance'}:
                raise ValueError('explicit fixed-MPS observable kind required')
            # sites=[0] is a supervisor anchor, NOT the observable support.
            if sites != [0]:
                raise ValueError('global observable tasks use the explicit [0] supervisor anchor')
            result[-1]['observable'] = observable
    return result


def _rss_bytes(pid: int) -> int:
    """Observe RSS without importing numeric libraries into the supervisor."""
    if sys.platform.startswith('linux'):
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    return int(line.split()[1])*1024
        except (FileNotFoundError, ProcessLookupError):
            return 0
    else:
        try:
            return int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(pid)],
                                               text=True, timeout=1).strip() or 0)*1024
        except (subprocess.SubprocessError, ValueError, OSError):
            return 0
    return 0


def bounded_subprocess(command: Sequence[str], *, wall_seconds: float,
                       memory_bytes: int, log_path: Path, env: dict[str, str]) -> dict:
    """Supervise one process group, killing only that group at a fixed cap."""
    wall = float(_positive(wall_seconds, 'wall_seconds'))
    memory = int(_positive(memory_bytes, 'memory_bytes', integer=True))
    start, peak, reason = time.monotonic(), 0, None
    with log_path.open('ab') as log:
        child = subprocess.Popen(list(command), stdout=log, stderr=log,
                                 env=env, start_new_session=True)
        while child.poll() is None:
            peak = max(peak, _rss_bytes(child.pid))
            if time.monotonic()-start >= wall:
                reason = 'wallclock_limit'
            elif peak > memory:
                reason = 'rss_limit'
            if reason is not None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(.025)
        code = child.wait()
    return dict(exit_code=code, supervisor_seconds=time.monotonic()-start,
                supervisor_peak_rss_bytes=peak, supervisor_rss_monitor_observed=peak>0,
                supervisor_stop_reason=reason)


def _mps_fingerprint(psi: Any) -> str:
    """Fingerprint stored MPS arrays without densifying charge-block tensors."""
    import numpy as np

    digest = hashlib.sha256()
    digest.update(repr((psi.L, psi.bc, psi.norm, psi.form)).encode())
    def add(array: Any) -> None:
        blocks = array._data if hasattr(array, '_data') else [array]
        for block in blocks:
            value = np.asarray(block)
            digest.update(repr((value.shape, value.dtype.str)).encode())
            digest.update(memoryview(np.ascontiguousarray(value)).cast('B'))
    for tensor in psi._B:
        digest.update(repr((tensor.get_leg_labels(), tensor.shape)).encode())
        add(tensor)
    for singular in psi._S:
        add(singular)
    return digest.hexdigest()


def raw_rdm_diagnostics(rho: Any, *, tolerance: float) -> dict:
    """Preserve even a finite invalid estimator, with no fabricated spectrum."""
    import numpy as np
    from su2_nqs.benchmarks.tensor_network_readout import physical_spin_block_readout

    matrix = np.asarray(rho, dtype=np.complex128)
    tolerance = float(_positive(tolerance, 'tolerance'))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError('raw RDM must be finite and square')
    try:
        result = physical_spin_block_readout(matrix, tolerance=tolerance)
        result.pop('invariant_readout', None)
        result['physical_matrix_valid'] = True
    except ValueError as error:
        result = dict(physical_matrix_valid=False, diagnostic_error=str(error),
            trace_error=float(abs(np.trace(matrix)-1)),
            hermiticity_frobenius=float(np.linalg.norm(matrix-matrix.conj().T)),
            minimum_eigenvalue_of_hermitian_part=float(np.linalg.eigvalsh((matrix+matrix.conj().T)/2).min()),
            strict_su2_eligible=False,
            warning='Raw invalid matrix retained; no normalized spin probability or entanglement spectrum interpretation.')
    result.update(strict_entanglement_spectrum_reported=False, input_repaired=False,
                  numerical_tolerance=tolerance, raw_trace_real=float(np.trace(matrix).real),
                  raw_trace_imaginary=float(np.trace(matrix).imag), raw_frobenius_norm=float(np.linalg.norm(matrix)),
                  strict_su2_eligibility_scope='Numerical regional factorization/physicality at the stated tolerance, not an exact global-S0 certificate.')
    return result


def _worker(request_path: Path) -> None:
    request = json.loads(request_path.read_text())
    limit = int(request['region']['limits']['memory_gib']*GIB)
    # RLIMIT_AS covers NumPy/BLAS allocations, unlike Python allocation tracers.
    # The independent supervisor additionally enforces measured RSS.
    address_limit = None
    if sys.platform.startswith('linux'):
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        address_limit = min(limit, hard) if hard != resource.RLIM_INFINITY else limit
        resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))
    elif _rss_bytes(os.getpid()) <= 0:
        # Darwin does not implement RLIMIT_AS. Never silently disable both
        # mechanisms when its process monitor is unavailable in a sandbox.
        raise RuntimeError('macOS worker needs an available RSS monitor; no unbounded fallback')
    start = time.monotonic()
    import h5py
    import numpy as np
    import tenpy
    from tenpy.tools import hdf5_io
    from su2_nqs.benchmarks.tensor_network_readout import tenpy_region_density

    checkpoint = Path(request['checkpoint'])
    before = checkpoint.stat()
    if file_sha256(checkpoint) != request['checkpoint_sha256']:
        raise ValueError('checkpoint differs from the frozen input before load')
    with h5py.File(checkpoint, 'r') as handle:
        # Load ONLY psi, not the entire engine, Lanczos cache or optimizer.
        if 'tenpy_version' in handle:
            version = handle['tenpy_version'][()]
            version = version.decode() if isinstance(version, bytes) else str(version)
            if version != tenpy.__version__:
                raise ValueError('checkpoint TeNPy version differs from this worker')
        psi = hdf5_io.Hdf5Loader(handle).load(request.get('psi_hdf_group', 'psi'))
    if psi.bc != 'finite' or psi.L != request['n_sites']:
        raise ValueError('only the declared finite MPS is supported')
    if any(form is None for form in psi.form):
        raise ValueError('input does not declare canonical tensor forms; never canonicalize the fixed parent here')
    chi = max(psi.chi, default=1)
    dtype = np.dtype(psi.dtype)
    declared = np.dtype(request['dtype'])
    if chi > request['chi_upper_bound'] or dtype.itemsize > declared.itemsize:
        raise ValueError('input exceeds the registered chi/dtype preflight')
    fingerprint = _mps_fingerprint(psi)
    loaded_seconds = time.monotonic()-start
    rho_start = time.monotonic()
    observable = request['region'].get('observable')
    if observable is None:
        rho = tenpy_region_density(psi, request['region']['sites'])
        payload = dict(rho=rho, sites=np.asarray(request['region']['sites'], np.int64))
        diagnostics = raw_rdm_diagnostics(rho, tolerance=request['region']['tolerance'])
    else:
        from su2_nqs.benchmarks.tensor_network_observables import tenpy_spin_correlations, tenpy_dimer_correlations
        if observable['kind'] == 'spin_correlations':
            diagnostics, correlations = tenpy_spin_correlations(psi, observable['rows'], observable['columns'])
            payload = dict(spin_correlations=correlations)
        elif observable['kind'] == 'dimer_correlations':
            means, connected = tenpy_dimer_correlations(psi, observable['bonds'])
            payload = dict(bond_means=means, connected=connected,
                           bonds=np.asarray(observable['bonds'], np.int64))
            diagnostics = dict(maximum_imaginary_bond_mean=float(np.max(abs(means.imag))),
                               maximum_imaginary_connected=float(np.max(abs(connected.imag))))
        else:
            from su2_nqs.benchmarks.tensor_network_readout import tenpy_graph_model
            model = tenpy_graph_model(psi.L, [tuple(b) for b in observable['physical_bonds']])
            energy = complex(model.H_MPO.expectation_value(psi))
            variance = complex(model.H_MPO.variance(psi, exp_val=energy))
            payload = dict(energy=np.asarray(energy), variance=np.asarray(variance))
            diagnostics = dict(energy_per_site=energy.real/psi.L,
                variance_per_site=variance.real/psi.L, energy_imaginary=energy.imag,
                variance_imaginary=variance.imag, variance_clipped=False,
                mpo_bond_dimensions=list(map(int,model.H_MPO.chi)))
        diagnostics.update(strict_entanglement_spectrum_reported=False,
            input_repaired=False, support='Declared global observable, not anchor site [0].')
    rdm_seconds = time.monotonic()-rho_start
    if _mps_fingerprint(psi) != fingerprint:
        raise RuntimeError('regional readout changed the stored MPS tensors')
    after = checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or file_sha256(checkpoint) != request['checkpoint_sha256']:
        raise RuntimeError('fixed input checkpoint changed during the readout')
    if 'jax' in sys.modules:
        raise RuntimeError('standalone TeNPy region probe unexpectedly imported JAX')
    directory = Path(request['directory'])
    candidate = directory/request['candidate_prefix']
    fd, name = tempfile.mkstemp(prefix='.rdm-', suffix='.npz', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as handle:
            np.savez_compressed(handle, **payload)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, candidate.with_suffix('.npz'))
    finally:
        if os.path.exists(name):
            os.unlink(name)
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = dict(status='complete', registration_sha256=request['registration_sha256'],
        region=request['region'], checkpoint_sha256=request['checkpoint_sha256'],
        rdm_sha256=file_sha256(candidate.with_suffix('.npz')), diagnostics=diagnostics,
        actual_chi=chi, actual_dtype=dtype.name, tenpy_version=tenpy.__version__,
        mps_fingerprint=fingerprint, mps_unchanged=True, checkpoint_unchanged=True,
        norm_attribute_before_after=float(abs(psi.norm)),
        load_and_fingerprint_seconds=loaded_seconds, raw_rdm_seconds=rdm_seconds,
        worker_seconds=time.monotonic()-start,
        worker_peak_rss_bytes=int(maximum if sys.platform=='darwin' else maximum*1024),
        address_space_limit_bytes=address_limit, requested_memory_cap_bytes=limit,
        memory_enforcement='RLIMIT_AS_and_supervisor_RSS' if address_limit is not None else 'supervisor_RSS_watchdog',
        raw_rdm_unmodified=True, no_jax_import=True,
        payload_kind='raw_rdm' if observable is None else 'fixed_mps_observables',
        new_optimization=0, new_H_calls=2 if observable is not None
            and observable['kind']=='hamiltonian_variance' else 0, new_ED=0,
        conventions='Requested site order; first requested site is MSB; bit0=down, bit1=up.',
        warning='Uses declared canonical MPS forms; trace/raw diagnostics are not a global ground-state or convergence certificate.')
    # Only the supervisor publishes a successful cut, after a clean child exit.
    # A child killed after writing this candidate never leaves a completed cut.
    _atomic_json(candidate.with_suffix('.json'), record)


def _collect_regions(config: Path, output: Path | None = None) -> dict:
    """Register sealed inputs, sequentially collect cuts, and resume successes."""
    config = config.resolve()
    spec = json.loads(config.read_text())
    regions = _settings(spec)
    checkpoint = (ROOT/spec['checkpoint']).resolve()
    expected = spec['checkpoint_sha256']
    if file_sha256(checkpoint) != expected:
        raise ValueError('input checkpoint SHA does not match the preregistration')
    output = (output or ROOT/spec['output']).resolve()
    if output == checkpoint.parent or output.is_relative_to(checkpoint.parent):
        raise ValueError('readout output must be outside the immutable checkpoint directory')
    inputs = {str(checkpoint): expected, str(config): file_sha256(config)}
    if spec.get('archive_receipt'):
        receipt = (ROOT/spec['archive_receipt']).resolve()
        if file_sha256(receipt) != spec['archive_receipt_sha256']:
            raise ValueError('archive receipt seal mismatch')
        if json.loads(receipt.read_text())['checkpoint_sha256'] != expected:
            raise ValueError('archive receipt belongs to a different MPS generation')
        inputs[str(receipt)] = spec['archive_receipt_sha256']
    sources = transitive_local_source_manifest((Path(__file__), _PROVENANCE_PATH), root=ROOT,
                    package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    registration = dict(schema=1, config=spec, config_sha256=file_sha256(config),
        checkpoint=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
        regions=regions, inputs=inputs, sources=sources,
        purpose='Fixed MPS raw RDM acquisition only; no optimization, canonicalization, ED, H calls, or symmetry repair.')
    digest = _canonical(registration)
    output.mkdir(parents=True, exist_ok=True)
    prior = output/'registration.json'
    if prior.exists():
        if json.loads(prior.read_text()) != registration:
            raise ValueError('resume input/config/source registration differs; use a fresh output')
        for relative, source_hash in sources.items():
            if file_sha256(output/'source_snapshots'/relative) != source_hash:
                raise ValueError('frozen source snapshot seal mismatch')
    else:
        if any(path.name != '.region-probe.lock' for path in output.iterdir()):
            raise FileExistsError('nonempty output lacks a region-probe registration')
        for relative, source_hash in sources.items():
            target = output/'source_snapshots'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/relative).read_bytes())
            if file_sha256(target) != source_hash:
                raise RuntimeError('source changed while registering the probe')
        _atomic_json(prior, registration)
    records, start = [], time.monotonic()
    for row in regions:
        directory = output/'regions'/row['name']
        directory.mkdir(parents=True, exist_ok=True)
        sealed = directory/'readout.json'
        if sealed.exists():
            record = json.loads(sealed.read_text())
            if (record['status'] != 'complete' or record['registration_sha256'] != digest
                    or record['region'] != row or file_sha256(directory/'rdm.npz') != record['rdm_sha256']
                    or record.get('record_sha256') != _canonical({k:v for k,v in record.items() if k!='record_sha256'})):
                raise ValueError('completed region seal mismatch; never silently recompute it')
            records.append(record)
            continue
        estimate = estimate_region_memory(row['sites'], spec['chi_upper_bound'],
            itemsize=8 if spec.get('dtype', 'complex128')=='float64' else 16,
            checkpoint_bytes=checkpoint.stat().st_size)
        if 'observable' in row:
            estimate = dict(estimated_total_bytes=2*checkpoint.stat().st_size+512*1024**2,
                estimates_input_loading_only=True, estimates_observable_intermediates=False,
                actual_wallclock_address_space_and_RSS_limits_enforced=True,
                warning='Not a one-site RDM estimate. Global MPO/H2 intermediates may exceed the input-loading estimate.')
        limit = int(row['limits']['memory_gib']*GIB)
        attempt = dict(region=row, estimate=estimate, registration_sha256=digest)
        stamp = f'{time.time_ns()}_{os.getpid()}'
        if estimate['estimated_total_bytes'] > limit and not row['dense_estimate_override']:
            attempt.update(status='preflight_refused', supervisor_stop_reason='dense_memory_estimate_exceeds_cap')
        else:
            request = dict(registration_sha256=digest, checkpoint=str(checkpoint),
                checkpoint_sha256=expected, directory=str(directory), region=row,
                n_sites=spec['n_sites'], chi_upper_bound=spec['chi_upper_bound'],
                dtype=spec.get('dtype', 'complex128'), psi_hdf_group=spec.get('psi_hdf_group', 'psi'),
                candidate_prefix=f'.candidate_{stamp}')
            request_path = directory/f'request_{stamp}.json'
            _atomic_json(request_path, request)
            env = os.environ.copy()
            for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
                env[name] = str(row['limits']['threads'])
            env['PYTHONPATH'] = str(ROOT)+os.pathsep+env.get('PYTHONPATH', '')
            attempt.update(bounded_subprocess([sys.executable, '-m', __name__ if __name__!='__main__'
                else 'projects.clebsch_tree_nqs.workflows.dmrg_region_probe', '--worker', str(request_path)],
                wall_seconds=row['limits']['wall_seconds'], memory_bytes=limit,
                log_path=directory/f'worker_{stamp}.log', env=env))
            candidate = directory/request['candidate_prefix']
            if attempt['exit_code']==0 and candidate.with_suffix('.json').exists():
                record = json.loads(candidate.with_suffix('.json').read_text())
                if (record['registration_sha256'] != digest or record['region'] != row
                        or file_sha256(candidate.with_suffix('.npz')) != record['rdm_sha256']):
                    raise RuntimeError('new worker output seal mismatch')
                record['supervisor'] = {k:v for k,v in attempt.items() if k.startswith('supervisor_') or k=='exit_code'}
                record['record_sha256'] = _canonical(record)
                os.replace(candidate.with_suffix('.npz'), directory/'rdm.npz')
                _atomic_json(sealed, record)
                attempt['status'] = 'complete'
                records.append(record)
            else:
                attempt['status'] = 'failed'
        _atomic_json(directory/f'attempt_{stamp}.json', attempt)
        if attempt['status'] != 'complete':
            records.append(attempt)
            if spec.get('stop_on_failure', True):
                break
    for path, source_hash in inputs.items():
        if file_sha256(Path(path)) != source_hash:
            raise RuntimeError('sealed input changed during acquisition')
    for relative, source_hash in sources.items():
        if file_sha256(ROOT/relative) != source_hash:
            raise RuntimeError('registered source changed during acquisition')
    result = dict(registration_sha256=digest, status='complete' if len(records)==len(regions)
        and all(r['status']=='complete' for r in records) else 'incomplete', regions=records,
        seconds=time.monotonic()-start, new_optimization=0,
        new_H_calls=sum(r.get('new_H_calls',0) for r in records), new_ED=0,
        source_modified=False, physical_state_replaced=False)
    _atomic_json(output/'results.json', result)
    _atomic_json(output/'completion.json', dict(status=result['status'],
        registration_sha256=file_sha256(prior), results_sha256=file_sha256(output/'results.json'),
        regions={row['region']['name']: dict(readout_sha256=file_sha256(output/'regions'/row['region']['name']/'readout.json'),
                    rdm_sha256=row['rdm_sha256']) for row in records if row['status']=='complete'}))
    return result


def collect_regions(config: Path, output: Path | None = None) -> dict:
    """Acquire/resume one fixed-parent probe under an exclusive output lock."""
    spec = json.loads(config.read_text())
    target = (output or ROOT/spec['output']).resolve()
    checkpoint = (ROOT/spec['checkpoint']).resolve()
    if target == checkpoint.parent or target.is_relative_to(checkpoint.parent):
        raise ValueError('readout output must be outside the immutable checkpoint directory')
    target.mkdir(parents=True, exist_ok=True)
    with (target/'.region-probe.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another region probe owns this output') from error
        return _collect_regions(config, target)


def materialize_completed_parent(config: Path) -> Path:
    """Fill checkpoint seals only after a preselected DMRG boundary completes."""
    spec = json.loads(config.read_text())
    declared = spec.pop('completed_parent')
    destination = ROOT/spec.pop('materialized_config')
    parent = ROOT/declared['directory']
    result = json.loads((parent/'results.json').read_text())
    registration = json.loads((parent/'registration.json').read_text())
    physical = registration['config']
    if (result['status'] != 'complete'
            or result['global_completed_sweeps'] != declared['sweep']
            or any(physical[k] != declared[k] for k in ('chi','rows','columns','j2'))):
        raise ValueError('preselected DMRG parent is incomplete or physically different')
    archive = parent/'archive.json'
    receipt = json.loads(archive.read_text())
    checkpoint = parent/'checkpoint.h5'
    digest = file_sha256(checkpoint)
    if digest != result['checkpoint_sha256'] or digest != receipt['checkpoint_sha256']:
        raise ValueError('completed DMRG parent archive/checkpoint mismatch')
    spec.update(checkpoint=str(checkpoint), checkpoint_sha256=digest,
                archive_receipt=str(archive), archive_receipt_sha256=file_sha256(archive))
    if destination.exists():
        if json.loads(destination.read_text()) != spec:
            raise ValueError('materialized fixed-parent readout changed')
    else:
        _atomic_json(destination,spec)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        _worker(args.worker)
    elif args.config is not None:
        config = (materialize_completed_parent(args.config)
                  if 'completed_parent' in json.loads(args.config.read_text()) else args.config)
        result = collect_regions(config, args.output)
        print(json.dumps(dict(status=result['status'], regions=len(result['regions']))), flush=True)
        if result['status'] != 'complete':
            raise SystemExit(2)
    else:
        parser.error('--config is required')


if __name__ == '__main__':
    main()
