"""Checkpointed finite-graph DMRG continuation with bounded CPU job chunks.

Checkpoints contain complete SWEEP-BOUNDARY states, adaptive Lanczos settings,
statistics and mixer schedule. A mid-sweep stop rolls back to the last sealed
boundary; it does not pretend an incomplete MPS sweep is resumable.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import resource
import shutil
import signal
import sys
from time import monotonic, time
from typing import Any, Callable

import numpy as np

from .spin_tensor_network_reference import ROOT, _json, file_sha256, transitive_local_source_manifest
from su2_nqs.benchmarks.tensor_network_readout import tenpy_graph_model, tenpy_region_density, physical_spin_block_readout
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds


class BoundaryRollback(RuntimeError):
    """A local update hit a deadline: use the existing completed-sweep file."""


@dataclass
class ChunkGuard:
    """Monotonic wall guard; signal handlers only request a safe rollback."""
    seconds: float
    reserve_seconds: float
    clock: Callable[[], float] = monotonic
    requested: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.reserve_seconds < self.seconds:
            raise ValueError('chunk duration must exceed its checkpoint reserve')
        self.started = self.clock()

    @property
    def elapsed(self) -> float:
        return self.clock()-self.started

    def request(self, *_: Any) -> None:
        self.requested = True

    def check_local(self) -> None:
        if self.requested or self.elapsed >= self.seconds-self.reserve_seconds:
            raise BoundaryRollback('signal or wall reserve reached; discard partial sweep')

    def can_start(self, predicted_seconds: float) -> bool:
        return not self.requested and self.seconds-self.reserve_seconds-self.elapsed > predicted_seconds


def guarded_engine_class() -> type:
    """Import optional TeNPy only at execution, not core readout import time."""
    from tenpy.algorithms.dmrg import TwoSiteDMRGEngine

    class GuardedDMRG(TwoSiteDMRGEngine):
        guard: ChunkGuard | None = None
        memory_cap_gib: float = float('inf')

        def prepare_update_local(self) -> Any:
            if self.guard is not None:
                self.guard.check_local()
            return super().prepare_update_local()

        def post_update_local(self, *args: Any, **kwargs: Any) -> None:
            super().post_update_local(*args, **kwargs)
            if self.guard is not None:
                self.guard.check_local()
            if peak_rss_gib() > self.memory_cap_gib:
                raise BoundaryRollback('local-update memory reserve reached')

    return GuardedDMRG


def save_engine(path: Path, engine: Any, metadata: dict) -> None:
    """Atomically save a completed-sweep engine, including mixer decay state."""
    import h5py
    import tenpy
    from tenpy.tools import hdf5_io
    mixer = engine.mixer
    if mixer is not None and type(mixer).__name__ != 'DensityMatrixMixer':
        raise ValueError('this sealed workflow validates only DensityMatrixMixer')
    payload = dict(schema=1, tenpy_version=tenpy.__version__, psi=engine.psi,
        resume_data=engine.get_resume_data(), options=engine.options.as_dict(),
        sweep_stats=engine.sweep_stats, update_stats=engine.update_stats,
        mixer=None if mixer is None else dict(class_name=type(mixer).__name__,
            amplitude=mixer.amplitude, decay=mixer.decay, disable_after=mixer.disable_after,
            sweep_activated=mixer.sweep_activated), metadata=metadata)
    tmp = path.with_suffix('.tmp.h5')
    with h5py.File(tmp, 'w') as handle:
        hdf5_io.save_to_hdf5(handle, payload)
        handle.flush()
    tmp.replace(path)


def load_engine(path: Path, model: Any, *, protocol_sha256: str) -> tuple[Any, dict]:
    """Restore the EXACT recorded policy; reject different versions/protocols."""
    import tenpy
    from tenpy.tools import hdf5_io
    saved = hdf5_io.load(path)
    if saved['schema'] != 1 or saved['tenpy_version'] != tenpy.__version__:
        raise ValueError('checkpoint schema or TeNPy version mismatch')
    if saved['metadata']['protocol_sha256'] != protocol_sha256:
        raise ValueError('checkpoint physical graph/policy mismatch')
    # Never momentarily reactivate a mixer that was already OFF at the
    # boundary. Its boolean option can outlive the actual decayed mixer.
    options = dict(saved['options'])
    if saved['mixer'] is None:
        options['mixer'] = False
    engine = guarded_engine_class()(saved['psi'], model, options, resume_data=saved['resume_data'])
    engine.pre_run_initialize()
    engine.sweep_stats = saved['sweep_stats']
    engine.update_stats = saved['update_stats']
    engine.time0 = time()-float(saved['metadata'].get('elapsed_seconds', 0.))-float(
        saved['metadata'].get('optimizer_elapsed_offset_seconds', 0.))
    if saved['mixer'] is None:
        engine.mixer = None
    else:
        if type(engine.mixer).__name__ != saved['mixer']['class_name']:
            raise ValueError('mixer class mismatch')
        for key in ('amplitude', 'decay', 'disable_after', 'sweep_activated'):
            setattr(engine.mixer, key, saved['mixer'][key])
    return engine, saved['metadata']


def inherit_engine(path: Path, model: Any, *, parent_protocol_sha256: str,
                   protocol_sha256: str, chi: int) -> tuple[Any, dict]:
    """Begin a new mixer-OFF stage without discarding the parent optimizer.

    Preserve environments, adaptive Lanczos tolerances and global statistics;
    only chi_max and the explicit OFF policy change. Stage sweep/wall budgets
    are separate from the retained global counters and parent wall ledger.
    """
    if isinstance(chi, bool) or not isinstance(chi, int) or chi < 1:
        raise ValueError('chi must be a positive integer')
    engine, parent = load_engine(path, model, protocol_sha256=parent_protocol_sha256)
    if engine.mixer is not None:
        raise ValueError('the declared transition requires an already mixer-OFF parent')
    old_chi = int(engine.trunc_params['chi_max'])
    if chi < old_chi or engine.chi_list:
        raise ValueError('only a nondecreasing fixed-chi transition is supported')
    engine.trunc_params['chi_max'] = chi
    engine.options['mixer'] = False
    metadata = dict(protocol_sha256=protocol_sha256, elapsed_seconds=0.,
        last_sweep_seconds=None, stage_start_sweeps=int(engine.sweeps),
        parent_protocol_sha256=parent_protocol_sha256,
        parent_checkpoint_sha256=file_sha256(path),
        parent_elapsed_seconds=float(parent['elapsed_seconds']),
        optimizer_elapsed_offset_seconds=float(parent['elapsed_seconds'])+float(
            parent.get('optimizer_elapsed_offset_seconds', 0.)),
        transition=dict(old_chi=old_chi, new_chi=chi, mixer=False,
                        inherited_optimizer=True, inherited_global_statistics=True))
    return engine, metadata


def validate_parent(spec: dict, bonds: list) -> dict:
    """Gate a new stage on a completed, sealed, same-H parent (not exit0 alone)."""
    initial = ROOT/spec['initial_checkpoint']
    directory = ROOT/spec['parent_output']
    if initial.resolve() != (directory/'checkpoint.h5').resolve():
        raise ValueError('parent checkpoint/output do not agree')
    parent = json.loads((directory/'results.json').read_text())
    registration = json.loads((directory/'registration.json').read_text())
    if parent['status'] != 'complete' or parent['registration'] != registration:
        raise ValueError('parent is not a sealed completed stage')
    actual = file_sha256(initial)
    if actual != parent['checkpoint_sha256']:
        raise ValueError('parent checkpoint no longer matches its completion seal')
    if spec.get('initial_checkpoint_sha256', actual) != actual:
        raise ValueError('parent differs from the predeclared checkpoint')
    if spec.get('parent_protocol_sha256', registration['protocol_sha256']) != registration['protocol_sha256']:
        raise ValueError('parent differs from the predeclared protocol')
    if registration['bonds'] != bonds:
        raise ValueError('parent physical Hamiltonian/boundaries changed')
    if registration['config']['regions'] != spec['regions']:
        raise ValueError('physical readout regions changed')
    if 'parent_config' in spec:
        expected = json.loads((ROOT/spec['parent_config']).read_text())
        if registration['config'] != expected or parent['completed_sweeps'] != expected['sweeps']:
            raise ValueError('dependency is not the predeclared completed parent stage')
    return dict(protocol_sha256=registration['protocol_sha256'],
                checkpoint_sha256=actual,
                result_sha256=file_sha256(directory/'results.json'),
                registration_sha256=file_sha256(directory/'registration.json'))


def save_boundary_readout(output: Path, engine: Any, model: Any, spec: dict) -> dict:
    """Save raw compact/winding RDMs of a completed boundary, without twirling.

    Canonicalize a COPY so the optimization environments remain untouched.
    If a previous chunk ended during readout, reconstruct the missing record
    from the recovered boundary before starting another optimization sweep.
    """
    sweep = int(engine.sweeps)
    directory = output/'sweep_readouts'
    record_path = directory/f'sweep{sweep:04d}.json'
    array_path = directory/f'sweep{sweep:04d}.npz'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record['rdms_sha256'] != file_sha256(array_path):
            raise ValueError('sealed boundary RDM changed')
        return record
    directory.mkdir(parents=True, exist_ok=True)
    start = monotonic()
    psi = engine.psi.copy()
    psi.canonical_form()
    energy = float(model.H_MPO.expectation_value(psi).real)/psi.L
    arrays, regional = {}, {}
    for name, sites in spec['regions'].items():
        rho = tenpy_region_density(psi, sites)
        arrays[f'chi{spec["chi"]}_{name}'] = rho
        regional[name] = physical_spin_block_readout(rho)
    temp = array_path.with_suffix('.tmp.npz')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temp.replace(array_path)
    record = dict(global_sweep=sweep, chi=spec['chi'], energy_per_site=energy,
        sweep_energy_closure_error=abs(energy-float(engine.sweep_stats['E'][-1])/psi.L),
        final_sweep_max_discarded_weight=float(engine.sweep_stats['max_trunc_err'][-1]),
        canonical_norm_error=float(np.linalg.norm(psi.norm_test())),
        regions=regional, rdms_sha256=file_sha256(array_path),
        seconds=monotonic()-start, variance_measured=False, symmetry_repair=False)
    _json(record_path, record)
    del psi
    gc.collect()
    return record


def peak_rss_gib() -> float:
    """ru_maxrss is bytes on Darwin, KiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**30 if sys.platform == 'darwin' else 2**20)


def analyze_probe(directory: Path) -> dict:
    """Compare each frozen chi1024 trajectory with its OWN chi256/512 history.

    Include a captured first-sweep state if available, with actual raw RDMs.
    Energy differences between initializations are not phase evidence.
    """
    from tenpy.tools import hdf5_io
    from su2_nqs.benchmarks.readout_quality import regional_readout_distance
    from su2_nqs.benchmarks.tensor_network_observables import tenpy_spin_correlations
    directory = directory.resolve()
    output = directory/'audit'
    if (output/'results.json').exists():
        raise FileExistsError('completed quality probe audit is immutable')
    inputs, records, arrays = {}, {}, {}
    start = monotonic()
    for arm in ('dimer', 'stripe'):
        path = directory/arm
        new = json.loads((path/'results.json').read_text())
        parent = ROOT/'projects/clebsch_tree_nqs/results/spin_tensor_network_reference_v2'/arm
        old = json.loads((parent/'results.json').read_text())
        if old['registration']['bonds'] != new['registration']['bonds']:
            raise ValueError('physical Hamiltonian changed between quality points')
        for source in (path/'results.json', path/'rdms.npz', path/'chi1024.h5',
                       parent/'results.json', parent/'rdms.npz'):
            inputs[str(source.relative_to(ROOT))] = file_sha256(source)
        with np.load(path/'rdms.npz') as z:
            final = {k:z[k] for k in z.files}
        with np.load(parent/'rdms.npz') as z:
            earlier = {k:z[k] for k in z.files}
        record = dict(initial_energy_per_site=old['history'][-1]['energy_per_site'],
            final_energy_per_site=new['history'][-1]['energy_per_site'],
            energy_improvement_per_site=old['history'][-1]['energy_per_site']-new['history'][-1]['energy_per_site'],
            energy_error_per_site_to_published_ED=new['history'][-1]['energy_per_site']+.515564,
            actual_sweeps=new['history'][-1]['sweeps'], seconds=new['seconds'], peak_rss_gib=new['peak_rss_gib'],
            maximum_discarded_weight=new['history'][-1]['max_discarded_weight'], regional={}, variance_measured=False)
        spec = new['registration']['config']
        first_path = path/'sweep1_checkpoint.h5'
        first = None
        if first_path.exists():
            inputs[str(first_path.relative_to(ROOT))] = file_sha256(first_path)
            first = hdf5_io.load(first_path)
            if first['metadata'].get('sweeps') != 1:
                raise ValueError('captured first-sweep checkpoint has another counter')
            first['psi'].canonical_form()
        for name, sites in spec['regions'].items():
            rho = final[f'chi1024_{name}']
            r = dict(chi256_to_512=regional_readout_distance(earlier[f'chi256_{name}'], earlier[f'chi512_{name}']),
                chi512_to_1024=regional_readout_distance(earlier[f'chi512_{name}'], rho),
                final_readout=physical_spin_block_readout(rho))
            if first is not None:
                rho1 = tenpy_region_density(first['psi'], sites)
                arrays[f'{arm}_sweep1_{name}'] = rho1
                r['chi1024_sweep1_to_2'] = regional_readout_distance(rho1, rho)
            arrays[f'{arm}_final_{name}'] = rho
            record['regional'][name] = r
        psi = hdf5_io.load(path/'chi1024.h5')['psi']
        spin, c = tenpy_spin_correlations(psi, spec['rows'], spec['columns'])
        record['global_spin'] = spin
        n = spec['rows']*spec['columns']
        bond_energy = sum(w*c[i, j] for i, j, w in new['registration']['bonds'])/n
        record['independent_bond_energy_closure_error'] = abs(bond_energy-record['final_energy_per_site'])
        if record['independent_bond_energy_closure_error'] > 1e-10:
            raise RuntimeError('independent spin-pair sum disagrees with MPO energy')
        arrays[f'{arm}_spin_correlations'] = c
        records[arm] = record
        print('audited probe', arm, record['final_energy_per_site'], flush=True)
    best = min(records, key=lambda a: records[a]['final_energy_per_site'])
    output.mkdir(parents=True, exist_ok=True)
    sources = transitive_local_source_manifest((Path(__file__),), root=ROOT,
                     package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    for path in sources:
        target = output/'source_snapshots'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/path, target)
    _json(output/'registration.json', dict(inputs=inputs, sources=sources,
        protocol='Same-H, each-own refinement and first-to-second sweep raw-RDM comparison. No symmetry repair, H², new ED or phase assignment.'))
    np.savez_compressed(output/'readouts.npz', **arrays)
    result = dict(records=records, lower_energy_initialization=best,
        initialization_energy_gap_per_site=abs(records['dimer']['final_energy_per_site']-records['stripe']['final_energy_per_site']),
        seconds=monotonic()-start, peak_rss_gib=peak_rss_gib(),
        new_ED=0, new_training=0, new_samples=0, new_H2_contractions=0, phase_claim=False,
        variance_measured=False,
        disposition='Bounded reference-acquisition evidence only. Two sweeps at larger chi confound extra relaxation and capacity; no isolated diagnosis of either, and no competing-phase claim.')
    _json(output/'results.json', result)
    return result


def run(config: Path) -> dict:
    """One bounded chunk of one fixed-H, fixed-chi reference trajectory."""
    import tenpy
    from tenpy.tools import hdf5_io
    spec = json.loads(config.read_text())
    output = ROOT/spec['output']
    if (output/'results.json').exists():
        raise FileExistsError('completed reference is immutable')
    if tenpy.__version__ != spec['tenpy_version']:
        raise ValueError('use the predeclared isolated TeNPy environment')
    guard = ChunkGuard(spec['chunk_seconds'], spec['reserve_seconds'])
    initial = ROOT/spec['initial_checkpoint']
    bonds = triangular_j1j2_bonds(spec['rows'], spec['columns'], j2=spec['j2'], periodic=True)
    n = spec['rows']*spec['columns']
    scientific = dict(config=spec, initial_checkpoint_sha256=file_sha256(initial), bonds=[list(b) for b in bonds])
    if spec.get('inherit_optimizer', False):
        if spec.get('mixer') is not False:
            raise ValueError('inherited stages require the explicit mixer-OFF policy')
        scientific['parent_seal'] = validate_parent(spec, scientific['bonds'])
    digest = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    sources = transitive_local_source_manifest((Path(__file__),), root=ROOT,
                    package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
    sources['su2_nqs/frontier/provenance.py'] = file_sha256(ROOT/'su2_nqs/frontier/provenance.py')
    registration = dict(**scientific, protocol_sha256=digest, sources=sources,
                        physical_order='row-major, physical torus; open virtual MPS')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'registration.json').exists():
        if json.loads((output/'registration.json').read_text()) != registration:
            raise ValueError('source/protocol registration changed; no silent continuation')
    else:
        _json(output/'registration.json', registration)
        for path in sources:
            target = output/'source_snapshots'/path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/path, target)
    model = tenpy_graph_model(n, bonds)
    checkpoint = output/'checkpoint.h5'
    if checkpoint.exists():
        engine, metadata = load_engine(checkpoint, model, protocol_sha256=digest)
    elif spec.get('inherit_optimizer', False):
        engine, metadata = inherit_engine(initial, model,
            parent_protocol_sha256=scientific['parent_seal']['protocol_sha256'],
            protocol_sha256=digest, chi=spec['chi'])
        save_engine(checkpoint, engine, metadata)
        _json(output/'transition.json', metadata)
    else:
        psi = hdf5_io.load(initial)['psi']
        if psi.L != n or psi.bc != 'finite':
            raise ValueError('initial MPS is incompatible with the physical graph')
        options = dict(N_sweeps_check=1, mixer=spec.get('mixer', True), mixer_params=spec['mixer_params'],
            trunc_params=dict(chi_max=spec['chi'], svd_min=1e-12), max_trunc_err=None,
            lanczos_params=dict(N_min=3, N_max=40, E_tol=1e-10, P_tol=1e-8))
        engine = guarded_engine_class()(psi, model, options)
        engine.pre_run_initialize()
        metadata = dict(protocol_sha256=digest, elapsed_seconds=0., last_sweep_seconds=None)
        save_engine(checkpoint, engine, metadata)
    engine.guard = guard
    engine.memory_cap_gib = float(spec['memory_cap_gib'])
    prior = float(metadata['elapsed_seconds'])
    # A discarded partial sweep still consumes the scientific wall budget.
    # Keep that ledger outside the unchanged completed-state checkpoint.
    if (output/'chunk_status.json').exists():
        prior = max(prior, float(json.loads((output/'chunk_status.json').read_text())['total_elapsed_seconds']))
    handlers = {s: signal.signal(s, guard.request) for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    reason = 'fixed sweep budget completed'
    stage_start = int(metadata.get('stage_start_sweeps', 0))
    target_sweeps = stage_start+spec['sweeps']
    try:
        if spec.get('save_each_sweep_rdms', False) and engine.sweeps > stage_start:
            save_boundary_readout(output, engine, model, spec)
        while engine.sweeps < target_sweeps:
            observed = metadata['last_sweep_seconds']
            prediction = spec['initial_sweep_seconds_bound'] if observed is None else max(30., 1.3*observed)
            if prior+guard.elapsed+prediction > spec['total_seconds_cap']:
                reason = 'total predeclared wall budget would be exceeded'; break
            if not guard.can_start(prediction):
                reason = 'not enough chunk time for another complete sweep'; break
            if peak_rss_gib() > spec['memory_cap_gib']:
                reason = 'predeclared memory reserve reached'; break
            t = monotonic()
            try:
                engine.run_iteration()
            except BoundaryRollback:
                reason = 'local-update deadline/signal; rolled back to last completed sweep'; break
            # A pure gauge cleanup updates the environment too. This SAME
            # boundary policy is used by uninterrupted and restored trajectories.
            engine.mixer_cleanup()
            metadata = dict(metadata, elapsed_seconds=prior+guard.elapsed,
                            last_sweep_seconds=monotonic()-t)
            save_engine(checkpoint, engine, metadata)
            if spec.get('save_each_sweep_rdms', False):
                save_boundary_readout(output, engine, model, spec)
            _json(output/'progress.json', dict(sweeps=int(engine.sweeps),
                completed_stage_sweeps=int(engine.sweeps)-stage_start,
                stage_start_sweeps=stage_start, mixer_active=engine.mixer is not None,
                energy_per_site=float(engine.sweep_stats['E'][-1])/n,
                sweep_seconds=metadata['last_sweep_seconds'], peak_rss_gib=peak_rss_gib(),
                checkpoint_sha256=file_sha256(checkpoint), variance_measured=False))
            print(json.dumps(dict(sweep=int(engine.sweeps), E_per_site=float(engine.sweep_stats['E'][-1])/n,
                                  seconds=metadata['last_sweep_seconds'], peak_rss_gib=peak_rss_gib())), flush=True)
        # Read the sealed boundary afresh: NEVER retain a partial in-memory sweep.
        saved = hdf5_io.load(checkpoint)
        global_completed = int(saved['resume_data']['sweeps'])
        completed = global_completed-stage_start
        status = dict(status='complete' if completed >= spec['sweeps'] else 'paused', reason=reason,
            completed_sweeps=completed, global_completed_sweeps=global_completed,
            stage_start_sweeps=stage_start, mixer_active=saved['mixer'] is not None,
            chunk_seconds=guard.elapsed, checkpoint_sha256=file_sha256(checkpoint),
            total_elapsed_seconds=prior+guard.elapsed, peak_rss_gib=peak_rss_gib(),
            new_ED=0, phase_claim=False, variance_measured=False)
        _json(output/'chunk_status.json', status)
        if completed < spec['sweeps']:
            return status
        # Do not retain the optimization environment while separately
        # contracting final physical RDMs/energy from the sealed boundary.
        del engine
        gc.collect()
        psi = saved['psi']
        psi.canonical_form()
        energy = float(model.H_MPO.expectation_value(psi).real)/n
        arrays, regional = {}, {}
        for name, sites in spec['regions'].items():
            rho = tenpy_region_density(psi, sites)
            arrays[f'chi{spec["chi"]}_{name}'] = rho
            regional[name] = physical_spin_block_readout(rho)
        temp = output/'rdms.tmp'
        with temp.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        temp.replace(output/'rdms.npz')
        shutil.copy2(checkpoint, output/f'chi{spec["chi"]}.h5')
        status.update(chunk_seconds=guard.elapsed, total_elapsed_seconds=prior+guard.elapsed,
                      peak_rss_gib=peak_rss_gib())
        _json(output/'chunk_status.json', status)
        result = dict(**status, registration=registration, energy_per_site=energy,
            actual_chi=int(max(psi.chi)), canonical_norm_error=float(np.linalg.norm(psi.norm_test())),
            maximum_discarded_weight=float(max(saved['sweep_stats']['max_trunc_err'])),
            final_sweep_max_discarded_weight=float(saved['sweep_stats']['max_trunc_err'][-1]),
            boundary_readout_count=len(list((output/'sweep_readouts').glob('sweep*.json'))),
            regions=regional, rdms_sha256=file_sha256(output/'rdms.npz'),
            disposition='Finite-chi reference candidate; require quality/readout stability before physical competition claims.')
        _json(output/'results.json', result)
        return result
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--config', type=Path)
    modes.add_argument('--analyze-probe', type=Path)
    parser.add_argument('--require-complete', action='store_true',
        help='exit nonzero on a safe pause so afterok cannot release the next stage')
    args = parser.parse_args()
    result = run(args.config) if args.config else analyze_probe(args.analyze_probe)
    print(json.dumps({k:result[k] for k in ('status', 'completed_sweeps', 'reason') if k in result}), flush=True)
    if args.require_complete and result.get('status') != 'complete':
        raise SystemExit(2)
