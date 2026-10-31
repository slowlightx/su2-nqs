"""Bounded acquisition/readout of optional finite TeNPy reference states.

Execute in the EXISTING tenpy environment from the nqs workspace. External
repositories and environments are not modified. Every completed stage has an
MPS checkpoint; failed convergence is recorded, never relabelled as a phase.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import signal
import sys
from time import perf_counter

import numpy as np

from su2_nqs.benchmarks.tensor_network_readout import tenpy_graph_model, tenpy_region_density, physical_spin_block_readout
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
ROOT=Path(__file__).resolve().parents[3]
DEFAULT=ROOT/'projects/clebsch_tree_nqs/configs/validation/spin_tensor_network_reference.json'
# The legacy frontier package __init__ eagerly imports JAX. Reuse its pure
# provenance implementation standalone in TeNPy-only environments, without
# changing any historically sealed source or installing training dependencies.
_path=ROOT/'su2_nqs/frontier/provenance.py'
_loader=importlib.util.spec_from_file_location('nqs_reference_provenance',_path)
_provenance=importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(_provenance)
file_sha256=_provenance.file_sha256
transitive_local_source_manifest=_provenance.transitive_local_source_manifest


def _json(path: Path, value: object) -> None:
    """Atomic numerical artifact writer without the training/JAX dependency."""
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)


def _checkpoint(path: Path, psi: object, metadata: dict) -> None:
    import h5py
    from tenpy.tools import hdf5_io
    tmp=path.with_suffix('.tmp.h5')
    with h5py.File(tmp,'w') as handle:
        hdf5_io.save_to_hdf5(handle,dict(psi=psi,metadata=metadata))
    tmp.replace(path)


def run(config: Path, arm: str) -> dict:
    """Acquire one predeclared initialization, with fixed chi/sweep budgets."""
    import h5py, tenpy
    from tenpy.algorithms import dmrg
    from tenpy.networks.mps import MPS
    from tenpy.tools import hdf5_io
    spec=json.loads(config.read_text())
    if arm not in spec['initializations']:
        raise ValueError('unknown preregistered initialization')
    output=ROOT/spec['output']/arm
    if (output/'results.json').exists():
        raise FileExistsError('completed run is immutable')
    output.mkdir(parents=True,exist_ok=True)
    bonds=triangular_j1j2_bonds(spec['rows'],spec['columns'],j2=spec['j2'],periodic=True)
    n=spec['rows']*spec['columns']
    model=tenpy_graph_model(n,bonds)
    initial_path=ROOT/spec['initial_checkpoints'][arm] if 'initial_checkpoints' in spec else None
    if arm == 'warm' and initial_path is None:
        raise ValueError('warm reference requires an explicit immutable initial MPS')
    registration=dict(config=spec,config_sha256=file_sha256(config),arm=arm,
        sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,tenpy=tenpy.__version__),
        bonds=[list(b) for b in bonds],physical_order='row-major, bit0 down; finite virtual MPS, physical torus',
        warning='Finite-chi endpoints are candidate references, NOT LE/HE evidence or a QSL diagnosis.')
    registration['sources']['su2_nqs/frontier/provenance.py']=file_sha256(_path)
    if initial_path is not None:
        registration['initial_checkpoint_sha256']=file_sha256(initial_path)
        declared = spec.get('initial_checkpoint_sha256', {}).get(arm)
        if declared is not None and registration['initial_checkpoint_sha256'] != declared:
            raise ValueError('warm initial physical MPS changed')
    if (output/'registration.json').exists():
        if json.loads((output/'registration.json').read_text())!=registration:
            raise ValueError('resume registration mismatch')
    else:
        _json(output/'registration.json',registration)
    checkpoint=output/'checkpoint.h5'
    if checkpoint.exists():
        with h5py.File(checkpoint,'r') as handle:
            saved=hdf5_io.load_from_hdf5(handle)
        psi=saved['psi']; metadata=saved['metadata']
        history=metadata['history']; done=int(metadata['completed_stages'])
        prior_seconds=float(metadata.get('elapsed_seconds',0.))
        interrupted_sweeps=int(metadata.get('sweeps',0)) if metadata.get('interrupted_stage')==done else 0
    else:
        sites=model.lat.mps_sites(); history=[]; done=0; prior_seconds=0.; interrupted_sweeps=0
        if initial_path is not None:
            with h5py.File(initial_path,'r') as handle:
                psi=hdf5_io.load_from_hdf5(handle)['psi']
            if psi.L!=n or psi.bc!='finite':
                raise ValueError('initial checkpoint does not match the finite physical graph')
        elif arm=='dimer':
            psi=MPS.from_singlets(sites[0],n,[(i,i+1) for i in range(0,n,2)],bc='finite')
        else:
            pattern=['up' if (i//spec['columns'])%2 else 'down' for i in range(n)]
            psi=MPS.from_product_state(sites,pattern,bc='finite')
    metadata = dict(history=history, completed_stages=done, elapsed_seconds=prior_seconds)
    start=perf_counter(); rho_archive={}
    guard = None
    if spec.get('safe_local_update_guard', False):
        from .dmrg_continuation import ChunkGuard, guarded_engine_class, BoundaryRollback
        guard = ChunkGuard(spec['seconds_per_arm']+spec['reserve_seconds'], spec['reserve_seconds'])
        signal.signal(signal.SIGUSR1, guard.request)
        signal.signal(signal.SIGTERM, guard.request)
        if not checkpoint.exists():
            _checkpoint(checkpoint, psi, dict(history=history, completed_stages=done,
                elapsed_seconds=prior_seconds, inherited_optimizer=False))
    if (output/'rdms.npz').exists():
        with np.load(output/'rdms.npz') as archive:
            rho_archive={key:archive[key] for key in archive.files}
    for stage_index,stage in enumerate(spec['stages']):
        if stage_index<done:
            continue
        remaining=spec['seconds_per_arm']-prior_seconds-(perf_counter()-start)
        if remaining<30:
            break
        already=interrupted_sweeps if stage_index==done else 0
        budget=stage['sweeps']-already
        if budget<=0:
            raise ValueError('checkpoint stage budget is exhausted but lacks a completed readout')
        # TeNPy 1.1 stops when sweeps > max_sweeps, not >=. With one sweep
        # per iteration, subtract one to enforce the requested actual count.
        options=dict(N_sweeps_check=1,max_sweeps=budget-1,min_sweeps=budget-1,
            max_hours=remaining/3600,max_E_err=1e-9,max_S_err=1e-8,
            trunc_params=dict(chi_max=stage['chi'],svd_min=1e-12),
            mixer=spec.get('mixer',False),max_trunc_err=None,
            lanczos_params=dict(N_min=3,N_max=40,E_tol=1e-10,P_tol=1e-8))
        engine=(dmrg.TwoSiteDMRGEngine if guard is None else guarded_engine_class())(psi,model,options)
        if guard is not None:
            engine.guard = guard
            engine.memory_cap_gib = spec['memory_cap_gib']
        def save_sweep(algo):
            _checkpoint(checkpoint,algo.psi,dict(history=history,completed_stages=stage_index,
                        interrupted_stage=stage_index,sweeps=already+algo.sweeps,
                        elapsed_seconds=prior_seconds+perf_counter()-start))
            print(arm,'chi',stage['chi'],'sweep',algo.sweeps,'E/N',float(algo.sweep_stats['E'][-1])/n,flush=True)
        engine.checkpoint.connect(save_sweep)
        stage_start=perf_counter()
        try:
            energy,psi=engine.run()
        except Exception as error:
            if guard is None or not isinstance(error, BoundaryRollback):
                raise
            # Never overwrite the last completed physical sweep with partial
            # tensors. A future acquisition can restart from that boundary,
            # explicitly resetting the optimizer (not identical continuation).
            result = dict(status='paused_safe_boundary', seconds=prior_seconds+perf_counter()-start,
                reason=str(error), checkpoint_sha256=file_sha256(checkpoint),
                partial_sweep_discarded=True, inherited_optimizer=False, phase_claim=False)
            _json(output/'paused.json', result)
            return result
        psi.canonical_form()
        energy=float(model.H_MPO.expectation_value(psi).real)
        record=dict(chi_max=stage['chi'],actual_chi=int(max(psi.chi)),sweeps=int(engine.sweeps),
                    energy_per_site=energy/n,seconds=perf_counter()-stage_start,
                    stage_sweeps_total=already+int(engine.sweeps),
                    canonical_norm_error=float(np.linalg.norm(psi.norm_test())),
                    last_energy_change_per_site=float(engine.sweep_stats['Delta_E'][-1])/n,
                    max_discarded_weight=float(engine.sweep_stats['max_trunc_err'][-1]),
                    regions={})
        # Measure bounded exact MPS RDMs; do NOT project an approximate state.
        for name,region in spec['regions'].items():
            rho=tenpy_region_density(psi,region)
            record['regions'][name]=physical_spin_block_readout(rho)
            rho_archive[f'chi{stage["chi"]}_{name}']=rho
        history.append(record)
        complete=already+engine.sweeps>=stage['sweeps']
        metadata=dict(history=history,completed_stages=stage_index+1 if complete else stage_index,
                      elapsed_seconds=prior_seconds+perf_counter()-start)
        if not complete:
            metadata.update(interrupted_stage=stage_index,sweeps=already+engine.sweeps)
        _checkpoint(checkpoint,psi,metadata)
        _checkpoint(output/f'chi{stage["chi"]}.h5',psi,metadata)
        tmp=output/'rdms.tmp'
        with tmp.open('wb') as handle:
            np.savez_compressed(handle,**rho_archive)
        tmp.replace(output/'rdms.npz')
        _json(output/'progress.json',dict(history=history,seconds=perf_counter()-start))
        print(arm,'completed chi',stage['chi'],'E/N',record['energy_per_site'],'RDM SU2 defects',
              {k:v['su2_factorization_frobenius'] for k,v in record['regions'].items()},flush=True)
        if not complete:
            break
    result=dict(status='bounded_acquisition_complete' if metadata['completed_stages'] == len(spec['stages'])
                    else 'paused_acquisition_budget',registration=registration,history=history,
                seconds=prior_seconds+perf_counter()-start,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    *(1 if sys.platform == 'darwin' else 1024)/2**30,
                checkpoint_sha256=file_sha256(checkpoint),rdms_sha256=file_sha256(output/'rdms.npz'),
                new_ED=0,phase_claim=False,variance_measured=False,
                disposition='Use only after refinement, independent initialization, SU2 and correlation checks. No forced A1/S0 repair.')
    _json(output/'results.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT); p.add_argument('--arm',choices=['dimer','stripe','warm'],required=True)
    args=p.parse_args(); result=run(args.config,args.arm)
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'])))
    if result['status'] != 'bounded_acquisition_complete':
        raise SystemExit(2)
