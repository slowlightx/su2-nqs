"""Frozen N144 GPU cost -> support -> fresh H-bridge audit, without training.

Preparation creates deterministic registrations and a short CPU replay oracle.
The GPU allocation is capped at three hours. Every stage is released only from
the preceding fixed endpoint; neither probabilities nor energies select states.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from projects.clebsch_tree_nqs.workflows.triangle_hierarchy_audit import (
    bridge_projection_chunk, checked_active_widths, finite_relative_error, run_h_bridge_endpoint_audit,
    run_pair_support_cost, run_pair_support_stage, support_coverage_diagnostics,
)

ROOT = Path.cwd()
OLD = Path('projects/clebsch_tree_nqs/results/ct_conditional_nested_initialization_20261023')
OUT = Path('projects/clebsch_tree_nqs/results/ct_transfer_gpu_audit_20261025')
CONFIG = Path('projects/clebsch_tree_nqs/configs/remote/ct_transfer_gpu_20261025')
BASE_OUT, BASE_CONFIG = OUT, CONFIG


def prepare() -> None:
    """Register the fixed budgets before generating the short CPU oracle."""
    if (CONFIG/'plan.json').exists():
        raise FileExistsError('Frozen campaign plan already exists')
    spec = json.loads((OLD/'support_stage_2048_plan.json').read_text())
    sample = OLD/'support_stage_2048/support_prefix.npz'
    with np.load(sample, allow_pickle=False) as saved:
        rng = saved['rng'].tolist()
    spec.update(task='same_frozen_sampler_GPU_10240_then_H_bridge',
        bond_ledger_path=str(OLD/'H_bridge_bond_ledger.json'),
        bond_ledger_sha256=file_sha256(OLD/'H_bridge_bond_ledger.json'),
        initial_samples_path=str(sample), initial_samples_sha256=file_sha256(sample),
        initial_rng_key=rng, execution_profile='registered_single_gpu',
        segments=80, maximum_registered_proposals=10240, seconds_cap=5400.,
        cost_seconds_cap=1200., cpu_reference_only=False,
        previous_support_proposals=2048, fixed_total_support_proposals=12288,
        coverage_gate=dict(window_records=512, maximum_split_rhat=1.1,
            maximum_drift_standard_errors=2., minimum_effective_fraction=.25,
            maximum_chain_fraction=.35),
        no_automatic_H_release=False,
        scope='Fixed eight frozen chains; preserve the saved RNG and the original128-proposal segmentation.',
        gate_reason='The predeclared final512 records span8192 single exchanges, rather than testing only a short final window.',
        no_further_automatic_extension=True)
    spec.pop('continuation_reason', None)
    spec['CPU_serialization'] = 'Only the two-proposal replay oracle runs locally.'
    cpu = spec | dict(cpu_reference_only=True, cost_seconds_cap=180.)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(CONFIG/'cpu_reference_plan.json', cpu)
    atomic_json(CONFIG/'campaign_registration.json', dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),
        support_proposals=10240, support_segments=80, chains=8,
        gate_window_records=512, diagnostic_only_windows=[16,64],
        audit_records_per_chain=32, audit_burn=128, audit_between=128,
        audit_seed=2610258300, audit_H_centers=516,
        extra_GPU_cost_H_centers=4, CPU_H_queries=0,
        requested_GPU_hours_upper_bound=3., maximum_simultaneous_GPUs=1,
        overall_kernel_boundary_cap_seconds=10200.,
        stage_caps='support<=5400 and audit<=3600, further bounded by actual remaining allocation; never reduce the fixed record/proposal counts',
        no_training_or_attention_fit=True, no_adaptive_budget_extension=True,
        runtime_release='Cost and replay -> all10240 proposals -> fixed512 stability gate -> fresh audit'))
    result = run_pair_support_cost(cpu, OUT/'cpu_reference')
    if result['status'] != 'CPU_short_reference_complete_no_long_sampling':
        raise RuntimeError('CPU reference did not complete')
    reference = OUT/'cpu_reference/short_reference.npz'
    spec.update(cpu_reference_path=str(reference), cpu_reference_sha256=file_sha256(reference))
    atomic_json(CONFIG/'plan.json', spec)


def seal() -> None:
    """Freeze only the imported source, exact inputs and executable wrapper."""
    spec = json.loads((CONFIG/'plan.json').read_text())
    entry = [Path(__file__), BASE_OUT/'verify.py']
    manifest = transitive_local_source_manifest(entry, root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    extra = [CONFIG/'plan.json', CONFIG/'cpu_reference_plan.json',
        CONFIG/'campaign_registration.json', CONFIG/'run.sbatch',
        Path('AGENTS.md'), Path(spec['control_checkpoint']),
        Path(spec['candidate_checkpoint']), Path(spec['initial_samples_path']),
        Path(spec['cpu_reference_path']), Path(spec['cpu_reference_path']).parent/'results.json',
        Path(spec['cpu_reference_path']).parent/'registration.json',
        Path(spec['cpu_reference_path']).parent/'registered_workflow.py',
        Path(spec['bond_ledger_path']),
        OLD/'bridge_cost_4/cost_products.npz']
    for path in extra:
        manifest[str(path)] = file_sha256(path)
    atomic_json(CONFIG/'manifest.json', manifest)
    (CONFIG/'sync_files.txt').write_text('\n'.join(sorted(manifest))+ '\n'+
        str(CONFIG/'manifest.json')+'\n')
    print(json.dumps(dict(files=len(manifest), bytes=sum(Path(p).stat().st_size for p in manifest))), flush=True)


def bridge_cost(spec: dict, *, deadline: float) -> dict:
    """Four same-center products against the saved independent CPU H oracle."""
    from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from su2_nqs.models.triangular_clebsch_active_layout import repack_active_clebsch_bank
    from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank
    from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
    from su2_nqs.vmc.checkpoint import load_checkpoint
    from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_bridge

    folder = OUT/'gpu_H_cost'
    folder.mkdir()
    cfg = MixedClebschConfig(**spec['model'])
    active = checked_active_widths(spec['active_top_widths'], nodes=15)
    with np.load(OLD/'bridge_cost_4/cost_products.npz', allow_pickle=False) as saved:
        rows = saved['rows'].copy()
        expected = np.stack((saved['psi'], saved['Hpsi'], saved['pushed_probability']), axis=1)
    bonds = triangular_j1j2_bonds(12,12,j2=.125,periodic=True)
    p = np.abs([v for _,_,v in bonds]); p /= p.sum()
    images = np.asarray(triangular_hnf_gamma_a1_group(((12,0),(0,12))),np.int32)
    H_chunk = bridge_projection_chunk(spec)
    kernel, metadata = make_cached_projected_heisenberg_bridge(cfg,images,bonds,p,
        projection_chunk=H_chunk,active_top_widths=active)
    timing = {}; values = []; completed = 0
    for k, name in enumerate(('control','candidate')):
        bank = prepare_feature_lut_bank(repack_active_clebsch_bank(
            load_checkpoint(spec[name+'_checkpoint'])['params'],cfg,active),cfg)
        for label in ('first','warm'):
            if perf_counter()>deadline-30:
                raise TimeoutError('Registered full-H cost budget exhausted before '+name+'_'+label)
            atomic_json(folder/'progress.json',dict(operation=name+'_'+label,
                completed_full_H_centers=completed,in_flight_H_centers_upper_bound=1,timings=timing))
            tick=perf_counter(); answer=jax.block_until_ready(kernel(bank,rows))
            timing[name+'_'+label]=perf_counter()-tick
            completed+=1
            atomic_json(folder/'progress.json',dict(operation='center_complete',
                completed_full_H_centers=completed,timings=timing))
        actual = np.asarray([np.asarray(a)[0] for a in answer])
        for a,b in zip(actual,expected[k],strict=True):
            if finite_relative_error(np.asarray([a]),np.asarray([b]))>2e-10:
                raise ArithmeticError('CPU/GPU full-H bridge products failed replay')
        values.append(actual)
    values=np.asarray(values)
    np.savez_compressed(folder/'cost_products.npz',rows=rows,psi=values[:,0],
        Hpsi=values[:,1],pushed_probability=values[:,2].real)
    report=dict(status='GPU_H_bridge_replay_complete',full_H_centers=4,timings=timing,
        H_projection_chunk=H_chunk,
        kernel_metadata=metadata,relative_errors=[[
            finite_relative_error(np.asarray([a]),np.asarray([b])) for a,b in zip(row,ref,strict=True)]
            for row,ref in zip(values,expected,strict=True)],
        memory_stats=jax.local_devices()[0].memory_stats())
    atomic_json(folder/'results.json',report)
    return report


def execute() -> None:
    """Execute the prospective conditional protocol inside one Slurm allocation."""
    jax.config.update('jax_enable_x64',True)
    manifest=json.loads((CONFIG/'manifest.json').read_text())
    for path,digest in manifest.items():
        if file_sha256(Path(path))!=digest:
            raise ArithmeticError('Frozen campaign file changed: '+path)
    if (OUT/'GPU_execution.json').exists():
        raise FileExistsError('This frozen GPU campaign has already been attempted')
    started=perf_counter()
    campaign=json.loads((CONFIG/'campaign_registration.json').read_text())
    overall_cap=float(campaign['overall_kernel_boundary_cap_seconds'])
    status=dict(status='started',started_utc=datetime.now(timezone.utc).isoformat(),
        no_training=True,requested_GPU_hours=campaign['requested_GPU_hours_upper_bound'],
        manifest_sha256=file_sha256(CONFIG/'manifest.json'))
    atomic_json(OUT/'GPU_execution.json',status)
    try:
        spec=json.loads((CONFIG/'plan.json').read_text())
        pilot=run_pair_support_cost(spec,OUT/'gpu_cost')
        if pilot['status']!='GPU_support_cost_complete':
            status['status']='GPU_cost_incomplete_no_support_or_audit';return
        hc=bridge_cost(spec,deadline=min(started+overall_cap,perf_counter()+1200.))
        hcost=hc['timings']['control_warm']+hc['timings']['candidate_warm']
        proposal=pilot['warm_seconds_per_8chain_proposal']
        audit_forecast=1.5*((128+32*128+1)*proposal+256*hcost)+300.
        forecast=pilot['support_remaining_forecast_seconds']+audit_forecast
        status.update(support_forecast_seconds=pilot['support_remaining_forecast_seconds'],
            audit_forecast_seconds=audit_forecast,total_remaining_forecast_seconds=forecast)
        if forecast>overall_cap-(perf_counter()-started) or audit_forecast>3600:
            status['status']='measured_cost_exceeds_registered_budget_no_sampling';return
        support_cap=min(5400.,overall_cap-(perf_counter()-started)-audit_forecast)
        if pilot['support_remaining_forecast_seconds']>support_cap or support_cap<60:
            status['status']='insufficient_remaining_budget_for_fixed_support';return
        support_spec=spec|dict(seconds_cap=support_cap)
        support=run_pair_support_stage(support_spec,OUT/'support10240')
        if support.get('fixed_endpoint_reached',False):
            with np.load(OUT/'support10240/support_prefix.npz',allow_pickle=False) as saved:
                psi=saved['psi'].copy()
            diagnostics={str(n):support_coverage_diagnostics(psi,
                old_floor=support['old_floor_reference_only'],window_records=n) for n in (16,64)}
            atomic_json(OUT/'support10240/short_window_diagnostics.json',diagnostics)
        if not support.get('fixed_endpoint_reached',False) or not support.get('operational_gate_passed',False):
            status['status']='fixed_support_gate_failed_no_H_audit_or_attention';return
        audit_cap=min(3600.,overall_cap-(perf_counter()-started))
        if audit_forecast>audit_cap or audit_cap<60:
            status['status']='support_passed_but_remaining_audit_budget_insufficient';return
        audit=spec|dict(chains=8,records=32,burn_proposals=128,proposals_between=128,
            seed=2610258300,seconds_cap=audit_cap)
        audit.pop('initial_rng_key',None)
        audit['fresh_audit_rng_protocol']='PRNGKey(seed2610258300); support RNG and discarded pilot traces are not reused'
        for name,path in (('support_samples',OUT/'support10240/support_prefix.npz'),
            ('support_results',OUT/'support10240/results.json'),
            ('support_registration',OUT/'support10240/registration.json')):
            audit[name+'_path']=str(path);audit[name+'_sha256']=file_sha256(path)
        atomic_json(OUT/'audit_plan.json',audit)
        result=run_h_bridge_endpoint_audit(audit,OUT/'fresh_H_bridge')
        status['status']=result['status']
        status['attention_release']='Independent mathematical/statistical review required; no automatic training.'
    except Exception as error:
        status.update(status='failed',error=repr(error));raise
    finally:
        status['elapsed_seconds']=perf_counter()-started
        atomic_json(OUT/'GPU_execution.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','seal','execute'))
    parser.add_argument('--variant',choices=('chunk64',))
    args=parser.parse_args()
    if args.variant:
        OUT, CONFIG = BASE_OUT/args.variant, BASE_CONFIG/args.variant
        if args.phase=='prepare':
            raise ValueError('The batching-only variant is explicitly registered from the existing CPU oracle, not regenerated.')
    {'prepare':prepare,'seal':seal,'execute':execute}[args.phase]()
