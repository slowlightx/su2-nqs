"""Explicitly migrated CT12 w28 gate with budgeted exact physical sampling.

Old comparison artifacts and failed gates are immutable.  This new campaign
replays one frozen actual1024 full-W direction, uses wholly fresh independent
geometry/H panels, and does not automatically authorize training.  A separate
root release can enable at most eight attempts, with a fresh4096 endpoint audit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
from time import perf_counter, time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.vmc.budgeted_triangular_born import BudgetedTriangularOrbitDrawer
from su2_nqs.vmc.checkpoint import save_checkpoint
from su2_nqs.vmc.full_support_proposal import full_support_sz0_density
from su2_nqs.vmc.finite_candidate_calibration import common_proposal_FS
from su2_nqs.vmc.finite_candidate_calibration import bilateral_direction_moments
from . import ct12_actual1024_cached_gate_20261031 as old
from . import ct12_direction_gate as legacy
from . import triangle_active_two_step_pilot as active
from .triangle_iid_orbit_audit import paired_iid_moments

ROOT = old.ROOT
TASK = 'CT12_w28_global45_budgeted_physical_reentry_20261101_v1'
PHASES = ('gate', 'replay', 'pilot_prepare', 'pilot_replay', 'pilot_train', 'audit')
FACTORS = np.asarray([6., 4., 3., 2., 1.5, 1., .5, 0.])
OLD_PLAN_SHA = '59327cb4d14a39704b974b60c49709558b120561a149cd62068f18a5f87249ff'
OLD_SOURCE = 'a31daa17cf7c7d71432e08b6499d3934a06278c114138d7eadc56d385e1b28b0'
OLD_COMMIT = 'fe355965d5bac2e90c68baa0f94d63b883584446'
OLD_RNG_SHA = '8040507bb68b34a94442cbf6116b609730bd168328357d7c7be5ba5d9ae46a95'
OLD_FILES = {
    'registration.json':'1b529823dd743788d1b8fe6a5238e6bea1dacf04ad1ab393210343938a483cba',
    'compare_result.json':'86016719c880cfc76b4460c46d67dfbf57bed0d1132355f33cef094e1507d0b0',
    'wide28_ready.json':'d2df092ff448ce13b9fed78a524dbc9817c3c8b8e5954d163c8cf54c5422a39e',
    'wide28_fit_SR.json':'2c7dae14da5ec08942df02e8d75d61e3ba9030f7faffc86164fc657d8028b7df',
    'fit_panel.npz':'c82f6c7359b1c3785ca103146cba9a943fc0de8c971702bfee8b90cb4ed374b6',
    'wide28_full_proposal.pkl':'8c66754b7d1c093fc3bae9f282f21484cf2e2cb71f85b3f01e841119e6587d7c',
    'wide_actual1024_J.npy.json':'e25284c2f838d577d07f6d90d421bbdec3c64cd43b4a7facd20071bb9863fac0',
    'wide_cache_metadata.json':'02f6d89786c4667c5b283876aefa5590a642495fdecea337628be389513d8f19',
}
ROLES = ('selection', 'guard', 'bilateral', 'confirmation',
         'confirmation_statistics', 'audit', 'audit_statistics')


def default_plan() -> dict:
    """Return an unapproved, precisely scoped plan; root releases execution."""
    return dict(task=TASK, execution_authorized=False, legacy_plan_sha256=OLD_PLAN_SHA,
        legacy_source_identity=OLD_SOURCE, legacy_commit=OLD_COMMIT,
        legacy_directory='/home/yx789/nqs-wavefunctions/nqs/ct12_actual1024_20261031/run01',
        legacy_file_sha256=deepcopy(OLD_FILES), parent_global_step=45, j2=.125,
        fit_records=1024, calibration_records=1024, calibration_split=512,
        bilateral_records=512, bilateral_groups=32, confirmation_records=1024,
        confirmation_groups=32, audit_records=4096, audit_groups=64,
        maximum_attempts=8, damping=.5, learning_rate=.03, target_FS=.015,
        maximum_FS=.02, minimum_guard_FS=.0075, epsilon=.125,
        geometry_factors=FACTORS.tolist(), complete_A1_images=1728, physical_H_bonds=864,
        native_parameters=20748160, derivative_row_batch=2, cached_parameter_chunk=16384,
        H_chunk=8, maximum_peak_device_bytes=70*1024**3,
        maximum_host_array_bytes=240*1024**3, sampler_per_bank_bytes=8*1024**3,
        sampler_aggregate_bytes=64*1024**3, sampler_other_live_bytes=16*1024**3,
        requested_host_GiB=320, gate_wall_cap_seconds=18*3600,
        pilot_wall_cap_seconds=72*3600, restore_reserve_seconds=5400,
        save_reserve_seconds=300, source_cache_reuse='read_only_verified_legacy_migration',
        pilot_cache_policy='volatile_full_exact_current_parent_only',
        automatically_start_pilot=False, automatically_submit=False,
        seeds={key:4101101001+i for i,key in enumerate(ROLES)})


def validate_plan(plan: dict, *, executing: bool=False) -> None:
    """Forbid changes to physics, statistics, budgets and the frozen cache source."""
    expected=default_plan()
    if set(plan)!=set(expected):raise ValueError('exact new reentry plan fields required')
    def same(value,reference):
        if isinstance(reference,bool):return isinstance(value,bool) and value is reference
        if isinstance(reference,int):return isinstance(value,int) and not isinstance(value,bool) and value==reference
        if isinstance(reference,float):return isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value) and value==reference
        if isinstance(reference,list):return isinstance(value,list) and len(value)==len(reference) and all(same(a,b) for a,b in zip(value,reference,strict=True))
        if isinstance(reference,dict):return isinstance(value,dict) and set(value)==set(reference) and all(same(value[k],v) for k,v in reference.items())
        return type(value) is type(reference) and value==reference
    for key,value in expected.items():
        if key not in ('execution_authorized','seeds') and not same(plan[key],value):
            raise ValueError('registered reentry protocol changed: '+key)
    if not isinstance(plan['execution_authorized'],bool):raise ValueError('boolean execution authorization required')
    seeds=plan['seeds']
    if (set(seeds)!=set(ROLES) or len(set(seeds.values()))!=len(ROLES)
            or any(isinstance(x,bool) or not isinstance(x,int) or not 0<=x<2**32 for x in seeds.values())):
        raise ValueError('distinct explicit uint32 role seeds required')
    if executing and plan['execution_authorized'] is not True:
        raise PermissionError('root execution release required')


def legacy_plan() -> dict:
    """Read the sealed old protocol; never rewrite it to match this workflow."""
    plan=json.loads((ROOT/'configs/remote/ct12_actual1024_20261031/plan.json').read_text())
    old.validate_plan(plan)
    if canonical_sha256(plan)!=OLD_PLAN_SHA:raise ValueError('legacy plan changed')
    return plan


def source_identity(plan: dict) -> str:
    """Bind new execution to complete immutable deployment and changed modules."""
    from su2_nqs.vmc import budgeted_triangular_born as sampler
    from su2_nqs.vmc import cached_amplitude_operators as engine
    return canonical_sha256(dict(plan=plan,driver=old.file_sha(Path(__file__)),
        sampler=old.file_sha(Path(sampler.__file__)),engine=old.file_sha(Path(engine.__file__)),
        commit=os.environ.get('NQS_CODE_COMMIT','unbound')))


def role_ledger(plan: dict, inherited_rng: dict, prior: dict, *, historical: set[int]|None=None,
                historical_evidence: dict|None=None) -> dict:
    """Preserve next native RNG; fail closed on external, native or old collisions."""
    used=set(prior['seeds'].values()); history=set(historical or ());future=[]; following=deepcopy(inherited_rng)
    for _ in range(8):
        seeds,following=active._seeds(following);future.extend(seeds)
    external=list(plan['seeds'].values())
    if (len(set(future))!=len(future) or set(future)&(used|history)
            or set(external)&(used|history|set(future))):
        raise ValueError('new external/native/consumed legacy role collision')
    return dict(legacy_external_seeds=sorted(used),available_historical_seeds=sorted(history),
        available_history_evidence=historical_evidence or {},new_external_seeds=deepcopy(plan['seeds']),
        future_native_role_seeds=future,original_next_rng_sha256=canonical_sha256(inherited_rng),
        consumed_failed_legacy_roles_remain_excluded=True,native_rng_not_advanced_by_gate=True)


def verify_legacy_physical_sources() -> dict:
    """Prove transitive legacy physics bytes equal their frozen Git version.

    A cache identity based only on the two old entry files is not sufficient
    if an imported amplitude/H implementation has changed.  Missing historical
    Git objects fail closed rather than silently accepting a new environment.
    """
    from su2_nqs.frontier.provenance import transitive_local_source_manifest
    manifest=transitive_local_source_manifest([Path(old.__file__)],root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    for path,digest in manifest.items():
        try:raw=subprocess.check_output(['git','-C',str(ROOT),'show',OLD_COMMIT+':'+path],timeout=15,stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
            raise ValueError('historical physical source unavailable: '+path) from e
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('legacy transitive physical source changed: '+path)
    return dict(commit=OLD_COMMIT,files=manifest,all_legacy_transitive_bytes_equal=True)


def verify_legacy(plan: dict) -> tuple[dict, dict, dict]:
    """Verify all small files, complete inputs, source version and inherited RNG."""
    from su2_nqs.vmc import cached_amplitude_operators as engine
    prior=legacy_plan(); directory=Path(plan['legacy_directory'])
    for name,digest in OLD_FILES.items():
        if old.file_sha(directory/name)!=digest:raise ValueError('legacy artifact seal changed: '+name)
    source=canonical_sha256(dict(plan=prior,driver=old.file_sha(Path(old.__file__)),
        engine=old.file_sha(Path(engine.__file__)),commit=OLD_COMMIT))
    reg=json.loads((directory/'registration.json').read_text())
    if (source!=OLD_SOURCE or reg['plan']!=prior or reg['source_identity']!=source
            or reg['code_commit']!=OLD_COMMIT):
        raise ValueError('explicit legacy numerical source migration failed')
    states,profiles=old.load_states(prior)
    state=states['wide28']
    if canonical_sha256(state['metadata']['rng'])!=OLD_RNG_SHA:
        raise ValueError('original next native RNG changed')
    registration_path=Path(prior['inputs']['parent_registration']['path'])
    parent_registration=json.loads(registration_path.read_text())
    historical,history_evidence=legacy.historical_role_seeds(registration_path.parent)
    historical.update(parent_registration.get('plan',{}).get('seeds',{}).values())
    for key in ('gate_transition_seeds','future_native_role_seeds'):
        historical.update(parent_registration.get('role_streams',{}).get(key,[]))
    ledger=role_ledger(plan,state['metadata']['rng'],prior,historical=historical,historical_evidence=history_evidence)
    physical_sources=verify_legacy_physical_sources()
    ready=json.loads((directory/'wide28_ready.json').read_text())
    if (ready['source_identity']!=OLD_SOURCE or ready['plan_sha256']!=OLD_PLAN_SHA
            or ready['native_rng_sha256']!=OLD_RNG_SHA or ready['native_RNG_advanced'] is not False
            or ready['timings_sha256']!=canonical_sha256(ready['timings'])
            or ready['fit_panel_sha256']!=OLD_FILES['fit_panel.npz']
            or ready['proposal_sha256']!=OLD_FILES['wide28_full_proposal.pkl']):
        raise ValueError('legacy ready panel/proposal/timing binding failed')
    return state,profiles['wide28'],dict(role_streams=ledger,timing=ready['timings'],legacy_physical_sources=physical_sources,
        verified_legacy_file_sha256=deepcopy(OLD_FILES),legacy_source_identity=source)


def cached_fit(plan: dict, rt: dict, state: tuple, rows, psi, hpsi, q, *, cache=None,
               identity: str, progress=None) -> tuple[tuple, dict, object]:
    """One exact all-native-column SR solve; volatile J is rebuilt for each parent."""
    from su2_nqs.vmc.cached_amplitude_operators import build_cached_amplitude_jacobian,solve_cached_projected_sr
    theta,unravel,batch=old.batch_function(rt,state[0])
    if cache is None:
        cache=build_cached_amplitude_jacobian(batch,theta,rows,psi,q,row_batch_size=2,
            maximum_host_bytes=plan['maximum_host_array_bytes'],source_identity=identity,progress=progress)
    cache.verify(theta,rows,psi,q,source_identity=identity)
    d,g,info=solve_cached_projected_sr(cache,psi,hpsi,q,damping=.5,
        parameter_chunk_size=16384,maximum_host_bytes=plan['maximum_host_array_bytes'],
        maximum_device_bytes=plan['maximum_peak_device_bytes'],progress=progress)
    norm=float(info['raw_fisher_norm'])
    if (not info['numerically_valid'] or not np.isfinite(norm) or norm<=0
            or float(info['relative_linear_residual'])>=1e-7):
        raise ArithmeticError('full-native cached SR direction invalid')
    scale=.015/norm
    proposed=unravel(jnp.asarray(np.asarray(theta)-scale*np.asarray(d)))
    moved=(state[1][0]+1,state[1][1],state[1][2])
    result=dict(info,accepted=True,update_scale=scale,sampled_tangent_FS=.015,
        applied_energy_directional_derivative=-scale*float(np.vdot(g,d).real),
        full_current_parent_J_rebuilt_after_each_update=True)
    return (proposed,moved,deepcopy(state[2])),result,cache


def release_cache(cache) -> None:
    """Close the read-only mapped J before constructing sampler bank caches."""
    raw=cache.jacobian
    if isinstance(raw,np.memmap):raw._mmap.close()
    gc.collect()


def migrate_fit(plan: dict, rt: dict, parent: tuple, out: Path, *, replay: bool=False) -> tuple:
    """Hash all J bytes, identity-check, exact solve replay and native JVP replay."""
    from su2_nqs.vmc.cached_amplitude_operators import load_cached_amplitude_jacobian
    directory=Path(plan['legacy_directory']);tick=perf_counter()
    with np.load(directory/'fit_panel.npz') as f:
        rows,psi,hpsi,q=(np.asarray(f[k]) for k in ('rows','psi','Hpsi','density'))
    if rows.shape!=(1024,144) or psi.shape!=(1024,) or q.shape!=(1024,):
        raise ValueError('complete legacy fit panel required')
    cache=load_cached_amplitude_jacobian(directory/'wide_actual1024_J.npy',
        maximum_host_bytes=plan['maximum_host_array_bytes'],verify_content=True)
    proposal,sr,cache=cached_fit(plan,rt,parent,rows,psi,hpsi,q,cache=cache,identity=OLD_SOURCE)
    expected,_=legacy.restart.load_state(directory/'wide28_full_proposal.pkl')
    errors=[old.replay_error(a,b) for a,b in zip(jax.tree.leaves(proposal[:2]),jax.tree.leaves(expected[:2]),strict=True)]
    if canonical_sha256(proposal[2])!=canonical_sha256(expected[2]):raise ArithmeticError('proposal RNG replay changed')
    # Independently differentiate the actual full native projected amplitude.
    actual_FS=old.measured_tangent_FS(rt,parent[0],proposal[0],rows,psi,q)
    # Verify every component, not a tuple coerced to one scalar amplitude.
    with np.load(ROOT/legacy_plan()['inputs']['saved_H_products']['path']) as f:
        saved_rows,saved=np.asarray(f['rows']),np.asarray(f['wide'])
    current=jax.block_until_ready(rt['H'](parent[0],jnp.asarray(saved_rows)))
    H_errors=[old.replay_error(a,b) for a,b in zip(current,saved,strict=True)]
    release_cache(cache);del cache;gc.collect()
    receipt=dict(legacy_source_identity=OLD_SOURCE,new_source_identity=source_identity(plan),
        legacy_plan_sha256=OLD_PLAN_SHA,legacy_commit=OLD_COMMIT,
        legacy_input_seals=deepcopy(OLD_FILES),full_cache_content_verified=True,
        cache_identity_checked_against_exact_parent_rows_psi_Q=True,
        maximum_state_replay_error=max(errors,default=0.),native_H_component_replay_errors=H_errors,
        independently_measured_tangent_FS=actual_FS,full_native_direction=True,
        raw_J_mapping_closed_before_sampler_construction=True,
        native_RNG_advanced=False,wall_seconds=perf_counter()-tick,
        proposal_sha256=OLD_FILES['wide28_full_proposal.pkl'])
    if not replay:atomic_json(out/'cache_migration.json',receipt)
    return proposal,sr,receipt


def select_geometry(phi: np.ndarray, q: np.ndarray) -> tuple[int,dict]:
    """Amplitude-only selection and wholly separate veto; never energy selection."""
    parent=len(FACTORS)-1
    if phi.shape!=(1024,8) or q.shape!=(1024,):raise ValueError('complete1024/8-factor geometry required')
    selection=[common_proposal_FS(phi[:512][:,[parent,i]],q[:512],groups=16) for i in range(parent)]
    eligible=[i for i,x in enumerate(selection) if 1e-12<x['FS_radians'] and x['nominal_upper']<=.02]
    chosen=min(eligible,key=lambda i:abs(selection[i]['FS_radians']-.015)) if eligible else parent
    result=dict(selection=selection,selected_index=chosen,factors=FACTORS.tolist(),
        target_FS=.015,hard_upper_FS=.02,guard_not_used_for_reselection=True)
    if chosen==parent:return parent,dict(result,accepted=False,reason='no_eligible_finite_step')
    guard=common_proposal_FS(phi[512:][:,[parent,chosen]],q[512:],groups=16)
    ratio=guard['FS_radians']/selection[chosen]['FS_radians']
    passed=guard['nominal_upper']<=.02 and guard['FS_radians']>=.0075 and .5<=ratio<=2
    return (chosen if passed else parent),dict(result,accepted=bool(passed),guard=guard,
        guard_to_selection_ratio=ratio,reason='passed' if passed else 'independent_geometry_veto')


def drawer(plan: dict, rt: dict, banks: list) -> BudgetedTriangularOrbitDrawer:
    """No 158-GiB J is live here; account for native/canonical bank work explicitly."""
    return BudgetedTriangularOrbitDrawer(banks,rt,per_bank_bytes=plan['sampler_per_bank_bytes'],
        aggregate_bytes=plan['sampler_aggregate_bytes'],host_array_bytes=plan['maximum_host_array_bytes'],
        other_live_host_bytes=plan['sampler_other_live_bytes'])


def finite_geometry(plan: dict, rt: dict, parent: tuple, proposal: tuple, path: Path,
                    *, selection_seed: int, guard_seed: int, replay: bool=False) -> tuple[tuple,dict]:
    """Reuse frozen samplers across two independent streams, preserving all q/m."""
    started=perf_counter();sampling_seconds=None
    if not replay:path.mkdir()
    delta=jax.tree.map(lambda n,o:n-o,proposal[0],parent[0])
    candidates=[jax.tree.map(lambda p,d:p+f*d,parent[0],delta) for f in FACTORS]
    banks=[active.canonical_active_sampling_bank(p,rt['cfg'],rt['profile'])[0] for p in candidates]
    if replay:
        with np.load(path/'fixture.npz') as f:rows=f['rows']
    else:
        sampler=drawer(plan,rt,banks)
        selection,sd=sampler.draw(512,selection_seed,bridge=False)
        guard,gd=sampler.draw(512,guard_seed,bridge=False)
        rows=np.concatenate((selection,guard))
        save_checkpoint(path/'draws.pkl',(sd,gd),metadata=dict(role='separate_selection_guard',
            selection_seed=selection_seed,guard_seed=guard_seed))
        atomic_json(path/'sampler_budget.json',sampler.diagnostics)
        sampling_seconds=sampler.diagnostics['constructor_seconds']+sd['draw_seconds']+gd['draw_seconds']
        del sampler;gc.collect()
    raw=active.make_feature_lut_amplitude(rt['cfg'],active_top_widths=rt['profile'])
    groups=jnp.asarray(rt['images'].reshape(-1,16,144))
    def one(bank,row):
        def group(ids):
            v=jax.vmap(raw,in_axes=(None,0))(bank,row[ids])
            return jnp.stack((v.sum(),jnp.sum(v*v)))
        return jnp.sum(jax.lax.map(group,groups),axis=0)/1728
    query=jax.jit(lambda b,x:jax.lax.map(lambda r:one(b,r),x,batch_size=4))
    columns=[]
    for i,bank in enumerate(banks):
        active.stage_budget(rt,rt.get('audit_reserve_seconds',120.)+120.,'complete geometry column')
        tick=perf_counter()
        columns.append(np.asarray(jax.block_until_ready(query(rt['prepare'](bank),rows))))
        old.check_peak(plan)
        if not replay:atomic_json(path/'progress.json',dict(completed_columns=i+1,records=1024,seconds=perf_counter()-tick))
    values=np.stack(columns,axis=1);phi=values[:,:,0];q=values[:,:,1].mean(1)
    selected,info=select_geometry(phi,q)
    if replay:
        with np.load(path/'fixture.npz') as f:
            old.replay_error(phi,f['phi']);old.replay_error(q,f['density'])
            if selected!=int(f['selected']):raise ArithmeticError('geometry decision replay changed')
    else:
        np.savez_compressed(path/'fixture.npz',rows=rows,phi=phi,density=q,selected=selected)
        atomic_json(path/'result.json',old.clean(info))
    accepted=selected!=len(FACTORS)-1
    result=(candidates[selected],proposal[1],deepcopy(parent[2])) if accepted else parent
    return result,dict(info,selected_index=selected,selection_seed=selection_seed,guard_seed=guard_seed,
        maximum_factor=float(FACTORS[0]),records=1024,complete_coherent_A1=True,
        sampling_and_constructor_seconds=sampling_seconds,full_geometry_wall_seconds=perf_counter()-started)


def paired_panel(plan: dict, rt: dict, first: tuple, second: tuple, path: Path,
                 *, count: int, groups: int, seed: int, statistics: int) -> dict:
    """Full-H fresh endpoint-mixture panel, exact normalized H-bridge density."""
    path.mkdir();banks=[active.canonical_active_sampling_bank(s[0],rt['cfg'],rt['profile'])[0] for s in (first,second)]
    sampler=drawer(plan,rt,banks);rows,draw=sampler.draw(count,seed,bridge=True)
    sampling_seconds=sampler.diagnostics['constructor_seconds']+draw['draw_seconds']
    save_checkpoint(path/'draw.pkl',draw,metadata=dict(role='fresh_full_H_pair',seed=seed))
    del sampler;gc.collect()
    values,warm=active.products(rt,banks,rows,path,'paired',8)
    q=full_support_sz0_density(values[:,:,2].real.mean(1),values[:,:,3].real.mean(1),n_sites=144,epsilon=.125)
    moments=paired_iid_moments(values[:,:,0],values[:,:,1],q,np.repeat(np.arange(groups),count//groups),n_sites=144,seed=statistics)
    np.savez_compressed(path/'products.npz',rows=rows,products=values,proposal=q)
    report=dict(moments=moments,coverage=legacy.coverage_checks(moments,count),records=count,groups=groups,
        seed=seed,statistics_seed=statistics,full_H_centers=2*count,warm_H_seconds=warm,
        endpoint_parameter_sha256=[pytree_sha256(s[0]) for s in (first,second)],
        sampling_and_constructor_seconds=sampling_seconds,
        actual_peak_device_bytes=old.check_peak(plan),complete_A1_images=1728,physical_H_bonds=864)
    atomic_json(path/'audit.json',old.clean(report));return report


def bilateral_panel(plan: dict, rt: dict, parent: tuple, candidate: tuple, path: Path) -> dict:
    """Differentiate complete Hpsi, retaining both coherent energy-slope terms."""
    path.mkdir();bank,note=active.canonical_active_sampling_bank(parent[0],rt['cfg'],rt['profile'])
    sampler=drawer(plan,rt,[bank]);rows,draw=sampler.draw(512,plan['seeds']['bilateral'],bridge=True)
    sampling_seconds=sampler.diagnostics['constructor_seconds']+draw['draw_seconds']
    save_checkpoint(path/'draw.pkl',draw,metadata=dict(role='fresh_bilateral_H_JVP',seed=plan['seeds']['bilateral']))
    del sampler;gc.collect()
    delta=jax.tree.map(lambda n,o:n-o,candidate[0],parent[0])
    action=jax.jit(lambda p,d,x:legacy.native_h_jvp(rt,p,d,x))
    primals=[];tangents=[];seconds=[]
    for i,row in enumerate(rows):
        active.stage_budget(rt,120.,'complete native H/JVP center');tick=perf_counter()
        p,d=jax.block_until_ready(action(parent[0],delta,row[None]))
        primals.append(np.stack([np.asarray(x) for x in p],axis=1)[0])
        tangents.append(np.stack([np.asarray(x) for x in d],axis=1)[0]);seconds.append(perf_counter()-tick)
        old.check_peak(plan)
        if i%16==0:atomic_json(path/'progress.json',dict(completed=i+1,records=512,seconds=seconds))
    p,d=np.asarray(primals),np.asarray(tangents)
    q=full_support_sz0_density(p[:,2].real/note['original_raw_norm'],p[:,3].real/note['original_raw_norm'],n_sites=144,epsilon=.125)
    result=bilateral_direction_moments(p[:,0],p[:,1],d[:,0],d[:,1],q,groups=32,n_sites=144)
    np.savez_compressed(path/'products.npz',rows=rows,products=p,derivatives=d,proposal=q,seconds=seconds)
    result.update(seed=plan['seeds']['bilateral'],primal_H_centers=512,tangent_H_centers=512,
        sampling_and_constructor_seconds=sampling_seconds,
        both_coherent_terms=True,same_native_scale=True,actual_peak_device_bytes=old.check_peak(plan))
    atomic_json(path/'bilateral.json',old.clean(result));return result


def gate_forecast(plan: dict, timing: dict) -> dict:
    """Use actual full cold-parent work; solve-only warm time is not a step cost."""
    science=512*timing['H_JVP_warm_seconds_per_center']+2048*timing['H_seconds_per_center']+8*timing['actual1024_amplitude_warm_seconds']
    return dict(forecast_seconds=1.5*science+plan['restore_reserve_seconds']+1800+300,
        science_kernel_floor_seconds=science,restore_and_migration_reserve_seconds=plan['restore_reserve_seconds'],
        frozen_cache_math_replay_not_new_training=True,no_reduced_scientific_panels=True)


def pilot_forecast(plan: dict, timing: dict, *, gate_measurements: dict|None=None) -> dict:
    """Explicit 8 cold-parent updates + first replay + paired4096 audit estimate."""
    step=1024*timing['H_seconds_per_center']+timing['SR_cold_seconds']+8*timing['actual1024_amplitude_warm_seconds']
    terminal=8192*timing['H_seconds_per_center']
    floor=8*step+timing['SR_cold_seconds']+terminal
    before=1.5*floor+3600
    sampled=None
    if gate_measurements is not None:
        geometry=float(gate_measurements['geometry']['full_geometry_wall_seconds'])
        single=float(gate_measurements['bilateral']['sampling_and_constructor_seconds'])
        paired=float(gate_measurements['confirmation']['sampling_and_constructor_seconds'])
        if not np.isfinite([geometry,single,paired]).all() or min(geometry,single,paired)<0:
            raise ValueError('actual gate sampler/geometry timings required')
        # Conservative double charge of the geometry kernels already in floor:
        # each new parent needs its own canonical8-bank caches and both streams.
        sampled=1.5*(8*(geometry+2*single)+4*paired)
    return dict(raw_kernel_floor_seconds=floor,safety_factor=1.5,
        forecast_before_sampling_seconds=before,terminal_H_centers=8192,
        measured_sampler_geometry_reserve_seconds=sampled,
        full_forecast_seconds=None if sampled is None else before+sampled,
        requested_wall_cap_seconds=plan['pilot_wall_cap_seconds'],
        sampling_construction_and_compilation_must_be_measured=True,
        forty_eight_hours_not_certified=True)


def science_artifact_seals(out: Path) -> dict[str,str]:
    """Seal complete raw geometry, two-sided H/JVP and independent full-H data."""
    names=('geometry/fixture.npz','geometry/draws.pkl','geometry/result.json',
        'geometry/sampler_budget.json','bilateral/products.npz','bilateral/draw.pkl',
        'bilateral/bilateral.json','confirmation/products.npz','confirmation/draw.pkl',
        'confirmation/audit.json','candidate.pkl')
    return {name:old.file_sha(out/name) for name in names if (out/name).is_file()}


def verify_science_artifacts(out: Path, result: dict, *, complete: bool) -> None:
    """A gate certificate cannot be detached from its actual independent panels."""
    expected=result.get('science_artifact_sha256',{})
    minimum=11 if complete else 4
    if len(expected)<minimum:raise ValueError('incomplete scientific artifact seals')
    if any(Path(name).is_absolute() or '..' in Path(name).parts for name in expected):
        raise ValueError('scientific artifacts must remain within new campaign')
    current=science_artifact_seals(out)
    if current!=expected:raise ValueError('frozen independent scientific artifact changed')


def gate_phase(plan: dict, rt: dict, parent: tuple, out: Path, binding: dict) -> dict:
    """One frozen full-network direction; every validation role is fresh."""
    forecast=gate_forecast(plan,binding['timing']);atomic_json(out/'cost_forecast.json',forecast)
    active.stage_budget(rt,forecast['forecast_seconds'],'complete unchanged fresh wide direction gate')
    proposal,sr,migration=migrate_fit(plan,rt,parent,out)
    candidate,geometry=finite_geometry(plan,rt,parent,proposal,out/'geometry',
        selection_seed=plan['seeds']['selection'],guard_seed=plan['seeds']['guard'])
    if not geometry['accepted']:
        result=dict(status='fresh_geometry_veto_no_training',preliminary_passed=False,geometry=geometry)
    else:
        legacy.restart.save_state(out/'candidate.pkl',candidate,0,45)
        bilateral=bilateral_panel(plan,rt,parent,candidate,out/'bilateral')
        confirmation=paired_panel(plan,rt,parent,candidate,out/'confirmation',count=1024,groups=32,
            seed=plan['seeds']['confirmation'],statistics=plan['seeds']['confirmation_statistics'])
        decision=legacy.gate_decision(bilateral,confirmation,geometry_passed=True,replay_passed=False)
        result=dict(status='fresh_full_direction_requires_process_replay',geometry=geometry,
            bilateral=bilateral,confirmation=confirmation,decision=decision,
            preliminary_passed=all(v for k,v in decision['checks'].items() if k!='cross_process_replay'),
            candidate_sha256=old.file_sha(out/'candidate.pkl'),
            fit_predicted_energy_slope_per_site=float(sr['applied_energy_directional_derivative'])*float(FACTORS[geometry['selected_index']])/144)
    result.update(cache_migration_sha256=old.file_sha(out/'cache_migration.json'),
        science_artifact_sha256=science_artifact_seals(out),
        source_identity=source_identity(plan),native_RNG_advanced=False,gate_candidate_is_production_step=False)
    atomic_json(out/'gate_result.json',old.clean(result));return result


def replay_phase(plan: dict, rt: dict, parent: tuple, out: Path) -> dict:
    """Fresh process restores math/geometry only, never consumes confirmation twice."""
    result=json.loads((out/'gate_result.json').read_text())
    verify_science_artifacts(out,result,complete=bool(result.get('preliminary_passed')))
    cert=dict(passed=False,task=TASK,plan_sha256=canonical_sha256(plan),source_identity=source_identity(plan),
        parent_parameter_sha256=pytree_sha256(parent[0]),parent_optimizer_sha256=pytree_sha256(parent[1]),
        parent_rng_sha256=canonical_sha256(parent[2]),
        gate_result_sha256=old.file_sha(out/'gate_result.json'),cache_migration_sha256=old.file_sha(out/'cache_migration.json'),
        science_artifact_sha256=result['science_artifact_sha256'],
        gate_candidate_is_production_step=False)
    if result.get('preliminary_passed'):
        proposal,_,receipt=migrate_fit(plan,rt,parent,out,replay=True)
        candidate,geometry=finite_geometry(plan,rt,parent,proposal,out/'geometry',
            selection_seed=plan['seeds']['selection'],guard_seed=plan['seeds']['guard'],replay=True)
        frozen,_=legacy.restart.load_state(out/'candidate.pkl')
        if old.file_sha(out/'candidate.pkl')!=result['candidate_sha256']:raise ValueError('candidate seal changed')
        errors=[old.replay_error(a,b) for a,b in zip(jax.tree.leaves(candidate[:2]),jax.tree.leaves(frozen[:2]),strict=True)]
        decision=legacy.gate_decision(result['bilateral'],result['confirmation'],geometry_passed=geometry['accepted'],replay_passed=True)
        cert.update(decision,candidate_sha256=result['candidate_sha256'],maximum_replay_error=max(errors,default=0.),
            controls=dict(damping=.5,target_FS=.015,maximum_FS=.02),
            terminal_audit_must_be_new4096_not_gate_confirmation=True)
    atomic_json(out/'gate_certificate.json',old.clean(cert));return dict(status='direction_gate_passed' if cert['passed'] else 'direction_gate_failed_no_training',certificate=cert)


def validate_pilot_release(release: dict, plan: dict, out: Path, parent: tuple) -> dict:
    """A distinct root release, exactly six passed scientific gates, max8/nochain."""
    cert=json.loads((out/'gate_certificate.json').read_text())
    result=json.loads((out/'gate_result.json').read_text())
    verify_science_artifacts(out,result,complete=True)
    if (cert.get('passed') is not True or len(cert.get('checks',{}))!=6 or not all(cert['checks'].values())
            or cert.get('plan_sha256')!=canonical_sha256(plan) or cert.get('source_identity')!=source_identity(plan)
            or cert.get('parent_parameter_sha256')!=pytree_sha256(parent[0])
            or cert.get('parent_optimizer_sha256')!=pytree_sha256(parent[1])
            or cert.get('parent_rng_sha256')!=canonical_sha256(parent[2])
            or cert.get('gate_result_sha256')!=old.file_sha(out/'gate_result.json')
            or cert.get('cache_migration_sha256')!=old.file_sha(out/'cache_migration.json')
            or cert.get('science_artifact_sha256')!=result['science_artifact_sha256']
            or cert.get('candidate_sha256')!=old.file_sha(out/'candidate.pkl')
            or release!=dict(root_authorized=True,task=TASK,maximum_attempts=8,
                wall_cap_seconds=259200,gate_certificate_sha256=old.file_sha(out/'gate_certificate.json'),
                restore_original_global45=True,automatically_chain=False)):
        raise PermissionError('new same-parent root release and passed six-check certificate required')
    return cert


def training_transition(plan: dict, rt: dict, state: tuple, path: Path, *, replay: bool=False) -> tuple:
    """All-W current-parent exact cached SR, independent finite FS and atomic state."""
    seeds,following=active._seeds(state[2]);tick=perf_counter()
    if not replay:path.mkdir()
    bank,note=active.canonical_active_sampling_bank(state[0],rt['cfg'],rt['profile'])
    if replay:
        with np.load(path/'fit_panel.npz') as f:rows,psi,hpsi,q=(f[k] for k in ('rows','psi','Hpsi','density'))
        # No redraw: replay fixture is not new statistical evidence.
        normalized=jax.block_until_ready(rt['H'](bank,jnp.asarray(rows[:2])))
        scale=np.sqrt(note['original_raw_norm'])
        old.replay_error(normalized[0]*scale,psi[:2]);old.replay_error(normalized[1]*scale,hpsi[:2])
    else:
        sampler=drawer(plan,rt,[bank]);rows,draw=sampler.draw(1024,seeds[0],bridge=True)
        save_checkpoint(path/'fit_draw.pkl',draw,metadata=dict(role='current_parent_training_fit',seed=seeds[0]))
        del sampler;gc.collect()
        products,warm=active.products(rt,[bank],rows,path,'fit',8)
        q=np.asarray(full_support_sz0_density(products[:,0,2].real,products[:,0,3].real,n_sites=144,epsilon=.125))
        scale=np.sqrt(note['original_raw_norm']);psi=products[:,0,0].real*scale;hpsi=products[:,0,1].real*scale
        np.savez_compressed(path/'fit_panel.npz',rows=rows,psi=psi,Hpsi=hpsi,density=q)
    identity=canonical_sha256(dict(source=source_identity(plan),parent=pytree_sha256(state[0]),
        optimizer=pytree_sha256(state[1]),rng=canonical_sha256(state[2])))
    proposal,sr,cache=cached_fit(plan,rt,state,rows,psi,hpsi,q,identity=identity,
        progress=lambda _:active.stage_budget(rt,rt['audit_reserve_seconds']+300,'complete current-parent derivative block'))
    release_cache(cache);del cache;gc.collect()
    candidate,geometry=finite_geometry(plan,rt,state,proposal,path/'geometry',selection_seed=seeds[1],guard_seed=seeds[2],replay=replay)
    candidate=(candidate[0],candidate[1],following)
    info=dict(accepted=bool(geometry['accepted']),seeds=seeds,geometry=geometry,
        parent_parameter_sha256=pytree_sha256(state[0]),next_rng_sha256=canonical_sha256(following),
        full_native_W_parameters=20748160,fit_records=1024,wall_seconds=perf_counter()-tick,
        SR={k:old.clean(v) for k,v in sr.items() if np.asarray(v).ndim==0},
        rejected_roles_consumed=True,current_parent_J_not_reused_after_update=True)
    if replay:
        expected,_=legacy.restart.load_state(path/'provisional.pkl')
        errors=[old.replay_error(a,b) for a,b in zip(jax.tree.leaves(candidate[:2]),jax.tree.leaves(expected[:2]),strict=True)]
        if canonical_sha256(candidate[2])!=canonical_sha256(expected[2]):raise ArithmeticError('first next RNG replay changed')
        info['maximum_state_replay_error']=max(errors,default=0.)
    else:
        legacy.restart.save_state(path/'provisional.pkl',candidate,0,int(state[1][0]))
        atomic_json(path/'transition.json',old.clean(info))
    return candidate,info


def pilot_phase(plan: dict, rt: dict, parent: tuple, out: Path, phase: str,
                release: dict, timing: dict) -> dict:
    """Separate bounded campaign, max8 attempts, indivisible4096 terminal panel."""
    validate_pilot_release(release,plan,out,parent);path=out/'pilot'
    forecast=pilot_forecast(plan,timing,gate_measurements=json.loads((out/'gate_result.json').read_text()))
    rt['audit_reserve_seconds']=1.5*8192*timing['H_seconds_per_center']+1800
    rt['sampling_reserve_seconds']=rt['audit_reserve_seconds']+300
    if phase=='audit':
        # The reserved terminal work is now being performed, not reserved a
        # second time; retain only a save/complete-row guard during its draw.
        rt['audit_reserve_seconds']=0.
        rt['sampling_reserve_seconds']=120.
    rt['warm_H_seconds']=timing['H_seconds_per_center']
    def freeze(state,attempts,accepted,reason):
        legacy.restart.save_state(path/'checkpoint.pkl',state,accepted,45+accepted)
        report=dict(attempts=attempts,completed_updates=accepted,absolute_step=45+accepted,
            parameter_sha256=pytree_sha256(state[0]),optimizer_sha256=pytree_sha256(state[1]),
            rng_sha256=canonical_sha256(state[2]),checkpoint_sha256=old.file_sha(path/'checkpoint.pkl'),
            stop_reason=reason,maximum_attempts=8,terminal_audit_not_used_for_selection=True)
        atomic_json(path/'endpoint_frozen.json',report);return report
    if phase=='pilot_prepare':
        path.mkdir();atomic_json(path/'selection.json',dict(started_epoch=time(),release=release,
            source_identity=source_identity(plan),cost_forecast=forecast))
        active.stage_budget(rt,forecast['full_forecast_seconds'],'measured8-step/sampling cost and untouched audit reserve')
        roles,following=active._seeds(parent[2])
        # Atomic recovery point before the very first random draw.  It is not
        # the committed checkpoint, and fresh-process replay cannot mistake it
        # for an accepted update.  Even a hard kill leaves used roles reserved.
        legacy.restart.save_state(path/'first_reserved_state.pkl',(parent[0],parent[1],following),0,45)
        atomic_json(path/'first_role_reservation.json',dict(reserved_consumed_roles=roles,
            inherited_rng_sha256=canonical_sha256(parent[2]),following_rng_sha256=canonical_sha256(following),
            source_parent_unchanged=True,maximum_attempts=8))
        try:candidate,info=training_transition(plan,rt,parent,path/'step1')
        except (TimeoutError,MemoryError) as e:
            roles,following=active._seeds(parent[2]);state=(parent[0],parent[1],following)
            frozen=freeze(state,1,0,'partial_first_attempt_roles_consumed')
            atomic_json(path/'first_prepare.json',dict(accepted=False,error=str(e),reserved_consumed_roles=roles))
            return dict(status='zero_update_partial_no_quality_claim',**frozen)
        legacy.restart.save_state(path/'first_provisional.pkl',candidate,0,45)
        atomic_json(path/'first_prepare.json',old.clean(info))
        if not info['accepted']:return dict(status='zero_update_geometry_veto',**freeze(candidate,1,0,'finite_FS_veto'))
        return dict(status='first_update_provisional_requires_process_replay',transition=info)
    selection=json.loads((path/'selection.json').read_text())
    if selection['release']!=release or selection['source_identity']!=source_identity(plan):raise ValueError('pilot release/source changed')
    if phase=='pilot_replay':
        info=json.loads((path/'first_prepare.json').read_text())
        if not info['accepted']:return dict(status='no_first_commit')
        if (path/'checkpoint.pkl').exists():raise FileExistsError('first update already committed')
        replayed,note=training_transition(plan,rt,parent,path/'step1',replay=True)
        atomic_json(path/'first_restore_replay.json',old.clean(note))
        return dict(status='first_restored_committed_once',**freeze(replayed,1,1,'first_verified'))
    state,meta=legacy.restart.load_state(path/'checkpoint.pkl')
    frozen=json.loads((path/'endpoint_frozen.json').read_text())
    if (old.file_sha(path/'checkpoint.pkl')!=frozen['checkpoint_sha256']
            or pytree_sha256(state[0])!=frozen['parameter_sha256']
            or canonical_sha256(state[2])!=frozen['rng_sha256']):raise ValueError('frozen prefix identity changed')
    attempts,accepted=frozen['attempts'],frozen['completed_updates']
    if phase=='pilot_train':
        if accepted<1:return dict(status='zero_updates_no_training')
        if not (path/'first_restore_replay.json').exists():raise PermissionError('first fresh-process replay required')
        last=json.loads((path/'first_prepare.json').read_text())['wall_seconds'];reason='maximum_eight_attempts'
        while attempts<8:
            if rt['remaining']()<rt['audit_reserve_seconds']+1.5*last+300:reason='terminal_audit_reserve';break
            attempts+=1
            # Reserve/consume roles atomically before heavy work; original coordinates survive interruption.
            roles,following=active._seeds(state[2])
            safe=(state[0],state[1],following)
            freeze(safe,attempts,accepted,'attempt_reserved_not_yet_committed')
            try:candidate,info=training_transition(plan,rt,state,path/f'step{attempts}')
            except (TimeoutError,MemoryError) as e:
                state=safe;reason='bounded_partial_roles_consumed';break
            state=candidate;last=info['wall_seconds']
            if info['accepted']:accepted+=1
            freeze(state,attempts,accepted,'update_committed' if info['accepted'] else 'finite_FS_veto')
            if not info['accepted']:reason='finite_FS_veto';break
        return dict(status='fixed_prefix_frozen_for_fresh4096',**freeze(state,attempts,accepted,reason))
    if phase!='audit':raise ValueError('unknown pilot phase')
    if accepted<1:return dict(status='zero_update_no_terminal_precision')
    audit=paired_panel(plan,rt,parent,state,path/'terminal_audit',count=4096,groups=64,
        seed=plan['seeds']['audit'],statistics=plan['seeds']['audit_statistics'])
    return dict(status='fresh4096_frozen_endpoint_audited',**frozen,terminal_audit=audit,
        confirmation_never_reused_as_final_precision=True)


def run(plan_path: Path, out: Path, phase: str, *, release_path: Path|None=None) -> dict:
    """One isolated process phase; old running/sealed campaign is never modified."""
    plan=json.loads(plan_path.read_text());validate_plan(plan,executing=True)
    if phase not in PHASES:raise ValueError('explicit registered phase required')
    if not out.is_absolute() or out.is_relative_to(ROOT):raise ValueError('output must be outside code release')
    if (jax.default_backend()!='gpu' or not all('A100' in d.device_kind for d in jax.devices())
            or int(os.environ.get('SLURM_MEM_PER_NODE','0'))<320*1024):
        raise RuntimeError('Kim A10080 with explicit320GiB host allocation required')
    if not re.fullmatch(r'[0-9a-f]{40}',os.environ.get('NQS_CODE_COMMIT','')):raise PermissionError('immutable full Git commit required')
    state,profile,binding=verify_legacy(plan);versions=old.verify_runtime_versions()
    parent=(state['params'],state['optimizer_state'],deepcopy(state['metadata']['rng']))
    report_path=out/(phase+'_result.json')
    if phase=='gate':
        if out.exists():raise FileExistsError('new immutable reentry output required')
        out.mkdir(parents=True)
        atomic_json(out/'registration.json',dict(plan=plan,plan_sha256=canonical_sha256(plan),
            source_identity=source_identity(plan),runtime_versions=versions,started_epoch=time(),**binding))
    reg=json.loads((out/'registration.json').read_text())
    if reg['plan']!=plan or reg['source_identity']!=source_identity(plan) or reg['runtime_versions']!=versions:
        raise ValueError('new registration plan/source/runtime changed')
    if report_path.exists() or (out/(phase+'_started.json')).exists():raise FileExistsError('scientific phase already consumed or partial')
    pilot=phase.startswith('pilot_') or phase=='audit'
    epoch=reg['started_epoch']
    if pilot and phase!='pilot_prepare':epoch=json.loads((out/'pilot/selection.json').read_text())['started_epoch']
    elif phase=='pilot_prepare':epoch=time()
    cap=plan['pilot_wall_cap_seconds'] if pilot else plan['gate_wall_cap_seconds']
    stop=[False]
    def stopping(*_):stop[0]=True
    for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,stopping)
    def remaining():return 0. if stop[0] else min(epoch+cap,float(os.environ.get('SLURM_JOB_END_TIME',epoch+cap)))-time()-300
    rt=old.make_runtime(legacy_plan(),'wide28',parent[0],profile);rt['remaining']=remaining
    atomic_json(out/(phase+'_started.json'),dict(started_epoch=time(),cap_seconds=cap,source_identity=source_identity(plan)))
    tick=perf_counter()
    try:
        if phase=='gate':result=gate_phase(plan,rt,parent,out,binding)
        elif phase=='replay':result=replay_phase(plan,rt,parent,out)
        else:
            if release_path is None:raise PermissionError('separate root pilot release required')
            result=pilot_phase(plan,rt,parent,out,phase,json.loads(release_path.read_text()),binding['timing'])
    except (TimeoutError,MemoryError) as e:
        result=dict(status='bounded_partial_no_width_or_statistical_fallback',error=str(e),
            original_parent_unchanged=True,partial_not_direction_certificate=True)
    result.update(phase=phase,wall_seconds=perf_counter()-tick)
    atomic_json(report_path,old.clean(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=PHASES,required=True);p.add_argument('--pilot-release',type=Path)
    a=p.parse_args();result=run(a.plan,a.output,a.phase,release_path=a.pilot_release)
    print(json.dumps({k:result[k] for k in ('status','phase','wall_seconds') if k in result}),flush=True)
    if result['status'].startswith('bounded_partial'):raise SystemExit(2)
