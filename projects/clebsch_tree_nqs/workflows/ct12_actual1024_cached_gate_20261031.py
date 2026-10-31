"""Same-global45 CT12 actual1024 comparison and two independently gated arms.

The original cached-score implementation is an immutable cost/control arm.
The true-w28 arm uses an exact host-resident full-W amplitude Jacobian once,
then reuses it for the unchanged weighted F SR algebra. No heads, truncation,
log-score approximation, sample reduction or automatic submission is allowed.
Gate proposals never consume the restored native training RNG or count as
training. Optional eight-step training requires a separate root release and
restores the selected arm's original global45 state, moments and next RNG.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import signal
from time import perf_counter, time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.full_support_proposal import full_support_sz0_density
from su2_nqs.vmc.importance_amplitude_sr import make_importance_amplitude_sr
from . import triangle_active_two_step_pilot as active
from . import ct12_direction_gate as legacy

ROOT = Path(__file__).resolve().parents[3]
TASK = 'CT12_global45_actual1024_exact_cached_comparison_gate_v1'
ARMS = ('original', 'wide28')
PHASES = ('compare', 'gate', 'replay', 'pilot_prepare', 'pilot_replay', 'pilot_train', 'audit')
ROLES = ('fit', 'original_selection', 'original_guard', 'wide28_selection', 'wide28_guard',
         'original_bilateral', 'wide28_bilateral', 'original_confirmation', 'wide28_confirmation',
         'original_confirmation_statistics', 'wide28_confirmation_statistics', 'audit', 'audit_statistics')
GEOMETRY_FACTORS = np.asarray([2.,1.5,1.,.75,.5,.25,.125,0.])


def clean(value):
    """Convert small diagnostics, never the cached [N,P] arrays, to JSON."""
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        a = np.asarray(value); return a.item() if a.ndim == 0 else a.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def file_sha(path: Path) -> str:
    """Hash an input without loading a checkpoint-sized duplicate into RAM."""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def verify_runtime_versions() -> dict:
    """Pin the existing shared environment without changing it or reading keys."""
    path=ROOT/'configs/remote/ct12_w28_gpu_preflight_20261030/runtime_packages.json'
    expected=json.loads(path.read_text())
    actual={name:importlib.metadata.version(name) for name in expected}
    if actual!=expected:raise RuntimeError('shared Python package versions differ from pinned passed-w28 runtime')
    return actual


def validate_plan(plan: dict, *, executing: bool = False) -> None:
    """Freeze full physical support, all scientific panels, and both real layouts."""
    fixed = dict(task=TASK, j2=.125, parent_global_step=45, fit_records=1024,
        calibration_records=1024, calibration_split=512, bilateral_records=512,
        bilateral_groups=32, confirmation_records=1024, confirmation_groups=32,
        audit_records=4096, audit_groups=64, steps=8, damping=.5,
        learning_rate=.03, epsilon=.125, target_FS=.015, maximum_FS=.02,
        complete_A1_images=1728, physical_H_bonds=864,
        original_parameters=1495080, wide_parameters=20748160,
        original_model=dict(length=12,multiplicity_cap=20,maximum_q=10),
        wide_model=dict(length=12,multiplicity_cap=28,maximum_q=10),
        maximum_peak_device_bytes=70*1024**3, maximum_host_array_bytes=240*1024**3,
        requested_host_GiB=320, compare_wall_cap_seconds=21600,
        original_gate_wall_cap_seconds=28800, wide_gate_wall_cap_seconds=64800,
        original_pilot_wall_cap_seconds=64800, wide_pilot_wall_cap_seconds=172800,
        restore_reserve_seconds=1800, save_reserve_seconds=300,
        derivative_row_batch=2, cached_parameter_chunk=16384, H_chunk=8,
        projection_chunk=16, action_chunk=4, score_chunk=1,
        gate_geometry_factors=GEOMETRY_FACTORS.tolist(), minimum_guard_FS=.0075,
        pilot_geometry_factors=active.FACTORS.tolist(),
        pilot_cache_policy='volatile_host_exact_cache_per_current_parent',
        automatically_submit=False, automatically_start_pilot=False)
    if any(plan.get(k) != v for k,v in fixed.items()):
        raise ValueError('unchanged true-width global45 actual1024/full-H protocol required')
    if any(k in plan for k in ('nodelist','heads','frozen_blocks','low_rank_J','reduced_records')):
        raise ValueError('no pinned node, heads, frozen columns, approximate J or smaller panels')
    seeds = plan.get('seeds', {})
    if set(seeds) != set(ROLES) or len(set(seeds.values())) != len(ROLES):
        raise ValueError('all external roles need distinct preregistered streams')
    if any(isinstance(s,bool) or not isinstance(s,int) or not 0 <= s < 2**32 for s in seeds.values()):
        raise ValueError('uint32 role seeds required')
    if set(plan.get('inputs', {})) != {'parent','expanded','parent_registration','diagnostic_rows','saved_H_products'}:
        raise ValueError('explicit source state, expansion, actual profile and fixed replay rows required')
    for spec in plan['inputs'].values():
        if not isinstance(spec.get('sha256'),str) or len(spec['sha256']) != 64:
            raise ValueError('all immutable input file SHA256 values required')
        if not Path(spec['path']).is_absolute() or Path(spec['path']).is_relative_to(ROOT):
            raise ValueError('scientific inputs must be absolute and outside code-only release')
    if executing and plan.get('execution_authorized') is not True:
        raise PermissionError('root execution release required; this driver does not submit')


def check_peak(plan: dict) -> int:
    """Require an actual allocator high-water value; missing telemetry is not zero."""
    peaks = []
    for device in jax.local_devices():
        stats = device.memory_stats()
        if stats is None or 'peak_bytes_in_use' not in stats:
            raise RuntimeError('device peak telemetry unavailable; memory not certified')
        peaks.append(int(stats['peak_bytes_in_use']))
    peak = max(peaks)
    if peak > plan['maximum_peak_device_bytes']: raise MemoryError('70GiB postcall guard exceeded')
    return peak


def replay_error(actual, reference, *, tolerance: float = 2e-11) -> float:
    """A scale-relative comparison, with explicit finite/shape and zero checks."""
    a,b = np.asarray(actual),np.asarray(reference)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ArithmeticError('nonfinite or shape-mismatched replay')
    norm = np.linalg.norm(b)
    error = float(np.linalg.norm(a-b)/norm) if norm else (0. if not np.any(a) else np.inf)
    # This tolerance is a replay diagnostic, never a wavefunction/density floor.
    coordinate_scale = np.max(abs(b), initial=0.)
    if error > tolerance or np.any(abs(a-b) > tolerance*(abs(b)+coordinate_scale)):
        raise ArithmeticError('same physical scale replay failed')
    return error


def load_states(plan: dict) -> tuple[dict, dict]:
    """Bind both saved states/moments and identical original next scientific RNG."""
    for spec in plan['inputs'].values():
        if file_sha(ROOT/spec['path']) != spec['sha256']: raise ValueError('input seal changed: '+spec['path'])
    states = {arm: load_checkpoint(ROOT/plan['inputs'][key]['path'])
              for arm,key in (('original','parent'),('wide28','expanded'))}
    reg = json.loads((ROOT/plan['inputs']['parent_registration']['path']).read_text())
    profile = tuple({int(q):int(d) for q,d in row.items()} for row in reg['plan']['active_profile'])
    if reg['plan']['model'] != plan['original_model']: raise ValueError('source actual model changed')
    native_seeds = []; following = deepcopy(states['original']['metadata']['rng'])
    for _ in range(8):
        seeds, following = active._seeds(following); native_seeds.extend(seeds)
    past = set(reg['plan'].get('seeds', {}).values())
    for key in ('gate_transition_seeds','future_native_role_seeds'):
        past.update(reg.get('role_streams',{}).get(key,[]))
    if set(plan['seeds'].values()) & (set(native_seeds)|past): raise ValueError('external/native/historical role collision')
    for arm,state in states.items():
        m = state['metadata']; count = sum(a.size for a in jax.tree.leaves(state['params']))
        if (m['absolute_completed_steps'] != 45 or int(state['optimizer_state'][0]) != 45
                or count != plan['original_parameters' if arm=='original' else 'wide_parameters']
                or pytree_sha256(state['params']) != m['parameter_sha256']
                or pytree_sha256(state['optimizer_state']) != m['optimizer_sha256']):
            raise ValueError('global45 native parameter/optimizer identity changed')
    if states['original']['metadata']['rng'] != states['wide28']['metadata']['rng']:
        raise ValueError('expansion consumed native scientific RNG')
    return states, dict(original=profile,
        wide28=tuple({q:28 for q in range(0,11,2)} for _ in range(14))+({0:1},))


def make_runtime(plan: dict, arm: str, params: dict, profile: tuple) -> dict:
    """Reuse established fullA1/H864, differentiating local LUT preparation."""
    from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
    from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
    from su2_nqs.vmc.exchange_bridge_importance import make_orbit_exchange_bridge_products
    cfg = MixedClebschConfig(**plan['original_model' if arm=='original' else 'wide_model'])
    images = np.asarray(active.triangular_hnf_gamma_a1_group(((12,0),(0,12))),np.int32)
    bonds = active.registered_bonds(.125); edges = np.asarray([(a,b) for a,b,_ in bonds],np.int32)
    probability = np.abs([w for _,_,w in bonds]); probability /= probability.sum()
    if len(images)!=1728 or len(bonds)!=864: raise ArithmeticError('incomplete physical support')
    phi = active.make_feature_lut_projected_amplitude(cfg,images,chunk=16,active_top_widths=profile)
    prepare = lambda p:active.prepare_feature_lut_bank(p,cfg)
    bridge,metadata = make_orbit_exchange_bridge_products(
        lambda e:make_feature_lut_exchange_queries(cfg,e,active_top_widths=profile),
        images,bonds,probability,n_sites=144,projection_chunk=16)
    rt = dict(cfg=cfg,profile=profile,images=images,inverse=np.argsort(images,axis=1),
        bonds=bonds,edges=edges,probability=probability,metadata=metadata,prepare=prepare,
        amplitude=jax.jit(jax.vmap(phi,in_axes=(None,0))),
        H=jax.jit(lambda p,x:bridge(prepare(p),x)),epsilon=.125,j2=.125)
    rt['update'] = make_importance_amplitude_sr(lambda p,x:phi(prepare(p),x),params,
        damping=.5,score_chunk=1,parameter_chunk_size=8192,sr_tangent_fs_radius=.015,
        prepare=prepare,prepared_amplitude=phi) if arm=='original' else None
    return rt


def batch_function(rt: dict, params: dict):
    """Free native coordinates include preparation inside every AD closure."""
    theta,unravel = ravel_pytree(params)
    return theta,unravel,lambda flat,rows:rt['amplitude'](rt['prepare'](unravel(flat)),rows)


def cache_source(plan: dict) -> str:
    """Bind cache to plan and implementation bytes, not only a nominal width."""
    from su2_nqs.vmc import cached_amplitude_operators as engine
    return canonical_sha256(dict(plan=plan,driver=file_sha(Path(__file__)),engine=file_sha(Path(engine.__file__)),
        commit=os.environ.get('NQS_CODE_COMMIT','unspecified_by_launcher')))


def cached_proposal(plan: dict, rt: dict, params: dict, moments: tuple, rows,
                    psi, hpsi, density, *, cache=None, storage: Path|None=None,
                    progress=None):
    """Exact cached F solve, unchanged ridge/force and physical tangent radius."""
    from su2_nqs.vmc.cached_amplitude_operators import build_cached_amplitude_jacobian,solve_cached_projected_sr
    theta,unravel,batch = batch_function(rt,params); identity = cache_source(plan)
    if cache is None:
        cache = build_cached_amplitude_jacobian(batch,theta,rows,np.asarray(psi,np.float64),
            np.asarray(density,np.float64),row_batch_size=2,maximum_host_bytes=plan['maximum_host_array_bytes'],
            storage_path=storage,source_identity=identity,progress=progress)
    cache.verify(theta,rows,psi,density,source_identity=identity)
    direction,gradient,info = solve_cached_projected_sr(cache,psi,np.asarray(hpsi,np.float64),density,
        damping=.5,parameter_chunk_size=16384,maximum_host_bytes=plan['maximum_host_array_bytes'],
        maximum_device_bytes=plan['maximum_peak_device_bytes'],progress=progress)
    norm = float(info['raw_fisher_norm'])
    if not np.isfinite(norm) or norm<=0:raise ArithmeticError('zero/invalid physical SR direction')
    scale = .015/norm
    valid = bool(info['numerically_valid']) and float(info['relative_linear_residual'])<1e-7
    proposed = np.asarray(theta)-scale*np.asarray(direction)
    if not valid or not np.isfinite(proposed).all(): raise ArithmeticError('cached full-W SR proposal invalid')
    next_moments = (moments[0]+1,moments[1],moments[2])
    diagnostics = dict(info,accepted=True,update_scale=scale,sampled_tangent_FS=scale*norm,
        relative_SR_residual=info['relative_linear_residual'],energy_gradient_norm=info['gradient_norm'],
        applied_energy_directional_derivative=-scale*float(np.vdot(gradient,direction).real),
        no_cached_parent_reuse_after_update=True)
    return unravel(jnp.asarray(proposed)),next_moments,diagnostics,cache


def normalize_legacy_proposal(parent: dict, proposal: tuple) -> tuple:
    """Preserve the exact fitted legacy direction, not its coordinate LR cap."""
    params,moments,info=proposal;info=dict(info)
    measured=float(info['sampled_tangent_FS'])
    if not bool(info['accepted']) or not np.isfinite(measured) or measured<=0:
        raise ArithmeticError('nonzero valid legacy physical direction required')
    factor=.015/measured
    normalized=jax.tree.map(lambda old,new:old+factor*(new-old),parent,params)
    for key in ('update_scale','raw_energy_directional_derivative','applied_energy_directional_derivative'):
        if key in info:info[key]=float(info[key])*factor
    info.update(legacy_actual_tangent_FS_before_normalization=measured,
        physical_direction_rescale=factor,sampled_tangent_FS=.015,
        coordinate_learning_rate_not_a_physical_step_cap=True)
    return normalized,moments,info


def select_geometry(phi: np.ndarray, density: np.ndarray) -> tuple[int,dict]:
    """Amplitude-only FS bracket, then held-out veto with no post-veto reselect."""
    from su2_nqs.vmc.finite_candidate_calibration import common_proposal_FS
    parent=len(GEOMETRY_FACTORS)-1
    if phi.shape!=(1024,len(GEOMETRY_FACTORS)) or density.shape!=(1024,):
        raise ValueError('full1024 geometry panel and all registered factors required')
    selection=[common_proposal_FS(phi[:512][:,[parent,i]],density[:512],groups=16) for i in range(parent)]
    eligible=[i for i,m in enumerate(selection) if 1e-12<m['FS_radians'] and m['nominal_upper']<=.02]
    chosen=min(eligible,key=lambda i:abs(selection[i]['FS_radians']-.015)) if eligible else parent
    result=dict(selection=selection,selected_index=chosen,geometry_factors=GEOMETRY_FACTORS.tolist(),
        target_FS=.015,maximum_FS=.02,guard_not_used_for_reselection=True)
    if chosen==parent:return parent,dict(result,accepted=False,reason='no_eligible_nonzero_FS')
    guard=common_proposal_FS(phi[512:][:,[parent,chosen]],density[512:],groups=16)
    ratio=guard['FS_radians']/selection[chosen]['FS_radians']
    passed=guard['nominal_upper']<=.02 and guard['FS_radians']>=.0075 and .5<=ratio<=2
    return (chosen if passed else parent),dict(result,accepted=bool(passed),guard=guard,
        guard_to_selection_ratio=ratio,minimum_guard_FS=.0075,reason='passed' if passed else 'independent_FS_veto')


def timed(plan: dict, path: Path, label: str, fn):
    """Record completed cold/warm work and real allocator high water."""
    tick = perf_counter(); value = jax.block_until_ready(fn()); elapsed = perf_counter()-tick
    atomic_json(path/(label+'.json'),dict(seconds=elapsed,observed_peak_device_bytes=check_peak(plan)))
    return value,elapsed


def measured_tangent_FS(rt: dict, params: dict, candidate: dict, rows, psi, density) -> float:
    """Measure the implemented delta, not an LR label, with a direct full-A1 JVP."""
    theta,_,batch=batch_function(rt,params)
    delta,_=ravel_pytree(jax.tree.map(lambda n,o:n-o,candidate,params))
    derivative=jax.jit(lambda t,d:jax.jvp(lambda x:batch(x,jnp.asarray(rows)),(t,),(d,))[1])(theta,delta)
    value=np.asarray(jax.block_until_ready(derivative));psi=np.asarray(psi);q=np.asarray(density)
    scale=1/np.sqrt(q*np.sum(psi**2/q));u=psi*scale;z=value*scale;z-=u*np.dot(u,z)
    actual=float(np.linalg.norm(z))
    if not np.isfinite(actual) or abs(actual-.015)>.015*1e-5:
        raise ArithmeticError('implemented physical tangent differs from registered.015')
    return actual


def finite_geometry(plan: dict, rt: dict, parent: tuple, proposal: tuple, path: Path,
                    arm: str, *, replay: bool=False) -> tuple[tuple, dict]:
    """Preregistered amplitude bracket with independent512 selection/512 veto."""
    path.mkdir(exist_ok=True)
    delta = jax.tree.map(lambda n,o:n-o,proposal[0],parent[0])
    candidates = [jax.tree.map(lambda p,d:p+f*d,parent[0],delta) for f in GEOMETRY_FACTORS]
    banks = [active.canonical_active_sampling_bank(p,rt['cfg'],rt['profile'])[0] for p in candidates]
    if replay:
        saved = np.load(path/'fixture.npz'); rows = saved['rows']
    else:
        selection,sd = active.exact_draw(banks,rt,512,plan['seeds'][arm+'_selection'],bridge=False)
        guard,gd = active.exact_draw(banks,rt,512,plan['seeds'][arm+'_guard'],bridge=False)
        rows = np.concatenate((selection,guard)); save_checkpoint(path/'draws.pkl',(sd,gd),metadata=dict(role='independent_FS_selection_and_veto'))
    raw = active.make_feature_lut_amplitude(rt['cfg'],active_top_widths=rt['profile'])
    groups = jnp.asarray(rt['images'].reshape(-1,16,144))
    def one(bank,row):
        def group(ids):
            v = jax.vmap(raw,in_axes=(None,0))(bank,row[ids])
            return jnp.stack((v.sum(),jnp.sum(v*v)))
        return jnp.sum(jax.lax.map(group,groups),axis=0)/1728
    query = jax.jit(lambda b,x:jax.lax.map(lambda row:one(b,row),x,batch_size=4))
    columns = []
    for bank in banks:
        active.stage_budget(rt,plan['restore_reserve_seconds']+120,'full fixed FS column')
        columns.append(np.asarray(jax.block_until_ready(query(rt['prepare'](bank),rows)))); check_peak(plan)
    values = np.stack(columns,axis=1); phi = values[:,:,0]; q = values[:,:,1].mean(1)
    selected,info = select_geometry(phi,q)
    if replay:
        replay_error(phi,saved['phi']); replay_error(q,saved['density'])
        if selected != int(saved['selected']): raise ArithmeticError('independent geometry decision changed')
    else:
        np.savez_compressed(path/'fixture.npz',rows=rows,phi=phi,density=q,selected=selected)
        atomic_json(path/'result.json',clean(info))
    candidate = (candidates[selected],proposal[1],deepcopy(parent[2])) if selected!=len(GEOMETRY_FACTORS)-1 else parent
    return candidate,dict(info,selected_index=int(selected),records=1024,selection_records=512,guard_records=512)


def comparison(plan: dict, states: dict, profiles: dict, out: Path, remaining) -> dict:
    """Exactly1024 common physical records, original score versus wide exact cache."""
    free=shutil.disk_usage(out).free
    if free<220*1024**3:raise MemoryError('one exact gate cache needs at least220GiB external free disk; never use/tmp')
    rt = {a:make_runtime(plan,a,s['params'],profiles[a]) for a,s in states.items()}
    for runtime in rt.values(): runtime['remaining']=remaining
    original = states['original']; parent = original['params']
    normalized,note = active.canonical_active_sampling_bank(parent,rt['original']['cfg'],profiles['original'])
    rows,draw = active.exact_draw([normalized],rt['original'],1024,plan['seeds']['fit'],bridge=True)
    save_checkpoint(out/'fit_draw.pkl',draw,metadata=dict(role='shared_actual1024_comparison_fit_not_audit'))
    products,warm_H = active.products(rt['original'],[normalized],rows,out,'common_fit',8)
    if not np.isfinite(products).all() or np.max(abs(products.imag))>2e-11*max(np.max(abs(products)),1e-300):
        raise ArithmeticError('common full-H panel is not finite real CT data')
    density = np.asarray(full_support_sz0_density(products[:,0,2].real,products[:,0,3].real,n_sites=144,epsilon=.125))
    native_scale = np.sqrt(note['original_raw_norm']); psi = np.asarray(products[:,0,0].real*native_scale)
    hpsi = np.asarray(products[:,0,1].real*native_scale)
    np.savez_compressed(out/'fit_panel.npz',rows=rows,psi=psi,Hpsi=hpsi,density=density)
    fit_seal=file_sha(out/'fit_panel.npz')
    with np.load(ROOT/plan['inputs']['diagnostic_rows']['path']) as f: diagnostic=f['rows'][:1]
    timings = {}; replay = {}; fitted = {}; readiness={a:'pending' for a in ARMS}
    def persist_ready():
        atomic_json(out/'compare_result.json',clean(dict(status='partial_actual1024_arm_readiness',
            fit_records=1024,fit_panel_sha256=fit_seal,timings=timings,replay=replay,fitted=fitted,
            arm_readiness=readiness,shared_full_H_science_centers=1024,
            original_may_gate_if_wide_incomplete=True,native_RNG_advanced=False)))
    persist_ready()
    for arm in ARMS:
        p = states[arm]['params']; runtime = rt[arm]
        with np.load(ROOT/plan['inputs']['saved_H_products']['path']) as frozen:
            saved_rows=frozen['rows'];saved_products=frozen['parent' if arm=='original' else 'wide']
        actual,_=timed(plan,out,arm+'_saved_native_H_replay',lambda:runtime['H'](p,jnp.asarray(saved_rows)))
        replay[arm+'_saved_H_components']=[replay_error(np.asarray(a),b) for a,b in zip(actual,saved_products,strict=True)]
        native = lambda:runtime['amplitude'](runtime['prepare'](p),jnp.asarray(rows))
        value,cold = timed(plan,out,arm+'_actual1024_amp_cold',native)
        _,warm = timed(plan,out,arm+'_actual1024_amp_warm',native)
        replay[arm] = replay_error(value,psi)
        Hfn = lambda:runtime['H'](p,jnp.asarray(rows[:8]))
        H,coldH = timed(plan,out,arm+'_H8_cold',Hfn)
        _,warmH = timed(plan,out,arm+'_H8_warm',Hfn)
        expected = products[:8,0]*np.asarray([native_scale,native_scale,native_scale**2,native_scale**2])
        replay[arm+'_H_components'] = [replay_error(np.asarray(v),expected[:,i]) for i,v in enumerate(H)]
        from projects.clebsch_tree_nqs.explorations.ct12_w28_gpu_preflight_20261030 import root_direction
        d = root_direction(p,runtime['cfg'],new_copy=None,profile=profiles[arm])
        hjvp = jax.jit(lambda p,d,x:legacy.native_h_jvp(runtime,p,d,x))
        _,coldJ = timed(plan,out,arm+'_H_JVP_cold',lambda:hjvp(p,d,jnp.asarray(diagnostic)))
        _,warmJ = timed(plan,out,arm+'_H_JVP_warm',lambda:hjvp(p,d,jnp.asarray(diagnostic)))
        timings[arm] = dict(actual1024_amplitude_cold_seconds=cold,actual1024_amplitude_warm_seconds=warm,
            H8_cold_seconds=coldH,H_seconds_per_center=warmH/8,H_JVP_cold_seconds=coldJ,H_JVP_warm_seconds_per_center=warmJ)
        active.stage_budget(runtime,plan['restore_reserve_seconds']+1800,'complete1024 full-native derivative and exact solve')
        if arm=='original':
            fn=lambda:runtime['update'](p,states[arm]['optimizer_state'],jnp.asarray(rows),jnp.asarray(hpsi),jnp.asarray(density),.03)
            proposal,coldSR = timed(plan,out,arm+'_actual1024_score_SR_cold',fn)
            repeated,warmSR = timed(plan,out,arm+'_actual1024_score_SR_warm',fn)
            for a,b in zip(jax.tree.leaves(proposal[:2]),jax.tree.leaves(repeated[:2]),strict=True): replay_error(a,b)
            params,moments,sr = normalize_legacy_proposal(p,proposal)
            if not bool(sr['accepted']): raise ArithmeticError('original full score/SR proposal invalid')
        else:
            def progress(message):
                active.stage_budget(runtime,plan['restore_reserve_seconds']+300,'next registered exact derivative/algebra block')
                check_peak(plan); atomic_json(out/'wide_cache_progress.json',clean(message))
            tick=perf_counter()
            params,moments,sr,cache=cached_proposal(plan,runtime,p,states[arm]['optimizer_state'],rows,psi,hpsi,density,
                storage=out/'wide_actual1024_J.npy',progress=progress)
            coldSR=perf_counter()-tick; tick=perf_counter()
            rp,rm,ri,_=cached_proposal(plan,runtime,p,states[arm]['optimizer_state'],rows,psi,hpsi,density,cache=cache,progress=progress)
            warmSR=perf_counter()-tick
            for a,b in zip(jax.tree.leaves((params,moments)),jax.tree.leaves((rp,rm)),strict=True): replay_error(a,b)
            atomic_json(out/'wide_cache_metadata.json',clean(cache.metadata)); del cache; gc.collect()
        timings[arm].update(SR_cold_seconds=coldSR,SR_warm_seconds=warmSR,
            warm_is_solve_only_no_AD=(arm=='wide28'),sampled_physical_tangent_FS=float(sr['sampled_tangent_FS']))
        actual_FS,fs_seconds=timed(plan,out,arm+'_actual1024_implemented_tangent_JVP',
            lambda:measured_tangent_FS(runtime,p,params,rows,psi,density))
        timings[arm].update(implemented_tangent_FS=actual_FS,implemented_tangent_JVP_seconds=fs_seconds)
        legacy.restart.save_state(out/(arm+'_full_proposal.pkl'),(params,moments,deepcopy(states[arm]['metadata']['rng'])),0,45)
        atomic_json(out/(arm+'_fit_SR.json'),clean(sr))
        fitted[arm] = dict(physical_FS=float(sr['sampled_tangent_FS']),native_parameters=plan['original_parameters' if arm=='original' else 'wide_parameters'],
            full_proposal_sha256=file_sha(out/(arm+'_full_proposal.pkl')))
        ready=dict(status='ready',arm=arm,plan_sha256=canonical_sha256(plan),source_identity=cache_source(plan),
            fit_records=1024,fit_panel_sha256=fit_seal,proposal_sha256=fitted[arm]['full_proposal_sha256'],
            SR_sha256=file_sha(out/(arm+'_fit_SR.json')),timings=timings[arm],
            timings_sha256=canonical_sha256(timings[arm]),
            native_rng_sha256=canonical_sha256(states[arm]['metadata']['rng']),native_RNG_advanced=False)
        atomic_json(out/(arm+'_ready.json'),clean(ready))
        readiness[arm]='ready';persist_ready()
    report = dict(status='complete_actual1024_comparison_no_training',fit_records=1024,
        shared_full_H_science_centers=1024,extra_H_cost_centers_per_arm=16,
        H_JVP_cost_primal_and_tangent_centers_per_arm=2,
        common_H_reuse_requires_full1024_native_amplitude_and_H8_scale_replay=True,
        common_original_H_warm_seconds_per_center=warm_H,replay=replay,timings=timings,fitted=fitted,
        arm_readiness=readiness,fit_panel_sha256=fit_seal,
        comparison_is_execution_and_capacity_not_architecture_win=True,gate_candidate_committed=False,
        physical_FS_matched_not_native_coefficient_directions=True,native_RNG_advanced=False,
        no_training_speedup_claim_from_wide_solve_only_vs_original_AD_warm=True,
        saved_H_replay_centers_per_arm=len(saved_rows),
        all_fit_and_diagnostic_queries_charged_not_new_quality_precision=True)
    report['external_disk_free_bytes_before_draw']=free
    atomic_json(out/'compare_result.json',clean(report)); return report


def gate_forecast(plan: dict, timing: dict, *, replay: bool=False) -> dict:
    """Charge warmed real kernels, compilation once, full immutable record counts."""
    h=float(timing['H_seconds_per_center']); j=float(timing['H_JVP_warm_seconds_per_center'])
    if not np.isfinite([h,j]).all() or min(h,j)<=0: raise ValueError('actual warm full kernels required')
    components=dict(bilateral_512=512*j,paired_confirmation_2048=2048*h,
        full_FS_8x1024=8*timing['actual1024_amplitude_warm_seconds'],
        freshprocess_restore_and_cache_replay=plan['restore_reserve_seconds'],
        new_shape_and_candidate_compile=1800,save=plan['save_reserve_seconds'])
    return dict(components_seconds=components,forecast_seconds=1.5*sum(v for k,v in components.items()
        if k in ('bilateral_512','paired_confirmation_2048','full_FS_8x1024'))+
        sum(v for k,v in components.items() if k not in ('bilateral_512','paired_confirmation_2048','full_FS_8x1024')),
        no_scientific_records_reduced=True,cold_calls_not_extrapolated_as_throughput=True)


def require_arm_ready(plan: dict, out: Path, arm: str) -> dict:
    """A partial other arm never blocks an independently sealed complete target arm."""
    ready=json.loads((out/(arm+'_ready.json')).read_text())
    if (ready.get('status')!='ready' or ready.get('arm')!=arm or ready.get('fit_records')!=1024
            or ready.get('plan_sha256')!=canonical_sha256(plan) or ready.get('source_identity')!=cache_source(plan)
            or ready.get('timings_sha256')!=canonical_sha256(ready['timings'])
            or ready.get('native_RNG_advanced') is not False):
        raise PermissionError('target arm readiness source/plan/record/timing seal invalid')
    for field,path in [('fit_panel_sha256',out/'fit_panel.npz'),
                       ('proposal_sha256',out/(arm+'_full_proposal.pkl')),
                       ('SR_sha256',out/(arm+'_fit_SR.json'))]:
        if ready[field]!=file_sha(path):raise ValueError('target arm readiness file changed: '+field)
    return ready


def arm_gate(plan: dict, states: dict, profiles: dict, out: Path, arm: str, remaining) -> dict:
    """Each arm gets its own candidate, geometry, bilateral and finite-energy gate."""
    ready=require_arm_ready(plan,out,arm);timing=ready['timings']
    if ready['native_rng_sha256']!=canonical_sha256(states[arm]['metadata']['rng']):raise ValueError('original next RNG readiness seal changed')
    parent=(states[arm]['params'],states[arm]['optimizer_state'],deepcopy(states[arm]['metadata']['rng']))
    rt=make_runtime(plan,arm,parent[0],profiles[arm]);rt['remaining']=remaining
    target=out/arm;target.mkdir();forecast=gate_forecast(plan,timing)
    atomic_json(target/'cost_forecast.json',forecast)
    active.stage_budget(rt,forecast['forecast_seconds'],'entire unchanged arm direction gate plus restore')
    proposal,_=legacy.restart.load_state(out/(arm+'_full_proposal.pkl'))
    candidate,geometry=finite_geometry(plan,rt,parent,proposal,target/'geometry',arm)
    if not geometry['accepted']:
        result=dict(status='finite_geometry_veto_no_training',arm=arm,passed=False,geometry=geometry)
    else:
        legacy.restart.save_state(target/'candidate.pkl',candidate,0,45)
        armplan=dict(plan,seeds=dict(plan['seeds'],bilateral=plan['seeds'][arm+'_bilateral']))
        bilateral=legacy.bilateral_panel(armplan,rt,parent,candidate,target/'bilateral');check_peak(plan)
        confirmation=legacy.paired_panel(armplan,rt,parent,candidate,target/'confirmation',count=1024,groups=32,
            seed=plan['seeds'][arm+'_confirmation'],statistics=plan['seeds'][arm+'_confirmation_statistics']);check_peak(plan)
        decision=legacy.gate_decision(bilateral,confirmation,geometry_passed=True,replay_passed=False)
        result=dict(status='frozen_independent_direction_requires_process_replay',arm=arm,geometry=geometry,
            bilateral=bilateral,confirmation=confirmation,decision=decision,
            preliminary_passed=all(v for k,v in decision['checks'].items() if k!='cross_process_replay'),
            candidate_sha256=file_sha(target/'candidate.pkl'),gate_candidate_is_production_step=False)
    atomic_json(target/'gate_result.json',clean(result)); return result


def joint_guard_report(out: Path) -> dict:
    """Independent valid arms are not automatically matched-radius energy comparisons."""
    paths=[out/a/'gate_result.json' for a in ARMS]
    if not all(p.exists() for p in paths):return dict(status='awaiting_second_arm')
    results=[json.loads(p.read_text()) for p in paths]
    guards=[r.get('geometry',{}).get('guard') for r in results]
    if any(g is None for g in guards):report=dict(status='one_or_both_geometry_veto',comparative_energy_claim_allowed=False)
    else:
        radius=np.asarray([g['FS_radians'] for g in guards],float)
        errors=np.asarray([g['group_jackknife_SE'] for g in guards],float)
        allowance=.2*float(np.max(radius));difference=float(abs(radius[0]-radius[1]))
        upper=difference+2*float(np.linalg.norm(errors))
        matched=bool(np.isfinite(upper) and upper<=allowance)
        report=dict(status='matched_guard_radii' if matched else 'guard_radii_not_matched',
            actual_guard_FS=dict(zip(ARMS,radius.tolist())),guard_group_jackknife_SE=dict(zip(ARMS,errors.tolist())),
            absolute_difference=difference,difference_plus2SE=upper,registered_relative_tolerance=.2,
            matched_with_uncertainty=matched,comparative_energy_claim_allowed=False,
            no_architecture_or_energy_win_claim_from_this_two_arm_gate=True,
            each_arm_direction_certificate_remains_separate=True)
    atomic_json(out/'joint_guard_FS_comparison.json',report);return report


def arm_replay(plan: dict, states: dict, profiles: dict, out: Path, arm: str, remaining) -> dict:
    """Fresh process, same frozen fit cache/data and geometry rows; no science redraw."""
    target=out/arm;result=json.loads((target/'gate_result.json').read_text())
    require_arm_ready(plan,out,arm)
    certificate=dict(passed=False,arm=arm,plan_sha256=canonical_sha256(plan),
        parent_checkpoint_sha256=plan['inputs']['parent' if arm=='original' else 'expanded']['sha256'],
        parent_rng_sha256=canonical_sha256(states[arm]['metadata']['rng']),gate_candidate_is_production_step=False,
        gate_result_sha256=file_sha(target/'gate_result.json'))
    if result.get('preliminary_passed'):
        p=states[arm]['params'];moments=states[arm]['optimizer_state'];rt=make_runtime(plan,arm,p,profiles[arm]);rt['remaining']=remaining
        with np.load(out/'fit_panel.npz') as f: rows,psi,hpsi,q=(f[k] for k in ('rows','psi','Hpsi','density'))
        if arm=='original':
            params,m,sr=jax.block_until_ready(rt['update'](p,moments,jnp.asarray(rows),jnp.asarray(hpsi),jnp.asarray(q),.03))
            params,m,sr=normalize_legacy_proposal(p,(params,m,sr))
        else:
            from su2_nqs.vmc.cached_amplitude_operators import load_cached_amplitude_jacobian
            cache=load_cached_amplitude_jacobian(out/'wide_actual1024_J.npy',maximum_host_bytes=plan['maximum_host_array_bytes'],verify_content=True)
            params,m,sr,_=cached_proposal(plan,rt,p,moments,rows,psi,hpsi,q,cache=cache)
        expected,_=legacy.restart.load_state(out/(arm+'_full_proposal.pkl'))
        errors=[replay_error(a,b) for a,b in zip(jax.tree.leaves((params,m)),jax.tree.leaves(expected[:2]),strict=True)]
        parent=(p,moments,deepcopy(states[arm]['metadata']['rng']))
        candidate,geometry=finite_geometry(plan,rt,parent,(params,m,parent[2]),target/'geometry',arm,replay=True)
        frozen,_=legacy.restart.load_state(target/'candidate.pkl')
        if file_sha(target/'candidate.pkl')!=result['candidate_sha256']:raise ValueError('frozen candidate changed')
        errors += [replay_error(a,b) for a,b in zip(jax.tree.leaves(candidate[:2]),jax.tree.leaves(frozen[:2]),strict=True)]
        decision=legacy.gate_decision(result['bilateral'],result['confirmation'],geometry_passed=geometry['accepted'],replay_passed=True)
        certificate.update(decision,candidate_sha256=result['candidate_sha256'],maximum_numerical_replay_error=max(errors,default=0.),
            controls=dict(damping=.5,target_FS=.015,maximum_FS=.02),scientific_confirmation_not_reused_as_terminal_audit=True)
    atomic_json(target/'gate_certificate.json',clean(certificate)); return dict(status='gate_passed' if certificate['passed'] else 'gate_failed_no_training',certificate=certificate)


def validate_pilot_release(release: dict, plan: dict, out: Path, arm: str) -> dict:
    """Root chooses at most one arm after actual gates/cost, not by training means."""
    target=out/arm;cert=json.loads((target/'gate_certificate.json').read_text())
    cap=plan['original_pilot_wall_cap_seconds' if arm=='original' else 'wide_pilot_wall_cap_seconds']
    if (not cert.get('passed') or cert.get('plan_sha256')!=canonical_sha256(plan)
            or len(cert.get('checks',{}))!=6 or not all(cert.get('checks',{}).values())
            or cert.get('gate_result_sha256')!=file_sha(target/'gate_result.json')
            or cert.get('candidate_sha256')!=file_sha(target/'candidate.pkl')
            or cert.get('parent_checkpoint_sha256')!=plan['inputs']['parent' if arm=='original' else 'expanded']['sha256']
            or cert.get('controls')!=dict(damping=.5,target_FS=.015,maximum_FS=.02)
            or release.get('selected_arm')!=arm or release.get('maximum_steps')!=8
            or release.get('gate_certificate_sha256')!=file_sha(target/'gate_certificate.json')
            or release.get('wall_cap_seconds')!=cap or release.get('root_authorized') is not True
            or release.get('restore_original_global45') is not True):
        raise PermissionError('separate same-plan passed-certificate root pilot release required')
    return cert


def physical_transition(plan: dict, rt: dict, state: tuple, path: Path, *, replay: bool=False):
    """Reuse guarded consumer but forbid silent fallback to microscopic steps."""
    candidate,info=legacy.transition_once(plan,rt,state,path,replay=replay)
    if info['accepted']:
        radius=float(info['calibration']['guard']['FS_radians'])
        if not np.isfinite(radius) or radius<plan['minimum_guard_FS']:
            _,following=active._seeds(state[2]);candidate=(state[0],state[1],following)
            info=dict(info,accepted=False,new_trust_profile_floor_veto=True,
                actual_guard_FS=radius,minimum_guard_FS=plan['minimum_guard_FS'],
                rejected_update_roles_consumed=True,no_post_veto_reselection=True)
            if not replay:
                legacy.restart.save_state(path/'new_trust_floor_rejected.pkl',candidate,0,int(state[1][0]))
                atomic_json(path/'new_trust_floor_veto.json',clean(info))
    return candidate,info


def pilot_phase(plan: dict, states: dict, profiles: dict, out: Path, arm: str, remaining,
                release: dict, phase: str) -> dict:
    """Original guarded consumer, a fresh-process first replay, and fresh4096 audit.

    Exact caches are volatile host arrays for one CURRENT parent, not eight
    files. The only retained disk J is the independent comparison/gate cache.
    Native role RNG advances even on rejection/interruption; no replay can
    turn an already consumed role into another scientific attempt.
    """
    validate_pilot_release(release,plan,out,arm)
    path=out/'pilot'
    if phase=='pilot_prepare':
        path.mkdir()  # One common path forbids a second arm in this campaign.
        atomic_json(path/'selection.json',dict(arm=arm,release=release,original_global_step=45,started_epoch=time()))
    selection=json.loads((path/'selection.json').read_text())
    if selection['arm']!=arm or selection['release']!=release:raise ValueError('pilot arm/release changed')
    parent=(states[arm]['params'],states[arm]['optimizer_state'],deepcopy(states[arm]['metadata']['rng']))
    state=parent
    rt=make_runtime(plan,arm,state[0],profiles[arm]);rt['remaining']=remaining
    timing=json.loads((out/'compare_result.json').read_text())['timings'][arm]
    rt['warm_H_seconds']=timing['H_seconds_per_center']
    rt['audit_reserve_seconds']=1.5*8192*rt['warm_H_seconds']+1800
    if arm=='wide28':
        def update(p,m,rows,hpsi,q,lr):
            if float(lr)!=.03:raise ValueError('registered learning rate changed')
            _,_,batch=batch_function(rt,p);theta,_=ravel_pytree(p)
            psi=np.asarray(jax.block_until_ready(batch(theta,rows)))
            proposal,moved,info,cache=cached_proposal(plan,rt,p,m,rows,psi,np.asarray(hpsi),np.asarray(q),
                progress=lambda _:active.stage_budget(rt,rt['audit_reserve_seconds']+300,'exact current-parent derivative block'))
            del cache;gc.collect()
            scalar_info={k:v for k,v in info.items() if np.asarray(v).ndim==0}
            return proposal,moved,scalar_info
        rt['update']=update
    else:
        original_update=rt['update']
        rt['update']=lambda p,m,x,h,q,lr:normalize_legacy_proposal(p,jax.block_until_ready(original_update(p,m,x,h,q,lr)))
    def freeze(current: tuple, done: int, reason: str) -> dict:
        legacy.restart.save_state(path/'checkpoint.pkl',current,done,45+done)
        frozen=dict(completed_updates=done,absolute_step=45+done,parameter_sha256=pytree_sha256(current[0]),
            checkpoint_sha256=file_sha(path/'checkpoint.pkl'),target_eight_reached=done==8,stop_reason=reason)
        atomic_json(path/'endpoint_frozen.json',frozen);return frozen
    if phase=='pilot_prepare':
        active.stage_budget(rt,rt['audit_reserve_seconds']+2*max(1800,timing['SR_cold_seconds']),
            'full first update, fresh-process restore, and untouched terminal audit')
        try:
            candidate,info=physical_transition(plan,rt,parent,path/'step1')
        except (TimeoutError,MemoryError) as error:
            used,following=active._seeds(parent[2]);consumed=(parent[0],parent[1],following)
            frozen=freeze(consumed,0,'first_partial_attempt_roles_reserved_consumed')
            atomic_json(path/'first_interrupted_roles.json',dict(reserved_consumed_roles=used,
                no_parameter_or_optimizer_commit=True,following_rng_sha256=canonical_sha256(following),error=str(error)))
            atomic_json(path/'first_prepare.json',dict(accepted=False,partial_attempt=True))
            return dict(status='first_partial_no_commit_roles_consumed',**frozen)
        legacy.restart.save_state(path/'first_provisional.pkl',candidate,0,45)
        atomic_json(path/'first_prepare.json',clean(info))
        if not info['accepted']:freeze(candidate,0,'first_geometry_veto_roles_consumed')
        return dict(status='first_update_provisional_requires_fresh_process_replay' if info['accepted'] else 'first_veto_no_training',transition=info)
    if phase=='pilot_replay':
        info=json.loads((path/'first_prepare.json').read_text())
        if not info['accepted']:return dict(status='no_first_proposal_no_commit')
        if (path/'checkpoint.pkl').exists():raise FileExistsError('first update already committed')
        replayed,replay=physical_transition(plan,rt,parent,path/'step1',replay=True)
        expected,_=legacy.restart.load_state(path/'first_provisional.pkl')
        for a,b in zip(jax.tree.leaves(replayed[:2]),jax.tree.leaves(expected[:2]),strict=True):replay_error(a,b)
        if canonical_sha256(replayed[2])!=canonical_sha256(expected[2]):raise ArithmeticError('first native next RNG replay failed')
        legacy.restart.save_state(path/'checkpoint.pkl',expected,1,46)
        atomic_json(path/'first_restore_replay.json',clean(replay))
        return dict(status='fresh_process_replay_first_committed_once',committed_updates=1)
    if not (path/'checkpoint.pkl').exists():return dict(status='no_committed_endpoint_no_quality_claim')
    state,metadata=legacy.restart.load_state(path/'checkpoint.pkl');committed=int(metadata['completed_updates'])
    if phase=='pilot_train':
        if committed<1:return dict(status='zero_updates_no_training')
        if not (path/'first_restore_replay.json').exists():raise PermissionError('fresh-process first replay required')
        last=float(json.loads((path/'first_prepare.json').read_text())['wall_seconds']);reason='eight_requested'
        for i in range(committed+1,9):
            if remaining()<rt['audit_reserve_seconds']+max(1800,1.5*last):reason='terminal_audit_reserve';break
            try:
                candidate,info=physical_transition(plan,rt,state,path/f'step{i}')
            except (TimeoutError,MemoryError):
                # All three roles are conservatively reserved once this attempt
                # begins; a partially consumed attempt cannot be silently retried.
                used,following=active._seeds(state[2]);state=(state[0],state[1],following)
                atomic_json(path/f'step{i}_interrupted_roles.json',dict(reserved_consumed_roles=used,
                    no_parameter_or_optimizer_commit=True,following_rng_sha256=canonical_sha256(following)))
                reason='bounded_attempt_parent_and_consumed_RNG_preserved';break
            last=info['wall_seconds'];check_peak(plan);state=candidate
            if info['accepted']:committed=i
            legacy.restart.save_state(path/'checkpoint.pkl',state,committed,45+committed)
            if not info['accepted']:reason='independent_finite_geometry_veto';break
        return dict(status='fixed_prefix_frozen_for_untouched_audit',**freeze(state,committed,reason))
    if phase!='audit':raise ValueError('unknown pilot phase')
    if committed<1:return dict(status='zero_updates_no_terminal_precision')
    frozen=json.loads((path/'endpoint_frozen.json').read_text())
    if frozen['checkpoint_sha256']!=file_sha(path/'checkpoint.pkl') or frozen['parameter_sha256']!=pytree_sha256(state[0]):
        raise ValueError('frozen endpoint changed before untouched audit')
    audit=legacy.paired_panel(plan,rt,parent,state,path/'terminal_audit',count=4096,groups=64,
        seed=plan['seeds']['audit'],statistics=plan['seeds']['audit_statistics'])
    return dict(status='complete_fixed_prefix_independent_audit',**frozen,terminal_audit=audit,
        gate_fit_geometry_terminal_roles_separate=True,terminal_audit_not_used_for_endpoint_selection=True)


def fresh_phase_path(out: Path, phase: str, arm: str|None) -> Path:
    """Reject duplicate/partial attempts before altering any budget anchor."""
    path=out/(phase+('_'+arm if arm else '')+'_result.json')
    if path.exists():raise FileExistsError('never rerun already consumed scientific phase')
    if phase=='gate' and ((out/(arm+'_gate_budget.json')).exists() or (out/arm).exists()):
        raise FileExistsError('existing gate budget/attempt is immutable, including partial work')
    if phase=='pilot_prepare' and (out/'pilot').exists():raise FileExistsError('pilot attempt already reserved')
    return path


def run(plan_path: Path, out: Path, phase: str, *, arm: str|None=None, release_path: Path|None=None) -> dict:
    """Run one bounded A100 phase; failed work never becomes a training endpoint."""
    plan=json.loads(plan_path.read_text());validate_plan(plan,executing=True)
    if phase not in PHASES or (phase!='compare' and arm not in ARMS):raise ValueError('explicit registered phase/arm required')
    if not out.is_absolute() or out.is_relative_to(ROOT):raise ValueError('all output/cache data must be outside code-only release')
    if jax.default_backend()!='gpu' or not all('A100' in d.device_kind for d in jax.devices()):raise RuntimeError('real allocated A100 required')
    if not re.fullmatch(r'[0-9a-f]{40}',os.environ.get('NQS_CODE_COMMIT','')):
        raise PermissionError('launcher must bind a complete immutable Git commit SHA')
    allocated=int(os.environ.get('SLURM_MEM_PER_NODE','0'))
    minimum=320 if phase=='compare' or arm=='wide28' else 128
    if allocated<minimum*1024:raise MemoryError('explicit registered Slurm host memory allocation is required')
    if phase in ('gate','replay'):
        try:require_arm_ready(plan,out,arm)
        except (FileNotFoundError,PermissionError,ValueError) as error:
            out.mkdir(parents=True,exist_ok=True)
            result=dict(status='arm_not_ready_no_scientific_queries',phase=phase,arm=arm,
                reason=str(error),new_draws=0,H_queries=0,comparison_not_fabricated=True)
            atomic_json(out/(phase+'_'+arm+'_refusal.json'),result);return result
    states,profiles=load_states(plan)
    versions=verify_runtime_versions()
    if phase=='compare':
        if out.exists():raise FileExistsError('fresh immutable campaign output required')
        out.mkdir(parents=True)
        atomic_json(out/'registration.json',dict(plan=plan,plan_sha256=canonical_sha256(plan),
            source_identity=cache_source(plan),native_rng_sha256=canonical_sha256(states['original']['metadata']['rng']),
            runtime_versions=versions,code_commit=os.environ['NQS_CODE_COMMIT'],baseline_not_production_release=True))
    else:
        reg=json.loads((out/'registration.json').read_text())
        if reg['plan']!=plan or reg['source_identity']!=cache_source(plan) or reg['runtime_versions']!=versions:
            raise ValueError('registered source/plan/runtime changed')
    is_pilot=phase.startswith('pilot_') or phase=='audit'
    cap=plan['compare_wall_cap_seconds'] if phase=='compare' else (
        plan['original_pilot_wall_cap_seconds' if arm=='original' else 'wide_pilot_wall_cap_seconds'] if is_pilot
        else plan['original_gate_wall_cap_seconds' if arm=='original' else 'wide_gate_wall_cap_seconds'])
    result_path=fresh_phase_path(out,phase,arm)
    start=time();stop=[False]
    epoch=start
    if phase=='gate':atomic_json(out/(arm+'_gate_budget.json'),dict(started_epoch=start,wall_cap_seconds=cap))
    elif phase=='replay':epoch=json.loads((out/(arm+'_gate_budget.json')).read_text())['started_epoch']
    elif is_pilot and phase!='pilot_prepare':epoch=json.loads((out/'pilot/selection.json').read_text())['started_epoch']
    def stopping(*_):stop[0]=True
    for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,stopping)
    def remaining():return 0. if stop[0] else min(epoch+cap,float(os.environ.get('SLURM_JOB_END_TIME',epoch+cap)))-time()-plan['save_reserve_seconds']
    try:
        if phase=='compare':result=comparison(plan,states,profiles,out,remaining)
        elif phase=='gate':result=arm_gate(plan,states,profiles,out,arm,remaining)
        elif phase=='replay':result=arm_replay(plan,states,profiles,out,arm,remaining)
        else:
            if release_path is None:raise PermissionError('explicit root pilot release file required')
            result=pilot_phase(plan,states,profiles,out,arm,remaining,json.loads(release_path.read_text()),phase)
    except (TimeoutError,MemoryError) as error:
        result=json.loads((out/'compare_result.json').read_text()) if phase=='compare' and (out/'compare_result.json').exists() else {}
        result.update(status='bounded_partial_no_statistics_or_width_fallback',phase=phase,arm=arm,error=str(error),
            original_parent_unchanged=True,partial_not_a_direction_certificate=True)
    result.update(wall_seconds=time()-start,phase=phase,arm=arm)
    if phase in ('gate','replay'):joint_guard_report(out)
    atomic_json(result_path,clean(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=PHASES,required=True);p.add_argument('--arm',choices=ARMS)
    p.add_argument('--pilot-release',type=Path)
    a=p.parse_args();r=run(a.plan,a.output,a.phase,arm=a.arm,release_path=a.pilot_release)
    print(json.dumps({k:r[k] for k in ('status','phase','arm','wall_seconds') if k in r}),flush=True)
    if r['status'].startswith('bounded_partial'):raise SystemExit(2)
