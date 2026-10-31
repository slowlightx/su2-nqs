"""Isolated two-step active CT12 pilot; original CT6 workflows are untouched.

Three process phases: prepare step1 provisionally, replay its saved full-H
fixture and commit once before step2, then audit the frozen final prefix.
All physics kernels, amplitude SR, exact physical Born sampling and paired
statistics are shared implementations, not surrogate H or log-score updates.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
from time import perf_counter, time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_active_train import canonical_active_sampling_bank
from su2_nqs.models.triangular_clebsch_active_layout import repack_active_clebsch_bank, mixed_active_layout
from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
from su2_nqs.models.triangular_clebsch_feature_lut import (
    prepare_feature_lut_bank, make_feature_lut_amplitude, make_feature_lut_projected_amplitude,
)
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.exchange_bridge_importance import make_orbit_exchange_bridge_products
from su2_nqs.vmc.finite_candidate_calibration import common_proposal_FS
from su2_nqs.vmc.full_support_proposal import draw_full_support_sz0, full_support_sz0_density
from su2_nqs.vmc.importance_amplitude_sr import make_importance_amplitude_sr
from su2_nqs.vmc.triangular_raw_born import make_triangular_raw_born_sampler
from .triangle_iid_orbit_audit import paired_iid_moments

FACTORS = np.asarray([1., .5, .25, .125, 0.])
ROOT=Path(__file__).resolve().parents[3]


def checked_plan(plan: dict) -> tuple[MixedClebschConfig, tuple[dict, ...]]:
    """Check a sealed two-step CT12 protocol at registered J2=0 or 1/8.

    Record counts are explicit protocol inputs, never silently reduced by a
    cost gate. The original J2=1/8 plan retains its512/512/1024 panels.
    """
    required = dict(task='CT12_fixed_active_full_A1_IID_SR_two_step_v1', steps=2,
        damping=.5, learning_rate=.03,
        target_FS=.003, maximum_FS=.003, epsilon=.125, wall_cap_seconds=14400,
        maximum_peak_device_bytes=70*1024**3, automatic_production=False,
        model=dict(length=12, multiplicity_cap=20, maximum_q=10))
    if any(plan.get(k) != v for k,v in required.items()):
        raise ValueError('registered two-step active CT12 protocol required')
    registered_bonds(plan.get('j2'))
    for key in ('fit_records','calibration_records','calibration_split','audit_records','audit_groups'):
        value=plan.get(key)
        if isinstance(value,bool) or not isinstance(value,int) or value<2:
            raise ValueError('explicit positive integer scientific record counts required')
    if (plan['fit_records']<32 or plan['calibration_split']<32 or plan['audit_groups']<4
            or plan['fit_records']%16 or plan['calibration_split']%16
            or plan['calibration_records']!=2*plan['calibration_split']
            or plan['audit_records']%plan['audit_groups']):
        raise ValueError('complete panels, two equal geometry halves and divisible audit groups required')
    if not isinstance(plan.get('initial_endpoint'),str) or not plan['initial_endpoint']:
        raise ValueError('an explicit preselected parent-state identity is required')
    if ('active_real_parameters' in plan and (isinstance(plan['active_real_parameters'],bool)
            or not isinstance(plan['active_real_parameters'],int) or plan['active_real_parameters']<1)):
        raise ValueError('sealed active real-parameter count must be a positive integer')
    seeds = plan['seeds']
    if (set(seeds) != {'training', 'audit', 'statistics'} or len(set(seeds.values())) != 3
            or any(isinstance(s,bool) or not isinstance(s,int) or not 0<=s<2**32 for s in seeds.values())):
        raise ValueError('three distinct fresh uint32 streams required')
    for key in ('projection_chunk', 'action_chunk', 'score_chunk', 'H_chunk'):
        if isinstance(plan[key],bool) or not isinstance(plan[key],int) or plan[key]<1:
            raise ValueError('positive kernel chunk sizes required')
    cfg = MixedClebschConfig(**plan['model'])
    if any(isinstance(d,bool) or not isinstance(d,int) for row in plan['active_profile'] for d in row.values()):
        raise ValueError('actual active multiplicities must be integers, not booleans')
    profile = tuple({int(q): int(d) for q,d in row.items()} for row in plan['active_profile'])
    if len(profile)!=15 or profile[-1]!={0:1}:
        raise ValueError('complete fixed singlet upper profile required')
    mixed_active_layout(cfg,profile)
    return cfg, profile


def registered_bonds(j2: float) -> tuple[tuple[int,int,float], ...]:
    """Return only physical nonzero CT12 bonds at the registered coupling.

    J2=0 has432 nearest-neighbor bonds, never864 padded bond entries.
    This host-side check precedes construction of any JAX H/query kernel.
    """
    if (isinstance(j2,bool) or not isinstance(j2,(int,float))
            or not np.isfinite(j2) or float(j2) not in (0.,.125)):
        raise ValueError('this pilot permits only physical J2=0 or J2=0.125')
    bonds=triangular_j1j2_bonds(12,12,j2=float(j2),periodic=True)
    expected=432 if j2==0 else 864
    if len(bonds)!=expected or any(not np.isfinite(w) or w==0 for _,_,w in bonds):
        raise ArithmeticError('registered nonzero physical bond count changed')
    return bonds


def validate_parent_coupling(plan: dict, metadata: dict) -> None:
    """Reject relabeling a J2=1/8 optimized initialization as a J2=0 one."""
    value=metadata.get('j2')
    if (isinstance(value,bool) or not isinstance(value,(int,float))
            or not np.isfinite(value) or float(value)!=float(plan['j2'])):
        raise ValueError('frozen parent metadata must identify its actual optimized J2')


def validate_runtime_fixture(plan: dict, fixture: dict) -> None:
    """Bind new CPU replay products to parent SHA, J2 and nonzero bond set.

    The original sealed J2=1/8 fixture predates embedded identities. Its one
    exact SHA/parent pairing remains supported, but cannot be reused at J2=0.
    """
    legacy=('0f2709f7acf418b1cf74c72fb14a4b283da2777b49874fae6b7d6571e084ad86',
        '56cd517f5bf30805317a08994a4465a7ba6cced58a16a2b33f1fd2e2282896a3')
    if (plan['initial_replay_fixture_sha256'],plan['parent_sha256'])==legacy and plan['j2']==.125:
        pass
    else:
        required=dict(j2=float(plan['j2']),n_sites=144,physical_H_bonds=len(registered_bonds(plan['j2'])),
            complete_A1_images=1728,parent_checkpoint_sha256=plan['parent_sha256'])
        for key,expected in required.items():
            if key not in fixture or np.asarray(fixture[key]).shape!=() or np.asarray(fixture[key]).item()!=expected:
                raise ValueError('runtime fixture Hamiltonian/parent mismatch: '+key)
    rows=np.asarray(fixture['rows']); values=np.asarray(fixture['products'])
    if (rows.ndim!=2 or rows.shape[1]!=144 or len(rows)<1
            or not np.issubdtype(rows.dtype,np.integer) or not np.all((rows==0)|(rows==1))
            or not np.all(rows.sum(1)==72) or values.shape!=(len(rows),4) or not np.isfinite(values).all()):
        raise ValueError('finite complete-H replay products on physical Sz=0 rows required')


def select_step(phi: np.ndarray, q: np.ndarray, *, split: int=256,
                maximum_FS: float=.003) -> tuple[int, dict]:
    """Frozen decreasing finite grid, selection and independent veto at .003."""
    if phi.shape!=(2*split,5) or q.shape!=(2*split,):
        raise ValueError('the complete registered finite-geometry panel is required')
    fit = [common_proposal_FS(phi[:split][:,[4,i]], q[:split], groups=16) for i in range(4)]
    candidates = [i for i,m in enumerate(fit) if 1e-12 < m['FS_radians'] and m['nominal_upper'] <= maximum_FS]
    chosen = candidates[0] if candidates else 4
    report = dict(selection=fit, proposed_index=chosen, parent_index=4,
        guard_used_to_select_smaller_fraction=False, target_FS=maximum_FS, maximum_FS=maximum_FS,
        selection_uses_nominal_upper_before_guard=True)
    if chosen==4:
        return 4, dict(report, accepted=False, reason='no_nonzero_geometry_eligible_candidate')
    guard = common_proposal_FS(phi[split:][:,[4,chosen]], q[split:], groups=16)
    ratio = guard['FS_radians']/fit[chosen]['FS_radians']
    passed = guard['nominal_upper']<=maximum_FS and .5<=ratio<=2
    return (chosen if passed else 4), dict(report, guard=guard,
        guard_to_selection_ratio=ratio, accepted=bool(passed), reason='passed' if passed else 'geometry_veto')


def runtime(plan: dict, params: dict) -> dict:
    cfg, profile = checked_plan(plan)
    images = np.asarray(triangular_hnf_gamma_a1_group(((12,0),(0,12))), np.int32)
    bonds = registered_bonds(plan['j2'])
    if len(images)!=1728:
        raise ArithmeticError('complete registered lattice group or physical bond count changed')
    probability = np.abs([v for _,_,v in bonds]); probability /= probability.sum()
    edges = np.asarray([(a,b) for a,b,_ in bonds], np.int32)
    factory = lambda e: make_feature_lut_exchange_queries(cfg,e,active_top_widths=profile)
    bridge, metadata = make_orbit_exchange_bridge_products(factory,images,bonds,probability,
        n_sites=144,projection_chunk=plan['projection_chunk'])
    phi = make_feature_lut_projected_amplitude(cfg,images,chunk=plan['projection_chunk'],active_top_widths=profile)
    prepare = lambda p: prepare_feature_lut_bank(p,cfg)
    update = make_importance_amplitude_sr(lambda p,x:phi(prepare(p),x),params,damping=plan['damping'],
        score_chunk=plan['score_chunk'],parameter_chunk_size=8192,sr_tangent_fs_radius=plan['target_FS'],
        prepare=prepare,prepared_amplitude=phi)
    # Hquery is batched by the established exchange/space-image kernel.
    H = jax.jit(lambda p,x:bridge(prepare(p),x))
    amplitude = jax.jit(jax.vmap(phi,in_axes=(None,0)))
    return dict(cfg=cfg,profile=profile,images=images,inverse=np.argsort(images,axis=1),bonds=bonds,
        probability=probability,edges=edges,metadata=metadata,H=H,amplitude=amplitude,
        prepare=prepare,update=update,epsilon=plan['epsilon'],j2=float(plan['j2']))


def stage_budget(rt: dict, reserve: float, label: str) -> None:
    """A stage may stop, but may never silently shorten a frozen sample panel."""
    if 'remaining' in rt and rt['remaining']()<reserve:
        raise TimeoutError('insufficient reserved time before '+label)


def exact_draw(banks: list, rt: dict, count: int, seed: int, *, bridge: bool) -> tuple[np.ndarray, dict]:
    """Independent physical raw-orbit draws, optionally exact exchange/uniform."""
    def base(number, base_seed):
        samplers = [make_triangular_raw_born_sampler(b,rt['cfg'],active_top_widths=rt['profile']) for b in banks]
        rng=np.random.default_rng(base_seed); before=deepcopy(rng.bit_generator.state)
        rows=[]; latents=[]; errors=[]
        for _ in range(number):
            stage_budget(rt,rt.get('sampling_reserve_seconds',120.),'next exact physical IID draw')
            k=int(rng.integers(len(banks))); draw=samplers[k].draw(rng)
            image=int(rng.integers(len(rt['images']))); row=draw.bits[rt['inverse'][image]].copy()
            pushed=bool(bridge and rng.random()<.5); bond=int(rng.choice(len(rt['edges']),p=rt['probability']))
            if pushed:
                a,b=rt['edges'][bond]; row[a],row[b]=row[b],row[a]
            rows.append(row); latents.append((k,image,int(pushed),bond)); errors.append(draw.log_probability_replay_error)
        return np.asarray(rows,np.int32), dict(seed=int(base_seed),rng_before=before,
            rng_after=rng.bit_generator.state,latents=latents,maximum_Born_replay_error=max(errors,default=0.))
    if bridge:
        return draw_full_support_sz0(base,count,n_sites=144,seed=seed,epsilon=rt.get('epsilon',.125))
    return base(count,seed)


def _state_save(path: Path, state: tuple, **metadata) -> None:
    params,moments,rng=state
    save_checkpoint(path,params,optimizer_state=moments,metadata=dict(metadata,rng=rng,
        parameter_sha256=pytree_sha256(params),optimizer_sha256=pytree_sha256(moments),rng_sha256=canonical_sha256(rng)))


def _state_load(path: Path) -> tuple:
    payload=load_checkpoint(path); m=payload['metadata']
    state=(payload['params'],payload['optimizer_state'],m['rng'])
    if (pytree_sha256(state[0])!=m['parameter_sha256'] or pytree_sha256(state[1])!=m['optimizer_sha256']
            or canonical_sha256(state[2])!=m['rng_sha256']):
        raise ValueError('parameter/optimizer/RNG restoration seal mismatch')
    return state


def _seeds(rng_state: dict) -> tuple[list[int], dict]:
    rng=np.random.default_rng();rng.bit_generator.state=deepcopy(rng_state)
    seeds=[int(s) for s in rng.integers(0,2**32,size=3,dtype=np.uint32)]
    if len(set(seeds))!=3:raise ArithmeticError('role seed collision')
    return seeds,deepcopy(rng.bit_generator.state)


def products(rt: dict, banks: list, rows: np.ndarray, directory: Path, label: str, chunk: int) -> tuple[np.ndarray, float]:
    """Complete projected amplitudes/H/known guide components, with partial IO."""
    out=[]; times=[]
    for first in range(0,len(rows),chunk):
        warm = rt.get('warm_H_seconds', 0.)
        reserve = rt.get('audit_reserve_seconds', 0.) if label=='fit' else 120.
        if 'remaining' in rt and rt['remaining']()<1.3*(len(rows)-first)*len(banks)*warm+reserve:
            raise TimeoutError('measured complete fixed-panel H work plus audit reserve exceeds budget')
        tick=perf_counter(); panel=[]
        for bank in banks:
            values=jax.block_until_ready(rt['H'](bank,rows[first:first+chunk]))
            panel.append(np.stack([np.asarray(v) for v in values],axis=1))
        out.append(np.stack(panel,axis=1));times.append(perf_counter()-tick)
        rt['warm_H_seconds']=max(warm,times[-1]/len(out[-1])/len(banks)) if first else warm
        if first==0 or (first//chunk)%8==0:
            atomic_json(directory/(label+'_progress.json'),dict(completed=first+len(out[-1]),records=len(rows),
                scalar_H_centers=sum(len(x) for x in out)*len(banks),batch_seconds=times))
            np.savez_compressed(directory/(label+'_partial_products.npz'),rows=rows[:sum(len(x) for x in out)],products=np.concatenate(out))
    data=np.concatenate(out)
    np.savez_compressed(directory/(label+'_products.npz'),rows=rows,products=data,batch_seconds=times)
    warm=sum(times[1:])/sum(len(x) for x in out[1:])/len(banks) if len(times)>1 else times[0]/len(rows)/len(banks)
    return data,warm


def transition(plan: dict, rt: dict, state: tuple, directory: Path, *, replay: bool=False) -> tuple[tuple,dict]:
    """All active-W update; replay uses fixed H data, not another science draw."""
    directory.mkdir(parents=True,exist_ok=True); parent,moments,rng=state
    seeds,following=_seeds(rng);tick=perf_counter()
    rt['sampling_reserve_seconds']=rt.get('audit_reserve_seconds',0.)+120
    stage_budget(rt,rt['sampling_reserve_seconds'],'training normalization and fit draw')
    normalized,note=canonical_active_sampling_bank(parent,rt['cfg'],rt['profile'])
    fit_records=plan.get('fit_records',512)
    split=plan.get('calibration_split',256)
    epsilon=plan.get('epsilon',.125)
    rows,draw=exact_draw([normalized],rt,fit_records,seeds[0],bridge=True)
    if not replay:
        save_checkpoint(directory/'fit_draw.pkl',draw,metadata=dict(role='fixed_fit_sampling',seed=seeds[0]))
    if replay:
        cache=np.load(directory/'fit_products.npz')
        if not np.array_equal(rows,cache['rows']):raise ArithmeticError('fit RNG replay changed rows')
        data=cache['products'];warm=float(json.loads((directory/'transition.json').read_text())['warm_H_seconds'])
    else:
        data,warm=products(rt,[normalized],rows,directory,'fit',plan['H_chunk'])
    if not np.isfinite(data).all() or np.max(abs(data.imag))>2e-11*max(np.max(abs(data)),1e-300):
        raise ArithmeticError('real active tree produced nonreal or invalid complete-H products')
    checked=np.stack([np.asarray(v) for v in jax.block_until_ready(rt['H'](normalized,rows[:2]))],axis=1)
    replay_error=float(np.max(np.max(abs(checked-data[:2,0]),axis=0)/np.maximum(np.max(abs(data[:2,0]),axis=0),1e-300)))
    if not np.isfinite(replay_error) or replay_error>2e-11:
        raise ArithmeticError('restored active complete-H product replay failed')
    density=full_support_sz0_density(data[:,0,2].real,data[:,0,3].real,n_sites=144,epsilon=epsilon)
    scale=np.sqrt(note['original_raw_norm']); hpsi=data[:,0,1].real*scale
    stage_budget(rt,rt.get('audit_reserve_seconds',0.)+120,'full active SR Jacobian/solve')
    proposal,next_moments,sr=jax.block_until_ready(rt['update'](parent,moments,jnp.asarray(rows),
        jnp.asarray(hpsi),jnp.asarray(density),plan.get('learning_rate',.03)))
    peak=max(int((d.memory_stats() or {}).get('peak_bytes_in_use',0)) for d in jax.devices())
    if peak>plan['maximum_peak_device_bytes']:raise MemoryError('registered70GiB peak ceiling exceeded')
    if not bool(sr['accepted']):raise ArithmeticError('full active amplitude-SR numerical proposal failed')
    delta=jax.tree.map(lambda a,b:a-b,proposal,parent)
    candidates=[jax.tree.map(lambda p,d:p+f*d,parent,delta) for f in FACTORS]
    stage_budget(rt,rt.get('audit_reserve_seconds',0.)+120,'finite candidate normalization and geometry')
    canonical=[canonical_active_sampling_bank(p,rt['cfg'],rt['profile'])[0] for p in candidates]
    selection,sel_draw=exact_draw(canonical,rt,split,seeds[1],bridge=False)
    guard,guard_draw=exact_draw(canonical,rt,split,seeds[2],bridge=False)
    if not replay:
        save_checkpoint(directory/'geometry_draw.pkl',(sel_draw,guard_draw),metadata=dict(role='independent_selection_and_veto'))
    geometry=np.concatenate((selection,guard)); columns=[]; guides=[]
    # Known mixture guide Qorbit, evaluated by the SAME exact query kernel;
    # H is unnecessary here: raw-orbit guide is accumulated over all images.
    raw=make_feature_lut_amplitude(rt['cfg'],active_top_widths=rt['profile'])
    groups=jnp.asarray(rt['images'].reshape(-1,plan['projection_chunk'],144))
    def phi_guide(bank,row):
        def image_group(ids):
            v=jax.vmap(raw,in_axes=(None,0))(bank,row[ids])
            return jnp.stack((v.sum(),jnp.sum(v*v)))
        return jnp.sum(jax.lax.map(image_group,groups),axis=0)/len(rt['images'])
    query=jax.jit(lambda bank,x:jax.lax.map(lambda row:phi_guide(bank,row),x,batch_size=plan['action_chunk']))
    for bank in canonical:
        stage_budget(rt,rt.get('audit_reserve_seconds',0.)+120,'next complete-A1 finite geometry column')
        values=np.asarray(jax.block_until_ready(query(rt['prepare'](bank),geometry)))
        columns.append(values[:,0]);guides.append(values[:,1])
    phi=np.stack(columns,axis=1);q=np.stack(guides,axis=1).mean(1)
    chosen,calibration=select_step(phi,q,split=split,maximum_FS=plan.get('maximum_FS',.003))
    info=dict(accepted=chosen!=4,selected_index=int(chosen),seeds=seeds,
        parent_parameter_sha256=pytree_sha256(parent),next_rng_sha256=canonical_sha256(following),
        SR={k:np.asarray(v).item() for k,v in sr.items()},calibration=calibration,
        warm_H_seconds=warm,wall_seconds=perf_counter()-tick,
        full_H_centers=2 if replay else fit_records+2,complete_H_replay_relative_error=replay_error,
        full_active_jacobian_rows=fit_records,
        peak_device_bytes=peak,
        actual_profile=rt['profile'],native_padded_parameter_gradients_not_claimed=True)
    candidate=(candidates[chosen],next_moments,following) if chosen!=4 else state
    if replay:
        expected=_state_load(directory/'provisional.pkl')
        # Tight relative floating comparison rather than host/node equality.
        differences=[]
        for actual,old in zip(jax.tree.leaves(candidate[:2]),jax.tree.leaves(expected[:2]),strict=True):
            a,b=np.asarray(actual),np.asarray(old)
            differences.append(float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)))
        if max(differences,default=0)>2e-11 or canonical_sha256(candidate[2])!=canonical_sha256(expected[2]):
            raise ArithmeticError('fresh-process full active update/optimizer/RNG replay failed')
        info['maximum_state_relative_replay_error']=max(differences,default=0)
        expected_info=json.loads((directory/'transition.json').read_text())
        if chosen!=expected_info['selected_index']:raise ArithmeticError('geometry decision changed in replay')
        atomic_json(directory/'replay.json',info)
    else:
        np.savez_compressed(directory/'geometry_fixture.npz',rows=geometry,phi=phi,proposal=q)
        _state_save(directory/'provisional.pkl',candidate,provisional=True,transition=info)
        atomic_json(directory/'transition.json',info)
    return candidate,info


def run(plan_path: Path, output: Path, phase: str) -> dict:
    plan=json.loads(plan_path.read_text());cfg,profile=checked_plan(plan)
    if not plan.get('execution_authorized'):raise PermissionError('root must authorize this independent pilot')
    if jax.default_backend()!='gpu' or not any('A100' in d.device_kind for d in jax.devices()):
        raise RuntimeError('Kim pilot requires a real allocated A100; no CPU training')
    source=Path(plan['parent_checkpoint'])
    if file_sha256(source)!=plan['parent_sha256']:raise ValueError('frozen initial checkpoint changed')
    output.mkdir(parents=True,exist_ok=True)
    registration_path=output/'registration.json'
    if phase=='prepare':
        if registration_path.exists():raise FileExistsError('fresh pilot registration required')
        payload=load_checkpoint(source)
        validate_parent_coupling(plan,payload['metadata'])
        actual=payload['metadata']['embedding']['active_top_widths']
        actual=tuple({int(q):int(d) for q,d in actual[str(i) if str(i) in actual else i].items()} for i in range(15))
        if actual!=profile:raise ValueError('initial actual profile differs from plan')
        params=repack_active_clebsch_bank(payload['params'],cfg,profile)
        count=sum(np.size(v) for v in jax.tree.leaves(params))
        if count<1 or ('active_real_parameters' in plan and count!=plan['active_real_parameters']):
            raise ValueError('preselected active parameter count differs from sealed plan')
        zero=jax.tree.map(jnp.zeros_like,params)
        rng=np.random.default_rng(plan['seeds']['training']).bit_generator.state
        state=(params,(jnp.asarray(0),zero,zero),rng)
        _state_save(output/'parent.pkl',state,completed_updates=0)
        atomic_json(registration_path,dict(plan=plan,plan_sha256=canonical_sha256(plan),
            started_epoch=time(),active_real_parameters=count,legacy_native_padded_parameters=7856880,
            sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
                package_prefixes=('su2_nqs','projects.clebsch_tree_nqs')),
            runtime=dict(jax=jax.__version__,devices=[d.device_kind for d in jax.devices()]),
            separate_CT12_budget_not_CT6_campaign=True))
    registration=json.loads(registration_path.read_text())
    if canonical_sha256(plan)!=registration['plan_sha256']:raise ValueError('pilot plan changed after registration')
    current_sources=transitive_local_source_manifest((Path(__file__),),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    if current_sources!=registration['sources']:
        raise ValueError('source manifest changed between independent pilot processes')
    stopping=[False]
    def requested_stop(*_):stopping[0]=True
    for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,requested_stop)
    scheduler_end=os.environ.get('SLURM_JOB_END_TIME')
    if scheduler_end is not None:
        scheduler_end=float(scheduler_end)
        if not np.isfinite(scheduler_end):raise ValueError('invalid scheduler absolute deadline')
    def remaining():
        if stopping[0]:return 0.
        cap=14400-(time()-registration['started_epoch'])
        return cap if scheduler_end is None else min(cap,scheduler_end-time()-180.)
    parent=_state_load(output/'parent.pkl');rt=runtime(plan,parent[0])
    rt['remaining']=remaining
    preflight_path=output/'cost_preflight.json'
    if preflight_path.exists():
        preflight=json.loads(preflight_path.read_text())
        rt['warm_H_seconds']=preflight['warm_seconds']/preflight['records']
        rt['audit_reserve_seconds']=1.3*2*plan['audit_records']*rt['warm_H_seconds']+600
    if phase=='prepare':
        # Fixed complete-H cold/warm throughput preflight before costly fit.
        canonical,note=canonical_active_sampling_bank(parent[0],cfg,profile)
        fixture_path=Path(plan['initial_replay_fixture'])
        if file_sha256(fixture_path)!=plan['initial_replay_fixture_sha256']:
            raise ValueError('CPU complete-H runtime fixture changed')
        fixture=np.load(fixture_path)
        validate_runtime_fixture(plan,fixture)
        start=perf_counter()
        measured=np.stack([np.asarray(v) for v in jax.block_until_ready(rt['H'](canonical,fixture['rows']))],axis=1)
        Z=note['original_raw_norm'];measured=measured*np.asarray([np.sqrt(Z),np.sqrt(Z),Z,Z])[None,:]
        golden=fixture['products']
        errors=np.max(abs(measured-golden),axis=0)/np.maximum(np.max(abs(golden),axis=0),1e-300)
        atomic_json(output/'initial_complete_H_replay.json',dict(component_order=['Phi','Hphi','Qorbit','Qpush'],
            maximum_component_relative_errors=errors.tolist(),passed=bool(np.isfinite(errors).all() and max(errors)<=2e-11),
            runtime_fixture_sha256=plan['initial_replay_fixture_sha256'],full_H_centers=len(measured),
            raw_original_norm=Z,cold_seconds=perf_counter()-start))
        if not np.isfinite(errors).all() or max(errors)>2e-11:
            raise ArithmeticError('CPU-to-A100 complete-H saved-state replay failed')
        preflight_records=plan['H_chunk']
        rows,_=exact_draw([canonical],rt,preflight_records,plan['seeds']['training']+31,bridge=True)
        times=[]
        for _ in range(2):
            start=perf_counter();jax.block_until_ready(rt['H'](canonical,rows));times.append(perf_counter()-start)
        requested_centers=2*(plan['fit_records']+plan['audit_records'])
        expected=1.3*requested_centers*times[1]/preflight_records+1800
        atomic_json(output/'cost_preflight.json',dict(cold_seconds=times[0],warm_seconds=times[1],
            records=preflight_records,total_requested_science_H_centers=requested_centers,
            full_H_preflight_centers=2*preflight_records+len(measured),
            conservative_H_plus_compilation_and_restore_seconds=expected,remaining_seconds=remaining(),
            full_spatial_images=len(rt['images']),physical_bonds=len(rt['bonds']),j2=plan['j2']))
        if expected>remaining():
            report=dict(status='cost_gate_parent_preserved_no_update',cost_forecast=expected)
            atomic_json(output/'prepare_result.json',report)
            return report
        rt['warm_H_seconds']=times[1]/preflight_records
        rt['audit_reserve_seconds']=1.3*2*plan['audit_records']*rt['warm_H_seconds']+600
        candidate,info=transition(plan,rt,parent,output/'step1')
        report=dict(status='step1_provisional_requires_fresh_process_replay',step1=info,remaining_seconds=remaining())
    elif phase=='replay_train':
        prepared=json.loads((output/'prepare_result.json').read_text())
        if prepared['status']=='cost_gate_parent_preserved_no_update':
            report=dict(status='prepare_cost_gate_no_replay_or_training')
            atomic_json(output/'replay_train_result.json',report)
            return report
        if (output/'checkpoint.pkl').exists():
            raise FileExistsError('a committed prefix exists; never replay/commit step1 twice')
        candidate,replay=transition(plan,rt,parent,output/'step1',replay=True)
        if not replay['accepted']:
            report=dict(status='geometry_rejected_zero_updates',replay=replay)
            atomic_json(output/'replay_train_result.json',report)
            return report
        _state_save(output/'checkpoint.pkl',candidate,completed_updates=1,replay_passed=True)
        state=_state_load(output/'checkpoint.pkl')
        reserve=1.3*2*plan['audit_records']*replay['warm_H_seconds']+600
        if remaining()<reserve+1.3*replay['wall_seconds']:
            report=dict(status='cost_gate_one_update_prefix_frozen',completed_updates=1,reserved_audit_seconds=reserve)
        else:
            final,info=transition(plan,rt,state,output/'step2')
            completed=2 if info['accepted'] else 1
            _state_save(output/'checkpoint.pkl',final,completed_updates=completed,replay_passed=True)
            report=dict(status='fixed_two_updates_complete' if completed==2 else 'geometry_one_update_prefix',
                completed_updates=completed,step2=info)
    elif phase=='audit':
        if not (output/'checkpoint.pkl').exists():
            report=dict(status='no_committed_update_no_final_quality_claim')
            atomic_json(output/'audit_result.json',report)
            return report
        saved=load_checkpoint(output/'checkpoint.pkl');final=_state_load(output/'checkpoint.pkl')
        banks=[canonical_active_sampling_bank(s[0],cfg,profile)[0] for s in (parent,final)]
        audit=output/'audit';audit.mkdir(exist_ok=False)
        rt['sampling_reserve_seconds']=1.3*2*plan['audit_records']*rt.get('warm_H_seconds',0.)+120
        rows,draw=exact_draw(banks,rt,plan['audit_records'],plan['seeds']['audit'],bridge=True)
        product,warm=products(rt,banks,rows,audit,'untouched',plan['H_chunk'])
        q=full_support_sz0_density(product[:,:,2].real.mean(1),product[:,:,3].real.mean(1),n_sites=144,epsilon=plan['epsilon'])
        moments=paired_iid_moments(product[:,:,0],product[:,:,1],q,
            np.repeat(np.arange(plan['audit_groups']),plan['audit_records']//plan['audit_groups']),
            n_sites=144,seed=plan['seeds']['statistics'])
        report=dict(status='complete_fixed_prefix_independent_audit',completed_updates=saved['metadata']['completed_updates'],
            moments=moments,complete_A1=True,images=len(rt['images']),physical_bonds=len(rt['bonds']),j2=plan['j2'],
            independent_from_training_and_geometry=True,scalar_full_H_centers=2*plan['audit_records'],
            remaining_seconds=remaining(),warm_H_seconds=warm)
    else:raise ValueError('phase is prepare, replay_train or audit')
    atomic_json(output/(phase+'_result.json'),report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('prepare','replay_train','audit'),required=True)
    a=p.parse_args()
    try:
        result=run(a.plan,a.output,a.phase)
    except (TimeoutError,MemoryError) as error:
        result=dict(status='bounded_partial_no_record_count_reduction',phase=a.phase,error=str(error),
            committed_checkpoint_preserved=(a.output/'checkpoint.pkl').exists())
        a.output.mkdir(parents=True,exist_ok=True)
        atomic_json(a.output/(a.phase+'_interruption.json'),result)
    print(json.dumps(result,indent=2))
