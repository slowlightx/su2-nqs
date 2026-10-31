"""Executable isolated Kim128 protocol; preparation never submits a job.

One full-native-W proposal:128 fit,128 finite FS,128 independent complete
H/JVP and finite-energy veto, then512 NEW terminal IID records. The known
positive F_K tilt has unknown common Z. Only scale-free ratios are reported.
The physical source, singlet carrier, CT reference and affine scale are fixed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import signal
import sys
from time import perf_counter,time
from typing import Any

import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,canonical_sha256,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank,make_feature_lut_projected_amplitude
from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
from su2_nqs.models.triangular_sector_amplitude import triangular_sector_projector
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.complex_importance_amplitude_sr import make_complex_importance_amplitude_sr
from su2_nqs.vmc.finite_candidate_calibration import select_finite_grid,common_proposal_FS,bilateral_direction_moments
from su2_nqs.vmc.full_support_proposal import full_support_sz0_density
from su2_nqs.vmc.spin_source_action import make_spin_source_products
from projects.clebsch_tree_nqs.workflows.ct_sector_excitation_pilot import (
    make_runtime,deadline,save_state,restore_state,next_seed,family_norms)
from projects.clebsch_tree_nqs.workflows.ct6_native_direction_calibration import normalized_raw_bank
from projects.clebsch_tree_nqs.workflows.triangle_iid_orbit_audit import paired_iid_moments
from .ct_source_aware_proposal_20261030 import triangular_K_tilt
from .ct_source_aware_gate_20261030 import OUTPUT as GATE,concentration
from .ct_protected_triplet_continue_20261030 import OUTPUT as PARENT_DIRECTORY,SOURCE_DIRECTORY
from .ct_physical_spin_seed_20261029 import PARENTS

ROOT=Path(__file__).resolve().parents[3]
PREPARATION=ROOT/"projects/clebsch_tree_nqs/results/ct_source_aware_S1K_20261030/v2_preparation"
TASK="S1K_known_source_tilt_one_complete_W_direction_fit128_FS128_bilateral128_terminal512_v2"
ROLES=("fit128","finite_FS128","independent_bilateral128","new_terminal512","statistics")


def paired_relative_moments(psi: np.ndarray,hpsi: np.ndarray,qtilde: np.ndarray,
    groups: np.ndarray,*,n_sites: int,seed: int) -> dict[str,Any]:
    """Node-safe paired physical ratios; never label unknown-Z mass as norm."""
    result=paired_iid_moments(psi,hpsi,qtilde,groups,n_sites=n_sites,seed=seed)
    for name in ("endpoint_values","endpoint_iid_delta_SE","endpoint_independent_block_jackknife_SE"):
        result[name]=np.asarray(result[name])[:,:3].tolist()
    result["endpoint_block_bootstrap_95"]=np.asarray(result["endpoint_block_bootstrap_95"])[:,:3,:].tolist()
    for name in ("candidate_minus_control","paired_iid_delta_SE","paired_independent_block_jackknife_SE"):
        result[name]=result[name][:3]
    result["paired_block_bootstrap_95"]=result["paired_block_bootstrap_95"][:3]
    h2=abs(np.asarray(hpsi))**2/np.asarray(qtilde)[:,None]
    result["target_node_H2_fraction"]=(np.where(np.asarray(psi)==0,h2,0).sum(0)/h2.sum(0)).tolist()
    for name in ("target_node_H2_contributions","norm_weight_maximum","raw_norm_numerators_unbiased"):
        result.pop(name,None)
    result.update(metric_order=result["metric_order"][:3],normalizer_known=False,
        absolute_target_norm_not_reported=True,unknown_common_Z_cancels_in_all_reported_ratios=True,
        raw_relative_sums_estimate_norm_over_unknown_Z_not_norm=True)
    return result


def coverage(products: np.ndarray,qtilde: np.ndarray,*,minimum_fraction: float=.25,
    maximum_fraction: float=.2) -> dict[str,Any]:
    """Actual norm/H²/residual contribution coverage; no local-energy division."""
    p,h=np.asarray(products).T;q=np.asarray(qtilde)
    energy=np.sum(p.conj()*h/q)/np.sum(abs(p)**2/q)
    masses=dict(norm=abs(p)**2/q,H2=abs(h)**2/q,residual=abs(h-energy*p)**2/q)
    metrics={name:concentration(mass) for name,mass in masses.items() if mass.sum()>0}
    passed=len(metrics)==3 and all(x["ESS"]>=minimum_fraction*len(q)
        and x["maximum_fraction"]<=maximum_fraction for x in metrics.values())
    return dict(passed=bool(passed),metrics=metrics,records=len(q),minimum_ESS=minimum_fraction*len(q),
        maximum_single_fraction=maximum_fraction,unseen_tail_coverage_not_certified=True)


def independent_direction_statistics(products: np.ndarray,tangent: np.ndarray,qtilde: np.ndarray,
    *,n_sites: int=36,groups: int=16) -> dict[str,Any]:
    """Keep BOTH coherent bra/ket derivatives and their finite-panel errors."""
    p,h=np.asarray(products).T;d,hd=np.asarray(tangent).T;q=np.asarray(qtilde)
    bilateral=bilateral_direction_moments(p,h,d,hd,q,groups=groups,n_sites=n_sites)
    E=np.sum(p.conj()*h/q)/np.sum(abs(p)**2/q)
    # Physical ray coverage must not depend on a parallel normalization or
    # global-phase component. Project H*d consistently, not just d.
    ray_coefficient=np.sum(p.conj()*d/q)/np.sum(abs(p)**2/q)
    d_perp=d-ray_coefficient*p;hd_perp=hd-ray_coefficient*h
    tangent_mass=abs(d_perp)**2/q
    force=np.real(d_perp.conj()*(h-E.real*p)+p.conj()*(hd_perp-E.real*d_perp))/q
    resolved_tangent=(tangent_mass.sum()>(64*np.finfo(float).eps)**2*np.sum(abs(d)**2/q))
    tangent_metric=concentration(tangent_mass) if resolved_tangent else dict(ESS=0.,maximum_fraction=1.,records=len(q))
    force_metric=concentration(abs(force)) if abs(force).sum()>0 else dict(ESS=0.,maximum_fraction=1.,records=len(q))
    slope=bilateral["values"][2];se=bilateral["independent_group_jackknife_SE"][2]
    passed=(tangent_metric["ESS"]>=len(q)/4 and force_metric["ESS"]>=len(q)/4
        and tangent_metric["maximum_fraction"]<=.2 and force_metric["maximum_fraction"]<=.2
        and slope+2*se<0)
    return dict(passed=bool(passed),bilateral=bilateral,chosen_tangent_square=tangent_metric,
        absolute_bilateral_force_projection=force_metric,
        both_coherent_derivatives_retained=True,slope_upper_two_SE=slope+2*se,
        complex_ray_projection_applied_to_tangent_and_H_tangent=True,
        removed_ray_coefficient=[float(ray_coefficient.real),float(ray_coefficient.imag)],
        projected_tangent_resolved_above_roundoff=bool(resolved_tangent),
        chosen_direction_not_a_global_tangent_metric_certificate=True)


def record_nested_seed_lineage(rng_state: dict,children: list[int]) -> dict:
    """Reject aliases before H work and preserve every used child for restart."""
    if any(type(v) is not int or not 0<=v<2**32 for v in children):raise ValueError("uint32 child seed labels required")
    old=set(rng_state["used_seeds"]+rng_state.get("excluded_seeds",[]))
    if len(set(children))!=len(children) or set(children)&old:
        raise ArithmeticError("nested rejection-base seeds alias old/reserved/development/previous streams")
    following=deepcopy(rng_state)
    following["excluded_seeds"]=sorted(set(following.get("excluded_seeds",[]))|set(children))
    return following


def observed_device_peak_bytes(devices: list[Any]|None=None) -> int|None:
    """Return a measured peak, or None when the backend exposes no certificate."""
    devices=list(jax.local_devices()) if devices is None else devices
    peaks=[]
    for device in devices:
        stats=device.memory_stats()
        if stats is None or "peak_bytes_in_use" not in stats:return None
        peak=stats["peak_bytes_in_use"]
        if isinstance(peak,bool) or not isinstance(peak,(int,np.integer)) or peak<0:return None
        peaks.append(int(peak))
    return max(peaks) if peaks else None


def reserved_rng_and_seeds(parent_rng: dict,gate: Path=GATE) -> tuple[dict,dict]:
    """Preserve inherited generator/history and reserve all disjoint role seeds."""
    following=deepcopy(parent_rng)
    gate_reg=json.loads((gate/"registration.json").read_text())
    gate_draw=json.loads((gate/"guided_draw.json").read_text())
    excluded=set(following.get("excluded_seeds",[]))|set(gate_reg["seeds"].values())|set(gate_draw["base_batch_seeds"])
    following["excluded_seeds"]=sorted(excluded)
    seeds={}
    for role in ROLES:
        seeds[role],following=next_seed(following,"registered_"+role)
    return following,seeds


def make_plan() -> dict[str,Any]:
    """Construct the single reviewed128 protocol without executing physics."""
    state,meta=restore_state(PARENT_DIRECTORY/"checkpoint.pkl")
    rng,seeds=reserved_rng_and_seeds(state[2])
    return dict(task=TASK,target_q=2,target_m2=0,momentum=[[1,3],[1,3]],little_character="all_plus_one",
        j2=.125,model=dict(length=6,multiplicity_cap=28,maximum_q=10),arity="binary_truncated",binary_rank=8,
        fit_records=128,FS_records=128,FS_split=64,bilateral_records=128,bilateral_groups=16,
        terminal_records=512,terminal_groups=32,maximum_updates=1,fixed_attempts=1,
        source_tilt_floor=.1,source_factor_squared_bound=3.,base_uniform_fraction=.125,
        common_normalizer_known=False,density_field="q_tilde_not_normalized_Q",
        damping=.05,learning_rate=1.,tangent_FS=.01,target_FS=.01,maximum_FS=.015,
        maximum_source_FS=.04,maximum_source_FS_upper=.06,
        fractions=[1.,.5,.25,.125,0.],minimum_ESS_fraction=.25,maximum_single_fraction=.2,
        require_residual_ESS_before_SR=True,require_independent_bilateral_upper_two_SE_negative=True,
        require_independent_paired_energy_upper_two_SE_negative=True,
        projection_chunk=16,H_chunk=2,action_chunk=1,source_query_chunk=8,score_chunk=1,parameter_chunk_size=8192,
        parent_checkpoint=str((PARENT_DIRECTORY/"checkpoint.pkl").relative_to(ROOT)),
        parent_checkpoint_sha256=file_sha256(PARENT_DIRECTORY/"checkpoint.pkl"),parent_parameter_sha256=pytree_sha256(state[0]),
        parent_optimizer_sha256=pytree_sha256(state[1]),parent_rng_sha256=canonical_sha256(state[2]),
        fixed_source_carrier=str((PARENTS/"ct6_j0125/checkpoint.pkl").relative_to(ROOT)),
        fixed_source_carrier_sha256=file_sha256(PARENTS/"ct6_j0125/checkpoint.pkl"),
        fixed_CT_reference=str((SOURCE_DIRECTORY/"initialized.pkl").relative_to(ROOT)),
        fixed_CT_reference_sha256=file_sha256(SOURCE_DIRECTORY/"initialized.pkl"),fixed_scale=meta["fixed_scale"],
        real_native_parameters=sum(x.size for x in jax.tree.leaves(state[0])),
        inherited_generator_unchanged_before_role_reservation=True,reserved_role_seeds=seeds,
        reserved_rng_sha256=canonical_sha256(rng),roles_reserved_before_any_data_draw=True,
        no_development_rows_used_for_fit_FS_direction_or_terminal=True,
        runtime_replay=str((GATE/"runtime_replay_reference.npz").relative_to(ROOT)),
        runtime_replay_sha256=file_sha256(GATE/"runtime_replay_reference.npz"),
        resources=dict(partition="kim",account="kim",qos="normal",gres="gpu:nvidia_a100_80gb_pcie:1",
            allocation_wall_seconds=7200,science_wall_seconds=6600,save_reserve_seconds=300,
            host_memory_GiB=128,maximum_device_GiB=70,requested_node=None),
        conservative_AD_compile_reserve_seconds=900,cost_safety_factor=1.5,
        no_statistical_record_reductions=True,no_automatic_continuation=True,no_remote_submission_by_prepare=True,
        release_update_only_after_fixed_terminal_completes=True,terminal_values_never_select_endpoint=True)


def validate_plan(plan: dict[str,Any]) -> None:
    """Fail closed on changed spin, records, source identities or resource caps."""
    fixed=dict(task=TASK,target_q=2,target_m2=0,momentum=[[1,3],[1,3]],little_character="all_plus_one",
        j2=.125,model=dict(length=6,multiplicity_cap=28,maximum_q=10),arity="binary_truncated",binary_rank=8,
        fit_records=128,FS_records=128,FS_split=64,bilateral_records=128,bilateral_groups=16,
        terminal_records=512,terminal_groups=32,maximum_updates=1,fixed_attempts=1,
        source_tilt_floor=.1,source_factor_squared_bound=3.,base_uniform_fraction=.125,
        common_normalizer_known=False,density_field="q_tilde_not_normalized_Q",
        minimum_ESS_fraction=.25,maximum_single_fraction=.2,fractions=[1.,.5,.25,.125,0.],
        damping=.05,learning_rate=1.,tangent_FS=.01,target_FS=.01,maximum_FS=.015,
        maximum_source_FS=.04,maximum_source_FS_upper=.06)
    if any(plan.get(k)!=v for k,v in fixed.items()):raise ValueError("single strict128/128/128/512 source-aware protocol required")
    for name in ("target_q","target_m2","fit_records","FS_records","FS_split","bilateral_records","terminal_records"):
        if type(plan.get(name)) is not int:raise TypeError("exact doubled labels and integer record counts required")
    if any(type(v) is not int for pair in plan["momentum"] for v in pair):
        raise TypeError("exact integer momentum fractions required")
    if plan.get("real_native_parameters")!=1542328 or not np.isfinite(plan.get("fixed_scale",np.nan)) or plan["fixed_scale"]<=0:
        raise ValueError("full fixed1542328 native coordinates and positive fixed source scale required")
    r=plan.get("resources",{})
    if (r.get("allocation_wall_seconds")!=7200 or r.get("science_wall_seconds")!=6600
            or r.get("maximum_device_GiB")!=70 or r.get("requested_node") is not None
            or r.get("gres")!="gpu:nvidia_a100_80gb_pcie:1"):
        raise ValueError("one2h A10080 allocation without node pinning required")
    if set(plan.get("reserved_role_seeds",{}))!=set(ROLES) or len(set(plan["reserved_role_seeds"].values()))!=len(ROLES):
        raise ValueError("disjoint pre-reserved statistical streams required")
    for name in ("parent_checkpoint_sha256","parent_parameter_sha256","parent_optimizer_sha256","parent_rng_sha256",
                 "fixed_source_carrier_sha256","fixed_CT_reference_sha256","runtime_replay_sha256","reserved_rng_sha256"):
        if len(plan.get(name,""))!=64:raise ValueError("sealed physical/optimizer/RNG/replay provenance required")
    for name in ("require_residual_ESS_before_SR","require_independent_bilateral_upper_two_SE_negative",
                 "require_independent_paired_energy_upper_two_SE_negative","no_statistical_record_reductions",
                 "no_automatic_continuation","release_update_only_after_fixed_terminal_completes",
                 "terminal_values_never_select_endpoint"):
        if plan.get(name) is not True:raise ValueError("strict statistical veto and immutable terminal policy required")


class SourceRuntime:
    """Existing verified physical kernels plus the independent known source tilt."""

    def __init__(self,plan: dict[str,Any],parent: dict,reference: Any):
        p=plan
        runtime_plan=dict(model=p["model"],j2=p["j2"],arity=p["arity"],binary_rank=p["binary_rank"],epsilon=.125,
            projection_chunk=p["projection_chunk"],H_chunk=p["H_chunk"],action_chunk=p["action_chunk"],
            sector=dict(target_q=2,target_m2=0,momentum=((1,3),(1,3))))
        self.raw=make_runtime(runtime_plan,parent);self.p=p;self.guide=triangular_K_tilt(6,floor=.1)
        self.reference=reference;self.prepared_ref=self.raw["prepare"](reference)
        cfg=self.raw["cfg"];cell=((6,0),(0,6))
        projection=triangular_sector_projector(cfg,((1,3),(1,3)))
        if projection.group_order!=216:raise ValueError("K.A1 little-space-group mismatch")
        self.raw_bank,self.normalization=normalized_raw_bank(parent["params"],dict(cfg=cfg,
            bank=lambda x:arity_clebsch_bank(x,cfg,p["arity"],p["binary_rank"])))
        self.ground=prepare_feature_lut_bank(self.raw_bank,cfg)
        images=np.asarray(triangular_hnf_gamma_a1_group(cell),np.int32)
        phases=np.exp(2j*np.pi*np.sum(np.asarray([divmod(i,6) for i in range(36)]),axis=1)/3)
        probs=np.abs([c for a,b,c in self.raw["bonds"]]);probs/=probs.sum()
        self.source_action,self.source_note=make_spin_source_products(lambda edges:make_feature_lut_exchange_queries(cfg,edges),
            images,self.raw["bonds"],probs,phases[None],n_sites=36,projection_chunk=p["projection_chunk"])
        parent_amplitude=make_feature_lut_projected_amplitude(cfg,images,chunk=p["projection_chunk"])
        def source(row):return jnp.dot(jnp.asarray(phases),jnp.asarray(row,jnp.int32)-.5)/6*parent_amplitude(self.ground,row)
        def amplitude(prepared,row):
            return source(row)+p["fixed_scale"]*(self.raw["amplitude"](prepared,row)-self.raw["amplitude"](self.prepared_ref,row))
        self.amplitude=amplitude
        self.counts=dict(IID_rows=0,base_rejection_candidates=0,source_full_H_Q_centers=0,
            native_CT_H_centers=0,native_CT_amplitude_centers=0,full_W_complex_J_rows=0,complete_H_JVP_centers=0)
        self.base_seeds=[]
        self.forbidden_base_seeds=set()

    def check_resources(self) -> None:
        peak=observed_device_peak_bytes()
        if peak is None:raise MemoryError("backend peak memory unavailable; cannot certify the registered70GiB guard")
        if peak>self.p["resources"]["maximum_device_GiB"]*1024**3:raise MemoryError("registered70GiB device cap exceeded")

    def draw(self,count: int,seed: int) -> tuple[np.ndarray,dict]:
        def base_draw(n,child_seed):
            if child_seed in self.forbidden_base_seeds or child_seed in self.base_seeds:
                raise ArithmeticError("rejection-base RNG alias detected before physical queries")
            self.base_seeds.append(child_seed)
            return self.raw["draw"](n,child_seed)
        rows,receipt=self.guide.draw(base_draw,count,np.random.default_rng(seed),batch_size=256,
            maximum_candidates=100_000)
        self.counts["IID_rows"]+=count;self.counts["base_rejection_candidates"]+=receipt["base_candidates"]
        return rows,receipt

    def source(self,rows: np.ndarray,*,persist: Path|None=None) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        if persist is not None:persist.mkdir(parents=True,exist_ok=False)
        chunks=[]
        for i in range(0,len(rows),self.p["source_query_chunk"]):
            self.check_resources();x=rows[i:i+self.p["source_query_chunk"]]
            part=tuple(np.asarray(v) for v in jax.block_until_ready(self.source_action(self.ground,jnp.asarray(x))))
            chunks.append(part)
            self.counts["source_full_H_Q_centers"]+=len(x)
            if persist is not None:
                np.savez_compressed(persist/f"chunk_{i:04d}.npz",rows=x,phi=part[0],hphi=part[1],
                    orbit_density=part[2],Hpush_density=part[3])
        phi,hp,qo,qpush=[np.concatenate([v[k] for v in chunks]) for k in range(4)]
        base=full_support_sz0_density(qo,qpush,n_sites=36,epsilon=.125)
        return np.stack((phi,hp),axis=-1),base,self.guide.relative_density(base,rows)

    def H(self,params: Any,rows: np.ndarray,source: np.ndarray,ref: np.ndarray|None=None) -> tuple[np.ndarray,np.ndarray]:
        if ref is None:
            a,b=self.raw["H"](self.reference,rows);ref=np.column_stack((a,b));self.counts["native_CT_H_centers"]+=len(rows)
        a,b=self.raw["H"](params,rows);self.counts["native_CT_H_centers"]+=len(rows)
        return source+self.p["fixed_scale"]*(np.column_stack((a,b))-ref),ref

    def amplitudes(self,params: Any,rows: np.ndarray,source: np.ndarray,ref: np.ndarray|None=None) -> tuple[np.ndarray,np.ndarray]:
        if ref is None:
            ref=self.raw["amplitudes"](self.reference,rows);self.counts["native_CT_amplitude_centers"]+=len(rows)
        actual=self.raw["amplitudes"](params,rows);self.counts["native_CT_amplitude_centers"]+=len(rows)
        return source+self.p["fixed_scale"]*(actual-ref),ref

    def H_JVP(self,params: Any,direction: Any,rows: np.ndarray,source: np.ndarray,ref: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        def product(p):return jnp.asarray(source)+self.p["fixed_scale"]*(
            self.raw["prepared_action"](self.raw["prepare"](p),jnp.asarray(rows))-jnp.asarray(ref))
        values,tangent=jax.block_until_ready(jax.jvp(product,(params,),(direction,)))
        self.counts["complete_H_JVP_centers"]+=len(rows)
        return np.asarray(values),np.asarray(tangent)


def remaining_budget(plan: dict,start: float) -> float:
    """Respect the2h scheduler deadline and the shorter registered science cap."""
    remaining=plan["resources"]["science_wall_seconds"]-(perf_counter()-start)
    if os.environ.get("SLURM_JOB_END_TIME"):
        remaining=min(remaining,float(os.environ["SLURM_JOB_END_TIME"])-time()-plan["resources"]["save_reserve_seconds"])
    return remaining


def phase_costs(plan: dict,timings: dict) -> dict[str,float]:
    """Measured conservative remaining-work gate, without reducing records."""
    s,h,a,d,j=(timings[k] for k in ("source_Q","CT_H","CT_amp","guided_draw","H_JVP"))
    factor=plan["cost_safety_factor"]
    return dict(fit=factor*(128*(s+2*h+d)+plan["conservative_AD_compile_reserve_seconds"]),
        FS=factor*128*(s+6*a+d),independent=factor*128*(s+3*h+j+d),
        terminal=factor*512*(s+3*h+d),save=plan["resources"]["save_reserve_seconds"])


def execute(plan: dict[str,Any],output: Path,*,approved_plan_sha256: str) -> dict[str,Any]:
    """Run one registered GPU direction only after root approves the sealed plan."""
    validate_plan(plan)
    seal=canonical_sha256(plan)
    if approved_plan_sha256!=seal:raise PermissionError("explicit approved sealed-plan hash required")
    if jax.default_backend()!="gpu":raise RuntimeError("this128 protocol executes only in its reviewed GPU allocation")
    if output.exists():raise FileExistsError("fresh isolated GPU output required")
    for path_key,hash_key in (("parent_checkpoint","parent_checkpoint_sha256"),("fixed_source_carrier","fixed_source_carrier_sha256"),
        ("fixed_CT_reference","fixed_CT_reference_sha256"),("runtime_replay","runtime_replay_sha256")):
        if file_sha256(ROOT/plan[path_key])!=plan[hash_key]:raise ValueError("sealed physical/replay file differs")
    inherited,inherited_meta=restore_state(ROOT/plan["parent_checkpoint"])
    if (pytree_sha256(inherited[0])!=plan["parent_parameter_sha256"] or pytree_sha256(inherited[1])!=plan["parent_optimizer_sha256"]
            or canonical_sha256(inherited[2])!=plan["parent_rng_sha256"]):raise ValueError("native parameters/optimizer/RNG identity differs")
    if inherited_meta["fixed_scale"]!=plan["fixed_scale"]:raise ValueError("fixed physical-source affine scale differs")
    rng,seeds=reserved_rng_and_seeds(inherited[2])
    if seeds!=plan["reserved_role_seeds"] or canonical_sha256(rng)!=plan["reserved_rng_sha256"]:
        raise ValueError("reserved next-RNG replay differs")
    state=(inherited[0],inherited[1],rng);initial=state[0];moments=state[1]
    forbidden_initial_seed_history=set(rng["used_seeds"]+rng["excluded_seeds"])
    output.mkdir(parents=True);start=perf_counter();consumed=[];attempted=[]
    metadata=dict(plan=plan,plan_sha256=seal,role_seeds=seeds,consumed_data_roles=consumed,
        attempted_data_roles=attempted,fixed_scale=plan["fixed_scale"],fixed_source_carrier=plan["fixed_source_carrier"],fixed_reference=plan["fixed_CT_reference"])
    atomic_json(output/"registration.json",dict(plan=plan,plan_sha256=seal,approved_plan_sha256=approved_plan_sha256))
    save_state(output/"initial.pkl",state,completed_updates=0,metadata=metadata)
    result=dict(task=TASK,plan_sha256=seal,committed_updates=0,provisional_candidate_accepted=False,
        physical_source_and_scale_fixed=True,all_native_W_in_AD=True,old_development_rows_not_new_audit=True)
    runtime=None;selected=initial;selected_moments=moments;accepted=False
    def save_progress():
        atomic_json(output/"progress.json",dict(result,seconds=perf_counter()-start,consumed_data_roles=consumed))
    def draw_role(role,count):
        nonlocal rng
        attempted.append(role)
        runtime.forbidden_base_seeds=set(rng["used_seeds"]+rng["excluded_seeds"])
        rows,note=runtime.draw(count,seeds[role]);consumed.append(role)
        rng=record_nested_seed_lineage(rng,note["base_batch_seeds"])
        atomic_json(output/(role+"_draw.json"),note);np.savez_compressed(output/(role+"_rows.npz"),rows=rows)
        return rows
    previous_usr1=signal.getsignal(signal.SIGUSR1)
    def allocation_stop(*_:Any):raise TimeoutError("scheduler save signal: preserve zero released update")
    signal.signal(signal.SIGUSR1,allocation_stop)
    try:
        with deadline(plan["resources"]["science_wall_seconds"]):
            parent=load_checkpoint(ROOT/plan["fixed_source_carrier"])
            reference=load_checkpoint(ROOT/plan["fixed_CT_reference"])["params"]
            runtime=SourceRuntime(plan,parent,reference)
            replay=np.load(ROOT/plan["runtime_replay"]);rows=replay["rows"]
            tick=perf_counter();source,base,q=runtime.source(rows);actual,ref=runtime.H(initial,rows,source[:,1])
            replay_errors={}
            for name,observed,expected in (("source",source[:,1],replay["source_products"]),("physical",actual,replay["physical_products"]),
                ("base_Q",base,replay["normalized_base_density"]),("relative_Q",q,replay["relative_density"])):
                err=float(np.max(abs(observed-expected))/np.max(abs(expected)));replay_errors[name]=err
                if err>2e-11:raise ArithmeticError("fixed complex-scale CPU/GPU replay failed: "+name)
            result["CPU_GPU_replay"]=dict(relative_errors=replay_errors,phase_aligned=False,seconds=perf_counter()-tick)
            timings={}
            tick=perf_counter();runtime.source(rows);timings["source_Q"]=(perf_counter()-tick)/len(rows)
            tick=perf_counter();runtime.raw["H"](initial,rows);timings["CT_H"]=(perf_counter()-tick)/len(rows)
            runtime.counts["native_CT_H_centers"]+=len(rows)
            tick=perf_counter();runtime.raw["amplitudes"](initial,rows);timings["CT_amp"]=(perf_counter()-tick)/len(rows)
            runtime.counts["native_CT_amplitude_centers"]+=len(rows)
            # A deterministic all-W JVP capability/cost probe on REPLAY rows is
            # not a small fit or an energy-direction estimate.
            probe=jax.tree.map(lambda x:jnp.sin(jnp.arange(x.size).reshape(x.shape)+.17)*1e-6,initial)
            tick=perf_counter();runtime.H_JVP(initial,probe,rows,source[:,1],ref)
            timings["H_JVP"]=(perf_counter()-tick)/len(rows)
            timings["guided_draw"]=json.loads((GATE/"results.json").read_text())["timings"]["guided_draw_seconds_per_record"]
            costs=phase_costs(plan,timings);result["preflight"]=dict(timings=timings,costs=costs,
                remaining_seconds=remaining_budget(plan,start),raw_complex_J_GiB=16*128*plan["real_native_parameters"]/1024**3,
                two_row_probe_is_not_a_fit_or_a_memory_certificate=True)
            save_progress()
            if sum(costs.values())>remaining_budget(plan,start):
                result.update(status="zero_update_budget_gate_before128_fit",terminal_records_completed=0)
                raise TimeoutError("complete fixed-record protocol exceeds measured remaining allocation; no reduction to16")
            rows=draw_role("fit128",128);source,base,q=runtime.source(rows,persist=output/"fit128_source_chunks");fit,ref=runtime.H(initial,rows,source[:,1])
            np.savez_compressed(output/"fit128_products.npz",rows=rows,relative_density=q,normalized_base_density=base,
                source=source,physical=fit,reference_CT=ref)
            result["fit_coverage"]=coverage(fit,q)
            candidate=initial;proposed_moments=moments;can_propose=result["fit_coverage"]["passed"]
            if can_propose:
                update=make_complex_importance_amplitude_sr(lambda p,x:runtime.amplitude(runtime.raw["prepare"](p),x),initial,
                    damping=plan["damping"],score_chunk=1,parameter_chunk_size=8192,sr_tangent_fs_radius=plan["tangent_FS"],
                    prepare=runtime.raw["prepare"],prepared_amplitude=runtime.amplitude)
                candidate,proposed_moments,info=jax.block_until_ready(update(initial,moments,jnp.asarray(rows),
                    jnp.asarray(fit[:,1]),jnp.asarray(q),plan["learning_rate"]))
                runtime.counts["full_W_complex_J_rows"]+=128
                result["SR"]={k:np.asarray(v).item() for k,v in info.items()}
                save_state(output/"provisional_full_W_proposal.pkl",(candidate,proposed_moments,rng),
                    completed_updates=0,metadata=dict(metadata,provisional_only_not_committed=True))
                can_propose=(bool(info["numerically_valid"]) and float(info["relative_SR_residual"])<1e-7
                    and float(info["chosen_direction_tangent_square_ESS"])>=32
                    and float(info["chosen_direction_absolute_force_projection_ESS"])>=32
                    and float(info["chosen_direction_tangent_square_maximum_fraction"])<=.2
                    and float(info["chosen_direction_absolute_force_projection_maximum_fraction"])<=.2)
            result["fit_direction_gate_passed"]=bool(can_propose);save_progress()
            if can_propose:
                if costs["FS"]+costs["independent"]+costs["terminal"]+costs["save"]>remaining_budget(plan,start):
                    raise TimeoutError("insufficient complete-protocol reserve after128 fit; zero release")
                rows=draw_role("finite_FS128",128);source,base,q=runtime.source(rows,persist=output/"finite_FS128_source_chunks")
                candidates=[jax.tree.map(lambda a,b:a+float(f)*(b-a),initial,candidate) for f in plan["fractions"]]
                phi=[];ref_amp=None
                for p in candidates:
                    amp,ref_amp=runtime.amplitudes(p,rows,source[:,1,0],ref_amp);phi.append(amp)
                phi=np.column_stack(phi)
                index,finite=select_finite_grid(phi,q,np.asarray(plan["fractions"]),split=64,groups=8,
                    target=plan["target_FS"],maximum=plan["maximum_FS"])
                source_FS=common_proposal_FS(np.column_stack((source[:,1,0],phi[:,index])),q,groups=16)
                # The helper's norm_estimates are RELATIVE masses; never copy
                # them into a physical absolute-norm report.
                def relative_FS(x):
                    if isinstance(x,dict):return {("relative_mass_estimates" if k=="norm_estimates" else k):relative_FS(v) for k,v in x.items()}
                    if isinstance(x,list):return [relative_FS(v) for v in x]
                    return x
                finite=relative_FS(finite);source_FS=relative_FS(source_FS)
                can_propose=(bool(finite["accepted"]) and index!=len(candidates)-1 and source_FS["FS_radians"]<=.04
                    and source_FS["nominal_upper"]<=.06)
                result.update(finite_FS=finite,source_FS=source_FS,finite_FS_gate_passed=bool(can_propose),
                    chosen_fraction=float(plan["fractions"][index]) if can_propose else 0.)
                np.savez_compressed(output/"finite_FS128.npz",rows=rows,relative_density=q,phi=phi,source=source[:,1,0])
                candidate=candidates[index] if can_propose else initial;save_progress()
                if can_propose:save_state(output/"provisional_finite_candidate.pkl",(candidate,proposed_moments,rng),
                    completed_updates=0,metadata=dict(metadata,provisional_only_not_committed=True))
            if can_propose:
                rows=draw_role("independent_bilateral128",128);source,base,q=runtime.source(rows,persist=output/"independent_bilateral128_source_chunks")
                current,ref=runtime.H(initial,rows,source[:,1]);finite,ref=runtime.H(candidate,rows,source[:,1],ref)
                direction=jax.tree.map(lambda a,b:a-b,candidate,initial)
                value,tangent=runtime.H_JVP(initial,direction,rows,source[:,1],ref)
                np.testing.assert_allclose(value,current,rtol=2e-11,atol=2e-14)
                direction_stats=independent_direction_statistics(value,tangent,q)
                labels=np.repeat(np.arange(16),8)
                finite_stats=paired_relative_moments(np.column_stack((current[:,0],finite[:,0])),
                    np.column_stack((current[:,1],finite[:,1])),q,labels,n_sites=36,seed=seeds["statistics"])
                finite_upper=finite_stats["candidate_minus_control"][0]+2*finite_stats["paired_independent_block_jackknife_SE"][0]
                variance_delta=finite_stats["candidate_minus_control"][2]
                variance_SE=finite_stats["paired_independent_block_jackknife_SE"][2]
                accepted=(direction_stats["passed"] and coverage(current,q)["passed"] and coverage(finite,q)["passed"]
                    and finite_upper<0 and variance_delta<=2*variance_SE)
                result.update(independent_direction=direction_stats,independent_finite_step=finite_stats,
                    independent_finite_energy_upper_two_SE=finite_upper,independent_veto_passed=bool(accepted))
                np.savez_compressed(output/"independent_bilateral128.npz",rows=rows,relative_density=q,
                    initial=value,candidate=finite,tangent=tangent,source=source)
                selected=candidate if accepted else initial;selected_moments=proposed_moments if accepted else moments
                save_progress()
            if costs["terminal"]+costs["save"]>remaining_budget(plan,start):
                accepted=False;selected=initial;selected_moments=moments
                raise TimeoutError("insufficient512 terminal reserve: zero released updates, no sample reduction")
            selected_state=(selected,selected_moments,rng)
            save_state(output/"selected_before_terminal.pkl",selected_state,completed_updates=int(accepted),metadata=metadata)
            atomic_json(output/"endpoint_frozen.json",dict(parameter_sha256=pytree_sha256(selected),
                checkpoint_sha256=file_sha256(output/"selected_before_terminal.pkl"),provisional_updates=int(accepted),
                terminal_rows_not_drawn=True,selection_based_only_on_independent_preterminal_veto=True))
            rows=draw_role("new_terminal512",512);source,base,q=runtime.source(rows,persist=output/"new_terminal512_source_chunks")
            first,ref=runtime.H(initial,rows,source[:,1])
            final=first if not accepted else runtime.H(selected,rows,source[:,1],ref)[0]
            labels=np.repeat(np.arange(32),16)
            terminal=paired_relative_moments(np.column_stack((first[:,0],final[:,0])),np.column_stack((first[:,1],final[:,1])),
                q,labels,n_sites=36,seed=seeds["statistics"]^0x5A17D0B1)
            source_comparison=paired_relative_moments(np.column_stack((source[:,1,0],final[:,0])),
                np.column_stack((source[:,1,1],final[:,1])),q,labels,n_sites=36,seed=seeds["statistics"]^0x34AE67C9)
            np.savez_compressed(output/"terminal512_products.npz",rows=rows,relative_density=q,normalized_base_density=base,
                source=source,initial=first,final=final,groups=labels)
            # Data completion, not terminal numerical values, releases the
            # already frozen independently accepted candidate.
            state=(selected,selected_moments,rng)
            save_state(output/"checkpoint.pkl",state,completed_updates=int(accepted),metadata=metadata)
            restored,_=restore_state(output/"checkpoint.pkl")
            seed_a,rng_a=next_seed(state[2],"next_unused_fit");seed_b,rng_b=next_seed(restored[2],"next_unused_fit")
            if (pytree_sha256(state[:2])!=pytree_sha256(restored[:2]) or seed_a!=seed_b or rng_a!=rng_b):
                raise ArithmeticError("all parameters/optimizer/next RNG restoration failed")
            rs,rb,rq=runtime.source(rows[:2]);restored_H,_=runtime.H(restored[0],rows[:2],rs[:,1])
            replay_err=float(np.max(abs(restored_H-final[:2]))/np.max(abs(final[:2])))
            if replay_err>2e-11:raise ArithmeticError("restored physical complete-H replay failed")
            nested=set(runtime.base_seeds)
            # A child seed may not alias a registered old/reserved role seed.
            if (len(nested)!=len(runtime.base_seeds) or nested&forbidden_initial_seed_history
                    or not nested.issubset(set(rng["excluded_seeds"]))):raise ArithmeticError("nested rejection stream lineage failed")
            final_status=("complete_full_W128_proposal_and_new512_terminal" if runtime.counts["full_W_complex_J_rows"]==128
                else "complete_fit128_coverage_stop_zero_update_and_new512_parent_audit")
            result.update(status=final_status,committed_updates=int(accepted),
                actual_full_W128_direction_computed=runtime.counts["full_W_complex_J_rows"]==128,
                provisional_candidate_accepted=bool(accepted),terminal_records_completed=512,
                terminal=terminal,physical_source_comparison=source_comparison,
                selected_parameter_sha256=pytree_sha256(selected),all_family_changes=family_norms(jax.tree.map(lambda a,b:a-b,selected,initial)),
                checkpoint_replay=dict(all_native_parameters_optimizer_next_RNG_identical=True,next_unused_role_seed=seed_a,
                    physical_full_H_relative_error=replay_err),terminal_values_used_for_selection=False,
                strict_sector=dict(target_q=2,target_m2=0,momentum=[[1,3],[1,3]],little_character="all_plus_one"),
                not_a_lowest_sector_energy_or_certified_gap=True)
    except (TimeoutError,MemoryError,ArithmeticError,AssertionError) as error:
        # A budget/coverage failure never releases an unaudited physical update.
        if runtime is not None:
            rng=deepcopy(rng);rng["excluded_seeds"]=sorted(set(rng["excluded_seeds"])|set(runtime.base_seeds))
        state=(initial,moments,rng)
        result.update(status=result.get("status","zero_update_bounded_or_failed_protocol"),error=str(error),
            committed_updates=0,terminal_not_certified_complete=True)
        save_state(output/"checkpoint.pkl",state,completed_updates=0,metadata=metadata)
    finally:
        signal.signal(signal.SIGUSR1,previous_usr1)
    result["wall_seconds"]=perf_counter()-start
    result["consumed_data_roles"]=consumed
    result["work"]={} if runtime is None else runtime.counts
    result["peak_device_bytes"]=observed_device_peak_bytes()
    result["device_peak_measurement_available"]=result["peak_device_bytes"] is not None
    result["maximum_RSS_GiB"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform=="darwin" else 1024**2)
    result["original_parent_checkpoint_unchanged"]=file_sha256(ROOT/plan["parent_checkpoint"])==plan["parent_checkpoint_sha256"]
    atomic_json(output/"results.json",result)
    return result


def prepare() -> dict[str,Any]:
    """Seal the reviewable runner/plan/payload; no allocation, SSH or Slurm call."""
    if PREPARATION.exists():raise FileExistsError("immutable prepared bundle already exists")
    PREPARATION.mkdir(parents=True)
    plan=make_plan();validate_plan(plan);atomic_json(PREPARATION/"plan.json",plan)
    template=Path(__file__).with_suffix(".sbatch")
    (PREPARATION/"run.sbatch").write_text(template.read_text())
    entries=[Path(__file__),ROOT/"tests/test_ct_source_aware_direction.py",ROOT/"tests/test_ct_source_aware_proposal.py"]
    sources=transitive_local_source_manifest(entries,root=ROOT,package_prefixes=("su2_nqs","projects.clebsch_tree_nqs"))
    files=set(sources)|{"AGENTS.md",plan["parent_checkpoint"],plan["fixed_source_carrier"],plan["fixed_CT_reference"],
        plan["runtime_replay"],str((GATE/"registration.json").relative_to(ROOT)),str((GATE/"guided_draw.json").relative_to(ROOT)),
        str((GATE/"results.json").relative_to(ROOT)),str((PREPARATION/"plan.json").relative_to(ROOT))}
    files.update((str(template.relative_to(ROOT)),str((PREPARATION/"run.sbatch").relative_to(ROOT))))
    manifest={name:file_sha256(ROOT/name) for name in sorted(files)}
    atomic_json(PREPARATION/"payload_manifest.json",manifest)
    (PREPARATION/"sync_files.txt").write_text("\n".join(sorted(files|{
        str((PREPARATION/"payload_manifest.json").relative_to(ROOT))}))+"\n")
    approved=canonical_sha256(plan)
    command=("JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu python -m "
        "projects.clebsch_tree_nqs.explorations.ct_source_aware_direction_20261030 --phase execute "
        "--plan projects/clebsch_tree_nqs/results/ct_source_aware_S1K_20261030/v2_preparation/plan.json "
        "--output NEW_ISOLATED_OUTPUT --approved-plan-sha256 "+approved)
    report=dict(status="prepared_executable_single128_protocol_not_submitted",plan_sha256=approved,
        payload_files=len(manifest),payload_bytes=sum((ROOT/p).stat().st_size for p in manifest),
        payload_manifest_sha256=file_sha256(PREPARATION/"payload_manifest.json"),
        command_for_root_review_only=command,resources=plan["resources"],source_manifest=sources,
        reviewed_A100_hours_upper=2,existing_allocations_changed=False,remote_submission=False,
        actual128_direction_executed=False,actual512_terminal_executed=False)
    atomic_json(PREPARATION/"prepared.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--phase",choices=("prepare","execute"),required=True)
    parser.add_argument("--plan",type=Path);parser.add_argument("--output",type=Path)
    parser.add_argument("--approved-plan-sha256",default="");args=parser.parse_args()
    if args.phase=="prepare":report=prepare()
    else:
        if args.plan is None or args.output is None:parser.error("execution requires sealed plan and fresh output")
        report=execute(json.loads(args.plan.read_text()),args.output,approved_plan_sha256=args.approved_plan_sha256)
    print(json.dumps({k:report.get(k) for k in ("status","committed_updates","wall_seconds","plan_sha256")}),flush=True)
