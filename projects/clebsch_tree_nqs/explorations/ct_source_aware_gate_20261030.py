"""Bounded fresh S1K source-aware IID importance coverage and cost gate.

No native parameter is updated. Two independent64-row panels compare the
normalized original proposal and a known positive F_K tilt with UNKNOWN Z.
Their scale-free ESS and normalized physical ratios are legitimate; absolute
norm estimates from the tilted relative density are deliberately absent.
"""
from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json,pytree_sha256
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank
from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.full_support_proposal import full_support_sz0_density,full_support_sz0_constants
from su2_nqs.vmc.spin_source_action import make_spin_source_products
from projects.clebsch_tree_nqs.workflows.ct_sector_excitation_pilot import make_runtime,deadline,restore_state
from projects.clebsch_tree_nqs.workflows.ct6_native_direction_calibration import normalized_raw_bank
from .ct_physical_spin_seed_20261029 import PARENTS
from .ct_protected_triplet_continue_20261030 import OUTPUT as CONTINUATION,SOURCE_DIRECTORY
from .ct_source_aware_proposal_20261030 import triangular_K_tilt

ROOT=Path(__file__).resolve().parents[3]
OUTPUT=ROOT/"projects/clebsch_tree_nqs/results/ct_source_aware_S1K_20261030/v1"


def concentration(mass: np.ndarray) -> dict:
    """Common-scale-free finite contribution concentration, not tail precision."""
    mass=np.asarray(mass,float);total=mass.sum()
    return dict(ESS=float(total**2/np.sum(mass*mass)),maximum_fraction=float(mass.max()/total),records=len(mass))


def relative_moments(products: np.ndarray,qtilde: np.ndarray,groups: int=8) -> dict:
    """Node-safe normalized physical ratios with an arbitrary shared Q scale."""
    p,h=products.T;labels=np.repeat(np.arange(groups),len(p)//groups)
    def values(ids):
        z=np.sum(abs(p[ids])**2/qtilde[ids]);e=np.sum(p[ids].conj()*h[ids]/qtilde[ids])/z
        v=np.sum(abs(h[ids]-e*p[ids])**2/qtilde[ids])/z
        return np.asarray([e.real/36,e.imag/36,v/36])
    point=values(np.arange(len(p)))
    leave=np.asarray([values(np.flatnonzero(labels!=g)) for g in range(groups)])
    se=np.sqrt((groups-1)/groups*np.sum((leave-leave.mean(0))**2,axis=0))
    E=np.sum(p.conj()*h/qtilde)/np.sum(abs(p)**2/qtilde)
    return dict(metric_order=["energy_per_site","imaginary_energy_per_site","variance_per_site"],
        values=point.tolist(),group_jackknife_SE=se.tolist(),
        norm_concentration=concentration(abs(p)**2/qtilde),H2_concentration=concentration(abs(h)**2/qtilde),
        residual_concentration=concentration(abs(h-E*p)**2/qtilde),
        normalized_ratios_finite_sample_biased=True,unknown_common_Q_normalizer_cancels=True,
        absolute_norm_not_reported=True,unseen_tail_precision_not_certified=True)


def gpu_plan(fit_records: int,registration: dict,timings: dict,reference_file: Path) -> dict:
    """Prepare a concrete128/256 ALL-W protocol; never allocate or submit it."""
    if fit_records not in (128,256):raise ValueError("full-W fits are128 or256, never16")
    seed_offset=0 if fit_records==128 else 100
    role_seeds={name:3960302001+seed_offset+i for i,name in enumerate(
        ("fit","FS_selection_guard","independent_H_JVP","energy_confirmation","new_terminal","statistics"))}
    count=fit_records+128+64+128+256
    return dict(task="S1K_source_aware_complete_native_W_physical_direction_confirmation",
        status="prepared_not_executed_or_submitted",execution_authorized=False,
        target_q=2,target_m2=0,momentum=[[1,3],[1,3]],little_group="A1_all_plus_one",
        parent_checkpoint=registration["frozen_protected_checkpoint"],
        parent_checkpoint_sha256=registration["frozen_protected_checkpoint_sha256"],
        parent_parameter_sha256=registration["frozen_parameter_sha256"],
        fixed_source_carrier=registration["singlet_carrier_checkpoint"],
        fixed_source_carrier_sha256=registration["singlet_carrier_checkpoint_sha256"],
        original_CT_reference=registration["original_CT_reference"],
        fixed_affine_scale=registration["fixed_affine_scale"],
        sampler=dict(base="normalized raw-orbit/H-push/uniform epsilon=1/8",tilt_floor=.1,
            tilt_factor_squared_bound=3.,law="q=Qbase*(.1+|F_K|²)/Z",normalizer_known=False,
            relative_density_only=True,estimated_Z_never_inserted_into_Q=True,
            exact_IID_rejection=True,source_nodes_and_full_tangent_supported=True),
        role_seeds=role_seeds,fit_records=fit_records,all_real_native_parameters=1542328,
        all_local_six_top_and_root_trainable=True,no_new_head_or_trainable_scale=True,
        fixed_proposals=1,learning_rate=1.,damping=.05,tangent_FS=.01,
        complex_SR_parameter_chunk=8192,score_chunk=1,
        raw_complex128_J_bytes=16*fit_records*1542328,
        finite_FS=dict(records=128,selection=64,independent_guard=64,groups=8,
            fractions=[1.,.5,.25,.125,0.],target=.01,maximum_guard=.015,
            maximum_cumulative_source_FS=.04,maximum_source_nominal_upper=.06,
            failed_guard_cannot_reselect_smaller_fraction=True),
        independent_direction=dict(records=64,groups=8,complete_complex_H_JVP=True,
            bilateral_slope_both_sides=True,no_target_wavefunction_division=True,
            not_reused_for_fit_or_FS=True),
        independent_energy_confirmation=dict(records=128,groups=16,
            accept_only_if_paired_delta_energy_plus_2_SE_below_zero=True,
            requires_positive_norm_H2_tangent_coverage_and_numerical_SR_replay=True,
            veto_only_no_retuning=True),
        new_terminal=dict(records=256,groups=32,draw_only_after_endpoint_frozen=True,
            full_H_nodes_retained=True,not_used_for_acceptance_or_endpoint_selection=True),
        data_disjoint_from_local_gate_and_prior32_audit=True,
        runtime_replay=dict(file=str(reference_file.relative_to(ROOT)),sha256=file_sha256(reference_file),
            same_complex_scale_no_phase_alignment=True,require_amplitude_H_Qtilde_replay=True,
            require_full_parameters_optimizer_next_RNG_restore=True),
        measured_CPU_seconds=dict(timings,approximate_source_Q_cost_for_roles=count*timings["source_H_Q_warm_seconds_per_center"]),
        resources=dict(partition="kim",account="kim",qos="normal",
            gres="gpu:nvidia_a100_80gb_pcie:1",cpus=4,host_memory_GiB=128,
            science_wall_seconds=3600 if fit_records==128 else 5400,
            maximum_device_memory_GiB=70,requested_node=None),
        conditional_execution="root/user allocation authority and successful CPU-to-GPU numerical replay required",
        no_remote_job_submission=True,keep_existing_allocations_unchanged=True,
        no_claim_of_optimized_sector_energy_until_actual_fit_and_new_terminal_finish=True)


def run() -> dict:
    """One predeclared360s CPU coverage gate, immutable physics endpoints."""
    if OUTPUT.exists():raise FileExistsError("new immutable gate output required")
    OUTPUT.mkdir(parents=True)
    state,meta=restore_state(CONTINUATION/"checkpoint.pkl")
    validation=json.loads((CONTINUATION/"validation_extension_results.json").read_text())
    if validation["status"]!="completed_separate_read_only_validation_of_frozen_endpoint":
        raise ValueError("independently replayed protected endpoint required")
    parent_path=PARENTS/"ct6_j0125/checkpoint.pkl";parent=load_checkpoint(parent_path)
    ref_path=SOURCE_DIRECTORY/"initialized.pkl";reference=load_checkpoint(ref_path)["params"]
    guide=triangular_K_tilt(6,floor=.1)
    seeds=dict(baseline=3960301001,guided=3960301002,statistics=3960301003)
    excluded=set(state[2]["used_seeds"]+state[2]["excluded_seeds"])
    if set(seeds.values())&excluded:raise ValueError("new role collides with frozen history")
    registration=dict(task="known_FK_positive_tilt_source_coverage_and_cost_gate",date="2026-10-30",
        wall_cap_seconds=360,maximum_RSS_GiB=6.,records_per_law=64,groups=8,seeds=seeds,
        floor=.1,source_factor_squared_bound=3.,rejection_envelope=guide.envelope,
        base_uniform_fraction=.125,base_H_bridge_fraction=.5,
        normalized_density_for_guided_law_not_known=True,
        reported_guided_weight="q_tilde=Qbase*(.1+|F_K|²); common Z cancels in ratios only",
        frozen_protected_checkpoint=str((CONTINUATION/"checkpoint.pkl").relative_to(ROOT)),
        frozen_protected_checkpoint_sha256=file_sha256(CONTINUATION/"checkpoint.pkl"),
        frozen_parameter_sha256=pytree_sha256(state[0]),
        singlet_carrier_checkpoint=str(parent_path.relative_to(ROOT)),
        singlet_carrier_checkpoint_sha256=file_sha256(parent_path),
        original_CT_reference=str(ref_path.relative_to(ROOT)),original_CT_reference_sha256=file_sha256(ref_path),
        fixed_affine_scale=meta["fixed_scale"],no_optimization_or_checkpoint_selection=True,
        gate_criteria=dict(minimum_guided_actual_target_norm_ESS=16,minimum_guided_source_H2_ESS=16,
            maximum_guided_target_weight_fraction=.2,minimum_online_ESS_per_second_ratio=1.1),
        local_128_direction_requires_full_disjoint_protocol_fits_remaining_wall_and_6GiB=True,
        no_16_record_direction_substitution=True,no_remote_submission=True)
    atomic_json(OUTPUT/"registration.json",registration)
    atomic_json(OUTPUT/"source_manifest.json",transitive_local_source_manifest([Path(__file__)],root=ROOT,
        package_prefixes=("su2_nqs","projects.clebsch_tree_nqs")))
    start=perf_counter();result=dict(registration=registration,panels={},work=dict(new_rows=0,source_full_H_Q_centers=0,
        CT_amplitude_centers=0,CT_H_profile_centers=0,optimizer_updates=0,fit_direction_rows=0))
    work=result["work"]
    try:
        with deadline(360):
            plan=dict(model=dict(length=6,multiplicity_cap=28,maximum_q=10),j2=.125,
                arity="binary_truncated",binary_rank=8,epsilon=.125,projection_chunk=4,H_chunk=1,action_chunk=1,
                sector=dict(target_q=2,target_m2=0,momentum=((1,3),(1,3))))
            runtime=make_runtime(plan,parent);cfg=runtime["cfg"]
            raw_bank,norm_note=normalized_raw_bank(parent["params"],dict(cfg=cfg,
                bank=lambda p:arity_clebsch_bank(p,cfg,"binary_truncated",8)))
            ground=prepare_feature_lut_bank(raw_bank,cfg)
            images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
            phases=np.exp(2j*np.pi*np.sum(np.asarray([divmod(i,6) for i in range(36)]),axis=1)/3)
            couplings=np.abs([c for a,b,c in runtime["bonds"]]);prob=couplings/couplings.sum()
            action,_=make_spin_source_products(lambda edges:make_feature_lut_exchange_queries(cfg,edges),
                images,runtime["bonds"],prob,phases[None],n_sites=36,projection_chunk=4)
            scalar=jax.jit(runtime["amplitude"]);prepared=runtime["prepare"](state[0]);prepared_ref=runtime["prepare"](reference)
            rows_by_law={};draw_seconds={};draw_receipts={}
            before=perf_counter();rows_by_law["baseline"],base_note=runtime["draw"](64,seeds["baseline"])
            draw_seconds["baseline"]=perf_counter()-before
            draw_receipts["baseline"]=dict(records=64,base_candidates=64,empirical_acceptance=1.,normalizer_known=True)
            before=perf_counter()
            rows_by_law["guided"],draw_receipts["guided"]=guide.draw(runtime["draw"],64,
                np.random.default_rng(seeds["guided"]),batch_size=256,maximum_candidates=20_000)
            draw_seconds["guided"]=perf_counter()-before
            for name in rows_by_law:
                atomic_json(OUTPUT/(name+"_draw.json"),draw_receipts[name])
                np.savez_compressed(OUTPUT/(name+"_rows.npz"),rows=rows_by_law[name])
                work["new_rows"]+=64
            print("draws",draw_seconds,"guided candidates",draw_receipts["guided"]["base_candidates"],flush=True)
            stored={};batch_times=[]
            for name,rows in rows_by_law.items():
                begin=perf_counter();parts=[];law_times=[]
                for first in range(0,len(rows),8):
                    tick=perf_counter();part=tuple(np.asarray(x) for x in jax.block_until_ready(action(ground,jnp.asarray(rows[first:first+8]))))
                    law_times.append(perf_counter()-tick);parts.append(part);work["source_full_H_Q_centers"]+=8
                    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform=="darwin" else 1024**2)
                    if rss>6:raise MemoryError("registered6GiB cap exceeded")
                    print(name,"source-H/Q",first+8,"/64",flush=True)
                batch_times+=law_times[1:]
                phi,hp,qo,qpush=[np.concatenate([x[i] for x in parts]) for i in range(4)]
                Qbase=full_support_sz0_density(qo,qpush,n_sites=36,epsilon=.125)
                qrelative=Qbase if name=="baseline" else guide.relative_density(Qbase,rows)
                amp_start=perf_counter()
                current=np.asarray([complex(scalar(prepared,jnp.asarray(row))) for row in rows])
                ref=np.asarray([complex(scalar(prepared_ref,jnp.asarray(row))) for row in rows])
                protected=phi[:,1]+meta["fixed_scale"]*(current-ref);work["CT_amplitude_centers"]+=128
                amp_seconds=perf_counter()-amp_start
                query_seconds=perf_counter()-begin
                source_products=np.column_stack((phi[:,1],hp[:,1]))
                node_counts=np.stack([rows[:,np.flatnonzero(np.asarray([(i//6+i%6)%3 for i in range(36)])==c)].sum(axis=1) for c in range(3)],axis=1)
                known_nodes=np.all(node_counts==6,axis=1)
                source_stats=relative_moments(source_products,qrelative)
                actual_stats=concentration(abs(protected)**2/qrelative)
                online=query_seconds+draw_seconds[name]
                panel=dict(actual_protected_target_norm=actual_stats,source_physical_ratios=source_stats,
                    sampling_seconds=draw_seconds[name],source_H_Q_batch_seconds=law_times,
                    CT_amplitude_seconds=amp_seconds,total_online_seconds=online,
                    actual_target_norm_ESS_per_second=actual_stats["ESS"]/online,
                    source_H2_ESS_per_second=source_stats["H2_concentration"]["ESS"]/online,
                    known_FK_node_records=int(known_nodes.sum()),
                    relative_node_H2_mass_fraction=float(np.sum(abs(hp[known_nodes,1])**2/qrelative[known_nodes])/
                        np.sum(abs(hp[:,1])**2/qrelative)),
                    normalizer_known=name=="baseline",absolute_norm_not_reported=True,
                    wavefunction_and_frozen_checkpoint_unchanged=True)
                result["panels"][name]=panel
                stored[name]=dict(rows=rows,source=source_products,Qbase=Qbase,qrelative=qrelative,protected=protected)
                np.savez_compressed(OUTPUT/(name+"_products.npz"),rows=rows,source_products=source_products,
                    normalized_base_density=Qbase,relative_density=qrelative,protected_amplitude=protected,
                    guide_weight=guide.weight(rows),known_FK_nodes=known_nodes)
                atomic_json(OUTPUT/"progress.json",result)
                print("gate",name,"target ESS",actual_stats["ESS"],"H2 ESS",source_stats["H2_concentration"]["ESS"],flush=True)
            # CPU/GPU replay reference and warm H throughput, not new training or
            # another independent physical audit: these are consumed gate rows.
            replay_rows=rows_by_law["guided"][:2];tick=perf_counter()
            cp,ch=runtime["H"](state[0],replay_rows);rp,rh=runtime["H"](reference,replay_rows)
            warm_start=perf_counter();cp2,ch2=runtime["H"](state[0],replay_rows)
            warm_H=(perf_counter()-warm_start)/2;work["CT_H_profile_centers"]+=6
            np.testing.assert_allclose(np.column_stack((cp,ch)),np.column_stack((cp2,ch2)),rtol=0,atol=0)
            physical=stored["guided"]["source"][:2]+meta["fixed_scale"]*(np.column_stack((cp,ch))-np.column_stack((rp,rh)))
            replay_file=OUTPUT/"runtime_replay_reference.npz"
            np.savez_compressed(replay_file,rows=replay_rows,source_products=stored["guided"]["source"][:2],
                physical_products=physical,normalized_base_density=stored["guided"]["Qbase"][:2],
                relative_density=stored["guided"]["qrelative"][:2],guide_weight=guide.weight(replay_rows))
            timings=dict(source_H_Q_warm_seconds_per_center=float(np.median(batch_times)/8),
                CT_H_warm_seconds_per_center=warm_H,CT_H_profile_compile_and_warm_seconds=perf_counter()-tick,
                base_draw_seconds_per_record=draw_seconds["baseline"]/64,
                guided_draw_seconds_per_record=draw_seconds["guided"]/64)
            base=result["panels"]["baseline"];guided=result["panels"]["guided"];criteria=registration["gate_criteria"]
            coverage_pass=(guided["actual_protected_target_norm"]["ESS"]>=criteria["minimum_guided_actual_target_norm_ESS"]
                and guided["source_physical_ratios"]["H2_concentration"]["ESS"]>=criteria["minimum_guided_source_H2_ESS"]
                and guided["actual_protected_target_norm"]["maximum_fraction"]<=criteria["maximum_guided_target_weight_fraction"])
            speed_ratio=guided["actual_target_norm_ESS_per_second"]/base["actual_target_norm_ESS_per_second"]
            result.update(status="completed_fresh_source_aware_coverage_gate",timings=timings,
                coverage_criteria_passed=bool(coverage_pass),online_effective_rate_ratio=speed_ratio,
                source_known_factor_helpfulness_not_statistical_architecture_win=True,
                proposal_mechanism="known cheap spin factor, no learned attention",
                physical_parent_and_source_unchanged=True,all_W_direction_computed=False)
            plans={}
            for n in (128,256):
                p=gpu_plan(n,registration,timings,replay_file)
                atomic_json(OUTPUT/(f"Kim_fit{n}_protocol.json"),p);plans[str(n)]=p
            estimate=plans["128"]["measured_CPU_seconds"]["approximate_source_Q_cost_for_roles"]
            result["local_128_direction_decision"]=dict(executed=False,
                reason="Complete128 fit plus disjoint FS/H-JVP/confirmation/terminal exceeds remaining registered CPU cap; no reduction to16",
                estimated_source_Q_seconds_for_full_protocol=estimate,
                remaining_registered_seconds=360-(perf_counter()-start),
                exact_retained_J_GiB_128=16*128*1542328/1024**3,
                exact_retained_J_GiB_256=16*256*1542328/1024**3,
                J_bytes_are_lower_bound_not_peak_memory_certificate=True,
                prepared_Kim_protocols=["Kim_fit128_protocol.json","Kim_fit256_protocol.json"],
                protocol_prepared_is_not_a_completed_physical_direction=True)
            result["analytic_normalized_guided_uniform_floor_lower_bound"]=(
                guide.lower_density_ratio_bound*full_support_sz0_constants(36,epsilon=.125)["minimum_density"])
            result["normalization_note"]="Guided q_tilde is NOT normalized. Z is unknown and never estimated for weights. Only common-scale-free ratios and ESS are reported."
            result["raw_bank_normalization"]=norm_note
    except (TimeoutError,MemoryError,ArithmeticError) as error:
        result.update(status="bounded_partial_source_aware_gate",error=str(error))
    result["wall_seconds"]=perf_counter()-start
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_RSS_GiB"]=rss/(1024**3 if sys.platform=="darwin" else 1024**2)
    result["frozen_checkpoint_unchanged"]=(file_sha256(CONTINUATION/"checkpoint.pkl")==registration["frozen_protected_checkpoint_sha256"])
    atomic_json(OUTPUT/"results.json",result)
    print(json.dumps({k:result.get(k) for k in ("status","wall_seconds","peak_RSS_GiB","error")}),flush=True)
    return result


if __name__=="__main__":
    run()
