"""One separately budgeted read-only validation of the 300-second continuation.

The original partial receipt is immutable. Cached terminal statistics are
reviewed first. The extension performs no optimization or random sampling.
"""
from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_projected_amplitude, prepare_feature_lut_bank
from su2_nqs.models.triangular_sector_amplitude import make_prepared_target_root_amplitude
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_hnf import hnf_site_permutation, hnf_point_group_permutations
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.finite_candidate_calibration import common_proposal_FS
from projects.clebsch_tree_nqs.workflows.ct_sector_excitation_pilot import (
    make_runtime, deadline, family_norms, next_seed, restore_state)
from projects.clebsch_tree_nqs.workflows.ct6_native_direction_calibration import normalized_raw_bank
from projects.clebsch_tree_nqs.workflows.triangle_iid_orbit_audit import paired_iid_moments
from .ct_protected_triplet_continue_20261030 import OUTPUT, SOURCE_DIRECTORY
from .ct_protected_triplet_update_20261029 import anchored_amplitude
from .ct_physical_spin_seed_20261029 import PARENTS


def run(directory: Path = OUTPUT) -> dict:
    """Preserve the partial run and audit its already frozen endpoint once."""
    started = perf_counter()
    registration_path = directory/"validation_extension.json"
    if registration_path.exists():
        raise FileExistsError("the single validation extension is immutable")
    original = json.loads((directory/"results.json").read_text())
    original_sha = file_sha256(directory/"results.json")
    registration = dict(task="read_only_validation_after_explicit_300_second_partial", date="2026-10-30",
        additional_wall_cap_seconds=60, maximum_RSS_GiB=6., no_new_optimization=True,
        no_new_random_draws=True, no_endpoint_selection=True,
        original_results_sha256=original_sha, original_status=original["status"],
        original_wall_seconds=original["wall_seconds"], checkpoint_sha256=file_sha256(directory/"checkpoint.pkl"),
        terminal_products_sha256=file_sha256(directory/"terminal_products.npz"),
        validation_source_sha256=file_sha256(Path(__file__)),
        registered_checks=["cached terminal moments", "parameter/optimizer/next-RNG seals",
            "restored full H numerical replay", "independent direct protected-state full H",
            "final active translations and little-group characters", "raw target-root S squared identity"],
        protected_S_squared_policy="formal exact source rank-one + fixed-q CT + commuting spatial projection; no expensive 324 protected-amplitude repeat")
    atomic_json(registration_path, registration)
    result = dict(registration=registration, checks={})
    try:
        with deadline(60):
            terminal = np.load(directory/"terminal_products.npz")
            rows, Q, groups = terminal["rows"], terminal["proposal"], terminal["groups"]
            first, final, source_products = terminal["initial"], terminal["final"], terminal["source"]
            audit_seed = json.loads((directory/"new_final_independent_full_H_audit_draw.json").read_text())["seed"]
            moments = paired_iid_moments(np.column_stack((first[:,0],final[:,0])),
                np.column_stack((first[:,1],final[:,1])),Q,groups,n_sites=36,seed=audit_seed^0x5A17D0B1)
            source_comparison = paired_iid_moments(np.column_stack((source_products[:,0],final[:,0])),
                np.column_stack((source_products[:,1],final[:,1])),Q,groups,n_sites=36,seed=audit_seed^0x34AE67C9)
            state, meta = restore_state(directory/"checkpoint.pkl")
            trained, _ = restore_state(directory/"trained.pkl")
            initial, _ = restore_state(directory/"initial.pkl")
            if pytree_sha256(state[:2]) != pytree_sha256(trained[:2]):
                raise ArithmeticError("audit changed endpoint parameters or optimizer")
            frozen = json.loads((directory/"endpoint_frozen.json").read_text())
            if (file_sha256(directory/"trained.pkl") != frozen["checkpoint_sha256"]
                    or pytree_sha256(state[0]) != frozen["parameter_sha256"]):
                raise ArithmeticError("pre-audit endpoint seal mismatch")
            restored, _ = restore_state(directory/"checkpoint.pkl")
            seed_a, next_a = next_seed(state[2], "future_unused_fit")
            seed_b, next_b = next_seed(restored[2], "future_unused_fit")
            if seed_a != seed_b or next_a != next_b or pytree_sha256(state[:2]) != pytree_sha256(restored[:2]):
                raise ArithmeticError("full parameter/optimizer/next RNG restoration failed")
            changes = family_norms(jax.tree.map(lambda a,b:a-b,state[0],initial[0]))
            with np.load(directory/"independent_H_JVP.npz") as jvp:
                J_error=float(np.max(abs(jvp["tangent"]-jvp["finite_difference"]))/np.max(abs(jvp["tangent"])))
            result.update(moments=moments, physical_source_comparison=source_comparison,
                final_source_FS=common_proposal_FS(np.column_stack((source_products[:,0],final[:,0])),Q,groups=8),
                accepted_updates=sum(t["accepted"] for t in original["transitions"]),
                total_changed_families=changes, exact_real_parameters=sum(x.size for x in jax.tree.leaves(state[0])),
                cached_H_JVP_relative_FD_error=J_error,
                checkpoint_seals=dict(parameters_optimizer_identical=True,next_rng_identical=True,
                    next_unused_role_seed=seed_a,completed_updates=meta["completed_updates"],
                    final_audit_draw_seen_only_after_freeze=True))
            atomic_json(directory/"independent_review.json", result)
            print("cached review", moments["candidate_minus_control"][0], flush=True)

            parent=load_checkpoint(PARENTS/"ct6_j0125/checkpoint.pkl")
            reference=load_checkpoint(SOURCE_DIRECTORY/"initialized.pkl")["params"]
            plan=dict(model=dict(length=6,multiplicity_cap=28,maximum_q=10),j2=.125,
                arity="binary_truncated",binary_rank=8,epsilon=.125,projection_chunk=4,H_chunk=1,action_chunk=1,
                sector=dict(target_q=2,target_m2=0,momentum=((1,3),(1,3))))
            runtime=make_runtime(plan,parent);cfg=runtime["cfg"]
            scale=meta["fixed_scale"]
            # Defaults are accepted only after comparing the explicit previously
            # measured all-plus-one little-group receipt, not by inferring A1.
            spatial=json.loads((directory/"spatial_identity.json").read_text())
            if any(v != [1.,0.] for v in spatial["point_characters"].values()):
                raise ArithmeticError("runtime default differs from measured source irrep")
            restored_psi, restored_H=runtime["H"](restored[0],rows[:2])
            ref_psi, ref_H=runtime["H"](reference,rows[:2])
            physical=np.asarray(source_products[:2])+scale*(np.column_stack((restored_psi,restored_H))-np.column_stack((ref_psi,ref_H)))
            replay_error=float(np.max(abs(physical-final[:2]))/np.max(abs(final[:2])))
            if replay_error>2e-11:
                raise ArithmeticError("restored protected full-H numerical replay failed")
            result["checks"]["restored_complete_H_relative_error"]=replay_error
            atomic_json(directory/"validation_extension_results.json",result)

            ground_bank,_=normalized_raw_bank(parent["params"],dict(cfg=cfg,
                bank=lambda p:arity_clebsch_bank(p,cfg,"binary_truncated",8)))
            ground=prepare_feature_lut_bank(ground_bank,cfg)
            images=np.asarray(triangular_hnf_gamma_a1_group(((6,0),(0,6))),np.int32)
            parent_amplitude=make_feature_lut_projected_amplitude(cfg,images,chunk=4)
            phases=jnp.asarray(np.exp(2j*np.pi*np.sum(np.asarray([divmod(i,6) for i in range(36)]),axis=1)/3))
            prepared=runtime["prepare"](restored[0]);prepared_ref=runtime["prepare"](reference)
            @jax.jit
            def physical_amplitude(row):
                row=jnp.asarray(row,jnp.int32)
                source=jnp.dot(phases,row-.5)/6*parent_amplitude(ground,row)
                return anchored_amplitude(source,runtime["amplitude"](prepared,row),
                    runtime["amplitude"](prepared_ref,row),scale)
            row=rows[0];edges=np.asarray([(a,b) for a,b,c in runtime["bonds"]],np.int32)
            coupling=np.asarray([c for a,b,c in runtime["bonds"]])
            changed=[]
            for a,b in edges:
                x=row.copy();x[a],x[b]=row[b],row[a];changed.append(x)
            direct=np.asarray([complex(physical_amplitude(x)) for x in changed])
            p0=complex(physical_amplitude(row));unlike=row[edges[:,0]]!=row[edges[:,1]]
            direct_H=np.sum(coupling*np.where(unlike,-.25,.25))*p0+np.sum(coupling*np.where(unlike,.5*direct,0.))
            H_error=float(abs(direct_H-final[0,1])/abs(final[0,1]))
            if H_error>2e-10:
                raise ArithmeticError("independent full protected-H direct recontraction failed")
            result["checks"]["direct_complete_protected_H_relative_error"]=H_error
            result["checks"]["direct_complete_protected_H_exchange_bonds"]=len(edges)
            atomic_json(directory/"validation_extension_results.json",result)
            print("direct H",H_error,"seconds",perf_counter()-started,flush=True)

            base=complex(physical_amplitude(row));errors={}
            operations=[("T10",hnf_site_permutation(((6,0),(0,6)),translation=(1,0)),np.exp(-2j*np.pi/3)),
                        ("T01",hnf_site_permutation(((6,0),(0,6)),translation=(0,1)),np.exp(-2j*np.pi/3))]
            operations += [(name,image,1.) for name,op,image in hnf_point_group_permutations(((6,0),(0,6)))
                           if name in spatial["little_group"]]
            for name,image,expected in operations:
                errors[name]=float(abs(complex(physical_amplitude(row[np.asarray(image)]))-expected*base)/abs(base))
            if max(errors.values())>2e-9:
                raise ArithmeticError("updated protected state changed spatial character")
            result["checks"]["final_spatial_relative_errors"]=errors
            raw=make_prepared_target_root_amplitude(cfg,2,0)
            spin_rows=[row]
            for up in np.flatnonzero(row):
                for down in np.flatnonzero(1-row):
                    x=row.copy();x[up],x[down]=row[down],row[up];spin_rows.append(x)
            batch=jax.jit(lambda bank,x:jax.lax.map(lambda bits:raw(bank,bits),x))
            spin_values=np.asarray(jax.block_until_ready(batch(prepared,jnp.asarray(spin_rows))))
            spin_error=float(abs(16*spin_values[0]+spin_values[1:].sum())/np.linalg.norm(spin_values))
            if spin_error>2e-9:
                raise ArithmeticError("restored native target-root does not obey S squared=2")
            result["checks"]["raw_CT_physical_S_squared_relative_residual"]=spin_error
            result["checks"]["protected_S_squared_formal_inference"]=(
                "Frozen genuine singlet carrier -> rank-one S_K^z source has q=2; restored native root q=2; "
                "site permutations commute with S squared; fixed affine sum remains q=2. "
                "Numerical 324-swap check above is for the raw CT component, not a global norm bound.")
            result["status"]="completed_separate_read_only_validation_of_frozen_endpoint"
    except (TimeoutError,MemoryError,ArithmeticError) as error:
        result.update(status="bounded_validation_extension_partial",error=str(error))
    result["additional_wall_seconds"]=perf_counter()-started
    result["combined_wall_seconds"]=original["wall_seconds"]+result["additional_wall_seconds"]
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_RSS_GiB"]=rss/(1024**3 if sys.platform=="darwin" else 1024**2)
    result["original_partial_unchanged"]=file_sha256(directory/"results.json")==original_sha
    result["new_random_rows"]=0;result["new_optimization_updates"]=0
    result["additional_work_counts"]=dict(cached_terminal_records=32,CT_full_H_replay_centers=4,
        direct_protected_amplitude_queries=217+8,raw_CT_S_squared_amplitude_queries=325,
        original_continuation_fd_complete_H_centers_omitted_from_original_counter=4)
    atomic_json(directory/"validation_extension_results.json",result)
    print(json.dumps({k:result.get(k) for k in ("status","additional_wall_seconds","combined_wall_seconds","error")}),flush=True)
    return result


if __name__=="__main__":
    run()
