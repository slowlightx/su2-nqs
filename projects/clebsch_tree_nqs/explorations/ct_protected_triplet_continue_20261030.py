"""Bounded all-W physical-energy continuation of the protected S=1 K state.

The physical source, its singlet carrier, the original CT reference and the
affine scale remain frozen. New IID fit and finite-FS calibration panels are
separate from the final IID audit. Native local/six/top/root parameters are
all retained; no new head or trainable affine scalar is introduced.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest
from su2_nqs.lattices.triangular_hnf import hnf_point_group_permutations, hnf_site_permutation
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group, triangular_hnf_reciprocal_little_group
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank
from su2_nqs.models.triangular_clebsch_feature_lut import make_feature_lut_projected_amplitude, prepare_feature_lut_bank
from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
from su2_nqs.models.triangular_sector_amplitude import (make_sector_prepare,
    make_prepared_sector_amplitude, make_prepared_sector_exchange_queries,
    triangular_sector_projector)
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.complex_importance_amplitude_sr import make_complex_importance_amplitude_sr
from su2_nqs.vmc.finite_candidate_calibration import common_proposal_FS, select_finite_grid
from su2_nqs.vmc.full_support_proposal import full_support_sz0_density
from su2_nqs.vmc.spin_source_action import make_spin_source_products
from projects.clebsch_tree_nqs.workflows.ct_sector_excitation_pilot import (
    make_runtime, deadline, family_norms, next_seed, save_state, restore_state)
from projects.clebsch_tree_nqs.workflows.ct6_native_direction_calibration import normalized_raw_bank
from projects.clebsch_tree_nqs.workflows.triangle_iid_orbit_audit import paired_iid_moments
from .ct_physical_spin_seed_20261029 import PARENTS
from .ct_protected_triplet_update_20261029 import anchored_amplitude

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIRECTORY = ROOT/"projects/clebsch_tree_nqs/results/ct_sector_physical_seed_20261029/j0125_v2"
OUTPUT = ROOT/"projects/clebsch_tree_nqs/results/ct_protected_triplet_continue_20261030/j0125_v1"
FRACTIONS = np.asarray((1., .5, .25, .125, 0.))


def run(output: Path = OUTPUT) -> dict:
    """Attempt exactly two full-W updates inside the 300-second foreground cap."""
    if output.exists():
        raise FileExistsError("new immutable continuation directory required")
    old_directory = SOURCE_DIRECTORY/"protected_physical_update_v1"
    old = json.loads((old_directory/"results.json").read_text())
    frozen = json.loads((old_directory/"endpoint_frozen.json").read_text())
    if old["accepted_updates"] != 1 or file_sha256(old_directory/"checkpoint.pkl") != frozen["checkpoint_sha256"]:
        raise ValueError("verified protected J2=.125 endpoint required")
    inherited = load_checkpoint(old_directory/"checkpoint.pkl")
    initial = inherited["params"]
    if pytree_sha256(initial) != frozen["parameter_sha256"]:
        raise ValueError("protected parameter seal mismatch")
    reference_path = SOURCE_DIRECTORY/"initialized.pkl"
    reference = load_checkpoint(reference_path)["params"]
    parent_path = PARENTS/"ct6_j0125/checkpoint.pkl"
    parent = load_checkpoint(parent_path)
    if pytree_sha256(parent["params"]) != old["registration"]["parent_parameter_sha256"]:
        raise ValueError("physical singlet carrier differs from protected source")
    scale = float(inherited["metadata"]["fixed_scale"])
    if scale != old["fixed_scale"]:
        raise ValueError("frozen source scale mismatch")
    output.mkdir(parents=True)
    master_seed = 3950301001
    excluded = set(parent["metadata"]["rng"]["consumed_role_seeds"]+parent["metadata"]["rng"]["excluded_seeds"])
    excluded.update(json.loads((SOURCE_DIRECTORY/"results.json").read_text())["seeds"].values())
    excluded.update((3940291801, 3940291802, 3940291803, master_seed))
    rng = dict(generator=np.random.default_rng(master_seed).bit_generator.state,
               used_seeds=[], excluded_seeds=sorted(excluded), roles=[])
    state = (initial, inherited["optimizer_state"], rng)
    registration = dict(task="source_protected_S1_K_all_W_physical_energy_continuation", date="2026-10-30",
        j2=.125, target_q=2, target_m2=0, momentum=[[1, 3], [1, 3]],
        fixed_attempts=2, fit_records_per_attempt=16, calibration_records_per_attempt=16,
        fresh_final_audit_records=32, final_groups=8, diagnostic_H_JVP_records=2,
        master_seed=master_seed, fractions=FRACTIONS.tolist(), damping=.05,
        learning_rate=1., tangent_FS=.01, selection_target_FS=.01, maximum_guard_FS=.015,
        maximum_source_FS=.04, maximum_source_FS_upper=.06,
        wall_cap_seconds=300, maximum_RSS_GiB=6.,
        inherited_checkpoint_sha256=file_sha256(old_directory/"checkpoint.pkl"),
        initial_parameter_sha256=pytree_sha256(initial),
        reference_checkpoint_sha256=file_sha256(reference_path),
        singlet_carrier_checkpoint_sha256=file_sha256(parent_path), fixed_scale=scale,
        physical_source_fixed=True, no_new_head=True, all_native_W_trainable=True,
        data_policy="new disjoint fit/calibration/diagnostic/audit; old rows for replay only",
        final_audit_never_selects_checkpoint=True, no_remote_changes=True)
    atomic_json(output/"registration.json", registration)
    atomic_json(output/"source_manifest.json", transitive_local_source_manifest(
        [Path(__file__)], root=ROOT, package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs")))
    started = perf_counter()
    work = dict(new_IID_rows=0, source_full_H_centers=0, CT_full_H_centers=0,
                CT_amplitude_centers=0, complex_SR_J_rows=0, full_H_JVP_centers=0,
                old_rows_for_numerical_replay_only=2)
    report = dict(registration=registration, transitions=[], work=work)
    metadata = dict(registration=registration, reference_checkpoint=str(reference_path),
                    physical_source_carrier=str(parent_path), fixed_scale=scale)
    save_state(output/"initial.pkl", state, completed_updates=0, metadata=metadata)

    def check() -> None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        gib = rss/(1024**3 if sys.platform == "darwin" else 1024**2)
        if gib > 6:
            raise MemoryError("registered 6 GiB RSS cap exceeded")

    try:
        with deadline(300):
            plan = dict(model=dict(length=6, multiplicity_cap=28, maximum_q=10), j2=.125,
                arity="binary_truncated", binary_rank=8, epsilon=.125,
                projection_chunk=4, H_chunk=1, action_chunk=1,
                sector=dict(target_q=2, target_m2=0, momentum=((1, 3), (1, 3))))
            proposal_runtime = make_runtime(plan, parent)
            cfg = proposal_runtime["cfg"]
            cell = ((6, 0), (0, 6))
            little = triangular_hnf_reciprocal_little_group(cell, (12, 12))
            point_characters = {name: 1.+0j for name in little}
            projector = triangular_sector_projector(cfg, ((1, 3), (1, 3)), point_characters=point_characters)
            prepare = jax.jit(make_sector_prepare(cfg, arity="binary_truncated", binary_rank=8))
            amplitude = make_prepared_sector_amplitude(cfg, 2, 0, ((1, 3), (1, 3)),
                point_characters=point_characters, chunk=4)
            scalar = jax.jit(amplitude)
            edges = np.asarray([(a, b) for a, b, _ in proposal_runtime["bonds"]], np.int32)
            coupling = jnp.asarray([v for a, b, v in proposal_runtime["bonds"]])
            exchange = make_prepared_sector_exchange_queries(cfg, 2, 0, ((1, 3), (1, 3)),
                tuple(map(tuple, edges)), point_characters=point_characters, chunk=4)
            def one_h(prepared, row):
                psi, swapped = exchange(prepared, row)
                unlike = row[edges[:, 0]] != row[edges[:, 1]]
                hp = jnp.sum(coupling*jnp.where(unlike, -.25, .25))*psi
                hp += jnp.sum(coupling*jnp.where(unlike, .5*swapped, 0.))
                return jnp.stack((psi, hp))
            action = jax.jit(lambda prepared, rows:jax.lax.map(lambda row:one_h(prepared, row), rows))
            prepared_ref = prepare(reference)
            raw_bank, _ = normalized_raw_bank(parent["params"], dict(cfg=cfg,
                bank=lambda p:arity_clebsch_bank(p, cfg, "binary_truncated", 8)))
            ground = prepare_feature_lut_bank(raw_bank, cfg)
            images = np.asarray(triangular_hnf_gamma_a1_group(cell), np.int32)
            parent_amp = make_feature_lut_projected_amplitude(cfg, images, chunk=4)
            phases = np.exp(2j*np.pi*np.sum(np.asarray([divmod(i, 6) for i in range(36)]), axis=1)/3)
            def source(row):
                row = jnp.asarray(row, jnp.int32)
                return jnp.dot(jnp.asarray(phases), row-.5)/6*parent_amp(ground, row)
            source_scalar = jax.jit(source)
            source_action, _ = make_spin_source_products(lambda pairs:make_feature_lut_exchange_queries(cfg, pairs),
                images, proposal_runtime["bonds"], np.asarray(abs(coupling)/jnp.sum(abs(coupling))),
                phases[None], n_sites=36, projection_chunk=4)
            def anchored(prepared, row):
                return anchored_amplitude(source(row), amplitude(prepared, row),
                    amplitude(prepared_ref, row), scale)
            anchored_scalar = jax.jit(anchored)
            def raw_amplitudes(p, rows):
                bank = prepare(p)
                work["CT_amplitude_centers"] += len(rows)
                return np.asarray([complex(scalar(bank, jnp.asarray(row))) for row in rows])
            def physical_amplitudes(p, rows):
                bank = prepare(p)
                work["CT_amplitude_centers"] += 2*len(rows)
                return np.asarray([complex(anchored_scalar(bank, jnp.asarray(row))) for row in rows])
            def raw_h(p, rows):
                work["CT_full_H_centers"] += len(rows)
                return np.asarray(jax.block_until_ready(action(prepare(p), jnp.asarray(rows))))
            def query_source(rows):
                parts=[]
                for first in range(0, len(rows), 8):
                    check()
                    chunk = rows[first:first+8]
                    parts.append(tuple(np.asarray(x) for x in jax.block_until_ready(source_action(ground, jnp.asarray(chunk)))))
                    work["source_full_H_centers"] += len(chunk)
                phi, hp, qo, qpush = [np.concatenate([x[i] for x in parts]) for i in range(4)]
                Q = full_support_sz0_density(qo, qpush, n_sites=36, epsilon=.125)
                return np.column_stack((phi[:, 1], hp[:, 1])), Q
            def draw_role(count, role):
                nonlocal state
                seed, following = next_seed(state[2], role)
                state = (state[0], state[1], following)
                rows, note = proposal_runtime["draw"](count, seed)
                work["new_IID_rows"] += count
                atomic_json(output/(role+"_draw.json"), dict(seed=seed, role=role,
                    raw_draw_seed=int(note.get("seed", seed))))
                return rows, seed

            # Character checks precede optimization, using consumed rows only
            # for deterministic replay/identity, never as new audit records.
            with np.load(old_directory/"terminal_products.npz") as old_products:
                replay_rows = old_products["rows"][:2]
                replay_source = np.column_stack((old_products["psi"][:2, 0], old_products["hpsi"][:2, 0]))
                replay_expected = np.column_stack((old_products["psi"][:2, 1], old_products["hpsi"][:2, 1]))
            prepare_initial = prepare(initial)
            center = replay_rows[0]
            identity = dict(little_group=list(little), point_characters={k:[v.real, v.imag] for k,v in point_characters.items()},
                group_order=projector.group_order, requested_label="S=1, m2=0, K.A1", checks={})
            base = np.asarray([complex(source_scalar(center)), complex(scalar(prepare_initial, center)),
                               complex(anchored_scalar(prepare_initial, center))])
            if np.min(abs(base)) < 1e-14:
                raise ArithmeticError("nontrivial source/CT identity center required")
            operations = [("T10", hnf_site_permutation(cell, translation=(1, 0)), np.exp(-2j*np.pi/3)),
                          ("T01", hnf_site_permutation(cell, translation=(0, 1)), np.exp(-2j*np.pi/3))]
            operations += [(name, image, point_characters[name]) for name, operation, image in
                           hnf_point_group_permutations(cell) if name in little]
            for name, image, expected in operations:
                transformed = center[np.asarray(image)]
                actual = np.asarray([complex(source_scalar(transformed)), complex(scalar(prepare_initial, transformed)),
                                     complex(anchored_scalar(prepare_initial, transformed))])
                error = abs(actual-expected*base)/abs(base)
                identity["checks"][name] = dict(relative_errors_source_CT_protected=error.tolist(),
                    expected_character=[float(np.real(expected)), float(np.imag(expected))])
                if np.max(error) > 2e-9:
                    atomic_json(output/"spatial_identity.json", identity)
                    raise ArithmeticError("source and CT do not share the requested K little-group character")
            identity["maximum_relative_error"] = max(max(x["relative_errors_source_CT_protected"]) for x in identity["checks"].values())
            atomic_json(output/"spatial_identity.json", identity)
            print("spatial identity", identity["maximum_relative_error"], flush=True)
            replay_current = raw_h(initial, replay_rows)
            replay_reference = raw_h(reference, replay_rows)
            replay_actual = anchored_amplitude(replay_source, replay_current, replay_reference, scale)
            replay_error = float(np.max(abs(replay_actual-replay_expected))/np.max(abs(replay_expected)))
            if replay_error > 2e-10:
                raise ArithmeticError("inherited protected full-H replay failed")
            report["inherited_full_H_replay_relative_error"] = replay_error
            update = make_complex_importance_amplitude_sr(lambda p, row:anchored(prepare(p), row), initial,
                damping=.05, score_chunk=1, parameter_chunk_size=8192, sr_tangent_fs_radius=.01,
                prepare=prepare, prepared_amplitude=anchored)
            for index in (1, 2):
                before = perf_counter()
                params, moments, _ = state
                rows, fit_seed = draw_role(16, f"step{index}_fit")
                source_products, Q = query_source(rows)
                ref_products = raw_h(reference, rows)
                current = anchored_amplitude(source_products, raw_h(params, rows), ref_products, scale)
                np.savez_compressed(output/f"step{index}_fit.npz", rows=rows, proposal=Q, products=current)
                candidate, proposed_moments, sr = jax.block_until_ready(update(params, moments, jnp.asarray(rows),
                    jnp.asarray(current[:, 1]), jnp.asarray(Q), 1.))
                work["complex_SR_J_rows"] += len(rows)
                if not bool(sr["numerically_valid"]) or float(sr["relative_SR_residual"]) >= 1e-7:
                    raise ArithmeticError("all-W complex SR proposal failed exact numerical guard")
                cal_rows, cal_seed = draw_role(16, f"step{index}_finite_FS")
                cal_source, cal_Q = query_source(cal_rows)
                candidates = [jax.tree.map(lambda a,b:a+float(f)*(b-a), params, candidate) for f in FRACTIONS]
                phi = np.column_stack([physical_amplitudes(p, cal_rows) for p in candidates])
                selected, guard = select_finite_grid(phi, cal_Q, FRACTIONS, split=8, groups=4,
                    target=.01, maximum=.015)
                accepted = bool(guard["accepted"]) and selected != len(FRACTIONS)-1
                source_guard = common_proposal_FS(np.column_stack((cal_source[:, 0], phi[:, selected])), cal_Q, groups=4)
                accepted &= source_guard["FS_radians"] <= .04 and source_guard["nominal_upper"] <= .06
                chosen = candidates[selected] if accepted else params
                state = (chosen, proposed_moments if accepted else moments, state[2])
                transition = dict(attempt=index, accepted=bool(accepted), fit_seed=fit_seed, calibration_seed=cal_seed,
                    selected_fraction=float(FRACTIONS[selected]) if accepted else 0., finite_geometry=guard,
                    physical_source_guard=source_guard, full_W_changes=family_norms(jax.tree.map(lambda a,b:a-b, chosen, params)),
                    SR={k:np.asarray(v).item() for k,v in sr.items()}, seconds=perf_counter()-before)
                report["transitions"].append(transition)
                save_state(output/f"step{index}.pkl", state, completed_updates=sum(x["accepted"] for x in report["transitions"]), metadata=metadata)
                atomic_json(output/"progress.json", dict(transitions=report["transitions"], seconds=perf_counter()-started))
                np.savez_compressed(output/f"step{index}_FS.npz", rows=cal_rows, proposal=cal_Q,
                    phi=phi, source=cal_source[:, 0], fractions=FRACTIONS)
                print("protected step", index, "accepted", accepted, "seconds", transition["seconds"], flush=True)

            # Independent full-H JVP is diagnostic, never an update selector.
            rows, diagnostic_seed = draw_role(2, "independent_full_H_JVP")
            diagnostic_source, diagnostic_Q = query_source(rows)
            diagnostic_ref = raw_h(reference, rows)
            delta = jax.tree.map(lambda a,b:a-b, state[0], initial)
            def diagnostic_products(p):
                return jnp.asarray(diagnostic_source)+scale*(action(prepare(p), jnp.asarray(rows))-jnp.asarray(diagnostic_ref))
            values, tangent = jax.block_until_ready(jax.jvp(diagnostic_products, (initial,), (delta,)))
            work["full_H_JVP_centers"] += len(rows)
            eps = 1e-3
            plus = jax.tree.map(lambda a,b:a+eps*b, initial, delta)
            minus = jax.tree.map(lambda a,b:a-eps*b, initial, delta)
            finite = (np.asarray(diagnostic_products(plus))-np.asarray(diagnostic_products(minus)))/(2*eps)
            error = float(np.max(abs(finite-np.asarray(tangent)))/max(np.max(abs(tangent)), 1e-30))
            if error > 2e-6:
                raise ArithmeticError("complete complex H/JVP finite difference failed")
            report["independent_H_JVP"] = dict(seed=diagnostic_seed, records=2,
                relative_finite_difference_error=error, diagnostic_only=True, finite_step_epsilon=eps)
            np.savez_compressed(output/"independent_H_JVP.npz", rows=rows, proposal=diagnostic_Q,
                products=np.asarray(values), tangent=np.asarray(tangent), finite_difference=finite)
            accepted_count = sum(x["accepted"] for x in report["transitions"])
            save_state(output/"trained.pkl", state, completed_updates=accepted_count, metadata=metadata)
            atomic_json(output/"endpoint_frozen.json", dict(parameter_sha256=pytree_sha256(state[0]),
                checkpoint_sha256=file_sha256(output/"trained.pkl"), no_terminal_rows_seen=True))
            rows, audit_seed = draw_role(32, "new_final_independent_full_H_audit")
            source_products, Q = query_source(rows)
            reference_products = raw_h(reference, rows)
            first = anchored_amplitude(source_products, raw_h(initial, rows), reference_products, scale)
            final = anchored_amplitude(source_products, raw_h(state[0], rows), reference_products, scale)
            labels = np.repeat(np.arange(8), 4)
            moments = paired_iid_moments(np.column_stack((first[:, 0], final[:, 0])),
                np.column_stack((first[:, 1], final[:, 1])), Q, labels, n_sites=36, seed=audit_seed^0x5A17D0B1)
            source_comparison = paired_iid_moments(np.column_stack((source_products[:, 0], final[:, 0])),
                np.column_stack((source_products[:, 1], final[:, 1])), Q, labels, n_sites=36, seed=audit_seed^0x34AE67C9)
            np.savez_compressed(output/"terminal_products.npz", rows=rows, proposal=Q,
                initial=first, final=final, source=source_products, groups=labels)
            save_state(output/"checkpoint.pkl", state, completed_updates=accepted_count, metadata=metadata)
            restored, restored_metadata = restore_state(output/"checkpoint.pkl")
            next_a, rng_a = next_seed(state[2], "next_unused_fit_replay")
            next_b, rng_b = next_seed(restored[2], "next_unused_fit_replay")
            if (next_a != next_b or rng_a != rng_b or pytree_sha256(state[:2]) != pytree_sha256(restored[:2])):
                raise ArithmeticError("parameters/optimizer/next RNG restoration failed")
            replay_final = anchored_amplitude(source_products[:2], raw_h(restored[0], rows[:2]), reference_products[:2], scale)
            final_replay_error = float(np.max(abs(replay_final-final[:2]))/np.max(abs(final[:2])))
            if final_replay_error > 2e-11:
                raise ArithmeticError("restored physical full-H replay failed")
            # Complete independent direct physical swaps, no ancestor cache.
            row = rows[0]
            changed=[]
            for a,b in edges:
                x=row.copy(); x[a],x[b]=row[b],row[a]; changed.append(x)
            direct = physical_amplitudes(state[0], np.asarray(changed))
            p0 = physical_amplitudes(state[0], row[None])[0]
            unlike = row[edges[:,0]] != row[edges[:,1]]
            direct_H = np.sum(np.asarray(coupling)*np.where(unlike, -.25, .25))*p0
            direct_H += np.sum(np.asarray(coupling)*np.where(unlike, .5*direct, 0.))
            H_error = float(abs(direct_H-final[0,1])/max(abs(final[0,1]),1e-30))
            if H_error > 2e-10:
                raise ArithmeticError("complete independent direct-H oracle failed")
            spin_rows = []
            for up in np.flatnonzero(row):
                for down in np.flatnonzero(1-row):
                    changed = row.copy()
                    changed[up], changed[down] = row[down], row[up]
                    spin_rows.append(changed)
            spin_values = physical_amplitudes(state[0], np.asarray(spin_rows))
            spin_residual = float(abs(16*p0+spin_values.sum())/np.linalg.norm(np.r_[p0, spin_values]))
            if spin_residual > 2e-9:
                raise ArithmeticError("physical S squared does not give S(S+1)=2")
            final_character_errors = {}
            prepared_final = prepare(state[0])
            final_center = complex(anchored_scalar(prepared_final, center))
            for name, image, expected in operations:
                observed = complex(anchored_scalar(prepared_final, center[np.asarray(image)]))
                final_character_errors[name] = float(abs(observed-expected*final_center)/abs(final_center))
            if max(final_character_errors.values()) > 2e-9:
                raise ArithmeticError("updated physical state lost its K.A1 character")
            report.update(status="completed_two_fixed_full_W_energy_attempts", accepted_updates=accepted_count,
                audit_seed=audit_seed, moments=moments, physical_source_comparison=source_comparison,
                final_source_FS=common_proposal_FS(np.column_stack((source_products[:,0], final[:,0])), Q, groups=8),
                checkpoint_replay=dict(parameter_optimizer_rng_identical=True, next_unused_role_seed=next_a,
                    full_H_relative_error=final_replay_error, direct_full_H_relative_error=H_error),
                total_changed_families=family_norms(jax.tree.map(lambda a,b:a-b,state[0],initial)),
                target_q=2,target_m2=0,spatial_identity=identity, fixed_source_scale=scale,
                physical_spin_squared_relative_residual=spin_residual,
                final_spatial_character_relative_errors=final_character_errors,
                final_audit_used_for_state_selection=False, not_a_converged_sector_energy_or_excitation_gap=True)
    except (TimeoutError, MemoryError, ArithmeticError) as error:
        report.update(status="bounded_partial_or_failed_diagnostic", error=str(error))
        save_state(output/"partial.pkl", state, completed_updates=sum(x["accepted"] for x in report["transitions"]), metadata=metadata)
    report["wall_seconds"] = perf_counter()-started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_RSS_GiB"] = rss/(1024**3 if sys.platform == "darwin" else 1024**2)
    report["old_endpoint_unchanged"] = file_sha256(old_directory/"checkpoint.pkl") == registration["inherited_checkpoint_sha256"]
    atomic_json(output/"results.json", report)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({k:result.get(k) for k in ("status","accepted_updates","wall_seconds","peak_RSS_GiB","error")}), flush=True)
