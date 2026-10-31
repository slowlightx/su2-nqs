"""Ten-update projected 6x6 energy smoke for the strict triangle-patch tree.

Training uses the exact-singlet diagonal ``3 Hz`` force and therefore only
projected center amplitudes.  Full-Hamiltonian local energies are evaluated on
independent audit chains at updates 0, 5, and 10.  The diagonal-estimator
variance is never reported as the physical Hamiltonian variance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import array_sha256, atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import rectangular_translation_group, triangular_hnf_translation_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector, spatial_projected_amplitude
from su2_nqs.models.triangle_patch_clebsch_tree import (
    init_triangle_patch_clebsch_tree,
    triangle_patch_6x6_canary_plan,
    triangle_patch_amplitude,
    triangle_patch_parameter_count,
)
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains, run_exchange_sweeps
from su2_nqs.vmc.physical_sampled_adam_center_only import (
    SampledAdamState,
    initialize_sampled_adam,
    sampled_adam_update,
)
from su2_nqs.vmc.resumable_checkpoint import load_latest_resumable_checkpoint, save_resumable_checkpoint_generation
from su2_nqs.vmc.singlet_diagonal_force import sampled_singlet_diagonal_value_and_grad, three_hz_local_energies
from su2_nqs.vmc.spin_physical import padded_spin_connectivity, spin_physical_hpsi


jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[3]
TASK = "triangle_patch_n36_energy_smoke"
CLAIM_BOUNDARY = (
    "Ten-update single-seed Adam plumbing and early-energy-slope smoke only; "
    "no converged energy, efficiency, phase, QSL, scaling, or PRL win is claimed. "
    "A failure does not kill the ansatz; production comparison requires matrix-free SR."
)


@dataclass(frozen=True)
class _Runtime:
    plan: object
    params: object
    projector: PhysicalSpatialProjector
    amplitude: Callable[[object, jax.Array], jax.Array]


def _validate(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 1 or spec.get("domain") != "spin" or spec.get("task") != TASK:
        raise ValueError("invalid N36 energy-smoke identity")
    if spec.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("claim boundary mismatch")
    model = spec.get("model")
    if not isinstance(model, dict) or model != {
        "family": "strict_triangle_patch_clebsch_tree",
        "feature_dim": 2,
        "maximum_intermediate_q": 10,
        "parameter_count": 1252,
        "initialization_scale": 0.6,
        "route_gate_scale": 0.05,
    }:
        raise ValueError("model must be the registered D2/qmax10/P1252 canary")
    projection = spec.get("projection")
    if not isinstance(projection, dict) or projection != {
        "kind": "translation_k0",
        "group_order": 36,
        "orbit_chunk_size": 6,
        "rematerialize_orbit": True,
        "applied_from_update": 0,
    }:
        raise ValueError("exact 36-translation projection must apply from update zero")
    cases = spec.get("cases")
    expected = {
        "square_6x6_j2_0p5": ("square", 0.5, -0.503810, -0.503765),
        "triangular_6x6_j2_0p125": ("triangular", 0.125, -0.515564, -0.515386),
    }
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError("exactly two cases are required")
    for case in cases:
        if not isinstance(case, dict) or str(case.get("case_id")) not in expected:
            raise ValueError("unregistered case")
        geometry, ratio, exact, raw = expected[str(case["case_id"])]
        if (case.get("geometry"), case.get("rows"), case.get("columns"), case.get("periodic"), case.get("ed")) != (geometry, 6, 6, True, False):
            raise ValueError("case must be a periodic non-ED 6x6 torus")
        if float(case.get("j2_over_j1")) != ratio or float(case.get("exact_energy_per_site")) != exact or float(case.get("raw_literature_energy_per_site")) != raw:
            raise ValueError("case or literature target mismatch")
        if not isinstance(case.get("model_seed"), int) or isinstance(case.get("model_seed"), bool):
            raise TypeError("model_seed must be an integer")
    sampler, optimizer, audit = spec.get("sampler"), spec.get("optimizer"), spec.get("audit")
    if not isinstance(sampler, dict) or (sampler.get("chains"), sampler.get("initial_burn_proposals_per_chain"), sampler.get("proposals_between_updates")) != (16, 288, 72):
        raise ValueError("sampler must use 16 chains, 288 burn, and 72 inter-update proposals")
    if not isinstance(optimizer, dict) or (optimizer.get("updates"), optimizer.get("learning_rate"), optimizer.get("maximum_relative_update_norm"), optimizer.get("checkpoint_every")) != (10, 0.001, 0.05, 5):
        raise ValueError("optimizer must be the registered ten-step Adam smoke")
    if optimizer.get("training_microbatch_size") != 2:
        raise ValueError("training microbatch must be two")
    if not isinstance(audit, dict) or audit.get("updates") != [0, 5, 10] or audit.get("full_h_microbatch_size") != 2:
        raise ValueError("full-H paired audits are required at 0/5/10 with batch two")


def _geometry(case: dict[str, object]):
    geometry = str(case["geometry"])
    ratio = float(case["j2_over_j1"])
    if geometry == "square":
        return square_j1j2_bonds(6, 6, j1=1.0, j2=ratio, periodic=True), rectangular_translation_group(6, 6)
    return triangular_j1j2_bonds(6, 6, j1=1.0, j2=ratio, periodic=True), triangular_hnf_translation_group(((6, 0), (0, 6)))


def _runtime(case: dict[str, object], spec: dict[str, object]) -> _Runtime:
    plan = triangle_patch_6x6_canary_plan(str(case["geometry"]))
    params = init_triangle_patch_clebsch_tree(
        jax.random.PRNGKey(int(case["model_seed"])), plan,
        scale=float(spec["model"]["initialization_scale"]),
        route_gate_scale=float(spec["model"]["route_gate_scale"]),
    )
    if triangle_patch_parameter_count(params) != 1252:
        raise RuntimeError("canary parameter count changed")
    _, group = _geometry(case)
    projector = PhysicalSpatialProjector(
        group, orbit_chunk_size=6, rematerialize_orbit=True
    )
    def amplitude(candidate, state):
        return spatial_projected_amplitude(
            candidate, state, plan, projector, amplitude_fn=triangle_patch_amplitude
        )
    return _Runtime(plan, params, projector, amplitude)


def _kernels(runtime: _Runtime, bonds):
    edges = tuple((i, j) for i, j, _ in bonds)
    def uncached(p, key, states, attempts, probability):
        return run_exchange_sweeps(p, key, states, amplitude_fn=runtime.amplitude, particle_kind="spin", sweeps=attempts, amplitudes=None, exchange_edges=edges, bond_proposal_probability=probability)
    def cached(p, key, states, amplitudes, attempts, probability):
        return run_exchange_sweeps(p, key, states, amplitude_fn=runtime.amplitude, particle_kind="spin", sweeps=attempts, amplitudes=amplitudes, exchange_edges=edges, bond_proposal_probability=probability)
    def centers(p, states):
        return jax.vmap(lambda state: runtime.amplitude(p, state))(states)
    def hpsi(p, batch):
        samples, connected, elements, valid = batch
        return spin_physical_hpsi(p, samples, connected, elements, valid, runtime.plan, amplitude_fn=lambda q, state, _plan: runtime.amplitude(q, state))
    return (
        jax.jit(uncached, static_argnums=(3, 4)),
        jax.jit(cached, static_argnums=(4, 5)),
        jax.jit(centers),
        jax.jit(hpsi),
    )


def _float_row(row):
    return {name: float(value) for name, value in row.items()}


def _audit_fold(case_id: str, update: int, seed: int) -> int:
    digest = hashlib.sha256(f"{case_id}:{update}:{seed}:full_h_audit".encode()).hexdigest()
    return int(digest[:8], 16)


def _audit(params, training_states, *, case, bonds, spec, uncached, cached, hpsi):
    audit = spec["audit"]
    update = int(audit["current_update"])
    key = jax.random.fold_in(jax.random.PRNGKey(int(audit["seed"])), _audit_fold(str(case["case_id"]), update, int(audit["seed"])))
    states = jnp.asarray(np.array(training_states, copy=True))
    states, amplitudes, key, burn = uncached(params, key, states, int(audit["burn_proposals_per_chain"]), float(spec["sampler"]["bond_proposal_probability"]))
    jax.block_until_ready((states, amplitudes, key, burn))
    sampler_rows = [_float_row(burn)]
    full_rows, diagonal_rows = [], []
    padded_width = len(bonds) + 1
    for _ in range(int(audit["snapshots"])):
        states, amplitudes, key, row = cached(params, key, states, amplitudes, int(audit["proposals_between_snapshots"]), float(spec["sampler"]["bond_proposal_probability"]))
        jax.block_until_ready((states, amplitudes, key, row))
        sampler_rows.append(_float_row(row))
        host = np.asarray(states, dtype=np.int32)
        connected, elements, valid, _ = padded_spin_connectivity(host, bonds, padded_width=padded_width)
        full_chunks = []
        for start in range(0, len(host), 2):
            batch = tuple(jnp.asarray(x[start:start + 2]) for x in (host, connected, elements, valid))
            centers, actions = hpsi(params, batch)
            jax.block_until_ready((centers, actions))
            full_chunks.append(np.asarray(actions) / np.asarray(centers))
        full_rows.append(np.concatenate(full_chunks))
        diagonal_rows.append(np.asarray(three_hz_local_energies(host, bonds)))
    full = np.stack(full_rows); diagonal = np.stack(diagonal_rows)
    if np.any(~np.isfinite(full)) or np.any(~np.isfinite(diagonal)):
        raise FloatingPointError("nonfinite paired audit")
    n_sites = 36
    full_mean = complex(np.mean(full)); diagonal_mean = float(np.mean(diagonal))
    full_chain = np.mean(full.real, axis=0) / n_sites
    diagonal_chain = np.mean(diagonal, axis=0) / n_sites
    paired_chain = full_chain - diagonal_chain
    variance = float(np.mean(np.abs(full - full_mean) ** 2).real / n_sites)
    variance_chain = np.mean(np.abs(full - full_mean) ** 2, axis=0).real / n_sites
    chains = len(full_chain)
    result = {
        "update": update,
        "params_sha256": pytree_sha256(params),
        "independent_audit_stream": True,
        "training_states_copied_not_mutated": True,
        "same_samples_used_for_full_h_and_three_hz": True,
        "sample_count": int(full.size),
        "chain_count": chains,
        "snapshots_per_chain": int(full.shape[0]),
        "samples_are_temporally_autocorrelated": True,
        "mcse_is_across_chain_time_averages": True,
        "energy_per_site": float(full_mean.real / n_sites),
        "energy_per_site_chain_mcse": float(np.std(full_chain, ddof=1) / np.sqrt(chains)),
        "variance_h_per_site": variance,
        "variance_h_per_site_chain_mcse": float(np.std(variance_chain, ddof=1) / np.sqrt(chains)),
        "energy_imaginary_abs": float(abs(full_mean.imag)),
        "three_hz_energy_per_site": diagonal_mean / n_sites,
        "three_hz_sample_variance_per_site_estimator_diagnostic_only": float(np.var(diagonal) / n_sites),
        "three_hz_variance_is_not_physical_h_variance": True,
        "full_h_minus_three_hz_per_site": float(np.mean(paired_chain)),
        "full_h_minus_three_hz_chain_mcse": float(np.std(paired_chain, ddof=1) / np.sqrt(chains)),
        "gap_to_exact_energy_per_site": float(full_mean.real / n_sites - float(case["exact_energy_per_site"])),
        "gap_to_raw_literature_energy_per_site": float(full_mean.real / n_sites - float(case["raw_literature_energy_per_site"])),
        "sampler_accepted_move_fraction_mean": float(np.mean([x["accepted_move_fraction"] for x in sampler_rows])),
        "minimum_center_magnitude": float(np.min(np.abs(amplitudes))),
    }
    work = {
        "audit_sampler_proposals": chains * (int(audit["burn_proposals_per_chain"]) + int(audit["snapshots"]) * int(audit["proposals_between_snapshots"])),
        "audit_full_h_projected_amplitudes": int(full.size) * (1 + padded_width),
        "audit_base_amplitude_orbit_images_proxy": (chains * (1 + int(audit["burn_proposals_per_chain"]) + int(audit["snapshots"]) * int(audit["proposals_between_snapshots"])) + int(full.size) * (1 + padded_width)) * 36,
    }
    return result, work


def _checkpoint_metadata(states, key, audits, initial, work, resume_count):
    host_states = np.asarray(states, dtype=np.int32); host_key = np.asarray(key, dtype=np.uint32)
    return {
        "chain_states": host_states,
        "chain_states_sha256": array_sha256(host_states),
        "sampling_key": host_key,
        "sampling_key_sha256": array_sha256(host_key),
        "audits": audits,
        "initial": initial,
        "work": work,
        "resume_count": resume_count,
        "amplitude_cache_present": False,
    }


def _run_case(case, spec, output, *, overwrite):
    runtime = _runtime(case, spec); bonds, group = _geometry(case)
    uncached, cached, centers, hpsi = _kernels(runtime, bonds)
    config_hash = canonical_sha256(spec)
    initial_key = jax.random.PRNGKey(int(case["sampler_seed"]))
    chain_key, sampling_key = jax.random.split(initial_key)
    initial_states = initialize_spin_exchange_chains(chain_key, n_chains=16, n_sites=36, n_up=18)
    initial_centers = centers(runtime.params, initial_states); jax.block_until_ready(initial_centers)
    if bool(np.any(np.abs(np.asarray(initial_centers)) == 0.0)):
        raise FloatingPointError("projected initialization contains a zero center")
    initial = {
        "params_sha256": pytree_sha256(runtime.params),
        "chain_states_sha256": array_sha256(np.asarray(initial_states)),
        "projected_center_magnitude_minimum": float(np.min(np.abs(np.asarray(initial_centers)))),
        "projected_center_magnitude_maximum": float(np.max(np.abs(np.asarray(initial_centers)))),
    }
    identity = {
        "task": TASK, "config_sha256": config_hash, "case_id": case["case_id"],
        "bonds_sha256": canonical_sha256(bonds), "group_sha256": canonical_sha256(group),
        "initial": initial, "force": "exact_singlet_3hz_center_only_v1",
    }
    checkpoint_dir = output / "checkpoints" / str(case["case_id"])
    if overwrite and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint = None if overwrite else (load_latest_resumable_checkpoint(checkpoint_dir, expected_run_identity=identity) if checkpoint_dir.exists() else None)
    started = time.perf_counter()
    if checkpoint is None:
        params = runtime.params; optimizer_state = initialize_sampled_adam(params)
        states, amplitudes, sampling_key, burn = uncached(params, sampling_key, initial_states, 288, float(spec["sampler"]["bond_proposal_probability"]))
        jax.block_until_ready((states, amplitudes, sampling_key, burn))
        if float(burn["zero_amplitude_chain_fraction"]) != 0.0:
            raise FloatingPointError("post-burn projected chain has zero amplitude")
        del amplitudes
        audit_spec = dict(spec); audit_spec["audit"] = {**spec["audit"], "current_update": 0}
        audit0, audit_work = _audit(params, np.asarray(states), case=case, bonds=bonds, spec=audit_spec, uncached=uncached, cached=cached, hpsi=hpsi)
        audits = [audit0]; history = []; completed = 0; resume_count = 0
        work = {"initial_burn_sampler_proposals": 16 * 288, **audit_work}
        save_resumable_checkpoint_generation(checkpoint_dir, params, optimizer_state=optimizer_state, completed_steps=0, next_sample_index=0, run_identity=identity, history=history, metadata=_checkpoint_metadata(states, sampling_key, audits, initial, work, resume_count))
    else:
        params = checkpoint["params"]; optimizer_state = checkpoint["optimizer_state"]
        if not isinstance(optimizer_state, SampledAdamState):
            raise RuntimeError("checkpoint optimizer type mismatch")
        completed = int(checkpoint["completed_steps"]); history = list(checkpoint["history"])
        metadata = checkpoint["metadata"]; states = jnp.asarray(metadata["chain_states"]); sampling_key = jnp.asarray(metadata["sampling_key"])
        audits = list(metadata["audits"]); initial = dict(metadata["initial"]); work = dict(metadata["work"]); resume_count = int(metadata["resume_count"]) + 1
    hard_failure = None
    for update in range(completed + 1, 11):
        before = (params, optimizer_state, np.asarray(states), np.asarray(sampling_key))
        try:
            states, amplitudes, sampling_key, sampler_row = uncached(params, sampling_key, states, 72, float(spec["sampler"]["bond_proposal_probability"]))
            jax.block_until_ready((states, amplitudes, sampling_key, sampler_row))
            if float(sampler_row["zero_amplitude_chain_fraction"]) != 0.0:
                raise FloatingPointError("projected training chain reached zero amplitude")
            del amplitudes
            batches = tuple(jnp.asarray(states[start:start + 2]) for start in range(0, 16, 2))
            gradient = sampled_singlet_diagonal_value_and_grad(params, batches, center_fn=centers, bonds=bonds)
            params, optimizer_state, adam = sampled_adam_update(
                params, gradient.gradient, optimizer_state,
                learning_rate=0.001, max_relative_update=0.05,
                beta1=0.9, beta2=0.999, epsilon=1e-8,
            )
            jax.block_until_ready(params)
            audit_row = None
            if update in (5, 10):
                audit_spec = dict(spec); audit_spec["audit"] = {**spec["audit"], "current_update": update}
                audit_row, audit_work = _audit(params, np.asarray(states), case=case, bonds=bonds, spec=audit_spec, uncached=uncached, cached=cached, hpsi=hpsi)
                audits.append(audit_row)
                for name, value in audit_work.items(): work[name] = work.get(name, 0) + value
            row = {
                "update": update,
                "sampler": _float_row(sampler_row),
                "three_hz_energy_per_site_training_estimator": gradient.energy / 36,
                "three_hz_sample_variance_per_site_estimator_diagnostic_only": gradient.variance / 36,
                "three_hz_variance_is_not_physical_h_variance": True,
                "gradient_norm": gradient.gradient_norm,
                "minimum_center_magnitude": gradient.minimum_center_magnitude,
                "adam": asdict(adam),
                "full_h_audit": audit_row,
            }
            history.append(row); completed = update
            work["training_sampler_proposals"] = work.get("training_sampler_proposals", 0) + 16 * 72
            work["training_projected_center_evaluations_proxy"] = work.get("training_projected_center_evaluations_proxy", 0) + 2 * 16
            if update % 5 == 0:
                save_resumable_checkpoint_generation(checkpoint_dir, params, optimizer_state=optimizer_state, completed_steps=completed, next_sample_index=completed * 16, run_identity=identity, history=history, metadata=_checkpoint_metadata(states, sampling_key, audits, initial, work, resume_count))
        except (FloatingPointError, ValueError) as error:
            params, optimizer_state, old_states, old_key = before; states = jnp.asarray(old_states); sampling_key = jnp.asarray(old_key)
            hard_failure = {"attempted_update": update, "type": type(error).__name__, "message": str(error)}
            save_resumable_checkpoint_generation(checkpoint_dir, params, optimizer_state=optimizer_state, completed_steps=completed, next_sample_index=completed * 16, run_identity=identity, history=history, metadata=_checkpoint_metadata(states, sampling_key, audits, initial, work, resume_count))
            break
    final = audits[-1]
    return {
        "case_id": case["case_id"], "geometry": case["geometry"],
        "status": "complete" if completed == 10 and hard_failure is None else "hard_failed",
        "completed_updates": completed, "requested_updates": 10,
        "parameter_count": 1252, "translation_group_order": runtime.projector.group_order,
        "resume_count": resume_count, "initial": initial, "history": history,
        "full_h_paired_audits": audits, "final_full_h_energy_per_site": final["energy_per_site"],
        "final_gap_to_exact_energy_per_site": final["gap_to_exact_energy_per_site"],
        "final_gap_to_raw_literature_energy_per_site": final["gap_to_raw_literature_energy_per_site"],
        "work_ledger": work, "hard_failure": hard_failure,
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "checkpoint_directory": str(checkpoint_dir.relative_to(ROOT)),
    }


def run_triangle_patch_n36_energy_smoke(config_path: Path, output: Path, *, overwrite: bool = False):
    spec = json.loads(config_path.read_text()); _validate(spec)
    output.mkdir(parents=True, exist_ok=True); result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result exists; pass --overwrite: {result_path}")
    started = time.perf_counter()
    cases = [_run_case(case, spec, output, overwrite=overwrite) for case in spec["cases"]]
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": 1, "domain": "spin", "task": TASK,
        "status": "complete" if all(case["status"] == "complete" for case in cases) else "hard_failed",
        "completed_utc": now, "config": spec, "config_sha256": canonical_sha256(spec),
        "runtime_identity": numerical_runtime_identity(), "cases": cases,
        "total_wall_seconds": time.perf_counter() - started,
        "hamiltonian_diagonalization_performed": False,
        "translation_projection_applied_from_update_zero": True,
        "training_force": "exact_singlet_3hz_center_only",
        "physical_energy_and_variance_source": "independent_full_h_paired_audits_only",
        "promotion_decision": "runtime_and_early_slope_smoke_only",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["result_sha256"] = canonical_sha256(payload); atomic_json(result_path, payload)
    return payload


__all__ = ["CLAIM_BOUNDARY", "TASK", "run_triangle_patch_n36_energy_smoke"]
