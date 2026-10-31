"""Bounded full-network training of the unchanged native singlet Schur ViT.

This is a development experiment, NOT an exact SU(2)+A1 production model.
Native autoregressive draws are independent samples from the normalized S=0
Born law. The full lattice projector and its different Born law are absent.
All real parameters, including phase parameters, receive the two-sided complex
covariance SR force. No Hamiltonian columns, q support, or gradients are clipped.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_reduced_vit_viability import (
    _plain, measured, model_config, neural_bonds,
)
from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import (
    file_sha256, numerical_runtime_identity, transitive_local_source_manifest,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.many_cell_schur_vit import (
    many_cell_schur_log_amplitudes, many_cell_schur_vit_parameter_count,
    sample_many_cell_schur_vit,
)
from su2_nqs.models.schur_initialization import deepen_many_cell_schur_vit
from su2_nqs.su2.many_cell_schur import validate_layout_state
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.dual_sampled_sr import (
    make_dual_sampled_sr_update, sampled_overlap_diagnostics,
)
from su2_nqs.vmc.many_cell_tree import (
    compile_many_cell_device_hamiltonian, evaluate_many_cell_device_local_energy,
)

ROOT = Path(__file__).resolve().parents[3]
MODULE = "projects.clebsch_tree_nqs.workflows.spin_reduced_vit_train"


def local_energy_statistics(local_energies: np.ndarray, n_sites: int) -> dict:
    """IID estimates; variance includes imaginary local-energy fluctuations.

    ``Var(H)`` is estimated by the unbiased complex sample variance, not just
    the variance of Re(E_local). Its plug-in standard error is diagnostic at
    finite N; no Gaussian tail guarantee is implied. Native AR draws are iid,
    so these are not MCMC autocorrelation-adjusted estimates.
    """
    values = np.asarray(local_energies, dtype=np.complex128)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("need at least two finite complex local energies")
    if isinstance(n_sites, bool) or not isinstance(n_sites, int) or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    mean = values.mean()
    squared = np.abs(values-mean)**2
    variance = float(squared.sum()/(len(values)-1))
    return dict(sample_count=len(values), energy_per_site=float(mean.real/n_sites),
        energy_stderr_per_site=float(values.real.std(ddof=1)/np.sqrt(len(values))/n_sites),
        imaginary_energy_per_site=float(mean.imag/n_sites),
        imaginary_energy_stderr_per_site=float(values.imag.std(ddof=1)/np.sqrt(len(values))/n_sites),
        variance_per_site=variance/n_sites,
        variance_stderr_plugin_per_site=float(squared.std(ddof=1)/np.sqrt(len(values))/n_sites),
        variance_definition="sum |E_local-mean_complex(E_local)|^2/(N-1), divided by sites",
        samples_are_native_iid=True, full_A1=False)


def directional_energy_statistics(log_tangent: np.ndarray, local_energies: np.ndarray,
                                  n_sites: int) -> dict:
    """IID two-sided energy slope, split into magnitude and phase terms.

    ``log_tangent`` is d log(psi)/dt along an already fixed parameter update.
    This uses an independent old-state Born draw, not reweighted endpoint
    samples. Centered covariance has the N/(N-1) correction; its plug-in
    standard error is descriptive, especially for heavy local-energy tails.
    The slope is diagnostic only and never selects an update.
    """
    tangent = np.asarray(log_tangent, dtype=np.complex128)
    energy = np.asarray(local_energies, dtype=np.complex128)
    if tangent.ndim != 1 or tangent.shape != energy.shape or len(energy) < 2:
        raise ValueError("tangent and local energies require equal one-dimensional IID batches")
    if not np.isfinite(tangent).all() or not np.isfinite(energy).all() or n_sites < 1:
        raise ValueError("finite tangent/energies and positive site count required")
    tangent = tangent - tangent.mean()
    energy = energy - energy.mean()
    correction = len(energy)/(len(energy)-1)
    magnitude = 2*correction*tangent.real*energy.real/n_sites
    phase = 2*correction*tangent.imag*energy.imag/n_sites
    total = magnitude+phase
    return dict(sample_count=len(energy), energy_slope_per_site=float(total.mean()),
        energy_slope_stderr_plugin_per_site=float(total.std(ddof=1)/np.sqrt(len(total))),
        magnitude_slope_per_site=float(magnitude.mean()), phase_slope_per_site=float(phase.mean()),
        tangent_metric=float(np.mean(abs(tangent)**2)*correction),
        both_sided_complex_covariance=True, used_for_update_selection=False)


def native_spin_occupations(labels: np.ndarray, config) -> dict:
    """Native Born probabilities of doubled cell/inter-cell q; no RDM claim."""
    decoded = config.layout.decode(np.asarray(labels, dtype=np.int32))
    cell = np.asarray(decoded.cell_q)
    inter = np.asarray(decoded.intercell_q)
    per_draw_singlet = np.mean(cell == 0, axis=1)
    return dict(samples=len(cell), mean_cell_singlet_weight=float(per_draw_singlet.mean()),
        mean_cell_singlet_stderr=float(per_draw_singlet.std(ddof=1)/np.sqrt(len(cell))),
        cell_q_probabilities={str(q):np.mean(cell == q, axis=0).tolist() for q in (0,2,4)},
        intercell_q_nonzero_probabilities=np.mean(inter > 0, axis=0).tolist(),
        doubled_spin_labels=True, native_Born=True, full_A1=False)


def calibrated_sr_update(
    params: object, samples: jax.Array, local_energies: jax.Array,
    calibration_labels: jax.Array, sr_update: Callable, log_batch: Callable,
    settings: dict,
) -> tuple[object, dict]:
    """Backtrack one full SR direction using a separate old-state Born draw.

    Training and calibration sets both probe physical overlap; calibration
    never queries H, energies, or the untouched terminal audit stream. This
    cannot certify a global FS bound, but rules out parameter-norm matching.
    Both-sided complex score force and independent linear residual checks are
    supplied by the unchanged generic SR implementation.
    """
    candidate, raw = sr_update(params, samples, local_energies)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), candidate)
    diagnostics = _plain(jax.device_get(raw))
    old_train = log_batch(params, samples)
    old_calibration = log_batch(params, calibration_labels)
    chosen = params
    accepted = False
    train_diag = dict(sampled_fs_sine=0.0, reweighting_ess_fraction=1.0)
    heldout_diag = dict(sampled_fs_sine=0.0, reweighting_ess_fraction=1.0)
    applied_fraction = 0.0
    attempts = 0
    if bool(diagnostics["accepted"]):
        for index in range(settings["geometry_backtracking_steps"]):
            attempts += 1
            fraction = 0.5**index
            trial = jax.tree_util.tree_map(lambda p,c: p+fraction*(c-p), params, candidate)
            train_diag = _plain(jax.device_get(sampled_overlap_diagnostics(
                log_batch(trial, samples)-old_train)))
            heldout_diag = _plain(jax.device_get(sampled_overlap_diagnostics(
                log_batch(trial, calibration_labels)-old_calibration)))
            passed = all(np.isfinite(float(value)) for d in (train_diag,heldout_diag) for value in d.values())
            passed = passed and all(
                d["sampled_fs_sine"] <= settings["maximum_overlap_distance"]
                and d["reweighting_ess_fraction"] >= settings["minimum_ess_fraction"]
                for d in (train_diag,heldout_diag))
            if passed:
                chosen, accepted, applied_fraction = trial, True, fraction
                break
    diagnostics.update(accepted=accepted, independent_geometry_accepted=accepted,
        raw_parameter_step_norm=float(diagnostics["parameter_step_norm"]),
        final_parameter_step_norm=float(np.sqrt(sum(float(np.sum(np.asarray(c-p)**2))
            for c,p in zip(jax.tree_util.tree_leaves(chosen),jax.tree_util.tree_leaves(params),strict=True)))),
        last_trial_training_overlap=train_diag, last_trial_geometry_overlap=heldout_diag,
        training_overlap=train_diag if accepted else dict(sampled_fs_sine=0.0,reweighting_ess_fraction=1.0),
        independent_geometry_overlap=heldout_diag if accepted else dict(sampled_fs_sine=0.0,reweighting_ess_fraction=1.0),
        independent_geometry_backtracking_evaluations=attempts,
        extra_applied_fraction=applied_fraction,
        final_applied_scale=float(diagnostics["applied_scale"])*applied_fraction,
        final_fisher_norm=float(diagnostics["fisher_norm"])*applied_fraction,
        final_directional_derivative=float(diagnostics["directional_derivative"])*applied_fraction,
        geometry_is_global_certificate=False)
    return chosen, diagnostics


def save_training_checkpoint(path: Path, params: object, optimizer_state: dict,
                             rng: dict[str, jax.Array], metadata: dict) -> None:
    """Save trained parameters, explicit stateless-SR state, and both RNGs."""
    save_checkpoint(path, jax.tree_util.tree_map(np.asarray, params),
        optimizer_state=optimizer_state,
        metadata={**metadata, "rng": {name: np.asarray(key) for name,key in rng.items()},
                  "exact_SU2": True, "full_A1": False,
                  "native_normalized_Born": True})


def _validate(settings: dict) -> None:
    if settings["lx"] != 6 or settings["ly"] != 6 or settings["j2"] != 0.125:
        raise ValueError("this registration is restricted to triangular N36 J2=.125")
    for key in ("samples_per_step", "geometry_samples", "steps", "audit_samples",
                "hamiltonian_chunk_size", "total_seconds", "training_seconds"):
        if isinstance(settings[key], bool) or not isinstance(settings[key],int) or settings[key]<1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["total_seconds"] > 3600 or settings["training_seconds"] >= settings["total_seconds"]:
        raise ValueError("hard cap is one hour with a reserved audit window")
    if len({settings[key] for key in ("training_seed", "geometry_seed", "audit_seed")}) != 3:
        raise ValueError("training, geometry, and audit RNG seeds must be distinct")
    extra_seeds = [settings[k] for k in ("direction_seed", "parent_audit_seed") if k in settings]
    all_seeds = [settings[k] for k in ("training_seed", "geometry_seed", "audit_seed")]+extra_seeds
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("all diagnostic and audit RNG seeds must be distinct")
    if settings.get("direction_check_steps"):
        if "direction_seed" not in settings or settings.get("direction_samples",0) < 2:
            raise ValueError("direction checks require their own RNG and IID batch")
        if any(step < 1 or step > settings["steps"] for step in settings["direction_check_steps"]):
            raise ValueError("direction check steps must lie inside fixed training budget")
    caps = settings["exact_connected_capacities"]
    if not caps or any(isinstance(x,bool) or not isinstance(x,int) or x<1 for x in caps):
        raise ValueError("need positive exact kernel capacities")
    if sorted(set(caps)) != caps:
        raise ValueError("exact fallback capacities must increase without repeats")
    if settings.get("full_A1", False):
        raise ValueError("full A1 is missing; native singlet cannot advertise it")
    if not isinstance(settings.get('bucket_padded_amplitudes',False),bool):
        raise ValueError('bucket_padded_amplitudes must be boolean')
    if settings.get("platform", "cpu") not in ("cpu", "gpu"):
        raise ValueError("platform must be explicitly cpu or gpu")


def initialize_from_parent(settings: dict, loaded: dict) -> tuple[object, dict]:
    """Restore the parent, optionally appending state-preserving ViT blocks.

    Architecture growth has an explicit fresh stateless-SR registration;
    it is not an optimizer resume across different parameter spaces.
    """
    source_settings = loaded["metadata"].get("config", loaded["metadata"].get("model_config"))
    if source_settings is None:
        raise ValueError("parent lacks declared model configuration")
    if source_settings.get('conditional_mode','legacy')!=settings.get('conditional_mode','legacy'):
        raise ValueError('conditional mode changed; explicit new initialization required, not a resume')
    expansion = settings.get("capacity_expansion")
    for name in ("lx", "ly", "embedding_dim", "head_count", "depth", "direct_phase_initialization_scale"):
        expected = expansion["source_depth"] if expansion and name == "depth" else settings[name]
        if source_settings[name] != expected:
            raise ValueError("model differs from the frozen parent: " + name)
    params = jax.tree_util.tree_map(jnp.asarray, loaded["params"])
    if expansion is None:
        return params, dict(capacity_changed=False)
    source = model_config({**settings, "depth": expansion["source_depth"]})
    target = model_config(settings)
    parent_hash = pytree_sha256(params)
    if parent_hash != expansion["parent_params_sha256"]:
        raise ValueError("capacity expansion parent parameters differ from registration")
    grown = deepen_many_cell_schur_vit(params,source,target,jax.random.PRNGKey(expansion["seed"]))
    # A registered initialization-only draw does not touch training or audit
    # RNGs and performs no H queries. Exhaustive proof lives in exact tests.
    labels = sample_many_cell_schur_vit(params,jax.random.PRNGKey(expansion["probe_seed"]),16,source)
    before = np.asarray(many_cell_schur_log_amplitudes(params,labels,source))
    after = np.asarray(many_cell_schur_log_amplitudes(grown,labels,target))
    error = float(np.max(np.abs(after-before)))
    if not np.isfinite(error) or error > 2e-12:
        raise AssertionError("capacity expansion changed the physical parent state")
    return grown, dict(capacity_changed=True,mechanism="identity residual-block depth growth",
        source_depth=source.depth,target_depth=target.depth,
        source_parameters=many_cell_schur_vit_parameter_count(params),
        target_parameters=many_cell_schur_vit_parameter_count(grown),
        source_params_sha256=parent_hash,probe_log_amplitude_max_error=error,
        probe_native_draws=16,probe_H_calls=0,optimizer_reset_for_new_parameter_space=True,
        physical_parent_preserved=True,full_A1=False)


class _Hamiltonian:
    """Full H chunks, with registered exact capacity escalation, never pruning."""
    def __init__(self, config, settings, ledger, started):
        self.config, self.settings, self.ledger, self.started = config,settings,ledger,started
        self.bonds = neural_bonds(triangular_j1j2_bonds(6,6,j2=settings["j2"],periodic=True),6,6)
        self.capacity_index = 0
        self.kernel = self._compile()

    def _compile(self):
        capacity = self.settings["exact_connected_capacities"][self.capacity_index]
        kernel = compile_many_cell_device_hamiltonian(self.config,self.bonds,candidate_capacity=None,
            connected_capacity=capacity,amplitude_tile_size=self.settings["amplitude_tile_size"],
            bucket_padded_amplitudes=self.settings.get('bucket_padded_amplitudes',False))
        self.ledger["kernel_compilations"].append(dict(connected_capacity=capacity,
            table_seconds=kernel.compilation_seconds, raw_slots_per_sample=kernel.raw_transition_slots_per_sample))
        return kernel

    def evaluate(self, params, labels, *, stage):
        labels = np.asarray(labels,dtype=np.int32)
        chunk = self.settings["hamiltonian_chunk_size"]
        local, widths = [], []
        evaluated_amplitudes=0
        wall = 0.0
        for offset in range(0,len(labels),chunk):
            if time.perf_counter()-self.started >= self.settings["total_seconds"]-3:
                raise TimeoutError("registered total compute budget; no partial audit presented as complete")
            batch = labels[offset:offset+chunk]
            padded = np.concatenate((batch,np.tile(batch[-1],(chunk-len(batch),1))))
            while True:
                record = evaluate_many_cell_device_local_energy(self.kernel,params,padded,fail_on_overflow=False)
                self.ledger["full_H_calls"] += len(padded)
                self.ledger["full_H_bond_calls"] += record.full_h_term_calls
                self.ledger["stage_H_calls"][stage] = self.ledger["stage_H_calls"].get(stage,0)+len(padded)
                wall += record.evaluation_seconds
                evaluated_amplitudes+=record.evaluated_connected_amplitudes
                if not np.any(record.overflow):
                    break
                self.ledger["exact_capacity_escalations"].append(dict(stage=stage,
                    old_capacity=self.kernel.connected_capacity, rejected_batch_offset=offset,
                    overflow_samples=np.flatnonzero(record.overflow).tolist(),
                    reason="incomplete H rejected; repeat with larger pre-registered exact capacity"))
                self.capacity_index += 1
                if self.capacity_index >= len(self.settings["exact_connected_capacities"]):
                    raise RuntimeError("all registered exact connected capacities overflowed; no truncated ratio used")
                self.kernel = self._compile()
            if not np.isfinite(record.local_energies).all():
                raise FloatingPointError("non-finite full-H ratio: no clipping")
            local.extend(record.local_energies[:len(batch)])
            widths.extend(record.unique_widths[:len(batch)])
        return np.asarray(local), dict(full_H_seconds=wall,unique_width_max=int(max(widths)),
            unique_width_mean=float(np.mean(widths)),used_connected_capacity=self.kernel.connected_capacity,
            evaluated_connected_amplitudes=evaluated_amplitudes,
            bucketed_amplitude_dispatch=self.settings.get('bucket_padded_amplitudes',False))


def _sr_settings(settings: dict) -> dict:
    return {name: settings[name] for name in ("damping", "step_size", "fisher_trust_radius",
        "maximum_overlap_distance", "minimum_ess_fraction", "backtracking_steps", "score_chunk_size", "linear_solver")}


def _train(settings: dict, output: Path) -> dict:
    started = time.perf_counter()
    config = model_config(settings)
    parent = ROOT/settings["parent_checkpoint"]
    if file_sha256(parent) != settings["parent_checkpoint_sha256"]:
        raise ValueError("registered initial checkpoint hash mismatch")
    loaded = load_checkpoint(parent)
    if jax.default_backend() != settings.get("platform", "cpu"):
        raise RuntimeError("requested accelerator unavailable; refusing silent platform fallback")
    params, initialization = initialize_from_parent(settings,loaded)
    atomic_json(output/"initialization.json",initialization)
    initial_hash = pytree_sha256(params)
    if many_cell_schur_vit_parameter_count(params) != settings["expected_parameters"]:
        raise ValueError("parameter count changed")
    rng = dict(training=jax.random.PRNGKey(settings["training_seed"]),
               geometry=jax.random.PRNGKey(settings["geometry_seed"]))
    if settings.get("direction_check_steps"):
        rng["direction"] = jax.random.PRNGKey(settings["direction_seed"])
    sr_config = _sr_settings(settings)
    optimizer = dict(algorithm="full-network sample-space ridge SR; no momentum or hidden adaptive state",
        step=0, accepted_steps=0, sr_settings=sr_config,
        geometry_settings={key:settings[key] for key in ("geometry_backtracking_steps",
            "maximum_overlap_distance", "minimum_ess_fraction")})
    history = []
    metadata = dict(model_config={key:settings[key] for key in ("lx","ly","embedding_dim","head_count",
        "depth","direct_phase_initialization_scale")},parent_checkpoint_sha256=settings["parent_checkpoint_sha256"],
        initialization_params_sha256=initial_hash,initialization=initialization,history=history,training=True)
    metadata['model_config']['conditional_mode']=settings.get('conditional_mode','legacy')
    draw = jax.jit(lambda p,k: sample_many_cell_schur_vit(p,k,settings["samples_per_step"],config))
    geometry_draw = jax.jit(lambda p,k: sample_many_cell_schur_vit(p,k,settings["geometry_samples"],config))
    log_batch = jax.jit(lambda p,x: many_cell_schur_log_amplitudes(p,x,config))
    direction_draw = None
    if settings.get("direction_check_steps"):
        direction_draw = jax.jit(lambda p,k: sample_many_cell_schur_vit(
            p,k,settings["direction_samples"],config))
    log_jvp = jax.jit(lambda p,d,x: jax.jvp(lambda pp: log_batch(pp,x),(p,),(d,))[1])
    sr_update = make_dual_sampled_sr_update(lambda p,x: many_cell_schur_log_amplitudes(p,x[None,:],config)[0],
                                          params,**sr_config)
    ledger = dict(full_H_calls=0,full_H_bond_calls=0,stage_H_calls={},kernel_compilations=[],
                  exact_capacity_escalations=[],training_draws=0,geometry_draws=0,audit_draws=0,
                  checkpoint_replay_training_draws=0,checkpoint_replay_geometry_draws=0,
                  direction_diagnostic_draws=0,parent_audit_draws=0)
    hamiltonian = _Hamiltonian(config,settings,ledger,started)
    checkpoint = output/"trained_checkpoint.pkl"
    save_training_checkpoint(checkpoint,params,optimizer,rng,metadata)
    save_training_checkpoint(output/"expanded_initial_checkpoint.pkl",params,optimizer,rng,metadata)
    training_stopped_reason = None
    for step in range(settings["steps"]):
        if time.perf_counter()-started >= settings["training_seconds"]:
            training_stopped_reason = "reserved untouched-audit window, no extension of registered training budget"
            break
        before = time.perf_counter()
        next_training,train_key = jax.random.split(rng["training"])
        next_geometry,geometry_key = jax.random.split(rng["geometry"])
        samples,sample_seconds = measured(lambda: draw(params,train_key))
        calibration,geometry_seconds = measured(lambda: geometry_draw(params,geometry_key))
        ledger["training_draws"] += len(samples)
        ledger["geometry_draws"] += len(calibration)
        for row in np.asarray(samples):
            validate_layout_state(config.layout,row,target_q=0)
        energies,hdiag = hamiltonian.evaluate(params,samples,stage="training")
        (candidate,diagnostics),update_seconds = measured(lambda: calibrated_sr_update(
            params,samples,jnp.asarray(energies),calibration,sr_update,log_batch,settings))
        if float(diagnostics["relative_linear_residual"]) > 1e-7:
            raise RuntimeError("full SR solve residual exceeded unchanged 1e-7 guard")
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(candidate)):
            raise FloatingPointError("non-finite full-network candidate")
        direction_diagnostics = None
        if step+1 in settings.get("direction_check_steps",[]):
            next_direction, direction_key = jax.random.split(rng["direction"])
            direction_labels = direction_draw(params,direction_key)
            ledger["direction_diagnostic_draws"] += len(direction_labels)
            direction_local,direction_h = hamiltonian.evaluate(params,direction_labels,stage="direction_diagnostic")
            direction = jax.tree_util.tree_map(lambda new,old: new-old,candidate,params)
            tangent = log_jvp(params,direction,direction_labels)
            direction_diagnostics = dict(**directional_energy_statistics(tangent,direction_local,config.n),
                full_H=direction_h, independent_of_training_and_geometry=True,
                diagnostic_only=True, native_spin=native_spin_occupations(direction_labels,config))
            rng["direction"] = next_direction
        params = candidate
        rng = {**rng,"training":next_training,"geometry":next_geometry}
        optimizer["step"] = step+1
        optimizer["accepted_steps"] += int(diagnostics["accepted"])
        row = dict(step=step+1, energy_before_update=local_energy_statistics(energies,config.n),
            sr=diagnostics,samples_seconds=sample_seconds,geometry_samples_seconds=geometry_seconds,
            update_seconds=update_seconds,full_H=hdiag,step_seconds=time.perf_counter()-before)
        if direction_diagnostics is not None:
            row["independent_direction"] = direction_diagnostics
        history.append(row)
        _,checkpoint_seconds = measured(lambda: save_training_checkpoint(checkpoint,params,optimizer,rng,metadata))
        row["checkpoint_write_seconds"] = checkpoint_seconds
        # The actual subsequent training step continues from serialized state,
        # not just an in-memory imitation of checkpoint recovery.
        resumed_state = load_checkpoint(checkpoint)
        params = jax.tree_util.tree_map(jnp.asarray,resumed_state["params"])
        optimizer = resumed_state["optimizer_state"]
        rng = {name:jnp.asarray(key) for name,key in resumed_state["metadata"]["rng"].items()}
        row["next_training_step_uses_restored_checkpoint"] = True
        atomic_json(output/"progress.json",_plain(dict(completed_steps=step+1,history=history,ledger=ledger,
            wall_seconds=time.perf_counter()-started,exact_SU2=True,full_A1=False,status="training")))
        print(json.dumps(_plain(dict(step=step+1,energy=row["energy_before_update"]["energy_per_site"],
            accepted=diagnostics["accepted"],fs=diagnostics["independent_geometry_overlap"]["sampled_fs_sine"],
            wall_seconds=time.perf_counter()-started))),flush=True)
    # The endpoint is frozen BEFORE either audit values or replay gradients.
    save_training_checkpoint(checkpoint,params,optimizer,rng,metadata)
    endpoint_hash = pytree_sha256(params)
    frozen = dict(step=optimizer["step"],accepted_steps=optimizer["accepted_steps"],
        params_sha256=endpoint_hash,checkpoint_sha256=file_sha256(checkpoint),
        training_stopped_reason=training_stopped_reason,full_A1=False,exact_SU2=True)
    atomic_json(output/"frozen_endpoint.json",frozen)
    replay_started = time.perf_counter()
    restored = load_checkpoint(checkpoint)
    restored_params = jax.tree_util.tree_map(jnp.asarray,restored["params"])
    assert pytree_sha256(restored_params) == endpoint_hash
    assert restored["optimizer_state"] == optimizer
    restored_rng = {name:jnp.asarray(key) for name,key in restored["metadata"]["rng"].items()}
    assert set(restored_rng) == set(rng)
    for name in rng:
        np.testing.assert_array_equal(restored_rng[name],rng[name])
    next_training,train_key = jax.random.split(rng["training"])
    next_geometry,geometry_key = jax.random.split(rng["geometry"])
    replay_training,replay_key = jax.random.split(restored_rng["training"])
    replay_geometry,replay_geometry_key = jax.random.split(restored_rng["geometry"])
    samples = draw(params,train_key)
    repeated = draw(restored_params,replay_key)
    calibration = geometry_draw(params,geometry_key)
    repeated_calibration = geometry_draw(restored_params,replay_geometry_key)
    ledger["checkpoint_replay_training_draws"] += 2*len(samples)
    ledger["checkpoint_replay_geometry_draws"] += 2*len(calibration)
    np.testing.assert_array_equal(samples,repeated)
    np.testing.assert_array_equal(calibration,repeated_calibration)
    np.testing.assert_array_equal(next_training,replay_training)
    np.testing.assert_array_equal(next_geometry,replay_geometry)
    amplitude_error = float(np.max(np.abs(np.asarray(log_batch(params,samples))-np.asarray(log_batch(restored_params,repeated)))))
    energies,_ = hamiltonian.evaluate(params,samples,stage="checkpoint_replay")
    # An independently restored full H chunk checks more than amplitude replay.
    repeated_energy,_ = hamiltonian.evaluate(restored_params,repeated[:settings["hamiltonian_chunk_size"]],stage="checkpoint_replay")
    h_error = float(np.max(np.abs(energies[:len(repeated_energy)]-repeated_energy)))
    continued,forward_diag = calibrated_sr_update(params,samples,jnp.asarray(energies),calibration,sr_update,log_batch,settings)
    # Reconstruct the stateless optimizer from its saved hyperparameters, rather
    # than accidentally relying on an unsaved Python optimizer closure.
    resumed_update = make_dual_sampled_sr_update(
        lambda p,x: many_cell_schur_log_amplitudes(p,x[None,:],config)[0],
        restored_params,**restored["optimizer_state"]["sr_settings"])
    resumed,resumed_diag = calibrated_sr_update(restored_params,repeated,jnp.asarray(energies),
        repeated_calibration,resumed_update,log_batch,settings)
    continued_error = max(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(
        jax.tree_util.tree_leaves(continued),jax.tree_util.tree_leaves(resumed),strict=True))
    if max(amplitude_error,h_error,continued_error) > 1e-12 or forward_diag != resumed_diag:
        raise AssertionError("trained optimizer/RNG continuation replay mismatch")
    replay = dict(trained_checkpoint_step=optimizer["step"],optimizer_state_identical=True,
        optimizer_reconstructed_from_saved_settings=True,
        next_training_samples_identical=True,next_geometry_samples_identical=True,next_rng_identical=True,
        amplitude_max_error=amplitude_error,H_ratio_max_error=h_error,
        next_full_network_SR_update_max_error=continued_error,
        speculative_update_not_applied=True,replay_seconds=time.perf_counter()-replay_started,
        checkpoint_bytes=checkpoint.stat().st_size)
    atomic_json(output/"checkpoint_replay.json",replay)
    # Audit key is constructed only after a hashed frozen endpoint exists.
    audit_started = time.perf_counter()
    audit_draw = jax.jit(lambda p,k: sample_many_cell_schur_vit(p,k,settings["audit_samples"],config))
    audit_labels,audit_sample_seconds = measured(lambda: audit_draw(params,jax.random.PRNGKey(settings["audit_seed"])))
    ledger["audit_draws"] = len(audit_labels)
    audit_energies,audit_h = hamiltonian.evaluate(params,audit_labels,stage="untouched_audit")
    if pytree_sha256(params) != endpoint_hash or file_sha256(checkpoint) != frozen["checkpoint_sha256"]:
        raise AssertionError("endpoint changed during independent audit")
    np.savez_compressed(output/"untouched_audit.npz",labels=np.asarray(audit_labels),local_energies=audit_energies,
                        frozen_params_sha256=endpoint_hash,audit_seed=settings["audit_seed"])
    audit = dict(**local_energy_statistics(audit_energies,config.n),full_H=audit_h,
        frozen_params_sha256=endpoint_hash,frozen_checkpoint_step=optimizer["step"],
        native_AR_iid=True,initial_or_training_energy_is_not_paired_audit=True,
        audit_seed=settings["audit_seed"],audit_sampling_seconds=audit_sample_seconds,
        audit_seconds=time.perf_counter()-audit_started,selection_or_training_used_audit=False)
    if "parent_audit_seed" in settings:
        audit["native_spin"] = native_spin_occupations(audit_labels,config)
    atomic_json(output/"untouched_audit.json",audit)
    parent_audit = None
    if "parent_audit_seed" in settings:
        # Parent quality is measured only after the child endpoint was frozen;
        # this independent batch is never used to adapt training or select it.
        parent_params,_ = initialize_from_parent(settings,loaded)
        parent_labels = audit_draw(parent_params,jax.random.PRNGKey(settings["parent_audit_seed"]))
        ledger["parent_audit_draws"] += len(parent_labels)
        parent_energy,parent_h = hamiltonian.evaluate(parent_params,parent_labels,stage="parent_untouched_audit")
        parent_audit = dict(**local_energy_statistics(parent_energy,config.n),full_H=parent_h,
            frozen_params_sha256=pytree_sha256(parent_params), audit_seed=settings["parent_audit_seed"],
            selection_or_training_used_audit=False, native_spin=native_spin_occupations(parent_labels,config))
        np.savez_compressed(output/"parent_untouched_audit.npz",labels=np.asarray(parent_labels),
            local_energies=parent_energy,frozen_params_sha256=pytree_sha256(parent_params),
            audit_seed=settings["parent_audit_seed"])
        atomic_json(output/"parent_untouched_audit.json",parent_audit)
    result = dict(status="completed" if optimizer["step"] == settings["steps"] else "budget_limited_training_audit_completed",
        training_steps=optimizer["step"],accepted_steps=optimizer["accepted_steps"],
        optimized_parameter_count=settings["expected_parameters"],all_parameters_trainable=True,
        initial_params_sha256=initial_hash,terminal_params_sha256=endpoint_hash,
        initialization=initialization,runtime=numerical_runtime_identity(),
        parameters_changed=initial_hash != endpoint_hash,exact_SU2=True,full_A1=False,
        optimized_energy_available=True,native_normalized_Born=True,
        history=history,ledger=ledger,audit=audit,checkpoint_replay=replay,
        compute_wall_seconds=time.perf_counter()-started,
        conclusion="Bounded native S=0 development training; not SU2+A1, not a capacity/work-matched ordinary ViT comparison, and no variational win claim.")
    if parent_audit is not None:
        result["parent_audit"] = parent_audit
        result["independent_audit_difference"] = dict(
            energy_per_site=audit["energy_per_site"]-parent_audit["energy_per_site"],
            energy_stderr_per_site=float(np.hypot(audit["energy_stderr_per_site"],parent_audit["energy_stderr_per_site"])),
            variance_per_site=audit["variance_per_site"]-parent_audit["variance_per_site"],
            variance_stderr_plugin_per_site=float(np.hypot(audit["variance_stderr_plugin_per_site"],
                parent_audit["variance_stderr_plugin_per_site"])), paired=False,
            common_architecture=True, common_H=True, same_sampling_count=True)
    atomic_json(output/"results.json",_plain(result))
    return result


def run(config_path: Path) -> dict:
    """Register all fixed settings/sources, then enforce one-hour child cap."""
    settings = json.loads(config_path.read_text())
    _validate(settings)
    output = ROOT/settings["output"]
    output.mkdir(parents=True,exist_ok=False)
    # The package initializer eagerly resolves its public local_energy alias
    # with import_module; include that runtime dependency beyond static AST imports.
    manifest = transitive_local_source_manifest([Path(__file__),ROOT/"su2_nqs/vmc/local_energy.py"],root=ROOT,
        package_prefixes=("su2_nqs","projects.clebsch_tree_nqs"))
    manifest[str(config_path.relative_to(ROOT))] = file_sha256(config_path)
    for name in manifest:
        saved = output/"source_snapshot"/name
        saved.parent.mkdir(parents=True,exist_ok=True)
        saved.write_bytes((ROOT/name).read_bytes())
    parent = ROOT/settings["parent_checkpoint"]
    if file_sha256(parent) != settings["parent_checkpoint_sha256"]:
        raise ValueError("parent checkpoint differs before registration")
    atomic_json(output/"registration.json",dict(config=settings,sources=manifest,
        parent_checkpoint_sha256=file_sha256(parent),runtime=numerical_runtime_identity(),
        registered_UTC=datetime.now(timezone.utc).isoformat(),
        full_A1=False,exact_SU2=True,model_changes=bool(settings.get("capacity_expansion")),head_changes=False,
        audit_used_for_training_or_selection=False,hard_compute_cap_seconds=settings["total_seconds"]))
    started = time.perf_counter()
    platform = settings.get("platform", "cpu")
    env = {**os.environ,"JAX_PLATFORM_NAME":platform,"JAX_PLATFORMS":"cuda" if platform=="gpu" else "cpu",
           "JAX_ENABLE_X64":"true","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2"}
    with (output/"compute.log").open("w") as stream:
        process = subprocess.Popen([sys.executable,"-m",MODULE,"--config",str(config_path),"--child"],
            cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        try:
            exit_code = process.wait(timeout=settings["total_seconds"])
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            exit_code = process.returncode
            atomic_json(output/"budget_stop.json",dict(status="hard_compute_cap",compute_seconds=time.perf_counter()-started,
                checkpoint_retained=(output/"trained_checkpoint.pkl").exists(),
                complete_512_audit=(output/"untouched_audit.json").exists()))
    result_path = output/"results.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else dict(status="incomplete",
        optimized_energy_available=False,full_A1=False,
        trained_checkpoint_retained=(output/"trained_checkpoint.pkl").exists())
    completion = dict(exit_code=exit_code,compute_wall_seconds=time.perf_counter()-started,
        results_sha256=file_sha256(result_path) if result_path.exists() else None,
        audit_complete=(output/"untouched_audit.json").exists(),
        hard_compute_cap_seconds=settings["total_seconds"])
    atomic_json(output/"completion.json",completion)
    if exit_code:
        raise RuntimeError(f"bounded training child failed ({exit_code}); see {output/'compute.log'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--child",action="store_true")
    args = parser.parse_args()
    path = args.config.resolve()
    if args.child:
        jax.config.update("jax_enable_x64",True)
        settings = json.loads(path.read_text())
        try:
            _train(settings,ROOT/settings["output"])
        except BaseException as error:
            atomic_json(ROOT/settings["output"]/"failure.json",dict(exception=type(error).__name__,
                message=str(error),traceback=traceback.format_exc()))
            raise
    else:
        print(json.dumps(run(path),indent=2))


if __name__ == "__main__":
    main()
