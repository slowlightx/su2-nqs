"""Bounded N36 full-ViT sampled feasibility, after the N16 reference gates.

This is not production, an energy win, or an SU(2) architecture comparison.
N16 is a prerequisite only: its exact energy is never used in this loss.
"""

import argparse
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.spatial_vit_sampling import (
    projected_log_function, make_full_heisenberg_ratios, make_log_exchange_sampler,
)
from su2_nqs.models.fixed_sz_spatial_vit import FixedSzSpatialViTConfig, init_fixed_sz_spatial_vit, fixed_sz_spatial_vit_parameter_count
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.adam import _manual_adam_init, _manual_adam_update
from su2_nqs.vmc.checkpoint import save_checkpoint, load_checkpoint
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity, transitive_local_source_manifest
from .spin import _groups, _bonds
from .spin_full_vit_budget import ROOT, DEFAULT_CONFIG, DEFAULT_OUTPUT

jax.config.update("jax_enable_x64", True)
PILOT_CONFIG = ROOT/"projects/clebsch_tree_nqs/configs/validation/spin_full_vit_n36_pilot.json"
CONTINUATION_CONFIG = PILOT_CONFIG.with_name("spin_full_vit_n36_continuation.json")


def make_sampled_adam_update(log_batch: Callable, spec: dict) -> Callable:
    """Real-parameter complex-wavefunction VMC gradient; identical pilot Adam."""
    @jax.jit
    def update(p, opt, x, energies):
        centered = jax.lax.stop_gradient(energies-jnp.mean(energies))
        def loss(candidate):
            return 2*jnp.real(jnp.mean(jnp.conj(log_batch(candidate,x))*centered))
        gradient = jax.grad(loss)(p)
        norm = jnp.sqrt(sum(jnp.sum(v*v) for v in jax.tree_util.tree_leaves(gradient)))
        gradient = jax.tree_util.tree_map(lambda v: v*jnp.minimum(
            1.,spec["gradient_clip_norm"]/jnp.maximum(norm,1e-30)),gradient)
        new, opt = _manual_adam_update(p,gradient,opt,spec["learning_rate"])
        return new,opt,norm
    return update


def sampled_transition(state: dict, moments: object, sample: Callable,
                       local: Callable, update: Callable) -> tuple:
    """One deterministic transition, including sampler key and Adam moments.

    Reject nonfinite proposals before the caller replaces a valid checkpoint.
    The chain sampler refreshes log amplitudes under the current parameters.
    """
    x,key,accept,moved = sample(state["model"],state["key"],state["chains"])
    energies = local(state["model"],x)
    params,next_moments,norm = update(state["model"],moments,x,energies)
    jax.block_until_ready((params,next_moments,energies))
    if not all(np.all(np.isfinite(v)) for v in jax.tree_util.tree_leaves(
            (params,next_moments,energies,norm))):
        raise FloatingPointError("nonfinite transition; last valid checkpoint retained")
    return dict(model=params,chains=x,key=key),next_moments,energies,float(norm),float(moved)


def audit_statistics(values: np.ndarray, n_sites: int, *, bootstrap_seed: int,
                     replicates: int = 2000) -> dict:
    """Describe independent streams of (snapshots, chains) complex local E.

    Resample entire chains, not individual correlated snapshots. Finite burn-in
    bias is not included in these conditional uncertainty estimates.
    """
    values = np.asarray(values)
    if values.ndim != 3 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("need finite (streams, snapshots, chains), all axes >=2")
    streams,batches,chains = values.shape
    paths = values.transpose(0,2,1).reshape(streams*chains,batches)
    means,second = paths.mean(axis=1),np.mean(abs(paths)**2,axis=1)
    mean = means.mean()
    variance = np.mean(abs(paths-mean)**2)/n_sites
    se = means.real.std(ddof=1)/np.sqrt(len(means))/n_sites
    influence = np.mean(abs(paths-mean)**2,axis=1)/n_sites
    variance_se = influence.std(ddof=1)/np.sqrt(len(means))
    draws = np.random.default_rng(bootstrap_seed).integers(len(means),size=(replicates,len(means)))
    boot_mean = means[draws].mean(axis=1)
    boot_var = (second[draws].mean(axis=1)-abs(boot_mean)**2)/n_sites
    half = batches//2
    drift = (paths[:,half:].real.mean(axis=1)-paths[:,:half].real.mean(axis=1))/n_sites
    split = np.concatenate([paths[:,:half].real,paths[:,-half:].real],axis=0)
    within = np.var(split,axis=1,ddof=1).mean()
    between = half*np.var(split.mean(axis=1),ddof=1)
    rhat = np.sqrt(((half-1)*within+between)/(half*within)) if within>0 else None
    centered = paths.real-paths.real.mean(axis=1,keepdims=True)
    lag_denominator = np.sqrt(np.sum(centered[:,:-1]**2)*np.sum(centered[:,1:]**2))
    lag1 = np.sum(centered[:,:-1]*centered[:,1:])/lag_denominator if lag_denominator>0 else 0.
    tails = np.sort(abs(paths-mean).ravel()**2)
    count = max(1,int(np.ceil(.01*len(tails))))
    return dict(energy_per_site=float(mean.real/n_sites),
        energy_imaginary_per_site=float(mean.imag/n_sites),
        chain_mean_standard_error=float(se),variance_per_site=float(variance),
        variance_chain_standard_error=float(variance_se),
        energy_chain_bootstrap_95=np.quantile(boot_mean.real/n_sites,[.025,.975]).tolist(),
        variance_chain_bootstrap_95=np.quantile(boot_var,[.025,.975]).tolist(),
        stream_energies_per_site=(values.mean(axis=(1,2)).real/n_sites).tolist(),
        stream_variances_per_site=[float(np.mean(abs(v-v.mean())**2)/n_sites) for v in values],
        split_chain_energy_rhat=None if rhat is None else float(rhat),
        energy_lag1_descriptive=float(lag1),
        second_minus_first_half_per_site=float(drift.mean()),
        half_drift_chain_standard_error=float(drift.std(ddof=1)/np.sqrt(len(drift))),
        top_one_percent_variance_share=float(tails[-count:].sum()/tails.sum()) if tails.sum()>0 else 0.,
        samples=int(values.size),independent_chains=len(means),mixing_certified=False)


def validate_n16_reference(result: dict, baseline: dict) -> None:
    """Fail closed on stale/nonfinite references and the registered tail gate."""
    if result["result_sha256"] != canonical_sha256({k:v for k,v in result.items() if k != "result_sha256"}):
        raise ValueError("N16 artifact seal changed")
    if result["config_sha256"] != canonical_sha256(baseline):
        raise ValueError("N16 protocol differs from the current registered baseline")
    gate=baseline["pilot_reference_gates"]
    metrics=result["metrics"]
    errors=[*result["checks"]["initial"].values(),*result["checks"]["final"].values(),
            result["checks"]["checkpoint_amplitude_replay_max"]]
    if (not np.isfinite([*metrics.values(),*errors]).all() or max(errors)>2e-10
        or metrics["energy_error_per_site"] < -1e-10
        or metrics["energy_error_per_site"] > gate["maximum_energy_error_per_site"]
        or not 0 <= metrics["variance_per_site"] <= gate["maximum_variance_per_site"]):
        raise ValueError("N16 finite/accuracy/variance/replay references not satisfied")
    adam=[r for r in result["training"]["history"] if r["stage"]=="adam"]
    if not adam:
        raise ValueError("N16 Adam trajectory is missing")
    terminal=adam[-1]
    earlier=[r for r in adam if r["step"]<=terminal["step"]-100]
    if not earlier:
        raise ValueError("N16 last-100-step reference is missing")
    previous=earlier[-1]["energy_error_per_site"]
    fraction=(previous-terminal["energy_error_per_site"])/max(previous,1e-15)
    if not np.isfinite(fraction) or fraction < gate["minimum_fraction_energy_error_removed_by_last_100_adam_steps"]:
        raise ValueError("N16 registered training-tail reference not satisfied")


def run_pilot(geometry: str, output: Path) -> dict:
    """Run the registered primary width without selecting a winning seed."""
    spec, baseline = json.loads(PILOT_CONFIG.read_text()), json.loads(DEFAULT_CONFIG.read_text())
    arm = next(a for a in baseline["arms"] if a["id"] == spec["primary_arm"])
    anchors = []
    for seed in baseline["seeds"]:
        path = DEFAULT_OUTPUT/f'{geometry}_{arm["id"]}_s{seed}'/"results.json"
        result = json.loads(path.read_text())
        validate_n16_reference(result,baseline)
        anchors.append(result["result_sha256"])
    destination = output/geometry
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/"results.json").exists() or (destination/"checkpoint.pkl").exists():
        raise FileExistsError(destination)
    started = perf_counter()
    config = FixedSzSpatialViTConfig(geometry=geometry, rows=6,columns=6,
        **baseline["model"],**{k:v for k,v in arm.items() if k!="id"})
    params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(spec["seed"]),config)
    initial_hash = pytree_sha256(params)
    groups = _groups(geometry,6,6)
    ratio = .125 if geometry=="triangular" else .5
    bonds = _bonds(geometry,6,6,ratio)
    log_fn, image_count = projected_log_function(config,groups["gamma_a1"])
    log_batch = jax.jit(jax.vmap(log_fn,in_axes=(None,0)))
    local = make_full_heisenberg_ratios(log_fn,bonds)
    sample = make_log_exchange_sampler(log_fn,spec["proposals_per_update"])
    thermal = make_log_exchange_sampler(log_fn,spec["initial_proposals"])
    audit_thermal = make_log_exchange_sampler(log_fn,spec["audit_proposals"])
    key = jax.random.PRNGKey(spec["seed"]+1)
    states = initialize_spin_exchange_chains(key,n_chains=spec["chains"],n_sites=36,n_up=18)
    moments = _manual_adam_init(params)

    update = make_sampled_adam_update(log_batch,spec)

    compile_started = perf_counter()
    # Warm calls do not modify the initialized training state or optimizer.
    jax.block_until_ready(sample(params,key,states))
    warm_local = local(params,states); jax.block_until_ready(warm_local)
    jax.block_until_ready(update(params,moments,states,warm_local))
    compile_seconds = perf_counter()-compile_started
    print(geometry,"N36 compiled",compile_seconds,"s; projected images",image_count,flush=True)

    def audit(p, audit_seed):
        audit_started = perf_counter()
        rng = jax.random.PRNGKey(audit_seed)
        x = initialize_spin_exchange_chains(rng,n_chains=spec["chains"],n_sites=36,n_up=18)
        rng, draw = jax.random.split(rng)
        x,rng,_,_ = audit_thermal(p,draw,x)
        samples,values,moves = [],[],[]
        for _ in range(spec["audit_batches"]):
            x,rng,accept,move = sample(p,rng,x)
            y = np.asarray(local(p,x))
            samples.append(np.asarray(x)); values.append(y); moves.append(float(move))
        values = np.asarray(values)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("nonfinite untouched local energies")
        mean = values.mean()
        means = values.real.mean(axis=0)/36
        return dict(energy_per_site=float(mean.real/36),
                    energy_imaginary_per_site=float(mean.imag/36),
                    chain_mean_standard_error=float(means.std(ddof=1)/np.sqrt(len(means))),
                    variance_per_site=float(np.mean(abs(values-mean)**2)/36),
                    mean_nontrivial_acceptance=float(np.mean(moves)),
                    samples=int(values.size), chains=spec["chains"],
                    unique_configurations=int(len(np.unique(np.asarray(samples).reshape(-1,36),axis=0))),
                    seconds=perf_counter()-audit_started,
                    mixing_certified=False)

    initial = audit(params,spec["seed"]+100)
    states,key,_,_ = thermal(params,key,states)
    jax.block_until_ready(states)
    history=[]; active_seconds=0.
    for step in range(1,spec["maximum_steps"]+1):
        if active_seconds >= spec["maximum_training_seconds"]:
            break
        step_started=perf_counter()
        states,key,accept,moved = sample(params,key,states)
        energies=local(params,states)
        proposed,next_moments,gradient_norm = update(params,moments,states,energies)
        jax.block_until_ready((proposed,energies))
        if not all(np.all(np.isfinite(v)) for v in jax.tree_util.tree_leaves(proposed)) or not np.all(np.isfinite(energies)):
            raise FloatingPointError("nonfinite VMC step; no production release")
        params,moments = proposed,next_moments
        elapsed=perf_counter()-step_started; active_seconds+=elapsed
        row=dict(step=step,energy_before_update_per_site=float(jnp.mean(energies).real/36),
            variance_before_update_per_site=float(jnp.mean(abs(energies-jnp.mean(energies))**2)/36),
            seconds=elapsed,active_seconds=active_seconds,gradient_norm=float(gradient_norm),
            nontrivial_acceptance=float(moved))
        history.append(row)
        if step%8==0 or step==1:
            print(geometry,row,flush=True)
        save_checkpoint(destination/"checkpoint.pkl",dict(model=params,chains=states,key=key),
            optimizer_state=moments,metadata=dict(step=step,history=history,config=spec,
            model_config=asdict(config),active_seconds=active_seconds))
    final=audit(params,spec["seed"]+200)
    saved=load_checkpoint(destination/"checkpoint.pkl")
    replay=float(np.max(abs(np.asarray(log_batch(params,states))-np.asarray(log_batch(saved["params"]["model"],states)))))
    reference=np.asarray(log_batch(params,states))
    symmetry_error=0.
    for perm in groups["gamma_a1"][::max(1,len(groups["gamma_a1"])//8)]:
        transformed=np.asarray(log_batch(params,states[:,jnp.asarray(perm)]))
        symmetry_error=max(symmetry_error,float(np.max(abs(np.exp(transformed-reference)-1))))
    result=dict(task=spec["task"],config=spec,geometry=geometry,j2_over_j1=ratio,
        model_config=asdict(config),parameter_count=fixed_sz_spatial_vit_parameter_count(params),
        n16_reference_seals=anchors,initial_params_sha256=initial_hash,
        final_params_sha256=pytree_sha256(params),status="bounded_complete",history=history,
        initial_audit=initial,final_audit=final,compile_seconds=compile_seconds,
        training_seconds=active_seconds,wall_seconds=perf_counter()-started,
        completed_steps=len(history),checkpoint_log_amplitude_replay_max=replay,
        sampled_A1_relative_error=symmetry_error,spatial_group_order=len(groups["gamma_a1"]),
        amplitude_images=image_count,bond_count=len(bonds),
        full_H_configuration_calls=spec["chains"]*(1+len(history)+2*spec["audit_batches"]),
        stop="steps" if len(history)==spec["maximum_steps"] else "time",
        remote_submitted=False,production_released=False,variational_win_claimed=False,
        runtime=numerical_runtime_identity(),source_manifest=transitive_local_source_manifest(
            (Path(__file__),),root=ROOT,package_prefixes=("su2_nqs","projects.clebsch_tree_nqs")))
    result["result_sha256"]=canonical_sha256(result)
    atomic_json(destination/"results.json",result)
    print(geometry,"initial",initial,"final",final,flush=True)
    return result


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed_json(path: Path) -> dict:
    result = json.loads(path.read_text())
    if result["result_sha256"] != canonical_sha256({k:v for k,v in result.items() if k!="result_sha256"}):
        raise ValueError(f"artifact seal changed: {path}")
    return result


def validate_parent(spec: dict, geometry: str) -> tuple[dict, dict]:
    """Pin parent bytes, model, Adam count, sector and unchanged update protocol."""
    source = ROOT/spec["parent_directory"]/geometry
    expected = spec["parents"][geometry]
    result = _sealed_json(source/"results.json")
    if (result["result_sha256"] != expected["result_sha256"] or
            _file_sha256(source/"checkpoint.pkl") != expected["checkpoint_sha256"]):
        raise ValueError("parent artifacts differ from preregistration")
    saved = load_checkpoint(source/"checkpoint.pkl")
    if (saved["metadata"]["step"] != spec["parent_step"] or
            int(saved["optimizer_state"][0]) != spec["parent_step"] or
            pytree_sha256(saved["params"]["model"]) != result["final_params_sha256"] or
            saved["metadata"]["model_config"] != result["model_config"]):
        raise ValueError("parent model/optimizer identity mismatch")
    for key in ("chains","proposals_per_update","learning_rate","gradient_clip_norm"):
        if spec[key] != saved["metadata"]["config"][key]:
            raise ValueError(f"continuation changed {key}")
    states = np.asarray(saved["params"]["chains"])
    if states.shape != (spec["chains"],36) or not np.all(states.sum(axis=1)==18):
        raise ValueError("parent chain sector mismatch")
    return saved,result


def run_continuation(geometry: str, output: Path, *, resume: bool = False,
                     config_path: Path = CONTINUATION_CONFIG) -> dict:
    """Fixed-length full-backbone continuation, then independent frozen-state audits.

    Audit results never enter the optimizer or stopping rule. A completed run
    cannot be overwritten. Explicit resume requires the identical source and
    registration, and restores all training state, including the sampler RNG.
    """
    started = perf_counter()
    spec = json.loads(config_path.read_text())
    parent,parent_result = validate_parent(spec,geometry)
    seeds = [s for group in spec["audit"]["seeds"].values() for pool in group.values() for s in pool]
    if len(seeds) != len(set(seeds)) or spec["additional_steps"] != 1000:
        raise ValueError("registered fresh seeds / fixed 1000-step budget invalid")
    destination = output/geometry
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/"results.json").exists():
        raise FileExistsError("completed result is immutable")
    registration_path = destination/"registration.json"
    registration = dict(config=spec,config_sha256=canonical_sha256(spec),geometry=geometry,
        parent_state_sha256=pytree_sha256((parent["params"],parent["optimizer_state"])),
        source_manifest=transitive_local_source_manifest((Path(__file__),),root=ROOT,
            package_prefixes=("su2_nqs","projects.clebsch_tree_nqs")))
    if registration_path.exists():
        if not resume:
            raise FileExistsError("use explicit --resume for an incomplete registered run")
        stored = _sealed_json(registration_path)
        if any(stored[k] != v for k,v in registration.items()):
            raise ValueError("resume registration/source changed")
        registration = stored
    else:
        if resume:
            raise FileNotFoundError("cannot resume before registration")
        registration["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        registration["result_sha256"] = canonical_sha256(registration)
        atomic_json(registration_path,registration)
    checkpoint = destination/"checkpoint.pkl"
    if resume and checkpoint.exists():
        saved = load_checkpoint(checkpoint)
        state,moments,meta = saved["params"],saved["optimizer_state"],saved["metadata"]
        if (meta["registration_sha256"] != registration["result_sha256"] or
                int(moments[0]) != spec["parent_step"]+meta["completed_steps"] or
                len(meta["history"]) != meta["completed_steps"] or
                not 0 <= meta["completed_steps"] <= spec["additional_steps"]):
            raise ValueError("continuation checkpoint identity/count mismatch")
    else:
        state,moments = parent["params"],parent["optimizer_state"]
        meta = dict(registration_sha256=registration["result_sha256"],completed_steps=0,
            step=spec["parent_step"],history=[],active_seconds=0.,compile_seconds=0.,
            full_H_configuration_calls=0,compile_invocations=0,model_config=parent_result["model_config"])
        save_checkpoint(checkpoint,state,optimizer_state=moments,metadata=meta)
    config = FixedSzSpatialViTConfig(**parent_result["model_config"])
    groups = _groups(geometry,6,6)
    bonds = _bonds(geometry,6,6,parent_result["j2_over_j1"])
    log_fn,image_count = projected_log_function(config,groups["gamma_a1"])
    log_batch = jax.jit(jax.vmap(log_fn,in_axes=(None,0)))
    local = make_full_heisenberg_ratios(log_fn,bonds)
    sample = make_log_exchange_sampler(log_fn,spec["proposals_per_update"])
    update = make_sampled_adam_update(log_batch,spec)
    compile_started = perf_counter()
    # Warm replay is discarded: do not advance training chains, PRNG or moments.
    jax.block_until_ready(sample(state["model"],state["key"],state["chains"]))
    warm = local(state["model"],state["chains"])
    jax.block_until_ready(update(state["model"],moments,state["chains"],warm))
    meta["compile_seconds"] += perf_counter()-compile_started
    meta["full_H_configuration_calls"] += spec["chains"]
    meta["compile_invocations"] += 1
    save_checkpoint(checkpoint,state,optimizer_state=moments,metadata=meta)
    print(geometry,"registered",registration["result_sha256"],"resume at",meta["step"],flush=True)
    for added_step in range(meta["completed_steps"]+1,spec["additional_steps"]+1):
        tick = perf_counter()
        state,moments,energies,norm,moved = sampled_transition(state,moments,sample,local,update)
        elapsed = perf_counter()-tick
        meta["active_seconds"] += elapsed
        meta["completed_steps"] = added_step
        meta["step"] = spec["parent_step"]+added_step
        meta["full_H_configuration_calls"] += spec["chains"]
        e = np.asarray(energies)
        row = dict(step=meta["step"],added_step=added_step,
            energy_before_update_per_site=float(e.mean().real/36),
            variance_before_update_per_site=float(np.mean(abs(e-e.mean())**2)/36),
            seconds=elapsed,active_seconds=meta["active_seconds"],gradient_norm=norm,
            nontrivial_acceptance=moved)
        meta["history"].append(row)
        save_checkpoint(checkpoint,state,optimizer_state=moments,metadata=meta)
        if added_step==1 or added_step%50==0:
            print(geometry,row,flush=True)

    # Both starting and terminal states are measured only after all 1000 steps.
    # No audit PRNG is shared with training or with another frozen state.
    training_hash = pytree_sha256((state,moments))
    audit_spec = spec["audit"]
    thermal = make_log_exchange_sampler(log_fn,audit_spec["burn_proposals"])
    audit_sample = make_log_exchange_sampler(log_fn,audit_spec["proposals_between_snapshots"])
    audits = {}
    audit_values = {}
    for label,p in (("parent",parent["params"]["model"]),("terminal",state["model"])):
        streams,records = [],[]
        for seed in audit_spec["seeds"][geometry][label]:
            record_path = destination/f"audit_{label}_{seed}.json"
            values_path = record_path.with_suffix(".npz")
            if record_path.exists():
                record = _sealed_json(record_path)
                if (record["registration_sha256"] != registration["result_sha256"] or
                        record["params_sha256"] != pytree_sha256(p) or
                        record["raw_sha256"] != _file_sha256(values_path)):
                    raise ValueError("audit resume seal mismatch")
                with np.load(values_path) as raw:
                    values = raw["local_energies"]
            else:
                audit_started = perf_counter()
                init_key,rng = jax.random.split(jax.random.PRNGKey(seed))
                x = initialize_spin_exchange_chains(init_key,n_chains=spec["chains"],n_sites=36,n_up=18)
                x,rng,_,_ = thermal(p,rng,x)
                samples,values,moves = [],[],[]
                for _ in range(audit_spec["snapshots"]):
                    x,rng,_,move = audit_sample(p,rng,x)
                    values.append(np.asarray(local(p,x)))
                    samples.append(np.asarray(x)); moves.append(float(move))
                values = np.asarray(values)
                if not np.isfinite(values).all():
                    raise FloatingPointError("nonfinite independent audit; do not retune training")
                np.savez_compressed(values_path,local_energies=values,states=np.asarray(samples),
                    final_key=np.asarray(rng),nontrivial_acceptance=np.asarray(moves))
                record = dict(seed=seed,label=label,registration_sha256=registration["result_sha256"],
                    params_sha256=pytree_sha256(p),raw_sha256=_file_sha256(values_path),
                    samples=int(values.size),full_H_configuration_calls=int(values.size),
                    seconds=perf_counter()-audit_started,mean_nontrivial_acceptance=float(np.mean(moves)),
                    unique_configurations=int(len(np.unique(np.asarray(samples).reshape(-1,36),axis=0))))
                record["result_sha256"] = canonical_sha256(record)
                atomic_json(record_path,record)
            streams.append(values); records.append(record)
            print(geometry,label,"untouched seed",seed,"E/N",float(values.mean().real/36),flush=True)
        audit_values[label] = np.asarray(streams)
        audits[label] = audit_statistics(np.asarray(streams),36,
            bootstrap_seed=audit_spec["bootstrap_seed"],replicates=audit_spec["bootstrap_replicates"])
        audits[label]["streams"] = records
    # Independent parent/terminal streams: differences are not paired estimates.
    differences = {}
    for quantity,se_key in (("energy_per_site","chain_mean_standard_error"),
                            ("variance_per_site","variance_chain_standard_error")):
        delta = audits["terminal"][quantity]-audits["parent"][quantity]
        se = np.hypot(audits["terminal"][se_key],audits["parent"][se_key])
        differences[quantity] = dict(terminal_minus_parent=delta,unpaired_chain_standard_error=float(se),
            normal_approximation_95=[float(delta-1.96*se),float(delta+1.96*se)])
    # Bootstrap the difference with independent chain resamples in each state.
    rng = np.random.default_rng(audit_spec["bootstrap_seed"]+1)
    bootstrap = {}
    for label,values in audit_values.items():
        paths = values.transpose(0,2,1).reshape(-1,values.shape[1])
        means,second = paths.mean(axis=1),np.mean(abs(paths)**2,axis=1)
        indices = rng.integers(len(means),size=(audit_spec["bootstrap_replicates"],len(means)))
        mean = means[indices].mean(axis=1)
        bootstrap[label] = (mean.real/36,(second[indices].mean(axis=1)-abs(mean)**2)/36)
    for k,quantity in enumerate(("energy_per_site","variance_per_site")):
        differences[quantity]["unpaired_chain_bootstrap_95"] = np.quantile(
            bootstrap["terminal"][k]-bootstrap["parent"][k],[.025,.975]).tolist()
    saved = load_checkpoint(checkpoint)
    if training_hash != pytree_sha256((state,moments)) or training_hash != pytree_sha256((saved["params"],saved["optimizer_state"])):
        raise ValueError("audit altered training or checkpoint state")
    reference = np.asarray(log_batch(state["model"],state["chains"]))
    replay = float(np.max(abs(reference-np.asarray(log_batch(saved["params"]["model"],state["chains"])))))
    symmetry_error = 0.
    for perm in groups["gamma_a1"][::max(1,len(groups["gamma_a1"])//8)]:
        transformed = np.asarray(log_batch(state["model"],state["chains"][:,jnp.asarray(perm)]))
        symmetry_error = max(symmetry_error,float(np.max(abs(np.exp(transformed-reference)-1))))
    result = dict(task=spec["task"],geometry=geometry,config=spec,
        registration_sha256=registration["result_sha256"],parent_result_sha256=parent_result["result_sha256"],
        status="fixed_budget_complete",completed_additional_steps=meta["completed_steps"],
        total_steps=meta["step"],parameter_count=fixed_sz_spatial_vit_parameter_count(state["model"]),
        model_config=asdict(config),j2_over_j1=parent_result["j2_over_j1"],
        history=meta["history"],audits=audits,independent_differences=differences,
        training_seconds=meta["active_seconds"],compile_seconds=meta["compile_seconds"],
        wall_seconds_this_invocation=perf_counter()-started,compile_invocations=meta["compile_invocations"],
        full_H_configuration_calls=meta["full_H_configuration_calls"]+sum(a["samples"] for a in audits.values()),
        checkpoint_sha256=_file_sha256(checkpoint),training_state_sha256=training_hash,
        final_params_sha256=pytree_sha256(state["model"]),checkpoint_log_amplitude_replay_max=replay,
        sampled_A1_relative_error=symmetry_error,amplitude_images=image_count,bond_count=len(bonds),
        remote_submitted=False,production_released=False,variational_win_claimed=False,
        spin_symmetry="fixed Sz=0; NOT exact S=0",spatial_symmetry="exact Gamma-A1",
        runtime=numerical_runtime_identity())
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination/"results.json",result)
    print(geometry,"COMPLETE",differences,flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry",choices=("triangular","square"),required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--continuation",action="store_true",help="registered 1000 extra updates from small-model checkpoint")
    parser.add_argument("--resume",action="store_true",help="resume an incomplete continuation, never overwrite a result")
    args=parser.parse_args()
    if args.continuation:
        run_continuation(args.geometry,args.output or ROOT/"projects/clebsch_tree_nqs/results/spin_full_vit_n36_continuation_v1",resume=args.resume)
    else:
        if args.resume:
            parser.error("--resume is only implemented for --continuation")
        run_pilot(args.geometry,args.output or ROOT/"projects/clebsch_tree_nqs/results/spin_full_vit_n36_pilot_v1")
