"""One fixed follow-up: correlated Born-targeted teachers, independent IS audit.

The Metropolis chains supply training configurations only. They do not estimate
energies, certify equilibration, or replace the exactly known proposal law in
the fresh final importance panels. Architecture and optimizer match the first
probe without tuning to its heldout configurations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from experiments.attention_sampling_probe import concentration, fit, save_json
from experiments.learned_proposals import DefensiveLearnedProposal, FixedMagnetizationAR
from peps_vmc.contraction import make_square_torus_plan
from peps_vmc.model import SquareGraphSU2PEPS
from peps_vmc.proposals import CoveringBridgeProposal, torus_coverings
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint


def main() -> None:
    started = time.monotonic()
    output = Path("peps-vmc/results/attention_sampling_20261029/target_aware_teacher")
    output.mkdir(parents=False, exist_ok=False)
    endpoint = Path("peps-vmc/results/triangular_6x6_j2_0125_local_20261029/frozen_endpoint.pkl")
    config = dict(chains=16, burn_in_steps=64, saved_intervals=96, save_stride=4,
        mcmc_seed=29201, regularizer_seed=29202, train_resampling_seed=29203,
        evaluation_seeds=dict(bridge=29210, attention=29211, gru=29212),
        regularizer_records=384, train_steps=300, batch_size=128,
        architecture_and_optimizer="same as first registered probe; no tuning",
        learned_fraction=.75, empirical_MC_training_fraction=.9,
        training_total_wall_cap_seconds=120, overall_wall_cap_seconds=180,
        parameter_seed=29102, independent_records_per_proposal=512,
        endpoint_sha256=hashlib.sha256(endpoint.read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path(__file__).with_name("attention_sampling_probe.py"),
             Path(__file__).with_name("learned_proposals.py"))},
        MC_scope="training data only; correlated, finite burn-in, no mixing certificate",
        checkpoint_policy="fixed final 300 updates; no audit selection")
    save_json(output/"registration.json", config)
    bonds = triangular_j1j2_bonds(6, 6, j1=1., j2=.125, periodic=True)
    pairs = np.asarray([(a, b) for a, b, _ in bonds], np.int32)
    couplings = np.asarray([v for _, _, v in bonds])
    bridge = CoveringBridgeProposal(torus_coverings(6, 6, count=48, seed=29002),
        pairs, np.abs(couplings)/np.abs(couplings).sum(), epsilon=.25)
    model = SquareGraphSU2PEPS(6, 6)
    plan = make_square_torus_plan(6, 6, 3)
    tensors = model.tensors(jnp.asarray(load_checkpoint(endpoint)["params"]["w"]))
    amplitude = jax.jit(plan.amplitude)
    def amplitudes(rows):
        return np.asarray([float(amplitude(tensors, jnp.asarray(row))) for row in rows])
    def resource_check():
        if time.monotonic()-started > config["overall_wall_cap_seconds"]:
            raise TimeoutError("fixed follow-up total wall cap reached")
    teacher_start = time.monotonic()
    rng = np.random.default_rng(config["mcmc_seed"])
    chains, _ = bridge.draw(config["chains"], rng)
    values = amplitudes(chains)
    initial_log = np.log(abs(values))
    snapshots, saved_values = [], []
    log_trace = [initial_log.copy()]
    proposed_nonnull = np.zeros(config["chains"], np.int64)
    accepted_nonnull = np.zeros(config["chains"], np.int64)
    accepted = np.zeros(config["chains"], np.int64)
    steps = config["burn_in_steps"]+config["saved_intervals"]*config["save_stride"]
    for step in range(steps):
        candidate = chains.copy()
        for row in candidate:
            if rng.random() < .5:
                # Constant n_up*n_down makes this long-range swap symmetric.
                a = int(rng.choice(np.flatnonzero(row == 0)))
                b = int(rng.choice(np.flatnonzero(row == 1)))
            else:
                # Including null bond swaps keeps this proposal symmetric.
                a, b = pairs[int(rng.choice(len(pairs), p=bridge.exchange_probabilities))]
            row[a], row[b] = row[b], row[a]
        candidate_values = amplitudes(candidate)
        nonnull = np.any(candidate != chains, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = 2*(np.log(abs(candidate_values))-np.log(abs(values)))
        accept = np.log(rng.random(len(chains))) < np.minimum(0., log_ratio)
        accept |= (values == 0) & (candidate_values != 0)
        chains[accept], values[accept] = candidate[accept], candidate_values[accept]
        proposed_nonnull += nonnull
        accepted_nonnull += accept & nonnull
        accepted += accept
        if step == config["burn_in_steps"]-1:
            burnin_log = np.log(abs(values))
        if step >= config["burn_in_steps"] and (step+1-config["burn_in_steps"]) % config["save_stride"] == 0:
            snapshots.append(chains.copy())
            saved_values.append(values.copy())
            log_trace.append(np.log(abs(values)))
        if (step+1) % 64 == 0:
            resource_check()
    saved_rows = np.asarray(snapshots)
    saved_amps = np.asarray(saved_values)
    teacher = saved_rows.reshape((-1, 36))
    regularizer, _ = bridge.draw(config["regularizer_records"], np.random.default_rng(config["regularizer_seed"]))
    training_rows = np.concatenate((teacher, regularizer))
    probability = np.concatenate((np.full(len(teacher), .9/len(teacher)),
                                   np.full(len(regularizer), .1/len(regularizer))))
    indices = np.random.default_rng(config["train_resampling_seed"]).choice(
        len(training_rows), size=(300, 128), p=probability)
    log_saved = np.log(abs(saved_amps))
    centered = log_saved-log_saved.mean(axis=0)
    lag1 = np.sum(centered[:-1]*centered[1:], axis=0)/np.sqrt(
        np.sum(centered[:-1]**2, axis=0)*np.sum(centered[1:]**2, axis=0))
    teacher_seconds = time.monotonic()-teacher_start
    teacher_info = dict(seconds=teacher_seconds, exact_amplitude_queries=16*(1+steps),
        correlated_MC_records=len(teacher), independent_initialization="16 fresh bridge draws",
        proposal="half uniform opposite-spin exchange; half |J|-weighted bond exchange including null swaps",
        acceptance_including_null=(accepted/steps).tolist(),
        nonnull_acceptance=(accepted_nonnull/proposed_nonnull).tolist(),
        initial_log_abs_psi=initial_log.tolist(), post_burn_in_log_abs_psi=burnin_log.tolist(),
        first_quarter_mean_log_abs_psi=log_saved[:24].mean(axis=0).tolist(),
        last_quarter_mean_log_abs_psi=log_saved[-24:].mean(axis=0).tolist(),
        lag1_saved_log_abs_psi_correlation=lag1.tolist(),
        unique_MC_rows=len({bytes(row) for row in teacher.astype(np.int8)}),
        mixing_certified=False, measurements_for_training_only=True)
    np.savez_compressed(output/"teacher.npz", chain_rows=saved_rows, chain_amplitudes=saved_amps,
        log_abs_trace=np.asarray(log_trace), training_rows=training_rows,
        training_indices=indices, training_probability=probability)
    save_json(output/"teacher.json", teacher_info)
    print(json.dumps(dict(event="mcmc_teacher", seconds=teacher_seconds,
        mean_nonnull_acceptance=float(np.mean(accepted_nonnull/proposed_nonnull)),
        unique_MC_rows=teacher_info["unique_MC_rows"])), flush=True)
    fit_info, proposals = {}, dict(bridge=bridge)
    training_start = time.monotonic()
    for kind in ("attention", "gru"):
        network = FixedMagnetizationAR(36, kind)
        parameters, info = fit(network, training_rows, probability, indices,
            config["training_total_wall_cap_seconds"]-(time.monotonic()-training_start))
        info["nominal_training_weight_ess_not_independent_MC_ess"] = info.pop("weighted_teacher_ess")
        fit_info[kind] = info
        proposals[kind] = DefensiveLearnedProposal(network, parameters, bridge, .75)
        save_checkpoint(output/(kind+"_proposal.pkl"), parameters, metadata=dict(config=config, fit=info))
        resource_check()
    result = dict(config=config, teacher=teacher_info, training=fit_info, evaluation={})
    rows_by_kind, amps_by_kind, density_by_kind = {}, {}, {}
    for kind, proposal in proposals.items():
        before = time.monotonic()
        rows, density = proposal.draw(512, np.random.default_rng(config["evaluation_seeds"][kind]))
        draw_seconds = time.monotonic()-before
        before = time.monotonic()
        values = amplitudes(rows)
        amp_seconds = time.monotonic()-before
        stats = concentration(values**2/density)
        online = draw_seconds+amp_seconds
        setup = 0. if kind == "bridge" else teacher_seconds+fit_info[kind]["seconds"]
        result["evaluation"][kind] = dict(norm=stats, online_seconds=online,
            draw_and_exact_density_seconds=draw_seconds, exact_amplitude_seconds=amp_seconds,
            effective_norm_records_per_online_second=stats["ess"]/online,
            effective_norm_records_per_second_including_teacher_and_fit=stats["ess"]/(online+setup))
        np.savez_compressed(output/(kind+"_independent.npz"), rows=rows, density=density, amplitude=values)
        rows_by_kind[kind], amps_by_kind[kind], density_by_kind[kind] = rows, values, density
        print(json.dumps(dict(event="evaluation", kind=kind, **result["evaluation"][kind])), flush=True)
        resource_check()
    panel, weights, pq = rows_by_kind["bridge"], amps_by_kind["bridge"]**2/density_by_kind["bridge"], density_by_kind["bridge"]
    result["paired_heldout_log_density_gain"] = {}
    for kind in ("attention", "gru"):
        gain = np.log(proposals[kind].density(panel)/pq)
        result["paired_heldout_log_density_gain"][kind] = float(weights@gain/weights.sum())
    result["total_seconds"] = time.monotonic()-started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_RSS_GiB"] = rss/(1024**3 if sys.platform == "darwin" else 1024**2)
    result["work_counts"] = dict(new_amplitude_centers=teacher_info["exact_amplitude_queries"]+1536,
        new_full_H_centers=0, old_evaluation_records_reused=0, PEPS_parameter_updates=0)
    result["interpretation_limits"] = ["Correlated finite-burn-in teacher is not a certified Born sample",
        "Fresh final law is exact; finite importance ESS does not certify unseen tails",
        "Single fixed follow-up; no checkpoint selection, architecture or optimizer tuning",
        "No full-H, gradient, energy or variance claim from this follow-up"]
    save_json(output/"summary.json", result)
    print(json.dumps(dict(event="complete", seconds=result["total_seconds"], work=result["work_counts"])), flush=True)


if __name__ == "__main__":
    main()
