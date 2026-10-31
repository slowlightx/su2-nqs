"""Bounded fresh-data comparison of global attention and GRU PEPS proposals.

Run from the repository root with PYTHONPATH=peps-vmc and JAX_PLATFORM_NAME=cpu.
The fixed endpoint and original source manifest are read only. No energy
optimization occurs; all reported energies are small engineering diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path
import resource
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from experiments.learned_proposals import DefensiveLearnedProposal, FixedMagnetizationAR
from peps_vmc.contraction import make_heisenberg_action, make_square_torus_plan
from peps_vmc.diagnostics import paired_moments
from peps_vmc.model import SquareGraphSU2PEPS
from peps_vmc.proposals import CoveringBridgeProposal, torus_coverings
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint


def save_json(path: Path, value: dict) -> None:
    """Write finite small audit metadata."""
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")


def concentration(weights: np.ndarray) -> dict:
    """Finite-panel ESS; not a certificate against unseen importance tails."""
    weights = np.asarray(weights, np.float64)
    return dict(ess=float(weights.sum()**2/(weights@weights)),
                maximum_weight_fraction=float(weights.max()/weights.sum()),
                mean=float(weights.mean()), records=len(weights))


def tree_count(parameters: dict) -> int:
    return sum(int(x.size) for x in jax.tree.leaves(parameters))


def fit(model: FixedMagnetizationAR, teacher: np.ndarray, probability: np.ndarray,
        indices: np.ndarray, seconds_limit: float) -> tuple[dict, dict]:
    """Fixed-step Adam MLE; no evaluation-based checkpoint choice."""
    start = time.monotonic()
    parameters = model.initialize(29102)
    zero = jax.tree.map(jnp.zeros_like, parameters)

    def objective(p, rows):
        nll = -jnp.mean(model.symmetric_log_probability(p, rows))
        reg = 1e-4*sum(jnp.sum(x*x) for x in jax.tree.leaves(p))
        return nll+reg, nll

    @jax.jit
    def step(p, m, v, count, rows):
        (loss, nll), gradient = jax.value_and_grad(objective, has_aux=True)(p, rows)
        norm = jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(gradient)))
        gradient = jax.tree.map(lambda g: g*jnp.minimum(1., 2./(norm+1e-12)), gradient)
        count = count+1
        m = jax.tree.map(lambda a, g: .9*a+.1*g, m, gradient)
        v = jax.tree.map(lambda a, g: .999*a+.001*g*g, v, gradient)
        p = jax.tree.map(lambda x, a, b: x-.003*(a/(1-.9**count))/(jnp.sqrt(b/(1-.999**count))+1e-8), p, m, v)
        return p, m, v, count, loss, nll

    m, v = zero, zero
    count = jnp.asarray(0)
    trace = []
    for i, selection in enumerate(indices):
        parameters, m, v, count, loss, nll = step(parameters, m, v, count, jnp.asarray(teacher[selection]))
        if i == 0 or (i+1) % 50 == 0:
            jax.block_until_ready(parameters)
            trace.append(dict(step=i+1, loss=float(loss), batch_nll=float(nll), seconds=time.monotonic()-start))
            print(json.dumps(dict(event="fit", kind=model.kind, **trace[-1])), flush=True)
        if time.monotonic()-start > seconds_limit:
            raise TimeoutError("predeclared total proposal-training wall budget exceeded")
    jax.block_until_ready(parameters)
    return parameters, dict(kind=model.kind, train_steps=len(indices), batch_size=indices.shape[1],
        real_parameters=tree_count(parameters), seconds=time.monotonic()-start, trace=trace,
        fixed_checkpoint=True, weighted_teacher_ess=float(1/(probability@probability)))


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path("peps-vmc/results/attention_sampling_20261029"))
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a fresh output; existing evidence is immutable")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    endpoint = Path("peps-vmc/results/triangular_6x6_j2_0125_local_20261029/frozen_endpoint.pkl")
    original_registration = json.loads(endpoint.with_name("registration.json").read_text())
    c = original_registration["config"]
    config = dict(teacher_records=1536, independent_records_per_proposal=512,
        full_H_centers_per_proposal=32, train_steps=args.steps, batch_size=128,
        teacher_seed=29101, parameter_seed=29102, train_resampling_seed=29103,
        evaluation_seeds=dict(bridge=29110, attention=29111, gru=29112),
        learned_fraction=.75, teacher_target_smoothing=.1, learning_rate=.003,
        l2_coefficient=1e-4, training_total_wall_cap_seconds=480,
        endpoint_sha256=hashlib.sha256(endpoint.read_bytes()).hexdigest(),
        checkpoint_policy="fixed final update; no heldout selection",
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path(__file__).with_name("learned_proposals.py"))},
        evaluation_scope="small engineering probe; no competitive energy/variance or unseen-tail certificate")
    save_json(args.output/"registration.json", config)
    bonds = triangular_j1j2_bonds(6, 6, j1=1., j2=.125, periodic=True)
    pairs = np.asarray([(a, b) for a, b, _ in bonds], np.int32)
    couplings = np.asarray([v for _, _, v in bonds])
    bridge = CoveringBridgeProposal(torus_coverings(6, 6, count=c["coverings"], seed=c["covering_seed"]),
                                   pairs, np.abs(couplings)/np.abs(couplings).sum(), epsilon=c["uniform_fraction"])
    model = SquareGraphSU2PEPS(6, 6)
    plan = make_square_torus_plan(6, 6, 3)
    parameters = load_checkpoint(endpoint)["params"]
    tensors = model.tensors(jnp.asarray(parameters["w"]))
    amp = jax.jit(plan.amplitude)
    def amplitudes(rows):
        return np.asarray([float(amp(tensors, jnp.asarray(row))) for row in rows])
    prep = time.monotonic()
    teacher, tq = bridge.draw(config["teacher_records"], np.random.default_rng(config["teacher_seed"]))
    ta = amplitudes(teacher)
    teacher_seconds = time.monotonic()-prep
    weights = ta**2/tq
    probability = .9*weights/weights.sum()+.1/len(weights)
    indices = np.random.default_rng(config["train_resampling_seed"]).choice(
        len(teacher), size=(args.steps, config["batch_size"]), p=probability)
    np.savez_compressed(args.output/"teacher.npz", rows=teacher, density=tq, amplitude=ta,
                        resampling_probability=probability, training_indices=indices)
    teacher_info = dict(seconds=teacher_seconds, norm=concentration(weights),
        nonzero_amplitudes=int(np.count_nonzero(ta)),
        exact_global_spin_flip_average=True, teacher_only=True)
    save_json(args.output/"teacher.json", teacher_info)
    print(json.dumps(dict(event="teacher", **teacher_info)), flush=True)
    proposals = dict(bridge=bridge)
    fit_info = {}
    training_start = time.monotonic()
    for kind in ("attention", "gru"):
        network = FixedMagnetizationAR(36, kind)
        p, fit_info[kind] = fit(network, teacher, probability, indices,
            config["training_total_wall_cap_seconds"]-(time.monotonic()-training_start))
        proposals[kind] = DefensiveLearnedProposal(network, p, bridge, learned_fraction=.75)
        save_checkpoint(args.output/(kind+"_proposal.pkl"), p, metadata=dict(config=config, fit=fit_info[kind]))
    save_json(args.output/"fit.json", fit_info)
    result = dict(config=config, teacher=teacher_info, training=fit_info, evaluation={})
    all_rows = {}
    all_amplitudes = {}
    all_density = {}
    for kind, proposal in proposals.items():
        draw_start = time.monotonic()
        rows, density = proposal.draw(config["independent_records_per_proposal"],
                                     np.random.default_rng(config["evaluation_seeds"][kind]))
        draw_seconds = time.monotonic()-draw_start
        eval_start = time.monotonic()
        values = amplitudes(rows)
        eval_seconds = time.monotonic()-eval_start
        stats = concentration(values**2/density)
        online_seconds = draw_seconds+eval_seconds
        setup_seconds = 0. if kind == "bridge" else teacher_seconds+fit_info[kind]["seconds"]
        result["evaluation"][kind] = dict(norm=stats, draw_and_exact_density_seconds=draw_seconds,
            exact_amplitude_seconds=eval_seconds, online_seconds=online_seconds,
            effective_norm_records_per_online_second=stats["ess"]/online_seconds,
            effective_norm_records_per_second_including_teacher_and_fit=stats["ess"]/(online_seconds+setup_seconds),
            total_cost_assigned_teacher_seconds=0. if kind == "bridge" else teacher_seconds,
            analytic_uniform_floor=float(proposal.uniform_floor) if kind != "bridge" else
                float(bridge.epsilon/comb(36, 18)))
        all_rows[kind], all_amplitudes[kind], all_density[kind] = rows, values, density
        np.savez_compressed(args.output/(kind+"_independent.npz"), rows=rows, density=density, amplitude=values)
        print(json.dumps(dict(event="evaluation", kind=kind, **result["evaluation"][kind])), flush=True)
    # Same fresh independent baseline panel for likelihood diagnostics only.
    # The panel is never consulted during optimization or checkpoint choice.
    panel, panel_a, panel_q = all_rows["bridge"], all_amplitudes["bridge"], all_density["bridge"]
    panel_weights = panel_a**2/panel_q
    result["paired_heldout_log_density_gain"] = {}
    for kind in ("attention", "gru"):
        gain = np.log(proposals[kind].density(panel)/panel_q)
        value = float(panel_weights@gain/panel_weights.sum())
        result["paired_heldout_log_density_gain"][kind] = dict(mean=value,
            interpretation="target-weighted ratio estimate from fresh baseline panel; large teacher/evaluation tail uncertainty")
    # Exact full H action retains node contributions |H psi|²/q. The original
    # bridge is part of each learned proposal and supplies a strict floor.
    action_start = time.monotonic()
    action = make_heisenberg_action(plan, pairs.tolist(), couplings, max_intermediate_elements=20_000_000)
    hp = jax.jit(action.products)
    result["full_H_engineering_audit"] = {}
    for kind in proposals:
        begin = time.monotonic()
        rows = all_rows[kind][:32]
        density = all_density[kind][:32]
        products = np.asarray([np.asarray(hp(tensors, jnp.asarray(row))) for row in rows])
        # Reuse the validated two-endpoint diagnostic with identical endpoints;
        # discard its vacuous pair difference and report only one endpoint.
        paired = paired_moments(np.repeat(products[:, None, :], 2, axis=1), density,
                                n_sites=36, groups=4)
        audit = dict(metric_order=paired["metric_order"], values=paired["endpoint_values"][0],
            group_jackknife_SE=paired["group_jackknife_SE"][0], records=len(rows), groups=4,
            norm_ess=paired["norm_concentration"]["effective_contributions"][0],
            H2_ess=paired["H2_concentration"]["effective_contributions"][0],
            target_node_records=paired["target_node_records"][0],
            target_node_H2_contributions=paired["target_node_H2_contributions"][0])
        result["full_H_engineering_audit"][kind] = dict(seconds=time.monotonic()-begin,
            metrics=audit, engineering_only=True, includes_node_H2=True)
        np.savez_compressed(args.output/(kind+"_full_H.npz"), rows=rows, density=density, products=products)
        print(json.dumps(dict(event="full_H", kind=kind, seconds=time.monotonic()-begin)), flush=True)
    result["full_H_total_seconds"] = time.monotonic()-action_start
    result["total_seconds"] = time.monotonic()-started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_RSS_GiB"] = rss/(1024**3 if sys.platform == "darwin" else 1024**2)
    result["work_counts"] = dict(new_amplitude_centers=len(teacher)+sum(len(x) for x in all_rows.values()),
        new_full_H_centers=96, old_pilot_audit_records_reused=0, PEPS_parameter_updates=0)
    result["interpretation_limits"] = ["Single endpoint, seed, fixed budget and small panels",
        "Finite-panel ESS does not certify global mode coverage or variance",
        "H audit has 32 centers per method; no gradient or energy improvement claim",
        "Proposal symmetry is not required; exact target PEPS singlet is unchanged",
        "Attention and GRU have similar parameter budgets but different compute costs"]
    save_json(args.output/"summary.json", result)
    print(json.dumps(dict(event="complete", seconds=result["total_seconds"], work=result["work_counts"])), flush=True)


if __name__ == "__main__":
    main()
