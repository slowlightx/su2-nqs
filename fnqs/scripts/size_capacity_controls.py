"""Sealed development controls for size sharing, optimization and GRU width.

Fixed models receive exactly the same two fields per size/epoch as shared
models. Their aggregate Hamiltonian/Born budget matches a shared endpoint.
Neither fixed-model aggregation nor this development grid establishes a class.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import jax
import numpy as np

from fnqs.artifacts import load_checkpoint, save_checkpoint, source_hashes, write_json
from fnqs.exact import TFIMReference
from fnqs.experiments import _sampled_audit_values, provenance
from fnqs.hamiltonians import IsingCouplings
from fnqs.initialization import initialize_run
from fnqs.models import GRUConfig, sample
from fnqs.observables import quantum_geometric_tensor, sampled_correlations
from fnqs.training import Task, train


def launcher_hash() -> str:
    """Identify orchestration separately from the unchanged numerical library."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def parameter_count(width: int) -> int:
    """Count all GRU parameters, including the two output biases."""
    GRUConfig(width)
    return 3 * width * width + 44 * width + 2


def field_schedule(seed: int, steps: int, sizes: list[int], fields: list[float]) -> np.ndarray:
    """Two distinct fields per size; longer schedules preserve every prefix."""
    if steps < 1 or len(fields) < 2:
        raise ValueError("positive steps and at least two fields required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x51CE]))
    return np.array([[rng.choice(len(fields), 2, replace=False) for _ in sizes]
                     for _ in range(steps)], dtype=np.int64)


def build_protocol(seeds: list[int], base_steps: int = 1000, audit_samples: int = 8192) -> dict:
    """Build a crossed width/budget experiment and an aggregate-budget control."""
    if (not seeds or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
        or type(base_steps) is not int or base_steps < 1
        or type(audit_samples) is not int or audit_samples < 2):
        raise ValueError("valid unique uint32 seeds and positive budgets required")
    sizes = [16, 24, 32]
    fields = [round(.95 + .005 * i, 8) for i in range(21)]
    development = [round(.952 + .005 * i, 8) for i in range(20)]
    confirmation = [round(.9509375 + .0025 * i, 8) for i in range(40)]
    if set(fields) & set(development) or set(confirmation) & set(fields + development):
        raise ValueError("field leakage")
    arms = {f"fixed16_L{n}": {"sizes": [n], "width": 16, "milestones": [base_steps]}
            for n in sizes}
    arms.update({f"shared{width}": {"sizes": sizes, "width": width,
                 "milestones": [base_steps, 2 * base_steps]} for width in [16, 32]})
    options = {"mode": "vmc", "optimizer": "sr", "learning_rate": .5,
               "n_samples": 128, "sr_shift": 1e-4, "sr_solver": "cg",
               "sr_tolerance": 1e-7, "sr_maxiter": 5000,
               "sr_max_step_norm": .25, "sr_backtracking": False}
    for arm in arms.values():
        tasks = 2 * len(arm["sizes"])
        arm["tasks_per_epoch"] = tasks
        arm["parameter_count"] = parameter_count(arm["width"])
        arm["endpoint_costs"] = {str(step): {
            "gradient_hamiltonians": step * tasks,
            "hamiltonian_evaluations": step * tasks,
            "born_samples": step * tasks * options["n_samples"]}
            for step in arm["milestones"]}
    return {"kind": "size_capacity_development_control_protocol",
            "seeds": seeds, "sizes": sizes, "train_fields": fields,
            "development_fields": development, "reserved_confirmation_fields": confirmation,
            "base_steps": base_steps, "audit_samples": audit_samples,
            "arms": arms, "training_options": options,
            "cold_initialization_only": True, "two_fields_per_size_each_epoch": True,
            "fixed_aggregate_parameter_count": 3 * parameter_count(16),
            "shared32_parameter_count": parameter_count(32),
            "fairness_limits": "Fixed models have independent parameters/updates. Aggregate field exposure and H/Born budgets match shared16 at base_steps; shared32 has 60 more parameters than the fixed aggregate. Wider score/Jacobian/CG arithmetic is not equal wall-clock cost. Doubling epochs changes budget, not the optimizer or trajectory initialization.",
            "interpretation": "Paired seed/field development comparisons only; a width effect can involve optimization conditioning as well as expressivity. No causal attribution to capacity without convergence evidence.",
            "method_selection_uses_development_only": True,
            "confirmation_measured": False, "universality_established": False,
            "source_sha256": source_hashes(), "launcher_sha256": launcher_hash()}


def declare(output: Path, seeds: list[int], base_steps: int, audit_samples: int) -> None:
    """Seal schedules and costs before profiling/training touches observables."""
    output.mkdir(parents=True, exist_ok=False)
    protocol = {**provenance({}), **build_protocol(seeds, base_steps, audit_samples)}
    schedules = {}
    for seed in seeds:
        schedule = field_schedule(seed, 2 * base_steps, protocol["sizes"], protocol["train_fields"])
        path = output / f"schedule_{seed}.npy"
        np.save(path, schedule, allow_pickle=False)
        schedules[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol["schedule_sha256"] = schedules
    write_json(output / "protocol.json", protocol)


def check_runtime(protocol: dict) -> None:
    """Fail rather than silently label running old code with changed hashes."""
    if protocol["source_sha256"] != source_hashes() or protocol["launcher_sha256"] != launcher_hash():
        raise ValueError("library or control runner changed since declaration")


def selected_tasks(protocol: dict, arm: dict, schedule: np.ndarray, epoch: int) -> list[Task]:
    """Return the sealed epoch's Pauli TFIM tasks in deterministic size order."""
    return [Task(n, IsingCouplings(h=protocol["train_fields"][int(index)]), "A")
            for n in arm["sizes"]
            for index in schedule[epoch, protocol["sizes"].index(n)]]


def advance(params: dict, state: dict, key: jax.Array, model: GRUConfig,
            protocol: dict, arm: dict, schedule: np.ndarray, steps: int) -> tuple:
    """Advance the accepted trajectory without consuming training PRNG in audits."""
    if type(steps) is not int or steps < 1 or int(state["step"]) + steps > len(schedule):
        raise ValueError("advance must stay within the declared field schedule")
    history = []
    for _ in range(steps):
        epoch = int(state["step"])
        tasks = selected_tasks(protocol, arm, schedule, epoch)
        params, state, key, records = train(params, tasks, model, key, state=state,
                                           steps=1, **protocol["training_options"])
        records[0]["field_schedule_epoch"] = epoch
        history.extend(records)
    return params, state, key, history


@partial(jax.jit, static_argnames=("count", "n", "model"))
def compiled_sample(params: dict, key: jax.Array, count: int, n: int,
                    vector: jax.Array, model: GRUConfig) -> jax.Array:
    """Compile the unchanged sampler once per shape; parameters/fields are dynamic."""
    return sample(params, key, count, n, vector, model)


def audit_row(params: dict, model: GRUConfig, n: int, h: float,
              key: jax.Array, count: int) -> dict:
    """Same Born/QGT/bootstrap definitions as the existing sampled audit."""
    vector = jax.numpy.asarray(IsingCouplings(h=h).vector())
    bits = compiled_sample(params, key, count, n, vector, model)
    values = [_sampled_audit_values(params, bits[start:start+64], vector, model)
              for start in range(0, count, 64)]
    local = np.concatenate([np.asarray(value[0]) for value in values])
    scores = np.concatenate([np.asarray(value[1]) for value in values])
    metric = quantum_geometric_tensor(scores).real
    correlations, errors = sampled_correlations(np.asarray(bits))
    ref = TFIMReference(n, h)
    rng = np.random.default_rng(int(np.asarray(key)[-1]))
    bootstrap = [np.var(scores[rng.integers(count, size=count), 3]) for _ in range(200)]
    return {"n": n, "h": h, "energy_total": float(local.mean()),
            "energy_se": float(local.std(ddof=1) / np.sqrt(count)),
            "energy_relative_error": abs(float(local.mean())-ref.energy()) / abs(ref.energy()),
            "variance_total": float(local.var()), "correlations": correlations.tolist(),
            "correlation_se": errors.tolist(), "chi_total": float(metric[3, 3]),
            "chi_se": float(np.std(bootstrap, ddof=1)),
            "chi_reference": ref.fidelity_susceptibility(),
            "chi_relative_error": abs(float(metric[3, 3])-ref.fidelity_susceptibility()) / ref.fidelity_susceptibility(),
            "max_correlation_abs_error": float(max(abs(correlations[r]-ref.correlation(r)) for r in range(n))),
            "coupling_metric": metric.tolist(), "audit_samples": count,
            "audit_independent_of_training": True}


def run_arm(output: Path, name: str, seed: int, profile_steps: int | None = None) -> None:
    """Run/resume an arm, retaining both prefix endpoints and strict provenance."""
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    check_runtime(protocol)
    if name not in protocol["arms"] or seed not in protocol["seeds"]:
        raise ValueError("arm/seed not declared")
    schedule_path = output / f"schedule_{seed}.npy"
    if hashlib.sha256(schedule_path.read_bytes()).hexdigest() != protocol["schedule_sha256"][str(seed)]:
        raise ValueError("field schedule changed")
    schedule = np.load(schedule_path, allow_pickle=False)
    if profile_steps is not None and (type(profile_steps) is not int or not 1 <= profile_steps <= len(schedule)):
        raise ValueError("profile steps must fit the declared schedule")
    arm = protocol["arms"][name]
    model = GRUConfig(arm["width"], 64)
    directory = output / ("profiles" if profile_steps else "runs") / name / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    signature = {"arm": arm, "seed": seed, "training_options": protocol["training_options"],
                 "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                 "schedule_sha256": protocol["schedule_sha256"][str(seed)]}
    accepted = directory / "accepted.npz"
    history = []
    if accepted.exists():
        if profile_steps:
            raise FileExistsError("profiling must use a fresh declared output")
        params, state, key, metadata = load_checkpoint(accepted)
        if (metadata["signature"] != signature or metadata["source_sha256"] != source_hashes()
            or metadata["launcher_sha256"] != launcher_hash()):
            raise ValueError("resume signature/source differs")
        history = metadata["history"]
    else:
        params, state, key, origin = initialize_run(seed, model)
        assert origin is None
    started = time.monotonic()
    def save(path: Path) -> None:
        check_runtime(protocol)
        save_checkpoint(path, params, state, key, {
            "signature": signature, "seed": seed, "model": asdict(model),
            "source_sha256": source_hashes(), "launcher_sha256": launcher_hash(),
            "history": history, "warm_start": None})
    def measure(step: int) -> None:
        endpoint = directory / f"step_{step}"
        result_path = endpoint / "audit.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            if (existing["signature"] != signature
                or existing["source_sha256"] != protocol["source_sha256"]
                or existing["launcher_sha256"] != protocol["launcher_sha256"]
                or existing["checkpoint_sha256"] != hashlib.sha256((endpoint / "checkpoint.npz").read_bytes()).hexdigest()):
                raise ValueError("incompatible completed endpoint")
            return
        p, s, k, _ = load_checkpoint(endpoint / "checkpoint.npz")
        expected = arm["endpoint_costs"][str(step)]
        if any(s["work"][name] != value for name, value in expected.items()):
            raise ValueError("endpoint cost mismatch")
        rows = []
        audit_key = jax.random.fold_in(k, 0xA0D17)
        points = [(n, h) for n in arm["sizes"]
                  for h in sorted(protocol["train_fields"] + protocol["development_fields"])]
        for i, (n, h) in enumerate(points):
            row = audit_row(p, model, n, h, jax.random.fold_in(audit_key, i), protocol["audit_samples"])
            row["split"] = "train" if h in protocol["train_fields"] else "development"
            rows.append(row)
        check_runtime(protocol)
        write_json(result_path, {"kind": "size_capacity_development_endpoint", "arm": name,
            "seed": seed, "step": step, "signature": signature, "model": asdict(model),
            "source_sha256": source_hashes(), "launcher_sha256": launcher_hash(),
            "checkpoint_sha256": hashlib.sha256((endpoint / "checkpoint.npz").read_bytes()).hexdigest(),
            "parameter_count": sum(value.size for value in jax.tree.leaves(p)),
            "training_work": s["work"], "last_training_diagnostics":
            next(record for record in reversed(history) if record["step"] == step),
            "rows": rows, "audit_born_draws": sum(row["audit_samples"] for row in rows),
            "reserved_confirmation_measured": False, "universality_established": False})
        print(f"AUDIT complete arm={name} seed={seed} step={step}", flush=True)
    milestones = [profile_steps] if profile_steps else arm["milestones"]
    for milestone in milestones:
        if not profile_steps and (directory / f"step_{milestone}" / "checkpoint.npz").exists():
            measure(milestone)
        while int(state["step"]) < milestone:
            try:
                params, state, key, records = advance(params, state, key, model, protocol, arm, schedule, 1)
            except Exception as error:
                save(accepted)
                write_json(directory / "failure.json", {"kind": "stopped_control",
                    "error": repr(error), "last_accepted_step": int(state["step"]),
                    "additional_failed_attempt_hamiltonians": arm["tasks_per_epoch"],
                    "additional_failed_attempt_born_samples": arm["tasks_per_epoch"] * protocol["training_options"]["n_samples"],
                    "source_sha256": source_hashes(), "launcher_sha256": launcher_hash()})
                raise
            history.extend(records)
            step = int(state["step"])
            if step % 25 == 0 or step == 1:
                print(f"arm={name} seed={seed} step={step} E/N={records[-1]['energy_per_site']:.8f}", flush=True)
            if step % 100 == 0:
                save(accepted)
        save(accepted)
        if profile_steps:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            write_json(directory / "profile.json", {"kind": "control_resource_profile",
                "elapsed_seconds": time.monotonic()-started,
                "peak_rss_bytes": rss if sys.platform == "darwin" else 1024*rss,
                "signature": signature, "source_sha256": source_hashes(),
                "launcher_sha256": launcher_hash(), "training_work": state["work"],
                "neural_accuracy_audited": False})
        else:
            endpoint = directory / f"step_{milestone}"
            if not (endpoint / "checkpoint.npz").exists():
                save(endpoint / "checkpoint.npz")
            measure(milestone)
    write_json(directory / "complete.json", {"kind": "completed_control_trajectory",
        "profile_only": profile_steps is not None, "signature": signature,
        "source_sha256": source_hashes(), "launcher_sha256": launcher_hash(),
        "elapsed_seconds_this_invocation": time.monotonic()-started})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--declare", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[71, 83, 97])
    parser.add_argument("--base-steps", type=int, default=1000)
    parser.add_argument("--audit-samples", type=int, default=8192)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--profile-steps", type=int)
    args = parser.parse_args()
    if args.declare:
        declare(args.output, args.seeds, args.base_steps, args.audit_samples)
        return
    protocol = json.loads((args.output / "protocol.json").read_text())
    for name in args.arms or protocol["arms"]:
        for seed in args.seeds:
            run_arm(args.output, name, seed, args.profile_steps)


if __name__ == "__main__":
    main()
