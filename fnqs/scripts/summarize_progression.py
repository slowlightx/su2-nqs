"""Aggregate the sealed paired experiment, costs and research-stage readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fnqs.artifacts import write_json
from fnqs.exact import TFIMReference
from fnqs.staging import assess_stages


def _correlation_bound(row: dict) -> float:
    """Descriptive per-distance error plus two SE, matching the saved gate."""
    reference = TFIMReference(row["n"],row["h"])
    return max(abs(value - reference.correlation(r)) + 2 * row["correlation_se"][r]
               for r,value in enumerate(row["correlations"]))


def _unique_ancestral_costs(arms: list[dict]) -> dict:
    """Count each shared parent stage once across warm stochastic replicas."""
    stages = {}
    for arm in arms:
        origin = arm.get("warm_start")
        while origin:
            identity = origin.get("checkpoint_sha256") or origin.get("parent_sha256")
            work = origin.get("parent_work") or origin.get("parent_training_work")
            if not identity or work is None:
                raise ValueError("incomplete parent identity/cost record")
            if identity in stages and stages[identity]["work"] != work:
                raise ValueError("inconsistent shared-parent cost")
            stages[identity] = {"checkpoint":origin.get("checkpoint") or origin.get("parent"),"work":work}
            origin = origin.get("parent_origin")
    totals = {name:sum(stage["work"].get(name, 0) for stage in stages.values())
              for name in ["gradient_hamiltonians","hamiltonian_evaluations","basis_state_evaluations","born_samples"]}
    return {"unique_stages":stages,"totals":totals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.pair / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    reference = json.loads(args.references.read_text())
    reference_modules = ("basis.py","hamiltonians.py","exact.py","observables.py","sectors.py")
    if any(reference["source_sha256"][name] != protocol["source_sha256"][name]
           for name in reference_modules):
        raise ValueError("independent reference's physics sources differ")
    status_path = args.pair / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    results, summaries, hashes = {}, {}, {}
    for name in ["warm","scratch"]:
        path = args.pair / name / "neural.json"
        if not path.exists():
            summaries[name] = {"state":status.get(name,"running"),"completed_checkpoint_seeds":
                               [seed for seed in protocol["configs"][name]["seeds"]
                                if (args.pair / name / f"seed_{seed}.npz").exists()]}
            continue
        data = json.loads(path.read_text())
        if data["source_sha256"] != protocol["source_sha256"] or data["config"] != protocol["configs"][name]:
            raise ValueError("paired result differs from its predeclared protocol")
        results[name] = data; hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        seeds = []
        for arm in data["arms"]:
            if arm["final_step"] != 1000 or arm["training_work"]["gradient_hamiltonians"] != 6000:
                raise ValueError("paired experiment did not complete its declared budget")
            held = [row for row in arm["rows"] if row["split"] == "held_field"]
            seeds.append({"seed":arm["seed"],"observable_checks_passed":arm["observable_checks_passed"],
                "max_held_energy_relative_error":max(row["energy_relative_error"] for row in held),
                "max_held_chi_relative_error":max(row["chi_relative_error"] for row in held),
                "max_held_correlation_abs_error":max(row["max_correlation_abs_error"] for row in held),
                "max_held_energy_bound":max(row["energy_relative_error"] +
                    2 * row["energy_se"] / abs(TFIMReference(row["n"],row["h"]).energy()) for row in held),
                "max_held_chi_bound":max(row["chi_relative_error"] +
                    2 * row["chi_se"] / row["chi_reference"] for row in held),
                "max_held_correlation_bound":max(_correlation_bound(row) for row in held),
                "extracted":arm["extracted"],"last_training_diagnostics":arm["history"][-1],
                "work":arm["training_work"],
                "audit_born_draws":sum(row["audit_samples"] for row in arm["rows"])})
        summaries[name] = {"state":"complete","seeds":seeds,
            "phase0_scientific_gate":data["phase0_scientific_gate"],
            "elapsed_seconds":data["elapsed_seconds"],
            "ancestral_costs":_unique_ancestral_costs(data["arms"])}
    complete = len(results) == 2
    output = {"kind":"phase0_pair_progression_summary","complete":complete,"arms":summaries,
        "source_sha256":protocol["source_sha256"],"inputs_sha256":hashes,
        "protocol_sha256":hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "reference_sha256":hashlib.sha256(args.references.read_bytes()).hexdigest(),
        "launcher_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "readiness":assess_stages(results.get("scratch",{})),
        "warm_replicas_not_independent_pretraining_seeds":True,
        "equal_total_cost_or_foundation_advantage_claim":False,
        "phase1_reference_complete":True,"phase1_class_confirmation":False,
        "phase2_transfer_experiments_started":False}
    write_json(args.output, output)
    print({"complete":complete,"arm_status":{key:value["state"] for key,value in summaries.items()},
           "readiness":output["readiness"]})


if __name__ == "__main__":
    main()
