"""Predeclared Phase 0 optimizer, field-coverage and size-sharing controls.

Run with ``python -m fnqs.controls --output ...``. Development audit points
are for method comparison; a separate, predeclared grid is reserved for later
confirmation. Small-size success never satisfies the universality gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import write_json
from .experiments import provenance, run_neural


def control_configs(steps: int = 1200, seeds: tuple[int, ...] = (17,)) -> dict[str, dict]:
    """Same width, initialization seeds and five Hamiltonian queries/update."""
    sparse = [round(.8 + .1*i, 8) for i in range(5)]
    dense = [round(.8 + .02*i, 8) for i in range(21)]
    development = [round(.81 + .02*i, 8) for i in range(20)]
    configs = {}
    for sizes, scope in [([8], "fixed"), ([4, 6, 8], "shared")]:
        coverages = [(sparse, "sparse"), (dense, "dense")] if scope == "fixed" else [(dense, "dense")]
        for fields, coverage in coverages:
            for optimizer in ["adam", "sr"]:
                config = {"model": {"hidden_dim": 16, "size_scale": 64.},
                          "sizes": sizes, "train_fields": fields, "test_fields": development,
                          "seeds": list(seeds), "steps": steps,
                          "learning_rate": .003 if optimizer == "adam" else .5,
                          "mode": "exact", "optimizer": optimizer, "sr_shift": 1e-4,
                          "samples_per_task": 128, "audit_samples": 1024,
                          "tasks_per_step": 5, "sr_solver": "cg", "sr_tolerance": 1e-8,
                          "sr_maxiter": 2000, "sr_max_step_norm": .5,
                          "sr_backtracking": optimizer == "sr"}
                configs[f"{scope}_{coverage}_{optimizer}"] = config
    return configs


def summarize_control(result: dict) -> dict:
    """Keep train/development metrics separate, including worst-seed accuracy."""
    summary = {"phase0_scientific_gate": result["phase0_scientific_gate"],
               "elapsed_seconds": result["elapsed_seconds"],
               "gradient_hamiltonians_per_seed": result["gradient_hamiltonians_per_seed"],
               "parameter_count": 1474, "seeds": []}
    for arm in result["arms"]:
        entry = {"seed": arm["seed"], "observable_checks_passed": arm["observable_checks_passed"],
                 "training_work": arm["training_work"],
                 "final_training_diagnostics": arm["history"][-1]}
        for split in ["train", "held_field"]:
            rows = [r for r in arm["rows"] if r["split"] == split]
            entry[split] = {f"max_{name}": max(r[name] for r in rows) for name in [
                "energy_relative_error", "chi_relative_error", "max_correlation_abs_error",
                "infidelity", "variance_total", "norm_error", "ad_fidelity_abs_discrepancy"]}
        summary["seeds"].append(entry)
    return summary


def plot_controls(output: Path, results: dict[str, dict]) -> None:
    """Standalone curves with exact references and explicitly finite sizes."""
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"fnqs-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    reference = None
    for name, result in results.items():
        arm = result["arms"][0]
        rows = sorted([r for r in arm["rows"] if r["n"] == 8], key=lambda r: r["h"])
        fields = [r["h"] for r in rows]
        axes[0].plot(fields, [r["chi_total"] for r in rows], label=name, linewidth=1.1)
        axes[1].semilogy(fields, [max(r["energy_relative_error"], 1e-16) for r in rows], linewidth=1.1)
        history = arm["history"]
        axes[2].semilogy([r["step"] for r in history], [max(r["variance_per_site"], 1e-16) for r in history], linewidth=1.1)
        if reference is None or len(rows) > len(reference):
            reference = rows
    if reference:
        axes[0].plot([r["h"] for r in reference], [r["chi_reference"] for r in reference], "k--", label="exact L=8")
    axes[0].set(xlabel="h", ylabel="total fidelity susceptibility", title="Finite L=8 response; no exponent claim")
    axes[1].set(xlabel="h", ylabel="relative energy error", title="Energy accuracy at fixed trained parameters")
    axes[2].set(xlabel="updates (5 gradient Hamiltonians/update)", ylabel="active-task mean variance/site",
                title="Training diagnostics, before update")
    axes[0].legend(fontsize=7)
    fig.savefig(output/"controls.png", dpi=180)
    plt.close(fig)


def run_controls(output: Path, steps: int = 1200, seeds: tuple[int, ...] = (17,),
                 arms: list[str] | None = None) -> dict:
    """Execute immutable controls, reusing only completed matching artifacts."""
    from .configuration import validate_config
    configs = control_configs(steps, seeds)
    if arms is not None:
        if not arms or len(set(arms)) != len(arms) or set(arms)-configs.keys():
            raise ValueError("unknown or repeated control arms")
        configs = {name: configs[name] for name in arms}
    for config in configs.values():
        validate_config("train", config)
    protocol = {**provenance({"arms": configs}), "kind": "phase0_development_controls",
                "confirmation_fields": [round(.805+.01*i, 8) for i in range(40)],
                "budget": "5 gradient Hamiltonians/update for every arm; exact Born sums; SR backtracking queries counted separately",
                "fairness_limits": "Same primary gradient queries, updates and parameters. Different L changes enumerated-state cost; SR adds scores/CG/backtracking queries. All work counts saved; wall time includes audit. This is an optimizer/coverage control, not a matched-total-cost foundation advantage test.",
                "selection": "Development fields may inform method choice; confirmation fields and new seeds require a separate run.",
                "universality_established": False}
    output.mkdir(parents=True, exist_ok=True)
    path = output/"protocol.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["config"] != protocol["config"] or saved["source_sha256"] != protocol["source_sha256"]:
            raise ValueError("control protocol or numerical sources changed; use a new output directory")
    else:
        write_json(path, protocol)
    results, summary = {}, {"kind": "phase0_development_controls", "arms": {},
                            "universality_established": False, "source_sha256": protocol["source_sha256"]}
    for name, config in configs.items():
        directory = output/name
        write_json(output/f"{name}.json", config)
        completed = directory/"neural.json"
        if completed.exists():
            result = json.loads(completed.read_text())
            if result["config"] != config or result["source_sha256"] != protocol["source_sha256"]:
                raise ValueError(f"incompatible completed arm: {name}")
        else:
            print(f"CONTROL {name}", flush=True)
            result = run_neural(config, directory)
        results[name] = result
        summary["arms"][name] = summarize_control(result)
        write_json(output/"summary.json", summary)
        plot_controls(output, results)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--arms", nargs="+")
    args = parser.parse_args()
    result = run_controls(args.output, args.steps, tuple(args.seeds), args.arms)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
