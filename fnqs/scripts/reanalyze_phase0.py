"""Reanalyze sealed Born audits with the independently checked physical fitter.

Never modifies measurements, checkpoints, source seals, thresholds or budgets.
The window/degree/correction sensitivity grid follows the earlier declaration;
changing its statistical objective is explicitly a post-hoc numerical repair.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from fnqs.artifacts import source_hashes, write_json
from fnqs.scaling import collapse_fit, phase0_gate
from fnqs.staging import assess_stages


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit(rows: list[dict], degree: int = 6, correction: bool = True) -> dict:
    return collapse_fit([r["n"] for r in rows], [r["h"] for r in rows],
                        [r["chi_total"] for r in rows], degree=degree,
                        correction=correction, chi_se=[r["chi_se"] for r in rows])


def _gate(arms: list[dict], sizes: list[int]) -> dict:
    extracted = [arm["extracted"] for arm in arms]
    mean = lambda name: float(np.mean([fit[name] for fit in extracted]))
    gate = phase0_gate(mean("hc"), mean("nu"), mean("eta"),
                       all(a["observable_checks_passed"] for a in arms), sizes, len(arms))
    individual = []
    for arm in arms:
        fit = arm["extracted"]
        individual.append(phase0_gate(fit["hc"], fit["nu"], fit["eta"],
            arm["observable_checks_passed"], sizes, len(arms))["passed"]
            and fit["collapse"]["success"] and not fit["collapse"]["at_bound"])
    gate["checks"]["every_seed_extraction_passes"] = all(individual)
    gate["passed"] = gate["passed"] and all(individual)
    gate["missing"] = [name for name, value in gate["checks"].items() if not value]
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--sensitivity-protocol", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--declare", action="store_true")
    args = parser.parse_args()
    pair_path = args.pair / "protocol.json"
    pair = json.loads(pair_path.read_text())
    sensitivity = json.loads(args.sensitivity_protocol.read_text())
    if sensitivity["paired_protocol_sha256"] != _sha(pair_path):
        raise ValueError("sensitivity declaration belongs to a different paired batch")
    manifest = json.loads(args.analysis_manifest.read_text())
    expected_sources = manifest["source_sha256"]
    declaration_path = args.output / "protocol.json"
    if args.declare:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(declaration_path, {
            "kind": "physical_error_weighted_saved_audit_reanalysis_protocol",
            "posthoc_statistical_repair": True,
            "reason": "Varying nu must not suppress measurement noise by L^(-2/nu).",
            "objective": "fixed_physical_standard_errors",
            "original_primary_fit_window": "abs((h-1)*L)<=config collapse_window_x, default 1",
            "sensitivity_options": sensitivity["options"],
            "paired_protocol_sha256": _sha(pair_path),
            "original_sensitivity_protocol_sha256": _sha(args.sensitivity_protocol),
            "analysis_source_sha256": expected_sources,
            "launcher_sha256": _sha(Path(__file__)),
            "new_training_or_born_draws": 0,
            "changes_accuracy_thresholds": False,
            "checkpoint_selection": False,
            "confidence_intervals_or_class_confirmation": False,
        })
        print({"declared": True, "output": str(args.output)})
        return
    declaration = json.loads(declaration_path.read_text())
    if (declaration["paired_protocol_sha256"] != _sha(pair_path)
        or declaration["launcher_sha256"] != _sha(Path(__file__))
        or declaration["original_sensitivity_protocol_sha256"] != _sha(args.sensitivity_protocol)
        or declaration["analysis_source_sha256"] != source_hashes()):
        raise ValueError("reanalysis implementation, inputs or sources differ from declaration")
    if (args.output / "reanalysis.json").exists():
        raise ValueError("preserve existing results; choose a new versioned output")
    result_arms, evidence, input_hashes = {}, {}, {}
    for name in ("warm", "scratch"):
        path = args.pair / name / "neural.json"
        original = json.loads(path.read_text())
        if (original["source_sha256"] != pair["source_sha256"]
            or original["config"] != pair["configs"][name]
            or [a["seed"] for a in original["arms"]] != original["config"]["seeds"]):
            raise ValueError("incomplete or mismatched sealed measurement batch")
        input_hashes[name] = _sha(path)
        corrected_arms, summaries = [], []
        config = original["config"]
        for arm in original["arms"]:
            if arm["final_step"] != config["steps"]:
                raise ValueError("measurement audit did not complete its declared training budget")
            primary_rows = [r for r in arm["rows"] if r["n"] in config["sizes"]
                and abs((r["h"]-1)*r["n"]) <= config.get("collapse_window_x", 1.)]
            primary_fit = _fit(primary_rows)
            extracted = {"hc": primary_fit["hc"], "nu": primary_fit["nu"],
                         "eta": arm["extracted"]["eta"], "collapse": primary_fit}
            corrected = {"seed": arm["seed"], "warm_start": arm["warm_start"],
                         "extracted": extracted,
                         "observable_checks_passed": arm["observable_checks_passed"]}
            corrected_arms.append(corrected)
            fits = []
            options = declaration["sensitivity_options"]
            for window, degree, correction in itertools.product(
                    options["windows_x"], options["degrees"], options["corrections"]):
                rows = [r for r in arm["rows"] if r["split"] == "held_field"
                        and abs((r["h"]-1)*r["n"]) <= window]
                fits.append({"window_x": window, **_fit(rows, degree, correction)})
            summaries.append({**corrected, "legacy_extracted": arm["extracted"],
                "held_field_sensitivity": fits,
                "sensitivity_nu_range": [min(f["nu"] for f in fits), max(f["nu"] for f in fits)],
                "sensitivity_any_bound": any(f["at_bound"] for f in fits)})
        gate = _gate(corrected_arms, config["sizes"])
        evidence[name] = {"kind": "shared_neural_phase0", "arms": corrected_arms,
                          "phase0_scientific_gate": gate}
        result_arms[name] = {"seeds": summaries, "corrected_scientific_gate": gate,
                            "legacy_scientific_gate": original["phase0_scientific_gate"]}
    write_json(args.output / "reanalysis.json", {
        "kind": "phase0_saved_measurement_reanalysis", "complete": True,
        "posthoc_statistical_repair": True, "arms": result_arms,
        "measurement_source_sha256": pair["source_sha256"],
        "analysis_source_sha256": source_hashes(), "inputs_sha256": input_hashes,
        "protocol_sha256": _sha(declaration_path),
        "measurement_checks_and_costs_unchanged": True,
        "sampling_confidence_intervals": False,
        "readiness": assess_stages(evidence["scratch"]),
        "equal_total_cost_or_foundation_advantage_claim": False,
    })
    print({name: value["corrected_scientific_gate"] for name, value in result_arms.items()})


if __name__ == "__main__":
    main()
