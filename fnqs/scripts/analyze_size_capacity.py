"""Validate and summarize the sealed size/width/budget development controls.

No training, checkpoint selection, exponent fitting or confirmation queries.
Positive paired reductions mean that the second endpoint has smaller error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fnqs.artifacts import source_hashes, write_json
from fnqs.exact import TFIMReference


METRICS = ("chi_max_relative", "chi_max_relative_plus_2se", "chi_rms_relative",
           "correlation_max_absolute", "correlation_max_absolute_plus_2se",
           "energy_max_relative", "energy_max_relative_plus_2se")
LIMITS = {"chi_max_relative_plus_2se": .1,
          "correlation_max_absolute_plus_2se": .03,
          "energy_max_relative_plus_2se": .002}


def digest(path: Path) -> str:
    """SHA256 of one preserved input artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    """Read a strict JSON object."""
    return json.loads(path.read_text())


def declare(directory: Path) -> None:
    """Fix reductions and contrasts before any completed endpoint is inspected."""
    path = directory / "analysis_protocol.json"
    if path.exists():
        raise FileExistsError(path)
    if list((directory / "runs").glob("*/seed_*/step_*/audit.json")):
        raise ValueError("analysis must be declared before completed audits")
    write_json(path, {"kind": "size_capacity_development_analysis_declaration",
        "experiment_protocol_sha256": digest(directory / "protocol.json"),
        "analysis_sha256": digest(Path(__file__)), "metrics": list(METRICS),
        "primary_metric": "chi_max_relative_plus_2se on development fields",
        "split_metrics": ["train", "development"],
        "reductions": "Each endpoint: maximum over every declared field and size; RMS chi gives equal weight to all points. Fixed aggregate concatenates all three models, sums their parameters/work, and takes the maximum across their rows, rather than averaging three maxima.",
        "contrasts": ["fixed16_1000 -> shared16_1000", "fixed16_1000 -> shared32_1000",
            "shared16_1000 -> shared16_2000", "shared32_1000 -> shared32_2000",
            "shared16_1000 -> shared32_1000", "shared16_2000 -> shared32_2000"],
        "contrast_sign": "Positive error reduction means the second endpoint is better.",
        "interaction": "(shared16_2000-shared32_2000)-(shared16_1000-shared32_1000), in error units; positive means the width benefit grows with budget.",
        "seed_summary": "All three paired seed values, mean, minimum, maximum. No three-seed asymptotic confidence interval or significance claim.",
        "engineering_margins": LIMITS,
        "uncertainty_meaning": "error+2SE is the existing descriptive audit margin, not simultaneous coverage; no gate opened from development alone",
        "cost_accounting": "Each training trajectory counted once at its last checkpoint; both shared audits counted because they use independent draws. Resource profiling separately counted. No parents.",
        "confirmation_measured": False, "exponent_fits_performed": False,
        "universality_established": False})


def row_metrics(rows: list[dict]) -> dict:
    """Compare Born/QGT measurements with independent analytic Pauli references."""
    if not rows:
        raise ValueError("no rows")
    chi, chi_se, corr, corr_margin, energy, energy_margin, signed = [], [], [], [], [], [], []
    for row in rows:
        ref = TFIMReference(row["n"], row["h"])
        reference = ref.fidelity_susceptibility()
        if not np.isclose(row["chi_reference"], reference, rtol=0, atol=1e-12):
            raise ValueError("changed independent chi reference")
        relative = (row["chi_total"] - reference) / reference
        signed.append(relative)
        chi.append(abs(relative))
        chi_se.append(abs(relative) + 2 * row["chi_se"] / reference)
        exact_c = np.array([ref.correlation(r) for r in range(row["n"])])
        residual = abs(np.asarray(row["correlations"]) - exact_c)
        corr.append(float(residual.max()))
        corr_margin.append(float((residual + 2 * np.asarray(row["correlation_se"])).max()))
        e = abs(row["energy_total"] - ref.energy()) / abs(ref.energy())
        energy.append(e)
        energy_margin.append(e + 2 * row["energy_se"] / abs(ref.energy()))
    result = dict(zip(METRICS, [max(chi), max(chi_se), float(np.sqrt(np.mean(np.square(signed)))),
        max(corr), max(corr_margin), max(energy), max(energy_margin)]))
    result.update(chi_mean_signed_relative=float(np.mean(signed)), points=len(rows))
    result["all_point_engineering_margins_pass"] = all(result[k] <= v for k, v in LIMITS.items())
    return result


def read_endpoint(directory: Path, protocol: dict, arm_name: str, seed: int, step: int) -> dict:
    """Reject mismatched provenance, samples, fields, updates or parameter counts."""
    arm = protocol["arms"][arm_name]
    endpoint = directory / "runs" / arm_name / f"seed_{seed}" / f"step_{step}"
    audit = load_json(endpoint / "audit.json")
    signature = {"arm": arm, "seed": seed, "training_options": protocol["training_options"],
        "protocol_sha256": digest(directory / "protocol.json"),
        "schedule_sha256": protocol["schedule_sha256"][str(seed)]}
    if (audit["signature"] != signature or audit["source_sha256"] != protocol["source_sha256"]
        or audit["launcher_sha256"] != protocol["launcher_sha256"]
        or audit["checkpoint_sha256"] != digest(endpoint / "checkpoint.npz")
        or audit["arm"] != arm_name or audit["seed"] != seed or audit["step"] != step
        or audit["parameter_count"] != arm["parameter_count"]
        or audit["reserved_confirmation_measured"] or audit["universality_established"]):
        raise ValueError(f"invalid endpoint identity: {endpoint}")
    expected = arm["endpoint_costs"][str(step)]
    if any(audit["training_work"][key] != value for key, value in expected.items()):
        raise ValueError("training cost mismatch")
    with np.load(endpoint / "checkpoint.npz", allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        history = metadata["history"]
        if (int(archive["step"]) != step or metadata["signature"] != signature
            or metadata["source_sha256"] != protocol["source_sha256"]
            or metadata["launcher_sha256"] != protocol["launcher_sha256"]
            or metadata["warm_start"] is not None
            or json.loads(str(archive["work"])) != audit["training_work"]):
            raise ValueError("invalid checkpoint metadata")
        leaves = [archive[k] for k in archive.files if k.startswith("params__")]
        if sum(a.size for a in leaves) != arm["parameter_count"] or not all(np.isfinite(a).all() for a in leaves):
            raise ValueError("invalid checkpoint parameters")
    if [r["step"] for r in history] != list(range(1, step + 1)):
        raise ValueError("incomplete update history")
    if [r["field_schedule_epoch"] for r in history] != list(range(step)):
        raise ValueError("incorrect schedule prefix")
    if any(r["tasks_evaluated"] != arm["tasks_per_epoch"] or not np.isfinite(r["sr_relative_residual"])
           or r["sr_relative_residual"] > 1e-6 for r in history):
        raise ValueError("task count or SR guard mismatch")
    expected_points = {(n, h) for n in arm["sizes"] for h in protocol["train_fields"] + protocol["development_fields"]}
    points = {(r["n"], r["h"]) for r in audit["rows"]}
    if points != expected_points or len(points) != len(audit["rows"]):
        raise ValueError("incomplete/duplicate field grid")
    for r in audit["rows"]:
        expected_split = "train" if r["h"] in protocol["train_fields"] else "development"
        if r["split"] != expected_split or r["audit_samples"] != protocol["audit_samples"] or not r["audit_independent_of_training"]:
            raise ValueError("wrong field split or audit count")
    if audit["audit_born_draws"] != len(points) * protocol["audit_samples"]:
        raise ValueError("audit budget mismatch")
    tail = history[-100:]
    audit["convergence"] = {"last_100_mean_gradient_norm": float(np.mean([r["gradient_norm"] for r in tail])),
        "last_100_mean_variance_per_site": float(np.mean([r["variance_per_site"] for r in tail])),
        "last_100_max_variance_per_site": max(r["max_variance_per_site"] for r in tail),
        "last_100_mean_effective_learning_rate": float(np.mean([r["effective_learning_rate"] for r in tail])),
        "largest_true_sr_residual": max(r["sr_relative_residual"] for r in history),
        "additional_residual_solves": audit["training_work"].get("sr_refinement_solves", 0)}
    audit["_history"] = history
    return audit


def grouped_summary(endpoints: list[dict], sizes: list[int]) -> dict:
    """Aggregate independent fixed models without dropping their work or rows."""
    rows = [row for e in endpoints for row in e["rows"]]
    result = {"parameter_count": sum(e["parameter_count"] for e in endpoints),
        "training_work": {key: sum(e["training_work"].get(key, 0) for e in endpoints)
                          for key in endpoints[0]["training_work"]},
        "audit_born_draws": sum(e["audit_born_draws"] for e in endpoints),
        "checkpoint_sha256": [e["checkpoint_sha256"] for e in endpoints],
        "convergence_by_arm": {e["arm"]: e["convergence"] for e in endpoints}}
    for split in ["train", "development"]:
        selected = [r for r in rows if r["split"] == split]
        result[split] = row_metrics(selected)
        result[split]["by_size"] = {str(n): row_metrics([r for r in selected if r["n"] == n]) for n in sizes}
    return result


def contrast(before: dict, after: dict) -> dict:
    """Paired reduction in error; no significance inference from three seeds."""
    return {metric: before[metric] - after[metric] for metric in METRICS}


def analyze(directory: Path) -> dict:
    """Validate all 21 endpoints and count 15 trajectories exactly once."""
    declaration = load_json(directory / "analysis_protocol.json")
    if declaration["analysis_sha256"] != digest(Path(__file__)):
        raise ValueError("analysis changed after declaration")
    protocol = load_json(directory / "protocol.json")
    if (declaration["experiment_protocol_sha256"] != digest(directory / "protocol.json")
        or source_hashes() != protocol["source_sha256"]
        or digest(Path(__file__).with_name("size_capacity_controls.py")) != protocol["launcher_sha256"]):
        raise ValueError("experiment/library changed")
    for seed in protocol["seeds"]:
        if digest(directory / f"schedule_{seed}.npy") != protocol["schedule_sha256"][str(seed)]:
            raise ValueError("field schedule changed")
    base = protocol["base_steps"]
    all_endpoints, groups, raw = [], {}, {}
    training_work, audit_draws = {}, 0
    for seed in protocol["seeds"]:
        endpoints = {}
        for name, arm in protocol["arms"].items():
            if (directory / "runs" / name / f"seed_{seed}" / "failure.json").exists():
                raise ValueError("failed trajectory cannot form a completed experiment")
            complete = load_json(directory / "runs" / name / f"seed_{seed}" / "complete.json")
            if complete["profile_only"] or complete["source_sha256"] != protocol["source_sha256"] or complete["launcher_sha256"] != protocol["launcher_sha256"]:
                raise ValueError("invalid trajectory completion")
            for step in arm["milestones"]:
                e = read_endpoint(directory, protocol, name, seed, step)
                endpoints[f"{name}_{step}"] = e
                all_endpoints.append(e)
                audit_draws += e["audit_born_draws"]
            final = endpoints[f"{name}_{max(arm['milestones'])}"]
            root = directory / "runs" / name / f"seed_{seed}"
            with np.load(root / "accepted.npz", allow_pickle=False) as accepted, np.load(
                root / f"step_{max(arm['milestones'])}" / "checkpoint.npz", allow_pickle=False) as saved:
                if accepted.files != saved.files or any(not np.array_equal(accepted[k], saved[k]) for k in saved.files):
                    raise ValueError("final accepted state differs from audited checkpoint")
            for key, value in final["training_work"].items():
                training_work[key] = training_work.get(key, 0) + value
            if len(arm["milestones"]) == 2:
                prefix = endpoints[f"{name}_{base}"]["_history"]
                if prefix != final["_history"][:base]:
                    raise ValueError("longer budget did not preserve trajectory prefix")
        fixed = [endpoints[f"fixed16_L{n}_{base}"] for n in protocol["sizes"]]
        seed_groups = {f"fixed16_{base}": grouped_summary(fixed, protocol["sizes"])}
        for width in [16, 32]:
            for step in [base, 2 * base]:
                name = f"shared{width}_{step}"
                seed_groups[name] = grouped_summary([endpoints[name]], protocol["sizes"])
        groups[str(seed)] = seed_groups
        raw[str(seed)] = {name: e["rows"] for name, e in endpoints.items()}
    pairs = [(f"fixed16_{base}", f"shared16_{base}"), (f"fixed16_{base}", f"shared32_{base}"),
             (f"shared16_{base}", f"shared16_{2*base}"), (f"shared32_{base}", f"shared32_{2*base}"),
             (f"shared16_{base}", f"shared32_{base}"), (f"shared16_{2*base}", f"shared32_{2*base}")]
    contrasts = {}
    for first, second in pairs:
        values = {seed: contrast(g[first]["development"], g[second]["development"]) for seed, g in groups.items()}
        by_size = {seed: {str(n): contrast(g[first]["development"]["by_size"][str(n)], g[second]["development"]["by_size"][str(n)])
                         for n in protocol["sizes"]} for seed, g in groups.items()}
        contrasts[f"{first}_to_{second}"] = {"paired_error_reductions": values, "by_size": by_size,
            "summary": {metric: {"mean": float(np.mean([v[metric] for v in values.values()])),
                "min": min(v[metric] for v in values.values()), "max": max(v[metric] for v in values.values())} for metric in METRICS}}
    interaction = {seed: {m: (g[f"shared16_{2*base}"]["development"][m] - g[f"shared32_{2*base}"]["development"][m])
        - (g[f"shared16_{base}"]["development"][m] - g[f"shared32_{base}"]["development"][m]) for m in METRICS} for seed, g in groups.items()}
    input_hashes = {str(p.relative_to(directory)): digest(p) for p in sorted((directory / "runs").glob("*/seed_*/step_*/audit.json"))}
    profiles = [load_json(p) for p in sorted((directory / "profiles").glob("*/seed_*/profile.json"))]
    return {"kind": "validated_size_capacity_development_comparison", "seeds": protocol["seeds"],
        "protocol_sha256": digest(directory / "protocol.json"), "analysis_protocol_sha256": digest(directory / "analysis_protocol.json"),
        "analysis_sha256": digest(Path(__file__)), "source_sha256": protocol["source_sha256"],
        "launcher_sha256": protocol["launcher_sha256"], "validated_endpoints": len(all_endpoints),
        "unique_training_trajectories": len(protocol["seeds"]) * len(protocol["arms"]),
        "total_unique_training_work": training_work, "total_audit_born_draws": audit_draws,
        "profiles_separate_from_training": profiles, "endpoint_audit_sha256": input_hashes,
        "groups_by_seed": groups, "paired_contrasts": contrasts, "width_budget_interaction": interaction,
        "raw_rows_by_seed": raw, "fairness_limits": protocol["fairness_limits"],
        "engineering_margins": LIMITS, "confirmation_measured": False,
        "universality_established": False, "phase0_scientific_gate_opened": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--declare", action="store_true")
    args = parser.parse_args()
    if args.declare:
        declare(args.directory)
    else:
        write_json(args.directory / "comparison.json", analyze(args.directory))


if __name__ == "__main__":
    main()
