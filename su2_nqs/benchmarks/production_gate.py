"""Fail-closed evidence gate for production locality simulations.

The evaluator deliberately checks provenance as well as numerical thresholds.
An exact dense reference, a suffix-contraction oracle, or a different neural
model family cannot satisfy a production-model check merely by reporting a
good number.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


DEFAULT_REQUIREMENTS: dict[str, Any] = {
    "model_family": "joint_endpoint_vector_scalar",
    "engine": "joint_endpoint_vector_scalar_causal_recurrence",
    "required_j2_over_j1": [0.5, 0.6],
    "required_seeds": list(range(8)),
    "maximum_exact_error": 1.0e-10,
    "minimum_connected_residual_ess_fraction": 0.2,
    "minimum_recovery_fraction": 0.8,
    "maximum_normalization_error": 1.0e-10,
    "minimum_runtime_n_sites": 16,
    "maximum_triangular_graph_error": 1.0e-10,
}


def requirements_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized gate requirements from a cluster configuration."""
    result = dict(DEFAULT_REQUIREMENTS)
    result.update(config.get("production_gate", {}))
    result["required_j2_over_j1"] = [float(x) for x in result["required_j2_over_j1"]]
    result["required_seeds"] = [int(x) for x in result["required_seeds"]]
    return result


def _check(name: str, passed: bool, observed: Any, required: Any, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
        "reason": reason,
    }


def _identity(row: Mapping[str, Any], req: Mapping[str, Any]) -> bool:
    return (
        row.get("model_family") == req["model_family"]
        and row.get("engine") == req["engine"]
    )


def _numeric_extreme(values: Sequence[Any], *, largest: bool) -> float | int | None:
    """Return a finite numeric extreme while preserving fail-closed nulls.

    Evidence rows may legitimately record ``null`` when a held-out test was
    not performed.  Reporting that absence must not crash the gate evaluator.
    The corresponding predicate still fails because it requires every value
    to be numeric.
    """

    numeric = [
        value
        for value in values
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    ]
    if not numeric:
        return None
    return (max if largest else min)(numeric)


def _required_rows(
    rows: Sequence[Mapping[str, Any]], req: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    selected: list[Mapping[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for ratio in req["required_j2_over_j1"]:
        for seed in req["required_seeds"]:
            matches = []
            for row in rows:
                try:
                    same_ratio = float(row.get("j2_over_j1")) == ratio
                except (TypeError, ValueError):
                    same_ratio = False
                if (same_ratio and row.get("seed") == seed and _identity(row, req)
                        and row.get("trial_kind") == "production_causal"):
                    matches.append(row)
            if len(matches) == 1:
                selected.append(matches[0])
            else:
                missing.append({"j2_over_j1": ratio, "seed": seed, "matches": len(matches)})
    return selected, missing


def evaluate_production_gate(
    evidence: Mapping[str, Any], requirements: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate all production gates, rejecting absent or misattributed data."""
    req = dict(DEFAULT_REQUIREMENTS)
    req.update(requirements)
    checks: list[dict[str, Any]] = []

    exact = evidence.get("exact_validation", {})
    exact_errors = [exact.get("maximum_su2_error"), exact.get("maximum_reduced_error")]
    exact_ok = (
        _identity(exact, req)
        and exact.get("uses_hidden_suffix") is False
        and all(isinstance(x, (int, float)) and x <= req["maximum_exact_error"] for x in exact_errors)
    )
    checks.append(_check(
        "exact_su2_and_reduced", exact_ok,
        {"identity": _identity(exact, req), "errors": exact_errors},
        {"model_family": req["model_family"], "engine": req["engine"], "maximum": req["maximum_exact_error"]},
        "Exact SU(2) and reduced-operator validation must belong to the production family and be suffix-free.",
    ))

    sampling = evidence.get("connected_residual_sampling", {})
    ess = sampling.get("minimum_ess_fraction")
    ess_ok = (
        _identity(sampling, req)
        and sampling.get("n_sites", 0) >= 16
        and sampling.get("uses_hidden_suffix") is False
        and sampling.get("proposal_family") == "hamiltonian_connected"
        and isinstance(ess, (int, float)) and ess >= req["minimum_connected_residual_ess_fraction"]
    )
    checks.append(_check(
        "connected_residual_ess", ess_ok,
        {"identity": _identity(sampling, req), "minimum_ess_fraction": ess},
        {"minimum": req["minimum_connected_residual_ess_fraction"], "n_sites": 16},
        "ESS must be measured for the causal production engine, not merely for another model's proposal.",
    ))

    runtime = evidence.get("runtime", {})
    runtime_ok = (
        _identity(runtime, req)
        and runtime.get("n_sites", 0) >= req["minimum_runtime_n_sites"]
        and runtime.get("compiled") is True
        and runtime.get("scalable") is True
        and runtime.get("production_engine") is True
        and runtime.get("uses_dense_exhaustive_plan") is False
        and isinstance(runtime.get("throughput_samples_per_second"), (int, float))
        and runtime["throughput_samples_per_second"] > 0
    )
    checks.append(_check(
        "scalable_n16_runtime", runtime_ok,
        {k: runtime.get(k) for k in ("model_family", "engine", "n_sites", "compiled", "scalable", "production_engine", "uses_dense_exhaustive_plan", "throughput_samples_per_second")},
        "compiled, non-dense production-engine throughput at N>=16",
        "A dense causal reference or an isolated operator kernel is not a scalable autoregressive runtime.",
    ))

    norm = evidence.get("normalization", {})
    norm_error = norm.get("maximum_error")
    norm_ok = (
        _identity(norm, req)
        and norm.get("n_sites") == 16
        and norm.get("exhaustive") is True
        and norm.get("uses_hidden_suffix") is False
        and isinstance(norm_error, (int, float))
        and norm_error <= req["maximum_normalization_error"]
    )
    checks.append(_check(
        "n16_exhaustive_normalization", norm_ok,
        {"identity": _identity(norm, req), "n_sites": norm.get("n_sites"), "exhaustive": norm.get("exhaustive"), "maximum_error": norm_error},
        {"n_sites": 16, "exhaustive": True, "maximum_error": req["maximum_normalization_error"]},
        "N<=10 validation cannot stand in for exhaustive N=16 normalization of the production family.",
    ))

    trials, missing = _required_rows(evidence.get("trials", []), req)
    expected = len(req["required_j2_over_j1"]) * len(req["required_seeds"])
    coverage = not missing and len(trials) == expected
    suffix_free = coverage and all(row.get("uses_hidden_suffix") is False for row in trials)
    checks.append(_check(
        "strict_no_suffix", suffix_free,
        {"covered_trials": len(trials), "expected_trials": expected, "missing": missing, "all_suffix_free": suffix_free},
        "all required production trials use revealed-prefix information only",
        "Suffix-contraction oracle evidence is diagnostic only and is never deployable evidence.",
    ))

    recoveries = [row.get("recovery_fraction") for row in trials]
    recovery_ok = coverage and all(
        isinstance(x, (int, float)) and x >= req["minimum_recovery_fraction"] for x in recoveries
    )
    checks.append(_check(
        "per_seed_recovery", recovery_ok,
        {"covered_trials": len(trials), "minimum": _numeric_extreme(recoveries, largest=False), "missing": missing},
        {"every_j2_seed": req["minimum_recovery_fraction"]},
        "Every required J2/seed causal production trial must pass; medians and other model families do not count.",
    ))

    energy_deltas = [row.get("exact_energy_delta") for row in trials]
    energy_ok = coverage and all(isinstance(x, (int, float)) and x <= 0.0 for x in energy_deltas)
    checks.append(_check(
        "no_exact_energy_loss", energy_ok,
        {"maximum_exact_energy_delta": _numeric_extreme(energy_deltas, largest=True), "missing": missing},
        "candidate minus baseline exact energy <= 0 for every required trial",
        "Backtracking or sampled estimates cannot replace exact finite-system energy monotonicity.",
    ))

    ucbs = [row.get("paired_ucb95") for row in trials]
    ucb_ok = coverage and all(
        row.get("evaluation_independent") is True
        and isinstance(row.get("paired_ucb95"), (int, float))
        and row["paired_ucb95"] < 0.0
        for row in trials
    )
    checks.append(_check(
        "independent_paired_ucb", ucb_ok,
        {"maximum_ucb95": _numeric_extreme(ucbs, largest=True), "all_independent": coverage and all(r.get("evaluation_independent") is True for r in trials), "missing": missing},
        "independent paired 95% UCB < 0 for every required trial",
        "A significant gain from generic TreeGRU trials cannot be reassigned to the causal channel model.",
    ))

    tri = evidence.get("triangular_exact_graph", {})
    graph_errors = [tri.get("maximum_hermiticity_error"), tri.get("maximum_graph_error")]
    tri_ok = (
        tri.get("lattice_family") == "triangular_j1j2"
        and tri.get("n_sites", 0) >= 16
        and tri.get("exact") is True
        and tri.get("boundary_condition") == "pbc"
        and tri.get("j1_bonds", 0) > 0 and tri.get("j2_bonds", 0) > 0
        and all(isinstance(x, (int, float)) and x <= req["maximum_triangular_graph_error"] for x in graph_errors)
    )
    checks.append(_check(
        "triangular_exact_graph", tri_ok,
        {"n_sites": tri.get("n_sites"), "errors": graph_errors, "j1_bonds": tri.get("j1_bonds"), "j2_bonds": tri.get("j2_bonds")},
        {"lattice_family": "triangular_j1j2", "n_sites": 16, "maximum_error": req["maximum_triangular_graph_error"]},
        "Triangular production is gated on an independent exact 4x4 PBC graph/operator check.",
    ))

    failed = [item["name"] for item in checks if not item["passed"]]
    return {
        "schema_version": 1,
        "passed": not failed,
        "decision": "PASS" if not failed else "FAIL",
        "required_model_family": req["model_family"],
        "required_engine": req["engine"],
        "checks": checks,
        "failed_checks": failed,
        "fail_closed": True,
    }


def render_production_gate_report(result: Mapping[str, Any]) -> str:
    """Render a compact human-readable audit report."""
    lines = [
        "# Production readiness gate",
        "",
        f"**Decision: {result['decision']}**",
        "",
        f"Required family: `{result['required_model_family']}`  ",
        f"Required engine: `{result['required_engine']}`",
        "",
        "| Gate | Result | Observed |",
        "|---|---:|---|",
    ]
    for check in result["checks"]:
        display = check["observed"]
        if isinstance(display, dict) and isinstance(display.get("missing"), list):
            display = dict(display)
            missing = display.pop("missing")
            display["missing_required_trials"] = len(missing)
        observed = str(display).replace("|", "\\|")
        lines.append(f"| `{check['name']}` | {'PASS' if check['passed'] else 'FAIL'} | `{observed}` |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The gate is provenance-aware and fail-closed. Dense exhaustive causal references, "
        "hidden-suffix contraction oracles, isolated recoupling kernels, and generic TreeGRU "
        "trials cannot be relabeled as production causal-engine evidence.",
        "",
        "Cluster execution remains disabled until every row passes and the production backend "
        "is independently marked implemented in the cluster configuration.",
        "",
    ])
    return "\n".join(lines)
