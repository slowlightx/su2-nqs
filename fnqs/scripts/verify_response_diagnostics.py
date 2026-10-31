"""Replay saved diagnostic moments/bootstrap without new Born or H queries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fnqs.artifacts import source_hashes, write_json
from fnqs.diagnostics.response import bootstrap_response
from fnqs.exact import TFIMReference
from fnqs.scripts.analyze_size_capacity import digest, load_json, read_endpoint
from fnqs.scripts.diagnose_response import completed_case, validate


def verify(directory: Path) -> dict:
    """Reject changed inputs, rows, field identity, budgets or saved statistics."""
    protocol, parent, pp = validate(directory)
    result = load_json(directory/"diagnosis.json")
    if not result["complete"] or result["protocol_sha256"] != digest(directory/"protocol.json"):
        raise ValueError("diagnosis incomplete or protocol mismatch")
    if result["calibration_sha256"] != digest(directory/"calibration.json"):
        raise ValueError("changed calibration")
    endpoints, largest_difference = {}, 0.
    for c in protocol["cases"]:
        saved = completed_case(directory, c)
        if saved is None or saved != result["rows"][c["index"]]:
            raise ValueError("missing/changed case")
        path = directory/"cases"/f"case_{c['index']:03d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            bits = archive["bits"]
            if bits.shape != (protocol["samples_per_point"], c["n"]) or not np.isin(bits, (0, 1)).all():
                raise ValueError("invalid saved Born labels/count")
            records = {k: archive[k] for k in ("score", "energy", "h_score", "source", "fd_scores")}
        replay = bootstrap_response(records, protocol["prng_domain"]+c["index"], protocol["bootstrap_replicates"])
        if any(replay[key] != saved[key] for key in replay):
            raise ValueError("saved moments/paired bootstrap do not replay exactly")
        for key, value in replay.items():
            if isinstance(value, (float, int)):
                largest_difference = max(largest_difference, abs(value-saved[key]))
        identity = (c["arm"], c["seed"], c["step"])
        if identity not in endpoints:
            endpoints[identity] = read_endpoint(parent, pp, *identity)
        original = next(row for row in endpoints[identity]["rows"] if row["n"] == c["n"] and row["h"] == c["h"])
        ref = TFIMReference(c["n"], c["h"]).fidelity_susceptibility()
        if (saved["chi_reference"] != ref or saved["original_v8_chi_ratio"] != original["chi_total"]/ref
            or saved["original_v8_chi_ratio_se"] != original["chi_se"]/ref):
            raise ValueError("original independent audit/reference mismatch")
    points = len(protocol["cases"])
    if (len(result["rows"]) != points or result["new_diagnostic_born_draws"] != points*protocol["samples_per_point"]
        or result["hamiltonian_state_and_tangent_actions"] != 2*points
        or result["directional_operator_actions"] != points or result["optimizer_updates"] != 0
        or result["confirmation_measured"] or result["phase0_scientific_gate"]):
        raise ValueError("wrong cost/gate")
    plot = load_json(directory/"figures.json")
    if plot["input_sha256"] != digest(directory/"diagnosis.json"):
        raise ValueError("plot input mismatch")
    for name, sha in plot["figures"].items():
        if digest(directory/name) != sha:
            raise ValueError("changed figure")
    if plot["plot_source_sha256"] != digest(Path(__file__).with_name("plot_response_diagnostics.py")):
        raise ValueError("changed plot source")
    profile = load_json(directory/"profile.json")
    if profile["protocol_sha256"] != digest(directory/"protocol.json") or profile["born_draws"] != protocol["profile"]["born_draws"]:
        raise ValueError("profile identity/cost mismatch")
    report = {"kind": "saved_response_diagnostic_replay", "complete": True,
        "validated_points": points, "validated_original_endpoints": len(endpoints),
        "all_raw_labels_and_counts_valid": True, "all_moments_and_bootstrap_identical": True,
        "largest_replay_discrepancy": largest_difference, "replay_tolerance": 0.,
        "original_root_sources_unchanged": source_hashes() == pp["source_sha256"],
        "protocol_sha256": digest(directory/"protocol.json"), "diagnosis_sha256": digest(directory/"diagnosis.json"),
        "verification_source_sha256": digest(Path(__file__)),
        "new_born_draws": 0, "new_hamiltonian_queries": 0,
        "confirmation_measured": False, "phase0_scientific_gate": False}
    write_json(directory/"replay_validation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("fnqs/results/coupling_response_diagnostics_v9"))
    args = parser.parse_args()
    print(verify(args.directory))
