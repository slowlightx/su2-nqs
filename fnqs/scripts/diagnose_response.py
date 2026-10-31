"""Sealed, read-only development diagnostics on the original v8 checkpoints.

Declare before run. Resume validates sources, parents and every completed row.
No theta updates, confirmation fields, model selection or exponent fitting.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np

from fnqs.artifacts import load_checkpoint, source_hashes, write_json
from fnqs.basis import all_bits
from fnqs.diagnostics.response import (bootstrap_response, dense_response,
    response_records, summarize_response, tangent_comparison)
from fnqs.exact import TFIMReference
from fnqs.hamiltonians import IsingCouplings, dense_hamiltonian
from fnqs.models import GRUConfig, log_psi, sample
from fnqs.scripts.analyze_size_capacity import digest, load_json, read_endpoint
from fnqs.variants.size_features import SizeFeatureConfig, parameter_counts

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("fixed16_1000", "shared16_1000", "shared16_2000", "shared32_1000", "shared32_2000")
FIELDS = (.952, 1.002, 1.047)
FD_STEPS = (1e-3, 5e-4)


def runtime_sources() -> dict[str, str]:
    """Hash old root sources and all new runtime dependencies recursively."""
    files = list(ROOT.glob("*.py"))
    for subdir in ("diagnostics", "variants"):
        files += list((ROOT/subdir).glob("*.py"))
    files += [Path(__file__), ROOT/"scripts"/"analyze_size_capacity.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def case_grid(parent_protocol: dict) -> list[dict]:
    """Use only three already observed development fields, all seeds/groups."""
    result = []
    for seed in parent_protocol["seeds"]:
        for group in GROUPS:
            family, step = group.rsplit("_", 1)
            for n in parent_protocol["sizes"]:
                arm = f"fixed16_L{n}" if family == "fixed16" else family
                for h in FIELDS:
                    if h not in parent_protocol["development_fields"] or h in parent_protocol["reserved_confirmation_fields"]:
                        raise ValueError("diagnostic field leakage")
                    result.append({"index": len(result), "group": group, "arm": arm,
                                   "seed": seed, "step": int(step), "n": n, "h": h})
    return result


def parent_files(parent: Path, cases: list[dict]) -> dict[str, str]:
    """Bind all original input checkpoints/audits plus sealed parent protocols."""
    files = {"protocol.json", "analysis_protocol.json", "comparison.json"}
    for c in cases:
        endpoint = f"runs/{c['arm']}/seed_{c['seed']}/step_{c['step']}"
        files.update((endpoint+"/checkpoint.npz", endpoint+"/audit.json"))
    return {name: digest(parent/name) for name in sorted(files)}


def declare(parent: Path, output: Path, samples: int = 1024, chunk: int = 32,
            bootstrap: int = 200) -> dict:
    """Freeze all new diagnostic costs and inputs before new observations."""
    if (output/"protocol.json").exists() or (output/"cases").exists():
        raise FileExistsError("use a new version directory")
    if samples < 128 or chunk < 1 or samples % chunk or bootstrap < 2:
        raise ValueError("invalid diagnostic sample/chunk/bootstrap budget")
    p = load_json(parent/"protocol.json")
    if source_hashes() != p["source_sha256"]:
        raise ValueError("original model numerical sources changed")
    comparison = load_json(parent/"comparison.json")
    if comparison["validated_endpoints"] != 21 or comparison["unique_training_trajectories"] != 15:
        raise ValueError("parent experiment incomplete")
    cases = case_grid(p)
    protocol = {"kind": "fixed_theta_coupling_response_development_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent": str(parent.resolve()), "parent_sha256": parent_files(parent, cases),
        "source_sha256": runtime_sources(), "cases": cases, "direction": [0., 0., 0., 1.],
        "fd_steps": list(FD_STEPS), "samples_per_point": samples, "chunk": chunk,
        "bootstrap_replicates": bootstrap, "prng_domain": 31337,
        "diagnostic_born_draws": len(cases)*samples,
        "hamiltonian_state_and_tangent_actions": 2*len(cases),
        "directional_operator_actions": len(cases),
        "optimizer_updates": 0, "training_hamiltonians": 0,
        "profile": {"case_index": 0, "born_draws": 128,
                    "hamiltonian_state_and_tangent_actions": 2, "directional_operator_actions": 1},
        "dense_calibration": {"sizes": [4, 6, 8], "h": 1.02, "tangent_scales": [1., 1.25, 0.],
                              "born_draws": 0, "exact_state_witness": True},
        "metrics": ["AD versus two central FD scores/chi", "projected physical response residual",
                    "Hellmann-Feynman energy derivative defect", "state residual norm", "chi/reference"],
        "uncertainty": "iid-row paired bootstrap SE; recenter E, scores, source and Q in every replica; no simultaneous CI or new gate",
        "field_selection": "post-hoc development diagnosis of the known v8 response shape; left/middle/right development points",
        "interpretation": "AD/FD agreement validates input differentiation only. Physical residual is not a tangent-error bound for an inexact state without state/gap control.",
        "size_ablations": {"implemented_policies": ["legacy", "no_explicit_size", "no_length"],
            "counts_width16": {name: parameter_counts(SizeFeatureConfig(16, size_features=name))
                               for name in ("legacy", "no_explicit_size", "no_length")},
            "trained": False, "old_checkpoints_reinterpreted": False,
            "next_training_route": "independent model at each fixed L, sharing couplings within L; cold ablation only"},
        "confirmation_measured": False, "exponent_fits_performed": False,
        "phase0_scientific_gate": False, "universality_established": False}
    archive = output.parent/"runtime_v9"/"fnqs"
    if archive.exists():
        raise FileExistsError(archive)
    write_json(output/"protocol.json", protocol)
    for name, sha in protocol["source_sha256"].items():
        target = archive/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
        if digest(target) != sha:
            raise ValueError("archive copy mismatch")
    write_json(archive.parent/"manifest.json", {"source_sha256": protocol["source_sha256"],
        "protocol_sha256": digest(output/"protocol.json"), "training_performed": False})
    return protocol


def validate(output: Path) -> tuple[dict, Path, dict]:
    """Fail closed on modified runtime or input artifacts."""
    protocol = load_json(output/"protocol.json")
    if runtime_sources() != protocol["source_sha256"]:
        raise ValueError("diagnostic runtime changed after declaration")
    archive = output.parent/"runtime_v9"
    manifest = load_json(archive/"manifest.json")
    if (manifest["protocol_sha256"] != digest(output/"protocol.json")
        or manifest["source_sha256"] != protocol["source_sha256"]
        or any(digest(archive/"fnqs"/name) != sha for name, sha in protocol["source_sha256"].items())):
        raise ValueError("sealed protocol/archive changed")
    parent = Path(protocol["parent"])
    p = load_json(parent/"protocol.json")
    if parent_files(parent, protocol["cases"]) != protocol["parent_sha256"]:
        raise ValueError("parent artifacts changed")
    if protocol["cases"] != case_grid(p):
        raise ValueError("case grid changed")
    return protocol, parent, p


def witness_log(p: dict, bits: jax.Array, g: jax.Array) -> jax.Array:
    """Exact-state calibration witness with independently specified tangent."""
    n = bits.shape[1]
    labels = bits @ (1 << jnp.arange(n-1, -1, -1))
    psi = p["state"]+p["scale"]*(g[3]-p["h"])*p["tangent"]
    psi = psi/jnp.linalg.norm(psi)
    return jnp.log(psi[labels])


def calibrate(output: Path) -> None:
    """Record exact response, deliberately wrong derivative and prefix witness."""
    if (output/"calibration.json").exists():
        saved = load_json(output/"calibration.json")
        if saved["protocol_sha256"] != digest(output/"protocol.json") or len(saved["rows"]) != 9:
            raise ValueError("invalid exact calibration")
        if (saved["largest_action_discrepancy"] > 2e-11
            or saved["largest_relative_response_discrepancy"] > 2e-11
            or saved["largest_fd_score_rms_error"] > 2e-9):
            raise ArithmeticError("saved exact response calibration failed")
        return
    rows, max_action_error, max_residual_error, max_fd_error = [], 0., 0., 0.
    for n in (4, 6, 8):
        g = IsingCouplings(h=1.02)
        ref = dense_response(n, g)
        for scale in (1., 1.25, 0.):
            p = {"state": jnp.asarray(ref.state), "tangent": jnp.asarray(ref.tangent),
                 "h": g.h, "scale": scale}
            rec = {k: np.asarray(v) for k, v in response_records(p, all_bits(n), g.vector(),
                   witness_log, fd_steps=(1e-5, 5e-6)).items()}
            result = summarize_response(rec, ref.state**2)
            comparison = tangent_comparison(ref.state, rec["score"], ref)
            h = dense_hamiltonian(n, g)
            v = dense_hamiltonian(n, IsingCouplings(0., 0., 0., 1.))
            errors = [np.max(abs(rec["energy"]-h@ref.state/ref.state)),
                      np.max(abs(rec["source"]-v@ref.state/ref.state)),
                      np.max(abs(rec["h_score"]-h@(ref.state*rec["score"])/ref.state))]
            max_action_error = max(max_action_error, float(max(errors)))
            max_residual_error = max(max_residual_error, abs(result["response_relative_residual"]-abs(scale-1)))
            max_fd_error = max(max_fd_error, max(result["fd_score_rms_error"]))
            rows.append({"n": n, "h": g.h, "tangent_scale": scale,
                         "reference_chi": ref.chi_total, **result, **comparison})
    exact4 = dense_response(4, IsingCouplings()).state**2
    exact6 = (dense_response(6, IsingCouplings()).state**2).reshape(16, 4).sum(axis=1)
    write_json(output/"calibration.json", {"rows": rows, "exact_state_witness": True,
        "protocol_sha256": digest(output/"protocol.json"),
        "largest_action_discrepancy": max_action_error, "action_tolerance": 2e-11,
        "largest_relative_response_discrepancy": max_residual_error, "response_tolerance": 2e-11,
        "largest_fd_score_rms_error": max_fd_error, "fd_tolerance": 2e-9,
        "prefix_pbc_witness": {"h": 1., "sizes": [4, 6],
            "max_exact_probability_prefix_gap": float(np.max(abs(exact4-exact6))),
            "scope": "no_length recurrent AR plus Z2 mixture; not all architectures without a scalar L input"}})
    if max_action_error > 2e-11 or max_residual_error > 2e-11 or max_fd_error > 2e-9:
        raise ArithmeticError("exact response calibration failed")


def case_path(output: Path, index: int) -> Path:
    return output/"cases"/f"case_{index:03d}"


def completed_case(output: Path, case: dict) -> dict | None:
    path = case_path(output, case["index"])
    if not path.with_suffix(".json").exists():
        return None
    row = load_json(path.with_suffix(".json"))
    if (row["case"] != case or row["protocol_sha256"] != digest(output/"protocol.json")
        or row["records_sha256"] != digest(path.with_suffix(".npz"))):
        raise ValueError("invalid saved diagnostic case")
    return row


def run(output: Path, profile_only: bool = False, max_cases: int | None = None) -> dict:
    """Run with dynamic checkpoint parameters, static width/n and chunked JVP."""
    protocol, parent, pp = validate(output)
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be positive")
    endpoint_cache, kernels = {}, {}

    def get_endpoint(c):
        identity = (c["arm"], c["seed"], c["step"])
        if identity not in endpoint_cache:
            audit = read_endpoint(parent, pp, *identity)
            path = parent/"runs"/c["arm"]/f"seed_{c['seed']}"/f"step_{c['step']}"/"checkpoint.npz"
            params, state, _, metadata = load_checkpoint(path)
            cfg = GRUConfig(**metadata["model"])
            if asdict(cfg) != {"hidden_dim": pp["arms"][c["arm"]]["width"], "size_scale": 64.}:
                raise ValueError("unexpected original model semantics")
            endpoint_cache[identity] = (params, cfg, audit)
        return endpoint_cache[identity]

    def evaluate(c, draws, domain):
        params, cfg, audit = get_endpoint(c)
        identity = (cfg, c["n"], draws)
        if identity not in kernels:
            log = lambda p, b, g: log_psi(p, b, g, cfg)
            kernel = jax.jit(lambda p, b, g: response_records(p, b, g, log,
                tuple(protocol["direction"]), tuple(protocol["fd_steps"])))
            sampler = jax.jit(lambda p, k, g: sample(p, k, draws, c["n"], g, cfg))
            kernels[identity] = (sampler, kernel)
        sampler, kernel = kernels[identity]
        key = jax.random.fold_in(jax.random.PRNGKey(protocol["prng_domain"]), domain)
        g = jnp.asarray([1., 0., 0., c["h"]])
        start = time.monotonic()
        bits = sampler(params, key, g)
        pieces = {k: [] for k in ("score", "energy", "h_score", "source", "fd_scores")}
        for offset in range(0, draws, protocol["chunk"]):
            rec = kernel(params, bits[offset:offset+protocol["chunk"]], g)
            for name, value in rec.items():
                pieces[name].append(np.asarray(value))
        records = {k: np.concatenate(v) for k, v in pieces.items()}
        elapsed = time.monotonic()-start
        old = next(r for r in audit["rows"] if r["n"] == c["n"] and r["h"] == c["h"])
        return records, np.asarray(bits), elapsed, old

    if profile_only:
        if (output/"profile.json").exists():
            return load_json(output/"profile.json")
        records, bits, elapsed, _ = evaluate(protocol["cases"][0], 128, 1000000)
        result = {"kind": "resource_profile_only", "born_draws": 128, "seconds_including_compile": elapsed,
                  "finite_records": all(np.isfinite(v).all() for v in records.values()),
                  "protocol_sha256": digest(output/"protocol.json"), "scientific_accuracy_audited": False}
        write_json(output/"profile.json", result)
        return result
    calibrate(output)
    executed = 0
    for c in protocol["cases"]:
        if completed_case(output, c) is not None:
            continue
        records, bits, elapsed, old = evaluate(c, protocol["samples_per_point"], c["index"])
        stats = bootstrap_response(records, protocol["prng_domain"]+c["index"], protocol["bootstrap_replicates"])
        reference = TFIMReference(c["n"], c["h"]).fidelity_susceptibility()
        path = case_path(output, c["index"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".npz").open("wb") as stream:
            np.savez_compressed(stream, bits=bits, **records)
        row = {"case": c, "protocol_sha256": digest(output/"protocol.json"),
               "records_sha256": digest(path.with_suffix(".npz")),
               "samples": protocol["samples_per_point"], "seconds_including_compile": elapsed,
               "chi_reference": reference, "chi_ratio": stats["chi_total"]/reference,
               "original_v8_chi_ratio": old["chi_total"]/reference,
               "original_v8_chi_ratio_se": old["chi_se"]/reference, **stats}
        write_json(path.with_suffix(".json"), row)
        print(f"case {c['index']+1}/{len(protocol['cases'])}: {c['group']} seed={c['seed']} L={c['n']} h={c['h']} rho={stats['response_relative_residual']:.4f}", flush=True)
        executed += 1
        if max_cases is not None and executed >= max_cases:
            break
    rows = [row for c in protocol["cases"] if (row := completed_case(output, c)) is not None]
    complete = len(rows) == len(protocol["cases"])
    summary = {"complete": complete, "points_complete": len(rows), "points_declared": len(protocol["cases"]),
        "protocol_sha256": digest(output/"protocol.json"), "calibration_sha256": digest(output/"calibration.json"),
        "rows": rows, "new_diagnostic_born_draws": len(rows)*protocol["samples_per_point"],
        "hamiltonian_state_and_tangent_actions": 2*len(rows), "directional_operator_actions": len(rows),
        "profile_cost_separate": protocol["profile"], "optimizer_updates": 0,
        "confirmation_measured": False, "phase0_scientific_gate": False, "universality_established": False}
    if rows:
        summary["largest_fd_score_relative_rms_error"] = max(max(r["fd_score_relative_rms_error"]) for r in rows)
        summary["response_relative_residual_range"] = [min(r["response_relative_residual"] for r in rows),
                                                        max(r["response_relative_residual"] for r in rows)]
        summary["by_group_seed"] = {group: {str(seed): {
            "response_relative_residual_mean": float(np.mean([r["response_relative_residual"] for r in rows
                if r["case"]["group"] == group and r["case"]["seed"] == seed])),
            "response_relative_residual_max": max(r["response_relative_residual"] for r in rows
                if r["case"]["group"] == group and r["case"]["seed"] == seed)}
            for seed in pp["seeds"] if any(r["case"]["group"] == group and r["case"]["seed"] == seed for r in rows)}
            for group in GROUPS}
    write_json(output/("diagnosis.json" if complete else "diagnosis_partial.json"), summary)
    return {k: v for k, v in summary.items() if k not in ("rows", "by_group_seed")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "profile", "run"))
    parser.add_argument("--parent", type=Path, default=ROOT/"results"/"size_capacity_controls_v8")
    parser.add_argument("--output", type=Path, default=ROOT/"results"/"coupling_response_diagnostics_v9")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if args.command == "declare":
        p = declare(args.parent, args.output, args.samples, args.chunk, args.bootstrap)
        print({"cases": len(p["cases"]), "diagnostic_born_draws": p["diagnostic_born_draws"]})
    else:
        print(run(args.output, profile_only=args.command == "profile", max_cases=args.max_cases))


if __name__ == "__main__":
    main()
