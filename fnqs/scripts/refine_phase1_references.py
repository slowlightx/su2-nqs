"""Refine finite-size candidate peaks with parity-resolved exact references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from fnqs.artifacts import write_json
from fnqs.exact import TFIMReference
from fnqs.experiments import provenance
from fnqs.hamiltonians import IsingCouplings
from fnqs.observables import correlations_from_state, fidelity_metric, renyi2_from_state
from fnqs.sectors import parity_reference


def main() -> None:
    """Record a declared finite ED budget; retain response and spectrum audits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[6, 8, 10, 12, 14])
    parser.add_argument("--step", type=float, default=.005)
    parser.add_argument("--padding", type=float, default=.05)
    args = parser.parse_args()
    if not 0 < args.step <= .05 or not 0 <= args.padding <= .2:
        raise ValueError("invalid reference grid")
    if (args.output / "protocol.json").exists():
        raise FileExistsError("use a new reference output directory")
    previous = json.loads(args.input.read_text())
    tasks = []
    names = sorted({cut["family"] for cut in previous["cuts"]})
    for name in names:
        available = [cut for cut in previous["cuts"] if cut["family"] == name]
        for n in args.sizes:
            anchor = min(available, key=lambda cut: abs(cut["n"] - n))
            lo, hi = anchor["peak_bracket"]
            lo, hi = max(args.step, lo - args.padding), hi + args.padding
            fields = np.round(np.linspace(lo, hi, round((hi - lo) / args.step) + 1), 8).tolist()
            tasks.append({"name":name,"n":n,"j2":anchor["j2"],"jx":anchor["jx"],
                          "fields":fields,"anchor_n":anchor["n"]})
    config = {"sizes":args.sizes,"step":args.step,"padding":args.padding,"tasks":tasks,
              "grid_points":sum(len(task["fields"]) for task in tasks),
              "response_audit_extra_points":4 * len(tasks)}
    metadata = {**provenance(config), "previous_input":str(args.input),
                "previous_input_sha256":hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "launcher_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "universality_established":False,"neural_result":False,
                "large_L_reference_available":False}
    write_json(args.output / "protocol.json", metadata)
    started = time.monotonic(); completed = []
    audit = {"max_spectrum_residual":0.,"max_response_relative_residual":0.,
             "max_norm_error":0.,"max_tfim_energy_abs_error":0.,"max_tfim_chi_abs_error":0.}
    for task in tasks:
        n = task["n"]; rows = []
        for h in task["fields"]:
            g = IsingCouplings(j2=task["j2"], jx=task["jx"], h=h)
            ref = parity_reference(n, g)
            correlations = correlations_from_state(ref.state, n)
            rows.append({"h":h,"energy_total":ref.energy,"ground_parity":1,
                         "same_parity_gap":ref.same_parity_gap,"opposite_parity_gap":ref.opposite_parity_gap,
                         "chi_total":ref.chi_total,"correlations":correlations.tolist(),
                         "m2":float(correlations.mean()),"c_half":float(correlations[n // 2]),
                         "renyi2_half":renyi2_from_state(ref.state, n, n // 2),
                         "spectrum_residual":ref.spectrum_residual,
                         "response_relative_residual":ref.response_relative_residual})
            audit["max_spectrum_residual"] = max(audit["max_spectrum_residual"], ref.spectrum_residual)
            audit["max_response_relative_residual"] = max(audit["max_response_relative_residual"], ref.response_relative_residual)
            audit["max_norm_error"] = max(audit["max_norm_error"], abs(float(np.vdot(ref.state, ref.state)) - 1))
            if g.j2 == g.jx == 0:
                exact = TFIMReference(n, h)
                audit["max_tfim_energy_abs_error"] = max(audit["max_tfim_energy_abs_error"], abs(ref.energy - exact.energy()))
                audit["max_tfim_chi_abs_error"] = max(audit["max_tfim_chi_abs_error"], abs(ref.chi_total - exact.fidelity_susceptibility()))
        peak = int(np.argmax([row["chi_total"] for row in rows]))
        h = rows[peak]["h"]
        middle = parity_reference(n, IsingCouplings(j2=task["j2"], jx=task["jx"], h=h))
        fidelity_audit = []
        for delta in [args.step, args.step / 2]:
            plus, minus = [parity_reference(n, IsingCouplings(j2=task["j2"], jx=task["jx"], h=h + sign * delta))
                           for sign in [1, -1]]
            finite = fidelity_metric(middle.state, plus.state, minus.state, delta)
            fidelity_audit.append({"delta":delta,"chi_finite_difference":finite,
                                  "absolute_error":abs(finite - middle.chi_total)})
        completed.append({**task,"rows":rows,"peak_field":h,
            "peak_bracket":[rows[max(0, peak - 1)]["h"], rows[min(len(rows) - 1, peak + 1)]["h"]],
            "peak_on_grid_edge":peak in [0, len(rows) - 1],"fidelity_audit":fidelity_audit})
        write_json(args.output / "partial.json", {**metadata,"kind":"phase1_parity_ed_reference_in_progress",
                                                  "cuts":completed,"audit":audit})
        print({"cut":task["name"],"n":n,"peak":h,"seconds":time.monotonic() - started}, flush=True)
    result = {**metadata,"kind":"phase1_parity_ed_reference","cuts":completed,"audit":audit,
              "all_peaks_bracketed":all(not cut["peak_on_grid_edge"] for cut in completed),
              "elapsed_seconds":time.monotonic() - started,
              "next":"Obtain L>=16 independent references and verify candidate classes; finite peaks are not thermodynamic hc."}
    write_json(args.output / "reference.json", result)
    print({"elapsed_seconds":result["elapsed_seconds"],"audit":audit}, flush=True)


if __name__ == "__main__":
    main()
