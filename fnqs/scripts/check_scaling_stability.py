"""Declared held-field extraction diagnostics; never select a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from fnqs.artifacts import write_json
from fnqs.exact import TFIMReference
from fnqs.scaling import collapse_fit


def _fits(rows: list[dict], options: dict) -> list[dict]:
    results = []
    for window,degree,correction in itertools.product(options["windows_x"],options["degrees"],options["corrections"]):
        subset = [row for row in rows if abs((row["h"]-1)*row["n"]) <= window]
        fit = collapse_fit([r["n"] for r in subset],[r["h"] for r in subset],
                           [r["chi_total"] for r in subset],degree=degree,correction=correction)
        results.append({"window_x":window,**fit})
    return results


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--declare",action="store_true")
    args=parser.parse_args()
    pair_path=args.pair/"protocol.json"
    pair=json.loads(pair_path.read_text())
    declaration=args.output/"protocol.json"
    if args.declare:
        args.output.mkdir(parents=True,exist_ok=False)
        config=pair["configs"]["scratch"]
        options={"windows_x":[.8,1.,1.2],"degrees":[4,6],"corrections":[False,True]}
        protocol={"kind":"held_field_scaling_sensitivity_protocol","options":options,
                  "sizes":config["sizes"],"fields":config["test_fields"],
                  "centering_h_for_TFIM_calibration":1.,"fit_hc_and_nu_free":True,
                  "additional_training_or_born_draws":0,"checkpoint_selection":False,
                  "changes_scientific_thresholds":False,
                  "leave_smallest_size_out_available":len(config["sizes"])>=4,
                  "note":"Three sizes cannot support the >=3-size fitter after dropping the smallest size.",
                  "paired_protocol_sha256":hashlib.sha256(pair_path.read_bytes()).hexdigest(),
                  "launcher_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        write_json(declaration,protocol)
        oracle=[{"n":n,"h":h,"chi_total":TFIMReference(n,h).fidelity_susceptibility()}
                for n in config["sizes"] for h in config["test_fields"]]
        fits=_fits(oracle,options)
        write_json(args.output/"oracle.json",{"kind":"exact_oracle_extraction_sensitivity",
                   "neural_result":False,"fits":fits,
                   "protocol_sha256":hashlib.sha256(declaration.read_bytes()).hexdigest()})
        print({"declared":True,"oracle_nu_range":[min(f["nu"] for f in fits),max(f["nu"] for f in fits)]})
        return
    protocol=json.loads(declaration.read_text())
    if protocol["paired_protocol_sha256"] != hashlib.sha256(pair_path.read_bytes()).hexdigest():
        raise ValueError("paired protocol differs from the declaration")
    if protocol["launcher_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("diagnostic implementation changed since declaration")
    results={}
    for name in ("warm","scratch"):
        path=args.pair/name/"neural.json"
        data=json.loads(path.read_text())
        if data["source_sha256"] != pair["source_sha256"] or data["config"] != pair["configs"][name]:
            raise ValueError("neural result differs from its paired protocol")
        results[name]={"input_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"seeds":[]}
        for arm in data["arms"]:
            fits=_fits([r for r in arm["rows"] if r["split"]=="held_field"],protocol["options"])
            results[name]["seeds"].append({"seed":arm["seed"],"fits":fits,
                "nu_range":[min(f["nu"] for f in fits),max(f["nu"] for f in fits)],
                "hc_range":[min(f["hc"] for f in fits),max(f["hc"] for f in fits)],
                "any_fit_at_bound":any(f["at_bound"] for f in fits)})
    write_json(args.output/"neural.json",{"kind":"held_field_scaling_sensitivity_diagnostic",
        "arms":results,"protocol_sha256":hashlib.sha256(declaration.read_bytes()).hexdigest(),
        "sampling_confidence_intervals":False,"universality_established":False,
        "scientific_gate_unchanged":True})
    print({name:[{"seed":s["seed"],"nu_range":s["nu_range"],"any_fit_at_bound":s["any_fit_at_bound"]}
                for s in value["seeds"]] for name,value in results.items()})


if __name__ == "__main__":
    main()
