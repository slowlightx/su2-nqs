"""Explicit new optimization stage from a checkpoint, with a new audit grid.

This is not CLI resume: optimizer settings change and the origin is recorded.
The checkpoint is never resealed or overwritten to bypass resume validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from fnqs.artifacts import load_checkpoint, save_checkpoint, source_hashes, write_json
from fnqs.configuration import validate_config
from fnqs.experiments import _audit_exact, provenance
from fnqs.hamiltonians import IsingCouplings
from fnqs.models import GRUConfig
from fnqs.scaling import phase0_gate
from fnqs.training import Task, train


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    validate_config("train",config)
    if config["mode"]!="exact" or config["optimizer"]!="sr" or len(config["seeds"])!=1:
        raise ValueError("this refinement launcher requires exact SR and one parent seed")
    if args.output.exists():raise FileExistsError("use a new refinement output")
    params,parent_state,key,metadata=load_checkpoint(args.parent)
    parent_config=metadata.get("signature",metadata.get("config",{}))
    if metadata["source_sha256"]!=source_hashes() or parent_config.get("model")!=config["model"]:
        raise ValueError("parent numerical code/model differs")
    if metadata["seed"]!=config["seeds"][0]:raise ValueError("parent seed differs")
    seed=config["seeds"][0]
    expected_model_shapes={name:value.shape for name,value in params.items()}
    model=GRUConfig(**config["model"])
    args.output.mkdir(parents=True)
    origin={"parent":str(args.parent),"parent_sha256":hashlib.sha256(args.parent.read_bytes()).hexdigest(),
            "parent_steps":int(parent_state["step"]),"parent_training_work":parent_state["work"],
            "optimizer_restarted":True,"changed_settings":"full ensemble, lower shift and step norm",
            "launcher_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.output/"protocol.json",{**provenance(config),**origin,"universality_established":False})
    tasks=[Task(n,IsingCouplings(h=h)) for n in config["sizes"] for h in config["train_fields"]]
    started=time.monotonic()
    params,state,key,history=train(params,tasks,model,key,steps=config["steps"],learning_rate=config["learning_rate"],
        mode="exact",optimizer="sr",sr_shift=config["sr_shift"],sr_solver=config["sr_solver"],
        sr_tolerance=config["sr_tolerance"],sr_maxiter=config["sr_maxiter"],
        sr_max_step_norm=config["sr_max_step_norm"],sr_backtracking=config["sr_backtracking"],
        callback=lambda r:print({k:r[k] for k in ["step","energy_per_site","max_variance_per_site","gradient_norm"]},flush=True))
    assert {name:value.shape for name,value in params.items()}==expected_model_shapes
    save_checkpoint(args.output/f"seed_{seed}.npz",params,state,key,
        {"seed":seed,"config":config,"source_sha256":source_hashes(),"origin":origin,"history":history,
         "checkpoint_kind":"explicit_new_stage; use this launcher rather than CLI --resume"})
    rows=[]
    for n in config["sizes"]:
        for h in sorted(set(config["train_fields"]+config["test_fields"])):
            row=_audit_exact(params,model,n,IsingCouplings(h=h))
            row["split"]="train" if h in config["train_fields"] else "held_field"
            rows.append(row)
    accurate=all(r["energy_relative_error"]<=.002 and r["chi_relative_error"]<=.1 and
        r["max_correlation_abs_error"]<=.03 and r["norm_error"]<=1e-10 and
        r["ad_fidelity_abs_discrepancy"]<=1e-5*max(1.,r["chi_total"]) for r in rows)
    arm={"seed":seed,"initial_step":0,"final_step":int(state["step"]),"history":history,
         "rows":rows,"training_work":state["work"],"observable_checks_passed":accurate,
         "extracted":{"hc":None,"nu":None,"eta":None,"reason":"small-size response calibration only"}}
    result={**provenance(config),"kind":"full_ensemble_response_refinement",**origin,"arms":[arm],
            "phase0_scientific_gate":phase0_gate(None,None,None,accurate,config["sizes"],1),
            "gradient_hamiltonians_per_seed":len(tasks)*config["steps"],
            "elapsed_seconds":time.monotonic()-started,"universality_established":False}
    write_json(args.output/"neural.json",result)
    print({"small_response_passed":accurate,"seconds":result["elapsed_seconds"]},flush=True)


if __name__=="__main__":main()
