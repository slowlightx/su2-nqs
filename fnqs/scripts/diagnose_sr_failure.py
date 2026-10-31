"""Replay a failed cold SR seed and retain its unaccepted linear system."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fnqs import training
from fnqs.artifacts import source_hashes, write_json
from fnqs.configuration import validate_config
from fnqs.hamiltonians import IsingCouplings
from fnqs.initialization import initialize_run
from fnqs.models import GRUConfig
from fnqs.training import Task, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config("train", config)
    if source_hashes()["training.py"] != "57728dc8b20ee21663476345b66985aee8a3541ecab2552ac3bdf2c3695bd2fb":
        raise ValueError("Replay requires the frozen runtime_v5 package; run this script with that PYTHONPATH")
    if config.get("warm_start_checkpoint") or args.seed not in config["seeds"]:
        raise ValueError("diagnostic requires a declared cold seed")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"kind":"sr_failure_diagnostic_only","config":config,"seed":args.seed,
                "source_sha256":source_hashes(),"launcher_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "observable_audits":False,"unaccepted_step_captured":True}
    write_json(args.output / "protocol.json", protocol)
    calls = 0
    original = training._solve_sr

    def capture(a, target, shift, *, solver, tolerance, maxiter):
        nonlocal calls
        calls += 1
        direction, residual, gradient = original(a,target,shift,solver=solver,tolerance=tolerance,maxiter=maxiter)
        value = float(residual)
        if not np.isfinite(value) or value > max(1e-7,10*tolerance):
            with (args.output / "failed_system.npz").open("wb") as stream:
                np.savez_compressed(stream,a=np.asarray(a),target=np.asarray(target),
                                    direction=np.asarray(direction),shift=shift,tolerance=tolerance,maxiter=maxiter)
            write_json(args.output / "failure.json", {
                "failed_iteration":calls,"true_relative_residual":value,"stop_threshold":max(1e-7,10*tolerance),
                "direction_accepted":False,"gradient_hamiltonians_consumed":calls*config["tasks_per_step"],
                "training_born_draws_consumed":calls*config["tasks_per_step"]*config["samples_per_task"],
                "system_sha256":hashlib.sha256((args.output / "failed_system.npz").read_bytes()).hexdigest(),
                "source_sha256":source_hashes()})
        return direction,residual,gradient

    training._solve_sr = capture
    model = GRUConfig(**config["model"])
    params,state,key,_ = initialize_run(args.seed,model)
    tasks = [Task(n,IsingCouplings(h=h)) for n in config["sizes"] for h in config["train_fields"]]
    try:
        train(params,tasks,model,key,steps=config["steps"],learning_rate=config["learning_rate"],
              mode=config["mode"],optimizer=config["optimizer"],n_samples=config["samples_per_task"],
              sr_shift=config["sr_shift"],state=state,
              **{name:config[name] for name in ("sr_solver","sr_tolerance","sr_maxiter","sr_max_step_norm","sr_backtracking","tasks_per_step")},
              callback=lambda row: print(f"diagnostic seed={args.seed} step={row['step']}",flush=True))
    except FloatingPointError as error:
        print(str(error),flush=True)
        if not (args.output / "failed_system.npz").exists():
            raise
    else:
        raise RuntimeError("The predeclared SR failure was not reproduced")


if __name__ == "__main__":
    main()
