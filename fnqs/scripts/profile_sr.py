"""CPU resource calibration, without interpreting a short VMC run as physics.

From the repository root: python -m fnqs.scripts.profile_sr --output ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import jax

from fnqs.artifacts import save_checkpoint, write_json
from fnqs.experiments import provenance
from fnqs.configuration import validate_config
from fnqs.initialization import initialize_run
from fnqs.hamiltonians import IsingCouplings
from fnqs.models import GRUConfig, init_model
from fnqs.training import Task, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--config", type=Path, help="profile the first seed of a declared train config")
    args = parser.parse_args()
    if (args.output/"profile.json").exists():
        raise FileExistsError("use a new profiling output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text()) if args.config else {
        "model":{"hidden_dim":16,"size_scale":64},"sizes":[16,24,32],
        "train_fields":[.975,1.,1.025],"test_fields":[.9875,1.0125],"seeds":[17],
        "steps":args.steps,"mode":"vmc","optimizer":"sr","tasks_per_step":3,
        "samples_per_task":64,"audit_samples":128,"learning_rate":.5,"sr_shift":.001,
        "sr_solver":"cg","sr_tolerance":1e-7,"sr_maxiter":2000,"sr_max_step_norm":.5}
    config = dict(config, steps=args.steps, seeds=[config["seeds"][0]])
    validate_config("train",config)
    write_json(args.output/"config.json", config)
    model = GRUConfig(**config["model"])
    parent=Path(config["warm_start_checkpoint"]) if config.get("warm_start_checkpoint") else None
    params,state,key,origin = initialize_run(config["seeds"][0],model,parent)
    tasks = [Task(n, IsingCouplings(h=h)) for n in config["sizes"] for h in config["train_fields"]]
    times = []
    started = time.monotonic()
    def callback(record: dict) -> None:
        times.append({"step": record["step"], "seconds": time.monotonic()-started})
        print(times[-1], flush=True)
    params, state, key, history = train(params, tasks, model, key, steps=args.steps,
        mode=config["mode"],optimizer=config["optimizer"],learning_rate=config["learning_rate"],
        n_samples=config["samples_per_task"],sr_shift=config.get("sr_shift",.001),
        sr_solver=config.get("sr_solver","dense"),sr_tolerance=config.get("sr_tolerance",1e-8),
        sr_maxiter=config.get("sr_maxiter",500),sr_max_step_norm=config.get("sr_max_step_norm"),
        tasks_per_step=config.get("tasks_per_step"),state=state,callback=callback)
    jax.block_until_ready(params)
    elapsed = time.monotonic()-started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {**provenance(config), "kind": "sr_vmc_resource_profile", "neural_accuracy_audited": False,
              "universality_established": False, "elapsed_training_seconds": elapsed,
              "timings": times, "history": history, "training_work": state["work"],"warm_start":origin,
              "peak_rss_bytes": rss if sys.platform == "darwin" else 1024*rss,
              "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_checkpoint(args.output/"seed_17.npz", params, state, key, {"profile": result})
    write_json(args.output/"profile.json", result)
    print({"seconds": elapsed, "peak_rss_bytes": result["peak_rss_bytes"]}, flush=True)


if __name__ == "__main__":
    main()
