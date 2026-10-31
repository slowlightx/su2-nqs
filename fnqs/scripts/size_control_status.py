"""Read-only progress inspection plus an explicitly incomplete status artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np

from fnqs.artifacts import write_json


def status(directory: Path) -> dict:
    """Read atomic accepted checkpoints, without treating logs as audit evidence."""
    protocol = json.loads((directory / "protocol.json").read_text())
    trajectories = []
    for name, arm in protocol["arms"].items():
        for seed in protocol["seeds"]:
            root = directory / "runs" / name / f"seed_{seed}"
            accepted = root / "accepted.npz"
            committed = 0
            work = {}
            if accepted.exists():
                with np.load(accepted, allow_pickle=False) as archive:
                    committed = int(archive["step"])
                    work = json.loads(str(archive["work"]))
            trajectories.append({"arm": name, "seed": seed, "last_saved_accepted_step": committed,
                "target_step": max(arm["milestones"]), "saved_training_work": work,
                "completed_audits": [s for s in arm["milestones"] if (root / f"step_{s}" / "audit.json").exists()],
                "complete": (root / "complete.json").exists(), "failed": (root / "failure.json").exists()})
    frontiers = {}
    for path in sorted(directory.parent.glob("size_capacity_*_v8.log")):
        matches = re.findall(r"arm=(\S+) seed=(\d+) step=(\d+) E/N=", path.read_text())
        if matches:
            name, seed, step = matches[-1]
            frontiers[path.name] = {"arm": name, "seed": int(seed), "logged_step": int(step)}
    return {"kind": "size_capacity_running_status", "updated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest(),
        "completed_trajectories": sum(t["complete"] for t in trajectories),
        "total_trajectories": len(trajectories), "completed_audits": sum(len(t["completed_audits"]) for t in trajectories),
        "total_audits": sum(len(a["milestones"]) for a in protocol["arms"].values()) * len(protocol["seeds"]),
        "failed_trajectories": sum(t["failed"] for t in trajectories),
        "log_frontiers_not_audits": frontiers, "trajectories": trajectories,
        "science_interpretation": "incomplete unless all planned trajectories and audits are validated",
        "universality_established": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    result = status(args.directory)
    write_json(args.directory / "status.json", result)
    print(json.dumps({k:result[k] for k in ["updated_utc", "completed_trajectories", "total_trajectories", "completed_audits", "total_audits", "failed_trajectories", "log_frontiers_not_audits"]}))
