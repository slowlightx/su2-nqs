"""Verify paired checkpoints, sealed measurement provenance and declared costs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous verification; choose a new output")
    protocol_path = args.pair / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    checks = []
    for name in ("warm", "scratch"):
        config = protocol["configs"][name]
        result_path = args.pair / name / "neural.json"
        data = json.loads(result_path.read_text())
        assert data["kind"] == "shared_neural_phase0"
        assert data["source_sha256"] == protocol["source_sha256"]
        assert data["config"] == config
        assert [arm["seed"] for arm in data["arms"]] == config["seeds"]
        expected_points = {(n, h) for n in config["sizes"]
                           for h in config["train_fields"] + config["test_fields"]}
        assert not set(config["train_fields"]) & set(config["test_fields"])
        for arm in data["arms"]:
            checkpoint = args.pair / name / f"seed_{arm['seed']}.npz"
            with np.load(checkpoint, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
                work = json.loads(str(archive["work"]))
                assert int(archive["step"]) == arm["final_step"] == config["steps"]
                assert metadata["source_sha256"] == protocol["source_sha256"]
                assert metadata["seed"] == arm["seed"]
                assert metadata["warm_start"] == arm["warm_start"]
                assert work == arm["training_work"]
                parameters = [archive[key] for key in archive.files if key.startswith("params__")]
                assert parameters and all(np.isfinite(value).all() for value in parameters)
                parameter_count = sum(value.size for value in parameters)
            assert {(row["n"], row["h"]) for row in arm["rows"]} == expected_points
            assert len(arm["rows"]) == len(expected_points)
            assert all(row["audit_samples"] == config["audit_samples"] for row in arm["rows"])
            assert all(row["split"] == ("train" if row["h"] in config["train_fields"] else "held_field")
                       for row in arm["rows"])
            assert work["gradient_hamiltonians"] == protocol["primary_gradient_hamiltonians_per_arm_seed"]
            assert work["born_samples"] == protocol["training_born_draws_per_arm_seed"]
            audit_draws = sum(row["audit_samples"] for row in arm["rows"])
            assert audit_draws == protocol["audit_born_draws_per_arm_seed"]
            checks.append({"arm": name, "seed": arm["seed"], "passed": True,
                "checkpoint_sha256": _sha(checkpoint), "measurement_sha256": _sha(result_path),
                "parameter_count": parameter_count, "training_work": work,
                "audit_points": len(expected_points), "audit_born_draws": audit_draws})
    result = {"kind": "paired_checkpoint_and_measurement_integrity_validation",
              "passed": True, "checks": checks,
              "protocol_sha256": _sha(protocol_path),
              "measurement_source_sha256": protocol["source_sha256"],
              "launcher_sha256": _sha(Path(__file__)),
              "scientific_accuracy_implied": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print({"passed": True, "checkpoints": len(checks)})


if __name__ == "__main__":
    main()
