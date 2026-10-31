"""Physical Schur-channel diagnosis of sealed exact-SU2 N=36 trajectories."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import jax
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.vmc.resumable_checkpoint import load_resumable_checkpoint
from su2_nqs.vmc.scale_first_arms import EXACT_SU2_ARM

from .spin_scale_first_carrier_checkpoint_audit import (
    _build_carrier_runtime,
    _checkpoint_path,
    _load_contract as _load_checkpoint_audit_contract,
)


jax.config.update("jax_enable_x64", True)

TASK = "spin_scale_first_carrier_schur_channel_audit"
METRICS = (
    "cell_q0_fraction",
    "cell_q2_fraction",
    "cell_q4_fraction",
    "cell_channel_entropy",
    "intercell_nonzero_fraction",
)


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1) / math.sqrt(len(values)))


def _load_contract(
    config_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    Path,
]:
    spec = json.loads(config_path.read_text())
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid Schur-channel audit identity")
    audit_path = _resolve(config_path, spec["source_checkpoint_audit"])
    audit, training, base, initialization, artifact_root = (
        _load_checkpoint_audit_contract(audit_path)
    )
    controls = spec.get("controls")
    if not isinstance(controls, dict):
        raise TypeError("controls must be an object")
    seeds = controls.get("sample_seeds")
    count = controls.get("sample_count_per_seed")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("sample_seeds must contain at least two unique integers")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("sample_count_per_seed must be an integer of at least two")
    forbidden = set(audit["controls"]["audit_seeds"])
    forbidden.update(base.get("production_contract", {}).get("seeds", []))
    if forbidden.intersection(seeds):
        raise ValueError("Schur-channel seeds must be fresh")
    return spec, audit, training, base, initialization, artifact_root


def _metrics(labels: np.ndarray, layout) -> dict[str, float]:
    decoded = layout.decode(labels)
    cell = np.asarray(decoded.cell_q, dtype=np.int32)
    intercell = np.asarray(decoded.intercell_q, dtype=np.int32)
    entropies = []
    for column in range(cell.shape[1]):
        probabilities = [float(np.mean(cell[:, column] == q)) for q in (0, 2, 4)]
        entropies.append(-sum(p * math.log(p) for p in probabilities if p > 0.0))
    return {
        "cell_q0_fraction": float(np.mean(cell == 0)),
        "cell_q2_fraction": float(np.mean(cell == 2)),
        "cell_q4_fraction": float(np.mean(cell == 4)),
        "cell_channel_entropy": float(np.mean(entropies)),
        "intercell_nonzero_fraction": float(np.mean(intercell > 0)),
    }


def run_spin_scale_first_carrier_schur_channel_audit(
    config_path: Path,
    checkpoint_root: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Measure how training redistributes exact local/inter-cell spin channels."""

    spec, audit, training, base, initialization, artifact_root = _load_contract(
        config_path
    )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "results.json"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {destination}")
    rows = []
    summaries = []
    controls = spec["controls"]
    for case_index, case in enumerate(base["systems"]):
        runtime, initial, initial_hash, identity, _manifest = _build_carrier_runtime(
            training=training,
            base=base,
            initialization=initialization,
            artifact_root=artifact_root,
            case=case,
            case_index=case_index,
            arm_id=EXACT_SU2_ARM,
        )
        final_checkpoint = load_resumable_checkpoint(
            _checkpoint_path(
                checkpoint_root, str(case["case_id"]), EXACT_SU2_ARM, 64
            ),
            expected_run_identity=identity,
        )
        by_step: dict[int, dict[int, dict[str, float]]] = {0: {}, 64: {}}
        for step, params in ((0, initial), (64, final_checkpoint["params"])):
            for seed in controls["sample_seeds"]:
                seed = int(seed)
                labels = np.asarray(
                    runtime.sample_fn(
                        params,
                        jax.random.PRNGKey(seed),
                        int(controls["sample_count_per_seed"]),
                    )
                )
                values = _metrics(labels, runtime.model_config.layout)
                by_step[step][seed] = values
                rows.append(
                    {
                        "case_id": str(case["case_id"]),
                        "checkpoint_step": step,
                        "sample_seed": seed,
                        "sample_count": int(controls["sample_count_per_seed"]),
                        **values,
                    }
                )
        summary: dict[str, object] = {
            "case_id": str(case["case_id"]),
            "initial_params_sha256": initial_hash,
            "sample_seed_count": len(controls["sample_seeds"]),
        }
        for step in (0, 64):
            summary[f"step_{step}"] = {
                metric: {
                    "mean": float(np.mean([row[metric] for row in by_step[step].values()])),
                    "standard_error": _standard_error(
                        [row[metric] for row in by_step[step].values()]
                    ),
                }
                for metric in METRICS
            }
        summary["paired_change_64_minus_0"] = {
            metric: {
                "mean": float(
                    np.mean(
                        [
                            by_step[64][seed][metric] - by_step[0][seed][metric]
                            for seed in controls["sample_seeds"]
                        ]
                    )
                ),
                "standard_error": _standard_error(
                    [
                        by_step[64][seed][metric] - by_step[0][seed][metric]
                        for seed in controls["sample_seeds"]
                    ]
                ),
            }
            for metric in METRICS
        }
        summaries.append(summary)
    result: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_sha256(spec),
        "source_checkpoint_audit_config_sha256": canonical_sha256(audit),
        "rows": rows,
        "summaries": summaries,
        "runtime_identity": numerical_runtime_identity(),
        "training_performed": False,
        "production_release_allowed": False,
        "claim_boundary": spec["claim_boundary"],
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    return result


__all__ = ["TASK", "_load_contract", "_metrics", "run_spin_scale_first_carrier_schur_channel_audit"]
