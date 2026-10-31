#!/usr/bin/env python3
"""Freeze the single minimum-mean-CV candidate for one untouched audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from su2_nqs.benchmarks.crossfit_regularized_operator import (  # noqa: E402
    krylov_control_fit,
    regularized_generalized_ritz,
)
from su2_nqs.benchmarks.operator_vit_sampled import (  # noqa: E402
    centered_scalar_component_matrices,
    krylov_control_basis,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256  # noqa: E402
from su2_nqs.operators.triangular_a1_scalar import (  # noqa: E402
    TRIANGULAR_A1_SCALAR_LABELS,
)


def freeze(root: Path) -> dict[str, object]:
    source = json.loads((root / "results.json").read_text())
    sealed = dict(source)
    source_hash = sealed.pop("result_sha256")
    if canonical_sha256(sealed) != source_hash:
        raise ValueError("development result hash mismatch")
    if source.get("phase") != "development" or source.get("status") != "complete":
        raise ValueError("source must be a complete development artifact")
    count = int(source["sample_identity"]["count"])
    raw = []
    for index in range(count):
        with np.load(
            root / "component_cache" / f"develop_{index:05d}.npz",
            allow_pickle=False,
        ) as stored:
            raw.append(np.asarray(stored["actions"], dtype=np.complex128)[0])
    raw_array = np.asarray(raw)
    words = tuple(tuple(word) for word in source["words"])
    means = np.asarray(source["standardization"]["means"], dtype=np.float64)
    scales = np.asarray(source["standardization"]["scales"], dtype=np.float64)
    coupling = float(source["config"]["j2_over_j1"])
    components, action = centered_scalar_component_matrices(
        raw_array,
        words,
        labels=TRIANGULAR_A1_SCALAR_LABELS[:3],
        hamiltonian_weights=(1.0, coupling, 0.0),
        means=means,
        scales=scales,
    )
    centers = raw_array[:, words.index(())]
    components /= centers[:, None]
    action /= centers[:, None]
    basis = krylov_control_basis(
        hamiltonian_weights=(1.0, coupling, 0.0),
        means=means,
        scales=scales,
    )
    ridge = float(source["crossfit"]["best_mean_ridge"])
    selected_row = next(
        row for row in source["crossfit"]["grid"] if float(row["ridge"]) == ridge
    )
    candidate = regularized_generalized_ritz(
        components, action, ridge=ridge, control_basis=basis
    )
    control = krylov_control_fit(components, action, control_basis=basis)
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": "spin_g1_regularized_operator",
        "phase": "development",
        "status": "complete",
        "claim_status": "single_candidate_frozen_for_untouched_confirmation",
        "source_result_sha256": source_hash,
        "config": source["config"],
        "words": source["words"],
        "standardization": source["standardization"],
        "control_basis": basis.tolist(),
        "selection": {
            "rule": "minimum_mean_grouped_crossfit_energy_difference",
            "ridge": ridge,
            "mean_energy_difference_total": selected_row["mean_energy_difference"],
            "fold_standard_error_total": selected_row["fold_standard_error"],
            "development_difference_resolved": bool(
                selected_row["mean_energy_difference"]
                + 2.0 * selected_row["fold_standard_error"]
                < 0.0
            ),
            "candidate_count_opened_on_final_audit": 1,
        },
        "frozen_candidate": {
            "coefficients": candidate.coefficients.tolist(),
            "ridge": ridge,
            "non_krylov_norm_fraction": candidate.resonance_norm_squared,
        },
        "frozen_two_lanczos": {"coefficients": control.coefficients.tolist()},
        "decision": {
            "audit_licensed": True,
            "criterion": (
                "one minimum-mean grouped-CV candidate may be tested once; "
                "development is selection, not evidence"
            ),
            "untouched_audit_created": False,
        },
    }
    payload["result_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = freeze(args.result_root)
    atomic_json(args.output, payload)
    print(payload["result_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
