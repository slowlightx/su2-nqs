#!/usr/bin/env python3
"""Development-only comparison of preregistrable low-dimensional subspaces."""

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
    crossfit_select_ridge,
)
from su2_nqs.benchmarks.operator_vit_sampled import (  # noqa: E402
    carrier_standardization_for_labels,
    centered_scalar_component_matrices,
    effective_word_actions,
    expanded_base_words,
    scalar_component_words,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256  # noqa: E402
from su2_nqs.operators.triangular_a1_scalar import (  # noqa: E402
    TRIANGULAR_A1_SCALAR_LABELS,
)


RIDGE_GRID = (
    0.0,
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    3.0e-1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1.0e3,
    3.0e3,
    1.0e4,
)


def _cached_rows(root: Path, count: int) -> np.ndarray:
    rows = []
    for index in range(count):
        with np.load(
            root / "component_cache" / f"develop_{index:05d}.npz",
            allow_pickle=False,
        ) as stored:
            rows.append(np.asarray(stored["actions"], dtype=np.complex128)[0])
    return np.asarray(rows)


def _selector(component_count: int) -> np.ndarray:
    basis = np.zeros((component_count, 3), dtype=np.float64)
    basis[:3] = np.eye(3)
    return basis


def analyze(root: Path) -> dict[str, object]:
    source = json.loads((root / "results.json").read_text())
    sealed = dict(source)
    recorded = sealed.pop("result_sha256")
    if canonical_sha256(sealed) != recorded:
        raise ValueError("development result hash mismatch")
    if source.get("phase") != "development" or source.get("status") != "complete":
        raise ValueError("source must be a complete development artifact")
    count = int(source["sample_identity"]["count"])
    chain_count = int(source["sample_identity"]["chain_count"])
    coupling = float(source["config"]["j2_over_j1"])
    base_words = tuple(tuple(word) for word in source["words"])
    raw = _cached_rows(root, count)

    effective_labels = ("H_J1J2", "A1_ring")
    expansions = {
        effective_labels[0]: {
            TRIANGULAR_A1_SCALAR_LABELS[0]: 1.0,
            TRIANGULAR_A1_SCALAR_LABELS[1]: coupling,
        },
        effective_labels[1]: {TRIANGULAR_A1_SCALAR_LABELS[2]: 1.0},
    }
    effective_components = scalar_component_words(effective_labels)
    effective_words = tuple(
        dict.fromkeys(
            (
                *effective_components,
                *((effective_labels[0], *word) for word in effective_components),
            )
        )
    )
    required_base = expanded_base_words(effective_words, expansions)
    if any(word not in set(base_words) for word in required_base):
        raise ValueError("development cache lacks the H+ring effective basis")
    effective_raw = effective_word_actions(
        raw, base_words, effective_words, expansions
    )
    means, scales, _ = carrier_standardization_for_labels(
        effective_raw, effective_words, effective_labels
    )
    components, action = centered_scalar_component_matrices(
        effective_raw,
        effective_words,
        labels=effective_labels,
        hamiltonian_label=effective_labels[0],
        means=means,
        scales=scales,
    )
    centers = effective_raw[:, effective_words.index(())]
    components /= centers[:, None]
    action /= centers[:, None]
    # Effective ordering: I,H,R,HH,HR,RH,RR.  Put the exact nested control
    # first, followed by fixed physical response combinations.
    control = (0, 1, 3)
    candidates = {
        "linear_ring_4d": (
            components[:, control + (2,)],
            action[:, control + (2,)],
        ),
        "ring_symmetric_hr_5d": (
            np.column_stack(
                (components[:, control], components[:, 2], components[:, 4] + components[:, 5])
            ),
            np.column_stack(
                (action[:, control], action[:, 2], action[:, 4] + action[:, 5])
            ),
        ),
        "ring_commutator_hr_5d": (
            np.column_stack(
                (components[:, control], components[:, 2], components[:, 4] - components[:, 5])
            ),
            np.column_stack(
                (action[:, control], action[:, 2], action[:, 4] - action[:, 5])
            ),
        ),
        "ring_even_6d": (
            np.column_stack(
                (
                    components[:, control],
                    components[:, 2],
                    components[:, 4] + components[:, 5],
                    components[:, 6],
                )
            ),
            np.column_stack(
                (
                    action[:, control],
                    action[:, 2],
                    action[:, 4] + action[:, 5],
                    action[:, 6],
                )
            ),
        ),
        "full_h_ring_7d": (
            components[:, (0, 1, 3, 2, 4, 5, 6)],
            action[:, (0, 1, 3, 2, 4, 5, 6)],
        ),
    }
    rows = {}
    for name, (local_components, local_action) in candidates.items():
        fit, crossfit = crossfit_select_ridge(
            local_components,
            local_action,
            chain_count=chain_count,
            fold_count=8,
            ridge_grid=RIDGE_GRID,
            control_basis=_selector(local_components.shape[1]),
        )
        selected = next(
            row
            for row in crossfit["grid"]
            if row["ridge"] == crossfit["selected_ridge"]
        )
        rows[name] = {
            "dimension": local_components.shape[1],
            "crossfit": crossfit,
            "selected_non_krylov_norm_fraction": fit.resonance_norm_squared,
            "development_gate": bool(
                selected["mean_energy_difference"]
                + 2.0 * selected["fold_standard_error"]
                < 0.0
                and crossfit["selected_ridge"] < max(RIDGE_GRID)
                and fit.resonance_norm_squared > 1.0e-8
            ),
        }
    return {
        "schema_version": 1,
        "task": "spin_g1_regularized_development_subspaces",
        "claim_status": "development_only_no_audit_samples_created",
        "source_result_sha256": recorded,
        "families": rows,
        "any_audit_licensed": any(row["development_gate"] for row in rows.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = analyze(args.result_root)
    atomic_json(args.output, payload)
    print(json.dumps({name: row["development_gate"] for name, row in payload["families"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
