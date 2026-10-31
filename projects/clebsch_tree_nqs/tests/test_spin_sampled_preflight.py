from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.workflows import spin, spin_sampled_preflight as preflight
from su2_nqs.frontier.artifacts import paired_initialization_audit
from su2_nqs.frontier.route_policy import FROZEN_PRODUCTION_MODULE_PREFIXES


def _spec() -> dict[str, object]:
    return {
        "schema_version": 2,
        "domain": "spin",
        "task": "exact_born_sampled_estimator_preflight",
        "model": {
            "feature_dim": 2,
            "maximum_intermediate_q": 4,
            "initialization_scale": 1.0,
        },
        "model_seeds": [11],
        "sampling": {
            "sample_counts": [32, 8192],
            "replicates": 8,
            "seed": 20260916,
        },
        "arms": [
            {"arm_id": "u1_none", "backbone": "u1", "projection": "none"},
            {
                "arm_id": "u1_translation",
                "backbone": "u1",
                "projection": "translation",
            },
            {"arm_id": "su2_none", "backbone": "su2", "projection": "none"},
            {
                "arm_id": "su2_translation",
                "backbone": "su2",
                "projection": "translation",
            },
        ],
        "cases": [
            {
                "case_id": "square_2x4_test",
                "geometry": "square",
                "rows": 2,
                "columns": 4,
                "j2_over_j1": 0.5,
                "support": "fixed_m0",
                "exact_energy_anchor": -6.0,
                "anchor_provenance": "stored test anchor; no eigensolve in runner",
                "arm_ids": [
                    "u1_none",
                    "u1_translation",
                    "su2_none",
                    "su2_translation",
                ],
            }
        ],
    }


def test_preflight_rejects_large_support_and_nonregistered_projection() -> None:
    spec = _spec()
    preflight._validate_spec(spec)
    too_large = deepcopy(spec)
    too_large["cases"][0]["rows"] = 4
    too_large["cases"][0]["columns"] = 6
    with pytest.raises(ValueError, match="N <= 16"):
        preflight._validate_spec(too_large)
    wrong_projection = deepcopy(spec)
    wrong_projection["arms"][0]["projection"] = "gamma_a1"
    with pytest.raises(ValueError, match="none and translation"):
        preflight._validate_spec(wrong_projection)


def test_preflight_source_closure_excludes_frozen_routes() -> None:
    manifest = preflight._source_manifest()
    assert "projects/clebsch_tree_nqs/workflows/spin_sampled_preflight.py" in manifest
    assert "projects/clebsch_tree_nqs/workflows/spin.py" in manifest
    forbidden_paths = {
        f"{prefix.replace('.', '/')}.py"
        for prefix in FROZEN_PRODUCTION_MODULE_PREFIXES
    }
    assert forbidden_paths.isdisjoint(manifest)


def test_exact_born_energy_and_gradient_preflight_all_registered_arms() -> None:
    spec = _spec()
    case_spec = spec["cases"][0]
    data = spin._prepare_case(case_spec, "fixed_m0")
    trials = []
    for arm in spec["arms"]:
        trial = preflight._run_trial(
            arm,
            model_seed=11,
            spec=spec,
            data=data,
            case_id=case_spec["case_id"],
        )
        trials.append(trial)
        metrics = trial["metrics"]
        assert metrics["exact_gradient_norm"] > 1.0e-12
        assert metrics["exact_local_energy_imaginary_abs_maximum"] < 1.0e-12
        summaries = metrics["sample_summaries"]
        assert [row["sample_count"] for row in summaries] == [32, 8192]
        assert summaries[-1]["gradient_cosine_mean"] > 0.9
        assert summaries[-1]["energy_rmse"] < summaries[0]["energy_rmse"]
        for summary in summaries:
            for key in (
                "energy_bias",
                "energy_rmse",
                "gradient_mean_bias_relative_norm",
                "gradient_rmse_relative_norm",
                "gradient_cosine_mean",
            ):
                assert np.isfinite(summary[key])
    audits = paired_initialization_audit(trials)
    assert len(audits) == 2
    assert all(row["passed"] for row in audits)
