from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_schur_spatial_scaling_preflight import (
    _validate,
)
from su2_nqs.benchmarks.schur_spatial_scaling import (
    profile_schur_spatial_scaling,
    spin_half_singlet_dimension,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation"
    / "spin_schur_spatial_scaling_preflight.json"
)


def test_singlet_dimension_matches_catalan_sequence() -> None:
    assert [spin_half_singlet_dimension(n) for n in (2, 4, 6, 8, 10)] == [
        1,
        2,
        5,
        14,
        42,
    ]


def test_small_scaling_profile_keeps_exact_singlets_and_balanced_cost() -> None:
    report = profile_schur_spatial_scaling(
        geometry="triangular",
        linear_size=2,
        j2_over_j1=0.125,
        sample_count=4,
        connectivity_sample_count=2,
        seed=71,
        embedding_dim=8,
        head_count=2,
        depth=1,
        timing_repeats=1,
    )
    assert report["singlet_dimension"] == 2
    assert report["all_samples_valid_singlets"]
    assert report["balanced_connectivity"]["maximum_width"] >= 1
    assert report["forward_samples_per_second"] > 0.0
    assert report["gradient_samples_per_second"] > 0.0


def test_scaling_preflight_cli_and_config_are_registered() -> None:
    parsed = _build_parser().parse_args(
        ["spin-schur-spatial-scaling-preflight", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-schur-spatial-scaling-preflight"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert {(row["geometry"], row["linear_size"]) for row in spec["systems"]} == {
        ("triangular", 4),
        ("triangular", 6),
        ("triangular", 8),
        ("square", 4),
        ("square", 6),
        ("square", 8),
    }
