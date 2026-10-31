from __future__ import annotations

import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_n36_audit import (
    _autocorrelation_report,
    _configuration,
    _edge_bank,
)


def _spec() -> dict[str, object]:
    return {
        "case": {"j2_over_j1": 0.125},
        "model": {
            "feature_dim": 2,
            "maximum_intermediate_q": 10,
            "initialization_scale": 0.2,
            "embedding_dim": 8,
            "head_count": 2,
            "depth": 1,
            "mlp_ratio": 2,
            "modulation_rank": 1,
            "modulation_scale": 0.1,
        },
    }


def test_dimer_complete_n36_bank_adds_third_orientation_and_edges() -> None:
    config, _bonds, ledger = _configuration(_spec())
    assert config.n_sites == 36
    assert config.n_routes == 9
    assert "diagonal_dimers" in config.spatial.route_names
    assert ledger["physical_extension"] == "three_orientation_dimer_complete"
    row = _edge_bank(config, "row")
    complete = _edge_bank(config, "dimer_complete")
    assert len(row) == 18
    assert len(complete) == 54
    assert set(row) < set(complete)


def test_autocorrelation_report_distinguishes_mixed_and_sticky_series() -> None:
    rng = np.random.default_rng(9)
    mixed = rng.normal(size=(64, 8))
    sticky = np.cumsum(rng.normal(size=(64, 8)), axis=0)
    mixed_report = _autocorrelation_report(mixed)
    sticky_report = _autocorrelation_report(sticky)
    assert 0.0 < mixed_report["effective_sample_size_fraction"] <= 1.0
    assert sticky_report["effective_sample_size_fraction"] < mixed_report[
        "effective_sample_size_fraction"
    ]
