from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_symmetry_loop_coupling_scan import (
    _validate,
)
from su2_nqs.benchmarks.qsl_three_arm import triangular_pi_flux_operator_components
from su2_nqs.benchmarks.symmetry_labelled_loop_coupling_scan import (
    _response_metrics,
)
from su2_nqs.benchmarks.symmetry_labelled_loop_resonance import (
    triangular_labelled_loop_operators,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/"
    "spin_symmetry_loop_coupling_scan.json"
)


def test_symmetry_loop_coupling_scan_contract_is_frozen() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    parsed = _build_parser().parse_args(
        ["spin-symmetry-loop-coupling-scan", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-symmetry-loop-coupling-scan"


def test_A2_moment_and_return_are_consistent_on_exact_singlet() -> None:
    data = triangular_pi_flux_operator_components(0.125)
    operators = triangular_labelled_loop_operators()
    state = data.carrier / np.linalg.norm(data.carrier)
    metrics = _response_metrics(state, operators, data.hamiltonian)
    a2 = metrics["moments"]["A2_directed_ring"]
    image = operators.a2_current @ state
    np.testing.assert_allclose(a2["m0_total"], np.vdot(image, image).real, rtol=2e-14)
    assert a2["sma_gap"] > 0.0
    assert np.isfinite(a2["sma_susceptibility_proxy_per_site"])
    assert metrics["static_A2_expectation_abs"] < 1.0e-12
    assert metrics["ritz"]["A2_return"]["rank"] == 3
    assert metrics["ritz"]["matched_degree2_krylov"]["rank"] == 3
