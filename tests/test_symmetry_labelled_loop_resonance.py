from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_symmetry_labelled_loop_screen import (
    _validate,
)
from su2_nqs.benchmarks.qsl_three_arm import triangular_pi_flux_operator_components
from su2_nqs.benchmarks.symmetry_labelled_loop_resonance import (
    _covariance_diagnostics,
    _labelled_return_columns,
    triangular_labelled_loop_operators,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN = (
    ROOT
    / "projects/clebsch_tree_nqs/results/"
    "spin_qsl_pi_flux_three_arm_frontier_v1/results.json"
)
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/"
    "spin_symmetry_labelled_loop_screen.json"
)


def test_labelled_loop_screen_contract_is_frozen() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    parsed = _build_parser().parse_args(
        ["spin-symmetry-labelled-loop-screen", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-symmetry-labelled-loop-screen"


def _frozen_state():
    artifact = json.loads(FROZEN.read_text())
    coefficients = np.asarray(
        artifact["frontier"]["pi_flux_exact_su2_a1_operator_vit_trials"][0][
            "operator_coefficients"
        ]
    )
    data = triangular_pi_flux_operator_components(0.125)
    first, second = coefficients
    weights = np.concatenate(
        (np.ones(1), first + second, np.outer(second, first).reshape(-1))
    )
    state = data.component_matrix @ weights
    state /= np.linalg.norm(state)
    return data, state


def test_labelled_loop_operators_have_exact_e2_a2_and_return_a1() -> None:
    data, state = _frozen_state()
    operators = triangular_labelled_loop_operators()
    diagnostics = _covariance_diagnostics(
        state, operators, data.configurations, data.integer_basis
    )
    for key in (
        "e2_ring_covariance_max_relative_error",
        "e2_dimer_covariance_max_relative_error",
        "a2_covariance_max_relative_error",
        "returned_a1_max_relative_error",
        "maximum_splus_relative_norm",
        "maximum_elementary_hermiticity_error",
    ):
        assert diagnostics[key] < 1.0e-10, (key, diagnostics[key])
    assert diagnostics["minimum_return_expectation"] > -1.0e-10


def test_labelled_loop_returns_are_nontrivial_beyond_state() -> None:
    _data, state = _frozen_state()
    operators = triangular_labelled_loop_operators()
    returns = _labelled_return_columns(state, operators)
    for vector in returns:
        residual = vector - state * np.vdot(state, vector)
        assert np.linalg.norm(residual) / np.linalg.norm(vector) > 1.0e-4
