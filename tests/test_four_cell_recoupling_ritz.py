from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_four_cell_recoupling_ritz import (
    _validate,
)
from su2_nqs.benchmarks.four_cell_overlapping_recoupling_frontier import (
    _case_data,
)
from su2_nqs.benchmarks.four_cell_recoupling_ritz_frontier import _ritz_state


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation"
    / "spin_four_cell_recoupling_ritz_calibration.json"
)


def _energy(state: np.ndarray, hamiltonian: np.ndarray) -> float:
    return float(np.vdot(state, hamiltonian @ state).real / np.vdot(state, state).real)


def test_recoupling_ritz_config_and_command_are_preregistered() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    parsed = _build_parser().parse_args(
        ["spin-four-cell-recoupling-ritz", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-four-cell-recoupling-ritz"
    assert spec["optimization"]["seeds"] == [811, 812, 813]
    assert spec["comparison"]["report_one_lanczos"] is True


@pytest.mark.parametrize(
    "case_name", ("triangular_n12_j2_0125", "square_n16_j2_05")
)
def test_ritz_subspaces_are_variational_and_matched(case_name: str) -> None:
    data = _case_data(case_name)
    physical_a1 = np.asarray(data["physical_a1_basis"])
    hamiltonian = physical_a1.T @ (data["hamiltonian"] @ physical_a1)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
    state = np.random.default_rng(37).normal(size=hamiltonian.shape[0])
    state /= np.linalg.norm(state)
    raw_energy = _energy(state, hamiltonian)
    lanczos, lanczos_rank = _ritz_state(state, hamiltonian, None)
    physical, physical_rank = _ritz_state(
        state, hamiltonian, np.asarray(data["physical_operators"])
    )
    scrambled, scrambled_rank = _ritz_state(
        state, hamiltonian, np.asarray(data["scrambled_operators"])
    )
    assert _energy(lanczos, hamiltonian) <= raw_energy + 2.0e-12
    assert _energy(physical, hamiltonian) <= raw_energy + 2.0e-12
    assert _energy(scrambled, hamiltonian) <= raw_energy + 2.0e-12
    assert lanczos_rank <= 2
    assert physical_rank == scrambled_rank
    assert physical_rank <= 9
