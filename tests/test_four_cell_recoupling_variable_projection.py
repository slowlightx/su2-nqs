from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_four_cell_recoupling_variable_projection import (
    _validate,
)
from su2_nqs.benchmarks.clebsch_tree_exact import reynolds_project_state
from su2_nqs.benchmarks.four_cell_overlapping_recoupling_frontier import (
    _case_data,
)
from su2_nqs.benchmarks.four_cell_recoupling_variable_projection import (
    _variable_projection_energy_state,
)
from su2_nqs.models.su2_four_cell_sector_vit import (
    init_su2_four_cell_sector_vit,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation"
    / "spin_four_cell_recoupling_variable_projection_development.json"
)


def test_variable_projection_config_and_command_are_preregistered() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    parsed = _build_parser().parse_args(
        [
            "spin-four-cell-recoupling-variable-projection",
            "--config",
            str(CONFIG),
        ]
    )
    assert parsed.command == "spin-four-cell-recoupling-variable-projection"
    assert spec["optimization"]["seeds"] == [911, 912, 913]


@pytest.mark.parametrize(
    "case_name", ("triangular_n12_j2_0125", "square_n16_j2_05")
)
def test_variable_projection_loss_gradient_and_state_are_exact(
    case_name: str,
) -> None:
    data = _case_data(case_name)
    physical_a1 = np.asarray(data["physical_a1_basis"])
    hamiltonian = physical_a1.T @ (data["hamiltonian"] @ physical_a1)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
    params = init_su2_four_cell_sector_vit(
        jax.random.PRNGKey(41), data["config"]
    )
    for operators in (
        data["physical_operators"],
        data["scrambled_operators"],
    ):
        loss_fn = lambda candidate: _variable_projection_energy_state(
            candidate, data, hamiltonian, operators
        )[0]
        energy, gradients = jax.value_and_grad(loss_fn)(params)
        assert np.isfinite(float(energy))
        assert all(
            np.all(np.isfinite(np.asarray(leaf)))
            for leaf in jax.tree_util.tree_leaves(gradients)
        )
        _energy, reduced = _variable_projection_energy_state(
            params, data, hamiltonian, operators
        )
        state = physical_a1 @ np.asarray(reduced)
        projected = np.asarray(
            reynolds_project_state(state, data["indices_numpy"])
        )
        np.testing.assert_allclose(state, projected, atol=5.0e-13)
        singlet = np.asarray(data["singlet_basis"])
        np.testing.assert_allclose(
            state, singlet @ (singlet.T @ state), atol=5.0e-13
        )
