from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_symmetry_response_operator import (
    _validate,
)
from projects.clebsch_tree_nqs.workflows.spin_square_b2_response_audit import (
    _validate as validate_b2_audit,
)
from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.four_cell_overlapping_recoupling_frontier import (
    _case_data,
)
from su2_nqs.benchmarks.four_cell_recoupling_variable_projection import (
    _variable_projection_energy_state,
)
from su2_nqs.benchmarks.symmetry_labelled_response import (
    build_symmetry_labelled_response_a1,
)
from su2_nqs.models.su2_four_cell_sector_vit import (
    init_su2_four_cell_sector_vit,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation"
    / "spin_symmetry_response_operator_development.json"
)
B2_CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation"
    / "spin_square_b2_response_audit.json"
)


@pytest.fixture(scope="module")
def response_data():
    answer = {}
    for case_name in ("triangular_n12_j2_0125", "square_n16_j2_05"):
        data = _case_data(case_name)
        response = build_symmetry_labelled_response_a1(
            case_name,
            np.asarray(data["physical_a1_basis"]),
            np.asarray(data["singlet_basis"]),
            fixed_m_spin_basis(int(data["n_sites"]), 0),
        )
        answer[case_name] = (data, response)
    return answer


def test_response_config_and_command_are_preregistered() -> None:
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert spec["optimization"]["seeds"] == [1111, 1112, 1113, 1114, 1115]
    parsed = _build_parser().parse_args(
        ["spin-symmetry-response-operator", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-symmetry-response-operator"
    b2_spec = json.loads(B2_CONFIG.read_text())
    validate_b2_audit(b2_spec)
    assert b2_spec["optimization"]["seeds"] == [1211, 1212, 1213, 1214, 1215]
    parsed_b2 = _build_parser().parse_args(
        ["spin-square-b2-response-audit", "--config", str(B2_CONFIG)]
    )
    assert parsed_b2.command == "spin-square-b2-response-audit"


def test_response_loops_have_exact_irreps_and_isospectral_controls(
    response_data,
) -> None:
    expected = {
        "triangular_n12_j2_0125": (
            "E2_J1_nematic_return",
            "A2_chirality_return",
        ),
        "square_n16_j2_05": (
            "B1_J1_nematic_return",
            "B2_J2_nematic_return",
            "A2_chirality_return",
        ),
    }
    for case_name, (_data, response) in response_data.items():
        assert response.labels == expected[case_name]
        assert response.diagnostics["maximum_exact_certificate_error"] < 2.0e-12
        np.testing.assert_allclose(
            response.operators,
            response.operators.conj().transpose(0, 2, 1),
            atol=3.0e-13,
        )
        for physical, scrambled, moment in zip(
            response.operators,
            response.scrambled_operators,
            response.moment_operators,
            strict=True,
        ):
            np.testing.assert_allclose(
                np.linalg.eigvalsh(physical),
                np.linalg.eigvalsh(scrambled),
                atol=4.0e-13,
            )
            assert np.min(np.linalg.eigvalsh(moment)) > -3.0e-13


def test_augmented_response_variable_projection_has_finite_gradient(
    response_data,
) -> None:
    data, response = response_data["square_n16_j2_05"]
    physical_a1 = np.asarray(data["physical_a1_basis"])
    hamiltonian = physical_a1.T @ (data["hamiltonian"] @ physical_a1)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
    operators = np.concatenate(
        (np.asarray(data["physical_operators"]), response.operators)
    )
    params = init_su2_four_cell_sector_vit(
        jax.random.PRNGKey(73), data["config"]
    )

    def loss(candidate):
        return _variable_projection_energy_state(
            candidate, data, jax.numpy.asarray(hamiltonian), operators
        )[0]

    energy, gradients = jax.value_and_grad(loss)(params)
    assert np.isfinite(float(energy))
    assert all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(gradients)
    )
