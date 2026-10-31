from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.clebsch_tree_exact import (
    fixed_m_spin_basis,
    reynolds_project_state,
)
from su2_nqs.benchmarks.four_cell_conditioned_response_frontier import (
    _energy_state,
    _haar_orthogonal,
)
from su2_nqs.benchmarks.four_cell_overlapping_recoupling_frontier import (
    _case_data,
)
from su2_nqs.benchmarks.symmetry_labelled_response import (
    build_symmetry_labelled_response_a1,
)
from su2_nqs.models.su2_conditioned_response_gate import (
    init_su2_conditioned_response_gate,
    su2_conditioned_response_operators,
)
from su2_nqs.models.su2_four_cell_sector_vit import (
    su2_four_cell_sector_log_amplitudes,
)
from projects.clebsch_tree_nqs.workflows.spin_conditioned_response_gate import (
    _validate,
)


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[1]


def test_conditioned_response_development_contract_is_frozen() -> None:
    config = ROOT / (
        "projects/clebsch_tree_nqs/configs/validation/"
        "spin_conditioned_response_gate_development.json"
    )
    spec = json.loads(config.read_text())
    _validate(spec)
    mutated = json.loads(config.read_text())
    mutated["optimization"]["seeds"][-1] += 1
    with pytest.raises(ValueError, match="seeds changed"):
        _validate(mutated)


@pytest.fixture(scope="module")
def conditioned_cases():
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


def test_conditioned_operators_are_hermitian_and_pointwise_isospectral(
    conditioned_cases,
) -> None:
    for case_name, (data, response) in conditioned_cases.items():
        params = init_su2_conditioned_response_gate(
            jax.random.PRNGKey(201), data["config"], len(response.labels)
        )
        dimension = int(data["physical_a1_basis"].shape[1])
        rotation = _haar_orthogonal(dimension, 203)
        physical, gates = su2_conditioned_response_operators(
            params,
            data["actions"],
            data["config"],
            data["coefficient_a1_basis"],
            response.operators,
        )
        scrambled, scrambled_gates = su2_conditioned_response_operators(
            params,
            data["actions"],
            data["config"],
            data["coefficient_a1_basis"],
            response.operators,
            scramble_rotation=rotation,
        )
        physical = np.asarray(physical)
        scrambled = np.asarray(scrambled)
        np.testing.assert_allclose(gates, scrambled_gates, atol=0.0)
        np.testing.assert_allclose(
            physical, physical.conj().transpose(0, 2, 1), atol=4.0e-13
        )
        np.testing.assert_allclose(
            np.trace(physical, axis1=1, axis2=2), 0.0, atol=4.0e-13
        )
        np.testing.assert_allclose(
            np.linalg.norm(physical, axis=(1, 2)), 1.0, atol=4.0e-13
        )
        for left, right in zip(physical, scrambled, strict=True):
            np.testing.assert_allclose(
                np.linalg.eigvalsh(left),
                np.linalg.eigvalsh(right),
                atol=5.0e-13,
            )
        assert np.min(np.std(np.asarray(gates), axis=0)) > 1.0e-3
        logs = su2_four_cell_sector_log_amplitudes(
            params["backbone"], data["actions"], data["config"]
        )
        np.testing.assert_allclose(
            np.sum(np.exp(2.0 * np.real(np.asarray(logs)))),
            1.0,
            atol=3.0e-13,
        )


def test_conditioned_state_and_parameter_tangent_remain_exact_su2_a1(
    conditioned_cases,
) -> None:
    data, response = conditioned_cases["square_n16_j2_05"]
    params = init_su2_conditioned_response_gate(
        jax.random.PRNGKey(211), data["config"], len(response.labels)
    )

    def state_fn(candidate):
        logs = su2_four_cell_sector_log_amplitudes(
            candidate["backbone"], data["actions"], data["config"]
        )
        coefficients = jnp.exp(logs - jnp.max(jnp.real(logs)))
        reduced = data["coefficient_a1_basis"].T @ coefficients
        operators, _gates = su2_conditioned_response_operators(
            candidate,
            data["actions"],
            data["config"],
            data["coefficient_a1_basis"],
            response.operators,
        )
        reduced = operators[1] @ reduced
        physical = data["physical_a1_basis"] @ reduced
        return physical / jnp.linalg.norm(physical)

    direction = jax.tree_util.tree_map(jnp.ones_like, params)
    state, tangent = jax.jvp(state_fn, (params,), (direction,))
    singlet = np.asarray(data["singlet_basis"])
    for vector in (np.asarray(state), np.asarray(tangent)):
        norm = max(float(np.linalg.norm(vector)), 1.0e-30)
        assert np.linalg.norm(vector - singlet @ (singlet.T @ vector)) / norm < 5.0e-13
        projected = np.asarray(
            reynolds_project_state(vector, data["indices_numpy"])
        )
        assert np.linalg.norm(vector - projected) / norm < 5.0e-13


def test_conditioned_variable_projection_gradient_is_finite(
    conditioned_cases,
) -> None:
    data, response = conditioned_cases["square_n16_j2_05"]
    physical_a1 = np.asarray(data["physical_a1_basis"])
    hamiltonian = physical_a1.T @ (data["hamiltonian"] @ physical_a1)
    hamiltonian = jnp.asarray(0.5 * (hamiltonian + hamiltonian.T))
    params = init_su2_conditioned_response_gate(
        jax.random.PRNGKey(223), data["config"], len(response.labels)
    )
    rotation = jnp.asarray(_haar_orthogonal(physical_a1.shape[1], 227))

    def loss(candidate):
        return _energy_state(
            candidate,
            data,
            hamiltonian,
            data["physical_operators"],
            response.operators,
            rotation,
            "conditioned_labelled_response",
        )[0]

    energy, gradients = jax.value_and_grad(loss)(params)
    assert np.isfinite(float(energy))
    assert all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(gradients)
    )
    gate_gradient = gradients["response_gate"]["head"]
    assert np.linalg.norm(np.asarray(gate_gradient)) > 1.0e-12
