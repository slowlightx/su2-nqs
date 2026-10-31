from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.clebsch_tree_exact import reynolds_project_state
from su2_nqs.benchmarks.four_cell_operator_scalar_frontier import (
    OPERATOR_LABELS,
    _case_data,
    _init_params,
    _state,
)


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "case_name", ("triangular_n12_j2_0125", "square_n16_j2_05")
)
def test_operator_bank_is_hermitian_and_scramble_preserves_invariants(
    case_name: str,
) -> None:
    data = _case_data(case_name)
    physical = np.asarray(data["physical_operators"])
    scrambled = np.asarray(data["scrambled_operators"])
    assert physical.shape[0] == len(OPERATOR_LABELS)
    np.testing.assert_allclose(
        physical, physical.conj().transpose(0, 2, 1), atol=3.0e-13
    )
    np.testing.assert_allclose(
        scrambled, scrambled.conj().transpose(0, 2, 1), atol=3.0e-13
    )
    np.testing.assert_allclose(
        np.trace(physical, axis1=1, axis2=2),
        np.trace(scrambled, axis1=1, axis2=2),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        np.linalg.norm(physical, axis=(1, 2)),
        np.linalg.norm(scrambled, axis=(1, 2)),
        atol=2.0e-13,
    )


@pytest.mark.parametrize(
    "case_name", ("triangular_n12_j2_0125", "square_n16_j2_05")
)
def test_operator_state_and_tangent_are_exact_su2_a1(case_name: str) -> None:
    data = _case_data(case_name)
    params = _init_params(jax.random.PRNGKey(23), data, correlated=True)
    params["operator_correlator"] = {
        "real": jnp.linspace(-0.02, 0.02, len(OPERATOR_LABELS)),
        "imag": jnp.linspace(0.01, -0.01, len(OPERATOR_LABELS)),
    }
    tangent = jax.tree_util.tree_map(jnp.zeros_like, params)
    tangent["operator_correlator"]["real"] = jnp.linspace(
        0.01, -0.01, len(OPERATOR_LABELS)
    )
    state, derivative = jax.jvp(
        lambda candidate: _state(
            candidate, data, data["physical_operators"]
        ),
        (params,),
        (tangent,),
    )
    singlet_basis = np.asarray(data["singlet_basis"])
    for vector in (np.asarray(state), np.asarray(derivative)):
        norm = max(float(np.linalg.norm(vector)), np.finfo(np.float64).tiny)
        singlet_residual = np.linalg.norm(
            vector - singlet_basis @ (singlet_basis.conj().T @ vector)
        ) / norm
        projected = np.asarray(
            reynolds_project_state(jnp.asarray(vector), data["indices"])
        )
        spatial_residual = np.linalg.norm(vector - projected) / norm
        assert singlet_residual < 5.0e-13
        assert spatial_residual < 5.0e-13
