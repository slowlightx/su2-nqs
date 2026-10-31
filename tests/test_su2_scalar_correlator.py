from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.clebsch_tree_exact import reynolds_project_state
from su2_nqs.benchmarks.four_cell_scalar_correlator_frontier import (
    _correlator_case_data,
    _init_params,
    _state,
)
from su2_nqs.models.su2_scalar_correlator import (
    init_schur_scalar_correlator,
    schur_scalar_features,
    schur_scalar_log_correction,
)


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "case_name", ("triangular_n12_j2_0125", "square_n16_j2_05")
)
def test_scalar_features_are_nonconstant_standardized_reduced_data(
    case_name: str,
) -> None:
    data = _correlator_case_data(case_name)
    features = schur_scalar_features(np.asarray(data["actions"]), data["config"])
    assert features.values.shape[0] == len(data["actions"])
    assert features.values.shape[1] == len(features.labels)
    assert features.values.shape[1] >= 20
    np.testing.assert_allclose(np.mean(features.values, axis=0), 0.0, atol=2.0e-14)
    np.testing.assert_allclose(np.std(features.values, axis=0), 1.0, atol=2.0e-14)


def test_zero_scalar_correlator_is_exact_identity() -> None:
    features = jnp.asarray(((0.0, 1.0), (2.0, -1.0)), dtype=jnp.float64)
    params = init_schur_scalar_correlator(2)
    np.testing.assert_array_equal(
        schur_scalar_log_correction(params, features),
        np.zeros(2, dtype=np.complex128),
    )


@pytest.mark.parametrize(
    "case_name", ("triangular_n12_j2_0125", "square_n16_j2_05")
)
def test_scalar_state_and_parameter_tangent_remain_exact_su2_a1(
    case_name: str,
) -> None:
    data = _correlator_case_data(case_name)
    params = _init_params(jax.random.PRNGKey(17), data, correlated=True)
    count = data["physical_features"].shape[1]
    params["scalar_correlator"] = {
        "log_magnitude": jnp.linspace(-0.03, 0.03, count),
        "phase": jnp.linspace(0.02, -0.02, count),
    }
    tangent = jax.tree_util.tree_map(jnp.zeros_like, params)
    tangent["scalar_correlator"]["log_magnitude"] = jnp.linspace(
        0.01, -0.01, count
    )
    state, derivative = jax.jvp(
        lambda candidate: _state(
            candidate, data, data["physical_features"]
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
        assert singlet_residual < 4.0e-13
        assert spatial_residual < 4.0e-13
