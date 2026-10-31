from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.clebsch_tree_exact import reynolds_project_state
from su2_nqs.benchmarks.four_cell_overlapping_recoupling_frontier import (
    _case_data,
    _init_params,
    _state,
)
from su2_nqs.benchmarks.overlapping_schur_recoupling import (
    translated_tiling_orbit,
)


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("case_name", "expected_count", "site_count"),
    (
        ("triangular_n12_j2_0125", 6, 12),
        ("square_n16_j2_05", 4, 16),
    ),
)
def test_translated_tilings_are_distinct_overlapping_partitions(
    case_name: str,
    expected_count: int,
    site_count: int,
) -> None:
    tilings = translated_tiling_orbit(case_name)
    assert len(tilings) == expected_count
    signatures = set()
    for tiling in tilings:
        assert sorted(site for support in tiling for site in support) == list(
            range(site_count)
        )
        signature = tuple(sorted(tuple(sorted(support)) for support in tiling))
        signatures.add(signature)
    assert len(signatures) == expected_count
    primary_supports = {frozenset(support) for support in tilings[0]}
    assert any(
        any(
            0 < len(frozenset(left) & frozenset(right)) < len(left)
            for left in tilings[0]
            for right in tiling
        )
        and {frozenset(support) for support in tiling} != primary_supports
        for tiling in tilings[1:]
    )


@pytest.mark.parametrize(
    ("case_name", "a1_dimension", "tiling_count"),
    (
        ("triangular_n12_j2_0125", 4, 6),
        ("square_n16_j2_05", 31, 4),
    ),
)
def test_overlapping_operator_bank_is_exact_a1_and_isospectral(
    case_name: str,
    a1_dimension: int,
    tiling_count: int,
) -> None:
    data = _case_data(case_name)
    diagnostics = data["representation_diagnostics"]
    assert diagnostics["gamma_a1_dimension"] == a1_dimension
    assert diagnostics["tiling_count"] == tiling_count
    assert diagnostics["operator_count"] == 8
    assert diagnostics["spatial_residual"] < 3.0e-13
    assert diagnostics["maximum_view_gram_error"] < 3.0e-13
    physical = np.asarray(data["physical_operators"])
    scrambled = np.asarray(data["scrambled_operators"])
    np.testing.assert_allclose(
        physical, physical.transpose(0, 2, 1), atol=3.0e-13
    )
    np.testing.assert_allclose(
        scrambled, scrambled.transpose(0, 2, 1), atol=3.0e-13
    )
    for left, right in zip(physical, scrambled, strict=True):
        np.testing.assert_allclose(
            np.linalg.eigvalsh(left), np.linalg.eigvalsh(right), atol=3.0e-13
        )
    assert np.linalg.norm(physical - scrambled) > 1.0e-2


@pytest.mark.parametrize(
    "case_name", ("triangular_n12_j2_0125", "square_n16_j2_05")
)
def test_overlapping_state_and_parameter_tangent_are_exact_su2_a1(
    case_name: str,
) -> None:
    data = _case_data(case_name)
    params = _init_params(jax.random.PRNGKey(29), data, correlated=True)
    count = len(data["operator_labels"])
    params["recoupling_correlator"] = {
        "real": jnp.linspace(-0.02, 0.02, count),
        "imag": jnp.linspace(0.01, -0.01, count),
    }
    tangent = jax.tree_util.tree_map(jnp.zeros_like, params)
    tangent["recoupling_correlator"]["real"] = jnp.linspace(
        0.01, -0.01, count
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
            vector - singlet_basis @ (singlet_basis.T @ vector)
        ) / norm
        projected = np.asarray(
            reynolds_project_state(jnp.asarray(vector), data["indices"])
        )
        spatial_residual = np.linalg.norm(vector - projected) / norm
        assert singlet_residual < 5.0e-13
        assert spatial_residual < 5.0e-13
