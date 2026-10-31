from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.crossfit_regularized_operator import (
    RESONANCE_INDICES,
    crossfit_select_ridge,
    krylov_control_fit,
    regularized_generalized_ritz,
    self_normalized_energy,
)


def _sampled_problem(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    components = generator.normal(size=(280, 7))
    # The Krylov column 1 is the unique low-energy direction.  Resonance
    # columns are deliberately higher, so conservative CV should shrink them.
    spectrum = np.asarray([0.5, -2.0, 1.2, -0.5, 1.0, 1.4, 1.8])
    hamiltonian = components * spectrum[None, :]
    return components.astype(np.complex128), hamiltonian.astype(np.complex128)


def test_regularized_ritz_shrinks_exactly_toward_nested_krylov_space() -> None:
    components, hamiltonian = _sampled_problem()
    unregularized = regularized_generalized_ritz(
        components, hamiltonian, ridge=0.0
    )
    strongly_regularized = regularized_generalized_ritz(
        components, hamiltonian, ridge=1.0e8
    )
    control = krylov_control_fit(components, hamiltonian)

    assert strongly_regularized.resonance_norm_squared < 1.0e-12
    assert strongly_regularized.resonance_norm_squared <= unregularized.resonance_norm_squared
    assert np.isclose(
        strongly_regularized.training_energy,
        control.training_energy,
        atol=1.0e-8,
    )
    assert np.isclose(
        self_normalized_energy(
            strongly_regularized.coefficients, components, hamiltonian
        ),
        strongly_regularized.training_energy,
    )
    assert np.allclose(
        strongly_regularized.coefficients[np.asarray(RESONANCE_INDICES)],
        0.0,
        atol=1.0e-6,
    )


def test_crossfit_one_se_rule_returns_strongest_supported_shrinkage() -> None:
    components, hamiltonian = _sampled_problem()
    grid = (0.0, 1.0e-4, 1.0e-2, 1.0, 1.0e4)
    fit, diagnostics = crossfit_select_ridge(
        components,
        hamiltonian,
        chain_count=28,
        fold_count=4,
        ridge_grid=grid,
    )
    assert diagnostics["selected_ridge"] == max(grid)
    assert len(diagnostics["grid"]) == len(grid)
    assert fit.resonance_norm_squared < 1.0e-8
    assert diagnostics["selection_rule"].startswith("largest_ridge")


def test_crossfit_rejects_splitting_measurements_instead_of_chains() -> None:
    components, hamiltonian = _sampled_problem()
    try:
        crossfit_select_ridge(
            components[:-1],
            hamiltonian[:-1],
            chain_count=28,
            fold_count=4,
            ridge_grid=(0.0, 1.0),
        )
    except ValueError as error:
        assert "multiple" in str(error)
    else:
        raise AssertionError("misaligned persistent chains must be rejected")


def test_regularized_fit_accepts_non_axis_aligned_krylov_control() -> None:
    components, hamiltonian = _sampled_problem(91)
    basis = np.zeros((7, 3), dtype=np.float64)
    basis[:, 0] = (1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
    basis[:, 1] = (0.1, 1.0, 0.4, 0.2, 0.0, 0.0, 0.0)
    basis[:, 2] = (0.3, -0.1, 0.2, 1.0, 0.5, 0.2, 0.0)
    control = krylov_control_fit(
        components, hamiltonian, control_basis=basis
    )
    shrunk = regularized_generalized_ritz(
        components,
        hamiltonian,
        ridge=1.0e10,
        control_basis=basis,
    )
    assert np.isclose(shrunk.training_energy, control.training_energy, atol=1e-7)
    assert shrunk.resonance_norm_squared < 1.0e-12


def test_regularized_fit_supports_a_thirteen_column_three_token_space() -> None:
    generator = np.random.default_rng(20260920)
    components = generator.normal(size=(390, 13)).astype(np.complex128)
    hamiltonian = components * np.linspace(-1.2, 1.7, 13)[None, :]
    basis = generator.normal(size=(13, 3))
    control = krylov_control_fit(components, hamiltonian, control_basis=basis)
    shrunk = regularized_generalized_ritz(
        components,
        hamiltonian,
        ridge=1.0e11,
        control_basis=basis,
    )
    assert shrunk.coefficients.shape == (13,)
    assert np.isclose(shrunk.training_energy, control.training_energy, atol=1e-7)
    assert shrunk.resonance_norm_squared < 1.0e-12
