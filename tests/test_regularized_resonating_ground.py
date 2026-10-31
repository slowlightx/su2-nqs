from __future__ import annotations

import numpy as np

from su2_nqs.vmc.regularized_resonating_ground import (
    crossfit_resonating_ridge,
    evaluate_resonating_ground,
    fit_regularized_resonating_ground,
)


def _problem(seed: int = 2026092208):
    rng = np.random.default_rng(seed)
    dimension = 24
    raw = rng.normal(size=(dimension, dimension))
    hamiltonian = 0.5 * (raw + raw.T)
    carrier = rng.normal(size=dimension)
    # One useful direction and one deliberately noisy/high-energy direction.
    useful = hamiltonian @ carrier
    noisy = rng.normal(size=dimension)
    bank = np.column_stack((carrier, useful, noisy)).astype(np.complex128)
    actions = hamiltonian @ bank
    proposal = np.full(dimension, 1.0 / dimension)
    return hamiltonian, bank, actions, proposal


def test_zero_ridge_is_variational_and_large_ridge_recovers_nested_carrier() -> None:
    _, bank, actions, proposal = _problem()
    carrier = evaluate_resonating_ground((1.0, 0.0, 0.0), bank, actions, proposal)
    unregularized = fit_regularized_resonating_ground(
        bank, actions, proposal, ridge=0.0
    )
    assert unregularized.energy <= carrier["energy"] + 1.0e-12
    strong = fit_regularized_resonating_ground(
        bank, actions, proposal, ridge=1.0e8
    )
    observed = evaluate_resonating_ground(
        strong.coefficients, bank, actions, proposal
    )
    np.testing.assert_allclose(observed["energy"], carrier["energy"], atol=1.0e-6)
    assert strong.resonance_norm_fraction < 1.0e-12


def test_grouped_crossfit_is_deterministic_and_returns_full_sample_fit() -> None:
    _, bank, actions, proposal = _problem(2026092209)
    # Four persistent chains and six blocks; index modulo chain_count is the
    # chain identity used by production MCMC outputs.
    result = crossfit_resonating_ridge(
        bank,
        actions,
        proposal,
        chain_count=4,
        fold_count=4,
        ridge_grid=(0.0, 1.0e-3, 1.0e-2, 0.1, 1.0, 10.0, 1.0e3),
    )
    assert result["selected_ridge"] in {
        row["ridge"] for row in result["grid"]
    }
    assert result["fit"].sample_count == len(bank)
    assert len(result["fit"].coefficients) == bank.shape[1]
    repeated = crossfit_resonating_ridge(
        bank,
        actions,
        proposal,
        chain_count=4,
        fold_count=4,
        ridge_grid=(0.0, 1.0e-3, 1.0e-2, 0.1, 1.0, 10.0, 1.0e3),
    )
    assert repeated["selected_ridge"] == result["selected_ridge"]
    np.testing.assert_allclose(
        repeated["fit"].coefficients, result["fit"].coefficients, atol=1.0e-14
    )
