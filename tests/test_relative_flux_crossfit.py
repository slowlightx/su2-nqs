from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.relative_flux_crossfit import (
    apply_relative_flux_basis,
    fit_relative_flux_basis,
)


def test_development_relative_flux_modes_are_whitened() -> None:
    generator = np.random.default_rng(8)
    latent = generator.normal(size=(32, 3)) + 1j * generator.normal(size=(32, 3))
    mixing = generator.normal(size=(3, 4)) + 1j * generator.normal(size=(3, 4))
    ratios = latent @ mixing
    basis = fit_relative_flux_basis(
        ratios, relative_eigenvalue_cutoff=1.0e-10
    )
    modes = apply_relative_flux_basis(ratios, basis)
    gram = modes.conj().T @ modes / len(modes)
    np.testing.assert_allclose(gram, np.eye(3), atol=2.0e-12)
    assert basis.retained_count == 3


def test_relative_flux_basis_is_invariant_to_carrier_column_scale() -> None:
    generator = np.random.default_rng(11)
    ratios = generator.normal(size=(40, 4)) + 1j * generator.normal(size=(40, 4))
    phases = np.asarray((2.0, -3.0j, 0.25, 5.0j))
    first = fit_relative_flux_basis(ratios, relative_eigenvalue_cutoff=1.0e-8)
    second = fit_relative_flux_basis(
        ratios * phases[None, :], relative_eigenvalue_cutoff=1.0e-8
    )
    modes_first = apply_relative_flux_basis(ratios, first)
    modes_second = apply_relative_flux_basis(ratios * phases[None, :], second)
    overlap = modes_first.conj().T @ modes_second / len(ratios)
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    np.testing.assert_allclose(singular_values, np.ones(4), atol=2.0e-12)
