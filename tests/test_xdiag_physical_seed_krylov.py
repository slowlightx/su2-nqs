from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.exact_diagonalization import computational_basis
from su2_nqs.benchmarks.projected_singlet_agp import (
    triangular_hnf_gamma_a1_group,
)
from su2_nqs.benchmarks.xdiag_physical_seed_krylov import (
    SMALL_CASES,
    configuration_from_repository_integer,
    configuration_from_xdiag_uint,
    generated_hnf_gamma_a1_toml,
    ordered_gamma_a1_group,
    physical_seed_amplitudes,
    polynomial_block_krylov,
    repository_integer_from_configuration,
    xdiag_uint_from_configuration,
)


def _pullback(states: np.ndarray, image: tuple[int, ...]) -> np.ndarray:
    return states[:, np.asarray(image, dtype=np.int64)]


def test_xdiag_and_repository_integer_conversions_are_inverse() -> None:
    for n_sites in (12, 16, 36):
        generator = np.random.default_rng(2026091300 + n_sites)
        for _ in range(100):
            configuration = tuple(
                int(value) for value in generator.integers(0, 2, size=n_sites)
            )
            repository = repository_integer_from_configuration(configuration)
            xdiag = xdiag_uint_from_configuration(configuration)
            assert configuration_from_repository_integer(repository, n_sites) == configuration
            assert configuration_from_xdiag_uint(xdiag, n_sites) == configuration


def test_generated_inputs_have_registered_group_orders() -> None:
    for case in SMALL_CASES.values():
        text = generated_hnf_gamma_a1_toml(case)
        assert text.count("[1.0, 0.0]") == 12 * case.n_sites
        assert "'J1', 'SdotS'" in text
        assert "'J2', 'SdotS'" in text


def test_all_registered_exact_small_physical_seeds_are_A1() -> None:
    for case in SMALL_CASES.values():
        basis = computational_basis(case.n_sites, case.n_sites // 2)
        generator = np.random.default_rng(2026091400 + case.n_sites)
        selected = generator.choice(len(basis), min(64, len(basis)), replace=False)
        states = np.asarray(
            [
                configuration_from_repository_integer(basis[index], case.n_sites)
                for index in selected
            ],
            dtype=np.int8,
        )
        amplitudes = physical_seed_amplitudes(states, case)
        assert tuple(case.seed_names) == (
            ("pi0_DSL_A1", "stripe_tower_A1")
            if case.geometry_id == "square4"
            else ("120_tower_A1",)
        )
        assert np.all(np.linalg.norm(amplitudes, axis=0) > 1.0e-10)
        group = ordered_gamma_a1_group(case)
        # Generators occur somewhere in the closed group; checking every image
        # remains cheap at N<=16 and protects the direct three-term reductions.
        for image in group:
            transformed = physical_seed_amplitudes(_pullback(states, image), case)
            np.testing.assert_allclose(transformed, amplitudes, atol=3.0e-13)


def test_polynomial_block_krylov_matches_explicit_power_span() -> None:
    generator = np.random.default_rng(2026091318)
    raw = generator.normal(size=(9, 9))
    hamiltonian = 0.5 * (raw + raw.T)
    seeds = generator.normal(size=(9, 2))
    seeds /= np.linalg.norm(seeds, axis=0, keepdims=True)
    rows = polynomial_block_krylov(hamiltonian, seeds, 4)
    powers = []
    frontier = np.array(seeds, copy=True)
    for depth, row in enumerate(rows):
        powers.append(frontier)
        span = np.column_stack(powers)
        expected_rank = np.linalg.matrix_rank(span, tol=1.0e-11)
        assert row["basis_dimension"] == expected_rank
        frontier = hamiltonian @ frontier
        if depth == 4:
            np.testing.assert_allclose(
                row["energies"], np.linalg.eigvalsh(hamiltonian)[:3], atol=2.0e-12
            )
            assert max(row["direct_residual_norms"]) < 3.0e-12
