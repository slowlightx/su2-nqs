from __future__ import annotations

from math import sqrt

import numpy as np

from su2_nqs.benchmarks.xdiag_physical_seed_krylov import SMALL_CASES
from su2_nqs.benchmarks.xdiag_total_splus_bridge import (
    computational_total_splus,
    deterministic_probe_columns,
    dicke_m_zero_full_column,
    fixed_sz_sparse_hamiltonian_sector,
)


def test_dicke_positive_control_has_analytic_total_splus_norm() -> None:
    for n_sites in (4, 6, 12):
        source, basis = dicke_m_zero_full_column(n_sites)
        raised, target_basis = computational_total_splus(
            source, basis, n_sites
        )
        assert all(state.bit_count() == n_sites // 2 + 1 for state in target_basis)
        spin = n_sites / 2
        np.testing.assert_allclose(
            np.linalg.norm(raised), sqrt(spin * (spin + 1)), atol=2.0e-14
        )


def test_computational_total_splus_uses_repository_msb_site_convention() -> None:
    # |down,up,down,up> has repository integer 0101. Raising either down
    # spin yields 1101 and 0111, with unit spin-1/2 matrix elements.
    source_basis = (0b0101,)
    raised, target_basis = computational_total_splus(
        np.asarray([[2.5]]), source_basis, 4
    )
    observed = {
        state: raised[index, 0]
        for index, state in enumerate(target_basis)
        if raised[index, 0] != 0.0
    }
    assert observed == {0b0111: 2.5, 0b1101: 2.5}


def test_frozen_probe_formula_is_deterministic_and_normalized() -> None:
    first = deterministic_probe_columns(17)
    second = deterministic_probe_columns(17)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=0), 1.0, atol=2.0e-15)
    assert np.linalg.matrix_rank(first) == 3


def test_full_computational_hamiltonian_intertwines_total_splus() -> None:
    case = SMALL_CASES["hnf6x2t4_color"]
    source_nup = case.n_sites // 2
    h0, basis0 = fixed_sz_sparse_hamiltonian_sector(case, 0.125, source_nup)
    h1, basis1 = fixed_sz_sparse_hamiltonian_sector(
        case, 0.125, source_nup + 1
    )
    generator = np.random.default_rng(2026091402)
    source = generator.normal(size=(len(basis0), 2))
    source /= np.linalg.norm(source, axis=0, keepdims=True)
    splus_source, observed_basis1 = computational_total_splus(
        source, basis0, case.n_sites
    )
    splus_h_source, _ = computational_total_splus(
        h0 @ source, basis0, case.n_sites
    )
    assert observed_basis1 == basis1
    np.testing.assert_allclose(
        h1 @ splus_source, splus_h_source, atol=2.0e-14
    )


def test_sector_oracles_fail_closed_on_invalid_arguments() -> None:
    case = SMALL_CASES["square4"]
    try:
        fixed_sz_sparse_hamiltonian_sector(case, 0.125, True)
    except ValueError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("boolean n_up was accepted")
    try:
        computational_total_splus(np.ones((1, 1)), (), 4)
    except ValueError as error:
        assert "incompatible" in str(error) or "must not be empty" in str(error)
    else:
        raise AssertionError("empty source basis was accepted")
