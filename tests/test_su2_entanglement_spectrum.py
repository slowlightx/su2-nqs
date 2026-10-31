from __future__ import annotations

import math

import numpy as np
import pytest

from su2_nqs.benchmarks.su2_entanglement_spectrum import (
    compare_entanglement_reports,
    reduced_density_matrix,
    su2_resolved_entanglement,
    subsystem_total_spin_squared,
    singlet_entropy_components,
)
from su2_nqs.su2.sz_basis import enumerate_sz_configs


def _singlet_product(
    pairs: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    n_sites = 2 * len(pairs)
    basis = enumerate_sz_configs(n_sites, n_sites // 2)
    amplitudes = np.zeros(len(basis), dtype=np.float64)
    lookup = {bits: index for index, bits in enumerate(basis)}
    for orientations in np.ndindex(*(2,) * len(pairs)):
        bits = [0] * n_sites
        sign = 1.0
        for orientation, (first, second) in zip(orientations, pairs, strict=True):
            if orientation == 0:
                bits[first], bits[second] = 0, 1
            else:
                bits[first], bits[second] = 1, 0
                sign *= -1.0
        amplitudes[lookup[tuple(bits)]] = sign / np.sqrt(2.0 ** len(pairs))
    return basis, amplitudes


def test_two_site_singlet_has_one_spin_half_boundary_sector() -> None:
    basis, state = _singlet_product(((0, 1),))
    report = su2_resolved_entanglement(state, basis, (0,))
    np.testing.assert_allclose(report["rho"], 0.5 * np.eye(2), atol=2.0e-15)
    assert report["von_neumann_entropy"] == pytest.approx(math.log(2.0))
    assert report["commutator_frobenius"] < 1.0e-14
    assert set(report["sectors"]) == {"1"}
    assert report["sectors"]["1"]["weight"] == pytest.approx(1.0)


def test_internal_and_crossed_singlets_resolve_distinct_boundary_spins() -> None:
    internal_basis, internal = _singlet_product(((0, 1), (2, 3)))
    internal_report = su2_resolved_entanglement(
        internal, internal_basis, (0, 1)
    )
    assert internal_report["von_neumann_entropy"] == pytest.approx(0.0, abs=1.0e-14)
    assert internal_report["sectors"]["0"]["weight"] == pytest.approx(1.0)
    assert internal_report["sectors"]["2"]["weight"] == pytest.approx(0.0, abs=1.0e-14)

    crossed_basis, crossed = _singlet_product(((0, 2), (1, 3)))
    crossed_report = su2_resolved_entanglement(crossed, crossed_basis, (0, 1))
    np.testing.assert_allclose(crossed_report["rho"], 0.25 * np.eye(4), atol=2.0e-15)
    assert crossed_report["von_neumann_entropy"] == pytest.approx(math.log(4.0))
    assert crossed_report["sectors"]["0"]["weight"] == pytest.approx(0.25)
    assert crossed_report["sectors"]["2"]["weight"] == pytest.approx(0.75)


def test_total_spin_squared_has_expected_four_spin_multiplicities() -> None:
    eigenvalues = np.linalg.eigvalsh(subsystem_total_spin_squared(4))
    expected = np.asarray([0.0] * 2 + [2.0] * 9 + [6.0] * 5)
    np.testing.assert_allclose(eigenvalues, expected, atol=3.0e-14)


def test_comparison_and_input_validation() -> None:
    basis, state = _singlet_product(((0, 2), (1, 3)))
    report = su2_resolved_entanglement(state, basis, (0, 1))
    comparison = compare_entanglement_reports(report, report)
    assert max(comparison.values()) < 1.0e-14
    with pytest.raises(ValueError, match="proper"):
        reduced_density_matrix(state, basis, ())
    with pytest.raises(ValueError, match="unique"):
        reduced_density_matrix(state, basis, (0, 0))
    with pytest.raises(ValueError, match="nonzero norm"):
        reduced_density_matrix(np.zeros_like(state), basis, (0, 1))


def test_singlet_entropy_decomposition_and_noninvariant_rejection() -> None:
    basis, state = _singlet_product(((0, 2), (1, 3)))
    report = su2_resolved_entanglement(state, basis, (0, 1))
    parts = singlet_entropy_components(report)
    assert parts["magnetic_entropy"] == pytest.approx(0.75 * math.log(3))
    assert parts["multiplicity_entropy"] == pytest.approx(0, abs=2e-14)
    assert parts["sum_error"] < 1e-14
    bad = dict(report, rho=np.diag([1., 0., 0., 0.]))
    with pytest.raises(ValueError, match="invariant"):
        singlet_entropy_components(bad)
