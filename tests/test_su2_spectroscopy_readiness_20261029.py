"""Exact quadrature tests for node-safe spectral widths and source summaries."""
import numpy as np
import pytest

from projects.clebsch_tree_nqs.explorations.su2_spectroscopy_readiness_20261029 import source_diagnostics


def test_eigenstate_width_nodes_and_total_energy_units():
    h = np.array([[1., .4, 0.], [.4, 2., 0.], [0., 0., 3.]])
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    parent = eigenvectors[:, 0]
    # Source has an exact target node but Hsource does not vanish there.
    source = np.array([1., 0., 0.])
    p = np.column_stack((parent, source, eigenvectors[:, 1], eigenvectors[:, 2]))
    q = np.full(3, 1/3)
    result = source_diagnostics(p, h@p, q)
    assert result[0, 0] == pytest.approx(eigenvalues[0], abs=3e-16)
    assert result[0, 2] < 5e-16
    assert result[1, 0] == pytest.approx(1.)
    assert result[1, 2] == pytest.approx(.4)
    assert result[1, 4] == pytest.approx(1-eigenvalues[0])
    # Power-of-two scaling isolates exact scale cancellation from an extra
    # inexact factor-nine accumulation; retain the strict absolute tolerance.
    np.testing.assert_allclose(source_diagnostics(2j*p, 2j*(h@p), q), result, atol=5e-16, rtol=0)
    assert result[-1, 0] == pytest.approx((eigenvalues[1]+eigenvalues[2])/2)
    assert result[-1, 3] == pytest.approx(2.)


def test_nonpositive_proposal_is_not_repaired():
    with pytest.raises(ValueError):
        source_diagnostics(np.ones((3, 3)), np.ones((3, 3)), np.array([1., 0., 1.]))


def test_M_star_width_includes_between_channel_spread_without_coherence():
    h = np.diag([0., 1., 3., 5.])
    basis = np.eye(4)
    # Three sharp M sources have unequal norms 1,1,4 and energies 1,3,5.
    p = np.column_stack((basis[:, 0], basis[:, 1], basis[:, 1],
                         basis[:, 2], 2*basis[:, 3]))
    q = np.full(4, .25)
    result = source_diagnostics(p, h@p, q)
    np.testing.assert_allclose(result[2:5, 2], 0., atol=1e-15, rtol=0)
    assert result[-1, 0] == pytest.approx(4., abs=1e-15)
    assert result[-1, 2] == pytest.approx(np.sqrt(7/3), abs=1e-15)
    assert result[-1, 3] == pytest.approx(6., abs=1e-15)
    # Independent channel phases cannot affect an incoherent direct sum.
    phases = np.asarray([1., 1., 1j, -1., -1j])
    np.testing.assert_allclose(source_diagnostics(p*phases, (h@p)*phases, q),
                               result, atol=1e-15, rtol=0)
