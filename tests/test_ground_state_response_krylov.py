"""Physical spectral counterexamples and independently diagonalized checks."""
import numpy as np
import pytest

from su2_nqs.benchmarks.ground_state_response_krylov import (
    one_step_response_from_moments, response_krylov,
)


def test_identical_centroid_and_width_do_not_determine_lowest_pole():
    # Both normalized positive spectra have mean .75 and variance .5625.
    # Their lowest poles differ by more than a factor of three.
    distributions = [([.15, 1.6875], [25/41, 16/41]), ([.5, 3.], [.9, .1])]
    moments = []
    for energy, weight in distributions:
        e, w = np.array(energy), np.array(weight)
        m = np.array([w @ e**r for r in range(4)])
        moments.append(m)
        response = response_krylov(lambda x: e*x, np.sqrt(w), max_dimension=2)
        quadrature = one_step_response_from_moments(m)
        np.testing.assert_allclose(response.energies, e, atol=2e-14, rtol=0)
        np.testing.assert_allclose(response.source_weights, w, atol=2e-14, rtol=0)
        np.testing.assert_allclose(quadrature['energies'], e, atol=2e-14, rtol=0)
        np.testing.assert_allclose(quadrature['source_weights'], w, atol=2e-14, rtol=0)
        assert max(response.residual_norms) < 2e-14
    np.testing.assert_allclose(moments[0][:3], moments[1][:3], atol=2e-14, rtol=0)
    assert abs(moments[0][3] - moments[1][3]) > .5


def test_full_krylov_matches_dense_hermitian_spectrum_and_resolves_dark_source():
    rng = np.random.default_rng(140326)
    raw = rng.normal(size=(9, 9)) + 1j*rng.normal(size=(9, 9))
    h = raw + raw.conj().T
    energies, vectors = np.linalg.eigh(h)
    source = rng.normal(size=9) + 1j*rng.normal(size=9)
    source /= np.linalg.norm(source)
    response = response_krylov(lambda x: h @ x, source, max_dimension=9)
    np.testing.assert_allclose(response.energies, energies, atol=3e-14, rtol=0)
    np.testing.assert_allclose(response.source_weights, abs(vectors.conj().T @ source)**2,
                               atol=3e-14, rtol=0)
    assert response.orthonormality_error < 1e-14
    assert max(response.residual_norms) < 3e-14
    dark = response_krylov(lambda x: h @ x, vectors[:, 3], max_dimension=9)
    assert len(dark.energies) == 1
    np.testing.assert_allclose(dark.energies, energies[3:4], atol=3e-14, rtol=0)


def test_unresolved_moment_matrix_is_rejected():
    with pytest.raises(ValueError):
        one_step_response_from_moments(np.array([1., 2., 3., 4.]))
    with pytest.raises(ValueError):
        response_krylov(lambda x: np.roll(x, 1), np.array([1, 1j, .2]), max_dimension=3)
