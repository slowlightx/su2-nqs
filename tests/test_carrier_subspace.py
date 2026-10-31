"""Independent small dense references for a sampled fixed-bank Ritz fit."""
from __future__ import annotations

import json

import numpy as np
import pytest

from su2_nqs.vmc.carrier_subspace import fit_carrier_subspace
from su2_nqs.vmc.sampled_multistate import NumericalSubspaceRankError


def _fidelity(first, second, weights=None):
    w = np.ones(len(first)) if weights is None else weights
    return abs(np.sum(w * first.conj() * second)) ** 2 / (
        np.sum(w * abs(first) ** 2) * np.sum(w * abs(second) ** 2))


def test_exact_complex_ground_and_complete_bank_coefficient_mapping():
    # The lowest state of the first block is (1,i)/sqrt(2); restricting
    # coefficients to the real numbers would miss its entire energy gain.
    h = np.asarray([[0., 1j, 0., 0.], [-1j, 0., 0., 0.],
                    [0., 0., 2., 0.], [0., 0., 0., 3.]], dtype=complex)
    phi = np.eye(4, dtype=complex)[:, [0, 2, 1]]
    q = np.asarray([.1, .2, .3, .4])
    fit = fit_carrier_subspace(phi, h @ phi, q, indices=(0, 2),
                              observation_weights=q, n_sites=2)
    np.testing.assert_allclose(fit.statistics['energy_total'], -1., atol=1e-14)
    assert fit.coefficients[1] == 0
    assert abs((fit.coefficients[2] / fit.coefficients[0]).imag) > .9
    np.testing.assert_allclose(_fidelity(phi @ fit.coefficients,
        np.asarray([1., 1j, 0., 0.])), 1., atol=1e-14)
    assert fit.statistics['variance_total'] < 1e-28
    assert fit.statistics['relative_optimality_residual'] < 1e-14
    serialized = fit.as_dict()
    assert serialized['rank'] == 2 and not serialized['rank_truncated']
    json.dumps(serialized, allow_nan=False)


def test_dense_qr_reference_with_nonorthogonal_columns():
    rng = np.random.default_rng(103)
    phi = rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3))
    raw = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    h = (raw + raw.conj().T) / 2
    q = np.exp(rng.normal(size=8))
    fit = fit_carrier_subspace(phi, h @ phi, q, observation_weights=q)
    orthogonal, _ = np.linalg.qr(phi)
    energies, vectors = np.linalg.eigh(orthogonal.conj().T @ h @ orthogonal)
    expected = orthogonal @ vectors[:, 0]
    np.testing.assert_allclose(fit.statistics['energy_total'], energies[0], atol=3e-14)
    np.testing.assert_allclose(_fidelity(phi @ fit.coefficients, expected), 1., atol=3e-14)
    print('dense-QR Ritz energy error:', abs(fit.statistics['energy_total'] - energies[0]))


def _finite_sample():
    rng = np.random.default_rng(85)
    raw = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
    h = (raw + raw.conj().T) / 2
    phi = rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))
    q = np.exp(rng.normal(size=7))
    rows = np.asarray([0, 1, 1, 3, 4, 4, 5, 6, 6])
    return phi[rows], (h @ phi)[rows], q[rows]


def test_finite_q_hermitized_objective_preserves_raw_complex_energy():
    phi, hp, q = _finite_sample()
    fit = fit_carrier_subspace(phi, hp, q, n_sites=3)
    c = fit.coefficients
    state, action = phi @ c, hp @ c
    norm = np.sum(abs(state) ** 2 / q)
    energy = np.sum(state.conj() * action / q) / norm
    variance = np.sum(abs(action - energy * state) ** 2 / q) / norm
    assert abs(energy.imag) > 1e-3
    np.testing.assert_allclose(fit.statistics['energy_total'], energy.real, atol=1e-14)
    np.testing.assert_allclose(fit.statistics['imaginary_energy_total'], energy.imag, atol=1e-14)
    np.testing.assert_allclose(fit.statistics['variance_total'], variance, atol=2e-14)
    np.testing.assert_allclose(fit.ritz.energies[0], energy.real, atol=2e-14)
    assert fit.moments.hamiltonian_hermiticity.relative_frobenius_error > .1
    assert fit.statistics['real_rayleigh_hermitized_discrepancy'] < 1e-14
    # Raw anti-Hermiticity is retained, not replaced by zero after fitting.
    b = fit.moments.hamiltonian_raw
    np.testing.assert_allclose(fit.moments.hamiltonian, (b + b.conj().T) / 2, atol=1e-14)


def test_arbitrary_column_scales_phases_and_proposal_scale_preserve_ray():
    phi, hp, q = _finite_sample()
    expected = fit_carrier_subspace(phi, hp, q)
    scale = np.asarray([1e-9 * (1 + 2j), 1e8 * (-2 + 1j), .3 - .7j])
    changed = fit_carrier_subspace(phi * scale, hp * scale, q * 9e5)
    np.testing.assert_allclose(expected.ritz.energies, changed.ritz.energies, atol=5e-14)
    np.testing.assert_allclose(expected.ritz.overlap_condition_number,
                               changed.ritz.overlap_condition_number, atol=2e-13)
    np.testing.assert_allclose(_fidelity(phi @ expected.coefficients,
        (phi * scale) @ changed.coefficients, 1 / q), 1., atol=2e-14)
    for key in ('energy_total', 'imaginary_energy_total', 'variance_total'):
        np.testing.assert_allclose(expected.statistics[key], changed.statistics[key], atol=5e-14)


def test_nested_subspaces_cannot_raise_training_minimum():
    phi, hp, q = _finite_sample()
    fits = [fit_carrier_subspace(phi, hp, q, indices=tuple(range(k))) for k in (1, 2, 3)]
    energies = [f.statistics['energy_total'] for f in fits]
    assert energies[2] <= energies[1] + 1e-13 <= energies[0] + 2e-13
    assert [f.ritz.rank for f in fits] == [1, 2, 3]
    for k, fit in enumerate(fits, 1):
        assert np.all(fit.coefficients[k:] == 0)


def test_nodes_contribute_to_full_h_variance():
    phi = np.eye(3, dtype=complex)[:, :2]
    h = np.asarray([[-2., 0., .4], [0., 1., 0.], [.4, 0., 0.]])
    fit = fit_carrier_subspace(phi, h @ phi, np.ones(3), n_sites=3)
    state, action = phi @ fit.coefficients, h @ phi @ fit.coefficients
    assert state[2] == 0 and abs(action[2]) > .1
    np.testing.assert_allclose(fit.statistics['energy_total'], -2., atol=1e-14)
    np.testing.assert_allclose(fit.statistics['variance_total'], .16, atol=1e-14)


def test_rank_deficiency_and_zero_columns_fail_without_truncation():
    phi = np.asarray([[1., 0., 1.], [0., 1., 1.], [1., 1., 2.], [2., -1., 1.]])
    with pytest.raises(NumericalSubspaceRankError):
        fit_carrier_subspace(phi, phi, np.ones(4))
    with pytest.raises(NumericalSubspaceRankError):
        fit_carrier_subspace(np.eye(3), np.eye(3), np.ones(3),
                            observation_weights=np.asarray([1., 1., 0.]))
    phi[:, 2] = 0
    with pytest.raises(NumericalSubspaceRankError):
        fit_carrier_subspace(phi, phi, np.ones(4))


def test_condition_gate_not_silent_dimension_reduction():
    c = np.asarray([1., 0., 0.])
    phi = np.column_stack((c, c + np.asarray([0., .01, 0.])))
    with pytest.raises(NumericalSubspaceRankError, match='condition'):
        fit_carrier_subspace(phi, phi, np.ones(3), maximum_condition_number=1e3)
    fit = fit_carrier_subspace(phi, phi, np.ones(3), maximum_condition_number=1e6)
    assert fit.ritz.rank == 2


@pytest.mark.parametrize('change', (
    {'q': np.zeros(9)}, {'q': np.ones(8)}, {'indices': (0, 0)},
    {'indices': (0, 3)}, {'indices': (0, True)}, {'indices': ()},
    {'n_sites': True}, {'n_sites': 0}, {'relative_rank_tolerance': 0.},
    {'q': np.ones(9, dtype=complex)}, {'maximum_condition_number': None},
    {'maximum_condition_number': .5},
    {'observation_weights': np.zeros(9)}, {'observation_weights': np.ones(8)},
))
def test_invalid_inputs_fail(change):
    phi, hp, q = _finite_sample()
    args = dict(amplitudes=phi, h_amplitudes=hp, q=q)
    with pytest.raises(ValueError):
        fit_carrier_subspace(**{**args, **change})
