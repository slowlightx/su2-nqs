"""Physical parent-anchored fitting, independently checked against dense H."""
import numpy as np
import pytest

from su2_nqs.vmc.projected_spin_linear import (
    evaluate_linear_state, fit_parent_anchored_linear_update,
)


def example(seed=241023):
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(9, 5)) + 1j * rng.normal(size=(9, 5))
    phi[0] = 0  # H need not annihilate this exact amplitude node.
    raw = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    h = (raw + raw.conj().T) / 2
    q = np.arange(1., 10.) ** 2
    q /= q.sum()
    c = rng.normal(size=5)
    return phi, h @ phi, q, c, h


def dense_moments(phi, h, c):
    psi = phi @ c
    psi = psi / np.linalg.norm(psi)
    energy = np.vdot(psi, h @ psi)
    variance = np.linalg.norm(h @ psi - energy * psi) ** 2
    return energy, variance


def test_exhaustive_nonuniform_Q_has_full_complex_energy_and_node_complete_H2():
    phi, hp, q, c, h = example()
    before = evaluate_linear_state(phi, hp, q, c, observation_weights=q)
    energy, variance = dense_moments(phi, h, c)
    assert before['energy'] == pytest.approx(energy.real, abs=2e-14)
    assert before['energy_imaginary'] == pytest.approx(energy.imag, abs=2e-14)
    assert before['variance'] == pytest.approx(variance, abs=2e-14)
    assert before['exact_amplitude_node_rows'] == 1
    assert before['node_h2_contribution'] > .01
    state = phi @ c
    assert before['node_h2_contribution'] == pytest.approx(abs((h @ state)[0])**2 / np.vdot(state,state).real)
    # Omitting the amplitude node is demonstrably NOT the same variance.
    incomplete = evaluate_linear_state(phi[1:], hp[1:], q[1:], c, observation_weights=q[1:])
    assert before['variance'] - incomplete['variance'] == pytest.approx(before['node_h2_contribution'])
    # A projected Hamiltonian squared would also discard unrepresented Hpsi.
    gram, reduced_h = phi.conj().T @ phi, phi.conj().T @ hp
    wrong_h2 = np.vdot(c, reduced_h @ np.linalg.solve(gram, reduced_h @ c)).real / np.vdot(c, gram @ c).real
    assert before['h2'] - wrong_h2 > .1


def test_one_fit_step_reduces_exact_energy_and_respects_finite_FS_and_parent_norm():
    phi, hp, q, c, h = example()
    old = c.copy()
    updated, fit = fit_parent_anchored_linear_update(phi, hp, q, c,
        observation_weights=q, maximum_fs_sine=.08, initial_step=.1)
    assert fit['accepted']
    np.testing.assert_array_equal(c, old)
    energy, variance = dense_moments(phi, h, updated)
    assert energy.real < dense_moments(phi, h, c)[0].real
    assert fit['after']['energy'] == pytest.approx(energy.real, abs=2e-14)
    assert fit['after']['variance'] == pytest.approx(variance, abs=2e-14)
    assert fit['after']['fs_sine'] <= .08
    assert fit['after']['norm_ratio'] == pytest.approx(1., abs=2e-14)
    assert fit['after']['normalized_overlap_real'] > 0
    assert fit['metric_retained_rank'] == len(c) - 1
    assert fit['scale_gauge_metric_residual'] < 2e-14
    assert not fit['audit_used_for_selection'] and not fit['full_network_stationarity_certified']


def test_finite_Q_both_energy_sides_and_complex_residual_gradient():
    phi, hp, q, c, _ = example(241024)
    # Deliberately incomplete quadrature: K is NOT samplewise Hermitian.
    phi, hp, q = phi[[0, 1, 2, 4]], hp[[0, 1, 2, 4]], q[[0, 1, 2, 4]]
    _, fit = fit_parent_anchored_linear_update(phi, hp, q, c,
        variance_weight=.04, maximum_fs_sine=.2)
    direction = np.asarray(fit['direction'])
    direction_scale = max(float(np.linalg.norm(direction)), 1.)
    direction = direction / direction_scale
    eps = 1e-5
    plus = evaluate_linear_state(phi, hp, q, c + eps * direction)
    minus = evaluate_linear_state(phi, hp, q, c - eps * direction)
    assert abs(fit['before']['energy_imaginary']) > .01
    assert fit['hamiltonian_relative_antihermiticity'] > .01
    assert (plus['energy'] - minus['energy']) / (2*eps) == pytest.approx(fit['predicted_energy_slope']/direction_scale, rel=2e-8, abs=2e-8)
    assert (plus['variance'] - minus['variance']) / (2*eps) == pytest.approx(fit['predicted_variance_slope']/direction_scale, rel=2e-8, abs=2e-8)
    # The ordinary forward-only derivative is not the finite-Q derivative.
    p, h, d = phi @ c, hp @ c, phi @ direction
    n = np.sum(abs(p)**2/q)
    E = np.sum(p.conj()*h/q)/n
    forward_only = 2*np.sum(d.conj()*(h-E*p)/q).real/n
    assert abs(forward_only-fit['predicted_energy_slope']/direction_scale) > .1


def test_real_coordinates_complex_columns_remove_global_phase_and_scale_nulls():
    # First two real coefficients span u and i*u, i.e. only one physical ray.
    phi = np.array([[1, 1j, 0], [0, 0, 1]], complex)
    h = np.array([[1, .3+.2j], [.3-.2j, -1]])
    q, c = np.array([.2, .8]), np.array([1., 0., 0.])
    _, fit = fit_parent_anchored_linear_update(phi, h @ phi, q, c, observation_weights=q)
    assert fit['metric_resolved_rank'] == 1
    assert fit['direction'][0] == pytest.approx(0., abs=1e-14)
    assert fit['direction'][1] == pytest.approx(0., abs=1e-14)
    assert fit['direction'][2] != 0


def test_identical_ray_with_roundoff_is_not_a_resolved_direction():
    phi = np.array([[.3], [.7j], [1.1]], complex)
    h = np.diag([1., 2., -1.])
    c = np.array([1.3])
    candidate, fit = fit_parent_anchored_linear_update(phi, h @ phi, np.ones(3), c)
    np.testing.assert_array_equal(candidate, c)
    assert not fit['accepted'] and fit['metric_resolved_rank'] == 0


def test_rank_cap_and_duplicate_nulls_do_not_change_parent_or_claim_stationarity():
    phi, hp, q, c, _ = example()
    # An exactly redundant real column and unobserved coordinate remain legal.
    phi = np.c_[phi, phi[:, 0], np.zeros(len(phi))]
    hp = np.c_[hp, hp[:, 0], np.zeros(len(phi))]
    c = np.r_[c, 2., 4.]
    updated, fit = fit_parent_anchored_linear_update(phi, hp, q, c, observation_weights=q,
        maximum_tangent_rank=2, maximum_fs_sine=.1)
    assert fit['metric_resolved_rank'] == 4
    assert fit['metric_retained_rank'] == 2
    assert fit['discarded_gradient_fraction'] > .01
    # Zero-coordinate is not used to fit unsupported off-cache features.
    assert fit['direction'][-1] == 0
    assert not fit['full_network_stationarity_certified']
    assert np.isfinite(updated).all()


def test_small_samples_explicitly_report_discarded_reverse_action_gradient():
    # Column 2 vanishes only on OBSERVED samples, while Hphi_2 is nonzero there.
    phi = np.array([[1., 0, 0], [0, 1, 0]])
    hp = np.array([[1., -.3, 2.], [-.3, -1., 3.]])
    _, fit = fit_parent_anchored_linear_update(phi, hp, np.ones(2), np.array([1., .2, 0]))
    assert fit['direction'][2] == 0
    assert fit['discarded_gradient_fraction'] > .5
    assert not fit['full_network_stationarity_certified']


def test_common_complex_amplitude_and_Q_scaling_leave_update_unchanged():
    phi, hp, q, c, _ = example()
    candidate, fit = fit_parent_anchored_linear_update(phi, hp, q, c, observation_weights=q)
    for scale in (1e-100, 1e100*(1+1j)):
        other, row = fit_parent_anchored_linear_update(phi*scale, hp*scale, q*1e-80, c, observation_weights=q)
        np.testing.assert_allclose(other, candidate, atol=2e-12)
        assert row['after']['energy'] == pytest.approx(fit['after']['energy'], abs=2e-13)
        assert row['after']['variance'] == pytest.approx(fit['after']['variance'], abs=2e-13)


def test_orthogonal_real_coordinate_change_has_covariant_physical_update():
    phi, hp, q, c, _ = example()
    u, _ = np.linalg.qr(np.random.default_rng(12).normal(size=(len(c), len(c))))
    candidate, fit = fit_parent_anchored_linear_update(phi, hp, q, c, observation_weights=q)
    changed, row = fit_parent_anchored_linear_update(phi@u, hp@u, q, u.T@c, observation_weights=q)
    np.testing.assert_allclose(phi @ candidate, phi @ u @ changed, atol=3e-13)
    assert row['after']['fs_sine'] == pytest.approx(fit['after']['fs_sine'], abs=3e-13)


def test_failed_predeclared_line_returns_identical_parent():
    phi, hp, q, c, _ = example()
    candidate, fit = fit_parent_anchored_linear_update(phi, hp, q, c, observation_weights=q,
        maximum_fs_sine=1e-10, maximum_backtracks=0)
    assert not fit['accepted'] and fit['selected_step'] == 0
    np.testing.assert_array_equal(candidate, c)
    assert len(fit['trials']) == 1


def test_optional_fit_variance_guard_rejects_energy_only_descent():
    phi = np.eye(2)
    hp = np.diag([0., 2.])
    q, c = np.ones(2), np.array([.2, 1.])
    _, allowed = fit_parent_anchored_linear_update(phi, hp, q, c,
        initial_step=.01, maximum_fs_sine=.1)
    assert allowed['accepted']
    assert allowed['after']['energy'] < allowed['before']['energy']
    assert allowed['after']['variance'] > allowed['before']['variance']
    candidate, guarded = fit_parent_anchored_linear_update(phi, hp, q, c,
        initial_step=.01, maximum_fs_sine=.1, maximum_relative_variance_increase=0.)
    assert not guarded['accepted']
    np.testing.assert_array_equal(candidate, c)
    assert not any(row['variance_guard'] for row in guarded['trials'])


def test_exact_stationary_eigenstate_has_no_update():
    phi, hp, q, c = np.eye(2), np.diag([0., 2.]), np.array([.3, .7]), np.array([1., 0.])
    candidate, fit = fit_parent_anchored_linear_update(phi, hp, q, c, observation_weights=q)
    np.testing.assert_array_equal(candidate, c)
    assert not fit['accepted']
    assert fit['tangent_fs_speed'] == 0
    assert fit['before']['variance'] == 0


@pytest.mark.parametrize('kwargs', [dict(damping=0), dict(metric_rcond=np.nan),
    dict(maximum_tangent_rank=0), dict(maximum_tangent_rank=True), dict(maximum_backtracks=-1),
    dict(maximum_fs_sine=1), dict(initial_step=-1), dict(variance_weight=-1),
    dict(maximum_relative_variance_increase=-1)])
def test_invalid_fit_controls_rejected(kwargs):
    phi, hp, q, c, _ = example()
    with pytest.raises(ValueError):
        fit_parent_anchored_linear_update(phi, hp, q, c, **kwargs)


@pytest.mark.parametrize('case', ['qzero','qcomplex','coeffcomplex','zerostate','weightscomplex','weightsnegative'])
def test_invalid_observation_semantics_rejected(case):
    phi, hp, q, c, _ = example()
    weights = None
    if case == 'qzero': q[0] = 0
    if case == 'qcomplex': q = q.astype(complex)
    if case == 'coeffcomplex': c = c.astype(complex)
    if case == 'zerostate': c = c * 0
    if case == 'weightscomplex': weights = np.ones(len(q),complex)
    if case == 'weightsnegative': weights = -np.ones(len(q))
    with pytest.raises(ValueError):
        evaluate_linear_state(phi, hp, q, c, observation_weights=weights)
