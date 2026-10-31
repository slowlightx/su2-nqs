"""Exact tiny-state checks for Hamiltonian-independent physical step matching."""
from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.physical_step_calibration import make_fs_calibrator

jax.config.update('jax_enable_x64', True)


def _two_level_calibrator(**options):
    arguments = dict(target_fs_sine=.05, relative_tolerance=1e-6,
                     bracket_steps=8, bisection_steps=24)
    arguments.update(options)
    return make_fs_calibrator(lambda p: p, lambda p, row: p['x'][0]*row[0]+0j,
                              {'x': jnp.zeros(1)}, **arguments)


def _exact_fs(parent, changed):
    parent, changed = np.asarray(parent), np.asarray(changed)
    overlap = abs(np.vdot(parent, changed))**2
    norm = np.vdot(parent, parent).real*np.vdot(changed, changed).real
    return np.sqrt(max(0., 1.-overlap/norm))


def test_calibration_matches_actual_nonlinear_state_not_parameter_distance():
    parent, proposed = {'x': jnp.asarray([0.])}, {'x': jnp.asarray([.08])}
    # These two rows enumerate the exact parent Born distribution.
    changed, info = _two_level_calibrator()(parent, proposed, jnp.asarray([[-1.], [1.]]))
    parameter = float(changed['x'][0])
    fs = _exact_fs(np.ones(2), np.exp(np.asarray([-parameter, parameter])))
    assert bool(info['accepted'])
    assert fs == pytest.approx(.05, rel=1e-6)
    assert float(info['sampled_fs_sine']) == pytest.approx(fs, abs=2e-14)
    assert float(info['applied_factor']) == pytest.approx(parameter/.08, abs=1e-14)
    assert int(info['amplitude_batch_evaluations']) <= 2+8+24


def test_complex_logs_keep_physical_sign_flip_in_finite_distance():
    # A real amplitude changes sign on the accepted path. Taking log(abs(psi))
    # would yield the wrong distance; complex log retains the pi phase.
    def logarithm(parameter, row):
        amplitude = (1.-2*parameter[0]*row[0]).astype(jnp.complex128)
        return jnp.log(amplitude)

    calibrate = make_fs_calibrator(lambda p: p, logarithm, jnp.zeros(1),
        target_fs_sine=.8, relative_tolerance=1e-6, max_factor=1.,
        bracket_steps=4, bisection_steps=26, minimum_ess_fraction=.5)
    changed, info = calibrate(jnp.zeros(1), jnp.asarray([.9]), jnp.asarray([[0.], [1.]]))
    vector = np.asarray([1., 1.-2*float(changed[0])])
    assert bool(info['accepted'])
    assert vector[1] < 0
    assert _exact_fs(np.ones(2), vector) == pytest.approx(.8, rel=1e-6)
    assert _exact_fs(np.ones(2), abs(vector)) != pytest.approx(.8, abs=.1)


def test_phase_only_direction_matches_overlap_without_density_change():
    calibrate = make_fs_calibrator(lambda p: p,
        lambda p, row: 1j*p[0]*row[0], jnp.zeros(1),
        target_fs_sine=.2, relative_tolerance=1e-6,
        bracket_steps=6, bisection_steps=24)
    changed, info = calibrate(jnp.zeros(1), jnp.asarray([.3]), jnp.asarray([[-1.], [1.]]))
    vector = np.exp(1j*float(changed[0])*np.asarray([-1., 1.]))
    assert bool(info['accepted'])
    assert _exact_fs(np.ones(2), vector) == pytest.approx(.2, rel=1e-6)
    assert float(info['reweighting_ess_fraction']) == pytest.approx(1., abs=2e-14)


@pytest.mark.parametrize('gauge', [1., 1j, 1.+2j])
def test_uniform_normalization_and_global_phase_gauges_cannot_pass(gauge):
    calibrate = make_fs_calibrator(lambda p: p, lambda p, row: p[0]*gauge,
        jnp.zeros(1), target_fs_sine=.05, bracket_steps=5, bisection_steps=12)
    parent = jnp.zeros(1)
    changed, info = calibrate(parent, jnp.asarray([.4]), jnp.arange(4)[:, None])
    np.testing.assert_array_equal(changed, parent)
    assert not bool(info['accepted'])
    assert not bool(info['target_bracketed'])
    assert float(info['applied_factor']) == 0
    assert int(info['amplitude_batch_evaluations']) == 2+5


@pytest.mark.parametrize('reason', ['unbracketed', 'poor_ess', 'nan_direction', 'infinite_old_logs'])
def test_rejection_returns_exact_parent_without_partial_step(reason):
    parent, proposed = {'x': jnp.asarray([0.])}, {'x': jnp.asarray([.08])}
    calibrate = _two_level_calibrator()
    if reason == 'unbracketed':
        calibrate = _two_level_calibrator(max_factor=.01)
    elif reason == 'poor_ess':
        calibrate = _two_level_calibrator(target_fs_sine=.5, max_factor=20.,
                                        minimum_ess_fraction=.99)
    elif reason == 'nan_direction':
        proposed = {'x': jnp.asarray([jnp.nan])}
    else:
        calibrate = make_fs_calibrator(lambda p: p,
            lambda p, row: jnp.log(p['x'][0].astype(jnp.complex128)), parent,
            target_fs_sine=.05)
    changed, info = calibrate(parent, proposed, jnp.asarray([[-1.], [1.]]))
    np.testing.assert_array_equal(changed['x'], parent['x'])
    assert not bool(info['accepted'])
    assert float(info['applied_factor']) == 0
    assert float(info['sampled_fs_sine']) == 0
    assert int(info['amplitude_batch_evaluations']) <= 2+8+24


def test_nonmonotone_local_bracket_is_rejected_without_energy_selection():
    calibrate = make_fs_calibrator(lambda p: p,
        lambda p, row: 1j*jnp.sin(2*jnp.pi*p[0])*row[0], jnp.zeros(1),
        target_fs_sine=.6, max_factor=1., bracket_steps=3, bisection_steps=12)
    changed, info = calibrate(jnp.zeros(1), jnp.ones(1), jnp.asarray([[0.], [1.]]))
    assert not bool(info['locally_monotone_finite'])
    assert not bool(info['accepted'])
    np.testing.assert_array_equal(changed, jnp.zeros(1))
    assert int(info['amplitude_batch_evaluations']) == 2+3
    # Neither factory nor returned callable accepts an energy or H-action.
    assert set(inspect.signature(calibrate).parameters) == {'parent', 'proposed', 'rows'}
    assert not {'energy', 'hamiltonian', 'action'} & set(inspect.signature(make_fs_calibrator).parameters)


@pytest.mark.parametrize('arguments', [
    {'target_fs_sine': 0.}, {'target_fs_sine': 1.}, {'target_fs_sine': np.nan},
    {'relative_tolerance': 0.}, {'relative_tolerance': 1.},
    {'max_factor': 0.}, {'max_factor': np.inf},
    {'minimum_ess_fraction': 0.}, {'minimum_ess_fraction': 1.01},
    {'bracket_steps': 0}, {'bracket_steps': True}, {'bracket_steps': 2.5},
    {'bisection_steps': 0}, {'bisection_steps': False},
    {'batch_size': 0}, {'batch_size': True}, {'batch_size': 1.5},
])
def test_calibration_rejects_invalid_registered_search_scales(arguments):
    with pytest.raises(ValueError):
        _two_level_calibrator(**arguments)


def test_existing_campaign_reexports_the_same_generic_calibrator():
    from projects.clebsch_tree_nqs.workflows import triangle_dictionary_campaign
    assert triangle_dictionary_campaign.make_fs_calibrator is make_fs_calibrator


@pytest.mark.parametrize('batch_size', [1, 3])
def test_chunked_calibration_matches_default_values_and_gradients(batch_size):
    rows = jnp.asarray([[-1., .2], [-.7, -.3], [-.2, .5], [.1, -.8],
                        [.3, .9], [.7, -.1], [1., .4]])
    template = jnp.zeros(2)
    prepare = lambda p: jnp.tanh(p)
    logs = lambda p, x: jnp.dot(p, x)+.3j*p[1]*x[0]
    options = dict(target_fs_sine=.03, relative_tolerance=1e-6,
                   bracket_steps=8, bisection_steps=24)
    plain = make_fs_calibrator(prepare, logs, template, **options)
    chunked = make_fs_calibrator(prepare, logs, template, batch_size=batch_size, **options)
    proposed = jnp.asarray([.12, .03])
    a, ai = plain(template, proposed, rows)
    b, bi = chunked(template, proposed, rows)
    assert bool(ai['accepted']) and bool(bi['accepted'])
    np.testing.assert_allclose(a, b, rtol=0, atol=2e-13)
    for key in ai:
        np.testing.assert_allclose(ai[key], bi[key], rtol=0, atol=2e-12)
    ga = jax.jacrev(lambda p: plain(p, proposed, rows)[0])(template)
    gb = jax.jacrev(lambda p: chunked(p, proposed, rows)[0])(template)
    np.testing.assert_allclose(ga, gb, rtol=0, atol=2e-13)


def test_chunked_calibration_does_not_turn_pure_gauge_into_physical_progress():
    calibrate = make_fs_calibrator(lambda p: p, lambda p, x: (1+2j)*p[0],
        jnp.zeros(1), target_fs_sine=.001, batch_size=3)
    candidate, info = calibrate(jnp.zeros(1), jnp.ones(1), jnp.arange(7)[:, None])
    assert not bool(info['accepted'])
    np.testing.assert_array_equal(candidate, jnp.zeros(1))
