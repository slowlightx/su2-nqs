"""Synthetic cached-state continuation, finite-Q gradients and atomic resume."""
from dataclasses import replace

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.cached_energy_continuation import (
    CachedEnergyConfig, run_cached_energy_continuation,
)
from su2_nqs.vmc.matched_state_step import matched_state_step
from su2_nqs.vmc.resumable_checkpoint import load_resumable_checkpoint, save_resumable_checkpoint


def _problem():
    rng = np.random.default_rng(2026101701)
    matrix = rng.normal(size=(5, 5))+1j*rng.normal(size=(5, 5))
    h = (matrix+matrix.conj().T)/4
    carrier = np.array([0, 1, .3j, -.6, .2], dtype=complex)
    features = (rng.normal(size=(5, 3))+1j*rng.normal(size=(5, 3)))/3
    q = np.array([.3, 1.1, 2.7, .5, 1.4])
    def pair(x):
        psi = jnp.asarray(carrier)+jnp.asarray(features)@jnp.sin(x)
        return psi, jnp.asarray(h)@psi
    def dense(x):
        psi = carrier+features@np.sin(x)
        return psi, h@psi
    return pair, dense, q


def _config(steps=5):
    return CachedEnergyConfig(steps=steps, n_sites=5, target_fs=.01, bisection_steps=8)


def test_checkpoint_resume_matches_uninterrupted_parameters_and_moments_exactly(tmp_path):
    pair, _, q = _problem(); initial = np.array([.17, -.09, .04]); cfg = _config()
    complete = run_cached_energy_continuation(initial, pair, q, cfg, tmp_path/'complete.ckpt',
        {'arm':'synthetic', 'cache_hash':'fixed-features-v1'})
    path = tmp_path/'interrupted.ckpt'
    def interrupt(record):
        # Callback occurs after checksum/parameter/moments/history commit.
        saved = load_resumable_checkpoint(path)
        assert saved['completed_steps'] == record['completed_steps']
        if record['completed_steps'] == 2: raise RuntimeError('synthetic process interruption')
    with pytest.raises(RuntimeError, match='synthetic process interruption'):
        run_cached_energy_continuation(initial, pair, q, cfg, path,
            {'arm':'synthetic', 'cache_hash':'fixed-features-v1'}, progress=interrupt)
    resumed = run_cached_energy_continuation(initial, pair, q, cfg, path,
        {'arm':'synthetic', 'cache_hash':'fixed-features-v1'}, resume=True)
    assert resumed.resumed_from_step == 2 and resumed.completed_steps == cfg.steps
    for name in ('parameters', 'first_moment', 'second_moment', 'amplitudes', 'h_amplitudes'):
        np.testing.assert_array_equal(getattr(resumed, name), getattr(complete, name))
    for a, b in zip(resumed.history, complete.history):
        for key in ('energy_per_site','variance_per_site','gradient_norm','physical_step'):
            assert a[key] == b[key]
    assert resumed.ledger['completed_gradient_calls'] == cfg.steps
    assert resumed.ledger['completed_step_candidate_calls'] == cfg.steps*(cfg.bisection_steps+1)
    assert resumed.ledger['completed_cached_state_pair_evaluations'] == cfg.steps*(cfg.bisection_steps+4)
    assert resumed.ledger['completed_invocation_validation_state_pair_evaluations'] == 2
    assert resumed.ledger['new_H_queries'] == resumed.ledger['new_samples'] == 0


def test_one_step_uses_dual_sided_full_H_gradient_and_total_H_units(tmp_path):
    pair, dense, q = _problem(); initial = np.zeros(3); cfg = _config(1)
    def energy(x):
        psi, hp = dense(x)
        return (np.sum(psi.conj()*hp/q)/np.sum(abs(psi)**2/q)).real/cfg.n_sites
    eps = 1e-6; basis = np.eye(3)
    gradient = np.asarray([(energy(initial+eps*d)-energy(initial-eps*d))/(2*eps) for d in basis])
    direction = -gradient/(abs(gradient)+cfg.adam_epsilon)
    direction /= np.linalg.norm(direction)
    parent = dense(initial)[0]
    expected = matched_state_step(lambda alpha:dense(initial+alpha*direction)[0], parent, q,
        cfg.target_fs, cfg.max_alpha, cfg.bisection_steps,
        relative_tolerance=cfg.fs_relative_tolerance, absolute_tolerance=cfg.fs_absolute_tolerance)
    result = run_cached_energy_continuation(initial, pair, q, cfg, tmp_path/'one.ckpt', {'fixed':'synthetic'})
    np.testing.assert_allclose(result.parameters, initial+expected.alpha*direction, atol=2e-12, rtol=2e-9)
    np.testing.assert_allclose(result.first_moment, (1-cfg.beta1)*gradient, atol=2e-11, rtol=2e-8)
    psi, hp = dense(result.parameters)
    norm = np.sum(abs(psi)**2/q); e = np.sum(psi.conj()*hp/q)/norm
    variance = np.sum(abs(hp-e*psi)**2/q)/norm
    np.testing.assert_allclose(result.final_moments['energy_per_site'], e.real/cfg.n_sites, atol=2e-15)
    np.testing.assert_allclose(result.final_moments['variance_per_site'], variance/cfg.n_sites, atol=2e-15)
    assert parent[0] == 0 and abs((dense(initial)[1])[0]) > .1


def test_fresh_arbitrary_state_starts_with_zero_Adam_moments(tmp_path):
    pair, dense, q = _problem(); initial = np.array([.4, -.2, .1]); path = tmp_path/'zero.ckpt'
    result = run_cached_energy_continuation(initial, pair, q, _config(0), path, {'arm':'continued-teacher'})
    stored = load_resumable_checkpoint(path)
    np.testing.assert_array_equal(stored['params'], initial)
    for moment in stored['optimizer_state'].values(): np.testing.assert_array_equal(moment, np.zeros_like(initial))
    np.testing.assert_allclose(result.amplitudes, dense(initial)[0], atol=1e-16)
    assert stored['completed_steps'] == 0 and stored['next_sample_index'] == 0


def test_resume_rejects_changed_identity_proposal_initial_state_and_step_budget(tmp_path):
    pair, _, q = _problem(); initial = np.zeros(3); cfg = _config(0); path = tmp_path/'sealed.ckpt'
    run_cached_energy_continuation(initial, pair, q, cfg, path, {'source':'unchanged'})
    with pytest.raises(FileExistsError): run_cached_energy_continuation(initial, pair, q, cfg, path, {'source':'unchanged'})
    for x, proposal, config, identity in (
        (initial, q, cfg, {'source':'changed'}), (initial, q*2, cfg, {'source':'unchanged'}),
        (initial+.01, q, cfg, {'source':'unchanged'}), (initial, q, replace(cfg,steps=1), {'source':'unchanged'})):
        with pytest.raises(ValueError, match='identity'):
            run_cached_energy_continuation(x, pair, proposal, config, path, identity, resume=True)


@pytest.mark.parametrize('corruption', ('shape','nan','negative_variance','sample_index'))
def test_valid_checksum_but_invalid_optimizer_payload_is_rejected(tmp_path, corruption):
    pair, _, q = _problem(); initial = np.zeros(3); cfg = _config(0); path = tmp_path/'bad.ckpt'
    run_cached_energy_continuation(initial, pair, q, cfg, path, {'source':'unchanged'})
    saved = load_resumable_checkpoint(path)
    if corruption == 'shape': saved['optimizer_state']['m'] = np.zeros(4)
    elif corruption == 'nan':
        saved['params'] = np.array(saved['params'], copy=True); saved['params'][1] = np.nan
    elif corruption == 'negative_variance':
        saved['optimizer_state']['v'] = np.array(saved['optimizer_state']['v'], copy=True)
        saved['optimizer_state']['v'][0] = -.1
    elif corruption == 'sample_index': saved['next_sample_index'] = 1
    save_resumable_checkpoint(path, saved['params'], optimizer_state=saved['optimizer_state'],
        completed_steps=saved['completed_steps'], next_sample_index=saved['next_sample_index'],
        run_identity=saved['run_identity'], history=saved['history'], metadata=saved['metadata'])
    with pytest.raises(ValueError):
        run_cached_energy_continuation(initial, pair, q, cfg, path, {'source':'unchanged'}, resume=True)


def test_null_direction_does_not_skip_fixed_step_work_or_choose_by_energy(tmp_path):
    carrier = jnp.asarray([1., .4, -.2]); hc = jnp.asarray([-.3, .2, .1])
    def pair(x): return (1+x[0])*carrier, (1+x[0])*hc
    cfg = _config(2)
    result = run_cached_energy_continuation(np.zeros(1), pair, np.ones(3), cfg, tmp_path/'gauge.ckpt', {'arm':'gauge'})
    np.testing.assert_array_equal(result.parameters, np.zeros(1))
    assert all(row['physical_step']['alpha'] == 0 for row in result.history)
    assert all(row['physical_step']['evaluation_count'] == cfg.bisection_steps+1 for row in result.history)
    assert result.completed_steps == 2


def test_invalid_config_parameters_and_state_shapes_fail_closed(tmp_path):
    for kwargs in ({'steps':-1},{'n_sites':0},{'target_fs':0},{'beta2':1},{'bisection_steps':True}):
        with pytest.raises(ValueError): CachedEnergyConfig(**kwargs)
    pair, _, q = _problem()
    with pytest.raises(ValueError):
        run_cached_energy_continuation(np.zeros(3,dtype=complex), pair, q, _config(0), tmp_path/'none.ckpt', {'arm':'x'})
    with pytest.raises(ValueError):
        run_cached_energy_continuation(np.zeros(3), pair, -q, _config(0), tmp_path/'none.ckpt', {'arm':'x'})
    with pytest.raises(ValueError):
        run_cached_energy_continuation(np.zeros(3), pair, q[:-1], _config(0), tmp_path/'none.ckpt', {'arm':'x'})
