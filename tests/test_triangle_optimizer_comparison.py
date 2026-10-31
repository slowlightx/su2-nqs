"""Bounded native-optimizer controller checks; no N36 H contraction here."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import signal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import triangle_optimizer_comparison as comparison
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint

jax.config.update('jax_enable_x64', True)


@pytest.fixture
def plan(tmp_path):
    return dict(model=dict(length=6, multiplicity_cap=16, maximum_q=10),
        j2_over_j1=.125, binary_rank=8,
        parents={arity: {} for arity in comparison.ARITIES},
        optimizers=dict(sr_adam=dict(learning_rate=.001), direct_sr=dict(learning_rate=.001)),
        training=dict(samples=2048, chains=64, steps=8, refresh_proposals=1, spacing_proposals=1),
        calibration=dict(samples=128, chains=16, target_fs_sine=.001,
            relative_tolerance=.02, max_factor=2., bracket_steps=8,
            bisection_steps=16, minimum_ess_fraction=.8,
            initial_burn_proposals=1440, refresh_proposals=1, spacing_proposals=1),
        audit=dict(samples=512, chains=32, refresh_proposals=1, spacing_proposals=1),
        seeds={arity: dict(calibration=100+i*10, audit_sr_adam=101+i*10,
                          audit_direct_sr=102+i*10) for i, arity in enumerate(comparison.ARITIES)},
        pilot_steps=2, maximum_training_fs_sine=.02, minimum_training_ess_fraction=.8,
        damping=.05, projection_chunk=4, action_chunk=1, score_chunk=1,
        require_gpu=False, tolerance=2e-11,
        budget=dict(pilot_seconds=1800, replay_seconds=600, train_seconds=3600,
                    maximum_requested_A100_hours=23., maximum_peak_device_bytes=70_000_000_000),
        output=str(tmp_path/'output'))


def test_registered_plan_and_four_case_directories_are_distinct(plan):
    comparison.validate_plan(plan)
    paths = {comparison.case_directory(plan, arity, optimizer)
             for arity in comparison.ARITIES for optimizer in comparison.OPTIMIZERS}
    assert len(paths) == 4
    assert all(str(path).startswith(plan['output']) for path in paths)
    assert not Path(plan['output']).exists()
    for arity, optimizer in [('ternary', 'direct_sr'), ('ternary_joint', 'adam')]:
        with pytest.raises(ValueError, match='undeclared'):
            comparison.case_directory(plan, arity, optimizer)


@pytest.mark.parametrize('field,value', [
    (('model', 'length'), 12), (('model', 'multiplicity_cap'), 20),
    (('model', 'maximum_q'), 12), (('j2_over_j1',), 0.),
    (('binary_rank',), 4), (('training', 'samples'), 1024),
    (('training', 'chains'), 32), (('training', 'steps'), 2),
    (('pilot_steps',), 1), (('calibration', 'samples'), 129),
    (('audit', 'chains'), 1), (('maximum_training_fs_sine',), .0001),
    (('calibration', 'target_fs_sine'), 0.),
    (('training', 'samples'), 2048.), (('training', 'steps'), True),
    (('calibration', 'initial_burn_proposals'), 0),
    (('calibration', 'bracket_steps'), False), (('audit', 'spacing_proposals'), .5),
    (('damping',), np.nan), (('tolerance',), 3e-11),
    (('minimum_training_ess_fraction',), 1.01),
    (('optimizers', 'direct_sr', 'learning_rate'), np.inf),
    (('budget', 'train_seconds'), 3600.),
    (('budget', 'maximum_requested_A100_hours'), 1.),
    (('model', 'length'), 6.), (('model', 'multiplicity_cap'), 16.),
    (('training', 'steps'), 21),
    (('budget', 'maximum_requested_A100_hours'), np.nan),
    (('budget', 'maximum_requested_A100_hours'), 0.),
    (('budget', 'maximum_requested_A100_hours'), 25.),
    (('budget', 'maximum_peak_device_bytes'), np.nan),
    (('budget', 'maximum_peak_device_bytes'), 0),
    (('budget', 'maximum_peak_device_bytes'), True),
])
def test_plan_rejects_architecture_H_budget_or_calibration_changes(plan, field, value):
    invalid = deepcopy(plan)
    target = invalid
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    with pytest.raises(ValueError):
        comparison.validate_plan(invalid)


def test_plan_requires_both_arities_and_exactly_two_optimizer_recipes(plan):
    for key, removed in [('parents', 'binary_truncated'), ('optimizers', 'direct_sr')]:
        invalid = deepcopy(plan)
        del invalid[key][removed]
        with pytest.raises(ValueError):
            comparison.validate_plan(invalid)


@pytest.mark.parametrize('kind', ['duplicate', 'negative', 'outside_uint32', 'boolean'])
def test_plan_requires_distinct_valid_final_audit_and_calibration_seeds(plan, kind):
    row = plan['seeds']['ternary_joint']
    row['audit_direct_sr'] = {'duplicate': row['calibration'], 'negative': -1,
                             'outside_uint32': 2**32, 'boolean': True}[kind]
    with pytest.raises(ValueError, match='seeds'):
        comparison.validate_plan(plan)


def test_seed_names_cannot_be_replaced_by_undeclared_random_stream(plan):
    row = plan['seeds']['ternary_joint']
    row['unused_stream'] = row.pop('audit_direct_sr')
    with pytest.raises(ValueError, match='named calibration'):
        comparison.validate_plan(plan)


def _state():
    params = {'weights': jnp.asarray([.1, -.2], dtype=jnp.float64)}
    moments = (jnp.asarray(200), {'weights': jnp.asarray([.2, -.1])},
               {'weights': jnp.asarray([.03, .04])})
    training_key = jax.random.PRNGKey(1001)
    calibration_key = jax.random.PRNGKey(2001)
    states = jnp.tile(jnp.asarray([1]*18+[0]*18, jnp.int32), (64, 1))
    cal_states = jnp.roll(states[:16], 1, axis=1)
    return params, moments, training_key, states, calibration_key, cal_states


@pytest.mark.parametrize('fault', [None, 'not200', 'wrong_chain_shape'])
def test_parent_loader_reuses_sealed_same_H_fullstate_protocol(plan, monkeypatch, fault):
    state = _state()
    saved = dict(params=state[0], optimizer_state=state[1],
        metadata=dict(completed_steps=200, rng=state[2], sampler_states=state[3]))
    if fault == 'not200':
        saved['metadata']['completed_steps'] = 152
    elif fault == 'wrong_chain_shape':
        saved['metadata']['sampler_states'] = state[3][:32]
    seen = {}

    def checked(model, spec):
        seen.update(model=model, spec=spec)
        return None, None, dict(saved=saved)

    monkeypatch.setattr(comparison, 'checked_arity_endpoint', checked)
    if fault is not None:
        with pytest.raises(ValueError):
            comparison.load_parent(plan, 'binary_truncated')
    else:
        assert comparison.load_parent(plan, 'binary_truncated') is saved
        assert seen['spec']['endpoint_protocol'] == 'same_H_full_state_continuation'
        assert seen['spec']['arm'] == 'binary_truncated'
        assert seen['spec']['j2_over_j1'] == .125
        assert seen['spec']['additional_steps'] == plan['training']['steps']
        assert seen['spec']['steps'] == 200+plan['training']['steps']


def test_checkpoint_preserves_optimizer_and_both_rng_sampler_streams(tmp_path):
    state = _state()
    path = tmp_path/'checkpoint.pkl'
    metadata = dict(registration_sha256='frozen-registration', completed_steps=2, history=[])
    comparison.save_state(path, state, metadata)
    restored, saved_metadata = comparison.restore_state(path, 'frozen-registration')
    assert comparison.pytree_sha256(restored) == comparison.pytree_sha256(state)
    assert saved_metadata['completed_steps'] == 2
    for key_index in (2, 4):
        np.testing.assert_array_equal(jax.random.normal(restored[key_index], (12,)),
                                      jax.random.normal(state[key_index], (12,)))
    assert not np.array_equal(restored[2], restored[4])
    assert not np.array_equal(restored[3][:16], restored[5])
    assert 'params_sha256' not in metadata  # save does not mutate the caller.


@pytest.mark.parametrize('part', ['params', 'optimizer', 'training_key', 'training_states',
                                 'calibration_key', 'calibration_states', 'registration'])
def test_checkpoint_rejects_one_component_tamper(tmp_path, part):
    path = tmp_path/'checkpoint.pkl'
    comparison.save_state(path, _state(), dict(registration_sha256='frozen-registration'))
    payload = load_checkpoint(path)
    if part == 'params':
        payload['params']['weights'] = payload['params']['weights'].at[0].add(.01)
    elif part == 'optimizer':
        moments = payload['optimizer_state']
        payload['optimizer_state'] = (moments[0]+1, *moments[1:])
    elif part == 'registration':
        payload['metadata']['registration_sha256'] = 'other'
    else:
        value = np.array(payload['metadata'][part], copy=True)
        value.flat[0] += 1
        payload['metadata'][part] = value
    save_checkpoint(path, payload['params'], optimizer_state=payload['optimizer_state'],
                    metadata=payload['metadata'])
    with pytest.raises(ValueError, match='seal differs'):
        comparison.restore_state(path, 'frozen-registration')


def test_numeric_replay_fixture_omits_hash_strings_without_weakening_state_check():
    state = _state()
    info = dict(accepted=True, training_rows_sha256='traininghash',
        calibration_rows_sha256='calibrationhash', full_H_calls=2048,
        calibration=dict(accepted=True, sampled_fs_sine=.001),
        sr=dict(relative_SR_residual=3e-14))
    fixture = comparison.transition_fixture(state, info)
    assert fixture[0] is state
    assert 'training_rows_sha256' not in fixture[1]
    assert 'calibration_rows_sha256' not in fixture[1]
    assert 'training_rows_sha256' in info
    assert all(np.asarray(leaf).dtype.kind in 'biufc' for leaf in jax.tree.leaves(fixture))
    report = comparison.compare_replay(fixture, fixture, tolerance=2e-11,
                                       diagnostic_prefixes=('[1]',))
    assert report['passed'] and report['bitwise_exact']
    changed = (*state[:2], jax.random.PRNGKey(99), *state[3:])
    bad = comparison.compare_replay(fixture, comparison.transition_fixture(changed, info),
                                    tolerance=2e-11, diagnostic_prefixes=('[1]',))
    assert not bad['passed']


def _install_tiny_runtime(monkeypatch):
    initial = _state()
    parent = dict(params=initial[0], optimizer_state=initial[1],
                  metadata=dict(rng=initial[2], sampler_states=initial[3]))
    controls = dict(pause_after_step=None)
    handlers = {}

    def transition(state):
        p, moments, key, states, cal_key, cal_states = state
        key, draw = jax.random.split(key)
        cal_key, cal_draw = jax.random.split(cal_key)
        p = {'weights': p['weights']+.001*jax.random.normal(draw, (2,))+
                                    .002*jax.random.normal(cal_draw, (2,))}
        moments = (moments[0]+1, {'weights': moments[1]['weights']+.01},
                   {'weights': moments[2]['weights']+.001})
        changed = (p, moments, key, jnp.roll(states, 1, axis=1),
                   cal_key, jnp.roll(cal_states, 2, axis=1))
        info = dict(accepted=True, training_energy_per_site=-.5,
            training_variance_per_site=.01, full_H_calls=2048,
            training_rows_sha256=comparison.pytree_sha256(changed[3]),
            calibration_rows_sha256=comparison.pytree_sha256(changed[5]),
            calibration=dict(accepted=True, sampled_fs_sine=.001))
        if controls['pause_after_step'] == int(moments[0])-200:
            handlers[signal.SIGUSR1](signal.SIGUSR1, None)
        return changed, info, dict(sampling_seconds=.001, H_seconds=.001)

    monkeypatch.setattr(comparison, 'runtime', lambda *args: dict(transition=transition,
        prepare=lambda p: p, projected=lambda p, row: jnp.asarray(1.), edges=()))
    monkeypatch.setattr(comparison, 'run_exchange_sweeps',
        lambda p, key, rows, **kwargs: (rows, jnp.ones(len(rows)), jax.random.split(key)[0], {}))
    monkeypatch.setattr(comparison, 'load_parent', lambda *args: parent)
    monkeypatch.setattr(comparison, 'preflight',
                        lambda *args: dict(status='passed', full_H_centers=4))
    monkeypatch.setattr(comparison, 'terminal_audit',
        lambda rt, p, spec, seed, output: dict(energy_per_site=-.5, variance_per_site=.01,
            samples=spec['samples'], seed=seed, params_sha256=comparison.pytree_sha256(p)))
    monkeypatch.setattr(comparison, 'transitive_local_source_manifest', lambda *args, **kwargs: {})
    monkeypatch.setattr(comparison.signal, 'signal', lambda sig, callback: handlers.update({sig: callback}))
    return controls


def test_tiny_controller_pilot_isolated_replay_and_safe_pause_resume(plan, monkeypatch):
    controls = _install_tiny_runtime(monkeypatch)
    seal = comparison.canonical_sha256(plan)
    arity, optimizer = 'ternary_joint', 'direct_sr'
    output = comparison.case_directory(plan, arity, optimizer)
    pilot = comparison.run(plan, arity, optimizer, 'pilot', seal)
    assert pilot['status'] == 'passed' and pilot['completed_steps'] == 2
    cp_hash = comparison.file_sha256(output/'checkpoint.pkl')
    replay = comparison.run(plan, arity, optimizer, 'replay', seal)
    assert replay['status'] == 'replay_passed'
    assert comparison.file_sha256(output/'checkpoint.pkl') == cp_hash
    saved_replay = json.loads((output/'process_replay.json').read_text())
    assert saved_replay['optimizer_updates_committed'] == 0
    assert saved_replay['new_full_H_centers'] == 2048
    controls['pause_after_step'] = 3
    paused = comparison.run(plan, arity, optimizer, 'train', seal)
    assert paused['status'] == 'safe_pause' and paused['completed_steps'] == 3
    controls['pause_after_step'] = None
    result = comparison.run(plan, arity, optimizer, 'train', seal)
    assert result['status'] == 'fixed_steps_complete'
    assert result['completed_steps'] == plan['training']['steps']
    assert len(result['history']) == plan['training']['steps']
    restored, metadata = comparison.restore_state(output/'checkpoint.pkl', result['registration_sha256'])
    assert int(restored[1][0]) == 200+plan['training']['steps']
    assert metadata['completed_steps'] == plan['training']['steps']
    assert result['full_H_training_centers'] == 2048*plan['training']['steps']
    assert result['replay_H_centers'] == 4096
    assert result['preflight_H_centers'] == 4
    assert result['calibration_H_centers'] == 0
    assert result['terminal_audit_H_centers'] == plan['audit']['samples']
    assert result['equal_counts_not_identical_Born_rows_after_divergence']


def test_tiny_controller_rejects_changed_plan_before_parent_or_runtime(plan, monkeypatch):
    monkeypatch.setattr(comparison, 'load_parent', lambda *args: pytest.fail('must not load parent'))
    with pytest.raises(ValueError, match='registered plan changed'):
        comparison.run(plan, 'ternary_joint', 'direct_sr', 'pilot', 'wrong-seal')


def test_failed_pilot_budget_cannot_be_relabelled_or_replayed(plan, monkeypatch):
    _install_tiny_runtime(monkeypatch)
    plan['budget']['train_seconds'] = 1
    seal = comparison.canonical_sha256(plan)
    arity, optimizer = 'ternary_joint', 'direct_sr'
    output = comparison.case_directory(plan, arity, optimizer)
    with pytest.raises(RuntimeError, match='pilot exceeds'):
        comparison.run(plan, arity, optimizer, 'pilot', seal)
    pilot = json.loads((output/'pilot.json').read_text())
    assert pilot['status'] == 'failed'
    assert pilot['completed_steps'] == 2
    before = comparison.file_sha256(output/'checkpoint.pkl')
    with pytest.raises(ValueError, match='successful measured pilot'):
        comparison.run(plan, arity, optimizer, 'replay', seal)
    assert comparison.file_sha256(output/'checkpoint.pkl') == before
    assert not (output/'process_replay.json').exists()


@pytest.mark.parametrize('screen', ['accepted', 'sr_rejected', 'cal_rejected', 'too_far', 'poor_ess'])
def test_transition_commits_only_held_FS_and_training_guarded_update(plan, monkeypatch, screen):
    """Exercise the actual transition/controller guards with tiny amplitudes.

    The production closure is used, but CG preparation, H and sampling are
    replaced by exact two-level callbacks. No N36 contraction or training is
    performed. In particular, rejection must roll back optimizer moments too.
    """
    monkeypatch.setattr(comparison, 'arity_clebsch_bank', lambda p, *args: p)
    monkeypatch.setattr(comparison, 'prepare_feature_lut_bank', lambda p, cfg: p)
    projected = lambda p, row: jnp.exp(p['weights'][0]*(2.*row[0]-1.))
    monkeypatch.setattr(comparison, 'make_feature_lut_projected_amplitude', lambda *a, **k: projected)
    h_calls = []

    def action_factory(*args, **kwargs):
        def action(p, rows):
            h_calls.append('complete_H')
            psi = jax.vmap(lambda row: projected(p, row))(rows)
            return psi, -18.*psi
        return action

    monkeypatch.setattr(comparison, 'make_cached_projected_heisenberg_action', action_factory)

    def optimizer_factory(*args, **kwargs):
        def update(p, moments, rows, energies, rate, *, warm):
            proposed = {'weights': p['weights']+jnp.asarray([.02, 0.])}
            return proposed, (moments[0]+1, *moments[1:]), dict(accepted=jnp.asarray(screen != 'sr_rejected'))
        return update

    monkeypatch.setattr(comparison, 'make_physical_sr_adam', optimizer_factory)

    def calibration_factory(*args, **kwargs):
        def calibrate(parent, normalized, rows):
            chosen = normalized
            if screen == 'too_far':
                chosen = {'weights': parent['weights']+jnp.asarray([1., 0.])}
            if screen == 'cal_rejected':
                chosen = parent
            return chosen, dict(accepted=jnp.asarray(screen != 'cal_rejected'))
        return calibrate

    monkeypatch.setattr(comparison, 'make_fs_calibrator', calibration_factory)
    rows = jnp.zeros((2, 36), jnp.int32).at[1, 0].set(1)

    def sampler_factory(amplitude, settings, edges):
        actual_rows = rows if settings is plan['training'] else rows[::-1]
        return lambda prepared, key, states: (jax.random.split(key)[0], states, actual_rows)

    monkeypatch.setattr(comparison, 'make_record_sampler', sampler_factory)
    if screen == 'poor_ess':
        plan['minimum_training_ess_fraction'] = 1.
    state = _state()
    rt = comparison.runtime(plan, 'ternary_joint', 'direct_sr', state[0])
    changed, info, _ = rt['transition'](state)
    assert info['full_H_calls'] == 2
    assert h_calls == ['complete_H']  # no candidate-energy or calibration H call
    assert info['training_rows_sha256'] != info['calibration_rows_sha256']
    assert info['accepted'] == (screen == 'accepted')
    if screen == 'accepted':
        assert not np.array_equal(changed[0]['weights'], state[0]['weights'])
        assert int(changed[1][0]) == int(state[1][0])+1
        assert info['measured_training_step']['sampled_fs_sine'] <= plan['maximum_training_fs_sine']
    else:
        assert comparison.pytree_sha256(changed[0]) == comparison.pytree_sha256(state[0])
        assert comparison.pytree_sha256(changed[1]) == comparison.pytree_sha256(state[1])
    # Sampling consumes both streams even if the physical update is rejected.
    assert not np.array_equal(changed[2], state[2])
    assert not np.array_equal(changed[4], state[4])
