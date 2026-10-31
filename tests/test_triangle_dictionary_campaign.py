"""Independent physical-step and stream checks, without N36 optimization."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import triangle_dictionary_campaign as campaign
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint

jax.config.update('jax_enable_x64', True)


def test_actual_runtime_passes_complete_J1_J2_graph_to_H_factory(monkeypatch):
    seen = {}

    def action_factory(model, images, bonds, **options):
        seen.update(model=model, images=images, bonds=bonds)
        return lambda bank, bits: (jnp.ones(len(bits)), jnp.zeros(len(bits)))

    monkeypatch.setattr(campaign, 'make_cached_projected_heisenberg_action', action_factory)
    monkeypatch.setattr(campaign, 'make_feature_lut_projected_amplitude',
        lambda *args, **kwargs: lambda bank, bits: jnp.asarray(1.))
    spec = dict(model=dict(length=6), j2_over_j1=.125, projection_chunk=4, action_chunk=2)
    images, bonds, _, _ = campaign.physical_runtime(spec, MixedClebschConfig())
    assert bonds == seen['bonds']
    assert len(images) == 432 and len(bonds) == 216
    assert sum(J == 1. for _, _, J in bonds) == 108
    assert sum(J == .125 for _, _, J in bonds) == 108
    # Independently check geometric shell lengths in oblique120 coordinates.
    for a, b, coupling in bonds:
        ra, ca = divmod(a, 6)
        rb, cb = divmod(b, 6)
        distance = min((ra-rb+6*k)**2+(ca-cb+6*l)**2-
                       (ra-rb+6*k)*(ca-cb+6*l)
                       for k in (-1, 0, 1) for l in (-1, 0, 1))
        assert distance == (1 if coupling == 1 else 3)
    assert len(campaign.campaign_bonds(dict(spec, j2_over_j1=0.))) == 108
    large = dict(spec, model=dict(length=12))
    assert len(campaign.campaign_bonds(large)) == 864
    assert len(campaign.campaign_bonds(dict(large, j2_over_j1=0.))) == 432
    with pytest.raises(ValueError):
        campaign.campaign_bonds(dict(spec, model=dict(length=9)))


def _calibrator(*, target=.05, max_factor=2., minimum_ess=.8):
    return campaign.make_fs_calibrator(lambda params: params,
        lambda params, row: (params[0]*row[0]).astype(jnp.complex128),
        jnp.zeros(1), target_fs_sine=target, relative_tolerance=1e-5,
        max_factor=max_factor, bracket_steps=8, bisection_steps=24,
        minimum_ess_fraction=minimum_ess)


def test_calibration_matches_independent_analytic_two_level_overlap():
    # Parent is the equal-weight state; the two rows exhaust its exact Born
    # distribution. This analytic Hilbert-space overlap is independent of AD
    # and of the production sampled_overlap_diagnostics implementation.
    parent, candidate = jnp.asarray([0.]), jnp.asarray([.08])
    rows = jnp.asarray([[-1.], [1.]])
    changed, info = _calibrator()(parent, candidate, rows)
    amount = float(changed[0])
    analytic_fs = np.sqrt(1-np.cosh(amount)**2/np.cosh(2*amount))
    assert bool(info['accepted'])
    assert analytic_fs == pytest.approx(.05, rel=1e-5)
    assert float(info['sampled_fs_sine']) == pytest.approx(analytic_fs, abs=2e-14)
    assert 0 < float(info['applied_factor']) <= 2
    assert not np.array_equal(changed, candidate)


@pytest.mark.parametrize('reason', ['zero_direction', 'bounded_too_short', 'poor_ESS', 'nonfinite'])
def test_calibration_rejects_without_choosing_a_shorter_endpoint(reason):
    parent, candidate = jnp.asarray([0.]), jnp.asarray([.08])
    rows = jnp.asarray([[-1.], [1.]])
    calibrate = _calibrator()
    if reason == 'zero_direction':
        candidate = parent
    elif reason == 'bounded_too_short':
        calibrate = _calibrator(max_factor=.01)
    elif reason == 'poor_ESS':
        calibrate = _calibrator(target=.5, max_factor=20., minimum_ess=.99)
    else:
        candidate = jnp.asarray([jnp.nan])
    changed, info = calibrate(parent, candidate, rows)
    assert not bool(info['accepted'])
    np.testing.assert_array_equal(changed, parent)
    assert float(info['applied_factor']) == 0


def test_training_and_calibration_checkpoint_streams_resume_independently(tmp_path):
    seeds = dict(training_seed=91301, calibration_seed=91302)
    states, key, cal_states, cal_key = campaign.initialize_random_streams(
        seeds, n_sites=36, training_chains=4, calibration_chains=4)
    assert not np.array_equal(key, cal_key)
    assert not np.array_equal(states, cal_states)
    path = tmp_path/'checkpoint.pkl'
    save_checkpoint(path, jnp.asarray([.3]), optimizer_state=dict(completed_steps=1),
        metadata=dict(training_states=states, training_rng=key,
                      calibration_states=cal_states, calibration_rng=cal_key))
    saved = load_checkpoint(path)
    np.testing.assert_array_equal(jax.random.normal(key, (10,)),
        jax.random.normal(saved['metadata']['training_rng'], (10,)))
    np.testing.assert_array_equal(jax.random.normal(cal_key, (10,)),
        jax.random.normal(saved['metadata']['calibration_rng'], (10,)))
    # Advancing one stream cannot consume the other's future draws.
    _, next_training_key = jax.random.split(key)
    assert not np.array_equal(next_training_key, saved['metadata']['training_rng'])
    np.testing.assert_array_equal(saved['metadata']['calibration_rng'], cal_key)
    with pytest.raises(ValueError, match='must differ'):
        campaign.initialize_random_streams(dict(training_seed=1, calibration_seed=1),
            n_sites=36, training_chains=4, calibration_chains=4)


def test_controller_resume_and_frozen_process_replay_keep_both_streams(tmp_path, monkeypatch):
    """Cheap controller integration: real atomic checkpoints/audits, toy updates.

    The physical N36 kernel and SR are separately checked above/in the legacy
    independent tests. This exercises controller state flow without training a
    large model inside a unit test.
    """
    monkeypatch.setattr(campaign, 'ROOT', tmp_path)
    monkeypatch.setattr(campaign, 'transitive_local_source_manifest', lambda *a, **k: {})
    monkeypatch.setattr(campaign, 'prepare_feature_lut_bank', lambda bank, cfg: bank)
    model = dict(length=6, multiplicity_cap=9, maximum_q=3)
    parent = {'x': jnp.asarray([.1])}
    source = tmp_path/'parent'/'checkpoint.pkl'
    save_checkpoint(source, parent)
    registration = dict(config=dict(model=model, j2_over_j1=.125, arm='ternary_joint'))
    registration['sha256'] = campaign.canonical_sha256(registration)
    campaign.atomic_json(source.parent/'registration.json', registration)
    campaign.atomic_json(source.parent/'terminal_audit.json', dict(
        params_sha256=campaign.pytree_sha256(parent), registration_sha256=registration['sha256']))

    def runtime(spec, cfg):
        amplitude = lambda params, bits: jnp.exp(params['x'][0])

        @jax.jit
        def evaluate(params, rows):
            psi = jnp.full(len(rows), jnp.exp(params['x'][0]))
            return psi, -18*psi
        return np.arange(36)[None], campaign.campaign_bonds(spec), amplitude, evaluate

    def arm_functions(spec, cfg, parent, arm, seeds, amplitude, evaluate, bonds):
        def transition(params, states, key, cal_states, cal_key, step):
            key, draw = jax.random.split(key)
            cal_key, cal_draw = jax.random.split(cal_key)
            changed = {'x': params['x']+.001*jax.random.normal(draw, (1,))+
                                      .002*jax.random.normal(cal_draw, (1,))}
            rows = jnp.tile(states[None], (2, 1, 1))
            energy = jnp.full(rows.shape[0]*rows.shape[1], -18.)
            info = dict(full_H_centers=energy.size, full_H_action_batches=1,
                calibration_H_calls=0, accepted=True)
            return changed, states, key, cal_states, cal_key, rows, energy, info
        return cfg, parent, {}, 0., lambda params: params, transition

    monkeypatch.setattr(campaign, 'physical_runtime', runtime)
    monkeypatch.setattr(campaign, '_arm_functions', arm_functions)
    spec = dict(model=model, j2_over_j1=.125, output='out',
        source=dict(checkpoint='parent/checkpoint.pkl',
                    checkpoint_sha256=campaign.file_sha256(source),
                    params_sha256=campaign.pytree_sha256(parent)),
        seeds=[dict(generator_seed=123, training_seed=301, calibration_seed=401, audit_seed=501)],
        training=dict(steps=2, chains=2, records_per_chain=2),
        calibration=dict(chains=2), sr=dict(damping=.1),
        audit=dict(chains=2, records_per_chain=2, burn_proposals=1, proposals_between=1),
        sampler=dict(bond_proposal_probability=.75, lazy_probability=.05))
    config = tmp_path/'config.json'
    campaign.atomic_json(config, spec)
    partial = campaign.run(config, 0, until_step=1)
    assert partial['status'] == 'stopped_at_safe_boundary'
    assert all(row['completed_steps'] == 1 for row in partial['arms'].values())
    assert 'parent_audit' not in partial
    finished = campaign.run(config, 0, resume=True)
    assert finished['status'] == 'completed'
    assert all(row['same_process_full_next_transition_bitwise_exact']
               for row in finished['arms'].values())
    assert finished['full_H_work']['training_centers'] == 24
    assert finished['full_H_work']['untouched_audit_centers'] == 16
    assert finished['full_H_work']['uncommitted_same_process_replay_centers'] == 24
    replay = campaign.run(config, 0, replay=True)
    assert replay['status'] == 'passed' and replay['updates_committed'] == 0
    assert all(row['full_next_transition_bitwise_exact'] for row in replay['arms'].values())
    with pytest.raises(FileExistsError):
        campaign.run(config, 0, replay=True)
