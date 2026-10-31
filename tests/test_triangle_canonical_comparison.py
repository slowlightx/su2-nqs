"""Frozen same-parent coordinate protocol; exact state math is tested separately."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import triangle_canonical_comparison as trial
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, init_mixed_clebsch_family
from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint

jax.config.update('jax_enable_x64', True)


@pytest.fixture
def plan():
    directory = Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/configs/remote/ct_canonical_20261021'
    return json.loads((directory/'j0125.json').read_text())


def test_registered_plan_and_disjoint_physical_step_streams(plan):
    trial.validate_plan(plan)
    # Sequential records of each chain must never straddle the split.
    rows = jnp.tile(jnp.arange(128)[:, None], (8, 36))
    fit, unseen = trial.split_calibration_chains(rows, 128)
    assert fit.shape == unseen.shape == (512, 36)
    assert set(np.asarray(fit[:, 0])) == set(range(64))
    assert set(np.asarray(unseen[:, 0])) == set(range(64, 128))


@pytest.mark.parametrize('field,value', [
    (('model', 'multiplicity_cap'), 18), (('j2_over_j1',), .05),
    (('arity',), 'binary_truncated'), (('sr_solver',), 'woodbury_gradient'),
    (('training', 'steps'), 32), (('training', 'samples'), 1024),
    (('calibration', 'chains'), 64), (('calibration', 'samples'), 512),
    (('calibration', 'validation_relative_tolerance'), .5),
    (('calibration', 'target_fs_sine'), .002), (('calibration', 'relative_tolerance'), .1),
    (('audit', 'samples'), 4096), (('tolerance',), 3e-11),
    (('budget', 'train_seconds'), 20000), (('damping',), np.nan),
    (('require_gpu',), False), (('projection_chunk',), True),
])
def test_model_statistics_distance_and_budget_cannot_change_silently(plan, field, value):
    changed = deepcopy(plan); target = changed
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    with pytest.raises(ValueError):
        trial.validate_plan(changed)


@pytest.mark.parametrize('fs,ess,passes', [
    (.001, 1., True), (.0008, .99, True), (.0012, .99, True),
    (.0007, .99, False), (.0013, .99, False), (.001, .97, False),
    (0., 1., False), (np.nan, 1., False), (.001, np.nan, False),
])
def test_unseen_gate_validates_without_modifying_or_shrinking_step(plan, fs, ess, passes):
    report = dict(sampled_fs_sine=fs, reweighting_ess_fraction=ess)
    before = deepcopy(report)
    assert trial.validation_decision(report, plan['calibration']) is passes
    for key in report:
        np.testing.assert_equal(report[key], before[key])


def test_optimizer_reset_and_checkpoint_restore_all_streams(tmp_path):
    params = {'w': jnp.asarray([.2, -.3])}
    optimizer = _manual_adam_init(params)
    states = jnp.tile(jnp.asarray([1]*18+[0]*18), (4, 1))
    state = (params, optimizer, jax.random.PRNGKey(7), states,
             jax.random.PRNGKey(8), jnp.roll(states, 1, axis=1))
    path = tmp_path/'checkpoint.pkl'
    trial.shared.save_state(path, state, dict(registration_sha256='registered', completed_steps=0))
    replay, meta = trial.shared.restore_state(path, 'registered')
    assert trial.pytree_sha256(state) == trial.pytree_sha256(replay)
    assert int(replay[1][0]) == 0 and meta['completed_steps'] == 0
    assert all(np.count_nonzero(np.asarray(x)) == 0 for x in jax.tree.leaves(replay[1][1:]))
    payload = load_checkpoint(path)
    payload['metadata']['calibration_states'] = np.array(payload['metadata']['calibration_states'], copy=True)
    payload['metadata']['calibration_states'][0, 0] ^= 1
    save_checkpoint(path, payload['params'], optimizer_state=payload['optimizer_state'], metadata=payload['metadata'])
    with pytest.raises(ValueError, match='seal'):
        trial.shared.restore_state(path, 'registered')


def test_qr_arm_has_same_parameter_shapes_without_optimizer_inheritance():
    config = MixedClebschConfig(length=6, multiplicity_cap=9, maximum_q=3)
    params = init_mixed_clebsch_family(jax.random.PRNGKey(11), config)['ternary_free']
    raw, _ = trial.initial_parameters(params, 'raw', config)
    canonical, report = trial.initial_parameters(params, 'canonical', config)
    assert trial.mixed_parameter_count(raw) == trial.mixed_parameter_count(canonical)
    assert jax.tree.structure(raw) == jax.tree.structure(canonical)
    assert [x.shape for x in jax.tree.leaves(raw)] == [x.shape for x in jax.tree.leaves(canonical)]
    assert report
    with pytest.raises(ValueError):
        trial.initial_parameters(params, 'repeated_qr', config)
