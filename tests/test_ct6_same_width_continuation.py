"""Sealed width20 parents resume full optimizer/RNG under an immutable runtime."""
import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import triangle_clebsch_continuation as campaign
from su2_nqs.frontier.artifacts import canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.adam import _manual_adam_init
from su2_nqs.vmc.checkpoint import load_checkpoint

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'projects/clebsch_tree_nqs/configs/remote/ct6_continue_20261021'
SNAPSHOT = ROOT/'projects/clebsch_tree_nqs/results/remote_review_20261021/ct6_width_20261020'
PLANS = sorted(CONFIG.glob('*_w20_continue32.json'))


def test_fixed_four_parent_continuations_do_not_release_width24_or_reset_optimizer():
    assert len(PLANS) == 4
    seeds = []
    total = 0
    for path in PLANS:
        p = json.loads(path.read_text()); campaign.validate_plan(p)
        assert p['task'] == campaign.SAME_WIDTH_TASK
        assert p['model'] == p['parent']['model']
        assert p['model']['multiplicity_cap'] == 20
        assert p['training']['steps'] == 32 and p['training']['samples'] == 2048
        assert p['audit']['samples'] == 16384 and p['audit']['chains'] == 128
        assert p['continuation_profile'] == 'inherit_full_optimizer_sampler_rng'
        assert 'optimization_control' not in p
        assert p['runtime_contract']['node'] in {'kim-compute-03', 'kim-compute-04', 'kim-compute-05'}
        seeds += list(p['seeds'].values())
        total += sum(p['budget'][k] for k in ('pilot_seconds', 'replay_seconds', 'train_seconds'))
    assert len(seeds) == len(set(seeds)) and total == 25*3600
    summary = json.loads((CONFIG/'campaign.json').read_text())
    assert summary['released_widths'] == [20]
    assert len(summary['width18_reference_states_preserved']) >= 5
    for row in summary['cases']:
        assert row['conservative_train_seconds_forecast'] < 18000
        for stage in row['planned_submission_stages']:
            assert '--kill-on-invalid-dep=yes' in stage['command_without_dependency']


@pytest.mark.parametrize('change', ['width', 'optimizer', 'steps', 'seal', 'runtime'])
def test_same_width_contract_rejects_unregistered_changes(change):
    p = json.loads(PLANS[0].read_text())
    if change == 'width': p['model']['multiplicity_cap'] = 24
    if change == 'optimizer': p['optimization_control'] = 'same_parent_same_width_Adam_reset'
    if change == 'steps': p['training']['steps'] = 65
    if change == 'seal': del p['parent']['sealed_files']['checkpoint.pkl']
    if change == 'runtime': del p['runtime_contract']['node']
    with pytest.raises(ValueError): campaign.validate_plan(p)


@pytest.mark.parametrize('path', PLANS, ids=lambda p: p.stem)
def test_real_audited_parents_restore_all_moments_rng_and_chains_exactly(path, tmp_path):
    p = json.loads(path.read_text()); parent = Path(p['parent']['directory'])
    if not parent.exists():
        parent = SNAPSHOT/(p['case']+'_w20')
    assert parent.exists(), 'sealed parent must be present locally or at its remote production path'
    p['parent']['directory'] = str(parent)
    state, lineage = campaign.load_parent(p)
    original = load_checkpoint(parent/'checkpoint.pkl')
    assert pytree_sha256(state[0]) == original['metadata']['params_sha256']
    assert pytree_sha256(state[1]) == original['metadata']['optimizer_sha256']
    assert int(state[1][0]) == original['metadata']['optimizer_base_step'] + 16
    assert lineage['absolute_completed_steps'] == (264 if p['j2_over_j1'] == 0 else 284)
    assert np.shape(state[3]) == np.shape(state[5]) == (64, 36)
    out = tmp_path/'restored.pkl'
    campaign.shared.save_state(out, state, dict(original['metadata']))
    roundtrip, _ = campaign.shared.restore_state(out, lineage['registration_sha256'])
    for left, right in zip(jax.tree.leaves(state), jax.tree.leaves(roundtrip), strict=True):
        np.testing.assert_array_equal(left, right)
    replay = json.loads((parent/'process_replay.json').read_text())
    assert replay['report']['passed'] and replay['report']['tolerance'] == 2e-11
    assert all(row['passed'] for row in replay['report']['leaves'])


def test_first_pilot_seals_actual_runtime_and_later_process_cannot_drift(monkeypatch, tmp_path):
    p = json.loads(PLANS[0].read_text()); p['output'] = str(tmp_path/'output'); p['require_gpu'] = False
    pars = dict(x=jnp.array([.1, .2])); _, first, second = _manual_adam_init(pars)
    first = jax.tree.map(lambda x: x+.3, first); second = jax.tree.map(lambda x: x+.7, second)
    state = (pars, (jnp.asarray(16), first, second), jax.random.PRNGKey(3),
             jnp.zeros((64, 36), jnp.int32), jax.random.PRNGKey(4), jnp.ones((64, 36), jnp.int32))
    lineage = dict(absolute_completed_steps=p['parent']['absolute_completed_steps'],
                   checkpoint_sha256='sealed-parent', model=p['model'])
    monkeypatch.setattr(campaign, 'load_parent', lambda _:(state, lineage))
    monkeypatch.setattr(campaign, 'transitive_local_source_manifest', lambda *a, **k:{'test':'fixed'})
    actual = dict(node=p['runtime_contract']['node'], platform='gpu', package_versions={'jax':'remote-pilot'})
    monkeypatch.setattr(campaign, 'numerical_runtime_fingerprint', lambda:copy.deepcopy(actual))
    seen = []
    def transition(old):
        seen.append(old)
        new = (jax.tree.map(lambda x:x+.001, old[0]), (old[1][0]+1, old[1][1], old[1][2]),
               old[2]+1, old[3], old[4]+1, old[5])
        return new, dict(accepted=True, full_H_calls=2048), dict(sampling_seconds=.01)
    monkeypatch.setattr(campaign, 'runtime', lambda *a:dict(transition=transition))
    monkeypatch.setattr(campaign.shared, 'preflight', lambda *a:dict(status='passed'))
    monkeypatch.setattr(campaign.shared, 'terminal_audit', lambda rt, pars, spec, seed, out:
                        dict(params_sha256=pytree_sha256(pars), seed=seed, samples=spec['samples'],
                             independent_from_training_and_calibration=True))
    seal = canonical_sha256(p); out = Path(p['output'])
    campaign.run(p, 'pilot', seal)
    for left, right in zip(jax.tree.leaves(state), jax.tree.leaves(seen[0]), strict=True):
        np.testing.assert_array_equal(left, right)
    registration = json.loads((out/'registration.json').read_text())
    assert registration['numerical_runtime'] == actual
    frozen = file_sha256(out/'checkpoint.pkl'); before = len(seen)
    actual['package_versions']['jax'] = 'different-compiler-runtime'
    with pytest.raises(ValueError, match='immutable stage'): campaign.run(p, 'replay', seal)
    assert file_sha256(out/'checkpoint.pkl') == frozen and len(seen) == before
    actual['package_versions']['jax'] = 'remote-pilot'; actual['node'] = 'different-node'
    with pytest.raises(ValueError, match='pinned numerical node'): campaign.run(p, 'replay', seal)
    actual['node'] = p['runtime_contract']['node']
    assert campaign.run(p, 'replay', seal)['status'] == 'replay_passed'
    assert file_sha256(out/'checkpoint.pkl') == frozen
    result = campaign.run(p, 'train', seal)
    assert result['absolute_completed_steps'] == lineage['absolute_completed_steps']+32
    cp = load_checkpoint(out/'checkpoint.pkl')
    assert int(cp['optimizer_state'][0]) == 48
    assert cp['metadata']['optimizer_policy'] == 'inherit_full_SR_Adam_moments'
    assert result['full_H_training_centers'] == 65536


def test_runtime_fingerprint_records_actual_environment_not_a_plan_version():
    runtime = campaign.numerical_runtime_fingerprint()
    assert runtime['node'] and runtime['package_versions']['jax']
    assert runtime['platform'] == jax.default_backend() and runtime['devices']
    assert runtime['x64'] is True
