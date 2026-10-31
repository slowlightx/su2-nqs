"""Exact checkpoint/protocol checks; never allocate the N36 production model."""
from copy import deepcopy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import triangle_clebsch_midpoint_audit as audit
from projects.clebsch_tree_nqs.workflows import triangle_optimizer_comparison as shared
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint

jax.config.update('jax_enable_x64', True)


def make_snapshot(tmp_path):
    identity = dict(plan_sha256='fixture', sources={})
    registration = dict(identity, sha256=canonical_sha256(identity))
    p = {'w': jnp.array([1., 2.], jnp.float64)}
    state = (p, (jnp.array(46), p, p), jax.random.PRNGKey(31),
             jnp.array([[0, 1], [1, 0]]), jax.random.PRNGKey(32), jnp.array([[1, 0], [0, 1]]))
    meta = dict(registration_sha256=registration['sha256'], completed_steps=46,
                parent_absolute_completed_steps=296, absolute_completed_steps=342,
                optimizer_base_step=0, history=[{}] * 46)
    source = tmp_path / 'live.pkl'
    shared.save_state(source, state, meta)
    target = tmp_path / 'frozen' / 'checkpoint.pkl'
    return source, target, registration, state


def test_capture_preserves_complete_midpoint_not_128_and_is_exclusive(tmp_path):
    source, target, registration, state = make_snapshot(tmp_path)
    row = audit.capture_checkpoint(source, target, registration)
    assert row['completed_steps'] == 46 and row['absolute_completed_steps'] == 342
    assert source.read_bytes() == target.read_bytes()
    restored, _ = shared.restore_state(target, registration['sha256'])
    for a, b in zip(jax.tree.leaves(restored), jax.tree.leaves(state)):
        np.testing.assert_array_equal(a, b)
    assert target.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        audit.capture_checkpoint(source, target, registration)
    with pytest.raises(ValueError, match='live checkpoint'):
        audit.capture_checkpoint(source, source, registration)


def test_capture_rejects_incomplete_optimizer_or_counter_state(tmp_path):
    source, target, registration, _ = make_snapshot(tmp_path)
    payload = load_checkpoint(source)
    payload['metadata']['absolute_completed_steps'] = 128
    save_checkpoint(source, payload['params'], optimizer_state=payload['optimizer_state'], metadata=payload['metadata'])
    with pytest.raises(ValueError, match='counters'):
        audit.capture_checkpoint(source, target, registration)


def plan_fixture(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'projects/clebsch_tree_nqs/configs/remote/ct6_wide_production_20261023/j0125_w28.json'
    source = json.loads(path.read_text())
    plan = dict(task='ct6_immutable_midpoint_audit_v1', audit=dict(audit.AUDIT_SPEC),
                maximum_seconds=5400, requested_A100_hours=1.5, campaign_total_A100_hours=240,
                aggregate_reserved_A100_hours=167.0127777777778,
                seed=2610239201, forbidden_seeds=[2610238003, 2610238013],
                source_plan_sha256=canonical_sha256(source), cost_forecast_seconds=2800,
                output=str(tmp_path / 'audit'),
                snapshot={'snapshot_path': str(tmp_path / 'snapshot.pkl'),
                          'source_plan_sha256': canonical_sha256(source)})
    return plan, source


def test_fixed_audit_contract_accepts_midpoint_independent_of128_gate(tmp_path):
    plan, source = plan_fixture(tmp_path)
    audit.validate_plan(plan, source)
    assert source['training']['steps'] == 128


@pytest.mark.parametrize('change', ['samples', 'chains', 'budget', 'total', 'seed', 'old_seed',
                                    'mutable', 'source', 'snapshot_plan', 'live_output', 'forecast'])
def test_audit_rejects_scope_budget_or_seed_changes(tmp_path, change):
    plan, source = plan_fixture(tmp_path)
    if change == 'samples': plan['audit']['samples'] = 65536
    if change == 'chains': plan['audit']['chains'] = 128
    if change == 'budget': plan['maximum_seconds'] = 5401
    if change == 'total': plan['aggregate_reserved_A100_hours'] = 241
    if change == 'seed': plan['seed'] = True
    if change == 'old_seed': plan['seed'] = source['seeds']['audit']
    if change == 'mutable': plan['snapshot']['snapshot_path'] = str(Path(source['output']) / 'checkpoint.pkl')
    if change == 'source': source['j2_over_j1'] = 0.
    if change == 'snapshot_plan': plan['snapshot']['source_plan_sha256'] = 'different_coupling'
    if change == 'live_output': plan['output'] = source['output'] + '/audit'
    if change == 'forecast': plan['cost_forecast_seconds'] = float('nan')
    with pytest.raises(ValueError):
        audit.validate_plan(plan, source)


def test_runtime_is_audit_only_preserves_real_midpoint_and_raw_data(tmp_path, monkeypatch):
    source, target, registration, _ = make_snapshot(tmp_path)
    snapshot = audit.capture_checkpoint(source, target, registration)
    plan, old_plan = plan_fixture(tmp_path)
    source_path = tmp_path / 'source_plan.json'
    source_path.write_text(json.dumps(old_plan))
    plan.update(snapshot=snapshot, source_plan_path=str(source_path), output=str(tmp_path / 'audit'),
                campaign_deadline_utc='2030-01-01T00:00:00+00:00', cost_forecast_seconds=2800)
    plan['snapshot']['source_plan_sha256'] = canonical_sha256(old_plan)
    calls = []
    def forbidden(*args, **kwargs):
        raise AssertionError('training transition must never execute')
    monkeypatch.setattr(audit, 'verify_sources', lambda _: None)
    monkeypatch.setattr(audit.jax, 'default_backend', lambda: 'gpu')
    monkeypatch.setattr(audit.jax, 'devices', lambda: [type('Device', (), {'device_kind': 'NVIDIA A100 80GB PCIe'})()])
    monkeypatch.setattr(audit.production, 'runtime', lambda *args: dict(images=range(432), bonds=range(216), transition=forbidden))
    def terminal(rt, p, spec, seed, output):
        calls.append((spec, seed))
        np.savez_compressed(output / 'terminal_audit.npz', bits=np.array([[0, 1]]), local_energies=np.array([[-1.]]))
        return dict(energy_per_site=-.5, variance_per_site=.01, mixing_certified=False)
    monkeypatch.setattr(audit.shared, 'terminal_audit', terminal)
    before = target.read_bytes()
    result = audit.run(plan, canonical_sha256(plan))
    assert calls == [(audit.AUDIT_SPEC, plan['seed'])]
    assert result['snapshot']['completed_steps'] == 46
    assert result['optimizer_updates_committed'] == 0
    assert result['audit_full_H_centers'] == 16384
    assert result['variance_node_complete'] is False
    assert target.read_bytes() == before
    assert (Path(plan['output']) / 'terminal_audit.npz').is_file()
    with pytest.raises(FileExistsError):
        audit.run(plan, canonical_sha256(plan))


def test_two_gpus_are_rejected_before_reading_state_or_writing_outputs(tmp_path, monkeypatch):
    plan, old_plan = plan_fixture(tmp_path)
    source_path = tmp_path / 'source_plan.json'
    source_path.write_text(json.dumps(old_plan))
    plan.update(source_plan_path=str(source_path), campaign_deadline_utc='2030-01-01T00:00:00+00:00')
    monkeypatch.setattr(audit, 'verify_sources', lambda _: None)
    monkeypatch.setattr(audit.jax, 'default_backend', lambda: 'gpu')
    device = type('Device', (), {'device_kind': 'NVIDIA A100 80GB PCIe'})()
    monkeypatch.setattr(audit.jax, 'devices', lambda: [device, device])
    with pytest.raises(RuntimeError, match='exactly one'):
        audit.run(plan, canonical_sha256(plan))
    assert not Path(plan['output']).exists()
