"""Host portability must replay unchanged physics/state, never weaken gates."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.workflows import triangle_ct6_portable_runtime as portable
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint
from test_ct6_evidence_guarded_continuation import unresolved_workflow
from test_ct6_gate_replay_repair import gate_fixture
from test_ct6_validation_repair_launcher import manifest_fixture


def portable_manifest_fixture(monkeypatch, tmp_path):
    plan, original, current, old = manifest_fixture(monkeypatch, tmp_path)
    plan.update(output=str(tmp_path/'output'), budget={'train_seconds': 68400})
    original['plan_sha256'] = current['plan_sha256'] = old['plan_sha256'] = canonical_sha256(plan)
    old['registration_sha256'] = canonical_sha256(original)
    current['numerical_runtime']['node'] = 'any-other-host'
    frozen = tmp_path/'frozen.pkl'; frozen.write_bytes(b'sealed checkpoint')
    old['original_artifacts']['checkpoint.pkl'] = file_sha256(frozen)
    manifest = dict(type=portable.PORTABLE_TYPE, runtime_policy=portable.RUNTIME_POLICY,
        launcher_sha256=file_sha256(Path(portable.__file__)), plan_sha256=canonical_sha256(plan),
        replay_budget_seconds=7200, train_budget_seconds=68400,
        original_repair_manifest=old, original_repair_manifest_sha256=canonical_sha256(old),
        frozen_checkpoint_path=str(frozen))
    return plan, original, current, manifest


def test_portable_manifest_allows_only_hostname_and_records_host_agnostic_identity(monkeypatch, tmp_path):
    plan, original, current, manifest = portable_manifest_fixture(monkeypatch, tmp_path)
    record = portable.checked_portable_record(manifest, canonical_sha256(manifest),
        plan=plan, original=original, current=current)
    assert 'node' not in record['numerical_runtime_without_hostname']
    assert record['original_registration_preserved'] and record['replay_required_on_every_allocation']
    assert not record['new_direction_draw'] and not record['scientific_plan_changed']
    current['numerical_runtime']['node'] = 'yet-another-compatible-host'
    assert portable.checked_portable_record(manifest, canonical_sha256(manifest),
        plan=plan, original=original, current=current) == record


@pytest.mark.parametrize('change', ['runtime', 'dependency', 'plan', 'missing_node',
                                  'policy', 'launcher', 'old_manifest', 'frozen', 'budget'])
def test_portability_rejects_nonhost_identity_or_seal_changes(monkeypatch, tmp_path, change):
    plan, original, current, manifest = portable_manifest_fixture(monkeypatch, tmp_path)
    if change == 'runtime': current['numerical_runtime']['x64'] = False
    if change == 'dependency': current['sources']['dependency.py'] = 'changed'
    if change == 'plan': current['plan_sha256'] = 'changed'
    if change == 'missing_node': del current['numerical_runtime']['node']
    if change == 'policy': manifest['runtime_policy'] = 'any_runtime_without_checks'
    if change == 'launcher': manifest['launcher_sha256'] = '0'*64
    if change == 'old_manifest': manifest['original_repair_manifest']['plan_sha256'] = 'changed'
    if change == 'frozen': Path(manifest['frozen_checkpoint_path']).write_bytes(b'changed')
    if change == 'budget': manifest['replay_budget_seconds'] = 7201
    with pytest.raises(ValueError):
        portable.checked_portable_record(manifest, canonical_sha256(manifest),
            plan=plan, original=original, current=current)


def replay_report():
    return dict(protocol='exact_discrete_relative_float_v1', tolerance=2e-11, passed=True,
        leaves=[dict(path='[0][0]', shape=[2], discrete=True, finite=True, passed=True,
                     bitwise_exact=True),
                dict(path='[1]', shape=[2], discrete=False, finite=True, passed=True,
                     relative_vector_error=1e-14, absolute_error=1e-15)])


@pytest.mark.parametrize('change', ['none', 'bit', 'finite', 'relative', 'tolerance',
                                  'gate', 'classification', 'empty', 'protocol'])
def test_portable_replay_keeps_exact_discrete_and_original_float_gate_checks(change):
    report = replay_report(); gate = gate_fixture(); actual = deepcopy(gate)
    if change == 'bit': report['leaves'][0]['bitwise_exact'] = False
    if change == 'finite': report['leaves'][1]['finite'] = False
    if change == 'relative': report['leaves'][1]['relative_vector_error'] = 3e-11
    if change == 'tolerance': report['tolerance'] = 3e-11
    if change == 'gate': actual['passed'] = False
    if change == 'classification': actual['slope_evidence'] = 'confirmed_descent'
    if change == 'empty': report['leaves'] = []
    if change == 'protocol': report['protocol'] = 'approximate'
    if change == 'none':
        portable.require_replay_evidence(report, gate, actual, 2e-11)
    else:
        with pytest.raises(ValueError):
            portable.require_replay_evidence(report, gate, actual, 2e-11)


def test_second_portable_writer_cannot_acquire_output(tmp_path):
    with portable.single_writer(tmp_path):
        with pytest.raises(RuntimeError, match='another portable continuation'):
            with portable.single_writer(tmp_path):
                pass
    with portable.single_writer(tmp_path):
        pass


def runtime_fixture(monkeypatch, tmp_path):
    plan, _, calls = unresolved_workflow(monkeypatch, tmp_path)
    workflow = portable.workflow; digest = canonical_sha256(plan); out = Path(plan['output'])
    workflow.run(plan, 'pilot', digest); workflow.run(plan, 'replay', digest)
    originals = {name: file_sha256(out/name) for name in portable.repair.SEALED_ARTIFACTS}
    frozen = tmp_path/'frozen_checkpoint.pkl'; frozen.write_bytes((out/'checkpoint.pkl').read_bytes())
    manifest = dict(original_repair_manifest={'original_artifacts': originals},
        frozen_checkpoint_path=str(frozen), replay_budget_seconds=7200)
    monkeypatch.setattr(portable, 'checked_portable_record', lambda *a, **k: {'portable': True})
    monkeypatch.setattr(portable, 'transitive_local_source_manifest', lambda *a, **k: {'source': 'frozen'})
    # The historical plan stays pinned, but deployment may run on another host.
    monkeypatch.setattr(workflow, 'numerical_runtime_fingerprint', lambda: {'node': 'other-host'})
    return plan, digest, manifest, out, originals, calls


def test_fixed_replay_then32_updates_preserves_old_artifacts_and_rng_lineage(monkeypatch, tmp_path):
    plan, digest, manifest, out, originals, calls = runtime_fixture(monkeypatch, tmp_path)
    result = portable.run(plan, digest, manifest, canonical_sha256(manifest))
    assert result['completed_steps'] == 32 and result['initial_slope_evidence'] == 'unresolved'
    assert calls == dict(transition=36, confirmation=3, audit=1, branch=1)
    assert all(file_sha256(out/name) == seal for name, seal in originals.items() if name != 'checkpoint.pkl')
    assert file_sha256(Path(manifest['frozen_checkpoint_path'])) == originals['checkpoint.pkl']
    cp = load_checkpoint(out/'checkpoint.pkl'); meta = cp['metadata']
    assert meta['portable_execution_sha256'] == file_sha256(out/'portable_execution.json')
    assert meta['portable_replay_sha256'] == file_sha256(Path(meta['portable_replay_path']))
    replay = json.loads(Path(meta['portable_replay_path']).read_text())
    assert replay['actual_numerical_runtime']['node'] == 'other-host'
    assert replay['optimizer_updates_committed'] == 0 and not replay['new_scientific_confirmation_draw']
    assert result['numerical_method_changed_by_portability'] is False
    assert result['additional_portable_replay_work'] == dict(full_H_training_centers=4096,
        full_H_parent_centers=2048, full_H_JVP_tangent_centers=2048)
    before_calls = dict(calls); before_result = file_sha256(out/'results.json')
    cached = portable.run(plan, digest, manifest, canonical_sha256(manifest))
    assert cached['cached_verified_endpoint'] and calls == before_calls
    assert file_sha256(out/'results.json') == before_result


def test_failed_crosshost_replay_never_commits_or_redraws(monkeypatch, tmp_path):
    plan, digest, manifest, out, originals, calls = runtime_fixture(monkeypatch, tmp_path)
    original_runtime = portable.workflow.runtime
    runtime = original_runtime(plan, None); old = runtime['transition']
    def changed_transition(state):
        next_state, info, timing = old(state)
        next_state = (*next_state[:2], next_state[2]+1, *next_state[3:])
        return next_state, info, timing
    runtime['transition'] = changed_transition
    monkeypatch.setattr(portable.workflow, 'runtime', lambda *a: runtime)
    with pytest.raises(ArithmeticError, match='no training started'):
        portable.run(plan, digest, manifest, canonical_sha256(manifest))
    assert file_sha256(out/'checkpoint.pkl') == originals['checkpoint.pkl']
    assert calls == dict(transition=4, confirmation=3, audit=0, branch=1)
    latest = json.loads((out/'portable_runtime_latest.json').read_text())
    assert latest['status'] == 'failed'
