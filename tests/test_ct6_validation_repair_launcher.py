"""A validation-only launcher must preserve the registered numerical path."""
from copy import deepcopy
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.workflows import triangle_ct6_validation_repair as repair
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint
from test_ct6_evidence_guarded_continuation import unresolved_workflow


ORIGINAL = 'def checked_direction_release():\n    return False\n\ndef runtime(x):\n    return x+1\n'
REPAIRED = ('def checked_direction_release():\n    return True\n\n'
            'def direction_gate_replay_matches():\n    return True\n\ndef runtime(x):\n    return x+1\n')


def test_ast_allowlist_keeps_every_numerical_node_unchanged():
    repair.assert_validation_only_source_change(ORIGINAL, REPAIRED)
    for changed in (REPAIRED.replace('x+1', 'x+2'), REPAIRED+'\nimport os\n',
                    REPAIRED+'\nprint("extra execution")\n'):
        with pytest.raises(ValueError, match='outside the two replay validators'):
            repair.assert_validation_only_source_change(ORIGINAL, changed)


def manifest_fixture(monkeypatch, tmp_path):
    before, after = tmp_path/'before.py', tmp_path/'after.py'
    before.write_text(ORIGINAL); after.write_text(REPAIRED)
    monkeypatch.setattr(repair.workflow, '__file__', str(after))
    plan = {'numerical_plan': 'unchanged'}
    original = dict(plan_sha256=canonical_sha256(plan), numerical_runtime={'node': 'same'},
                    sources={repair.WORKFLOW_PATH: file_sha256(before), 'dependency.py': 'frozen'})
    current = deepcopy(original); current['sources'][repair.WORKFLOW_PATH] = file_sha256(after)
    manifest = dict(repair_type=repair.REPAIR_TYPE, original_workflow_snapshot=str(before),
        original_workflow_sha256=file_sha256(before), repaired_workflow_sha256=file_sha256(after),
        launcher_sha256=file_sha256(Path(repair.__file__)), plan_sha256=canonical_sha256(plan),
        registration_sha256=canonical_sha256(original),
        original_artifacts={name: '0'*64 for name in repair.SEALED_ARTIFACTS})
    return plan, original, current, manifest


def test_explicit_repair_manifest_seals_source_identity_and_original_registration(monkeypatch, tmp_path):
    plan, original, current, manifest = manifest_fixture(monkeypatch, tmp_path)
    row = repair.checked_repair_record(manifest, canonical_sha256(manifest),
        plan=plan, original=original, current=current)
    assert row['original_registration_sha256'] == canonical_sha256(original)
    assert row['actual_execution_identity_sha256'] == canonical_sha256(current)
    assert not row['numerical_method_changed'] and not row['new_direction_draw']
    for key in ('launcher_sha256', 'plan_sha256', 'registration_sha256', 'repaired_workflow_sha256'):
        invalid = deepcopy(manifest); invalid[key] = 'f'*64
        with pytest.raises(ValueError):
            repair.checked_repair_record(invalid, canonical_sha256(invalid),
                plan=plan, original=original, current=current)


@pytest.mark.parametrize('change', ['plan', 'node', 'dependency', 'extra_source', 'extra_field'])
def test_repair_rejects_changed_numerical_identity(monkeypatch, tmp_path, change):
    plan, original, current, manifest = manifest_fixture(monkeypatch, tmp_path)
    if change == 'plan': current['plan_sha256'] = 'other'
    if change == 'node': current['numerical_runtime']['node'] = 'other'
    if change == 'dependency': current['sources']['dependency.py'] = 'changed'
    if change == 'extra_source': current['sources']['new.py'] = 'changed'
    if change == 'extra_field': current['other'] = True
    with pytest.raises(ValueError):
        repair.checked_repair_record(manifest, canonical_sha256(manifest),
            plan=plan, original=original, current=current)


def test_train_only_launcher_reuses_sealed_state_and_records_execution_repair(monkeypatch, tmp_path):
    plan, _, calls = unresolved_workflow(monkeypatch, tmp_path)
    workflow = repair.workflow; digest = canonical_sha256(plan); out = Path(plan['output'])
    workflow.run(plan, 'pilot', digest); workflow.run(plan, 'replay', digest)
    manifest = {'original_artifacts': {name: file_sha256(out/name) for name in repair.SEALED_ARTIFACTS}}
    record = dict(validation_only=True, original_registration_preserved=True)
    monkeypatch.setattr(repair, 'checked_repair_record', lambda *args, **kwargs: record)
    monkeypatch.setattr(repair, 'transitive_local_source_manifest', lambda *args, **kwargs: {'source': 'frozen'})
    frozen = {name: digest for name, digest in manifest['original_artifacts'].items() if name != 'checkpoint.pkl'}
    result = repair.run(plan, digest, manifest, canonical_sha256(manifest))
    assert result['completed_steps'] == 32 and result['initial_slope_evidence'] == 'unresolved'
    assert result['numerical_method_changed_by_repair'] is False
    assert calls == dict(transition=35, confirmation=2, audit=1, branch=1)
    assert all(file_sha256(out/name) == seal for name, seal in frozen.items())
    checkpoint = load_checkpoint(out/'checkpoint.pkl')
    assert checkpoint['metadata']['execution_repair_sha256'] == file_sha256(out/'execution_repair.json')
    assert result['execution_repair_sha256'] == checkpoint['metadata']['execution_repair_sha256']


def test_launcher_rejects_changed_original_artifacts_before_training(monkeypatch, tmp_path):
    plan, _, calls = unresolved_workflow(monkeypatch, tmp_path)
    workflow = repair.workflow; digest = canonical_sha256(plan); out = Path(plan['output'])
    workflow.run(plan, 'pilot', digest); workflow.run(plan, 'replay', digest)
    manifest = {'original_artifacts': {name: file_sha256(out/name) for name in repair.SEALED_ARTIFACTS}}
    manifest['original_artifacts']['direction_check.npz'] = '0'*64
    monkeypatch.setattr(repair, 'checked_repair_record', lambda *args, **kwargs: {'validation_only': True})
    monkeypatch.setattr(repair, 'transitive_local_source_manifest', lambda *args, **kwargs: {'source': 'frozen'})
    before = dict(calls)
    with pytest.raises(ValueError, match='sealed original repair artifact changed'):
        repair.run(plan, digest, manifest, canonical_sha256(manifest))
    assert calls == before and not (out/'execution_repair.json').exists()
