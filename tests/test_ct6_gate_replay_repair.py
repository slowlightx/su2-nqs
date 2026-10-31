"""Replay-derived bounds may round; decisions and physical arrays stay exact."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import triangle_clebsch_continuation as campaign
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from test_ct6_evidence_guarded_continuation import unresolved_workflow


def gate_fixture():
    return dict(passed=True, checks={'no_resolved_ascent': True},
                slope_evidence='unresolved', confirmed_descent=False,
                thresholds={'slope_standard_errors': 2},
                slope_lower_bound_per_site=-1.4986205340798977e-6,
                slope_upper_bound_per_site=1.2218966436736859e-5)


def test_observed_gpu_roundoff_preserves_gate_at_original_relative_tolerance():
    expected = gate_fixture(); actual = deepcopy(expected)
    actual['slope_lower_bound_per_site'] = -1.4986205340798274e-6
    actual['slope_upper_bound_per_site'] = 1.2218966436736789e-5
    assert actual != expected
    assert campaign.direction_gate_replay_matches(expected, actual, tolerance=2e-11)


@pytest.mark.parametrize('change', ['decision', 'evidence', 'descent', 'threshold',
    'missing', 'extra', 'nan', 'inf', 'bool_bound', 'outside', 'tiny_relative'])
def test_gate_comparison_rejects_semantic_and_numerical_changes(change):
    expected = gate_fixture(); actual = deepcopy(expected)
    if change == 'decision': actual['checks']['no_resolved_ascent'] = False
    if change == 'evidence': actual['slope_evidence'] = 'confirmed_descent'
    if change == 'descent': actual['confirmed_descent'] = True
    if change == 'threshold': actual['thresholds']['slope_standard_errors'] = 1.999999999999
    if change == 'missing': del actual['slope_lower_bound_per_site']
    if change == 'extra': actual['new_field'] = False
    if change == 'nan': actual['slope_lower_bound_per_site'] = float('nan')
    if change == 'inf': actual['slope_lower_bound_per_site'] = float('inf')
    if change == 'bool_bound': actual['slope_lower_bound_per_site'] = False
    if change == 'outside': actual['slope_lower_bound_per_site'] *= 1+4e-11
    if change == 'tiny_relative':
        expected['slope_lower_bound_per_site'] = -1e-40
        actual['slope_lower_bound_per_site'] = -2e-40
    assert not campaign.direction_gate_replay_matches(expected, actual, tolerance=2e-11)


def test_training_uses_existing_passed_replay_with_bound_roundoff_without_redraw(monkeypatch, tmp_path):
    plan, _, calls = unresolved_workflow(monkeypatch, tmp_path)
    digest = canonical_sha256(plan); out = Path(plan['output'])
    campaign.run(plan, 'pilot', digest); campaign.run(plan, 'replay', digest)
    replay = json.loads((out/'process_replay.json').read_text())
    gate = replay['independently_recomputed_gate']
    for key in ('slope_lower_bound_per_site', 'slope_upper_bound_per_site'):
        gate[key] = float(np.nextafter(gate[key], np.inf))
    atomic_json(out/'process_replay.json', replay)
    result = campaign.run(plan, 'train', digest)
    assert result['completed_steps'] == 32 and result['initial_slope_evidence'] == 'unresolved'
    assert calls == dict(transition=35, confirmation=2, audit=1, branch=1)
