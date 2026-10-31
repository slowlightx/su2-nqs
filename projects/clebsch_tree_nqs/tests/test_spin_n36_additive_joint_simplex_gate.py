from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from projects.clebsch_tree_nqs.workflows import (
    spin_n36_additive_joint_simplex_gate as workflow,
)
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import file_sha256, transitive_local_source_manifest


ROOT = Path(__file__).resolve().parents[3]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/"
    "spin_n36_additive_joint_simplex_variance_gate.json"
)
RESULT = (
    ROOT
    / "projects/clebsch_tree_nqs/results/"
    "spin_n36_additive_joint_simplex_variance_gate_v1/results.json"
)


def _original_source_manifest(payload: dict, snapshot: Path) -> dict[str,str]:
    """Validate original bytes AND their complete local import closure.

    This is a historical negative experiment, not a current-code result.
    Never re-sign it when shared kernels evolve. The33 original file hashes
    come from its existing sealed result, not a newly generated manifest.
    """
    expected=payload['source_manifest']
    for relative,sha in expected.items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('invalid archived source path')
        saved=snapshot/path
        if not saved.is_file() or file_sha256(saved)!=sha:
            raise ValueError(f'changed archived source: {relative}')
    entry=snapshot/Path(workflow.__file__).resolve().relative_to(workflow.ROOT)
    manifest=transitive_local_source_manifest(
        (entry,),root=snapshot,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    if manifest!=expected:
        raise ValueError('archived local import closure differs from sealed manifest')
    return manifest


def test_additive_joint_simplex_contract_is_nested_matched_and_local() -> None:
    spec = json.loads(CONFIG.read_text())
    workflow._validate(spec)
    assert spec["model"]["joint_residual_backbone_training"] is True
    assert spec["model"]["magnetic_indices_visible"] is False
    assert spec["training"]["maximum_variance_increase"] == 0.0
    assert spec["audit"]["untouched_by_training"] is True
    assert spec["decision"]["remote_production_authorized_by_this_config"] is False


def test_sealed_joint_simplex_gate_retains_carrier_but_rejects_geometry_claim() -> None:
    payload = json.loads(RESULT.read_text())
    assert payload["status"] == "complete"
    assert payload["task"] == workflow.TASK
    unsealed = copy.deepcopy(payload)
    seal = unsealed.pop("result_sha256")
    assert seal == canonical_sha256(unsealed)
    manifest = _original_source_manifest(payload,RESULT.parent/'source_snapshot')
    assert payload["source_manifest"] == manifest
    assert payload["source_manifest_sha256"] == canonical_sha256(manifest)
    gates = payload["gates"]
    assert gates["exact_additive_nesting_and_local_f"] is True
    assert gates["joint_backbone_changed"] is True
    assert gates["checkpoint_resume"] is True
    assert gates["carrier_energy_and_variance_retained"] is True
    assert gates["multi_channel_support"] is True
    assert gates["importance_ess"] is True
    assert gates["physical_residual_beats_scramble"] is False
    assert gates["local_additive_joint_gate_passed"] is False
    assert gates["remote_production_authorized"] is False
    paired = payload["paired_summary"]
    assert paired["physical_minus_carrier_energy_per_site_mean"] < 0.0
    assert paired["physical_minus_carrier_variance_per_site_mean"] < 0.0
    assert paired["physical_minus_scramble_energy_per_site_mean"] > 0.0
    assert payload["minimum_audit_ess_fraction"] > 0.99
    assert payload["maximum_zero_residual_nesting_error"] < 1.0e-15


def test_historical_simplex_source_guard_rejects_a_changed_kernel(tmp_path) -> None:
    payload=json.loads(RESULT.read_text())
    snapshot=tmp_path/'source_snapshot'
    shutil.copytree(RESULT.parent/'source_snapshot',snapshot)
    kernel=snapshot/'su2_nqs/vmc/many_cell_tree.py'
    kernel.write_text(kernel.read_text()+'\n# intentional corrupt test fixture\n')
    with pytest.raises(ValueError,match='changed archived source'):
        _original_source_manifest(payload,snapshot)
