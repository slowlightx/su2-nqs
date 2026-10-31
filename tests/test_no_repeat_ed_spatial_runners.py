from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects/clebsch_tree_nqs"
EXACT_SUPPORT_SOURCES = (
    PROJECT / "run.py",
    PROJECT / "workflows/common.py",
    PROJECT / "workflows/spin.py",
    PROJECT / "workflows/spin_spatial_vit.py",
    PROJECT / "workflows/spin_spatial_vit_sr.py",
    PROJECT / "workflows/spin_strict_schur_spatial.py",
    PROJECT / "workflows/hubbard.py",
)


def test_active_campaign_does_not_call_eigensolvers() -> None:
    forbidden_names = {"eigsh", "eigs", "eigh", "eigvalsh"}
    violations: list[tuple[str, str]] = []
    for path in EXACT_SUPPORT_SOURCES:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
                violations.append((path.name, node.func.id))
            if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_names:
                violations.append((path.name, node.func.attr))
    assert violations == []


def test_active_configs_require_stored_scalar_energy_anchors() -> None:
    for path in sorted((PROJECT / "configs").glob("*.json")):
        config = json.loads(path.read_text())
        for case in config["cases"]:
            assert isinstance(case["exact_energy_anchor"], (int, float))
            provenance = case["anchor_provenance"].lower()
            assert "no eigensolve" in provenance or "stored" in provenance


def test_result_schema_explicitly_records_no_diagonalization() -> None:
    source = (PROJECT / "workflows/common.py").read_text()
    assert '"hamiltonian_diagonalization_performed": False' in source
