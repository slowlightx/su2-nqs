from __future__ import annotations

from pathlib import Path
import sys

from su2_nqs.frontier.provenance import transitive_local_source_manifest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_transitive_manifest_tracks_relative_local_imports(tmp_path: Path) -> None:
    package = tmp_path / "demo"
    package.mkdir()
    (package / "__init__.py").write_text("")
    dependency = package / "dependency.py"
    dependency.write_text("VALUE = 1\n")
    entry = package / "entry.py"
    entry.write_text("from .dependency import VALUE\n")

    before = transitive_local_source_manifest(
        (entry,), root=tmp_path, package_prefixes=("demo",)
    )
    assert set(before) == {
        "demo/__init__.py",
        "demo/dependency.py",
        "demo/entry.py",
    }
    dependency.write_text("VALUE = 2\n")
    after = transitive_local_source_manifest(
        (entry,), root=tmp_path, package_prefixes=("demo",)
    )
    assert before["demo/entry.py"] == after["demo/entry.py"]
    assert before["demo/dependency.py"] != after["demo/dependency.py"]


def test_spin_production_manifest_contains_group_theory_dependencies() -> None:
    from projects.clebsch_tree_nqs.workflows import spin
    from projects.clebsch_tree_nqs.workflows.common import source_manifest

    manifest = source_manifest(spin.WORKFLOW)
    for required in (
        "projects/clebsch_tree_nqs/workflows/common.py",
        "projects/clebsch_tree_nqs/workflows/spin.py",
        "su2_nqs/frontier/artifacts.py",
        "su2_nqs/frontier/provenance.py",
        "su2_nqs/su2/clebsch_gordan.py",
        "su2_nqs/su2/fusion_tree.py",
        "su2_nqs/su2/quantum_numbers.py",
        "su2_nqs/operators/sz_heisenberg.py",
    ):
        assert required in manifest
