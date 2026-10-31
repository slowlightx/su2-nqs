from __future__ import annotations

import json
from pathlib import Path

from projects.clebsch_tree_nqs.workflows.spin_symmetry_carrier_four_arm import (
    TASK,
    _geometry_only_singlet_screen,
    _generic_source_bank,
)
from su2_nqs.benchmarks.clebsch_tree_exact import physical_permutation_index_table
from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.su2.sz_basis import enumerate_sz_configs


ROOT = Path(__file__).resolve().parents[1]


def test_generic_source_bank_is_nested_shell_matched_and_deterministic() -> None:
    group = triangular_hnf_gamma_a1_group(((4, 0), (0, 4)))
    basis = tuple(enumerate_sz_configs(16, 8))
    actions = FixedSzPairActions(basis)
    screen = _geometry_only_singlet_screen(
        basis, physical_permutation_index_table(basis, group)
    )
    first = _generic_source_bank(
        seed=19,
        supercell=((4, 0), (0, 4)),
        group=group,
        screen_state=screen,
        screen_actions=actions,
    )
    second = _generic_source_bank(
        seed=19,
        supercell=((4, 0), (0, 4)),
        group=group,
        screen_state=screen,
        screen_actions=actions,
    )
    assert first == second
    assert tuple(row["shell_pair"] for row in first) == (
        "11", "12", "22", "11", "12", "22"
    )
    assert len({row["orbit_sha256"] for row in first}) == 6


def test_four_arm_config_and_sealed_artifact_contract() -> None:
    config = json.loads(
        (ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_symmetry_carrier_four_arm.json").read_text()
    )
    assert config["task"] == TASK
    path = ROOT / "projects/clebsch_tree_nqs/results/spin_symmetry_carrier_four_arm_v1/results.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    assert payload["status"] == "complete"
    assert payload["backbone_contract"]["same_cached_backbone_in_every_arm"]
    assert not payload["backbone_contract"]["generic_sources_see_hamiltonian_or_energy"]
    assert payload["summary"]["all_primary_arms_exact_singlets"]
    assert payload["summary"]["robust_comparison_count"] == 9
    assert payload["summary"]["maximum_generic_gamma_a1_relative_leakage"] < 1.0e-10
    assert payload["summary"]["minimum_generic_rank_outside_physical_span"] > 0
