from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.workflows import hubbard, spin
from projects.clebsch_tree_nqs.workflows.common import (
    source_manifest,
    validate_complete_payload,
)
from su2_nqs.frontier.route_policy import FROZEN_PRODUCTION_MODULE_PREFIXES
from su2_nqs.su2.hubbard_cg_irrep import DOUBLON, EMPTY


PROJECT = ROOT / "projects/clebsch_tree_nqs"
PRODUCTION_SOURCES = (
    PROJECT / "run.py",
    PROJECT / "workflows/common.py",
    PROJECT / "workflows/spin.py",
    PROJECT / "workflows/hubbard.py",
)


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return tuple(names)


def _load_result(name: str) -> dict[str, object]:
    return json.loads((PROJECT / "results" / name / "results.json").read_text())


def test_production_sources_exclude_historical_projects_and_frozen_modules() -> None:
    for path in PRODUCTION_SOURCES:
        imports = _imports(path)
        assert not any(
            name.startswith("projects.")
            and not name.startswith("projects.clebsch_tree_nqs")
            for name in imports
        )
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in imports
            for prefix in FROZEN_PRODUCTION_MODULE_PREFIXES
        )
        source = path.read_text()
        assert "eigsh(" not in source
        assert "eigh(" not in source


def test_one_cli_and_one_paper_notebook_boundary() -> None:
    assert (PROJECT / "run.py").is_file()
    assert not list(PROJECT.glob("run_*.py"))
    assert not list(PROJECT.glob("scripts/*.py"))
    source = (PROJECT / "build_notebook.py").read_text()
    exploration_source = (PROJECT / "build_exploration_notebook.py").read_text()
    assert "projects/clebsch_tree_nqs" in source
    assert "symmetry_ladder_nqs" not in source
    assert "multicut_fusion" not in source
    assert "figures/spin_article" not in source
    assert 'FIGURES = PROJECT / "figures/paper"' in source
    assert "spin_e2_resonance_audit" not in source
    assert 'FIGURES = PROJECT / "figures/exploration"' in exploration_source
    assert "spin_e2_resonance_audit" in exploration_source


def test_configs_declare_domain_and_scalar_anchor_policy() -> None:
    for path in sorted((PROJECT / "configs").glob("*.json")):
        config = json.loads(path.read_text())
        expected = "hubbard" if path.name.startswith("hubbard") else "spin"
        assert config["schema_version"] == 2
        assert config["domain"] == expected
        for case in config["cases"]:
            assert isinstance(case["exact_energy_anchor"], (int, float))
            assert case["anchor_provenance"]


def test_prl_local_pareto_is_preregistered_and_held_out() -> None:
    config = json.loads(
        (PROJECT / "configs/spin_prl_local_pareto.json").read_text()
    )
    assert config["optimization"]["seeds"] == list(range(21, 29))
    assert set(config["optimization"]["seeds"]).isdisjoint({11, 12, 13})
    translation_arms = [
        arm for arm in config["arms"] if arm["projection"] == "translation"
    ]
    for backbone in ("u1", "su2"):
        widths = {
            arm["model"]["feature_dim"]
            for arm in translation_arms
            if arm["backbone"] == backbone
        }
        assert widths == {2, 3, 4, 5, 6}
    assert config["analysis_contract"]["primary_sector"] == "translation"
    assert "matched-time" in config["analysis_contract"]["promotion_gate"]
    couplings = {
        (case["geometry"], case["j2_over_j1"])
        for case in config["cases"]
    }
    assert ("square", 0.55) in couplings
    assert ("triangular", 0.175) in couplings


def test_u1_long_control_covers_su2_time_range_without_protocol_drift() -> None:
    base = json.loads(
        (PROJECT / "configs/spin_prl_local_pareto.json").read_text()
    )
    long = json.loads(
        (PROJECT / "configs/spin_prl_local_u1_long.json").read_text()
    )
    assert long["optimization"]["seeds"] == base["optimization"]["seeds"]
    assert long["optimization"]["learning_rate"] == base["optimization"][
        "learning_rate"
    ]
    assert long["optimization"]["steps"] == 4 * base["optimization"]["steps"]
    assert len(long["arms"]) == 1
    arm = long["arms"][0]
    assert (arm["backbone"], arm["projection"], arm["model"]["feature_dim"]) == (
        "u1",
        "translation",
        6,
    )
    assert {
        (case["geometry"], case["j2_over_j1"]) for case in long["cases"]
    } == {
        (case["geometry"], case["j2_over_j1"]) for case in base["cases"]
    }


def test_su2_long_completes_the_censored_time_horizon_without_protocol_drift() -> None:
    base = json.loads(
        (PROJECT / "configs/spin_prl_local_pareto.json").read_text()
    )
    long = json.loads(
        (PROJECT / "configs/spin_prl_local_su2_long.json").read_text()
    )
    assert long["optimization"]["seeds"] == base["optimization"]["seeds"]
    assert long["optimization"]["learning_rate"] == base["optimization"][
        "learning_rate"
    ]
    assert long["optimization"]["steps"] == 2 * base["optimization"]["steps"]
    assert len(long["arms"]) == 1
    arm = long["arms"][0]
    assert (arm["backbone"], arm["projection"], arm["model"]["feature_dim"]) == (
        "su2",
        "translation",
        6,
    )
    assert {
        (case["geometry"], case["j2_over_j1"]) for case in long["cases"]
    } == {
        (case["geometry"], case["j2_over_j1"]) for case in base["cases"]
    }
    assert "after inspecting" in long["claim_boundary"]


def test_spin_manifest_contains_shared_harness_and_group_theory() -> None:
    manifest = source_manifest(spin.WORKFLOW)
    for required in (
        "projects/clebsch_tree_nqs/workflows/common.py",
        "projects/clebsch_tree_nqs/workflows/spin.py",
        "su2_nqs/frontier/artifacts.py",
        "su2_nqs/su2/clebsch_gordan.py",
        "su2_nqs/su2/fusion_tree.py",
        "su2_nqs/su2/quantum_numbers.py",
    ):
        assert required in manifest


def test_two_spin_singlet_and_triangular_momentum_conventions() -> None:
    basis = ((0, 1), (1, 0))
    state = np.asarray([1.0, -1.0]) / np.sqrt(2.0)
    correlations = spin._spin_correlations(state, basis)
    np.testing.assert_allclose(
        correlations,
        np.asarray([[0.75, -0.75], [-0.75, 0.75]]),
        atol=1.0e-14,
    )
    structure = spin._structure_factor_grid(correlations, 1, 2)
    assert abs(structure[0, 0]) < 1.0e-14
    assert abs(structure[0, 1] - 1.5) < 1.0e-14

    rows = columns = 3
    phase = np.asarray(
        [
            2.0 * np.pi * (row + column) / 3.0
            for row in range(rows)
            for column in range(columns)
        ]
    )
    classical = np.cos(phase[:, None] - phase[None, :])
    triangular = spin._structure_factor_grid(classical, rows, columns)
    peaks = {
        tuple(index)
        for index in np.argwhere(np.isclose(triangular, np.max(triangular)))
    }
    assert peaks == {(1, 1), (2, 2)}
    assert set(spin._groups("triangular", 3, 6)) == {"none", "translation"}


def test_hubbard_centered_charge_structure_retains_static_cdw() -> None:
    observables = hubbard._density_structure_factors(
        np.asarray([1.0]), ((DOUBLON, EMPTY),), rows=1, columns=2
    )
    assert abs(observables["charge_pi_along_chain"] - 2.0) < 1.0e-14


def test_current_spin_results_are_sealed_and_respect_claim_boundary() -> None:
    payload = _load_result("spin_validation_v1")
    validate_complete_payload(payload, spin.WORKFLOW)
    cases = {case["case_id"]: case for case in payload["cases"]}
    triangular = cases["triangular_4x4_j2_0p125_m0"]
    assert not triangular["k_120_commensurate"]
    assert all(
        trial["metrics"]["physical_observables"][
            "order_120_K_structure_factor_max"
        ]
        is None
        for trial in triangular["trials"]
    )
    square = cases["square_4x4_j2_0p5_m0"]
    errors = {
        arm_id: float(
            np.median(
                [
                    trial["metrics"]["energy_error_per_site"]
                    for trial in square["trials"]
                    if trial["arm_id"] == arm_id
                ]
            )
        )
        for arm_id in {trial["arm_id"] for trial in square["trials"]}
    }
    assert errors["su2_gamma_a1"] < errors["u1_gamma_a1"]
    assert all(
        audit["passed"]
        for case in payload["cases"]
        for audit in case["paired_initialization_audit"]
    )


def test_true_full_and_parameter_bracket_calibrations_are_explicit() -> None:
    full = _load_result("spin_true_full_v1")
    validate_complete_payload(full, spin.WORKFLOW)
    case = full["cases"][0]
    trial = case["trials"][0]
    assert case["support"] == "true_full"
    assert case["basis_size"] == 2 ** case["n_sites"]
    assert len(
        [
            weight
            for weight in trial["metrics"]["m2_sector_weights"].values()
            if weight > 1.0e-8
        ]
    ) > 2

    config = json.loads(
        (PROJECT / "configs/spin_square_parameter_brackets.json").read_text()
    )
    assert all("true_full" not in arm["arm_id"] for arm in config["arms"])


def test_current_hubbard_result_is_signed_projected_and_sealed() -> None:
    payload = _load_result("hubbard_validation_v1")
    validate_complete_payload(payload, hubbard.WORKFLOW)
    assert not payload["route_boundary"]["charge_only_control_available"]
    cases = {case["case_id"]: case for case in payload["cases"]}
    ladder = cases["hubbard_ladder_2x4_pbc_U4"]
    assert ladder["group_orders"]["gamma_a1"] == 16
    assert ladder["space_group_metadata"]["retained_point_quotient_order"] == 2
    for trial in ladder["trials"]:
        assert np.isfinite(trial["metrics"]["double_occupancy_per_site"])
        assert "charge_structure_factor_grid" in trial["metrics"][
            "physical_observables"
        ]
    su2 = next(
        trial for trial in ladder["trials"] if trial["arm_id"] == "su2_gamma_a1"
    )
    assert abs(su2["metrics"]["spin_squared_expectation"]) < 1.0e-10


def test_vieijra_nys_reference_records_cutoff_conversion() -> None:
    reference = json.loads(
        (PROJECT / "references/vieijra_nys_prb104_045123.json").read_text()
    )
    rows = reference["table_I_8x8_total_energies"]
    assert len([row for row in rows if row["method"] == "ClebschTree"]) == 5
    facts = reference["protocol_facts"]["systematic_scan"]
    assert facts["saturation_physical_j_max"] == 4
    assert facts["repository_doubled_q_max"] == 8
