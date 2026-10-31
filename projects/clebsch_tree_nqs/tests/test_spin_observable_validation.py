from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.clebsch_tree_nqs.workflows import (
    spin_observable_validation as validation,
)
from su2_nqs.frontier.artifacts import canonical_sha256, embedded_trial_manifest
from su2_nqs.frontier.route_policy import FROZEN_PRODUCTION_MODULE_PREFIXES
from su2_nqs.vmc.spin_observables import prepare_spin_structure_factor


CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/validation/spin_observable_validation.json"
)


def _spec() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_default_normalization_is_standard_structure_factor_and_m2_is_explicit() -> None:
    weights = np.ones(8, dtype=np.complex128)
    standard = prepare_spin_structure_factor(weights)
    assert standard.normalization == 8.0
    assert standard.normalization_kind == "structure_factor"
    m2 = prepare_spin_structure_factor(weights, normalization=64.0)
    assert m2.normalization == 64.0
    assert m2.normalization_kind == "m2"


def test_observable_validation_config_fails_closed() -> None:
    spec = _spec()
    validation._validate_spec(spec)
    wrong_normalization = deepcopy(spec)
    wrong_normalization["normalization"]["default"] = "structure_factor_1_over_N_squared"
    with pytest.raises(ValueError, match="normalization contract"):
        validation._validate_spec(wrong_normalization)
    wrong_size = deepcopy(spec)
    wrong_size["complex_state"]["n_sites"] = 6
    with pytest.raises(ValueError, match="fixed at N=8"):
        validation._validate_spec(wrong_size)


def test_observable_validation_source_closure_is_current_and_route_clean() -> None:
    manifest = validation._source_manifest()
    for required in (
        "projects/clebsch_tree_nqs/workflows/spin_observable_validation.py",
        "su2_nqs/vmc/spin_observables.py",
        "su2_nqs/operators/sz_graph_sparse.py",
        "su2_nqs/models/spin_clebsch_tree.py",
        "su2_nqs/su2/clebsch_gordan.py",
    ):
        assert required in manifest
    forbidden_paths = {
        f"{prefix.replace('.', '/')}.py"
        for prefix in FROZEN_PRODUCTION_MODULE_PREFIXES
    }
    assert forbidden_paths.isdisjoint(manifest)


def test_exact_small_observable_metrics_pass_strict_gates() -> None:
    trial = validation._run_trial(_spec())
    metrics = trial["metrics"]
    assert trial["all_gates_passed"]
    assert metrics["n8_complex_state_action_max_abs_discrepancy"] < 1.0e-12
    assert metrics["n8_born_expectation_abs_discrepancy"] < 1.0e-12
    assert metrics["singlet_q0_action_max_abs"] < 1.0e-12
    assert metrics["structure_factor_sum_target"] == 3.0
    assert metrics["structure_factor_sum_abs_error"] < 1.0e-12
    assert metrics["m2_sum_target"] == 0.75
    assert metrics["m2_sum_abs_error"] < 1.0e-12
    assert trial["normalization_contract"]["default_name"] == "structure_factor"
    assert trial["normalization_contract"]["m2_name"] == "m2"


def test_sealed_observable_artifact_is_reusable(tmp_path: Path) -> None:
    output = tmp_path / "observable_validation"
    payload = validation.run_spin_observable_validation(CONFIG, output)
    assert payload["status"] == "complete"
    assert payload["config_sha256"] == canonical_sha256(_spec())
    embedded = embedded_trial_manifest(payload)
    assert payload["trial_manifest"] == embedded
    assert payload["trial_manifest_sha256"] == canonical_sha256(embedded)
    reused = validation.run_spin_observable_validation(CONFIG, output)
    assert reused["trial_manifest_sha256"] == payload["trial_manifest_sha256"]
