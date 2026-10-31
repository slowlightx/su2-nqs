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
    spin_sampled_observable_probe as probe,
)
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.route_policy import FROZEN_PRODUCTION_MODULE_PREFIXES


CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/sampled/spin_sampled_observable_probe.json"
)


def _spec() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_probe_config_registers_square_and_triangular_full_spin_modes() -> None:
    spec = _spec()
    probe._validate_spec(spec)
    entries = {entry["geometry"]: entry for entry in spec["inputs"]}
    assert tuple(mode["name"] for mode in entries["square"]["modes"]) == (
        "q0",
        "neel",
        "stripe_x",
        "stripe_y",
    )
    assert tuple(mode["name"] for mode in entries["triangular"]["modes"]) == (
        "q0",
        "k",
        "m1",
        "m2",
        "m3",
    )
    broken = deepcopy(spec)
    broken["inputs"][0]["modes"][0]["name"] = "not_q0"
    with pytest.raises(ValueError, match="registered q0/order channels"):
        probe._validate_spec(broken)


def test_probe_source_closure_reuses_runtime_and_full_spin_kernel() -> None:
    manifest = probe._source_manifest()
    for required in (
        "projects/clebsch_tree_nqs/workflows/spin_sampled_observable_probe.py",
        "projects/clebsch_tree_nqs/workflows/spin_sampled_runtime.py",
        "su2_nqs/vmc/spin_observables.py",
        "su2_nqs/vmc/physical_exchange.py",
    ):
        assert required in manifest
    forbidden_paths = {
        f"{prefix.replace('.', '/')}.py"
        for prefix in FROZEN_PRODUCTION_MODULE_PREFIXES
    }
    assert forbidden_paths.isdisjoint(manifest)


def test_official_runtime_inputs_and_checkpoints_validate_fail_closed() -> None:
    for entry in _spec()["inputs"]:
        payload, case, result_path = probe._load_runtime_result(entry)
        assert payload["status"] == "complete"
        assert case["n_sites"] == 36
        for trial in case["trials"]:
            checkpoint, checkpoint_path = probe._load_runtime_checkpoint(
                result_path, payload, case, trial
            )
            assert checkpoint["status"] == "complete"
            assert checkpoint_path.name == trial["checkpoint"]["path"]
            assert checkpoint["checkpoint_sha256"] == trial["checkpoint"][
                "checkpoint_sha256"
            ]


def test_mode_specs_use_standard_N_inverse_normalization() -> None:
    for entry in _spec()["inputs"]:
        names, specs = probe._mode_specs(entry, rows=6, columns=6)
        assert names[0] == "q0"
        assert all(item.normalization == 36.0 for item in specs)
        assert all(item.normalization_kind == "structure_factor" for item in specs)


def test_chain_and_block_mcse_summary_is_finite() -> None:
    time = np.arange(9, dtype=np.float64)[:, None]
    chains = np.arange(16, dtype=np.float64)[None, :]
    values = 0.03 * time + 0.002 * chains + 1.0e-5j * (time - chains)
    summary = probe._summarize_mode(values, block_length=2)
    for key in (
        "mean_real",
        "iid_ess_mcse",
        "chain_mean_mcse",
        "blocked_chain_mcse",
        "tau_int",
        "ess",
    ):
        assert np.isfinite(summary[key])
    assert summary["block_count_per_chain"] == 4
    assert summary["ess"] <= values.size


def test_probe_checkpoint_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    identity = {
        "config_sha256": "a",
        "source_manifest_sha256": "b",
        "runtime_identity_sha256": "c",
        "case_id": "case",
        "arm_id": "arm",
        "backbone": "su2",
        "initial_params_sha256": "d",
        "upstream_trial_payload_sha256": "e",
        "upstream_checkpoint_sha256": "f",
    }
    written = probe._write_probe_checkpoint(
        path,
        identity=identity,
        status="running",
        states=np.asarray(((0, 1), (1, 0)), dtype=np.int32),
        amplitudes=np.asarray((1.0, -1.0), dtype=np.complex128),
        key=np.asarray((1, 2), dtype=np.uint32),
        completed_sweeps=1,
        diagnostics=[],
        local_values=np.ones((2, 2, 1), dtype=np.complex128),
        state_hashes=["x", "y"],
        cold_observable_seconds=0.1,
        warm_observable_seconds=[0.01],
        sampler_seconds=[0.02],
        resume_count=0,
    )
    loaded = probe._load_probe_checkpoint(path, identity)
    assert loaded is not None
    assert loaded["checkpoint_sha256"] == written["checkpoint_sha256"]
    tampered = json.loads(path.read_text())
    tampered["completed_sweeps"] = 2
    path.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="seal failed"):
        probe._load_probe_checkpoint(path, identity)


def test_cli_registers_reusable_probe_command() -> None:
    source = (ROOT / "projects/clebsch_tree_nqs/run.py").read_text()
    assert '"sampled-observable-probe"' in source
    assert "run_spin_sampled_observable_probe" in source
