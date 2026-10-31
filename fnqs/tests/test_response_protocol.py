from pathlib import Path
import json

import numpy as np
import pytest

from fnqs.artifacts import source_hashes, write_json
from fnqs.scripts.diagnose_response import (case_grid, completed_case, declare, digest,
    runtime_sources, validate)

PARENT = Path(__file__).resolve().parents[1]/"results"/"size_capacity_controls_v8"


def test_declared_cases_stay_in_development_and_cover_all_controls():
    parent = json.loads((PARENT/"protocol.json").read_text())
    cases = case_grid(parent)
    assert len(cases) == 135
    assert len({(c["group"], c["seed"], c["n"], c["h"]) for c in cases}) == 135
    assert all(c["h"] in parent["development_fields"] and c["h"] not in parent["reserved_confirmation_fields"] for c in cases)
    for c in cases:
        assert c["n"] in parent["arms"][c["arm"]]["sizes"]
    parent["reserved_confirmation_fields"].append(.952)
    with pytest.raises(ValueError, match="leakage"):
        case_grid(parent)


def test_declaration_and_archive_bind_original_checkpoints_and_new_code(tmp_path):
    output = tmp_path/"diagnosis"
    p = declare(PARENT, output, samples=128, chunk=32, bootstrap=10)
    assert p["diagnostic_born_draws"] == 17280
    assert p["hamiltonian_state_and_tangent_actions"] == 270
    assert p["directional_operator_actions"] == 135
    assert p["optimizer_updates"] == 0
    assert not p["confirmation_measured"]
    assert not p["size_ablations"]["trained"]
    assert validate(output)[0] == p
    assert source_hashes() == json.loads((PARENT/"protocol.json").read_text())["source_sha256"]
    for name, sha in runtime_sources().items():
        assert digest(tmp_path/"runtime_v9"/"fnqs"/name) == sha
    with pytest.raises(FileExistsError):
        declare(PARENT, output)
    p["source_sha256"]["diagnostics/response.py"] = "tampered"
    write_json(output/"protocol.json", p)
    with pytest.raises(ValueError, match="runtime changed"):
        validate(output)


def test_budget_cannot_be_changed_after_sealing(tmp_path):
    output = tmp_path/"diagnosis"
    p = declare(PARENT, output, samples=128, chunk=32, bootstrap=10)
    p["samples_per_point"] = 256
    write_json(output/"protocol.json", p)
    with pytest.raises(ValueError, match="sealed protocol"):
        validate(output)


def test_resume_rejects_changed_records_and_case_identity(tmp_path):
    case = {"index": 0, "n": 16}
    write_json(tmp_path/"protocol.json", {"test": True})
    assert completed_case(tmp_path, case) is None
    path = tmp_path/"cases"/"case_000"
    path.parent.mkdir()
    np.savez(path.with_suffix(".npz"), score=np.ones(3))
    row = {"case": case, "protocol_sha256": digest(tmp_path/"protocol.json"),
           "records_sha256": digest(path.with_suffix(".npz"))}
    write_json(path.with_suffix(".json"), row)
    assert completed_case(tmp_path, case) == row
    np.savez(path.with_suffix(".npz"), score=np.zeros(3))
    with pytest.raises(ValueError, match="invalid saved"):
        completed_case(tmp_path, case)


@pytest.mark.parametrize("samples,chunk,bootstrap", [(127, 1, 10), (129, 32, 10), (128, 0, 10), (128, 32, 1)])
def test_invalid_budget_is_rejected_before_declaration(tmp_path, samples, chunk, bootstrap):
    with pytest.raises(ValueError, match="budget"):
        declare(PARENT, tmp_path/"diagnosis", samples, chunk, bootstrap)
    assert not (tmp_path/"diagnosis").exists()
