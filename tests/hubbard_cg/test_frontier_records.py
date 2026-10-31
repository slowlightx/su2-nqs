import hashlib
import json
from pathlib import Path

import pytest

from su2_nqs.frontier.records import (
    FrontierRecord,
    WorkloadIdentity,
    assert_comparable_workloads,
    import_json_source,
    workload_from_mapping,
)


def _workload(**changes):
    values = dict(
        workload_id="chain_L6_U4_half_s0_raw",
        hamiltonian_hash="a" * 64,
        geometry="chain",
        boundary="obc",
        n_sites=6,
        n_electrons=6,
        target_q=0,
        target_m2=0,
        couplings=(("U", 4.0), ("t", 1.0)),
        evaluation_stage="raw_vmc",
    )
    values.update(changes)
    return WorkloadIdentity(**values)


def test_workload_fingerprint_is_stable_and_mapping_sorts_couplings():
    first = _workload()
    second = workload_from_mapping(
        {
            **{key: value for key, value in first.__dict__.items() if key != "couplings"},
            "couplings": {"U": 4.0, "t": 1.0},
        }
    )
    assert first == second
    assert first.fingerprint == second.fingerprint


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hamiltonian_hash", "b" * 64),
        ("boundary", "pbc"),
        ("n_sites", 8),
        ("n_electrons", 4),
        ("target_q", 2),
        ("couplings", (("U", 8.0), ("t", 1.0))),
        ("evaluation_stage", "lanczos_1"),
    ],
)
def test_comparison_rejects_every_physical_or_evaluation_mismatch(field, value):
    with pytest.raises(ValueError, match=field):
        assert_comparable_workloads(_workload(), _workload(**{field: value}))


def test_energy_without_exact_source_locator_is_rejected():
    with pytest.raises(ValueError, match="source locator"):
        FrontierRecord(
            schema_version="frontier-v3",
            stage="F2",
            execution_status="COMPLETE",
            algebra_status="PASS",
            benchmark_status="INCONCLUSIVE",
            accuracy_status="REFERENCE_VERIFIED",
            record_status="NOT_CLAIMED",
            workload=_workload(),
            method="HubbardCG",
            energy_per_site=-0.5,
        )


def test_content_addressed_json_import_is_idempotent_and_byte_exact(tmp_path: Path):
    source = tmp_path / "source.json"
    raw = json.dumps({"energy": -1.25}, indent=2).encode() + b"\n"
    source.write_bytes(raw)
    destination = tmp_path / "registry"
    first = import_json_source(source, destination)
    second = import_json_source(source, destination)
    assert first == second
    assert first.path.read_bytes() == raw
    assert first.sha256 == hashlib.sha256(raw).hexdigest()
