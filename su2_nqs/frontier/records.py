"""Fail-closed records for energy/accuracy/compute frontier comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping


EXECUTION_STATUSES = {"NOT_RUN", "RUNNING", "COMPLETE", "FAILED"}
ALGEBRA_STATUSES = {None, "PASS", "FAIL"}
BENCHMARK_STATUSES = {
    None,
    "CALIBRATED",
    "PROMISING_SCALING",
    "NO_GO",
    "INCONCLUSIVE",
}
ACCURACY_STATUSES = {
    "NOT_EVALUATED",
    "REFERENCE_VERIFIED",
    "VALIDATED_VARIATIONAL",
    "EXTRAPOLATED",
    "FAIL",
}
RECORD_STATUSES = {
    "NOT_CLAIMED",
    "REFERENCE_INCOMPLETE",
    "CANDIDATE",
    "CONFIRMED_IN_SCOPE",
    "REJECTED",
}


@dataclass(frozen=True)
class WorkloadIdentity:
    """Fields that must agree before two numerical rows are compared."""

    workload_id: str
    hamiltonian_hash: str
    geometry: str
    boundary: str
    n_sites: int
    n_electrons: int | None
    target_q: int
    target_m2: int
    couplings: tuple[tuple[str, float], ...]
    evaluation_stage: str = "raw_vmc"

    def __post_init__(self) -> None:
        if not self.workload_id or not self.hamiltonian_hash:
            raise ValueError("workload_id and hamiltonian_hash must be non-empty")
        if not self.geometry or not self.boundary or not self.evaluation_stage:
            raise ValueError("geometry, boundary and evaluation_stage must be non-empty")
        if isinstance(self.n_sites, bool) or not isinstance(self.n_sites, Integral) or self.n_sites < 1:
            raise ValueError("n_sites must be a positive integer")
        if self.n_electrons is not None and (
            isinstance(self.n_electrons, bool)
            or not isinstance(self.n_electrons, Integral)
            or not 0 <= self.n_electrons <= 2 * self.n_sites
        ):
            raise ValueError("n_electrons must be null or lie between 0 and 2*n_sites")
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (self.target_q, self.target_m2)):
            raise TypeError("target_q and target_m2 must be doubled integers")
        if self.target_q < 0 or abs(self.target_m2) > self.target_q or (self.target_q - self.target_m2) % 2:
            raise ValueError("target_m2 is not a projection of target_q")
        names = [name for name, _ in self.couplings]
        if len(names) != len(set(names)) or names != sorted(names):
            raise ValueError("couplings must have unique names in sorted order")
        if any(not name or not isinstance(value, Real) for name, value in self.couplings):
            raise ValueError("couplings must contain named real values")

    @property
    def fingerprint(self) -> str:
        """Return the deterministic identity digest."""

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def assert_comparable_workloads(
    first: WorkloadIdentity, second: WorkloadIdentity
) -> None:
    """Raise with every mismatched field instead of comparing unlike rows."""

    mismatches = [
        field
        for field in asdict(first)
        if getattr(first, field) != getattr(second, field)
    ]
    if mismatches:
        raise ValueError("workload identity mismatch: " + ", ".join(mismatches))


@dataclass(frozen=True)
class FrontierRecord:
    """One typed numerical row with independent status vocabularies."""

    schema_version: str
    stage: str
    execution_status: str
    algebra_status: str | None
    benchmark_status: str | None
    accuracy_status: str
    record_status: str
    workload: WorkloadIdentity
    method: str
    energy_per_site: float | None = None
    statistical_ci: tuple[float, float] | None = None
    source_locator: str | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "frontier-v3":
            raise ValueError("unsupported frontier schema version")
        if self.execution_status not in EXECUTION_STATUSES:
            raise ValueError("invalid execution_status")
        if self.algebra_status not in ALGEBRA_STATUSES:
            raise ValueError("invalid algebra_status")
        if self.benchmark_status not in BENCHMARK_STATUSES:
            raise ValueError("invalid benchmark_status")
        if self.accuracy_status not in ACCURACY_STATUSES:
            raise ValueError("invalid accuracy_status")
        if self.record_status not in RECORD_STATUSES:
            raise ValueError("invalid record_status")
        if not self.stage or not self.method:
            raise ValueError("stage and method must be non-empty")
        if self.energy_per_site is not None and (
            self.source_locator is None or self.source_sha256 is None
        ):
            raise ValueError("a numerical energy requires an exact source locator and hash")
        if (self.source_locator is None) != (self.source_sha256 is None):
            raise ValueError("source_locator and source_sha256 must be set together")
        if self.source_sha256 is not None and (
            len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            raise ValueError("source_sha256 must be a lowercase SHA256 digest")
        if self.statistical_ci is not None:
            lower, upper = self.statistical_ci
            if lower > upper:
                raise ValueError("statistical_ci must be ordered")


@dataclass(frozen=True)
class ImportedSource:
    """Content-addressed immutable JSON import metadata."""

    path: Path
    sha256: str
    byte_count: int


def import_json_source(source: str | Path, destination: str | Path) -> ImportedSource:
    """Copy one JSON file by content hash without rewriting an existing import."""

    source_path = Path(source)
    raw = source_path.read_bytes()
    json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    target = destination_path / f"{source_path.stem}-{digest[:16]}.json"
    if target.exists():
        if target.read_bytes() != raw:
            raise RuntimeError("content-addressed source path contains different bytes")
    else:
        with target.open("xb") as handle:
            handle.write(raw)
    return ImportedSource(path=target, sha256=digest, byte_count=len(raw))


def workload_from_mapping(payload: Mapping[str, object]) -> WorkloadIdentity:
    """Build a workload from JSON-compatible data with canonical couplings."""

    values = dict(payload)
    values["couplings"] = tuple(
        sorted((str(name), float(value)) for name, value in values["couplings"].items())
    )
    return WorkloadIdentity(**values)


__all__ = [
    "FrontierRecord",
    "ImportedSource",
    "WorkloadIdentity",
    "assert_comparable_workloads",
    "import_json_source",
    "workload_from_mapping",
]
