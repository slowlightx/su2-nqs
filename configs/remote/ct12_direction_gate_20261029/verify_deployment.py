"""Check the external deployment seal and pinned A100 runtime before execution."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).resolve().parent
manifest = CONFIG / "source_manifest.json"
if hashlib.sha256(manifest.read_bytes()).hexdigest() != os.environ.get("NQS_CT12_GATE_MANIFEST_SHA256"):
    raise ValueError("External CT12 direction-gate deployment SHA is absent or different")
for name, digest in json.loads(manifest.read_text()).items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise ValueError("Sealed source/config changed: " + name)
for name, version in json.loads((CONFIG / "runtime_packages.json").read_text()).items():
    if importlib.metadata.version(name) != version:
        raise ValueError("Pinned A100 package differs: " + name)
plan = json.loads((CONFIG / "j0125.json").read_text())
for name, digest in plan["parent_seals"].items():
    path = Path(plan["parent_directory"]) / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Frozen global37 parent changed: " + name)
print("CT12 source, runtime and full parent seals verified; no existing jobs changed.", flush=True)
