"""Verify the corrected retry deployment, A100 runtime and same frozen parent."""
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
    raise ValueError("External corrected CT12 deployment seal differs")
for name, digest in json.loads(manifest.read_text()).items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise ValueError("Sealed source/config/evidence changed: " + name)
for name, version in json.loads((CONFIG / "runtime_packages.json").read_text()).items():
    if importlib.metadata.version(name) != version:
        raise ValueError("Pinned A100 package differs: " + name)
plan = json.loads((CONFIG / "j0125.json").read_text())
for name, digest in plan["parent_seals"].items():
    if hashlib.sha256((Path(plan["parent_directory"]) / name).read_bytes()).hexdigest() != digest:
        raise ValueError("Frozen global37 parent changed: " + name)
print("CT12 retry source/runtime/parent and zero-fit failure receipts verified.", flush=True)
