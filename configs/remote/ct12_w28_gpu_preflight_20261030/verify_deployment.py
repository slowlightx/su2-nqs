"""Check exact private deployment files/runtime before any GPU work."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

root=Path.cwd();folder=root/'configs/remote/ct12_w28_gpu_preflight_20261030'
manifest=folder/'source_manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=os.environ.get('NQS_W28_PREFLIGHT_MANIFEST_SHA256'):
    raise SystemExit('preflight manifest identity differs')
for name,expected in json.loads(manifest.read_text()).items():
    digest=hashlib.sha256()
    with (root/name).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    if digest.hexdigest()!=expected:raise SystemExit('deployment SHA mismatch: '+name)
for package,version in json.loads((folder/'runtime_packages.json').read_text()).items():
    if importlib.metadata.version(package)!=version:raise SystemExit('runtime version mismatch: '+package)
plan=json.loads((folder/'plan.json').read_text())
if not plan['execution_authorized']:raise SystemExit('root has not released preflight execution')
print('True-w28 zero-update preflight deployment/runtime/inputs sealed.',flush=True)
