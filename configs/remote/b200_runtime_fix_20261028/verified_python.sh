#!/usr/bin/env bash
# Explicit opt-in only. This file never changes existing task launchers.
set -euo pipefail

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf '%s\n' \
    'Usage: NQS_B200_ACCEPTANCE_SHA256=<externally approved certificate SHA256> bash verified_python.sh -- <Python arguments>' \
    'Example: ... bash verified_python.sh -- -m projects.clebsch_tree_nqs.workflows.YOUR_WORKFLOW ...' \
    'Requires an allocated B200 and an independently accepted GPU certificate. No implicit production release.' >&2
  exit 2
fi
if [[ "$1" != "--" || $# -lt 2 ]]; then
  printf '%s\n' 'Expected -- followed by arguments to the verified Python interpreter.' >&2
  exit 2
fi
shift
if [[ ! "${NQS_B200_ACCEPTANCE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'Refusing: NQS_B200_ACCEPTANCE_SHA256 must be the independently approved certificate SHA256.' >&2
  exit 2
fi
if [[ -n "${LD_LIBRARY_PATH:-}" || -n "${LD_PRELOAD:-}" ]]; then
  printf '%s\n' 'Refusing: library path/preload overrides are not allowed for this verified pip-CUDA runtime.' >&2
  exit 2
fi

readonly NQS_FIXED_PYTHON='/home/yx789/nqs-b200-runtime-20261028/venv/bin/python'
export PYTHONNOUSERSITE=1
exec "$NQS_FIXED_PYTHON" - "$@" <<'PY'
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

root = Path('/home/yx789/nqs-b200-validation-20261028').resolve()
python = '/home/yx789/nqs-b200-runtime-20261028/venv/bin/python'
certificate_path = root/'gpu/gpu_check_result.json'
plan_path = root/'configs/remote/b200_runtime_fix_20261028/plan.json'

def refuse(reason):
    raise SystemExit('B200 verified runtime refused: ' + reason)

if Path(sys.executable).absolute() != Path(python):
    refuse('unexpected interpreter')
if not certificate_path.is_file() or not plan_path.is_file():
    refuse('accepted certificate or sealed plan is absent')
actual_sha = hashlib.sha256(certificate_path.read_bytes()).hexdigest()
if actual_sha != os.environ.get('NQS_B200_ACCEPTANCE_SHA256'):
    refuse('certificate differs from the external acceptance SHA256')
certificate = json.loads(certificate_path.read_text())
if certificate.get('status') != 'bounded_software_acceptance_complete' or certificate.get('passed') is not True:
    refuse('GPU acceptance is failed or incomplete')

# Import the frozen acceptance code, not a same-named module from the caller.
sys.path.insert(0, str(root))
runtime = importlib.import_module('projects.clebsch_tree_nqs.workflows.accelerator_runtime_validation')
entry = root/'projects/clebsch_tree_nqs/workflows/accelerator_runtime_validation.py'
if Path(runtime.__file__).resolve() != entry.resolve():
    refuse('runtime guard was imported from outside the sealed workspace')
from su2_nqs.frontier.artifacts import canonical_sha256
from su2_nqs.frontier.provenance import transitive_local_source_manifest
plan = json.loads(plan_path.read_text())
runtime.validate_plan(plan)
if certificate.get('plan_sha256') != canonical_sha256(plan):
    refuse('sealed acceptance plan changed')
manifest = transitive_local_source_manifest([entry], root=root,
    package_prefixes=('su2_nqs', 'projects.clebsch_tree_nqs'))
if certificate.get('source_manifest') != manifest:
    refuse('acceptance source or a transitive local dependency changed')
saved_stack = certificate.get('stack', {})
if saved_stack.get('distributions') != plan['approved_distributions']:
    refuse('certificate did not validate the approved distributions')
for name, expected in plan['approved_libraries'].items():
    if saved_stack.get('mapped_libraries', {}).get(name, {}).get('sha256') != expected:
        refuse('certificate native-library SHA differs: ' + name)

# This deliberately initializes JAX on the allocated B200 and checks the
# libraries it actually loaded, including native versions and file hashes.
# It does not mute warnings or disable JAX CUDA compatibility checks.
stack, _, _ = runtime.loaded_cuda_stack(plan)
print(json.dumps(dict(status='verified_B200_runtime_startup_passed',
    certificate_sha256=actual_sha, plan_sha256=canonical_sha256(plan),
    mapped_libraries=stack['mapped_libraries']), sort_keys=True), file=sys.stderr, flush=True)
os.execv(python, [python, *sys.argv[1:]])
PY
