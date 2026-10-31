#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-ct6-release-repair-20261026
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$PY" - <<'PY'
import hashlib,json,pathlib,os
from datetime import datetime,timedelta,timezone
base=pathlib.Path('projects/clebsch_tree_nqs/configs/remote')
repair=base/'ct6_release_repair_20261026'
manifest=json.loads((base/'ct6_regularized_update_20261026/source_manifest.json').read_text())
workflow='projects/clebsch_tree_nqs/workflows/triangle_clebsch_continuation.py'
patched='af053deb351a16c5030e01a02cf22d41a9c27a74c15144f41ad2b8137a716440'
for path,digest in manifest.items():
    expected=patched if path==workflow else digest
    if hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()!=expected:
        raise ValueError('frozen source/config changed:'+path)
campaign=json.loads((repair/'campaign.json').read_text())
total=campaign['previous_conservatively_reserved_A100_hours']+campaign['previous_v8_actual_A100_hours']+campaign['replacement_train_A100_hours']
if abs(total-campaign['aggregate_reserved_A100_hours'])>1e-9 or total>campaign['CT6_total_budget_A100_hours']:
    raise RuntimeError('repair exceeds existing campaign budget')
if datetime.now(timezone.utc)+timedelta(minutes=int(os.environ['NQS_ALLOC_MINUTES']))>datetime.fromisoformat(campaign['deadline_utc']):
    raise RuntimeError('allocation would exceed original campaign deadline')
print('original source/config manifest checked; validation-only patch declared',flush=True)
PY
if [[ "${NQS_PHASE:?phase}" == tests ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    export XLA_FLAGS=--xla_cpu_max_isa=AVX2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    "$PY" -m pytest -q tests/test_ct6_gate_replay_repair.py tests/test_ct6_validation_repair_launcher.py tests/test_ct6_direction_continuation.py tests/test_ct6_evidence_guarded_continuation.py tests/test_ct6_fixed_parent_direction_oracle.py
    exec "$PY" - <<'PY'
import json,pathlib
from projects.clebsch_tree_nqs.workflows.triangle_clebsch_continuation import checked_direction_release
from projects.clebsch_tree_nqs.workflows.triangle_ct6_validation_repair import assert_validation_only_source_change
base=pathlib.Path('projects/clebsch_tree_nqs/configs/remote')
for coupling in ('j0','j0125'):
    plan=json.loads((base/'ct6_regularized_update_20261026'/f'{coupling}.json').read_text())
    output=pathlib.Path(plan['output']); registration=json.loads((output/'registration.json').read_text())
    pilot=checked_direction_release(output,{'completed_steps':0},plan,registration['sha256'])
    print(coupling,'actual saved pilot/replay/raw-direction release:',pilot['status'],flush=True)
old=pathlib.Path('/home/yx789/nqs-ct6-regularized-update-20261026/projects/clebsch_tree_nqs/workflows/triangle_clebsch_continuation.py')
new=pathlib.Path('projects/clebsch_tree_nqs/workflows/triangle_clebsch_continuation.py')
assert_validation_only_source_change(old.read_text(),new.read_text())
print('actual frozen source AST numerical-path identity passed',flush=True)
PY
fi
unset XLA_FLAGS
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
REPAIR="projects/clebsch_tree_nqs/configs/remote/ct6_release_repair_20261026/${NQS_COUPLING:?coupling}_repair.json"
REPAIR_SHA=$("$PY" -c 'import json,sys; from su2_nqs.frontier.artifacts import canonical_sha256; print(canonical_sha256(json.load(open(sys.argv[1]))))' "$REPAIR")
exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_ct6_validation_repair --plan "${NQS_CONFIG:?config}" --plan-sha256 "${NQS_PLAN_SHA256:?seal}" --repair-manifest "$REPAIR" --repair-manifest-sha256 "$REPAIR_SHA"
