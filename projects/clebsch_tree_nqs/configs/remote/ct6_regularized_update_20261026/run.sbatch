#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-ct6-regularized-update-20261026
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$PY" - <<'PY'
import hashlib,json,pathlib,os
from datetime import datetime,timedelta,timezone
folder=pathlib.Path('projects/clebsch_tree_nqs/configs/remote/ct6_regularized_update_20261026')
manifest=json.loads((folder/'source_manifest.json').read_text())
for path,digest in manifest.items():
    if hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()!=digest:
        raise ValueError('frozen source/config changed:'+path)
campaign=json.loads((folder/'campaign.json').read_text())
assert sum(sum(row['phase_minutes'].values()) for row in campaign['cases'])/60 == campaign['new_requested_A100_hours']
assert campaign['aggregate_reserved_A100_hours'] == campaign['previous_conservatively_reserved_A100_hours']+campaign['new_requested_A100_hours']
if campaign['new_requested_A100_hours']!=48 or campaign['aggregate_reserved_A100_hours']>campaign['CT6_total_budget_A100_hours']:
    raise RuntimeError('fixed-parent continuation allocation exceeds frozen CT6 budget')
if datetime.now(timezone.utc)+timedelta(minutes=int(os.environ['NQS_ALLOC_MINUTES']))>datetime.fromisoformat(campaign['deadline_utc']):
    raise RuntimeError('allocation cannot finish before the campaign deadline')
print('source/config manifest verified',len(manifest),flush=True)
PY
if [[ "${NQS_PHASE:?phase}" == tests ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    export XLA_FLAGS=--xla_cpu_max_isa=AVX2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$PY" -m pytest -q tests/test_triangle_ct_campaign.py tests/test_triangle_clebsch_continuation.py tests/test_triangular_clebsch_expansion.py tests/test_physical_step_calibration.py tests/test_physical_sr_adam.py tests/test_parameter_block_sr.py tests/test_ct6_fixed_parent_direction_oracle.py tests/test_ct6_direction_continuation.py tests/test_ct6_evidence_guarded_continuation.py
fi
if [[ "$NQS_PHASE" == parents ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$PY" - <<'PY'
import json,pathlib
from projects.clebsch_tree_nqs.workflows.triangle_clebsch_continuation import validate_plan,load_parent
folder=pathlib.Path('projects/clebsch_tree_nqs/configs/remote/ct6_regularized_update_20261026')
for path in sorted(folder.glob('j*.json')):
    plan=json.loads(path.read_text());validate_plan(plan)
    state,lineage=load_parent(plan)
    print(json.dumps(dict(stage=plan['stage'],parent=lineage)),flush=True)
PY
fi
unset XLA_FLAGS
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_clebsch_continuation --plan "${NQS_CONFIG:?config}" --phase "$NQS_PHASE" --plan-sha256 "${NQS_PLAN_SHA256:?seal}"
