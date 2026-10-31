#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-ct6-sixday-20261022
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$PY" - <<'PY'
import hashlib,json,pathlib
from datetime import datetime,timedelta,timezone
manifest=json.loads(pathlib.Path('projects/clebsch_tree_nqs/configs/remote/ct6_sixday_20261022/source_manifest.json').read_text())
for p,h in manifest.items():
    if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h:
        raise ValueError('frozen source/config changed: '+p)
campaign=json.loads(pathlib.Path('projects/clebsch_tree_nqs/configs/remote/ct6_sixday_20261022/campaign.json').read_text())
if datetime.now(timezone.utc)+timedelta(minutes=int(__import__('os').environ['NQS_ALLOC_MINUTES']))>datetime.fromisoformat(campaign['deadline_utc']):
    raise RuntimeError('not enough time to finish this allocation before the fixed deadline')
print('source/config manifest verified',len(manifest),flush=True)
PY
if [[ "${NQS_PHASE:?phase}" == tests ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    export XLA_FLAGS=--xla_cpu_max_isa=AVX2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$PY" -m pytest -q tests/test_triangle_ct_campaign.py tests/test_triangle_clebsch_continuation.py tests/test_physical_step_calibration.py tests/test_physical_sr_adam.py tests/test_parameter_block_sr.py
fi
if [[ "$NQS_PHASE" == advance ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    export XLA_FLAGS=--xla_cpu_max_isa=AVX2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_ct_campaign --campaign "$NQS_CAMPAIGN" --command advance --coupling "$NQS_COUPLING" --stage "$NQS_STAGE"
fi
unset XLA_FLAGS
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_clebsch_continuation --plan "${NQS_CONFIG:?config}" --phase "$NQS_PHASE" --plan-sha256 "${NQS_PLAN_SHA256:?seal}"
