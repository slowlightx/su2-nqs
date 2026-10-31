#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-ct6-width-repair-20261020
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$PY" - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('projects/clebsch_tree_nqs/configs/remote/ct6_width_20261020/source_manifest.json')
manifest=json.loads(p.read_text())
for name,digest in manifest.items():
    if hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()!=digest:
        raise ValueError('immutable source changed: '+name)
print('immutable source manifest verified:',len(manifest),flush=True)
PY
if [[ "${NQS_PHASE:?test/pilot/replay/train}" == tests ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    exec "$PY" -m pytest -q tests/test_physical_sr_adam.py \
        tests/test_triangle_clebsch_continuation.py tests/test_triangle_clebsch_width_repair.py \
        --junitxml=logs/focused_tests.xml
fi
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_clebsch_continuation \
    --plan "${NQS_CONFIG:?immutable plan}" --plan-sha256 "${NQS_PLAN_SHA256:?plan seal}" \
    --phase "$NQS_PHASE"
