#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-ct6-gcnn-quality-20261021
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$PY" - <<'PY'
import json,hashlib,pathlib
m=json.loads(pathlib.Path('projects/clebsch_tree_nqs/configs/remote/ct6_gcnn_quality_20261021/source_manifest.json').read_text())
for p,h in m.items():
    if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h:raise ValueError('frozen source/config changed:'+p)
print('source/config manifest verified',len(m),flush=True)
PY
if [[ "${NQS_PHASE:?phase}" == tests ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    exec "$PY" -m pytest -q tests/test_parameter_block_sr.py tests/test_physical_sr_adam.py tests/test_triangle_clebsch_continuation.py
fi
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_clebsch_continuation --config "${NQS_CONFIG:?config}" --phase "$NQS_PHASE" --plan-sha256 "${NQS_PLAN_SHA256:?seal}"
