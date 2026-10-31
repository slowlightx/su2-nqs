#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-ct12-width-20261021
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$PY" - <<'PY'
import hashlib,json,pathlib
data=json.loads(pathlib.Path('projects/clebsch_tree_nqs/configs/remote/ct12_width_20261021/source_manifest.json').read_text())
for name,digest in data.items():
    if hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()!=digest:
        raise ValueError('immutable source/config changed: '+name)
print('verified immutable source/config:',len(data),flush=True)
PY
if [[ "${NQS_PHASE:?tests/pilot/replay/train}" == tests ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    exec "$PY" -u -m pytest -q tests/test_fusion_paths.py tests/test_coupled_basis.py tests/test_recoupling.py tests/test_heisenberg.py tests/test_models.py tests/test_triangle_n144_continue.py
fi
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_continue --config "${NQS_CONFIG:?frozen config}" --phase "$NQS_PHASE"
