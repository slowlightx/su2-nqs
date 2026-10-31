#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-ct12-sr-block-20261021
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$PY" - <<'PY'
import json,hashlib,pathlib
m=json.loads(pathlib.Path('projects/clebsch_tree_nqs/configs/remote/ct12_sr_block_20261021/source_manifest.json').read_text())
for p,h in m.items():
    if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h:raise ValueError('frozen source/config changed:'+p)
print('source/config manifest verified',len(m),flush=True)
PY
if [[ "${NQS_PHASE:?phase}" == tests ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    exec "$PY" -m pytest -q tests/test_parameter_block_sr.py tests/test_triangle_dictionary_vmc_independent.py tests/test_triangle_n144_continue.py tests/test_physical_sr_adam.py
fi
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
if [[ "$NQS_PHASE" == kernel ]]; then
    exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.parameter_block_sr_cost --output /home/yx789/nqs-ct12-sr-block-20261021/kernel_memory.json
fi
"$PY" - <<'PY'
import json,pathlib
m=json.loads(pathlib.Path('kernel_memory.json').read_text())
assert m['status']=='passed' and m['score_shape']==[256,2,7856880]
assert m['parameter_chunk_size']==8192 and m['peak_device_bytes']<70*1024**3
print('actual-size SR kernel-memory certificate passed',m['peak_device_bytes'],flush=True)
PY
exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_continue --config "${NQS_CONFIG:?config}" --phase "$NQS_PHASE"
