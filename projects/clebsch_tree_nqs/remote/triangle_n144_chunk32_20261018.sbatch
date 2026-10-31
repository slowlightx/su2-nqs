#!/usr/bin/env bash
# Independent chunk32 execution gate, caller requests one A100 /30min.
set -euo pipefail
cd /home/yx789/nqs-triangle-n144-native-20261018
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=True
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_CONFIG="projects/clebsch_tree_nqs/configs/remote/triangle_n144_native_20261018/${NQS_N144_CASE:?j0 or j0125}_chunk32_gate.json"
if [[ "${NQS_N144_COLLECT:-0}" == 1 ]]; then
    export JAX_PLATFORMS=cpu
    export JAX_PLATFORM_NAME=cpu
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_native_gate \
        --config "$NQS_CONFIG" --collect-job "${NQS_N144_JOB:?completed GPU job}"
fi
export JAX_PLATFORMS=cuda
export JAX_PLATFORM_NAME=gpu
"$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_native_gate \
    --config "$NQS_CONFIG"
# Independent Python replay adds no H calls and commits no updates.
exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_native_gate \
    --config "$NQS_CONFIG" --replay-gate
