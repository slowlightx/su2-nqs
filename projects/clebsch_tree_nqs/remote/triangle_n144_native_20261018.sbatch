#!/usr/bin/env bash
# Caller requests exactly one A100 and90min; no automatic production release.
set -euo pipefail
cd /home/yx789/nqs-triangle-n144-native-20261018
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=True
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_CONFIG="projects/clebsch_tree_nqs/configs/remote/triangle_n144_native_20261018/${NQS_N144_CASE:?j0 or j0125}_gate.json"
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
# Replay runs as a fresh Python process and does not call H or commit updates.
exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_native_gate \
    --config "$NQS_CONFIG" --replay-gate
