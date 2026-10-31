#!/usr/bin/env bash
# Independent full-H batch cost only; old sealed sources remain unchanged.
set -euo pipefail
cd /home/yx789/nqs-triangle-scale-batch-cost-20261017
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=True
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_CONFIG=projects/clebsch_tree_nqs/configs/triangle_scale_batch_cost_20261017.json
if [[ "${NQS_COST_COLLECT:-0}" == 1 ]]; then
    export JAX_PLATFORMS=cpu
    export JAX_PLATFORM_NAME=cpu
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_scale_batch_cost \
        --config "$NQS_CONFIG" --collect-job "${NQS_COST_JOB:?parent accounting job}"
fi
export JAX_PLATFORMS=cuda
export JAX_PLATFORM_NAME=gpu
exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_scale_batch_cost \
    --config "$NQS_CONFIG"
