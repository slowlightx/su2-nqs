#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-ct6-optimizer-20261019
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
exec /home/yx789/nqs-vit-quality/.conda-netket/bin/python -u -m \
    projects.clebsch_tree_nqs.workflows.triangle_optimizer_comparison \
    --plan projects/clebsch_tree_nqs/configs/remote/optimizer_comparison_20261019/plan.json \
    --plan-sha256 "${NQS_PLAN_SHA256:?registered plan}" \
    --arity "${NQS_ARITY:?native tree}" \
    --optimizer "${NQS_OPTIMIZER:?recipe}" \
    --phase "${NQS_PHASE:?pilot/replay/train}"
