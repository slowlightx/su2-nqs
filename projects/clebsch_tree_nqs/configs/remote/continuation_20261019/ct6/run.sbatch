#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-triangle-continuation-20261019
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
exec /home/yx789/nqs-vit-quality/.conda-netket/bin/python -u -m \
    projects.clebsch_tree_nqs.workflows.triangle_clebsch_continuation \
    --plan "${NQS_CONFIG:?frozen native CT6 plan}" \
    --plan-sha256 "${NQS_PLAN_SHA256:?registered stage plan}" \
    --phase "${NQS_PHASE:?pilot/replay/train}"
