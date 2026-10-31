#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-ct6-wide-production-v2-20261023
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
unset XLA_FLAGS
exec /home/yx789/nqs-vit-quality/.conda-netket/bin/python -u \
    /home/yx789/nqs-ct6-midpoint-audit-20261023/audit_driver.py \
    --plan "${NQS_AUDIT_PLAN:?frozen audit plan}" --plan-sha256 "${NQS_AUDIT_SHA256:?plan identity}"
