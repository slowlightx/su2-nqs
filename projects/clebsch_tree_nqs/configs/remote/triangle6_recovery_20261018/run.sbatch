#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-triangle6-recovery-20261018
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export NETKET_SHARDING=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
exec "$NQS_PY" -u projects/clebsch_tree_nqs/configs/remote/triangle6_recovery_20261018/recover.py \
  --plan projects/clebsch_tree_nqs/configs/remote/triangle6_recovery_20261018/plan.json \
  --plan-sha256 "${NQS_PLAN_SHA256:?frozen plan}" \
  --arm "${NQS_ARM:?registered arm}" --stage "${NQS_STAGE:?registered stage}"
