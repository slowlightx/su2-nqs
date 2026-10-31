#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-square-transfer-20261017
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=True JAX_PLATFORM_NAME=gpu NETKET_SHARDING=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
exec /home/yx789/nqs-vit-quality/.conda-netket/bin/python -u -m projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_initialization_compare \
    --plan projects/clebsch_tree_nqs/configs/remote/square_transfer_20261017/plan.json \
    --arm "${NQS_ARM:?registered arm}" --stage "${NQS_STAGE:?registered stage}"
