#!/usr/bin/env bash
set -euo pipefail
cd "/home/yx789/nqs-square-paper-20261020/${NQS_CASE:?case}"
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
exec /home/yx789/nqs-vit-quality/.conda-netket/bin/python -u -m \
    projects.clebsch_tree_nqs.workflows.square_clebsch_paper_continue \
    --plan "projects/clebsch_tree_nqs/configs/remote/square_paper_continue_20261020/${NQS_CASE:?case}_plan.json" \
    --phase "${NQS_PHASE:?phase}"
