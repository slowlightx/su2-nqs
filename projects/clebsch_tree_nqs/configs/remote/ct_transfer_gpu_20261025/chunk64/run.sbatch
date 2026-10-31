#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-ct-transfer-gpu-audit-chunk64-20261025
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset XLA_FLAGS
exec /home/yx789/nqs-vit-quality/.conda-netket/bin/python -u \
    projects/clebsch_tree_nqs/configs/remote/ct_transfer_gpu_20261025/run_campaign.py execute --variant chunk64
