#!/bin/bash
#SBATCH --partition=kim
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:20:00
#SBATCH --signal=B:USR1@120
set -euo pipefail
cd /home/yx789/nqs-triangular-arity-rank8-20261016
export JAX_ENABLE_X64=1
export JAX_PLATFORM_NAME=gpu
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
CONFIG=projects/clebsch_tree_nqs/configs/remote/triangle_arity_20261016/comparison.json
exec "$PYTHON" -u -m projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train --config "$CONFIG" --arm "$TRIANGLE_ARM" --session-seconds 3900
