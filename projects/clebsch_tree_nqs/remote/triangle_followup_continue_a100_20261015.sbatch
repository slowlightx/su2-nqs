#!/bin/bash
#SBATCH --partition=kim
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --signal=B:USR1@120
set -euo pipefail
cd /home/yx789/nqs-triangular-clebsch-followup-20261015
export JAX_ENABLE_X64=1 JAX_PLATFORM_NAME=gpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
CONFIG=projects/clebsch_tree_nqs/configs/remote/triangle_training_followup_20261015/${TRIANGLE_ARM}.json
"$PYTHON" -u -m projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_followup_gate --config "$CONFIG"
exec "$PYTHON" -u -m projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train --config "$CONFIG" --session-seconds 8400
