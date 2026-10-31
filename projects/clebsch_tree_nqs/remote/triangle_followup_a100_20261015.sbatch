#!/bin/bash
#SBATCH --partition=kim
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --signal=B:USR1@120
set -euo pipefail
cd /home/yx789/nqs-triangular-clebsch-followup-20261015
export JAX_ENABLE_X64=1
export JAX_PLATFORM_NAME=gpu
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
CONFIG=projects/clebsch_tree_nqs/configs/remote/triangle_training_followup_20261015/${TRIANGLE_ARM}.json
if [[ ${TRIANGLE_STAGE} == pilot ]]; then
  exec "$PYTHON" -u -m projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train --config "$CONFIG" --session-seconds 660 --until-step 2
elif [[ ${TRIANGLE_STAGE} == continuation ]]; then
  exec "$PYTHON" -u -m projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train --config "$CONFIG" --session-seconds 9300
else
  echo 'Unknown bounded triangle stage' >&2
  exit 2
fi
