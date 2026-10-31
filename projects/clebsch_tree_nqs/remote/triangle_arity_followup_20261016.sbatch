#!/bin/bash
#SBATCH --partition=kim
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --signal=B:USR1@120
set -euo pipefail
cd /home/yx789/nqs-triangular-arity-followup-20261016
export JAX_ENABLE_X64=1
export JAX_PLATFORM_NAME=gpu
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
WORKFLOW=projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train
CONFIG=projects/clebsch_tree_nqs/configs/remote/triangle_arity_followup_20261016/${TRIANGLE_CASE}.json
if [[ "$TRIANGLE_STAGE" == pilot ]]; then
    exec "$PYTHON" -u -m "$WORKFLOW" --config "$CONFIG" --until-step "$TRIANGLE_PILOT_STEP" --session-seconds 900
elif [[ "$TRIANGLE_STAGE" == continue ]]; then
    exec "$PYTHON" -u -m "$WORKFLOW" --config "$CONFIG" --require-pilot-step "$TRIANGLE_PILOT_STEP" --session-seconds 4800
else
    echo "Unregistered triangle stage" >&2
    exit 2
fi
