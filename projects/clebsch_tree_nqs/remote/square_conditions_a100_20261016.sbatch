#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=02:30:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/ct-condition-%j.out
#SBATCH --error=logs/ct-condition-%j.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export PYTHONPATH="$PWD"
if [[ "${NQS_CT_DIAGNOSE_DIRECTIONS:-0}" == "1" ]]; then
  exec /home/yx789/nqs-vit-quality/.conda-netket/bin/python -u \
    -m projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_radius_compare \
    --plan "${NQS_CT_CONDITION_PLAN:?registered direction plan}" \
    --plan-sha256 "${NQS_CT_CONDITION_PLAN_SHA256:?registered direction plan hash}" \
    --diagnose-directions
else
  exec /home/yx789/nqs-vit-quality/.conda-netket/bin/python -u \
    -m projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_radius_compare \
    --plan "${NQS_CT_CONDITION_PLAN:-projects/clebsch_tree_nqs/configs/remote/square_conditions_20261016/plan.json}" \
    --plan-sha256 "${NQS_CT_CONDITION_PLAN_SHA256:?registered condition plan hash}" \
    --arm "${NQS_CT_CONDITION_ARM:?one frozen optimizer recipe}"
fi
