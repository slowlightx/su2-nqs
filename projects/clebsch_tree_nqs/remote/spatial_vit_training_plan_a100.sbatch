#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=18:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/vit-opt-%j.out
#SBATCH --error=logs/vit-opt-%j.err
set -euo pipefail
export JAX_ENABLE_X64=True
export PYTHONPATH="$PWD"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MPLCONFIGDIR="$PWD/.matplotlib"
task_config="$(.conda-netket/bin/python -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_training_plan \
  --plan "${NQS_PLAN:?registered plan}" --plan-sha256 "${NQS_PLAN_SHA256:?frozen plan sha}")"
export NQS_CONFIG="$task_config"
exec bash projects/clebsch_tree_nqs/remote/spatial_vit_recovery_a100.sbatch "${1:?production,energy-stream,aggregate}"
