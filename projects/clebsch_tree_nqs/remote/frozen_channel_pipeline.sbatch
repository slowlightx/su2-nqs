#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=00:30:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/frozen-pipeline-%A_%a.out
#SBATCH --error=logs/frozen-pipeline-%A_%a.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD"
cases=(tri_j01_750 tri_j01_1000 tri_j0175_750 tri_j0175_1000)
config_set="${1:-pipeline}"
if [[ "${2:-}" == ordered_side ]]; then
  cases=(tri_j005_250 tri_j005_500)
fi
case "$config_set" in pipeline|pipeline_r2) ;; *) echo 'Unknown frozen plan set' >&2; exit 2 ;; esac
exec /home/yx789/nqs-vit-quality/.conda-netket/bin/python -u -m \
  projects.clebsch_tree_nqs.workflows.spin_schmidt_covariance \
  --pipeline-config "projects/clebsch_tree_nqs/configs/remote/frozen_channels_20261016/$config_set/${cases[$SLURM_ARRAY_TASK_ID]}.json"
