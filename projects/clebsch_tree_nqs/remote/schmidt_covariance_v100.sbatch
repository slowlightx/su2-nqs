#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --constraint=v100
#SBATCH --gres=gpu:tesla_v100s-pcie-32gb:1
#SBATCH --time=00:30:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/schmidt-cov-%A_%a.out
#SBATCH --error=logs/schmidt-cov-%A_%a.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"
cases=(tri_j0_750 tri_j0_1000 tri_j0125_100 tri_j0125_200)
config="projects/clebsch_tree_nqs/configs/remote/schmidt_covariance_20261015/${cases[$SLURM_ARRAY_TASK_ID]}.json"
exec /home/yx789/nqs-vit-quality/.conda-netket/bin/python -u -m \
  projects.clebsch_tree_nqs.workflows.spin_schmidt_covariance --config "$config"
