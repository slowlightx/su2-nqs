#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --constraint=v100
#SBATCH --gres=gpu:tesla_v100s-pcie-32gb:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/swap-probe-%j.out
#SBATCH --error=logs/swap-probe-%j.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"
config="${1:-projects/clebsch_tree_nqs/configs/remote/schmidt_covariance_20261015/swap_probe_v100.json}"
exec /home/yx789/nqs-vit-quality/.conda-netket/bin/python -u -m \
  projects.clebsch_tree_nqs.workflows.spin_schmidt_covariance \
  --config "$config" \
  --swap-probe-only
