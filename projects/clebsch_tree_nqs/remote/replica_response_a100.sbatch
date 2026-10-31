#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/replica-%j.out
#SBATCH --error=logs/replica-%j.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD"
exec /home/yx789/nqs-vit-quality/.conda-netket/bin/python -u -m \
  projects.clebsch_tree_nqs.workflows.spin_replica_response --config "${1:?config required}"
