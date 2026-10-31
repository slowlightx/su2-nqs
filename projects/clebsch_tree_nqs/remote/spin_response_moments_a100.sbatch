#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/spin-moments-%j.out
#SBATCH --error=logs/spin-moments-%j.err
set -euo pipefail
export PYTHONPATH="$PWD"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
/home/yx789/nqs-vit-quality/.conda-netket/bin/python -u \
  -m projects.clebsch_tree_nqs.workflows.spin_response_moment_readout --config "${1:?config required}"
