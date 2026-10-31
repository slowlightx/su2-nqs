#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ct-audit-%j.out
#SBATCH --error=logs/ct-audit-%j.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export PYTHONPATH="$PWD"
exec .conda-netket/bin/python -u -m projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_audit \
  --config "${NQS_CT_CONFIG:-projects/clebsch_tree_nqs/configs/remote/scale_capacity_20261014/vieijra_nys_square8_folded_audit20.json}"
