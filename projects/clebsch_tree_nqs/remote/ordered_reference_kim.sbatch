#!/usr/bin/env bash
#SBATCH --job-name=dmrg-j0-control
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --constraint=cpu
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=logs/ordered-reference-%j.out
#SBATCH --error=logs/ordered-reference-%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export PYTHONPATH="$PWD"
exec .venv-tenpy/bin/python -u -m projects.clebsch_tree_nqs.workflows.spin_tensor_network_reference \
  --config projects/clebsch_tree_nqs/configs/remote/scale_capacity_20261014/tri_j0_dmrg_ordered_reference.json --arm warm
