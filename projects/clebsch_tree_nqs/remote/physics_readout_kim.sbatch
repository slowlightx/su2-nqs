#!/usr/bin/env bash
# Reusable fixed-state readout, physically separate from running optimizers.
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/physics-%j.out
#SBATCH --error=logs/physics-%j.err
set -euo pipefail
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
task_mode="${1:?nqs or dmrg}"
shift
for task_config in "$@"; do
  case "$task_mode" in
    nqs)
      .conda-netket/bin/python -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality \
        cached-plan --config "$task_config"
      ;;
    dmrg)
      .venv-tenpy/bin/python -u -m projects.clebsch_tree_nqs.workflows.dmrg_region_probe \
        --config "$task_config"
      ;;
    *) exit 2 ;;
  esac
done
