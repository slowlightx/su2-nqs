#!/bin/bash
# Future stages only. The existing running job keeps its captured batch script
# and frozen registration; HOME compatibility links preserve old resume paths.
#SBATCH --job-name=nqs-dmrg-scratch
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=23:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/dmrg-%j.out
#SBATCH --error=logs/dmrg-%j.err

set -euo pipefail
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export PYTHONPATH="$PWD"
task_python="$PWD/.venv-tenpy/bin/python"
task_config="${1:?Pass a fresh registered config with scratch input/output}"
"$task_python" -m projects.clebsch_tree_nqs.workflows.dmrg_storage check \
  --config "$task_config" --scratch-root /scratch/yx51/nqs_reference_dmrg
exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.dmrg_continuation \
  --config "$task_config" --require-complete
