#!/bin/bash
#SBATCH --job-name=nqs-dmrg-ref
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=23:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/dmrg-%j.out
#SBATCH --error=logs/dmrg-%j.err

set -euo pipefail
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export PYTHONPATH="$PWD"
task_python="$PWD/.venv-tenpy/bin/python"
task_config="${1:?Pass an explicitly registered stage config to sbatch}"
# exec makes the Python guard receive Slurm's batch-shell USR1 signal.
# Stop/rollback checks occur around local updates; a native BLAS/Lanczos
# call must finish before Python can observe a pending signal.
exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.dmrg_continuation \
  --config "$task_config" --require-complete
