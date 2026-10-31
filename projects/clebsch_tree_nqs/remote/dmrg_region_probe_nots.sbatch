#!/bin/bash
# Independent fixed-state measurement: never modifies an optimizer checkpoint.
#SBATCH --job-name=nqs-dmrg-region
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dmrg-region-%j.out
#SBATCH --error=logs/dmrg-region-%j.err

set -euo pipefail
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export PYTHONPATH="$PWD"
if [ "$#" -eq 0 ]; then
  echo 'Pass one or more sealed region-probe configs' >&2
  exit 2
fi
for task_config in "$@"; do
  "$PWD/.venv-tenpy/bin/python" -u -m projects.clebsch_tree_nqs.workflows.dmrg_region_probe --config "$task_config"
done
