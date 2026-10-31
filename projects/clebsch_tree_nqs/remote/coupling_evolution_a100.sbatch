#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/coupling-%j.out
#SBATCH --error=logs/coupling-%j.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export PYTHONPATH="$PWD"
task_python=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
task_phase="${1:?freeze, train, or audit}"
task_config="${2:?configuration path}"
case "$task_phase" in
  freeze)
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
      --config "$task_config" --until-step 0 --require-endpoint
    ;;
  train)
    task_resume=()
    if [[ "${3:-}" == resume ]]; then task_resume=(--resume); fi
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
      --config "$task_config" --session-seconds 20700 --require-endpoint "${task_resume[@]}"
    ;;
  audit)
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality \
      audit-energy --config "$task_config" --steps "${3:?frozen step}" \
      --stream-index "${SLURM_ARRAY_TASK_ID:?independent audit stream}"
    ;;
  *) exit 2 ;;
esac
