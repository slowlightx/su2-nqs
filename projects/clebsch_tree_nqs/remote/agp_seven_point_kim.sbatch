#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:10:00
#SBATCH --output=logs/agp-seven-%A_%a.out
#SBATCH --error=logs/agp-seven-%A_%a.err
set -euo pipefail
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
task_mode="${1:?preflight or case}"
task_config="${2:?config path}"
task_python="/home/yx789/nqs-vit-quality/.conda-netket/bin/python"
"$task_python" -c 'import jax; print(jax.devices()); assert jax.default_backend()=="gpu" and jax.device_count()==1'
if [[ "$task_mode" == "case" ]]; then
  exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_agp_seven_point case \
    --config "$task_config" --index "${SLURM_ARRAY_TASK_ID:?registered 0..6 index}"
else
  exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_agp_seven_point preflight \
    --config "$task_config"
fi
