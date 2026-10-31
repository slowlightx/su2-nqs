#!/usr/bin/env bash
#SBATCH --job-name=nqs-vit-physics
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/vit-physics-%j.out
#SBATCH --error=logs/vit-physics-%j.err

set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export PYTHONPATH="$PWD"
task_python="/home/yx789/nqs-vit-quality/.conda-netket/bin/python"
task_config="projects/clebsch_tree_nqs/configs/remote/spin_spatial_vit_5600_physics.json"
"$task_python" -c 'import jax; print(jax.devices()); assert jax.default_backend()=="gpu" and jax.device_count()==1'
exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality \
  cached-physics --config "$task_config"
