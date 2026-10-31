#!/usr/bin/env bash
#SBATCH --job-name=su2-native-statistics
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=00:59:00
#SBATCH --output=logs/su2-statistics-%j.out
#SBATCH --error=logs/su2-statistics-%j.err

set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD"
task_python="$PWD/.conda-netket/bin/python"
task_config="${1:?registered reduced-space training configuration}"
"$task_python" -c 'import jax; print(jax.devices()); assert jax.default_backend()=="gpu" and jax.device_count()==1'
# Focused mathematical tests were run before submission. This allocation is
# bounded by 1 GPU-hour, with a 3300-second process cap and retained checkpoints.
exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_reduced_vit_train --config "$task_config"
