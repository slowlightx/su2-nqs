#!/usr/bin/env bash
#SBATCH --job-name=su2-revision-check
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/reduced-revision-%j.out
#SBATCH --error=logs/reduced-revision-%j.err

set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD"
task_python=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
task_config="${1:?registered serial revision configuration}"
"$task_python" -c 'import jax; print(jax.devices()); assert jax.default_backend()=="gpu" and jax.device_count()==1'
"$task_python" -m pytest tests/test_reduced_vap_orbit_projector.py -q
exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_reduced_vit_revision --config "$task_config"
