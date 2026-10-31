#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=00:30:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/schmidt-cov-%A_%a.out
#SBATCH --error=logs/schmidt-cov-%A_%a.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD"
if [[ -n "${NQS_SWAP_DIAGNOSTIC:-}" ]]; then
  /home/yx789/nqs-vit-quality/.conda-netket/bin/python -c \
    'import json,sys; r=json.load(open(sys.argv[1])); assert r["guard_passed"] is True, "prerequisite SWAP runtime replay failed"' \
    "$NQS_SWAP_DIAGNOSTIC"
fi
cases=(tri_j0_750 tri_j0_1000 tri_j0125_100 tri_j0125_200)
config_dir="${1:-projects/clebsch_tree_nqs/configs/remote/schmidt_covariance_20261015}"
config="$config_dir/${cases[$SLURM_ARRAY_TASK_ID]}.json"
exec /home/yx789/nqs-vit-quality/.conda-netket/bin/python -u -m \
  projects.clebsch_tree_nqs.workflows.spin_schmidt_covariance --config "$config"
