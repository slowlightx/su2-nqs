#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=00:10:00
#SBATCH --array=0-1%1
#SBATCH --output=logs/frozen-distance-%A_%a.out
#SBATCH --error=logs/frozen-distance-%A_%a.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD"
case "${1:-late}" in
  late) cases=(tri_j0_1000 tri_j0125_200) ;;
  early) cases=(tri_j0_750 tri_j0125_100) ;;
  *) echo 'Expected late or early frozen quality pair' >&2; exit 2 ;;
esac
exec /home/yx789/nqs-vit-quality/.conda-netket/bin/python -u -m \
  projects.clebsch_tree_nqs.workflows.spin_frozen_distance \
  --config "projects/clebsch_tree_nqs/configs/remote/frozen_distance_20261015/${cases[$SLURM_ARRAY_TASK_ID]}.json"
