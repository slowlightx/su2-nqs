#!/usr/bin/env bash
#SBATCH --job-name=nqs-vit-series
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:2
#SBATCH --time=16:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/vit-series-%j.out
#SBATCH --error=logs/vit-series-%j.err
set -euo pipefail
export PYTHONPATH="$PWD"
export JAX_PLATFORMS=cpu
task_config="$(.conda-netket/bin/python -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_schedule \
  --plan "${NQS_PLAN:?}" --stage "${NQS_STAGE:?}" --plan-sha256 "${NQS_PLAN_SHA256:?}")"
export NQS_CONFIG="$task_config"
exec bash projects/clebsch_tree_nqs/remote/spatial_vit_recovery_a100.sbatch "$@"
