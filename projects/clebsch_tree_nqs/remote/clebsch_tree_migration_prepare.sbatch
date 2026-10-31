#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=CT8-DAG-prepare
#SBATCH --output=logs/ct-dag-prepare-%j.out
#SBATCH --error=logs/ct-dag-prepare-%j.err
set -euo pipefail
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD"
exec .conda-netket/bin/python -u -m projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_migration \
  --config "${NQS_CT_MIGRATION_CONFIG:?sealed CPU migration configuration}" --submit
