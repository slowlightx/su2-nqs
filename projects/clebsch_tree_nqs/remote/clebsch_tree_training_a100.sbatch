#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/ct-training-%j.out
#SBATCH --error=logs/ct-training-%j.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export PYTHONPATH="$PWD"
if [[ "${NQS_CT_RUN_TESTS:-0}" == "1" ]]; then
  .conda-netket/bin/python -m pytest tests/test_clebsch_tree_training.py tests/test_physical_sr_adam.py -q
fi
exec .conda-netket/bin/python -u -m projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_training \
  --config "${NQS_CT_CONFIG:?CT training protocol}" --until-step "${NQS_CT_UNTIL_STEP:-2}" \
  --session-seconds "${NQS_CT_SESSION_SECONDS:-2700}"
