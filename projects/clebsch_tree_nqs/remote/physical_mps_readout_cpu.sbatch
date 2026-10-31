#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=logs/mps-readout-%j.out
#SBATCH --error=logs/mps-readout-%j.err
set -euo pipefail
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
/home/yx789/miniconda3/envs/tenpy/bin/python -u -m "${1:?workflow required}" \
  --config "${2:?config required}" --output "${3:?fresh output required}"
