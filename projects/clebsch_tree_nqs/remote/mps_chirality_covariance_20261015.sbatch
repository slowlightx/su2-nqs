#!/bin/bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:29:00
set -euo pipefail
cd /home/yx789/nqs-mps-chirality-covariance-20261015-r1
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
exec /home/yx789/miniconda3/envs/tenpy/bin/python -u -m projects.clebsch_tree_nqs.workflows.mps_chirality_covariance_probe \
  --config "projects/clebsch_tree_nqs/configs/remote/dmrg_row_matrix_20261015/${DMRG_ROW_CASE}.json" \
  --output "/home/yx789/nqs-wavefunctions/dmrg/readouts/dmrg_row_matrix_20261015/${DMRG_ROW_CASE}"
