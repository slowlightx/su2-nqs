#!/bin/bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:29:00
#SBATCH --nodelist=kim-compute-05
set -euo pipefail
cd /home/yx789/nqs-mps-chirality-covariance-20261015-r2
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
exec /home/yx789/miniconda3/envs/tenpy/bin/python -u -m projects.clebsch_tree_nqs.workflows.mps_chirality_covariance_probe \
  --config "${1:-projects/clebsch_tree_nqs/configs/remote/dmrg_row_matrix_20261015/chi10240_cut18_r2.json}" \
  --output "${2:-/home/yx789/nqs-wavefunctions/dmrg/readouts/dmrg_row_matrix_20261015_r2/chi10240_cut18}"
