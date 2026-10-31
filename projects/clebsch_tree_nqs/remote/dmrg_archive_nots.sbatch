#!/bin/bash
#SBATCH --job-name=nqs-dmrg-archive
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dmrg-archive-%j.out
#SBATCH --error=logs/dmrg-archive-%j.err

set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$PWD/.venv-tenpy/bin/python" -u -m projects.clebsch_tree_nqs.workflows.dmrg_archive \
  --source-root /scratch/yx51/nqs_reference_dmrg \
  --archive-root /storage/hpc/work/an9/yx51/nqs_reference_dmrg \
  --stage "${1:?Pass the existing completed or safely stopped stage}" \
  --receipt "/storage/hpc/work/an9/yx51/nqs_reference_dmrg/archive_receipts/job-${SLURM_JOB_ID}.json"
