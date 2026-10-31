#!/usr/bin/env bash
#SBATCH --job-name=nqs-pair-energy
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/adaptive-pair-energy-%j.out
#SBATCH --error=logs/adaptive-pair-energy-%j.err

set -euo pipefail
PROJECT_ROOT="${NQS_REMOTE_ROOT:?set NQS_REMOTE_ROOT to the synced repository root}"
cd "$PROJECT_ROOT"
mkdir -p logs projects/clebsch_tree_nqs/results
TASK_CONDA_EXE="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
source "$("$TASK_CONDA_EXE" info --base)/etc/profile.d/conda.sh"
conda activate nqs
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
OUTPUT="projects/clebsch_tree_nqs/results/remote_adaptive_pair_energy_${SLURM_JOB_ID}.json"
python projects/clebsch_tree_nqs/remote/validate_n36_adaptive_pair_energy.py \
  --output "$OUTPUT"
