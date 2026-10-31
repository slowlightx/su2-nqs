#!/usr/bin/env bash
#SBATCH --job-name=nqs-g1-opvit
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/g1-opvit-%j.out
#SBATCH --error=logs/g1-opvit-%j.err

set -euo pipefail

PROJECT_ROOT="${NQS_REMOTE_ROOT:?set NQS_REMOTE_ROOT to the synced repository root}"
cd "$PROJECT_ROOT"
mkdir -p logs

TASK_CONDA_EXE="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
source "$("$TASK_CONDA_EXE" info --base)/etc/profile.d/conda.sh"
conda activate nqs

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OUTPUT_ROOT="projects/clebsch_tree_nqs/results/remote_g1_operator_vit_${SLURM_JOB_ID}"
TASK_CONFIG="${G1_CONFIG:-projects/clebsch_tree_nqs/configs/remote/spin_g1_operator_vit_n36_canary.json}"
mkdir -p "$OUTPUT_ROOT"

python -c 'import json, jax, sys; row={"backend":jax.default_backend(),"devices":[str(x) for x in jax.devices()],"x64":bool(jax.config.jax_enable_x64)}; print(json.dumps(row)); sys.exit(0 if row["backend"]=="gpu" else 2)'
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv,noheader,nounits --loop=30 > "$OUTPUT_ROOT/gpu_telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT

python projects/clebsch_tree_nqs/run.py spin-g1-operator-vit \
  --config "$TASK_CONFIG" \
  --output "$OUTPUT_ROOT"

sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveRSS,MaxRSS,AveVMSize,MaxVMSize -P || true
