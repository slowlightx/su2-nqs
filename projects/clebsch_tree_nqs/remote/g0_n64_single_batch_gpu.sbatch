#!/usr/bin/env bash
#SBATCH --job-name=nqs-g0a
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=00:09:00
#SBATCH --output=logs/g0a-%j.out
#SBATCH --error=logs/g0a-%j.err

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

export OUTPUT_ROOT="projects/clebsch_tree_nqs/results/remote_g0_n64_single_batch_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_ROOT"

python -c 'import json, jax, sys; row={"backend":jax.default_backend(),"devices":[str(x) for x in jax.devices()],"x64":bool(jax.config.jax_enable_x64)}; print(json.dumps(row)); sys.exit(0 if row["backend"]=="gpu" else 2)'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
  --format=csv,noheader,nounits --loop=2 > "$OUTPUT_ROOT/gpu_telemetry.csv" &
TASK_MONITOR_PID=$!
trap 'kill "$TASK_MONITOR_PID" 2>/dev/null || true' EXIT

python projects/clebsch_tree_nqs/run.py sampled-runtime \
  --config projects/clebsch_tree_nqs/configs/remote/g0_square_n64_single_batch_gpu.json \
  --output "$OUTPUT_ROOT/square_n64"

kill "$TASK_MONITOR_PID" 2>/dev/null || true
wait "$TASK_MONITOR_PID" 2>/dev/null || true
trap - EXIT

python projects/clebsch_tree_nqs/remote/analyze_g0.py \
  --cpu-root projects/clebsch_tree_nqs/results/remote_g0_n64_single_batch_cpu_v1 \
  --gpu-root "$OUTPUT_ROOT" \
  --cases square_n64 \
  --output "$OUTPUT_ROOT/promotion_decision.json"
