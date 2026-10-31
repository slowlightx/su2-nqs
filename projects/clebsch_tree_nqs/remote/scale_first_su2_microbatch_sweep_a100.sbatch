#!/usr/bin/env bash
#SBATCH --job-name=nqs-su2-mbatch
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/su2-mbatch-%j.out
#SBATCH --error=logs/su2-mbatch-%j.err

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
export OUTPUT_ROOT="projects/clebsch_tree_nqs/results/remote_scale_first_microbatch_sweep_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_ROOT"

python -c 'import json, jax, sys; row={"backend":jax.default_backend(),"devices":[str(x) for x in jax.devices()],"x64":bool(jax.config.jax_enable_x64)}; print(json.dumps(row)); sys.exit(0 if row["backend"]=="gpu" else 2)'
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits --loop=1 > "$OUTPUT_ROOT/gpu_telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT

CONFIG="projects/clebsch_tree_nqs/configs/remote/spin_scale_first_gpu_preflight.json"
for MICROBATCH_SIZE in 4 8 16; do
  python projects/clebsch_tree_nqs/run.py \
    spin-scale-first-accelerator-slice \
    --config "$CONFIG" \
    --case-id triangular_6x6_j2_0.125 \
    --arm-id exact_su2_balanced_schur_spatial_vit \
    --batch-size 16 \
    --amplitude-tile-size 256 \
    --device-microbatch-size "$MICROBATCH_SIZE" \
    --output "$OUTPUT_ROOT/microbatch$MICROBATCH_SIZE"
done

sstat --jobs="${SLURM_JOB_ID}.batch" \
  --format=JobID,AveRSS,MaxRSS,AveVMSize,MaxVMSize -P || true
