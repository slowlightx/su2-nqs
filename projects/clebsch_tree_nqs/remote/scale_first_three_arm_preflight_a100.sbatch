#!/usr/bin/env bash
#SBATCH --job-name=nqs-scale3-pre
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/scale3-pre-%j.out
#SBATCH --error=logs/scale3-pre-%j.err

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
export OUTPUT_ROOT="projects/clebsch_tree_nqs/results/remote_scale_first_preflight_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_ROOT"

python -c 'import json, jax, sys; row={"backend":jax.default_backend(),"devices":[str(x) for x in jax.devices()],"x64":bool(jax.config.jax_enable_x64)}; print(json.dumps(row)); sys.exit(0 if row["backend"]=="gpu" else 2)'
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits --loop=1 > "$OUTPUT_ROOT/gpu_telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT

# Each static shape is deliberately executed in a fresh process.  Jobs
# 276262 and 280431 showed that same-process multi-shape XLA executables can
# drive cumulative demand beyond 80 GB even after jax.clear_caches(),
# invalidating both the memory audit and the throughput gate.
CONFIG="projects/clebsch_tree_nqs/configs/remote/spin_scale_first_gpu_preflight.json"
for CASE_ID in triangular_6x6_j2_0.125 square_6x6_j2_0.5; do
  for ARM_ID in \
    exact_su2_balanced_schur_spatial_vit \
    fixed_sz_shared_autoregressive_spatial_vit \
    symmetry_unaware_autoregressive_spatial_vit; do
    for BATCH_SIZE in 4 8 16; do
      SLICE_OUTPUT="$OUTPUT_ROOT/slices/$CASE_ID/$ARM_ID/batch$BATCH_SIZE"
      python projects/clebsch_tree_nqs/run.py \
        spin-scale-first-accelerator-slice \
        --config "$CONFIG" \
        --case-id "$CASE_ID" \
        --arm-id "$ARM_ID" \
        --batch-size "$BATCH_SIZE" \
        --output "$SLICE_OUTPUT"
    done
  done
done

python projects/clebsch_tree_nqs/run.py \
  spin-scale-first-accelerator-merge \
  --config "$CONFIG" \
  --input-root "$OUTPUT_ROOT/slices" \
  --output "$OUTPUT_ROOT"

sstat --jobs="${SLURM_JOB_ID}.batch" \
  --format=JobID,AveRSS,MaxRSS,AveVMSize,MaxVMSize -P || true
