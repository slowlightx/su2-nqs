#!/usr/bin/env bash
#SBATCH --job-name=nqs-init-pilot
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/init-pilot-%j.out
#SBATCH --error=logs/init-pilot-%j.err

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
OUTPUT_ROOT="projects/clebsch_tree_nqs/results/remote_scale_first_initialization_pilot_${SLURM_JOB_ID}"
CONFIG="projects/clebsch_tree_nqs/configs/validation/spin_scale_first_initialization_pilot.json"
mkdir -p "$OUTPUT_ROOT"

python -c 'import json, jax, sys; row={"backend":jax.default_backend(),"devices":[str(x) for x in jax.devices()],"x64":bool(jax.config.jax_enable_x64)}; print(json.dumps(row)); sys.exit(0 if row["backend"]=="gpu" else 2)'
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits --loop=1 > "$OUTPUT_ROOT/gpu_telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT

for CASE_ID in triangular_6x6_j2_0.125 square_6x6_j2_0.5; do
  for ARM_ID in \
    exact_su2_balanced_schur_spatial_vit \
    fixed_sz_shared_autoregressive_spatial_vit \
    symmetry_unaware_autoregressive_spatial_vit; do
    python projects/clebsch_tree_nqs/run.py \
      spin-scale-first-initialization-pilot \
      --config "$CONFIG" \
      --case-id "$CASE_ID" \
      --arm-id "$ARM_ID" \
      --output "$OUTPUT_ROOT/$CASE_ID/$ARM_ID"
    python -c 'import json,sys; row=json.load(open(sys.argv[1])); assert row["passed"] is True; assert row["case_filter"] == sys.argv[2]; assert row["arm_filter"] == sys.argv[3]; assert len(row["runs"]) == 1' \
      "$OUTPUT_ROOT/$CASE_ID/$ARM_ID/results.json" "$CASE_ID" "$ARM_ID"
  done
done

sstat --jobs="${SLURM_JOB_ID}.batch" \
  --format=JobID,AveRSS,MaxRSS,AveVMSize,MaxVMSize -P || true
