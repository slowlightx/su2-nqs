#!/usr/bin/env bash
#SBATCH --job-name=nqs-carrier-one
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/carrier-one-%A_%a.out
#SBATCH --error=logs/carrier-one-%A_%a.err

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

CASES=(
  triangular_6x6_j2_0.125 triangular_6x6_j2_0.125 triangular_6x6_j2_0.125
  square_6x6_j2_0.5 square_6x6_j2_0.5 square_6x6_j2_0.5
)
ARMS=(
  exact_su2_balanced_schur_spatial_vit
  fixed_sz_shared_autoregressive_spatial_vit
  symmetry_unaware_autoregressive_spatial_vit
  exact_su2_balanced_schur_spatial_vit
  fixed_sz_shared_autoregressive_spatial_vit
  symmetry_unaware_autoregressive_spatial_vit
)
CASE_ID="${CASES[$SLURM_ARRAY_TASK_ID]}"
ARM_ID="${ARMS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_ROOT="${NQS_OUTPUT_ROOT:-projects/clebsch_tree_nqs/results/remote_scale_first_carrier_single_seed_${SLURM_ARRAY_JOB_ID}}"
OUTPUT="$OUTPUT_ROOT/$CASE_ID/$ARM_ID"
CONFIG="projects/clebsch_tree_nqs/configs/remote/spin_scale_first_carrier_single_seed.json"
mkdir -p "$OUTPUT"

EXTRA_ARGS=()
EXPECTED_STEPS=64
if [[ -n "${NQS_MAX_STEPS_THIS_CALL:-}" ]]; then
  EXTRA_ARGS+=(--maximum-steps-this-call "$NQS_MAX_STEPS_THIS_CALL")
  EXPECTED_STEPS="$NQS_MAX_STEPS_THIS_CALL"
fi

python -c 'import json, jax, sys; row={"backend":jax.default_backend(),"devices":[str(x) for x in jax.devices()],"x64":bool(jax.config.jax_enable_x64)}; print(json.dumps(row)); sys.exit(0 if row["backend"]=="gpu" else 2)'
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits --loop=1 > "$OUTPUT/gpu_telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT

python projects/clebsch_tree_nqs/run.py \
  spin-scale-first-carrier-single-seed \
  --config "$CONFIG" \
  --case-id "$CASE_ID" \
  --arm-id "$ARM_ID" \
  --output "$OUTPUT" \
  "${EXTRA_ARGS[@]}"
python -c 'import json,sys; row=json.load(open(sys.argv[1])); expected=int(sys.argv[4]); assert row["case_id"] == sys.argv[2]; assert row["arm_id"] == sys.argv[3]; assert row["completed_steps"] == expected; assert row["production_budget_released"] is False; assert row["passed"] is (expected == 64)' \
  "$OUTPUT/results.json" "$CASE_ID" "$ARM_ID" "$EXPECTED_STEPS"

sstat --jobs="${SLURM_JOB_ID}.batch" \
  --format=JobID,AveRSS,MaxRSS,AveVMSize,MaxVMSize -P || true
