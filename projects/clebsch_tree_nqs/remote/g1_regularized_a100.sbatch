#!/usr/bin/env bash
#SBATCH --job-name=nqs-g1-reg-dev
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/g1-regularized-%j.out
#SBATCH --error=logs/g1-regularized-%j.err

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
export NQS_PLAN_WORKERS="${NQS_PLAN_WORKERS:-4}"
TASK_CONFIG="${G1R_CONFIG:-projects/clebsch_tree_nqs/configs/remote/spin_g1_regularized_n36_development.json}"
TASK_PHASE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "$TASK_CONFIG")"
OUTPUT_ROOT="${G1R_OUTPUT:-projects/clebsch_tree_nqs/results/remote_g1_regularized_${TASK_PHASE}_${SLURM_JOB_ID}}"
mkdir -p "$OUTPUT_ROOT"
python -c 'import json,jax,sys; row={"backend":jax.default_backend(),"devices":[str(x) for x in jax.devices()],"x64":bool(jax.config.jax_enable_x64)}; print(json.dumps(row)); sys.exit(0 if row["backend"]=="gpu" else 2)'
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv,noheader,nounits --loop=30 > "$OUTPUT_ROOT/gpu_telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT
python projects/clebsch_tree_nqs/run.py spin-g1-regularized-operator \
  --config "$TASK_CONFIG" \
  --output "$OUTPUT_ROOT"
/usr/local/slurm/current/bin/sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveRSS,MaxRSS,AveVMSize,MaxVMSize -P || true
