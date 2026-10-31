#!/usr/bin/env bash
#SBATCH --job-name=nqs-proj-mult
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/projected-multiplicity-%j.out
#SBATCH --error=logs/projected-multiplicity-%j.err

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
TASK_CONFIG="${PROJECTED_MULTIPLICITY_CONFIG:-projects/clebsch_tree_nqs/configs/remote/spin_n36_projected_multiplicity_a100_pilot.json}"
OUTPUT_ROOT="${PROJECTED_MULTIPLICITY_OUTPUT:-projects/clebsch_tree_nqs/results/remote_projected_multiplicity_${SLURM_JOB_ID}}"
mkdir -p "$OUTPUT_ROOT"
python -c 'import json,jax,sys; row={"backend":jax.default_backend(),"devices":[str(x) for x in jax.devices()],"x64":bool(jax.config.jax_enable_x64)}; print(json.dumps(row)); sys.exit(0 if row["backend"]=="gpu" else 2)'
python -c 'import pathlib,su2_nqs,sys; root=pathlib.Path(sys.argv[1]).resolve(); source=pathlib.Path(su2_nqs.__file__).resolve(); print({"project_root":str(root),"su2_nqs_source":str(source)}); sys.exit(0 if source.is_relative_to(root) else 3)' "$PROJECT_ROOT"
TELEMETRY_FILE="$OUTPUT_ROOT/gpu_telemetry.csv"
ACCOUNTING_FILE="$OUTPUT_ROOT/slurm_accounting_snapshots.txt"
{
  printf '# restart_utc=%s slurm_job_id=%s restart_count=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$SLURM_JOB_ID" \
    "${SLURM_RESTART_COUNT:-0}"
  exec nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu \
    --format=csv,noheader,nounits --loop=5
} >> "$TELEMETRY_FILE" &
TELEMETRY_PID=$!
capture_exit_accounting() {
  kill "$TELEMETRY_PID" 2>/dev/null || true
  {
    printf '# snapshot_utc=%s slurm_job_id=%s restart_count=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$SLURM_JOB_ID" \
      "${SLURM_RESTART_COUNT:-0}"
    /usr/local/slurm/current/bin/sstat \
      --jobs="${SLURM_JOB_ID}.batch" \
      --format=JobID,AveRSS,MaxRSS,AveVMSize,MaxVMSize -P || true
    /usr/local/slurm/current/bin/sacct \
      --jobs="$SLURM_JOB_ID" \
      --format=JobIDRaw,JobName,Partition,State,ElapsedRaw,AllocTRES,ReqTRES,NodeList \
      -P || true
  } >> "$ACCOUNTING_FILE" 2>&1
}
trap capture_exit_accounting EXIT
printf '%s\n' \
  "/usr/local/slurm/current/bin/sacct --jobs=${SLURM_JOB_ID} --format=JobIDRaw,JobName,Partition,State,ElapsedRaw,AllocTRES,ReqTRES,NodeList -P > ${OUTPUT_ROOT}/slurm_sacct_final.txt" \
  > "$OUTPUT_ROOT/POST_JOB_SACCT_COMMAND.txt"
python projects/clebsch_tree_nqs/run.py spin-n36-projected-multiplicity-preflight \
  --config "$TASK_CONFIG" \
  --output "$OUTPUT_ROOT"
