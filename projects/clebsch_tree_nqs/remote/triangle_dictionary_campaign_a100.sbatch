#!/usr/bin/env bash
# Separate, low-priority static-generator pilot. Never patch a mainline workspace.
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nice=10000
#SBATCH --signal=B:USR1@120
set -euo pipefail

cd "${NQS_DICTIONARY_WORKSPACE:?explicit isolated workspace required}"
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=1
unset JAX_PLATFORM_NAME
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
NQS_DICTIONARY_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_DICTIONARY_CONFIG=projects/clebsch_tree_nqs/configs/validation/triangle_dictionary_j0125_calibrated_gpu_pilot_20261018.json
NQS_CHILD_PID=""
NQS_MONITOR_PID=""
NQS_STOP_REQUESTED=0

cleanup_monitor() {
    if [[ -n "$NQS_MONITOR_PID" ]]; then
        kill "$NQS_MONITOR_PID" 2>/dev/null || true
        wait "$NQS_MONITOR_PID" 2>/dev/null || true
    fi
}
trap cleanup_monitor EXIT
mkdir -p logs
# Sample ONLY this allocation's GPU. Device-used memory is an execution
# diagnostic, not a claimed exact allocator peak or total-HBM requirement.
if command -v nvidia-smi >/dev/null && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
        --query-gpu=timestamp,index,memory.total,memory.used,utilization.gpu \
        --format=csv,nounits -l 5 > "logs/gpu_memory_${SLURM_JOB_ID}.csv" &
    NQS_MONITOR_PID=$!
fi

# Ask the active controller to stop at a checkpoint boundary. The Slurm time
# limit remains the external hard limit; no mainline process is signalled.
trap 'NQS_STOP_REQUESTED=1; if [[ -n "$NQS_CHILD_PID" ]]; then kill -USR1 "$NQS_CHILD_PID" 2>/dev/null || true; fi' USR1 TERM
run_phase() {
    # A signal can arrive between phases: do not start another process then.
    if [[ "$NQS_STOP_REQUESTED" == 1 ]]; then exit 0; fi
    "$NQS_DICTIONARY_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_dictionary_campaign \
        --config "$NQS_DICTIONARY_CONFIG" --seed-index 0 "$@" &
    NQS_CHILD_PID=$!
    local nqs_child_status=0
    # bash wait returns128+signal when the SHELL receives USR1/TERM, even
    # though the child is still checkpointing. Keep waiting for that same
    # child instead of allowing set -e to tear down the allocation early.
    while true; do
        if wait "$NQS_CHILD_PID"; then
            nqs_child_status=0
            break
        else
            nqs_child_status=$?
        fi
        if ! kill -0 "$NQS_CHILD_PID" 2>/dev/null; then
            if wait "$NQS_CHILD_PID"; then nqs_child_status=0; else nqs_child_status=$?; fi
            break
        fi
    done
    NQS_CHILD_PID=""
    if [[ "$NQS_STOP_REQUESTED" == 1 ]]; then exit "$nqs_child_status"; fi
    return "$nqs_child_status"
}

"$NQS_DICTIONARY_PY" -c 'import jax; assert jax.default_backend() == "gpu"; print(jax.devices())'
run_phase --until-step 1
run_phase --resume
run_phase --replay-completed
