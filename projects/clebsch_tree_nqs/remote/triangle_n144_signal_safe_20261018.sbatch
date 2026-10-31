#!/usr/bin/env bash
# Execution-only correction. Caller supplies unchanged90/30min allocation.
set -euo pipefail
cd /home/yx789/nqs-triangle-n144-native-20261018
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=True
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
case "${NQS_N144_GATE_LABEL:?explicit registered safe config required}" in
    j0125_gate_safe|j0_chunk32_gate_safe|j0125_chunk32_gate_safe) ;;
    *) echo "unsupported safe gate label" >&2; exit 2 ;;
esac
NQS_CONFIG="projects/clebsch_tree_nqs/configs/remote/triangle_n144_native_20261018/${NQS_N144_GATE_LABEL}.json"
if [[ "${NQS_N144_COLLECT:-0}" == 1 ]]; then
    export JAX_PLATFORMS=cpu
    export JAX_PLATFORM_NAME=cpu
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_native_gate \
        --config "$NQS_CONFIG" --collect-job "${NQS_N144_JOB:?completed GPU job}"
fi
export JAX_PLATFORMS=cuda
export JAX_PLATFORM_NAME=gpu
NQS_CHILD_PID=""
NQS_STOP_REQUESTED=0

# SIGNAL_SAFE_CHILD_BEGIN
# Reuse the tested triangle_dictionary_campaign_a100.sbatch trap/wait logic.
# The only generalization is accepting the child command as positional args.
trap 'NQS_STOP_REQUESTED=1; if [[ -n "$NQS_CHILD_PID" ]]; then kill -USR1 "$NQS_CHILD_PID" 2>/dev/null || true; fi' USR1 TERM
run_child() {
    if [[ "$NQS_STOP_REQUESTED" == 1 ]]; then exit 0; fi
    "$@" &
    NQS_CHILD_PID=$!
    local nqs_child_status=0
    # A trapped shell signal interrupts wait, not necessarily the child.
    # Keep waiting for that SAME checkpointing process despite set-e.
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
# SIGNAL_SAFE_CHILD_END

run_child "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_native_gate \
    --config "$NQS_CONFIG"
# A signal during or between phases prevents starting this fresh process.
run_child "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_native_gate \
    --config "$NQS_CONFIG" --replay-gate
