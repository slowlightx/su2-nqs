#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --no-requeue
#SBATCH --signal=USR1@600
set -euo pipefail
CT12_GATE_MODE="${1:-}"
case "$CT12_GATE_MODE" in gate) CT12_GATE_SECONDS=21600 ;; pilot) CT12_GATE_SECONDS=50400 ;; *) exit 2 ;; esac
CT12_GATE_WORKSPACE=/home/yx789/nqs-ct12-kim-direction-gate-20261029
CT12_GATE_PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
CT12_GATE_CONFIG=configs/remote/ct12_direction_gate_20261029
CT12_GATE_OUTPUT=/home/yx789/nqs-wavefunctions/nqs/ct12_kim_direction_gate_20261029/j0125/run01
CT12_GATE_EPOCH="$(date +%s)"
export SLURM_JOB_END_TIME="${SLURM_JOB_END_TIME:-$((CT12_GATE_EPOCH+CT12_GATE_SECONDS))}"
cd "$CT12_GATE_WORKSPACE"
export PYTHONPATH="$CT12_GATE_WORKSPACE" PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.90
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset XLA_PYTHON_CLIENT_MEM_FRACTION LD_PRELOAD LD_LIBRARY_PATH
unset JAX_DISABLE_JIT JAX_SKIP_CUDA_CONSTRAINTS_CHECK XLA_FLAGS
unset CUBLAS_EMULATE_DOUBLE_PRECISION CUBLAS_EMULATE_SINGLE_PRECISION
export JAX_COMPILATION_CACHE_DIR="$CT12_GATE_WORKSPACE/cache/A100-j0125"
"$CT12_GATE_PYTHON" "$CT12_GATE_CONFIG/verify_deployment.py"

if [ "$CT12_GATE_MODE" = gate ]; then
    test ! -e "$CT12_GATE_OUTPUT"
    mkdir -p "$(dirname "$CT12_GATE_OUTPUT")"
    nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$(dirname "$CT12_GATE_OUTPUT")/gate_allocation_gpu.csv"
    CT12_GATE_PHASES="gate gate_replay"
else
    # A failed gate can exit normally after saving useful scientific evidence.
    # Refuse here, before initializing JAX or constructing any GPU runtime.
    if ! "$CT12_GATE_PYTHON" - "$CT12_GATE_OUTPUT" <<'PY'
import hashlib,json,pathlib,sys
out=pathlib.Path(sys.argv[1]);certificate=out/'gate_certificate.json'
passed=False
try:
    gate=json.loads(certificate.read_text())
    reg=json.loads((out/'registration.json').read_text())
    passed=(gate.get('passed') is True and gate.get('plan_sha256')==reg['plan_sha256']
        and gate.get('parent_parameter_sha256')==reg['parent_parameter_sha256']
        and gate.get('parent_rng_sha256')==reg['parent_rng_sha256']
        and gate.get('gate_candidate_is_production_step') is False
        and hashlib.sha256((out/'gate_result.json').read_bytes()).hexdigest()==gate['gate_result_sha256'])
except (OSError,ValueError,KeyError):
    pass
if not passed:
    out.mkdir(parents=True,exist_ok=True)
    (out/'pilot_refusal.json').write_text(json.dumps(dict(
        status='direction_gate_failed_or_incomplete_no_training',
        no_GPU_runtime_or_update_started=True,existing_CT6_jobs_unchanged=True),indent=2))
    print('CT12 pilot refused: no replayed passed direction certificate.',flush=True)
    raise SystemExit(3)
print('Passed same-parent scientific and replay certificate found.',flush=True)
PY
    then
        exit 0
    fi
    nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$(dirname "$CT12_GATE_OUTPUT")/pilot_allocation_gpu.csv"
    # The workflow's train_prepare rechecks the replayed scientific certificate.
    # A failed/incomplete gate must leave a refusal receipt and perform no update.
    CT12_GATE_PHASES="train_prepare train_replay train audit"
fi
for CT12_GATE_PHASE in $CT12_GATE_PHASES; do
    if /usr/local/slurm/current/bin/srun --ntasks=1 "$CT12_GATE_PYTHON" -u -m \
        projects.clebsch_tree_nqs.workflows.ct12_direction_gate \
        --plan "$CT12_GATE_CONFIG/j0125.json" --output "$CT12_GATE_OUTPUT" --phase "$CT12_GATE_PHASE"; then
        continue
    else
        CT12_GATE_RETURN=$?
        # A bounded stop during train can still have an atomically committed
        # prefix. Use the reserved audit time for THAT frozen state only.
        # No numerical error, partial proposal or incomplete replay is ignored.
        if [ "$CT12_GATE_MODE" = pilot ] && [ "$CT12_GATE_PHASE" = train ] && [ "$CT12_GATE_RETURN" = 2 ]; then
            if "$CT12_GATE_PYTHON" - "$CT12_GATE_OUTPUT" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1])
try:
    result=json.loads((out/'train_result.json').read_text())
    frozen=json.loads((out/'endpoint_frozen.json').read_text())
    valid=(result.get('status')=='bounded_interruption_parent_or_committed_prefix_preserved'
        and result.get('committed_prefix_frozen_for_independent_audit') is True
        and frozen.get('completed_updates',0)>0 and (out/'checkpoint.pkl').is_file())
except (OSError,ValueError):
    valid=False
raise SystemExit(0 if valid else 2)
PY
            then
                continue
            fi
        fi
        exit "$CT12_GATE_RETURN"
    fi
done
