#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --signal=USR1@300
#SBATCH --output=/home/yx789/nqs-wavefunctions/nqs/slurm-ct6-one-extra-%j.out
set -euo pipefail
: "${NQS_CODE_RELEASE:?exact detached readonly code-only release required}"
: "${NQS_CODE_COMMIT:?full40-character commit SHA required}"
NQS_CT6_EXTRA_OUTPUT="${NQS_CT6_EXTRA_OUTPUT:-/home/yx789/nqs-wavefunctions/nqs/ct6_one_extra_segment_20261031/ct6_j0125}"
NQS_CT6_EXTRA_PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_CT6_EXTRA_CONFIG="$NQS_CODE_RELEASE/configs/remote/ct6_one_extra_segment_20261031"
NQS_CT6_EXTRA_PLAN="$NQS_CT6_EXTRA_CONFIG/plan.json"
NQS_CT6_EXTRA_MODULE=projects.clebsch_tree_nqs.workflows.ct6_one_extra_segment_20261031
test "$(git -C "$NQS_CODE_RELEASE" rev-parse HEAD)" = "$NQS_CODE_COMMIT"
test -z "$(git -C "$NQS_CODE_RELEASE" status --porcelain)"
test ! -e "$NQS_CT6_EXTRA_OUTPUT"
NQS_CT6_EXTRA_EPOCH="$(date +%s)"
export SLURM_JOB_END_TIME="${SLURM_JOB_END_TIME:-$((NQS_CT6_EXTRA_EPOCH+86400))}"
export PYTHONPATH="$NQS_CODE_RELEASE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export JAX_ENABLE_X64=True
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset LD_PRELOAD LD_LIBRARY_PATH JAX_DISABLE_JIT JAX_SKIP_CUDA_CONSTRAINTS_CHECK XLA_FLAGS
unset CUBLAS_EMULATE_DOUBLE_PRECISION CUBLAS_EMULATE_SINGLE_PRECISION XLA_CLIENT_MEM_FRACTION
cd "$NQS_CODE_RELEASE"
"$NQS_CT6_EXTRA_PYTHON" "$NQS_CT6_EXTRA_CONFIG/verify_runtime.py"
# afterany:763126 is submitted ONCE by the operator, never by this script.
# CPU-only immutable audit gate happens before any GPU backend/draw/H work.
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu "$NQS_CT6_EXTRA_PYTHON" -u -m "$NQS_CT6_EXTRA_MODULE" \
  --plan "$NQS_CT6_EXTRA_PLAN" --output "$NQS_CT6_EXTRA_OUTPUT" --phase gate
if ! "$NQS_CT6_EXTRA_PYTHON" - "$NQS_CT6_EXTRA_OUTPUT/parent_gate.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['status']=='passed' else 1)
PY
then
  echo 'Current terminal science/provenance gate refused; no new draws/H, no successor.'
  exit 0
fi
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=.90
unset XLA_PYTHON_CLIENT_MEM_FRACTION
export JAX_COMPILATION_CACHE_DIR="${NQS_CT6_EXTRA_OUTPUT}.xla-cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
for NQS_CT6_EXTRA_PHASE in prepare replay train audit; do
  /usr/local/slurm/current/bin/srun --ntasks=1 "$NQS_CT6_EXTRA_PYTHON" -u -m "$NQS_CT6_EXTRA_MODULE" \
    --plan "$NQS_CT6_EXTRA_PLAN" --output "$NQS_CT6_EXTRA_OUTPUT" --phase "$NQS_CT6_EXTRA_PHASE"
done
# No sbatch, requeue, successor, second coupling, or additional allocation.
