#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=18:00:00
#SBATCH --no-requeue
#SBATCH --signal=USR1@300
#SBATCH --output=/home/yx789/nqs-wavefunctions/nqs/slurm-ct12-reentry-%j.out
set -euo pipefail
: "${NQS_CODE_RELEASE:?root must supply immutable detached code-only release}"
: "${NQS_CODE_COMMIT:?root must supply full40-character commit}"
: "${NQS_CT12_OUTPUT:?root must supply fresh external scientific output}"
NQS_CT12_MODE="${1:-gate}"
NQS_CT12_PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_CT12_PLAN="$NQS_CODE_RELEASE/configs/remote/ct12_wide_reentry_20261101/plan.json"
NQS_CT12_MODULE=projects.clebsch_tree_nqs.workflows.ct12_wide_reentry_20261101
case "$NQS_CT12_MODE" in
  gate) NQS_CT12_CAP=64800 ;;
  pilot) NQS_CT12_CAP=259200; : "${NQS_CT12_PILOT_RELEASE:?separate same-certificate root release required}" ;;
  *) exit 64 ;;
esac
test "$(git -C "$NQS_CODE_RELEASE" rev-parse HEAD)" = "$NQS_CODE_COMMIT"
test -z "$(git -C "$NQS_CODE_RELEASE" status --porcelain)"
NQS_CT12_START="$(date +%s)"
export SLURM_JOB_END_TIME="${SLURM_JOB_END_TIME:-$((NQS_CT12_START+NQS_CT12_CAP))}"
export PYTHONPATH="$NQS_CODE_RELEASE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.85
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
unset LD_PRELOAD LD_LIBRARY_PATH JAX_DISABLE_JIT JAX_SKIP_CUDA_CONSTRAINTS_CHECK XLA_FLAGS
unset CUBLAS_EMULATE_DOUBLE_PRECISION CUBLAS_EMULATE_SINGLE_PRECISION XLA_CLIENT_MEM_FRACTION
export JAX_COMPILATION_CACHE_DIR="${NQS_CT12_OUTPUT}.xla-cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
cd "$NQS_CODE_RELEASE"
"$NQS_CT12_PYTHON" "$NQS_CODE_RELEASE/configs/remote/ct12_actual1024_20261031/verify_runtime.py"
if [ "$NQS_CT12_MODE" = gate ]; then
  /usr/local/slurm/current/bin/srun --ntasks=1 "$NQS_CT12_PYTHON" -u -m "$NQS_CT12_MODULE" \
    --plan "$NQS_CT12_PLAN" --output "$NQS_CT12_OUTPUT" --phase gate
  /usr/local/slurm/current/bin/srun --ntasks=1 "$NQS_CT12_PYTHON" -u -m "$NQS_CT12_MODULE" \
    --plan "$NQS_CT12_PLAN" --output "$NQS_CT12_OUTPUT" --phase replay
else
  # Root must explicitly request --time=72:00:00 when submitting pilot mode.
  # This script never releases or submits the pilot following a gate.
  for NQS_CT12_PHASE in pilot_prepare pilot_replay pilot_train audit; do
    /usr/local/slurm/current/bin/srun --ntasks=1 "$NQS_CT12_PYTHON" -u -m "$NQS_CT12_MODULE" \
      --plan "$NQS_CT12_PLAN" --output "$NQS_CT12_OUTPUT" --phase "$NQS_CT12_PHASE" \
      --pilot-release "$NQS_CT12_PILOT_RELEASE"
  done
fi
