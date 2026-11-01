#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/yx789/nqs-wavefunctions/nqs/slurm-S1K-full-W-%j.out
set -euo pipefail
: "${NQS_CODE_RELEASE:?exact detached readonly code release required}"
: "${NQS_CODE_COMMIT:?full Git commit SHA required}"
: "${NQS_S1K_OUTPUT:?fresh external output directory required}"
: "${NQS_S1K_APPROVED_PLAN_SHA256:?root-reviewed approved plan SHA required}"
NQS_S1K_PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_S1K_PLAN="$NQS_CODE_RELEASE/configs/remote/ct_s1k_full_network_20261101/plan.json"
NQS_S1K_MODULE=projects.clebsch_tree_nqs.workflows.ct_s1k_full_network_20261101
test "$(git -C "$NQS_CODE_RELEASE" rev-parse HEAD)" = "$NQS_CODE_COMMIT"
test -z "$(git -C "$NQS_CODE_RELEASE" status --porcelain)"
test ! -e "$NQS_S1K_OUTPUT"
export PYTHONPATH="$NQS_CODE_RELEASE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.85
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset LD_PRELOAD LD_LIBRARY_PATH XLA_FLAGS JAX_DISABLE_JIT JAX_SKIP_CUDA_CONSTRAINTS_CHECK
unset XLA_CLIENT_MEM_FRACTION CUBLAS_EMULATE_DOUBLE_PRECISION CUBLAS_EMULATE_SINGLE_PRECISION
export JAX_COMPILATION_CACHE_DIR="${NQS_S1K_OUTPUT}.xla-cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
cd "$NQS_CODE_RELEASE"
"$NQS_S1K_PYTHON" "$NQS_CODE_RELEASE/configs/remote/ct6_one_extra_segment_20261031/verify_runtime.py"
NQS_S1K_EPOCH="$(date +%s)"
export SLURM_JOB_END_TIME="${SLURM_JOB_END_TIME:-$((NQS_S1K_EPOCH+43200))}"
/usr/local/slurm/current/bin/srun --ntasks=1 "$NQS_S1K_PYTHON" -u -m "$NQS_S1K_MODULE" \
  --plan "$NQS_S1K_PLAN" --output "$NQS_S1K_OUTPUT" --phase train \
  --approved-plan-sha256 "$NQS_S1K_APPROVED_PLAN_SHA256"
# Separate Python process: original128 SR, full complex H/Q and exact next RNG.
/usr/local/slurm/current/bin/srun --ntasks=1 "$NQS_S1K_PYTHON" -u -m "$NQS_S1K_MODULE" \
  --plan "$NQS_S1K_PLAN" --output "$NQS_S1K_OUTPUT" --phase replay
# S0X requires a separate independent outcome review. No successor submission.
