#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --signal=USR1@300
set -euo pipefail
PREFLIGHT_ROOT=/home/yx789/nqs-ct12-w28-gpu-preflight-20261030
PREFLIGHT_PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
PREFLIGHT_CONFIG=configs/remote/ct12_w28_gpu_preflight_20261030
PREFLIGHT_OUTPUT=/home/yx789/nqs-wavefunctions/nqs/ct12_w28_gpu_preflight_20261030/run01
PREFLIGHT_EPOCH="$(date +%s)"
export SLURM_JOB_END_TIME="${SLURM_JOB_END_TIME:-$((PREFLIGHT_EPOCH+14400))}"
cd "$PREFLIGHT_ROOT"
export PYTHONPATH="$PREFLIGHT_ROOT" PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.85
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset LD_PRELOAD LD_LIBRARY_PATH JAX_DISABLE_JIT JAX_SKIP_CUDA_CONSTRAINTS_CHECK XLA_FLAGS
unset CUBLAS_EMULATE_DOUBLE_PRECISION CUBLAS_EMULATE_SINGLE_PRECISION XLA_CLIENT_MEM_FRACTION
export JAX_COMPILATION_CACHE_DIR="$PREFLIGHT_ROOT/cache/A100-w28"
"$PREFLIGHT_PYTHON" "$PREFLIGHT_CONFIG/verify_deployment.py"
test ! -e "$PREFLIGHT_OUTPUT"
/usr/local/slurm/current/bin/srun --ntasks=1 "$PREFLIGHT_PYTHON" -u -m \
  projects.clebsch_tree_nqs.explorations.ct12_w28_gpu_preflight_20261030 \
  --plan "$PREFLIGHT_CONFIG/plan.json" --output "$PREFLIGHT_OUTPUT"
