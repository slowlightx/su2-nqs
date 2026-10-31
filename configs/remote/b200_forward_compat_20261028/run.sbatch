#!/usr/bin/env bash
#SBATCH --account=kim
#SBATCH --qos=normal
#SBATCH --partition=aimi
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --job-name=b200_compat_acceptance
set -euo pipefail

# Software verification only, no optimizer step or production-budget release.
# Do not constrain a hostname or modify any existing A100 allocation.
NQS_RUNTIME_ROOT=/home/yx789/nqs-b200-validation-compat-20261028
NQS_RUNTIME_PYTHON=/home/yx789/nqs-b200-runtime-20261028/venv/bin/python
NQS_RUNTIME_CONFIG=configs/remote/b200_forward_compat_20261028
NQS_RUNTIME_OUTPUT=/home/yx789/nqs-b200-validation-compat-20261028/gpu
cd "$NQS_RUNTIME_ROOT"
export PYTHONPATH="$NQS_RUNTIME_ROOT" PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export TF_CPP_MIN_LOG_LEVEL=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset LD_LIBRARY_PATH LD_PRELOAD JAX_SKIP_CUDA_CONSTRAINTS_CHECK PYTHONWARNINGS
export LD_LIBRARY_PATH=/home/yx789/nqs-b200-forward-compat-20261028/lib
unset JAX_DISABLE_JIT XLA_FLAGS CUBLAS_EMULATE_DOUBLE_PRECISION CUBLAS_EMULATE_SINGLE_PRECISION
export JAX_COMPILATION_CACHE_DIR="$NQS_RUNTIME_ROOT/cache/B200-runtime-acceptance"

"$NQS_RUNTIME_PYTHON" - <<'PY'
import hashlib,json,os,pathlib
root=pathlib.Path.cwd()
manifest=root/'configs/remote/b200_forward_compat_20261028/source_manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=os.environ.get('NQS_RUNTIME_MANIFEST_SHA256'):
    raise ValueError('external deployment SHA256 required')
for name,digest in json.loads(manifest.read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('isolated software-verification deployment changed: '+name)
print('Frozen software-only B200 acceptance; no training or automatic production release.',flush=True)
PY

test ! -e "$NQS_RUNTIME_OUTPUT"
mkdir -p "$NQS_RUNTIME_OUTPUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$NQS_RUNTIME_OUTPUT/allocation_gpu.csv"
/usr/local/slurm/current/bin/srun --ntasks=1 "$NQS_RUNTIME_PYTHON" -u -m \
    projects.clebsch_tree_nqs.workflows.accelerator_runtime_validation \
    --phase gpu_check --plan "$NQS_RUNTIME_CONFIG/plan.json" \
    --input-root "$NQS_RUNTIME_ROOT" --output "$NQS_RUNTIME_OUTPUT" \
    --golden projects/clebsch_tree_nqs/results/b200_runtime_fix_20261028/cpu_golden_compat
