#!/usr/bin/env bash
# Prepared locally only. Root must review the sealed plan and authorize any submission.
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=USR1@600
#SBATCH --output=ct-S1K-source128-%j.log
set -euo pipefail
if [ "$#" -ne 2 ]; then
    echo "Requires the reviewed plan SHA256 and a fresh isolated output directory." >&2
    exit 2
fi
CT_S1K_APPROVED_PLAN_SHA256="$1"
CT_S1K_OUTPUT="$2"
CT_S1K_WORKSPACE="${CT_S1K_WORKSPACE:-/home/yx789/nqs-source-aware-S1K128-20261030-v2}"
CT_S1K_PYTHON="${CT_S1K_PYTHON:-/home/yx789/nqs-vit-quality/.conda-netket/bin/python}"
CT_S1K_PREPARATION=projects/clebsch_tree_nqs/results/ct_source_aware_S1K_20261030/v2_preparation
CT_S1K_EPOCH="$(date +%s)"
export SLURM_JOB_END_TIME="${SLURM_JOB_END_TIME:-$((CT_S1K_EPOCH+7200))}"
cd "$CT_S1K_WORKSPACE"
export PYTHONPATH="$CT_S1K_WORKSPACE" PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.90
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset XLA_PYTHON_CLIENT_MEM_FRACTION LD_PRELOAD LD_LIBRARY_PATH
unset JAX_DISABLE_JIT JAX_SKIP_CUDA_CONSTRAINTS_CHECK XLA_FLAGS
unset CUBLAS_EMULATE_DOUBLE_PRECISION CUBLAS_EMULATE_SINGLE_PRECISION
export JAX_COMPILATION_CACHE_DIR="$CT_S1K_WORKSPACE/cache/A100-S1K128-v2"
"$CT_S1K_PYTHON" - "$CT_S1K_PREPARATION/payload_manifest.json" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path.cwd()
for name,expected in json.loads(pathlib.Path(sys.argv[1]).read_text()).items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=expected:
        raise SystemExit("Sealed payload differs: "+name)
PY
test ! -e "$CT_S1K_OUTPUT"
/usr/local/slurm/current/bin/srun --ntasks=1 "$CT_S1K_PYTHON" -u -m \
    projects.clebsch_tree_nqs.explorations.ct_source_aware_direction_20261030 \
    --phase execute --plan "$CT_S1K_PREPARATION/plan.json" \
    --output "$CT_S1K_OUTPUT" --approved-plan-sha256 "$CT_S1K_APPROVED_PLAN_SHA256"
