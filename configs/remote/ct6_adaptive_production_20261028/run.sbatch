#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail

NQS_ADAPTIVE_TAG="${1:-}"
case "$NQS_ADAPTIVE_TAG" in
    ct6_j0|ct6_j0125) ;;
    *) echo 'Usage: run.sbatch ct6_j0|ct6_j0125' >&2; exit 2 ;;
esac
NQS_ADAPTIVE_WORKSPACE=/home/yx789/nqs-ct6-adaptive-production-20261028
NQS_ADAPTIVE_PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_ADAPTIVE_CONFIG=configs/remote/ct6_adaptive_production_20261028
NQS_ADAPTIVE_OUTPUT=/home/yx789/nqs-wavefunctions/nqs/ct6_adaptive_production_20261028/$NQS_ADAPTIVE_TAG
NQS_ADAPTIVE_STARTED_EPOCH="$(date +%s)"
export SLURM_JOB_END_TIME="${SLURM_JOB_END_TIME:-$((NQS_ADAPTIVE_STARTED_EPOCH+86400))}"
cd "$NQS_ADAPTIVE_WORKSPACE"
export PYTHONPATH="$NQS_ADAPTIVE_WORKSPACE" PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.90
unset XLA_PYTHON_CLIENT_MEM_FRACTION
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export JAX_COMPILATION_CACHE_DIR="$NQS_ADAPTIVE_WORKSPACE/cache/$NQS_ADAPTIVE_TAG"
"$NQS_ADAPTIVE_PYTHON" - "$NQS_ADAPTIVE_TAG" <<'PY'
import hashlib, importlib.metadata, json, os, pathlib, sys
root=pathlib.Path.cwd();config=root/'configs/remote/ct6_adaptive_production_20261028'
manifest=config/'source_manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=os.environ.get('NQS_CT6_ADAPTIVE_MANIFEST_SHA256'):
    raise ValueError('externally approved adaptive deployment manifest required')
for name,digest in json.loads(manifest.read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('sealed adaptive source/config differs: '+name)
for name,version in json.loads((config/'runtime_packages.json').read_text()).items():
    if importlib.metadata.version(name)!=version:
        raise RuntimeError('unchanged Kim numerical runtime differs: '+name)
parent=json.loads((config/'parent_inputs.json').read_text())[sys.argv[1]]
for name,digest in parent['sealed_files'].items():
    path=pathlib.Path(parent['remote_directory'])/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('frozen audited parent changed: '+str(path))
print('Adaptive CT6 package and independently audited parent sealed; no old workdir copied.',flush=True)
PY

# An allocation is never silently re-entered/requeued. Checkpoints preserve
# native parameters, original optimizer/RNG, and adaptive controller state.
# Any later allocation requires explicit new root authorization and review.
test ! -e "$NQS_ADAPTIVE_OUTPUT"
mkdir -p "$NQS_ADAPTIVE_OUTPUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$NQS_ADAPTIVE_OUTPUT/allocation_gpu.csv"
NQS_ADAPTIVE_RUN="$NQS_ADAPTIVE_OUTPUT/run01"
for NQS_ADAPTIVE_PHASE in prepare replay train audit; do
    /usr/local/slurm/current/bin/srun --ntasks=1 "$NQS_ADAPTIVE_PYTHON" -u -m \
        projects.clebsch_tree_nqs.workflows.ct6_adaptive_continuation \
        --plan "$NQS_ADAPTIVE_CONFIG/$NQS_ADAPTIVE_TAG.json" \
        --output "$NQS_ADAPTIVE_RUN" --phase "$NQS_ADAPTIVE_PHASE"
done
