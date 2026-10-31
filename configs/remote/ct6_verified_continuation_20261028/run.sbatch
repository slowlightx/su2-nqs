#!/usr/bin/env bash
#SBATCH --account=kim
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --signal=USR1@300
set -euo pipefail

NQS_FOLLOWUP_TAG="${1:?explicit ct6_j0/ct6_j0125 required}"
case "$NQS_FOLLOWUP_TAG" in
    ct6_j0) NQS_FOLLOWUP_DEVICE=A100; NQS_FOLLOWUP_LIMIT=28800 ;;
    ct6_j0125) NQS_FOLLOWUP_DEVICE=A100; NQS_FOLLOWUP_LIMIT=36000 ;;
    *) exit 2 ;;
esac
NQS_FOLLOWUP_WORKSPACE=/home/yx789/nqs-ct6-verified-continuation-20261028
NQS_FOLLOWUP_PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_FOLLOWUP_OUTPUT=/home/yx789/nqs-wavefunctions/nqs/ct6_verified_continuation_20261028/$NQS_FOLLOWUP_TAG
NQS_FOLLOWUP_STARTED_EPOCH="$(date +%s)"
export SLURM_JOB_END_TIME="${SLURM_JOB_END_TIME:-$((NQS_FOLLOWUP_STARTED_EPOCH+NQS_FOLLOWUP_LIMIT))}"
cd "$NQS_FOLLOWUP_WORKSPACE"
export PYTHONPATH="$NQS_FOLLOWUP_WORKSPACE" PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=.90
unset XLA_PYTHON_CLIENT_MEM_FRACTION
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export JAX_COMPILATION_CACHE_DIR="$NQS_FOLLOWUP_WORKSPACE/cache/$NQS_FOLLOWUP_DEVICE/$NQS_FOLLOWUP_TAG"

"$NQS_FOLLOWUP_PYTHON" - <<'PY'
import hashlib,json,os,pathlib,importlib.metadata
root=pathlib.Path.cwd();directory=root/'configs/remote/ct6_verified_continuation_20261028'
manifest=directory/'source_manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=os.environ.get('NQS_FOLLOWUP_MANIFEST_SHA256'):
    raise ValueError('external deployment manifest identity required')
for name,digest in json.loads(manifest.read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('new isolated deployment file changed: '+name)
for name,version in json.loads((directory/'runtime_packages.json').read_text()).items():
    if importlib.metadata.version(name)!=version:
        raise RuntimeError('read-only numerical environment version changed: '+name)
print('Sealed new bounded allocation; old jobs/checkpoints/environments are not modified.',flush=True)
PY

# This dedicated output is new. Preserve any existing completed/partial run;
# do not reuse its allocation or silently truncate a log/checkpoint.
test ! -e "$NQS_FOLLOWUP_OUTPUT"
mkdir -p "$NQS_FOLLOWUP_OUTPUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$NQS_FOLLOWUP_OUTPUT/allocation_gpu.csv"
# prepare expects a truly fresh science directory; hardware log lives beside it.
NQS_FOLLOWUP_RUN="$NQS_FOLLOWUP_OUTPUT/run01"
for NQS_FOLLOWUP_PHASE in prepare replay train audit; do
    /usr/local/slurm/current/bin/srun --ntasks=1 "$NQS_FOLLOWUP_PYTHON" -u -m \
        projects.clebsch_tree_nqs.workflows.ct_bounded_followup \
        --plan "configs/remote/ct6_verified_continuation_20261028/$NQS_FOLLOWUP_TAG.json" \
        --output "$NQS_FOLLOWUP_RUN" --phase "$NQS_FOLLOWUP_PHASE"
done
