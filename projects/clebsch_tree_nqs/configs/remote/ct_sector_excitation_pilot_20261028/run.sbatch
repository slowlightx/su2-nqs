#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail

NQS_SECTOR_TAG="${1:-}"
case "$NQS_SECTOR_TAG" in
    j0_q2_K|j0_q0_X|j0125_q2_K|j0125_q0_X) ;;
    *) echo 'Usage: run.sbatch j0_q2_K|j0_q0_X|j0125_q2_K|j0125_q0_X' >&2; exit 2 ;;
esac
NQS_SECTOR_WORKSPACE=/home/yx789/nqs-ct6-sector-excitation-pilot-20261028
NQS_SECTOR_PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_SECTOR_CONFIG=projects/clebsch_tree_nqs/configs/remote/ct_sector_excitation_pilot_20261028
NQS_SECTOR_OUTPUT=projects/clebsch_tree_nqs/results/ct_sector_excitation_pilot_20261028/gpu_$NQS_SECTOR_TAG
cd "$NQS_SECTOR_WORKSPACE"
export PYTHONPATH="$NQS_SECTOR_WORKSPACE" PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.90
unset XLA_PYTHON_CLIENT_MEM_FRACTION
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export JAX_COMPILATION_CACHE_DIR="$NQS_SECTOR_WORKSPACE/cache/$NQS_SECTOR_TAG"
"$NQS_SECTOR_PYTHON" - <<'PY'
import hashlib, importlib.metadata, json, os, pathlib
root=pathlib.Path.cwd()
config=root/'projects/clebsch_tree_nqs/configs/remote/ct_sector_excitation_pilot_20261028'
manifest=config/'source_manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=os.environ.get('NQS_SECTOR_DEPLOYMENT_SHA256'):
    raise ValueError('explicit externally approved deployment SHA is required')
for name, digest in json.loads(manifest.read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('sealed input/source differs: '+name)
for name, version in json.loads((config/'runtime_packages.json').read_text()).items():
    if importlib.metadata.version(name)!=version:
        raise RuntimeError('preserved Kim numerical runtime differs: '+name)
print('Verified isolated bounded sector deployment; old allocations unchanged.',flush=True)
PY
mkdir -p logs
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "logs/${NQS_SECTOR_TAG}-${SLURM_JOB_ID:-manual}-gpu.csv"
if [[ "${NQS_SECTOR_RESUME:-0}" == 1 ]]; then
    test -f "$NQS_SECTOR_OUTPUT/checkpoint.pkl"
    /usr/local/slurm/current/bin/srun --ntasks=1 "$NQS_SECTOR_PYTHON" -u -m \
        projects.clebsch_tree_nqs.workflows.ct_sector_excitation_pilot \
        --stage resume --plan "$NQS_SECTOR_CONFIG/$NQS_SECTOR_TAG.json" --output "$NQS_SECTOR_OUTPUT"
else
    test ! -e "$NQS_SECTOR_OUTPUT"
    for NQS_SECTOR_STAGE in prepare train; do
        /usr/local/slurm/current/bin/srun --ntasks=1 "$NQS_SECTOR_PYTHON" -u -m \
            projects.clebsch_tree_nqs.workflows.ct_sector_excitation_pilot \
            --stage "$NQS_SECTOR_STAGE" --plan "$NQS_SECTOR_CONFIG/$NQS_SECTOR_TAG.json" --output "$NQS_SECTOR_OUTPUT"
    done
fi
