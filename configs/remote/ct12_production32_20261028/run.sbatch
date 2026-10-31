#!/usr/bin/env bash
#SBATCH --account=kim
#SBATCH --qos=normal
#SBATCH --partition=aimi
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --no-requeue
#SBATCH --signal=USR1@600
#SBATCH --job-name=CT12_j0125_prod32
set -euo pipefail

# One explicitly authorized production segment; no host binding or auto-submit.
NQS_CT12_ROOT=/home/yx789/nqs-ct12-production32-20261028
NQS_CT12_PYTHON=/home/yx789/nqs-b200-runtime-20261028/venv/bin/python
NQS_CT12_CONFIG=configs/remote/ct12_production32_20261028
NQS_CT12_OUTPUT=/home/yx789/nqs-wavefunctions/nqs/ct12_production32_20261028/j0125
NQS_CT12_STARTED="$(date +%s)"
export SLURM_JOB_END_TIME="${SLURM_JOB_END_TIME:-$((NQS_CT12_STARTED+57600))}"
cd "$NQS_CT12_ROOT"
export PYTHONPATH="$NQS_CT12_ROOT" PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export TF_CPP_MIN_LOG_LEVEL=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.80
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# This sole path is the SHA-sealed official forward-compat DRIVER layer.
# BLAS remains the separately pinned pip wheel, verified in each live process.
export LD_LIBRARY_PATH=/home/yx789/nqs-b200-forward-compat-20261028/lib
unset LD_PRELOAD JAX_SKIP_CUDA_CONSTRAINTS_CHECK PYTHONWARNINGS JAX_DISABLE_JIT
unset XLA_FLAGS XLA_PYTHON_CLIENT_MEM_FRACTION
unset CUBLAS_EMULATE_DOUBLE_PRECISION CUBLAS_EMULATE_SINGLE_PRECISION
export JAX_COMPILATION_CACHE_DIR="$NQS_CT12_ROOT/cache/B200-CUDA13-compat-j0125"

"$NQS_CT12_PYTHON" - <<'PY'
import hashlib,json,os,pathlib
root=pathlib.Path.cwd();directory=root/'configs/remote/ct12_production32_20261028'
manifest=directory/'source_manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=os.environ.get('NQS_CT12_MANIFEST_SHA256'):
    raise ValueError('external production deployment SHA256 required')
for name,digest in json.loads(manifest.read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('production source/config/fixture changed: '+name)
p=json.loads((directory/'j0125.json').read_text());a=p['software_acceptance']
certificate=pathlib.Path(a['directory'])/'gpu/gpu_check_result.json'
result=json.loads(certificate.read_text())
if result.get('passed') is not True or result.get('status')!='bounded_software_acceptance_complete':
    raise ValueError('software acceptance failed or incomplete; NO training permitted')
print('Sealed J2=.125: unchanged1495080 native parameters,32 updates,1024 fit/step,4096 fresh audit.',flush=True)
PY

test ! -e "$NQS_CT12_OUTPUT"
mkdir -p "$NQS_CT12_OUTPUT"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$NQS_CT12_OUTPUT/allocation_gpu.csv"
for NQS_CT12_PHASE in prepare replay train audit; do
    /usr/local/slurm/current/bin/srun --ntasks=1 "$NQS_CT12_PYTHON" -u -m \
        projects.clebsch_tree_nqs.workflows.ct_bounded_followup \
        --plan "$NQS_CT12_CONFIG/j0125.json" --output "$NQS_CT12_OUTPUT/run01" \
        --phase "$NQS_CT12_PHASE"
done
"$NQS_CT12_PYTHON" - "$NQS_CT12_OUTPUT/run01" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);r=json.loads((root/'audit_result.json').read_text())
complete=(r.get('status')=='complete_fixed_prefix_independent_audit'
          and r.get('completed_updates')==32
          and r.get('terminal_audit',{}).get('records')==4096)
print(json.dumps(dict(fixed32_updates_and4096_audit_complete=complete,
                      actual_updates=r.get('completed_updates'),saved_state_preserved=True)),flush=True)
if not complete:raise SystemExit(2)
PY
