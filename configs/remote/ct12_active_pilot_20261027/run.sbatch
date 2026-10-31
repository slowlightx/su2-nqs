#!/usr/bin/env bash
#SBATCH --job-name=ct12_active_SR2
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --signal=USR1@180
#SBATCH --output=/home/yx789/nqs-ct12-active-pilot-20261027/logs/%x-%j.out
#SBATCH --error=/home/yx789/nqs-ct12-active-pilot-20261027/logs/%x-%j.err
set -euo pipefail

NQS_CT12_WORKSPACE=/home/yx789/nqs-ct12-active-pilot-20261027
NQS_CT12_PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_CT12_OUTPUT=/home/yx789/nqs-wavefunctions/nqs/ct12_active_pilot_20261027/run01
NQS_CT12_ALLOCATION_STARTED_EPOCH="$(date +%s)"
if [[ -z "${SLURM_JOB_END_TIME:-}" ]]; then
    export SLURM_JOB_END_TIME="$((NQS_CT12_ALLOCATION_STARTED_EPOCH+14400))"
fi
cd "$NQS_CT12_WORKSPACE"
export PYTHONPATH="$NQS_CT12_WORKSPACE" PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export JAX_COMPILATION_CACHE_DIR="$NQS_CT12_WORKSPACE/cache/jax"

# Only verify/install this new pilot's immutable inputs. No old campaign
# controller, deadline, hostname requirement or production checkpoint is used.
"$NQS_CT12_PYTHON" - <<'PY'
import hashlib,json,os,pathlib,shutil
root=pathlib.Path.cwd()
directory=root/'configs/remote/ct12_active_pilot_20261027'
manifest=directory/'source_manifest.json'
expected=os.environ.get('NQS_CT12_MANIFEST_SHA256')
if not expected or hashlib.sha256(manifest.read_bytes()).hexdigest()!=expected:
    raise ValueError('root must provide matching NQS_CT12_MANIFEST_SHA256 at submission')
for name,digest in json.loads(manifest.read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('new CT12 pilot deployment mismatch: '+name)
inputs=json.loads((directory/'input_transfers.json').read_text())
allowed=pathlib.Path('/home/yx789/nqs-wavefunctions/nqs/ct12_active_pilot_20261027/inputs')
for row in inputs:
    source=root/row['source_relative']
    target=pathlib.Path(row['remote_destination'])
    if target.parent!=allowed or hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:
        raise ValueError('input identity or precise destination mismatch')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:
            raise FileExistsError('refusing to overwrite another frozen input: '+str(target))
    else:
        temporary=target.with_suffix(target.suffix+'.copying')
        shutil.copy2(source,temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=row['sha256']:
            raise ArithmeticError('input copy failed byte-level verification')
        temporary.replace(target)
print('Independent CT12 budget: one A10080GB allocation, at most4GPU-hours. No CT6 budget or job changed.',flush=True)
PY

mkdir -p "$NQS_CT12_OUTPUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$NQS_CT12_OUTPUT/allocation_gpu.csv"
for NQS_CT12_PHASE in prepare replay_train audit; do
    # Three actual process boundaries. The default USR1 signal targets srun
    # job steps; Python preserves its last committed prefix at stage boundaries.
    /usr/local/slurm/current/bin/srun --ntasks=1 "$NQS_CT12_PYTHON" -u -m \
        projects.clebsch_tree_nqs.workflows.triangle_active_two_step_pilot \
        --plan configs/remote/ct12_active_pilot_20261027/plan.json \
        --output "$NQS_CT12_OUTPUT" --phase "$NQS_CT12_PHASE"
done
