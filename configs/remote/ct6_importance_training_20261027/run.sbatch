#!/usr/bin/env bash
#SBATCH --job-name=ct6_IID_SR_j0125
#SBATCH --partition=kim
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=/home/yx789/nqs-ct6-importance-training-20261027/logs/%x-%j.out
#SBATCH --error=/home/yx789/nqs-ct6-importance-training-20261027/logs/%x-%j.err
set -euo pipefail
NQS_ALLOCATION_STARTED_EPOCH="$(date +%s)"
export SLURM_JOB_END_TIME="$((NQS_ALLOCATION_STARTED_EPOCH+50400))"
cd /home/yx789/nqs-ct6-importance-training-20261027
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$PY" - <<'PY'
from datetime import datetime,timedelta,timezone
import hashlib,json,os,pathlib
folder=pathlib.Path('configs/remote/ct6_importance_training_20261027')
manifest=folder/'source_manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=os.environ['NQS_MANIFEST_SHA256']:
    raise ValueError('deployment manifest identity changed')
for name,digest in json.loads(manifest.read_text()).items():
    if hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()!=digest:
        raise ValueError('sealed deployment file changed: '+name)
if 222.345+14>240:
    raise ValueError('original conservative campaign ceiling exceeded')
if datetime.now(timezone.utc)+timedelta(hours=14)>datetime(2026,10,28,20,44,tzinfo=timezone.utc):
    raise ValueError('allocation cannot finish before original campaign deadline')
print('Sealed same-width IID amplitude-SR training; original476 parent; at most16 updates.',flush=True)
PY
for phase in prepare replay train; do
    "$PY" -u -m projects.clebsch_tree_nqs.workflows.ct6_importance_amplitude_continuation \
      --plan configs/remote/ct6_importance_training_20261027/j0125.json \
      --output /home/yx789/nqs-wavefunctions/nqs/ct6_importance_training_20261027/j0125 \
      --phase "$phase"
done
