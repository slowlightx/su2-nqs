#!/usr/bin/env bash
#SBATCH --job-name=ct6_W_geometry_resume
#SBATCH --partition=kim
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/yx789/nqs-ct6-native-direction-resume-20261027/logs/%x-%j.out
#SBATCH --error=/home/yx789/nqs-ct6-native-direction-resume-20261027/logs/%x-%j.err
set -euo pipefail
cd /home/yx789/nqs-ct6-native-direction-resume-20261027
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$PY" - <<'PY'
import hashlib,json,os,pathlib
from datetime import datetime,timedelta,timezone
folder=pathlib.Path('projects/clebsch_tree_nqs/configs/remote/ct6_native_direction_20261027')
manifest_path=folder/'resume_source_manifest.json'
if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=os.environ['NQS_MANIFEST_SHA256']:
    raise ValueError('resume deployment manifest changed')
for path,digest in json.loads(manifest_path.read_text()).items():
    if hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()!=digest:
        raise ValueError('sealed resume source changed: '+path)
inputs=folder/'resume_inputs.json'
if hashlib.sha256(inputs.read_bytes()).hexdigest()!=os.environ['NQS_INPUT_MANIFEST_SHA256']:
    raise ValueError('frozen prior-input manifest changed')
if 219.345+3>240:
    raise ValueError('original conservative campaign budget exceeded')
if datetime.now(timezone.utc)+timedelta(hours=3)>datetime(2026,10,28,20,44,tzinfo=timezone.utc):
    raise ValueError('original campaign deadline would be exceeded')
print('sealed resume from same direction and calibration rows; no refit or new calibration draws',flush=True)
PY
exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.ct6_native_direction_calibration \
  --resume-from /home/yx789/nqs-wavefunctions/nqs/ct6_native_direction_20261027/j0125 \
  --resume-manifest projects/clebsch_tree_nqs/configs/remote/ct6_native_direction_20261027/resume_inputs.json \
  --wall-cap-seconds 10800 \
  --output /home/yx789/nqs-wavefunctions/nqs/ct6_native_direction_20261027/j0125_resume1
