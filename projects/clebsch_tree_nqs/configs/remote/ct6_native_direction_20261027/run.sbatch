#!/usr/bin/env bash
#SBATCH --job-name=ct6_W_geometry_j0125
#SBATCH --partition=kim
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/yx789/nqs-ct6-native-direction-20261027/logs/%x-%j.out
#SBATCH --error=/home/yx789/nqs-ct6-native-direction-20261027/logs/%x-%j.err
set -euo pipefail
cd /home/yx789/nqs-ct6-native-direction-20261027
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$PY" - <<'PY'
import hashlib,json,os,pathlib
from datetime import datetime,timedelta,timezone
folder=pathlib.Path('projects/clebsch_tree_nqs/configs/remote/ct6_native_direction_20261027')
manifest=json.loads((folder/'source_manifest.json').read_text())
if hashlib.sha256((folder/'source_manifest.json').read_bytes()).hexdigest()!=os.environ['NQS_MANIFEST_SHA256']:
    raise ValueError('deployment source manifest changed')
for path,digest in manifest.items():
    if hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()!=digest:
        raise ValueError('sealed deployment changed: '+path)
plan=json.loads((folder/'j0125.json').read_text())
ledger=plan['resource_ledger']
if ledger['conservative_sum_A100_hours']>ledger['total_ceiling_A100_hours']:
    raise ValueError('original CT6 budget exceeded')
if datetime.now(timezone.utc)+timedelta(hours=4)>datetime.fromisoformat(plan['campaign_deadline_utc']):
    raise ValueError('original CT6 campaign deadline would be exceeded')
print('sealed isolated all-W calibration; any compatible Kim A100 node; no production writes',flush=True)
PY
exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.ct6_native_direction_calibration \
  --plan projects/clebsch_tree_nqs/configs/remote/ct6_native_direction_20261027/j0125.json \
  --output /home/yx789/nqs-wavefunctions/nqs/ct6_native_direction_20261027/j0125
