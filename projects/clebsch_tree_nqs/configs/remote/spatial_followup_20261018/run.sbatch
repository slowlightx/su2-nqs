#!/usr/bin/env bash
# Isolated nonnumerical dispatcher; original sealed producer source is read-only.
set -euo pipefail
cd /home/yx789/nqs-spatial-followup-20261018
export PYTHONPATH=/home/yx789/nqs-triangle-quality-20261017
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cuda
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MPLCONFIGDIR="$PWD/.matplotlib"
export NETKET_SHARDING=1
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_BASE="$PWD/projects/clebsch_tree_nqs/configs/remote/spatial_followup_20261018"
export NQS_CONFIG="$NQS_BASE/${NQS_CASE:?registered case}.json"
case "${NQS_PHASE:?registered phase}" in
pilot)
    "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$NQS_CONFIG" --until-step "${NQS_PILOT_STEP:?fixed pilot}" \
        --session-seconds 1500 --require-endpoint --require-sample-partitions 1
    ;;
replay)
    "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$NQS_CONFIG" --resume --until-step "${NQS_REPLAY_STEP:?fixed reload boundary}" \
        --session-seconds 1500 --require-endpoint --require-sample-partitions 1
    ;;
train)
    "$NQS_PY" - <<'PY'
import json, os
from pathlib import Path
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import read_sealed
s=json.loads(Path(os.environ['NQS_CONFIG']).read_text())
p=read_sealed(Path(s['output'])/'progress.json')
expected=int(os.environ['NQS_REPLAY_STEP'])
reference=float(os.environ['NQS_PARENT_ENERGY'])
rows=p['history'][-2:]
trust=s['netket']['sampled_trust']
if p['completed_steps']!=expected or len(rows)!=2 or not all(
    r['update_accepted'] and abs(r['energy_before_update_per_site']-reference)<.02
    and r['variance_before_update_per_site']<.5 and r['gpu_peak_bytes']<70*1024**3
    and r['sampled_fs_sine']<=trust['maximum_fs_sine']*(1+1e-10)
    and r['reweighting_ess_fraction']>=trust['minimum_ess_fraction']
    and r['unique_local_sample_partitions']==1 and r['device_count']==1
    for r in rows):
    raise ValueError('isolated process-resume pilot failed; no long training release')
print('Same-model production-size process resume passed; not an energy-win gate',flush=True)
PY
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$NQS_CONFIG" --resume --session-seconds "${NQS_SESSION_SECONDS:?bounded session}" \
        --require-endpoint --require-sample-partitions 1
    ;;
audit)
    export NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality \
        audit-energy --config "$NQS_CONFIG" --steps "${NQS_TERMINAL_STEP:?fixed terminal}" \
        --stream-index "${SLURM_ARRAY_TASK_ID:?two registered streams}"
    ;;
aggregate)
    unset JAX_PLATFORM_NAME
    export JAX_PLATFORMS=cpu NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality \
        audit-energy --config "$NQS_CONFIG" --steps "${NQS_TERMINAL_STEP:?fixed terminal}" --aggregate-only
    ;;
*) exit 2 ;;
esac
