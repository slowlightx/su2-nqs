#!/usr/bin/env bash
# New isolated workspace; never patch a source file beneath an active job.
set -euo pipefail
cd /home/yx789/nqs-triangle-quality-20261017
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=True
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MPLCONFIGDIR="$PWD/.matplotlib"
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_BASE=projects/clebsch_tree_nqs/configs/remote/triangle_quality_20261017
if [[ "${NQS_KIND:?registered family}" == clebsch ]]; then
    export NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_schedule \
        --plan "$NQS_BASE/${NQS_CASE:?case}.json" --plan-sha256 "${NQS_PLAN_SHA256:?plan seal}" \
        --stage "${NQS_STAGE:?stage}" --session-seconds "${NQS_SESSION_SECONDS:?fixed cap}"
elif [[ "$NQS_KIND" == spatial ]]; then
    export NETKET_SHARDING=1
    NQS_CONFIG="$NQS_BASE/${NQS_CASE:?case}.json"
    export NQS_CONFIG
    case "${NQS_PHASE:?phase}" in
    pilot)
        "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
            --config "$NQS_CONFIG" --until-step 2 --session-seconds 2100 \
            --require-endpoint --require-sample-partitions 1
        ;;
    replay)
        "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
            --config "$NQS_CONFIG" --resume --until-step 4 --session-seconds 2100 \
            --require-endpoint --require-sample-partitions 1
        ;;
    train)
        "$NQS_PY" - <<'PY'
import json, os
from pathlib import Path
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import read_sealed
s=json.loads(Path(os.environ['NQS_CONFIG']).read_text())
p=read_sealed(Path(s['output'])/'progress.json')
last=p['history'][-2:]
if p['completed_steps']!=4 or len(last)!=2 or not all(
    r['update_accepted'] and abs(r['energy_before_update_per_site']-s['quality_reference']['parent_energy_per_site'])<.02
    and r['variance_before_update_per_site']<.5
    and r['gpu_peak_bytes']<70*1024**3
    and r['sampled_fs_sine']<=s['netket']['sampled_trust']['maximum_fs_sine']*(1+1e-10)
    and r['reweighting_ess_fraction']>=s['netket']['sampled_trust']['minimum_ess_fraction']
    for r in last):
    raise ValueError('production-size resumed pilot failed; no long training')
PY
        exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
            --config "$NQS_CONFIG" --resume --session-seconds "${NQS_SESSION_SECONDS:?fixed cap}" \
            --require-endpoint --require-sample-partitions 1
        ;;
    audit)
        export NETKET_SHARDING=0
        exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality \
            audit-energy --config "$NQS_CONFIG" --steps "${NQS_AUDIT_STEP:?frozen endpoint}" \
            --stream-index "$SLURM_ARRAY_TASK_ID"
        ;;
    aggregate)
        unset JAX_PLATFORM_NAME
        export JAX_PLATFORMS=cpu NETKET_SHARDING=0
        exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality \
            audit-energy --config "$NQS_CONFIG" --steps "${NQS_AUDIT_STEP:?frozen endpoint}" --aggregate-only
        ;;
    *) exit 2 ;;
    esac
else
    exit 2
fi
