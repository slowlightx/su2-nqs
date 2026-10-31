#!/usr/bin/env bash
# Fixed128-step square8 size-transfer pilot, isolated reload and fresh audit.
set -euo pipefail
cd /home/yx789/nqs-square8-followup-20261019
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false NETKET_SHARDING=0
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
export NQS_CONFIG="$PWD/projects/clebsch_tree_nqs/configs/remote/square8_followup_20261019/${NQS_CASE:?j0 or j05}.json"
case "${NQS_PHASE:?pilot replay train audit aggregate}" in
pilot)
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$NQS_CONFIG" --until-step 2 --session-seconds 2400 \
        --require-endpoint --require-sample-partitions 1
    ;;
replay)
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$NQS_CONFIG" --resume --until-step 4 --session-seconds 2400 \
        --require-endpoint --require-sample-partitions 1
    ;;
train)
    "$NQS_PY" - <<'PY'
import json, os, math
from pathlib import Path
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import read_sealed
s=json.loads(Path(os.environ['NQS_CONFIG']).read_text())
p=read_sealed(Path(s['output'])/'progress.json'); rows=p['history'][-2:]
assert p['completed_steps']==4 and len(rows)==2
trust=s['netket']['sampled_trust']; j2=s['j2_over_j1']
for r in rows:
    assert all(math.isfinite(r[k]) for k in ('energy_before_update_per_site','variance_before_update_per_site','sampled_fs_sine'))
    assert r['update_accepted'] and r['variance_before_update_per_site']<1.
    assert -1.5*(1+j2)-.1 < r['energy_before_update_per_site'] < .5*(1+j2)+.1
    assert r['gpu_peak_bytes']<70*1024**3
    assert r['sampled_fs_sine']<=trust['maximum_fs_sine']*(1+1e-10)
    assert r['reweighting_ess_fraction']>=trust['minimum_ess_fraction']
    assert r['device_count']==r['unique_local_sample_partitions']==1
forecast=1200+124*max(r['seconds_including_compile'] for r in rows)
assert forecast < s['maximum_training_seconds'], 'measured N64 forecast exceeds registered active budget'
print('same-model isolated process reload passed; forecast seconds',forecast,flush=True)
PY
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$NQS_CONFIG" --resume --session-seconds 28800 \
        --require-endpoint --require-sample-partitions 1
    ;;
audit)
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality audit-energy \
        --config "$NQS_CONFIG" --steps 128 --stream-index "${SLURM_ARRAY_TASK_ID:?audit stream}"
    ;;
aggregate)
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality audit-energy \
        --config "$NQS_CONFIG" --steps 128 --aggregate-only
    ;;
*) exit 2 ;;
esac
