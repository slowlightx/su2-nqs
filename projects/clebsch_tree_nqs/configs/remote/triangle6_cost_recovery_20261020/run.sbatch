#!/usr/bin/env bash
# Scheduling-only recovery; old code, registrations and checkpoints are immutable.
set -euo pipefail
cd /home/yx789/nqs-triangle6-cost-recovery-20261020
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false NETKET_SHARDING=1
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
export NQS_CONFIG="$PWD/projects/clebsch_tree_nqs/configs/remote/triangle6_cost_recovery_20261020/${NQS_CASE:?case}.json"
"$NQS_PY" - <<'PY'
import os
from pathlib import Path
from su2_nqs.frontier.provenance import file_sha256
assert file_sha256(Path(os.environ['NQS_CONFIG'])) == os.environ['NQS_CONFIG_SHA256']
PY

check_cost() {
    "$NQS_PY" - <<'PY'
import json, math, os
from pathlib import Path
import numpy as np
from projects.clebsch_tree_nqs.workflows.spatial_vit_cost import forecast_training_cost
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import read_sealed, sealed_json
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.frontier.artifacts import pytree_sha256
spec=json.loads(Path(os.environ['NQS_CONFIG']).read_text()); out=Path(spec['output'])
p=read_sealed(out/'progress.json'); saved=load_checkpoint(out/'checkpoint.pkl')
expected=8 if os.environ['NQS_PHASE'] in ('warm','train1') else 300
assert p['completed_steps']==expected==saved['optimizer_state']['step']==saved['metadata']['completed_steps']
assert p['checkpoint_sha256']==file_sha256(out/'checkpoint.pkl')
assert p['final_params_sha256']==pytree_sha256(saved['params']['model'])
assert p['registration_sha256']==saved['metadata']['registration_sha256']
assert p['parameter_count']==174400 and p['training_full_H_calls']==4096*expected
spins=np.asarray(saved['params']['netket_sampler_state'].σ)
assert spins.shape==(128,36) and np.isin(spins,[-1,1]).all() and (spins.sum(-1)==0).all()
trust=spec['netket']['sampled_trust']
for row in p['history'][-3:]:
    assert row['update_accepted'] and 0<row['sampled_fs_sine']<=trust['maximum_fs_sine']*(1+1e-10)
    assert row['reweighting_ess_fraction']>=trust['minimum_ess_fraction']
    assert row['device_count']==row['unique_local_sample_partitions']==1
    assert row['images']==216 and row['projection']=='gamma_a1' and row['full_H_calls']==4096
    assert 0<row['gpu_peak_bytes']<70*1024**3
    assert math.isfinite(row['energy_before_update_per_site']) and 0<=row['variance_before_update_per_site']<.5
starts=[1,3,5] if expected==8 else [1,3,5,9]
forecast=forecast_training_cost(p['history'], process_first_steps=starts, target_step=600,
    active_training_seconds=p['active_training_seconds'],
    maximum_training_seconds=spec['maximum_training_seconds'], remaining_processes=2 if expected==8 else 1)
forecast.update(checkpoint_sha256=p['checkpoint_sha256'],progress_sha256=p['result_sha256'],
                registration_sha256=p['registration_sha256'],unchanged_full_A1_and_physical_trust=True)
sealed_json(Path(os.environ['NQS_CONFIG']).with_suffix('.'+os.environ['NQS_PHASE']+'.cost.json'),forecast)
print(json.dumps(forecast),flush=True)
PY
}

case "${NQS_PHASE:?phase}" in
warm)
    # Four actual updates: first pays compilation, then THREE warm measurements.
    # All four count toward the original fixed600 steps and cumulative time cap.
    "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$NQS_CONFIG" --until-step 8 --session-seconds 1800 --require-endpoint
    check_cost
    ;;
train1|train2)
    check_cost
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$NQS_CONFIG" --resume --until-step "${NQS_BOUNDARY:?boundary}" \
        --session-seconds 63000 --require-endpoint
    ;;
audit)
    export NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality audit-energy \
        --config "$NQS_CONFIG" --steps 600 --stream-index "${SLURM_ARRAY_TASK_ID:?stream}"
    ;;
aggregate)
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality audit-energy \
        --config "$NQS_CONFIG" --steps 600 --aggregate-only
    ;;
*) exit 2 ;;
esac
