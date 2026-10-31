#!/usr/bin/env bash
# Isolated, fixed-budget continuation of existing full Spatial-ViT.
set -euo pipefail
cd /home/yx789/nqs-triangle-continuation-20261019
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false NETKET_SHARDING=1
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_BASE="$PWD/projects/clebsch_tree_nqs/configs/remote/continuation_20261019/vit12"
export NQS_CONFIG="$NQS_BASE/${NQS_CASE:?case}.json"
case "${NQS_PHASE:?phase}" in
pilot|restart)
    # Pilot and restart are DIFFERENT processes with full on-disk state restore.
    # This is not advertised as an independently recomputed deterministic replay.
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$NQS_CONFIG" ${NQS_RESUME_ARG:-} --until-step "${NQS_BOUNDARY:?boundary}" \
        --session-seconds "${NQS_SESSION:?session}" --require-endpoint \
        --require-sample-partitions "${NQS_DEVICES:?devices}"
    ;;
train)
    "$NQS_PY" - <<'PY'
import json, math, os
from pathlib import Path
import numpy as np
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import read_sealed
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.frontier.artifacts import pytree_sha256
s = json.loads(Path(os.environ['NQS_CONFIG']).read_text())
p = read_sealed(Path(s['output'])/'progress.json')
checkpoint = Path(s['output'])/'checkpoint.pkl'
saved = load_checkpoint(checkpoint)
boundary = int(os.environ['NQS_BOUNDARY'])
start = 96 if os.environ['NQS_CASE'] == 'j0125' else 0
rows = [row for row in p['history'] if row['step'] > start]
trust = s['netket']['sampled_trust']; devices = int(os.environ['NQS_DEVICES'])
assert p['completed_steps'] == boundary == start+4 and len(rows) == 4
assert saved['optimizer_state']['step'] == boundary == saved['metadata']['completed_steps']
assert p['checkpoint_sha256'] == file_sha256(checkpoint)
assert p['final_params_sha256'] == pytree_sha256(saved['params']['model'])
assert p['registration_sha256'] == saved['metadata']['registration_sha256']
spins = np.asarray(saved['params']['netket_sampler_state'].σ)
assert spins.shape == (s['sampling']['chains'], 144)
assert np.isin(spins, [-1, 1]).all() and (spins.sum(-1) == 0).all()
if os.environ['NQS_CASE'] == 'j0':
    # This is an explicit weight warm-start with NEW chains/RNG/optimizer.
    assert p['initial_params_sha256'] == '6b70599252dd7a08aed1c41accf6113015e31e5a275c52fe7422496534bb5b4f'
    assert p['parent']['completed_steps'] == 164
    assert p['parent']['repair_is_new_experiment']
    assert p['parent']['sampler_reinitialized'] and p['parent']['optimizer_and_work_counters_reset']
    assert s['netket']['initialization'] == 'balanced_spin_half'
    assert s['sampling']['chains'] == 128 and p['training_full_H_calls'] == 4*4096
else:
    # Exact full-state continuation preserves the SAME global configurations,
    # RNG values and inherited work counters, not merely the parent weights.
    parent = p['continuation_parent']
    assert parent['completed_steps'] == 96 and parent['physics_and_SR_unchanged']
    assert not parent['parameters_reinitialized'] and not parent['sampler_reinitialized']
    assert parent['inherited_work_and_time_counters']
    assert s['sampling']['chains'] == 64 and p['training_full_H_calls'] == 100*4096
for row in rows:
    assert row['update_accepted'] and 0 < row['sampled_fs_sine'] <= trust['maximum_fs_sine']*(1+1e-10)
    assert row['reweighting_ess_fraction'] >= trust['minimum_ess_fraction']
    assert row['samples'] == row['full_H_calls'] == 4096
    assert row['device_count'] == devices == row['unique_local_sample_partitions']
    assert row['projection'] == 'gamma_a1' and row['images'] == 216
    assert 0 < row['gpu_peak_bytes'] < 70*1024**3
    assert math.isfinite(row['energy_before_update_per_site'])
    assert abs(row['energy_before_update_per_site']-float(os.environ['NQS_PARENT_ENERGY'])) < .02
    assert 0 <= row['variance_before_update_per_site'] < .25
assert rows[-1]['step'] == boundary
print('Four actual, finite full-A1 updates and checkpoint restore validated', flush=True)
PY
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$NQS_CONFIG" --resume --session-seconds "${NQS_SESSION:?session}" \
        --require-endpoint --require-sample-partitions "$NQS_DEVICES"
    ;;
audit)
    export NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality audit-energy \
        --config "$NQS_CONFIG" --steps "${NQS_TERMINAL:?terminal}" \
        --stream-index "${SLURM_ARRAY_TASK_ID:?audit stream}"
    ;;
aggregate)
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality audit-energy \
        --config "$NQS_CONFIG" --steps "${NQS_TERMINAL:?terminal}" --aggregate-only
    ;;
*) exit 2 ;;
esac
