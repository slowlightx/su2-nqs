#!/usr/bin/env bash
# Root submits each bounded phase explicitly; no self-submission or auto-chain.
set -euo pipefail
cd /home/yx789/nqs-vit-relaunch-20261022
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_BASE="$PWD/projects/clebsch_tree_nqs/configs/remote/vit_relaunch_20261022"
export NQS_CONFIG="$NQS_BASE/${NQS_KIND:?kind}_${NQS_CASE:?case}.json"

"$NQS_PY" - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('projects/clebsch_tree_nqs/configs/remote/vit_relaunch_20261022/source_manifest.json').read_text())
for name, digest in manifest.items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
        raise ValueError('frozen relaunch source/config changed: '+name)
print('frozen relaunch manifest verified', len(manifest), flush=True)
PY

if [[ "${NQS_PHASE:?phase}" == tests ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu NETKET_SHARDING=0
    "$NQS_PY" -m pytest -q -s tests/test_spatial_vit_relaunch.py tests/test_spatial_vit_netket.py tests/test_spatial_vit_training_plan.py tests/test_spatial_vit_cost.py
    # Fresh interpreter: JAX/NetKet sharding options must precede imports.
    export NETKET_SHARDING=1
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_force_host_platform_device_count=2"
    exec "$NQS_PY" -m pytest -q -s tests/test_spatial_vit_relaunch.py tests/test_spatial_vit_netket.py \
        -k 'lazy_full_A1_NTK_matches_explicit_and_replays_complete_update or NTK_direction_solves_independent_centered_complex_SR'
fi
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu NETKET_SHARDING=1

check_state() {
"$NQS_PY" - <<'PY'
import json, os
from pathlib import Path
import numpy as np
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import read_sealed
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.frontier.artifacts import pytree_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint
s = json.loads(Path(os.environ['NQS_CONFIG']).read_text()); out = Path(s['output'])
p = read_sealed(out/'progress.json'); c = load_checkpoint(out/'checkpoint.pkl')
assert p['completed_steps'] == c['metadata']['completed_steps'] == c['optimizer_state']['step']
assert p['checkpoint_sha256'] == file_sha256(out/'checkpoint.pkl')
assert p['final_params_sha256'] == pytree_sha256(c['params']['model'])
assert p['registration_sha256'] == c['metadata']['registration_sha256']
assert c['optimizer_state']['netket_version'] == '3.22.3'
spins = np.asarray(c['params']['netket_sampler_state'].σ)
n = s['model']['rows']*s['model']['columns']
assert spins.shape == (s['sampling']['chains'], n)
assert np.isin(spins, [-1, 1]).all() and (spins.sum(-1) == 0).all()
start = s.get('netket_continue_from', {}).get('completed_steps', 0)
recent = [r for r in p['history'] if r['step'] > start]
expected_count = 323392 if os.environ['NQS_KIND'] == 'vit6' else 100288
assert p['parameter_count'] == expected_count
if os.environ['NQS_KIND'] == 'vit6':
    assert s['netket']['on_the_fly'] is True and s['netket']['chunk_size_bwd'] == 1
for row in recent:
    assert row['projection'] == 'gamma_a1' and row['images'] == 216
    assert row['device_count'] == row['unique_local_sample_partitions'] == s['netket']['expected_device_count']
    assert row['samples'] == row['full_H_calls'] == s['sampling']['chains']*s['sampling']['snapshots_per_update']
    assert row['update_accepted']
    assert 0 < row['sampled_fs_sine'] <= s['netket']['sampled_trust']['maximum_fs_sine']*(1+1e-10)
    assert row['reweighting_ess_fraction'] >= s['netket']['sampled_trust']['minimum_ess_fraction']
    assert 0 < row['gpu_peak_bytes'] < 70*1024**3
    assert np.isfinite(row['energy_before_update_per_site'])
    assert 0 <= row['variance_before_update_per_site'] < .5
print('actual updates/checkpoint and complete native sampler support verified', p['completed_steps'], flush=True)
PY
}

case "$NQS_PHASE" in
pilot)
    # This constructs the registered continuation/expanded parent only once.
    "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$NQS_CONFIG" --until-step "${NQS_BOUNDARY:?boundary}" \
        --session-seconds "${NQS_SESSION_SECONDS:?session}" --require-endpoint \
        --require-sample-partitions "${NQS_DEVICES:?devices}"
    check_state ;;
restore|warm|train)
    check_state
    "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$NQS_CONFIG" --resume --until-step "${NQS_BOUNDARY:?boundary}" \
        --session-seconds "${NQS_SESSION_SECONDS:?session}" --require-endpoint \
        --require-sample-partitions "${NQS_DEVICES:?devices}"
    check_state ;;
audit)
    export NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality audit-energy \
        --config "$NQS_CONFIG" --steps "${NQS_TERMINAL:?terminal}" \
        --stream-index "${SLURM_ARRAY_TASK_ID:?stream}" ;;
aggregate)
    export NETKET_SHARDING=0 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality audit-energy \
        --config "$NQS_CONFIG" --steps "${NQS_TERMINAL:?terminal}" --aggregate-only ;;
*) exit 2 ;;
esac
