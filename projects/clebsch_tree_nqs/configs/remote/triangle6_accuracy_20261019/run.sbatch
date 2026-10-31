#!/usr/bin/env bash
# Separate capacity experiment: never alter a running producer workspace.
set -euo pipefail
cd /home/yx789/nqs-triangle6-accuracy-20261019
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false NETKET_SHARDING=1
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
export NQS_CONFIG="$PWD/projects/clebsch_tree_nqs/configs/remote/triangle6_accuracy_20261019/${NQS_CASE:?case}.json"
"$NQS_PY" - <<'PY'
import os
from pathlib import Path
from su2_nqs.frontier.provenance import file_sha256
assert file_sha256(Path(os.environ['NQS_CONFIG'])) == os.environ['NQS_CONFIG_SHA256']
PY
case "${NQS_PHASE:?phase}" in
pilot)
    # Check real parent amplitudes AND complete physical H, not only array shapes.
    "$NQS_PY" - <<'PY'
import json, os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from su2_nqs.models.literature_spatial_vit import LiteratureSpatialViTConfig
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.benchmarks.spatial_vit_sampling import make_full_heisenberg_ratios
from projects.clebsch_tree_nqs.workflows.spin import _groups, _bonds
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import initial_model, validate_spec, model_log, read_sealed, sealed_json
s=json.loads(Path(os.environ['NQS_CONFIG']).read_text())
p, info=initial_model(s)
declaration=s['repair_parameters_from']; d=Path(declaration['directory'])
old=read_sealed(d/'registration.json')['config']
saved=load_checkpoint(d/declaration['checkpoint_path']); parent=saved['params']['model']
rows=jnp.asarray(((np.asarray(saved['params']['netket_sampler_state'].σ)[:4]+1)/2).astype(np.int32))
group=_groups('triangular',6,6)['gamma_a1']
before, n0=model_log(old,LiteratureSpatialViTConfig(**old['model']),group)
after, n1=model_log(s,validate_spec(s),group)
a=jax.jit(jax.vmap(before,in_axes=(None,0)))(parent,rows)
b=jax.jit(jax.vmap(after,in_axes=(None,0)))(p,rows)
amplitude_error=float(jnp.max(jnp.abs(jnp.exp(b-a)-1)))
bonds=_bonds('triangular',6,6,s['j2_over_j1'])
h0=make_full_heisenberg_ratios(before,bonds)(parent,rows)
h1=make_full_heisenberg_ratios(after,bonds)(p,rows)
H_error=float(jnp.max(jnp.abs(h1-h0)))
assert n0==n1==216 and amplitude_error<2e-11 and H_error<2e-9
assert info['parent_parameter_count']==99904 and info['expanded_parameter_count']==174400
sealed_json(Path(os.environ['NQS_CONFIG']).with_suffix('.preflight.json'),dict(
    amplitude_ratio_max_error=amplitude_error,full_H_ratio_max_error=H_error,images=n1,
    complete_H_bonds=len(bonds),parameter_counts=[99904,174400],parent=info,
    rows_from_all_chain_parent=True,energy_or_phase_win_claimed=False))
print('GPU parent-state/full-H inclusion verified',amplitude_error,H_error,flush=True)
PY
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$NQS_CONFIG" --until-step 2 --session-seconds 2700 --require-endpoint
    ;;
restart)
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$NQS_CONFIG" --resume --until-step 4 --session-seconds 3000 --require-endpoint
    ;;
train1|train2)
    "$NQS_PY" - <<'PY'
import json, os, math
from pathlib import Path
import numpy as np
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import read_sealed
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.frontier.artifacts import pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
s=json.loads(Path(os.environ['NQS_CONFIG']).read_text()); d=Path(s['output'])
p=read_sealed(d/'progress.json'); saved=load_checkpoint(d/'checkpoint.pkl')
step=4 if os.environ['NQS_PHASE']=='train1' else 300
assert p['completed_steps']==step==saved['optimizer_state']['step']==saved['metadata']['completed_steps']
assert p['checkpoint_sha256']==file_sha256(d/'checkpoint.pkl')
assert p['final_params_sha256']==pytree_sha256(saved['params']['model'])
assert saved['metadata']['registration_sha256']==p['registration_sha256']
spins=np.asarray(saved['params']['netket_sampler_state'].σ)
assert spins.shape==(128,36) and np.isin(spins,[-1,1]).all() and (spins.sum(-1)==0).all()
assert p['parameter_count']==174400 and p['training_full_H_calls']==4096*step
rows=p['history'][-2:]; trust=s['netket']['sampled_trust']
for r in rows:
    assert r['update_accepted'] and 0<r['sampled_fs_sine']<=trust['maximum_fs_sine']*(1+1e-10)
    assert r['reweighting_ess_fraction']>=trust['minimum_ess_fraction']
    assert r['device_count']==r['unique_local_sample_partitions']==1
    assert r['images']==216 and r['projection']=='gamma_a1' and r['full_H_calls']==4096
    assert 0<r['gpu_peak_bytes']<70*1024**3
    assert math.isfinite(r['energy_before_update_per_site']) and 0<=r['variance_before_update_per_site']<.5
remaining=(600-step)*max(r['seconds_including_compile'] for r in rows)*1.15
assert remaining<=s['maximum_training_seconds']-p['active_training_seconds'], 'Measured cost exceeds fixed budget; keep checkpoint'
print('Physical updates, restore identity, memory and throughput verified; forecast remaining',remaining,flush=True)
PY
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
