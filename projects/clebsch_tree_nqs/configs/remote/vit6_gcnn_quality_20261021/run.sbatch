#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-vit6-gcnn-quality-20261021
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$PY" - <<'PY'
import json,pathlib,hashlib
m=json.loads(pathlib.Path('projects/clebsch_tree_nqs/configs/remote/vit6_gcnn_quality_20261021/source_manifest.json').read_text())
for p,h in m.items():
    if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h:raise ValueError('frozen source/config changed:'+p)
print('source/config manifest verified',len(m),flush=True)
PY
if [[ "${NQS_PHASE:?phase}" == tests ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu NETKET_SHARDING=0
    exec "$PY" -m pytest -q tests/test_spatial_vit_capacity.py tests/test_spatial_vit_cost.py tests/test_spatial_vit_quality.py
fi
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu NETKET_SHARDING=1
export NQS_CONFIG="$PWD/projects/clebsch_tree_nqs/configs/remote/vit6_gcnn_quality_20261021/${NQS_CASE:?case}.json"
check_state() {
"$PY" - <<'PY'
import os,json,numpy as np,jax
from pathlib import Path
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.frontier.artifacts import pytree_sha256
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import read_sealed,sealed_json
from projects.clebsch_tree_nqs.workflows.spatial_vit_cost import forecast_training_cost
s=json.loads(Path(os.environ['NQS_CONFIG']).read_text());o=Path(s['output'])
p=read_sealed(o/'progress.json');c=load_checkpoint(o/'checkpoint.pkl');n=p['completed_steps']
assert n==c['optimizer_state']['step']==c['metadata']['completed_steps']
assert p['parameter_count']==323392 and p['checkpoint_sha256']==file_sha256(o/'checkpoint.pkl')
assert p['final_params_sha256']==pytree_sha256(c['params']['model'])
assert p['registration_sha256']==c['metadata']['registration_sha256']
x=np.asarray(c['params']['netket_sampler_state'].σ)
assert x.shape==(128,36) and np.isin(x,[-1,1]).all() and (x.sum(-1)==0).all()
for h in p['history'][-3:]:
 assert h['device_count']==h['unique_local_sample_partitions']==2
 assert h['update_accepted'] and h['images']==216 and h['full_H_calls']==8192
 assert 0<h['sampled_fs_sine']<=.005*(1+1e-10) and h['reweighting_ess_fraction']>=.98
 assert 0<h['gpu_peak_bytes']<70*1024**3
 assert np.isfinite(h['energy_before_update_per_site']) and 0<=h['variance_before_update_per_site']<.5
if n>=8:
 starts=[step for step in (1,3,5,9,151) if step<=n]
 cost=forecast_training_cost(p['history'],process_first_steps=starts,target_step=300,
  active_training_seconds=p['active_training_seconds'],maximum_training_seconds=s['maximum_training_seconds'],
  remaining_processes=0 if n==300 else 2 if n==8 else 1)
 sealed_json(o/('cost_step%05d.json'%n),cost)
 print(json.dumps(cost),flush=True)
PY
}
case "$NQS_PHASE" in
pilot)
 "$PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket --config "$NQS_CONFIG" --until-step 2 --session-seconds 4800 --require-endpoint
 check_state ;;
restore|warm|train1|train2)
 check_state
 "$PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket --config "$NQS_CONFIG" --resume --until-step "${NQS_BOUNDARY:?boundary}" --session-seconds "${NQS_SESSION_SECONDS:?seconds}" --require-endpoint
 check_state ;;
audit)
 export NETKET_SHARDING=0
 exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality audit-energy --config "$NQS_CONFIG" --steps 300 --stream-index "${SLURM_ARRAY_TASK_ID:?stream}" ;;
aggregate)
 export NETKET_SHARDING=0 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
 exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality audit-energy --config "$NQS_CONFIG" --steps 300 --aggregate-only ;;
*) exit 2 ;;
esac
