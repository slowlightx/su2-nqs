#!/usr/bin/env bash
# New isolated workspace; never patch a source file beneath an active job.
set -euo pipefail
cd /home/yx789/nqs-triangle-quality-20261017
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=True
# --export=ALL can inherit JAX_PLATFORMS=cpu from a CPU-only submitter.
# JAX_PLATFORM_NAME=gpu alone does not override that backend allowlist.
export JAX_PLATFORMS=cuda
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MPLCONFIGDIR="$PWD/.matplotlib"
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_BASE=projects/clebsch_tree_nqs/configs/remote/triangle_quality_20261017
if [[ "${NQS_KIND:?registered family}" == backend ]]; then
    export NETKET_SHARDING=0
    "$NQS_PY" - <<'PY'
import json, os, socket
from pathlib import Path
import jax
import jax.numpy as jnp
from su2_nqs.frontier.artifacts import atomic_json, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint
assert os.environ['JAX_PLATFORMS']=='cuda' and jax.default_backend()=='gpu'
assert all(d.platform=='gpu' for d in jax.local_devices()) and len(jax.local_devices())==1
assert jax.jit(lambda x:jnp.vdot(x,x))(jnp.ones(32,jnp.float64)).item()==32
b=Path('projects/clebsch_tree_nqs/configs/remote/triangle_quality_20261017')
parents=[]
for path in sorted(b.glob('spatial*_plan.json')):
    config=json.loads((b/(path.stem.removesuffix('_plan')+'.json')).read_text())
    request=config['repair_parameters_from']; p=Path(request['directory'])/request['checkpoint_path']
    assert file_sha256(p)==request['checkpoint_sha256']
    saved=load_checkpoint(p)
    assert pytree_sha256(saved['params']['model'])==request['params_sha256']
    parents.append(dict(case=path.stem,exact_parent=True))
from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_schedule import materialize_stage
completed_clebsch=[]
for path in sorted(b.glob('clebsch_*.json')):
    plan=json.loads(path.read_text()); completed=Path(plan['parent']['directory'])/'results.json'
    if completed.exists() and json.loads(completed.read_text()).get('status')=='fixed_steps_complete':
        state=materialize_stage(plan,0)
        completed_clebsch.append(dict(case=path.stem,step=state['arity_endpoint_parent']['completed_steps'],
            checkpoint_sha256=state['arity_endpoint_parent']['checkpoint_sha256'],
            optimizer_sampler_RNG_and_independent_audit_verified=True))
result=dict(status='passed',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),
    backend=jax.default_backend(),devices=[str(x) for x in jax.devices()],
    CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),JAX_PLATFORMS=os.environ['JAX_PLATFORMS'],
    parent_replay=parents,completed_Clebsch_parents=completed_clebsch,new_training_steps=0,new_H_calls=0)
atomic_json(b/'device_preflight.json',result)
print(json.dumps(result),flush=True)
PY
elif [[ "$NQS_KIND" == clebsch ]]; then
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
