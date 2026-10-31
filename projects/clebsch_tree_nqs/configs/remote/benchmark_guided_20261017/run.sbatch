#!/usr/bin/env bash
# Shared dispatcher: existing trainers/readers, not a new numerical model.
set -euo pipefail
cd /home/yx789/nqs-benchmark-guided-20261017
export PYTHONPATH="$PWD"
export JAX_ENABLE_X64=True
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MPLCONFIGDIR="$PWD/.matplotlib"
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_BASE=projects/clebsch_tree_nqs/configs/remote/benchmark_guided_20261017
NQS_CONFIG="$NQS_BASE/${NQS_CASE:?registered case}.json"
export NQS_CONFIG
case "${NQS_KIND:?registered trainer}" in
triangle)
    export NETKET_SHARDING=0
    if [[ "$NQS_PHASE" == pilot ]]; then
        "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train --config "$NQS_CONFIG" --until-step 42 --session-seconds 1500
        "$NQS_PY" - <<'PY'
import json,os,shutil
from pathlib import Path
from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train import check_arity_pilot
p=Path(json.loads(Path(os.environ['NQS_CONFIG']).read_text())['output'])
check_arity_pilot(p,42)
if not (p/'pilot_checkpoint.pkl').exists():shutil.copy2(p/'checkpoint.pkl',p/'pilot_checkpoint.pkl')
PY
    else
        "$NQS_PY" - <<'PY'
import json
from pathlib import Path
from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train import check_arity_pilot
p=json.loads(Path('projects/clebsch_tree_nqs/configs/remote/benchmark_guided_20261017/campaign.json').read_text())
for t in p['tasks']:
    if t['kind']=='triangle':
        s=json.loads(Path(t['config']).read_text());check_arity_pilot(Path(s['output']),42)
PY
        exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train --config "$NQS_CONFIG" --require-pilot-step 42 --session-seconds "$NQS_SESSION_SECONDS"
    fi
    ;;
square)
    export NETKET_SHARDING=0
    if [[ "$NQS_PHASE" == pilot ]]; then
        "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_training --config "$NQS_CONFIG" --until-step 324 --session-seconds 2400
        "$NQS_PY" - <<'PY'
import json,os
from pathlib import Path
from su2_nqs.vmc.checkpoint import load_checkpoint
p=Path(json.loads(Path(os.environ['NQS_CONFIG']).read_text())['output'])
m=load_checkpoint(p/'checkpoint.pkl')['metadata']
if m['completed_steps']!=324 or not all(r['accepted'] and r['FS_sine']<=.05*(1+1e-10) for r in m['history'][-2:]):raise ValueError('square pilot did not finish safely')
PY
    else
        exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_radius_compare --plan "$NQS_BASE/square_plan.json" --plan-sha256 "$NQS_PLAN_SHA256" --arm direct_SR4096
    fi
    ;;
spatial)
    export NETKET_SHARDING=1
    if [[ "$NQS_PHASE" == pilot ]]; then
        "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket --config "$NQS_CONFIG" --until-step "${NQS_PILOT_STEP:-2}" --session-seconds 3300 --require-endpoint --require-sample-partitions "${NQS_SAMPLE_PARTITIONS:-2}"
        if [[ "$NQS_CASE" == spatial_j0_serial_continue ]]; then
            "$NQS_PY" - <<'PY'
import json,os
from pathlib import Path
s=json.loads(Path(os.environ['NQS_CONFIG']).read_text())
m=json.loads((Path(s['output'])/'progress.json').read_text())
rows=m['history'][-2:]
if m['completed_steps']!=134 or not all(abs(r['energy_before_update_per_site']+.5444488753507062)<.02 and r['variance_before_update_per_site']<1 for r in rows):
    raise ValueError('fixed-parent physical-moment anomaly guard failed; no release')
print('singleGPU full-state physical-moment guard passed')
PY
        fi
    elif [[ "$NQS_PHASE" == train ]]; then
        exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket --config "$NQS_CONFIG" --resume --session-seconds 25800 --require-endpoint --require-sample-partitions "${NQS_SAMPLE_PARTITIONS:-2}"
    elif [[ "$NQS_PHASE" == audit ]]; then
        export NETKET_SHARDING=0
        exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality audit-energy --config "$NQS_CONFIG" --steps "${NQS_TERMINAL_STEP:-32}" --stream-index "$SLURM_ARRAY_TASK_ID"
    else
        unset JAX_PLATFORM_NAME
        export JAX_PLATFORMS=cpu NETKET_SHARDING=0
        exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality audit-energy --config "$NQS_CONFIG" --steps "${NQS_TERMINAL_STEP:-32}" --aggregate-only
    fi
    ;;
readout)
    export NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality cached-plan --config "$NQS_CONFIG"
    ;;
*)
    echo 'Unregistered task kind' >&2
    exit 2
    ;;
esac
