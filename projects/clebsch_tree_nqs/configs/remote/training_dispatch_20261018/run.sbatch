#!/usr/bin/env bash
# Nonnumerical dispatcher over the existing sealed training implementations.
set -euo pipefail
cd /home/yx789/nqs-training-dispatch-20261018
export PYTHONPATH="${NQS_SOURCE:?frozen numerical workspace}"
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export NETKET_SHARDING=1
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_BASE="$PWD/projects/clebsch_tree_nqs/configs/remote/training_dispatch_20261018"
export NQS_CONFIG="$NQS_BASE/${NQS_CASE:?case}.json"
NQS_CHILD_PID=""
NQS_STOP_REQUESTED=0
trap 'NQS_STOP_REQUESTED=1; if [[ -n "$NQS_CHILD_PID" ]]; then kill -USR1 "$NQS_CHILD_PID" 2>/dev/null || true; fi' USR1 TERM
run_child() {
    if [[ "$NQS_STOP_REQUESTED" == 1 ]]; then exit 0; fi
    "$@" &
    NQS_CHILD_PID=$!
    local nqs_child_status=0
    while true; do
        if wait "$NQS_CHILD_PID"; then nqs_child_status=0; break; else nqs_child_status=$?; fi
        if ! kill -0 "$NQS_CHILD_PID" 2>/dev/null; then
            if wait "$NQS_CHILD_PID"; then nqs_child_status=0; else nqs_child_status=$?; fi
            break
        fi
    done
    NQS_CHILD_PID=""
    if [[ "$NQS_STOP_REQUESTED" == 1 ]]; then exit "$nqs_child_status"; fi
    return "$nqs_child_status"
}
case "${NQS_PHASE:?phase}" in
pilot|replay)
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket --config "$NQS_CONFIG" \
        ${NQS_RESUME_ARG:-} --until-step "${NQS_BOUNDARY:?boundary}" --session-seconds 1900 \
        --require-endpoint --require-sample-partitions "${NQS_DEVICES:?devices}"
    ;;
train)
    "$NQS_PY" - <<'PY'
import json, os
from pathlib import Path
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import read_sealed
s=json.loads(Path(os.environ['NQS_CONFIG']).read_text()); p=read_sealed(Path(s['output'])/'progress.json')
rows=p['history'][-2:]; trust=s['netket']['sampled_trust']; n=int(os.environ['NQS_DEVICES'])
assert p['completed_steps']==int(os.environ['NQS_BOUNDARY']) and len(rows)==2
assert all(r['update_accepted'] and abs(r['energy_before_update_per_site']-float(os.environ['NQS_PARENT_ENERGY']))<.02
    and r['variance_before_update_per_site']<.5 and r['gpu_peak_bytes']<70*1024**3
    and r['sampled_fs_sine']<=trust['maximum_fs_sine']*(1+1e-10)
    and r['reweighting_ess_fraction']>=trust['minimum_ess_fraction']
    and r['device_count']==n and r['unique_local_sample_partitions']==n for r in rows), 'isolated reload guard failed'
PY
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket --config "$NQS_CONFIG" \
        --resume --session-seconds "${NQS_SESSION:?session}" --require-endpoint --require-sample-partitions "$NQS_DEVICES"
    ;;
audit)
    export NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality audit-energy \
        --config "$NQS_CONFIG" --steps "$NQS_TERMINAL" --stream-index "$SLURM_ARRAY_TASK_ID"
    ;;
aggregate)
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality audit-energy \
        --config "$NQS_CONFIG" --steps "$NQS_TERMINAL" --aggregate-only
    ;;
square_train)
    export NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_radius_compare \
        --plan "$NQS_BASE/square_plan.json" --plan-sha256 "$NQS_PLAN_SHA256" --arm direct_SR4096
    ;;
square_cost)
    export NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_sampled --config "$NQS_CONFIG"
    ;;
square_j0_train)
    export NETKET_SHARDING=0
    "$NQS_PY" - <<'PY'
import json, os
from pathlib import Path
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
p=Path(os.environ['NQS_CONFIG']); template=p.with_name('square8_j0_template.json')
s=json.loads(template.read_text()); root=Path(s['cost_parent']['directory'])
gate=json.loads((root/'results.json').read_text())
assert gate['status']=='passed' and gate['config']['j2_over_j1']==0
s['cost_parent'].update(checkpoint_sha256=file_sha256(root/'checkpoint.pkl'), registration_sha256=gate['registration_sha256'])
if p.exists():
    assert json.loads(p.read_text())==s, 'training configuration changed'
else:
    atomic_json(p,s)
PY
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_training --config "$NQS_CONFIG" \
        --until-step "$NQS_BOUNDARY" --session-seconds "$NQS_SESSION"
    ;;
square_audit)
    export NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_clebsch_tree_audit --config "$NQS_CONFIG"
    ;;
attention)
    export NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_dictionary_campaign \
        --config "$NQS_CONFIG" --seed-index 0 ${NQS_ATTENTION_ARGS:-}
    ;;
n144_gate)
    export NETKET_SHARDING=0
    run_child "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_native_gate --config "$NQS_CONFIG"
    run_child "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_native_gate --config "$NQS_CONFIG" --replay-gate
    ;;
n144_account)
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu NETKET_SHARDING=0
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_native_gate --config "$NQS_CONFIG" --collect-job "$NQS_GATE_JOB"
    ;;
n144_train)
    export NETKET_SHARDING=0
    "$NQS_PY" - <<'PY'
import json, os
from pathlib import Path
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from projects.clebsch_tree_nqs.workflows.triangle_n144_train import checked_training_gate
p=Path(os.environ['NQS_CONFIG']); s=json.loads(p.with_name(p.stem.replace('_train2','_template')+'.json').read_text())
root=Path(s['native_gate']['directory'])
for key, filename in [('results','results.json'), ('registration','registration.json'), ('process_replay','process_replay.json'), ('accounting','accounting.json')]:
    s['native_gate'][key+'_sha256']=file_sha256(root/filename)
g=json.loads((root/'results.json').read_text()); t=g['timings']
# Conservative measured first-call costs; all training/audits/three replays.
mh_centers=2*8*((1440+4*144)+(144+4*144))+3*8*(720+4*144)+6*8*(144+4*144)
forecast=1200+mh_centers/(8*(720+4*144))*t['native_MH_initial_burn_and_record_pool']
forecast+=416/32*t['complete_H_entire_native_pool']
forecast+=10*max(t['full_original_parameter_SR_proposal_binary_truncated'],t['full_original_parameter_SR_proposal_ternary_joint'])
s['budget']['forecast_seconds']=forecast
checked_training_gate(s,root=Path(os.environ['NQS_SOURCE']))
if p.exists():
    assert json.loads(p.read_text())==s, 'measured release changed'
else:
    atomic_json(p,s)
print('Measured bounded forecast seconds',forecast,flush=True)
PY
    run_child "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_train --config "$NQS_CONFIG" --until-step 1
    run_child "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_train --config "$NQS_CONFIG" --resume
    run_child "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_train --config "$NQS_CONFIG" --replay-completed
    ;;
*) exit 2 ;;
esac
