#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-ct6-portable-runtime-20261026
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$PY" - <<'PY'
import hashlib,json,pathlib,os,subprocess
from datetime import datetime,timedelta,timezone
from su2_nqs.frontier.artifacts import canonical_sha256
base=pathlib.Path('projects/clebsch_tree_nqs/configs/remote')
portable=base/'ct6_portable_runtime_20261026'
source=json.loads((base/'ct6_regularized_update_20261026/source_manifest.json').read_text())
workflow='projects/clebsch_tree_nqs/workflows/triangle_clebsch_continuation.py'
for path,digest in source.items():
    expected='af053deb351a16c5030e01a02cf22d41a9c27a74c15144f41ad2b8137a716440' if path==workflow else digest
    if hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()!=expected:
        raise ValueError('frozen numerical source/config changed:'+path)
campaign=json.loads((portable/'campaign.json').read_text())
if canonical_sha256(campaign)!=os.environ['NQS_CAMPAIGN_SHA256']:
    raise ValueError('portable campaign seal changed')
for path,digest in campaign['deployment_files'].items():
    if hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()!=digest:
        raise ValueError('portable deployment source/config changed:'+path)
total=campaign['previous_conservatively_reserved_A100_hours']+campaign['previous_v8_actual_A100_hours']+campaign['retained_running_j0_A100_hours']+campaign['replacement_portable_j0125_A100_hours']
if abs(total-campaign['aggregate_reserved_A100_hours'])>1e-9 or total>campaign['CT6_total_budget_A100_hours']:
    raise RuntimeError('portable replacement exceeds existing CT6 budget')
if datetime.now(timezone.utc)+timedelta(minutes=int(os.environ['NQS_ALLOC_MINUTES']))>datetime.fromisoformat(campaign['deadline_utc']):
    raise RuntimeError('allocation would exceed original campaign deadline')
old=subprocess.run(['/usr/local/slurm/current/bin/squeue','-h','-j',campaign['superseded_pending_job_id'],'-o','%T|%r'],capture_output=True,text=True).stdout.strip()
if old!='PENDING|JobHeldUser':
    states=subprocess.run(['/usr/local/slurm/current/bin/sacct','-n','-X','-j',campaign['superseded_pending_job_id'],'--format=State','-P'],capture_output=True,text=True,check=True).stdout.splitlines()
    if old or not states or not all(state.strip().startswith('CANCELLED') for state in states):
        raise RuntimeError('superseded CT6 job is not held/cancelled; refuse duplicate writer:'+old)
print('sealed portable deployment checked; no hostname constraint; old writer held/cancelled',flush=True)
PY
if [[ "${NQS_PHASE:?phase}" == tests ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    export XLA_FLAGS=--xla_cpu_max_isa=AVX2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$PY" -m pytest -q tests/test_ct6_portable_runtime.py tests/test_ct6_gate_replay_repair.py tests/test_ct6_validation_repair_launcher.py tests/test_ct6_direction_continuation.py tests/test_ct6_evidence_guarded_continuation.py tests/test_ct6_fixed_parent_direction_oracle.py
fi
[[ "$NQS_PHASE" == run ]]
unset XLA_FLAGS
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_ct6_portable_runtime \
    --plan projects/clebsch_tree_nqs/configs/remote/ct6_regularized_update_20261026/j0125.json \
    --plan-sha256 043b7e1352d564ab45f96ce0b7946fdc8b65a2c8e14fa0a6920cbff26bf8930e \
    --portable-manifest projects/clebsch_tree_nqs/configs/remote/ct6_portable_runtime_20261026/j0125_portable.json \
    --portable-manifest-sha256 "${NQS_PORTABLE_SHA256:?seal}"
