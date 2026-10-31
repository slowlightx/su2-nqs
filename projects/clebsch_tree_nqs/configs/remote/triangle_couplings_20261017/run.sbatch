#!/usr/bin/env bash
# One reusable dispatcher for the registered coupling/arity combinations.
set -euo pipefail
cd /home/yx789/nqs-triangle-couplings-20261017
export PYTHONPATH="$PWD"
# Override a CPU-only submitter's inherited backend allowlist before Python.
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True JAX_PLATFORM_NAME=gpu NETKET_SHARDING=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
NQS_BASE=projects/clebsch_tree_nqs/configs/remote/triangle_couplings_20261017
# Array index selects a sealed physics config and representation, not new code.
read -r NQS_CASE NQS_ARM < <("$NQS_PY" - "$NQS_BASE/campaign.json" "${SLURM_ARRAY_TASK_ID:?registered task index}" <<'PY'
import json,sys
task=json.load(open(sys.argv[1]))['tasks'][int(sys.argv[2])]
print(task['case'],task['arm'])
PY
)
export NQS_CONFIG="$NQS_BASE/$NQS_CASE.json" NQS_ARM
if [[ "${NQS_PHASE:-pilot}" == pilot ]]; then
    "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train \
        --config "$NQS_CONFIG" --arm "$NQS_ARM" --until-step 2 --session-seconds 1500
    "$NQS_PY" - <<'PY'
import json,os,shutil
from pathlib import Path
from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train import check_arity_pilot,resolve_arm_spec
spec=resolve_arm_spec(json.loads(Path(os.environ['NQS_CONFIG']).read_text()),os.environ['NQS_ARM'])
output=Path(spec['output']);check_arity_pilot(output,2)
if not (output/'pilot_checkpoint.pkl').exists():
    shutil.copy2(output/'checkpoint.pkl',output/'pilot_checkpoint.pkl')
print('bounded paired-coupling pilot passed; no long job auto-released')
PY
else
    # Not submitted initially. A human-reviewed coupling PAIR must both pass.
    "$NQS_PY" - <<'PY'
import json,os
from pathlib import Path
from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train import check_arity_pilot,resolve_arm_spec
spec=json.loads(Path(os.environ['NQS_CONFIG']).read_text())
for arm in ('binary_truncated','ternary_joint'):
    selected=resolve_arm_spec(spec,arm);check_arity_pilot(Path(selected['output']),2)
PY
    exec "$NQS_PY" -u -m projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train \
        --config "$NQS_CONFIG" --arm "$NQS_ARM" --require-pilot-step 2 --session-seconds 19200
fi
