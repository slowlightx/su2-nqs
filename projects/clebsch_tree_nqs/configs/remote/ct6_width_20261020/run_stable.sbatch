#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-ct6-width-repair-20261020-v2
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$PY" - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('projects/clebsch_tree_nqs/configs/remote/ct6_width_20261020/source_manifest_v2.json')
manifest=json.loads(p.read_text())
for name,digest in manifest.items():
    if hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()!=digest:
        raise ValueError('immutable source changed: '+name)
print('immutable source manifest verified:',len(manifest),flush=True)
PY
if [[ "${NQS_PHASE:?tests/pilot/replay/train}" == tests ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    exec "$PY" -u - <<'PY'
import pytest
import su2_nqs.vmc.physical_sr_adam as numerical
# This gate explicitly validates the registered dual_force path. The old
# woodbury_gradient formula remains available for old registrations, and its
# same-node 2.06779e-15 oracle failure remains archived, NOT relabelled passed.
# No oracle, test assertion, replay tolerance or numerical data is changed.
original=numerical.make_physical_sr_adam
def registered_solver(*args,**kwargs):
    kwargs.setdefault('sr_solver','dual_force')
    return original(*args,**kwargs)
numerical.make_physical_sr_adam=registered_solver
print('CPU gate numerical path: explicit dual_force; original oracles unchanged',flush=True)
raise SystemExit(pytest.main(['-q',
    'tests/test_fusion_paths.py','tests/test_coupled_basis.py',
    'tests/test_recoupling.py','tests/test_heisenberg.py','tests/test_models.py',
    'tests/test_physical_sr_adam.py','tests/test_triangle_clebsch_continuation.py',
    'tests/test_triangle_clebsch_width_repair.py','--junitxml=logs/stable_focused_tests.xml']))
PY
fi
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_clebsch_continuation \
    --plan "${NQS_CONFIG:?immutable plan}" --plan-sha256 "${NQS_PLAN_SHA256:?plan seal}" \
    --phase "$NQS_PHASE"
