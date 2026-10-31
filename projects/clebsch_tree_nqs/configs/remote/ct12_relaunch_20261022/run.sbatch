#!/usr/bin/env bash
set -euo pipefail
cd /home/yx789/nqs-ct12-relaunch-20261022v2
export PYTHONPATH="$PWD" JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$PY" - <<'PY'
import hashlib,json,pathlib
path=pathlib.Path('projects/clebsch_tree_nqs/configs/remote/ct12_relaunch_20261022/source_manifest.json')
manifest=json.loads(path.read_text())
for name,expected in manifest.items():
    if hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()!=expected:
        raise ValueError('frozen source/config changed:'+name)
print('source/config manifest verified',len(manifest),flush=True)
PY
if [[ "${NQS_PHASE:?phase}" == tests ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    export XLA_FLAGS=--xla_cpu_max_isa=AVX2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$PY" -m pytest -q tests/test_parameter_block_sr.py tests/test_triangle_dictionary_vmc_independent.py tests/test_triangle_n144_continue.py tests/test_physical_sr_adam.py
fi
if [[ "$NQS_PHASE" == parents ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    export XLA_FLAGS=--xla_cpu_max_isa=AVX2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$PY" - <<'PY'
import json,pathlib
from projects.clebsch_tree_nqs.workflows.triangle_n144_continue import validate_plan,checked_parent,initialize_completed_parent
from su2_nqs.frontier.artifacts import pytree_sha256
directory=pathlib.Path('projects/clebsch_tree_nqs/configs/remote/ct12_relaunch_20261022')
for path in sorted(directory.glob('j*.json')):
    spec=json.loads(path.read_text())
    if not spec.get('training_release'):continue
    validate_plan(spec)
    saved,_=checked_parent(spec);stage,note=initialize_completed_parent(spec,saved)
    assert pytree_sha256(stage['params'])==pytree_sha256(saved['params'])
    assert pytree_sha256(stage['optimizer_state'])==pytree_sha256(saved['optimizer_state'])
    assert stage['metadata']['optimizer_base_step']==18
    assert not stage['metadata']['new_history']
    print(json.dumps(dict(config=str(path),parent_files_and_full_checkpoint_verified=True,initialization=note)),flush=True)
PY
fi
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
exec "$PY" -u -m projects.clebsch_tree_nqs.workflows.triangle_n144_continue --config "${NQS_CONFIG:?config}" --phase "$NQS_PHASE"
