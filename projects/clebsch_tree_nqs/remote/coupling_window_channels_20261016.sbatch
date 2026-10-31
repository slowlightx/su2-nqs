#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/coupling-window-channels-%j.out
#SBATCH --error=logs/coupling-window-channels-%j.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export PYTHONPATH="$PWD"
task_python=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
task_plan="${1:?pre-query frozen-channel plan}"
"$task_python" - "$task_plan" <<'PY'
import json, os, sys
from pathlib import Path
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import read_sealed
from projects.clebsch_tree_nqs.workflows.spin_chirality_covariance import validate_frozen_energy_audit
request_path = Path(sys.argv[1])
if file_sha256(request_path) != os.environ['NQS_READOUT_PLAN_SHA256']:
    raise ValueError('the original pre-query frozen-channel plan changed')
request = json.loads(request_path.read_text())
template_path = Path(request['template'])
if file_sha256(template_path) != request['template_sha256']:
    raise ValueError('the frozen region/operator/orbit template changed')
base = json.loads(template_path.read_text())
parent = Path(request['parent'])
registration = read_sealed(parent/'registration.json')
training = read_sealed(parent/'training.json')
if (registration['result_sha256'] != request['parent_registration_sha256']
        or training['status'] != 'fixed_steps_complete'
        or training['completed_steps'] != request['step']):
    raise ValueError('the requested fixed terminal training state is not complete')
audit_path = parent/'audits'/f"step{request['step']:05d}"/'energy.json'
audit = read_sealed(audit_path)
checkpoint = parent/'quality'/f"step{request['step']:05d}.pkl"
if (file_sha256(checkpoint) != audit['checkpoint_sha256']
        or audit['params_sha256'] != training['final_params_sha256']):
    raise ValueError('terminal parameter/checkpoint identity changed')
base.update(parent=str(parent), step=request['step'], checkpoint_sha256=audit['checkpoint_sha256'],
    energy_audit_filename='energy.json', name=request['name'],
    output=str(Path(request['output'])/'replica'),
    streams={f"stream{s['seed']}.npz":s['raw_sha256'] for s in audit['streams']})
validate_frozen_energy_audit(base, registration, audit)
for name, digest in base['streams'].items():
    if file_sha256(audit_path.parent/name) != digest:
        raise ValueError('completed raw Born cache changed')
base['frozen_input_fingerprints'] = dict(parent_registration=registration['result_sha256'],
    parent_registration_file_sha256=file_sha256(parent/'registration.json'),
    energy_audit_file_sha256=file_sha256(audit_path), energy_audit_result_sha256=audit['result_sha256'],
    checkpoint_sha256=audit['checkpoint_sha256'],
    old_reader_source_sha256=file_sha256(Path('projects/clebsch_tree_nqs/workflows/spin_replica_response.py')))
base['original_readout_plan_sha256'] = os.environ['NQS_READOUT_PLAN_SHA256']
config_parent = Path(request['materialized_base']).parent
config_parent.mkdir(parents=True, exist_ok=True)
for file in (request['materialized_base'], request['materialized_pipeline']):
    if Path(file).exists():
        raise FileExistsError('do not overwrite an existing readout configuration')
atomic_json(Path(request['materialized_base']), base)
pipeline = dict(base_config=request['materialized_base'], base_config_sha256=file_sha256(Path(request['materialized_base'])),
    output=request['output'], stage_wall_seconds=request['stage_wall_seconds'],
    original_readout_plan_sha256=os.environ['NQS_READOUT_PLAN_SHA256'],
    predeclared_before_new_readout=True,
    revision='Unchanged prefix18/six row probes and twelve relative orbits; actual sealed energy-only audit; no new Born/H/training.')
atomic_json(Path(request['materialized_pipeline']), pipeline)
print(json.dumps(dict(materialized_pipeline=request['materialized_pipeline'], state=audit['params_sha256'],
    original_plan_sha256=os.environ['NQS_READOUT_PLAN_SHA256'], new_H_calls=0, new_Born_samples=0)))
PY
task_pipeline="$("$task_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["materialized_pipeline"])' "$task_plan")"
exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_schmidt_covariance \
  --pipeline-config "$task_pipeline"
