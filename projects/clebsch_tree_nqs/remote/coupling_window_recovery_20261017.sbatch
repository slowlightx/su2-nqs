#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/coupling-recovery-%j.out
#SBATCH --error=logs/coupling-recovery-%j.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export NETKET_SHARDING=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export PYTHONPATH="$PWD"
task_python=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
task_mode="${1:?resume, audit, scalar}"
task_config="${2:?registered configuration}"
case "$task_mode" in
  resume)
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
      --config "$task_config" --resume --session-seconds 1500 --require-endpoint
    ;;
  audit)
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality \
      audit-energy --config "$task_config"
    ;;
  scalar)
    # The materializer MUST run in the foreground: background Python loses
    # a shell heredoc's stdin when noninteractive job control is disabled.
    "$task_python" - "$task_config" <<'PY'
import json, sys
from pathlib import Path
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import read_sealed
from projects.clebsch_tree_nqs.workflows.spin_chirality_covariance import validate_frozen_energy_audit
root=Path.cwd();spec=json.loads(Path(sys.argv[1]).read_text());parent=Path(spec['output'])
step=sum(s['steps'] for s in spec['stages'])
registration=read_sealed(parent/'registration.json');training=read_sealed(parent/'training.json')
if training['status']!='fixed_steps_complete' or training['completed_steps']!=step:
    raise ValueError('scalar readout needs the actual complete fixed endpoint')
audit_path=parent/'audits'/f'step{step:05d}'/'energy.json';audit=read_sealed(audit_path)
checkpoint=parent/'quality'/f'step{step:05d}.pkl'
if file_sha256(checkpoint)!=audit['checkpoint_sha256'] or training['final_params_sha256']!=audit['params_sha256']:
    raise ValueError('frozen scalar endpoint identity changed')
manifest=json.loads((root/'projects/clebsch_tree_nqs/remote/coupling_window_recovery_sync_20261017.json').read_text())
template='projects/clebsch_tree_nqs/configs/remote/frozen_channels_20261016/parents_r2/tri_j005_500.json'
old_request='projects/clebsch_tree_nqs/configs/remote/frozen_channels_20261016/tri_j005_500.json'
for source in (template,old_request):
    if file_sha256(root/source)!=manifest['files'][source]:
        raise ValueError('the original frozen scalar geometry/snapshot protocol changed')
base=json.loads((root/template).read_text())
base.update(parent=str(parent),step=step,checkpoint_sha256=audit['checkpoint_sha256'],
    energy_audit_filename='energy.json',name=parent.parent.name,
    streams={f"stream{s['seed']}.npz":s['raw_sha256'] for s in audit['streams']})
validate_frozen_energy_audit(base,registration,audit)
base['frozen_input_fingerprints']=dict(parent_registration=registration['result_sha256'],
    parent_registration_file_sha256=file_sha256(parent/'registration.json'),
    energy_audit_file_sha256=file_sha256(audit_path),energy_audit_result_sha256=audit['result_sha256'],
    checkpoint_sha256=audit['checkpoint_sha256'],
    old_reader_source_sha256=file_sha256(root/'projects/clebsch_tree_nqs/workflows/spin_replica_response.py'))
base['materializer_fix']='Foreground stdin; original failed scalar job never queried amplitudes.'
configs=root/'materialized_scalar';configs.mkdir(exist_ok=True)
base_path=configs/(parent.parent.name+'_parent.json');request_path=configs/(parent.parent.name+'_request.json')
if base_path.exists() or request_path.exists():raise FileExistsError('immutable scalar materialization already exists')
atomic_json(base_path,base)
request=json.loads((root/old_request).read_text())
request.update(base_config=str(base_path),output=str(parent/'frozen_scalar'),wall_seconds=540)
atomic_json(request_path,request)
print(json.dumps(dict(config=str(request_path),new_Born_samples=0,new_H_calls=0,new_training=0)),flush=True)
PY
    task_output="$("$task_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$task_config")"
    task_name="$(basename "$(dirname "$task_output")")"
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_frozen_distance \
      --config "$PWD/materialized_scalar/${task_name}_request.json"
    ;;
  *) exit 2 ;;
esac
