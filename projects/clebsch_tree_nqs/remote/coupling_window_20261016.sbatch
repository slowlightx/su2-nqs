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
#SBATCH --output=logs/coupling-window-%j.out
#SBATCH --error=logs/coupling-window-%j.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export NETKET_SHARDING=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export PYTHONPATH="$PWD"
task_python=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
task_mode="${1:?freeze-audit, train, audit, or scalar}"
task_config="${2:?registered configuration}"
task_child=''
trap 'if [[ -n "$task_child" ]]; then kill -USR1 "$task_child" 2>/dev/null || true; fi' USR1 TERM
run_python() {
  "$task_python" "$@" & task_child=$!
  local task_status=0
  wait "$task_child" || task_status=$?
  if kill -0 "$task_child" 2>/dev/null; then
    wait "$task_child" || task_status=$?
  fi
  task_child=''
  return "$task_status"
}
task_output="$("$task_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$task_config")"
case "$task_mode" in
  freeze-audit)
    run_python -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
      --config "$task_config" --until-step 0 --require-endpoint
    run_python -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality \
      audit-energy --config "$task_config" --steps 0
    ;;
  train)
    if [[ "${3:?continuation or warm-start}" == continuation ]]; then
      run_python -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$task_config" --until-step 502 --session-seconds 600 --require-endpoint
      task_start=500; task_compare=502
    else
      [[ -f "$task_output/audits/step00000/energy.json" ]]
      run_python -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$task_config" --until-step 1 --session-seconds 600 --require-endpoint
      run_python -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
        --config "$task_config" --resume --until-step 3 --session-seconds 600 --require-endpoint
      task_start=1; task_compare=3
    fi
    run_python -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_jump_probe \
      --config "$task_config" --checkpoint "$task_output/quality/$(printf 'step%05d.pkl' "$task_start")" \
      --compare-checkpoint "$task_output/quality/$(printf 'step%05d.pkl' "$task_compare")" \
      --output "$task_output/recovery_two_step_replay.json" --maximum-seconds 600
    run_python -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
      --config "$task_config" --resume --session-seconds "${4:?remaining training seconds}" --require-endpoint
    ;;
  audit)
    run_python -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality \
      audit-energy --config "$task_config"
    ;;
  scalar)
    # Both frozen operators and deterministic cache indices are fixed by this
    # registered template. Only unknown-until-completion state/data digests
    # are filled AFTER the independent audits; no channel or endpoint search.
    run_python - "$task_config" <<'PY'
import json, sys
from pathlib import Path
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import read_sealed
root = Path.cwd()
spec = json.loads(Path(sys.argv[1]).read_text())
parent = Path(spec['output'])
step = sum(s['steps'] for s in spec['stages'])
registration = read_sealed(parent/'registration.json')
audit_path = parent/'audits'/f'step{step:05d}'/'energy.json'
audit = read_sealed(audit_path)
template = root/'projects/clebsch_tree_nqs/configs/remote/frozen_channels_20261016/parents_r2/tri_j005_500.json'
base = json.loads(template.read_text())
base.update(parent=str(parent), step=step, checkpoint_sha256=audit['checkpoint_sha256'],
    energy_audit_filename='energy.json', name=parent.parent.name,
    streams={f"stream{s['seed']}.npz":s['raw_sha256'] for s in audit['streams']})
base['frozen_input_fingerprints'] = dict(parent_registration=registration['result_sha256'],
    parent_registration_file_sha256=file_sha256(parent/'registration.json'),
    energy_audit_file_sha256=file_sha256(audit_path), energy_audit_result_sha256=audit['result_sha256'],
    checkpoint_sha256=audit['checkpoint_sha256'],
    old_reader_source_sha256=file_sha256(root/'projects/clebsch_tree_nqs/workflows/spin_replica_response.py'))
base_path = parent/'frozen_scalar_parent.json'
atomic_json(base_path, base)
old_request = root/'projects/clebsch_tree_nqs/configs/remote/frozen_channels_20261016/tri_j005_500.json'
request = json.loads(old_request.read_text())
request.update(base_config=str(base_path), output=str(parent/'frozen_scalar'), wall_seconds=540)
atomic_json(parent/'frozen_scalar_request.json', request)
PY
    run_python -u -m projects.clebsch_tree_nqs.workflows.spin_frozen_distance \
      --config "$task_output/frozen_scalar_request.json"
    ;;
  *) exit 2 ;;
esac
