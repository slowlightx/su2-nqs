#!/usr/bin/env bash
#SBATCH --job-name=nqs-vit-recovery
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:2
#SBATCH --time=16:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/vit-recovery-%j.out
#SBATCH --error=logs/vit-recovery-%j.err

set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export NETKET_SHARDING=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export PYTHONPATH="$PWD"
export MPLCONFIGDIR="$PWD/.matplotlib"
task_python="$PWD/.conda-netket/bin/python"
task_config="${NQS_CONFIG:-projects/clebsch_tree_nqs/configs/remote/spin_spatial_vit_netket_recovery_n36.json}"
task_mode="${1:?cpu-test, preflight, production, energy-stream or aggregate}"

# Preserve historical two-device protocols unless a new sealed experiment
# explicitly chooses another count. The Python driver independently enforces
# the configured count before creating its Hamiltonian or sampler.
expected_training_devices() {
  "$task_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["netket"].get("expected_device_count",2))' "$task_config"
}

case "$task_mode" in
  jump-test)
    export JAX_PLATFORMS=cpu
    export NETKET_SHARDING=0
    exec "$task_python" -m pytest tests/test_spatial_vit_jump_probe.py -q -s
    ;;
  jump-test-sharded)
    export JAX_PLATFORMS=cpu
    export NETKET_SHARDING=1
    export XLA_FLAGS="--xla_force_host_platform_device_count=2"
    exec "$task_python" -m pytest tests/test_spatial_vit_jump_probe.py -q -s
    ;;
  jump-probe)
    task_checkpoint=()
    task_frozen=()
    task_replicated=()
    task_compare=()
    if [[ -n "${NQS_PROBE_CHECKPOINT:-}" ]]; then
      task_checkpoint=(--checkpoint "$NQS_PROBE_CHECKPOINT")
    fi
    if [[ "${NQS_PROBE_FREEZE_PARAMETERS:-0}" == "1" ]]; then
      task_frozen=(--freeze-parameters)
    fi
    if [[ "${NQS_PROBE_REPLICATE_CANDIDATE:-0}" == "1" ]]; then
      task_replicated=(--replicate-candidate)
    fi
    if [[ -n "${NQS_COMPARE_CHECKPOINT:-}" ]]; then
      task_compare=(--compare-checkpoint "$NQS_COMPARE_CHECKPOINT")
    fi
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_jump_probe \
      --config "$task_config" "${task_checkpoint[@]}" "${task_frozen[@]}" "${task_replicated[@]}" "${task_compare[@]}" \
      --output "${NQS_PROBE_OUTPUT:?new diagnostic output path}" \
      --steps "${NQS_PROBE_STEPS:-2}" --maximum-seconds "${NQS_PROBE_SECONDS:-1500}"
    ;;
  optimizer-conditions)
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_jump_probe \
      --config "$task_config" --checkpoint "${NQS_PROBE_CHECKPOINT:?fixed parent checkpoint}" \
      --optimizer-audit "${NQS_OPTIMIZER_AUDIT:?fixed parent audit}" \
      --output "${NQS_PROBE_OUTPUT:?new diagnostic path}" \
      --maximum-seconds "${NQS_PROBE_SECONDS:-2700}"
    ;;
  cpu-test)
    export JAX_PLATFORMS=cpu
    export NETKET_SHARDING=0
    exec "$task_python" -m pytest tests/test_spatial_vit_netket.py \
      tests/test_kim_repair_campaign.py tests/test_spatial_vit_quality.py -q -s
    ;;
  preflight)
    task_expected_devices="$(expected_training_devices)"
    "$task_python" -c 'import sys,jax,netket; print(jax.devices(),netket.__version__); assert jax.default_backend()=="gpu" and jax.device_count()==int(sys.argv[1])' "$task_expected_devices"
    # Includes BOTH NTK implementations versus an independent centered SR
    # equation, full RNG replay, and immutable-parent recovery checks.
    "$task_python" -m pytest tests/test_spatial_vit_netket.py -q -s
    # First declared production update, not an extra sampler/training batch.
    "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
      --config "$task_config" --until-step "${NQS_PREFLIGHT_ENDPOINT:-5501}" --session-seconds 6300 \
      --require-endpoint --require-sample-partitions "$task_expected_devices"
    # Optional preregistered coarse INITIAL-sampling screen, never an
    # independent-audit winner selection or parameter backtracking rule.
    if [[ -n "${NQS_INITIAL_ENERGY_CEILING:-}" ]]; then
      "$task_python" - "$task_config" <<'PY'
import json, math, os, sys
from pathlib import Path
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import ROOT, read_sealed
spec = json.loads(Path(sys.argv[1]).read_text())
progress = read_sealed(ROOT/spec['output']/'progress.json')
row = progress['history'][0]
energy = row['energy_before_update_per_site']
variance = row['variance_before_update_per_site']
if (row['step'] != 1 or row['projection'] != 'gamma_a1'
        or not all(map(math.isfinite, (energy, variance)))
        or energy > float(os.environ['NQS_INITIAL_ENERGY_CEILING'])
        or variance > float(os.environ['NQS_INITIAL_VARIANCE_CEILING'])):
    raise SystemExit('initial parent-sampling moments fail coarse plausibility screen; no long training released')
print('initial sampling plausibility passed; NOT an equilibrium certificate:', energy, variance)
if os.environ.get('NQS_BOUNDED_PILOT_SCREEN') == '1':
    rows = progress['history']
    endpoint = int(os.environ['NQS_PREFLIGHT_ENDPOINT'])
    if progress['completed_steps'] != endpoint or len(rows) != endpoint:
        raise SystemExit('bounded pilot did not reach registered checkpoint')
    expected_partitions = spec['netket'].get('expected_device_count', 2)
    if any(r['unique_local_sample_partitions'] != expected_partitions or r['projection'] != 'gamma_a1'
           or not all(math.isfinite(r[k]) for k in
                      ('energy_before_update_per_site','variance_before_update_per_site','sampled_fs_sine'))
           for r in rows):
        raise SystemExit('invalid support/moments/layout in bounded pilot')
    tail = rows[-5:]
    tail_e = sum(r['energy_before_update_per_site'] for r in tail)/len(tail)
    tail_v = sum(r['variance_before_update_per_site'] for r in tail)/len(tail)
    if (tail_e > energy+.025 or tail_v > 2*variance+.02
            or sum(r['update_accepted'] for r in rows) < endpoint//2):
        raise SystemExit('bounded pilot collapse/repeated-rejection screen failed; no long job released')
    print('bounded pilot sanity passed, not an independent accuracy audit:', tail_e, tail_v)
PY
    fi
    ;;
  production|complete-milestone)
    task_expected_devices="$(expected_training_devices)"
    task_endpoint=()
    if [[ -n "${NQS_UNTIL_STEP:-}" ]]; then task_endpoint=(--until-step "$NQS_UNTIL_STEP"); fi
    if [[ "$task_mode" == "complete-milestone" ]]; then
      # An afterany continuation is ONLY for an identified own writer that
      # safely pauses at its session cap. Never hide a failed/partial update,
      # reset cumulative budgets, or overwrite a completed quality snapshot.
      task_resume_needed="$("$task_python" - "$task_config" <<'PY'
import json, os, sys
from pathlib import Path
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import ROOT, register, read_sealed
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.frontier.artifacts import pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
protocol = Path(sys.argv[1])
spec, registration = register(protocol, ROOT/json.loads(protocol.read_text())['output'])
output = ROOT/spec['output']; progress = read_sealed(output/'progress.json')
saved = load_checkpoint(output/'checkpoint.pkl')
step = progress['completed_steps']; endpoint = int(os.environ['NQS_UNTIL_STEP'])
if (step != saved['optimizer_state']['step'] or step != saved['metadata']['completed_steps']
        or len(saved['metadata']['history']) != step
        or progress['checkpoint_sha256'] != file_sha256(output/'checkpoint.pkl')
        or progress['final_params_sha256'] != pytree_sha256(saved['params']['model'])
        or saved['metadata']['registration_sha256'] != registration['result_sha256']
        or progress['registration_sha256'] != registration['result_sha256']
        or endpoint not in spec['quality_milestones'] or not 0 < step <= endpoint):
    raise ValueError('session continuation checkpoint/registered milestone identity changed')
if step == endpoint:
    if progress['status'] not in ('paused_session_budget','fixed_steps_complete'):
        raise ValueError('completed boundary is not safely stopped')
    print('no')
elif progress['status'] != 'paused_session_budget' or (output/'training.json').exists():
    raise ValueError('only a safe session pause can continue; failed/time-capped work is not released')
else:
    print('yes')
PY
)"
      if [[ "$task_resume_needed" == "no" ]]; then
        echo "registered milestone already complete; no repeated update or sampler draw"
        exit 0
      fi
      [[ "$task_resume_needed" == "yes" ]]
    fi
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
      --config "$task_config" --resume --session-seconds "${NQS_SESSION_SECONDS:-54900}" "${task_endpoint[@]}" \
      --require-endpoint --require-sample-partitions "$task_expected_devices"
    ;;
  energy-stream)
    "$task_python" -c 'import jax; assert jax.default_backend()=="gpu" and jax.device_count()==1'
    task_audit_endpoint=()
    if [[ -n "${NQS_AUDIT_STEP:-}" ]]; then task_audit_endpoint=(--steps "$NQS_AUDIT_STEP"); fi
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality \
      audit-energy --config "$task_config" "${task_audit_endpoint[@]}" --stream-index "${2:-${SLURM_ARRAY_TASK_ID:?independent stream index}}"
    ;;
  milestone-audit)
    "$task_python" -c 'import jax; assert jax.default_backend()=="gpu" and jax.device_count()==1'
    # Existing immutable quality snapshots may be audited before the longer
    # registered trajectory ends. Reuse the complete existing audit protocol;
    # never relabel a paused trajectory as a fixed terminal endpoint.
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality \
      audit --config "$task_config" --steps "${NQS_AUDIT_STEP:?frozen registered milestone}"
    ;;
  aggregate)
    export JAX_PLATFORMS=cpu
    task_audit_endpoint=()
    if [[ -n "${NQS_AUDIT_STEP:-}" ]]; then task_audit_endpoint=(--steps "$NQS_AUDIT_STEP"); fi
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality \
      audit-energy --config "$task_config" "${task_audit_endpoint[@]}" --aggregate-only
    ;;
  *) exit 2 ;;
esac
