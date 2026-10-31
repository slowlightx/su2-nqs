#!/usr/bin/env bash
#SBATCH --job-name=nqs-dmrg8192
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=logs/dmrg-kim-%j.out
#SBATCH --error=logs/dmrg-kim-%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export PYTHONPATH="$PWD"
task_config="${1:?registered config required}"
# Kim permits HOME wavefunctions (user-authorized 2026-10-10). Derive the
# validated output stage from its config rather than hard-code a prior run.
# The separate immutable archive remains hash-verified on safe rollback.
mapfile -t task_paths < <(.venv-tenpy/bin/python - "$task_config" <<'PY'
import json, sys
from pathlib import Path
spec=json.loads(Path(sys.argv[1]).read_text())
output=Path(spec['output']).resolve()
if not output.name or output.parent == Path('/'):
    raise ValueError('explicit stage directory required')
print(output.parent)
print(output.name)
print(spec.get('archive_root', '/share/kim/yx789/nqs-reference-dmrg/archives'))
PY
)
[[ ${#task_paths[@]} == 3 ]]
set +e
.venv-tenpy/bin/python -u -m projects.clebsch_tree_nqs.workflows.dmrg_continuation \
  --config "$task_config" --require-complete &
task_pid=$!
trap 'kill -USR1 "$task_pid" 2>/dev/null || true' USR1 TERM INT
wait "$task_pid"
task_status=$?
if kill -0 "$task_pid" 2>/dev/null; then
  wait "$task_pid"
  task_status=$?
fi
set -e
.venv-tenpy/bin/python -m projects.clebsch_tree_nqs.workflows.dmrg_archive \
  --source-root "${task_paths[0]}" --archive-root "${task_paths[2]}" \
  --stage "${task_paths[1]}" --receipt "logs/dmrg-archive-${SLURM_JOB_ID:?}.json"
exit "$task_status"
