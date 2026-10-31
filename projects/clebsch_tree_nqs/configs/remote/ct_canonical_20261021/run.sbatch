#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=kim-compute-03
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@900
# Root coordinator chooses the allocated A100 node and submits stages serially.
# Pilot, replay, and train for an arm MUST stay on the same pinned node/runtime.
set -euo pipefail
cd "${NQS_WORKSPACE:?isolated immutable workspace}"
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
NQS_PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
"$NQS_PYTHON" - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('projects/clebsch_tree_nqs/configs/remote/ct_canonical_20261021')
manifest=json.loads((root/'source_manifest.json').read_text())
for name,digest in manifest.items():
    if hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()!=digest:
        raise ValueError('immutable source changed: '+name)
print('source manifest verified:',len(manifest),flush=True)
PY
if [[ "${NQS_PHASE:?tests/pilot/replay/train}" == tests ]]; then
    export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
    exec "$NQS_PYTHON" -m pytest -q \
        tests/test_fusion_paths.py tests/test_coupled_basis.py \
        tests/test_recoupling.py tests/test_heisenberg.py tests/test_models.py \
        tests/test_physical_step_calibration.py tests/test_triangular_clebsch_gauge.py \
        tests/test_triangle_canonical_comparison.py \
        --junitxml=logs/canonical_exact_tests.xml
fi
export JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
exec "$NQS_PYTHON" -u -m projects.clebsch_tree_nqs.workflows.triangle_canonical_comparison \
    --plan "${NQS_CONFIG:?immutable plan}" --plan-sha256 "${NQS_PLAN_SHA256:?plan seal}" \
    --arm "${NQS_ARM:?raw/canonical}" --phase "$NQS_PHASE"
