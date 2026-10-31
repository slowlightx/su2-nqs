#!/usr/bin/env bash
#SBATCH --job-name=nqs-vit-quality
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=36:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/vit-%j.out
#SBATCH --error=logs/vit-%j.err

set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export PYTHONPATH="$PWD"
export MPLCONFIGDIR="$PWD/.matplotlib"
task_python="$PWD/.conda-netket/bin/python"
task_mode="${1:?canary, pilot, production or audit}"
task_config="projects/clebsch_tree_nqs/configs/remote/spin_spatial_vit_netket_n36.json"
"$task_python" -c 'import jax,netket; print(jax.devices(),netket.__version__); assert jax.default_backend()=="gpu"'

case "$task_mode" in
  canary)
    "$task_python" -m pytest tests/test_spatial_vit_netket.py -q
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
      --config projects/clebsch_tree_nqs/configs/remote/spin_spatial_vit_netket_canary.json --session-seconds 13500 --require-endpoint
    ;;
  pilot)
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
      --config "$task_config" --until-step 200 --session-seconds 27900 --require-endpoint
    ;;
  production)
    "$task_python" - <<'PY'
from pathlib import Path
import numpy as np
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import read_sealed
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.frontier.artifacts import pytree_sha256
from su2_nqs.vmc.checkpoint import load_checkpoint
p=Path('projects/clebsch_tree_nqs/results/spin_spatial_vit_gpu_quality_v1/seed1')
r=read_sealed(p/'progress.json'); saved=load_checkpoint(p/'checkpoint.pkl')
assert r['completed_steps']==200 and r['backend']=='netket_ntk'
assert file_sha256(p/'checkpoint.pkl')==r['checkpoint_sha256']
assert pytree_sha256(saved['params']['model'])==r['final_params_sha256']
assert saved['optimizer_state']['step']==200
h=r['history']; assert len(h)==200 and r['runtime']['jax_backend']=='gpu'
for key in ('energy_before_update_per_site','variance_before_update_per_site','seconds_including_compile'):
    assert np.isfinite([x[key] for x in h]).all()
first,last=h[:20],h[-20:]
average=lambda rows,key: float(np.mean([x[key] for x in rows]))
e='energy_before_update_per_site'; v='variance_before_update_per_site'
assert average(last,e)<average(first,e), 'no early energy improvement: stop and inspect'
assert average(last,v)<1.25*average(first,v), 'unstable variance: stop and inspect'
assert max(x['gpu_peak_bytes'] for x in h)<75*1024**3, 'memory margin insufficient'
print('verified completed pilot/checkpoint; early/late E, variance; recent sec/update',
      average(first,e),average(last,e),average(first,v),average(last,v),
      average(last,'seconds_including_compile'))
PY
    # No fresh initialization. Resume the SAME registered trajectory, including
    # its sampler RNG and total time/work budget.
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket \
      --config "$task_config" --resume --session-seconds 126000 --require-endpoint
    ;;
  audit)
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality \
      audit --config "$task_config" --steps 5600
    ;;
  *) exit 2 ;;
esac
