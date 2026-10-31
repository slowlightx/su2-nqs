#!/usr/bin/env bash
#SBATCH --job-name=cg-capacity
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/cg-capacity-%j.out
#SBATCH --error=logs/cg-capacity-%j.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
export PYTHONPATH="$PWD"
task_python="$PWD/.conda-netket/bin/python"
task_dir=projects/clebsch_tree_nqs/configs/remote/scale_capacity_20261014
task_cost="${NQS_CG_COST_CONFIG:-$task_dir/cg_cost.json}"
task_config="${NQS_CG_TRAIN_CONFIG:-$task_dir/cg_train.json}"
"$task_python" -c 'import jax; assert jax.default_backend()=="gpu" and jax.device_count()==1; print(jax.devices())'
case "${1:?cost, prepare, train or audit}" in
  cost)
    if [[ "${NQS_CG_INIT_TESTS:-0}" == "1" ]]; then
      "$task_python" -m pytest tests/test_finite_schur_decoder.py tests/test_finite_schur_optimize.py -q
    fi
    "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_finite_schur_cost --config "$task_cost" --register
    "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_finite_schur_cost --config "$task_cost" --geometry triangular
    "$task_python" -c 'import json,sys; from pathlib import Path; s=json.load(open(sys.argv[1])); r=json.load(open(Path(s["output"])/"triangular/results.json")); assert r["passed"] and r["backend"]=="gpu"' "$task_cost"
    ;;
  prepare)
    "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_finite_schur_optimize --config "$task_config" --stage register
    "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_finite_schur_optimize --config "$task_config" --stage prepare
    ;;
  train)
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_finite_schur_optimize --config "$task_config" --stage train --arm "${2:?network or free}" --until-step "${3:?fixed milestone}"
    ;;
  audit)
    exec "$task_python" -u -m projects.clebsch_tree_nqs.workflows.spin_finite_schur_optimize --config "$task_config" --stage audit
    ;;
  *) exit 2 ;;
esac
