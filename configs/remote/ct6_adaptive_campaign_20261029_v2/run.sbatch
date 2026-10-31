#!/usr/bin/env bash
#SBATCH --partition=kim
#SBATCH --account=kim
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
CT_CAMPAIGN_TAG="${1:-}"
CT_CAMPAIGN_SEGMENT="${2:-}"
case "$CT_CAMPAIGN_TAG" in ct6_j0|ct6_j0125) ;; *) exit 2 ;; esac
case "$CT_CAMPAIGN_SEGMENT" in 01|02|03|04) ;; *) exit 2 ;; esac
CT_CAMPAIGN_WORKSPACE=/home/yx789/nqs-ct6-adaptive-campaign-20261029-v2
CT_CAMPAIGN_PYTHON=/home/yx789/nqs-vit-quality/.conda-netket/bin/python
CT_CAMPAIGN_CONFIG=configs/remote/ct6_adaptive_campaign_20261029_v2
CT_CAMPAIGN_PLAN=$CT_CAMPAIGN_CONFIG/${CT_CAMPAIGN_TAG}_segment${CT_CAMPAIGN_SEGMENT}.json
CT_CAMPAIGN_OUTPUT=/home/yx789/nqs-wavefunctions/nqs/ct6_adaptive_campaign_20261029_v2/$CT_CAMPAIGN_TAG/segment$CT_CAMPAIGN_SEGMENT
CT_CAMPAIGN_EPOCH="$(date +%s)"
export SLURM_JOB_END_TIME="${SLURM_JOB_END_TIME:-$((CT_CAMPAIGN_EPOCH+86400))}"
cd "$CT_CAMPAIGN_WORKSPACE"
export PYTHONPATH="$CT_CAMPAIGN_WORKSPACE" PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=.90
unset XLA_PYTHON_CLIENT_MEM_FRACTION
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export JAX_COMPILATION_CACHE_DIR="$CT_CAMPAIGN_WORKSPACE/cache/$CT_CAMPAIGN_TAG"
test ! -e "$CT_CAMPAIGN_OUTPUT"
mkdir -p "$CT_CAMPAIGN_OUTPUT"
"$CT_CAMPAIGN_PYTHON" - "$CT_CAMPAIGN_PLAN" "$CT_CAMPAIGN_OUTPUT" <<'PY'
import hashlib,importlib.metadata,json,os,pathlib,sys
root=pathlib.Path.cwd();config=root/'configs/remote/ct6_adaptive_campaign_20261029_v2'
manifest=config/'source_manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==os.environ['NQS_CT6_CAMPAIGN_MANIFEST_SHA256']
for name,digest in json.loads(manifest.read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
for name,version in json.loads((config/'runtime_packages.json').read_text()).items():
    assert importlib.metadata.version(name)==version,name
from projects.clebsch_tree_nqs.workflows.ct6_adaptive_campaign import validate_plan
plan=json.loads(pathlib.Path(sys.argv[1]).read_text());validate_plan(plan,executing=True)
parent=pathlib.Path(plan['parent_directory']);result=parent/'audit_result.json'
if not result.is_file() or json.loads(result.read_text()).get('status')!='complete_fixed_prefix_independent_audit':
    receipt={'status':'predecessor_lacks_complete_audit_no_training','parent':str(parent),
        'existing_jobs_cancelled':False,'unsafe_automatic_continuation':False}
    (pathlib.Path(sys.argv[2])/'refusal.json').write_text(json.dumps(receipt,indent=2))
    raise SystemExit('Predecessor has no completed independent audit: refusing unsealed continuation')
print('Campaign sources/runtime sealed; predecessor terminal audit exists; native binding follows.',flush=True)
PY
for CT_CAMPAIGN_PHASE in prepare replay train audit; do
    /usr/local/slurm/current/bin/srun --ntasks=1 "$CT_CAMPAIGN_PYTHON" -u -m \
        projects.clebsch_tree_nqs.workflows.ct6_adaptive_campaign \
        --plan "$CT_CAMPAIGN_PLAN" --output "$CT_CAMPAIGN_OUTPUT/run01" --phase "$CT_CAMPAIGN_PHASE"
done
