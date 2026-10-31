"""Generate and seal a standalone zero-update preflight, never submit it."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
FOLDER=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


def file_sha(path:Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorize-execution',action='store_true',
        help='Use only after root final confirmation; this still never submits.')
    args=parser.parse_args()
    from su2_nqs.frontier.provenance import transitive_local_source_manifest
    consumer=ROOT/'projects/clebsch_tree_nqs/explorations/ct12_w28_gpu_preflight_20261030.py'
    width=ROOT/'projects/clebsch_tree_nqs/explorations/ct12_live_w28_20261030.py'
    base='projects/clebsch_tree_nqs/results/'
    inputs=dict(parent=dict(path=base+'ct12_w28_reentry_20261030/diagnosis/global45_checkpoint.pkl',
        sha256='751aed76fe1de0862240f4b99b4523aac89f21d47773ae9262be47018a631d8f'),
        expanded=dict(path=base+'ct12_live_w28_20261030/projected_JVP_v2/expanded_checkpoint.pkl',
        sha256='c657c97756d832d7587df3b30494ba4e7780905869bff65c3c29211ce6140c1b'),
        rows=dict(path=base+'ct12_live_w28_20261030/projected_JVP_v2/actual_amplitude_checks.npz'),
        parent_registration=dict(path=base+'remote_status_20261030_morning/ct12_j0125_retry/registration.json'),
        CPU_inclusion=dict(path=base+'ct12_live_w28_20261030/projected_JVP_v2/results.json'))
    for name,record in inputs.items():
        actual=file_sha(ROOT/record['path'])
        if record.get('sha256',actual)!=actual:raise ValueError('immutable input differs: '+name)
        record['sha256']=actual;record['bytes']=(ROOT/record['path']).stat().st_size
    proof=json.loads((ROOT/inputs['CPU_inclusion']['path']).read_text())
    if proof['status']!='complete_same_state_w28_checkpoint_and_actual_amplitude_preflight_no_updates':
        raise ValueError('actual CPU inclusion and projected-JVP checks incomplete')
    plan=dict(task='CT12_w28_GPU_cost_replay_preflight_v1',execution_authorized=args.authorize_execution,
        submission_requires_root_final_confirmation=True,length=12,j2=.125,target_q=0,target_m2=0,maximum_q=10,
        multiplicity_cap=28,active_real_parameters=20748160,complete_A1_images=1728,physical_H_bonds=864,
        rows=4,row_indices=[0,1,2,3],maximum_updates=0,scientific_fit_records_consumed=0,
        scientific_RNG_consumed=False,wall_cap_seconds=14400,maximum_peak_device_bytes=70*1024**3,
        nodelist=None,full_dense_Jacobian_allowed=False,required_phases=['amplitude','full_H','derivatives','fisher'],
        phase_caps_seconds=dict(amplitude=1800,full_H=3600,derivatives=3600,fisher=3600,H_JVP=2400,full_H_four=1800),
        projection_chunk=16,replay_tolerance=2e-11,derivative_tolerance=2e-10,
        optional_full_H_four=False,optional_H_JVP=True,save_reserve_seconds=180,minimum_phase_launch_seconds=300,
        inputs=inputs,strict_S0_and_full_A1_unchanged=True,no_energy_or_gradient_fit_or_population_Fisher_claim=True,
        mathematical_Fisher_Q='constant positive Q over four frozen diagnostic rows; not population sampling',
        peak_guard_is_postcall_not_a_physical_hard_limit=True,
        allocator_fraction_requested=.85,actual_allocator_peak_required=True,
        stop_on_OOM_without_automatic_width_or_statistics_fallback=True,
        original_profile_replay_is_cost_control_not_new_training=True)
    from projects.clebsch_tree_nqs.explorations.ct12_w28_gpu_preflight_20261030 import validate_plan
    validate_plan(plan)
    (FOLDER/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    versions=json.loads((ROOT/'configs/remote/ct12_direction_gate_retry_20261029/runtime_packages.json').read_text())
    (FOLDER/'runtime_packages.json').write_text(json.dumps(versions,indent=2)+'\n')
    gram=ROOT/'projects/clebsch_tree_nqs/explorations/ct12_w28_tangent_strength_20261030.py'
    test_consumer=ROOT/'tests/test_ct12_w28_gpu_preflight_20261030.py'
    test_width=ROOT/'tests/test_ct12_live_w28_20261030.py'
    # vmc/__init__.py eagerly calls import_module('.local_energy', __name__)
    # for its public-name collision. Static AST imports do not discover this
    # dynamic dependency: seal its transitive closure explicitly, without
    # changing the mathematical source or relying on the local full checkout.
    eager_local_energy=ROOT/'su2_nqs/vmc/local_energy.py'
    manifest=transitive_local_source_manifest((consumer,width,gram,test_consumer,test_width,eager_local_energy),root=ROOT,
        package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    extras=[*FOLDER.glob('*.py'),FOLDER/'plan.json',FOLDER/'run.sbatch',FOLDER/'runtime_packages.json',
        ROOT/'tests/test_ct12_w28_gpu_preflight_20261030.py',ROOT/'tests/test_ct12_live_w28_20261030.py',
        *(ROOT/record['path'] for record in inputs.values())]
    for path in extras:manifest[str(path.relative_to(ROOT))]=file_sha(path)
    manifest=dict(sorted(manifest.items()))
    (FOLDER/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    transfer=[*manifest,str((FOLDER/'source_manifest.json').relative_to(ROOT)),
        str((FOLDER/'sync_files.txt').relative_to(ROOT))]
    (FOLDER/'sync_files.txt').write_text('\n'.join(transfer)+'\n')
    print(json.dumps(dict(status='local_standalone_preflight_sealed_not_submitted',
        execution_authorized=args.authorize_execution,files=len(manifest),
        bytes=sum((ROOT/name).stat().st_size for name in manifest),
        source_manifest_sha256=file_sha(FOLDER/'source_manifest.json'),
        remote_workspace='/home/yx789/nqs-ct12-w28-gpu-preflight-20261030',
        requested_A100_hours_upper=4,maximum_updates=0),indent=2))


if __name__=='__main__':main()
