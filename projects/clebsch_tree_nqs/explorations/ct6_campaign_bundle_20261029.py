"""Prepare/seal the user-authorized four-segment per-coupling campaign.

No SSH, Slurm submission or physical amplitude query occurs in this builder.
The parent of each segment is bound immutably AFTER its predecessor finishes.
"""
from __future__ import annotations

import json
from pathlib import Path

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.frontier.provenance import file_sha256,transitive_local_source_manifest
from projects.clebsch_tree_nqs.workflows import ct6_adaptive_campaign as campaign

ROOT=Path(__file__).resolve().parents[3]
CONFIG=ROOT/'configs/remote/ct6_adaptive_campaign_20261029_v2'
LOCAL_PARENTS=ROOT/'projects/clebsch_tree_nqs/results/ct_sector_excitation_pilot_20261028/completed_CT6'
WORKSPACE='/home/yx789/nqs-ct6-adaptive-campaign-20261029-v2'
DATA='/home/yx789/nqs-wavefunctions/nqs/ct6_adaptive_campaign_20261029_v2'


def run()->dict:
    """Exactly8 finite jobs, at most2 serial GPU lines and192 newGPU-hours."""
    CONFIG.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for tag,j2,offset in (('ct6_j0',0.,0),('ct6_j0125',.125,1000)):
        for segment in range(1,5):
            name=f'{tag}_segment{segment:02d}'
            roles=campaign.base.required_seed_roles()
            seeds={role:3930293001+offset+segment*20+i for i,role in enumerate(roles)}
            p=campaign.base.make_plan(j2,LOCAL_PARENTS/tag,seeds=seeds)
            for key in ('expected_parent_global_step','parent_registration_sha256',
                        'parent_checkpoint_sha256','parent_audit_sha256'):
                p.pop(key)
            previous=(f'/home/yx789/nqs-wavefunctions/nqs/ct6_adaptive_production_20261028/{tag}/run01'
                if segment==1 else f'{DATA}/{tag}/segment{segment-1:02d}/run01')
            p.update(task=campaign.TASK,parent_directory=previous,parent_format='audited_adaptive_predecessor',
                action_chunk=4,H_chunk=4,automatic_next_allocation=True,
                campaign_segments_per_coupling=4,campaign_A100_hours_upper_per_coupling=96,
                resource_horizon='next_eight_attempt_block',parent_binding='completed_frozen_audited_predecessor_only',
                control_window_boundary='inherit_values_archive_history_reset_monitor_window',segment_index=segment,
                execution_authorized=True)
            p=json.loads(json.dumps(p))  # Canonical tuple/list representation before idempotence check.
            path=CONFIG/(name+'.json')
            if path.exists() and json.loads(path.read_text())!=p:raise ValueError('existing campaign plan differs')
            campaign.validate_plan(p,executing=True);atomic_json(path,p)
            jobs.append(dict(tag=tag,segment=segment,plan=str(path.relative_to(ROOT)),
                predecessor_job=748163 if tag=='ct6_j0' and segment==1 else
                    748164 if tag=='ct6_j0125' and segment==1 else 'previous_new_segment',
                parent_directory=previous,output=f'{DATA}/{tag}/segment{segment:02d}/run01'))
    if not (CONFIG/'run.sbatch').exists():raise FileNotFoundError('reviewed explicit launcher required')
    runtime=ROOT/'configs/remote/ct6_adaptive_production_20261028/runtime_packages.json'
    atomic_json(CONFIG/'runtime_packages.json',json.loads(runtime.read_text()))
    entries=[Path(campaign.__file__),Path(__file__),ROOT/'tests/test_ct6_adaptive_campaign.py',
        ROOT/'tests/test_ct6_adaptive_continuation.py',ROOT/'tests/test_ct_bounded_followup.py']
    source=transitive_local_source_manifest(entries,root=ROOT,package_prefixes=('su2_nqs','projects.clebsch_tree_nqs'))
    files=set(source)|{'AGENTS.md'}|{job['plan'] for job in jobs}
    files.update(str((CONFIG/n).relative_to(ROOT)) for n in ('run.sbatch','runtime_packages.json'))
    manifest={name:file_sha256(ROOT/name) for name in sorted(files)}
    atomic_json(CONFIG/'source_manifest.json',manifest)
    (CONFIG/'sync_files.txt').write_text('\n'.join(sorted(files|{str((CONFIG/'source_manifest.json').relative_to(ROOT))}))+'\n')
    report=dict(task='bounded_CT6_automatic_campaign',new_requested_hours_upper=192,
        per_coupling_hours_upper=96,segments_per_coupling=4,attempts_per_segment=32,
        requested_node=None,dependency='afterany_then_strict_audited_parent_gate',
        no_cancellation_of_existing_jobs=True,workspace=WORKSPACE,code_files=len(source),
        deployment_files=len(files)+1,source_manifest_sha256=file_sha256(CONFIG/'source_manifest.json'),jobs=jobs,
        gate_failure='clear_refusal_receipt_no_unsafe_training_no_DependencyNeverSatisfied',
        no_automatic_fifth_segment=True)
    out=ROOT/'projects/clebsch_tree_nqs/results/ct6_adaptive_campaign_20261029';out.mkdir(exist_ok=True)
    atomic_json(out/'prepared_campaign.json',report)
    return report


if __name__=='__main__':print(json.dumps(run(),indent=2))
