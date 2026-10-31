"""Bounded read-only CT6 campaign mirror; never sample, evaluate H or submit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from math import comb
from pathlib import Path
import subprocess

import numpy as np

from su2_nqs.frontier.artifacts import atomic_json

BASE = "/home/yx789/nqs-wavefunctions/nqs/ct6_adaptive_campaign_20261029_v2"
REMOTE = r'''
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
base=Path('/home/yx789/nqs-wavefunctions/nqs/ct6_adaptive_campaign_20261029_v2')
packet=dict(checked_at_UTC=datetime.now(timezone.utc).isoformat(),remote_mutations=False,runs={})
for directory in sorted(base.glob('ct6*/segment*/run*')):
 tag='_'.join(directory.relative_to(base).parts)
 row=dict(directory=str(directory),json_files={},frozen_payloads={},checkpoint_files={})
 names=('registration.json','device_replay.json','cost.json','prepare_result.json',
  'replay_result.json','training_progress.json','train_result.json','endpoint_frozen.json',
  'audit_result.json','throughput_calibration.json','terminal_audit/audit.json')
 paths=[directory/name for name in names]
 paths+=sorted(directory.glob('step[0-9]*/transition.json'))
 paths+=sorted(directory.glob('monitor[0-9]*/decision.json'))
 for path in paths:
  if path.is_file():
   raw=path.read_bytes();name=str(path.relative_to(directory))
   row['json_files'][name]=dict(sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())
 if (directory/'audit_result.json').is_file():
  for name in ('terminal_audit/products.npz','terminal_audit/draw.npz'):
   path=directory/name
   if path.is_file():row['frozen_payloads'][name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
 for name in ('checkpoint.pkl','parent.pkl','provisional.pkl'):
  path=directory/name
  if path.is_file():row['checkpoint_files'][name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
 packet['runs'][tag]=row
for key,args in {
 'queue':['squeue','-u','yx789','-o','%.18i %.32j %.10T %.12M %.12l %.6D %R'],
 'accounting':['sacct','-S','2026-10-29','-u','yx789','-X','--format=JobID,JobName%40,State,Elapsed,ElapsedRaw,Timelimit,ExitCode','-P'],
}.items():
 reply=subprocess.run(['/usr/local/slurm/current/bin/'+args[0],*args[1:]],capture_output=True,text=True,timeout=25)
 lines=reply.stdout.splitlines();packet[key]=dict(returncode=reply.returncode,stdout='\n'.join([lines[0],*[line for line in lines[1:] if 'ct6' in line.lower()]]) if lines else '',stderr=reply.stderr)
packet['logs']={}
for path in sorted(Path('/home/yx789/nqs-ct6-adaptive-campaign-20261029-v2/logs').glob('*')):
 if path.is_file() and path.suffix in ('.out','.err'):
  with path.open('rb') as stream:stream.seek(max(0,path.stat().st_size-6000));tail=stream.read().decode(errors='replace')
  packet['logs'][path.name]=dict(bytes=path.stat().st_size,tail=tail)
print(json.dumps(packet,allow_nan=False))
'''


def digest(path: Path) -> str:
    """SHA256 of a frozen artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(output: Path) -> dict:
    """Mirror committed JSONs and completed audit NPZs into a fresh directory."""
    if output.exists():
        raise FileExistsError("fresh snapshot directory required")
    reply = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                            "unicorn", "python3 -"], input=REMOTE, text=True,
                           capture_output=True, check=True, timeout=90)
    packet = json.loads(reply.stdout)
    output.mkdir(parents=True)
    atomic_json(output / "snapshot.json", packet)
    for tag, row in packet["runs"].items():
        destination = output / tag
        for name, record in row["json_files"].items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(record["text"])
            if digest(path) != record["sha256"]:
                raise ArithmeticError("JSON mirror SHA mismatch: " + str(path))
        if row["frozen_payloads"]:
            args = ["rsync", "-aR", *[f"unicorn:{row['directory']}/./{name}"
                    for name in row["frozen_payloads"]], str(destination) + "/"]
            subprocess.run(args, check=True, timeout=180)
            for name, record in row["frozen_payloads"].items():
                if digest(destination / name) != record["sha256"]:
                    raise ArithmeticError("frozen NPZ mirror SHA mismatch: " + str(destination / name))
    return packet


def read(path: Path) -> dict:
    """Read a mirrored JSON artifact."""
    return json.loads(path.read_text())


def review(output: Path) -> dict:
    """Recompute frozen ratios with the independent NumPy moment oracle."""
    root=Path(__file__).resolve().parents[3]
    oracle_path=root/'projects/clebsch_tree_nqs/results/ct_status_20261028/ct6_independent_review/recompute.py'
    spec=importlib.util.spec_from_file_location('ct6_oct31_independent_moments',oracle_path)
    assert spec is not None and spec.loader is not None
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    chain_path=root/'projects/clebsch_tree_nqs/results/remote_status_20261029_evening/independent_review.py'
    spec=importlib.util.spec_from_file_location('ct6_oct31_record_chain',chain_path)
    assert spec is not None and spec.loader is not None
    chain=importlib.util.module_from_spec(spec);spec.loader.exec_module(chain)
    snapshot=read(output/'snapshot.json')
    registered_sources={tag:json.loads(row['json_files']['registration.json']['text'])['sources']
        for tag,row in snapshot['runs'].items() if 'registration.json' in row['json_files']}
    common_sources=next(iter(registered_sources.values()))
    assert all(sources==common_sources for sources in registered_sources.values())
    account={line.split('|')[0]:line.split('|') for line in snapshot['accounting']['stdout'].splitlines()[1:]}
    report=dict(checked_at_UTC=snapshot['checked_at_UTC'],read_only=True,runs={},segment_boundaries=[],
        oracle_sha256=digest(oracle_path),new_samples=0,new_H_queries=0,remote_jobs_changed=False,
        full_checkpoints_not_downloaded=True,checkpoint_parameters_optimizer_not_numerically_replayed=True)
    report['source_provenance']=dict(registered_entries_per_run=len(common_sources),
        all_eight_runs_same_frozen_manifest=True,
        sorted_JSON_manifest_sha256=hashlib.sha256(json.dumps(common_sources,sort_keys=True).encode()).hexdigest(),
        original_full_manifest_retained_in_each_registration=True,
        downloaded_JSONs=sum(len(row['json_files']) for row in snapshot['runs'].values()),
        downloaded_completed_NPZs=sum(len(row['frozen_payloads']) for row in snapshot['runs'].values()))
    for tag,remote in snapshot['runs'].items():
        directory=output/tag
        if not (directory/'training_progress.json').exists():continue
        reg=read(directory/'registration.json');plan=reg['plan'];progress=read(directory/'training_progress.json')
        row=dict(progress=chain.check_chain(directory,remote),controls=progress['controls'])
        steps=[read(directory/f'step{i}/transition.json') for i in range(1,progress['attempts']+1)]
        segment=int(tag.split('_segment')[1][:2]);coupling='j0125' if '_j0125_' in tag else 'j0'
        job=str((763123 if coupling=='j0125' else 763119)+segment-1)
        row['job_id']=job;row['slurm_accounting_fields']=account[job]
        row['job_A100_hours_observed']=int(account[job][4])/3600
        row['committed_transition_wall_seconds']=sum(step['wall_seconds'] for step in steps)
        row['committed_logical_work']={key:sum(step.get('work',{}).get(key,0) for step in steps)
            for key in ('fit_draws','fit_H_centers','fit_native_H_replay_centers','full_native_jacobian_rows',
                        'calibration_draws','calibration_bank_centers','native_amplitude_replay_centers')}
        row['work_count_limit']='Committed logical work is not the total process H count; first-attempt replay, profiling, monitoring and audit are separate. Job elapsed includes them; full ancestor cost is not reconstructed.'
        monitors=[read(path) for path in sorted(directory.glob('monitor[0-9]*/decision.json'))]
        row['monitoring_feedback_not_terminal_precision']=[dict(reason=m['reason'],monitor=m['monitor'],
            next_controls=m['next_controls'],terminal_audit_used=m['terminal_audit_used'],
            convergence_claimed=m['convergence_claimed']) for m in monitors]
        if (directory/'audit_result.json').exists():
            final=read(directory/'audit_result.json');frozen=read(directory/'endpoint_frozen.json')
            audit=read(directory/'terminal_audit/audit.json')
            assert final['status']=='complete_fixed_prefix_independent_audit' and audit['status']=='complete'
            assert final['terminal_audit']==audit and final['independent_from_training_geometry_and_adaptation']
            assert final['terminal_audit_used_to_adapt'] is False and audit['no_endpoint_selection']
            assert remote['checkpoint_files']['checkpoint.pkl']['sha256']==frozen['checkpoint_sha256']==final['checkpoint_sha256']
            assert audit['endpoint_params_sha256']==frozen['parameter_sha256']==final['parameter_sha256']==row['progress']['endpoint_parameter_sha256']
            assert audit['comparison_parent_global_step']==reg['parent_global_step']
            assert audit['seed']==plan['seeds']['audit'] and audit['statistics_seed']==plan['seeds']['audit_statistics']
            used={seed for step in steps for seed in (step['seeds'].values() if isinstance(step['seeds'],dict) else step['seeds'])}
            assert not used.intersection((audit['seed'],audit['statistics_seed']))
            with np.load(directory/'terminal_audit/products.npz') as data:
                rows,p,h,q,guide=[data[name] for name in ('rows','psi','hpsi','proposal','guide_components')]
            with np.load(directory/'terminal_audit/draw.npz') as data:assert np.array_equal(rows,data['rows'])
            assert rows.shape==(8192,36) and np.isin(rows,[0,1]).all() and np.all(rows.sum(1)==18)
            assert np.isfinite(p).all() and np.isfinite(h).all() and np.isfinite(q).all() and np.all(q>0)
            assert len(rows)==audit['records']==plan['audit_records'] and audit['scalar_H_centers']==2*len(rows)
            replay_q=(1-plan['epsilon'])*guide.real.mean((1,2))+plan['epsilon']/comb(36,18)
            q_error=float(np.max(abs(replay_q-q)/q));assert q_error<2e-13
            groups=audit['moments']['independent_groups'];assert groups==plan['audit_groups']==128
            computed=oracle.moments(p,h,q,groups)
            keys=('endpoint_values','candidate_minus_control','endpoint_iid_delta_SE','paired_iid_delta_SE',
                'endpoint_independent_block_jackknife_SE','paired_independent_block_jackknife_SE')
            differences={key:float(np.max(abs(np.asarray(computed[key])-np.asarray(audit['moments'][key])))) for key in keys}
            assert max(differences.values())<2e-10
            row['independent_audit']=dict(status='complete_frozen_terminal_independently_recomputed',
                records=len(rows),independent_groups=groups,parent_global_step=reg['parent_global_step'],
                endpoint_global_step=final['absolute_completed_steps'],parameter_count=reg['parameter_count'],
                parameter_sha256=frozen['parameter_sha256'],checkpoint_sha256=frozen['checkpoint_sha256'],
                optimizer_sha256=frozen['optimizer_sha256'],rng_sha256=frozen['rng_sha256'],
                moments=computed,numeric_differences=differences,maximum_numeric_difference=max(differences.values()),
                proposal_maximum_relative_difference=q_error,audit_scalar_H_centers=audit['scalar_H_centers'],
                audit_wall_seconds=audit['wall_seconds'],paired_energy_decrease_SE=-computed['candidate_minus_control'][0]/computed['paired_independent_block_jackknife_SE'][0],
                paired_variance_reduction_fraction=-computed['candidate_minus_control'][2]/computed['endpoint_values'][0][2],
                frozen_file_SHA_and_recorded_parameter_optimizer_rng_binding_verified=True,
                no_new_H_queries_or_sampling=True,full_checkpoint_not_downloaded=True)
        report['runs'][tag]=row
    for coupling in ('j0','j0125'):
        for segment in range(2,5):
            previous=output/f'ct6_{coupling}_segment{segment-1:02}_run01'
            current=output/f'ct6_{coupling}_segment{segment:02}_run01'
            if not (current/'registration.json').exists():continue
            frozen=read(previous/'endpoint_frozen.json');reg=read(current/'registration.json')
            for key in ('parameter','optimizer','rng'):
                assert frozen[f'{key}_sha256']==reg[f'original_{key}_sha256']
            assert frozen['absolute_completed_steps']==reg['parent_global_step']
            aliases={'checkpoint.pkl':'checkpoint.pkl','registration.json':'registration.json',
                'endpoint_frozen.json':'endpoint_frozen.json','audit.json':'terminal_audit/audit.json',
                'audit_result.json':'audit_result.json','products.npz':'terminal_audit/products.npz'}
            prior=snapshot['runs'][previous.name]
            for key,name in aliases.items():
                expected=(prior['checkpoint_files'][name]['sha256'] if key=='checkpoint.pkl'
                    else digest(previous/name))
                assert reg['parent_seals'][key]==expected
            report['segment_boundaries'].append(dict(coupling=coupling,from_segment=segment-1,to_segment=segment,
                parent_global_step=reg['parent_global_step'],parameter_optimizer_rng_and_six_parent_seals_verified=True))
    atomic_json(output/'independent_review.json',report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-existing",action="store_true")
    args = parser.parse_args()
    if args.review_existing:
        result=review(args.output)
        print(json.dumps({tag:dict(progress=row['progress'],audit=row.get('independent_audit',{}).get('moments',{}).get('endpoint_values'),
            paired=row.get('independent_audit',{}).get('moments',{}).get('candidate_minus_control'),job_hours=row['job_A100_hours_observed'])
            for tag,row in result['runs'].items()},indent=2))
        raise SystemExit(0)
    packet = fetch(args.output)
    print(json.dumps(dict(output=str(args.output),checked_at_UTC=packet["checked_at_UTC"],
        runs={tag:dict(json_files=len(row["json_files"]),frozen_payloads=len(row["frozen_payloads"]))
              for tag,row in packet["runs"].items()},queue=packet["queue"],accounting=packet["accounting"]),indent=2))
