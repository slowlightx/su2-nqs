"""One-shot, explicitly authorized pair of serial 90-minute A100 audits."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess

WORK = Path('/home/yx789/nqs-ct6-midpoint-audit-20261023')
SLURM = '/usr/local/slurm/current/bin'
receipt = json.loads((WORK / 'capture_receipt.json').read_text())
if receipt['new_requested_A100_hours'] != 3 or receipt['aggregate_reserved_A100_hours'] > 240:
    raise ValueError('two-audit allocation exceeds frozen budget')
if datetime.now(timezone.utc) + timedelta(hours=3) > datetime.fromisoformat(receipt['deadline_utc']):
    raise RuntimeError('audit pair cannot fit inside unchanged campaign deadline')
if hashlib.sha256((WORK / 'audit_driver.py').read_bytes()).hexdigest() != receipt['wrapper_sha256']:
    raise ValueError('reviewed audit wrapper changed')
with (WORK / 'submission_started.json').open('x') as handle:
    json.dump(dict(started_utc=datetime.now(timezone.utc).isoformat(),
                   policy='ambiguous failure never automatically resubmits'), handle)
(WORK / 'logs').mkdir(exist_ok=True)
jobs = []
for tag in ('j0125', 'j0'):
    row = receipt['cases'][tag]
    args = [SLURM + '/sbatch', '--parsable', '--partition=kim', '--account=kim',
            '--job-name=ct6_midpoint_' + tag, '--nodes=1', '--ntasks=1', '--cpus-per-task=4',
            '--mem=96G', '--gres=gpu:nvidia_a100_80gb_pcie:1', '--nodelist=kim-compute-04',
            '--time=01:30:00', '--no-requeue', '--nice=1000', '--kill-on-invalid-dep=yes',
            '--chdir=' + receipt['source_root'],
            '--output=' + str(WORK / 'logs' / (tag + '-%j.out')),
            '--error=' + str(WORK / 'logs' / (tag + '-%j.err')),
            '--export=ALL,NQS_AUDIT_PLAN=' + row['plan_path'] + ',NQS_AUDIT_SHA256=' + row['plan_sha256']]
    if jobs:
        args += ['--dependency=afterany:' + jobs[0]['job_id']]
    args += [str(WORK / 'run.sbatch')]
    process = subprocess.run(args, text=True, capture_output=True)
    record = dict(case=tag, command=args, returncode=process.returncode,
                  stdout=process.stdout, stderr=process.stderr,
                  requested_A100_hours=1.5, captured_snapshot=row['snapshot'],
                  seed=row['seed'], plan_sha256=row['plan_sha256'])
    with (WORK / (tag + '_submission.json')).open('x') as handle:
        json.dump(record, handle, indent=2)
    reply = process.stdout.strip().split(';')[0]
    if process.returncode or not reply.isdigit():
        raise RuntimeError('submission failed or ambiguous; inspect persisted receipt, do not retry blindly')
    record['job_id'] = reply
    jobs.append(record)
    result = dict(receipt, status='submitted' if len(jobs) == 2 else 'first_submitted', jobs=jobs,
                  submitted_utc=datetime.now(timezone.utc).isoformat(),
                  original_training_jobs_unchanged=['518104', '518107'],
                  runner_sha256=hashlib.sha256((WORK / 'run.sbatch').read_bytes()).hexdigest(),
                  submitter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    temporary = WORK / 'submission_receipt.tmp'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(WORK / 'submission_receipt.json')
    print(json.dumps(dict(case=tag, job_id=reply, time_limit='01:30:00', requested_GPUs=1,
                         dependency=None if len(jobs) == 1 else 'afterany:' + jobs[0]['job_id'])), flush=True)
