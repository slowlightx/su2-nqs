"""One read-only SSH snapshot of the explicitly registered CT continuations.

No submission, cancellation, GPU allocation, H query or remote write. Partial
terminal-audit products are deliberately not read as scientific conclusions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from su2_nqs.frontier.artifacts import atomic_json

REMOTE = r'''
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
roots={
 'ct6_j0':'/home/yx789/nqs-wavefunctions/nqs/ct6_verified_continuation_20261028/ct6_j0/run01',
 'ct6_j0125':'/home/yx789/nqs-wavefunctions/nqs/ct6_verified_continuation_20261028/ct6_j0125/run01',
 'ct12_j0125':'/home/yx789/nqs-wavefunctions/nqs/ct12_production32_20261028/j0125/run01',
}
packet={'checked_at_UTC':datetime.now(timezone.utc).isoformat(),'remote_mutations':False,'runs':{}}
def read(p):
 raw=p.read_bytes()
 return {'sha256':hashlib.sha256(raw).hexdigest(),'document':json.loads(raw)}
for tag,directory in roots.items():
 p=Path(directory);entry={'directory':directory,'exists':p.is_dir(),'documents':{},'steps':{}}
 for name in ('registration.json','device_replay.json','cost.json','prepare_result.json','replay_result.json',
              'training_progress.json','train_result.json','endpoint_frozen.json','audit_result.json',
              'interim_decision.json','interim_audit/audit.json','throughput_calibration.json'):
  f=p/name
  if f.is_file():entry['documents'][name]=read(f)
 for f in sorted(p.glob('step[0-9]*/transition.json')):
  entry['steps'][str(f.relative_to(p))]=read(f)
 audit=p/'terminal_audit/audit.json'
 if audit.is_file():entry['documents']['terminal_audit/audit.json']=read(audit)
 entry['checkpoint_files']={str(f.relative_to(p)):{'bytes':f.stat().st_size,'mtime':f.stat().st_mtime}
  for f in (p/'checkpoint.pkl',p/'provisional.pkl') if f.is_file()}
 entry['incomplete_terminal_products_present']=(p/'terminal_audit/partial_products.json').is_file()
 packet['runs'][tag]=entry
plan=Path('/home/yx789/nqs-ct12-production32-20261028/configs/remote/ct12_production32_20261028/j0125.json')
if plan.is_file():
 a=json.loads(plan.read_text())['software_acceptance']
 certificate=Path(a['directory'])/'gpu/gpu_check_result.json'
 packet['B200_certificate_path']=str(certificate)
 if certificate.is_file():packet['B200_certificate']=read(certificate)
for name,args in {
 'queue':['squeue','-u','yx789','-o','%.18i %.14P %.34j %.10T %.12M %.12l %.6D %R'],
 'accounting':['sacct','-X','-u','yx789','-S','2026-10-28','-j','733588,733589,735189,735301','--format=JobID,JobName%28,State,Elapsed,Timelimit,ExitCode,AllocTRES%90','-P'],
}.items():
 result=subprocess.run(['/usr/local/slurm/current/bin/'+args[0],*args[1:]],capture_output=True,text=True,timeout=20)
 packet[name]={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
logs={
 'ct6_j0':('/home/yx789/nqs-ct6-verified-continuation-20261028/logs/ct6_j0-733588.out','/home/yx789/nqs-ct6-verified-continuation-20261028/logs/ct6_j0-733588.err'),
 'ct6_j0125':('/home/yx789/nqs-ct6-verified-continuation-20261028/logs/ct6_j0125-733589.out','/home/yx789/nqs-ct6-verified-continuation-20261028/logs/ct6_j0125-733589.err'),
 'ct12_j0125':('/home/yx789/nqs-ct12-production32-20261028/logs/production-735301.out','/home/yx789/nqs-ct12-production32-20261028/logs/production-735301.err'),
}
packet['logs']={}
for tag,paths in logs.items():
 packet['logs'][tag]={}
 for path in paths:
  p=Path(path)
  if p.is_file():
   with p.open('rb') as stream:
    stream.seek(max(0,p.stat().st_size-20000));tail=stream.read()
   packet['logs'][tag][p.name]={'bytes':p.stat().st_size,'tail':tail.decode(errors='replace')}
print(json.dumps(packet,allow_nan=False))
'''


def run(output: Path) -> dict:
    """Save immutable local evidence, never an incomplete audit estimate."""
    if output.exists():
        raise FileExistsError('new local snapshot output required')
    reply = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
                            'unicorn','python3 -'],input=REMOTE,text=True,
                           capture_output=True,timeout=90,check=True)
    packet = json.loads(reply.stdout)
    output.mkdir(parents=True)
    atomic_json(output/'snapshot.json',packet)
    return packet


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    packet=run(args.output)
    print(json.dumps({'output':str(args.output),'checked_at_UTC':packet['checked_at_UTC'],
                      'run_documents':{k:list(v['documents']) for k,v in packet['runs'].items()}}))
