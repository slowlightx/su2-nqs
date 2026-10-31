"""Read-only CT6 adaptive/CT12 production snapshot; never query H or submit.

Partially written terminal products are not interpreted as audited energies.
Only already committed JSON reports and bounded log tails are copied locally.
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
 'ct6_j0':'/home/yx789/nqs-wavefunctions/nqs/ct6_adaptive_production_20261028/ct6_j0/run01',
 'ct6_j0125':'/home/yx789/nqs-wavefunctions/nqs/ct6_adaptive_production_20261028/ct6_j0125/run01',
 'ct12_j0125':'/home/yx789/nqs-wavefunctions/nqs/ct12_production32_20261028/j0125/run01',
}
for tag in ('ct6_j0','ct6_j0125'):
 for segment in range(1,5):
  roots[f'{tag}_auto{segment:02}']=(
   '/home/yx789/nqs-wavefunctions/nqs/ct6_adaptive_campaign_20261029_v2/'
   f'{tag}/segment{segment:02}/run01')
packet={'checked_at_UTC':datetime.now(timezone.utc).isoformat(),'remote_mutations':False,'runs':{}}
def read(p):
 raw=p.read_bytes()
 return {'sha256':hashlib.sha256(raw).hexdigest(),'document':json.loads(raw)}
for tag,directory in roots.items():
 p=Path(directory);e={'directory':directory,'documents':{},'steps':{},'monitors':{}}
 for name in ('registration.json','device_replay.json','cost.json','prepare_result.json',
              'replay_result.json','training_progress.json','train_result.json',
              'endpoint_frozen.json','audit_result.json','throughput_calibration.json',
              'terminal_audit/audit.json'):
  f=p/name
  if f.is_file():e['documents'][name]=read(f)
 for f in sorted(p.glob('step[0-9]*/transition.json')):
  e['steps'][str(f.relative_to(p))]=read(f)
 for f in sorted(p.glob('monitor[0-9]*/decision.json')):
  e['monitors'][str(f.relative_to(p))]=read(f)
 for f in sorted(p.glob('step[0-9]*/progress.json')):
  if not (f.parent/'transition.json').exists():
   e.setdefault('in_progress',{})[str(f.relative_to(p))]=read(f)
 e['checkpoint_files']={str(f.relative_to(p)):{'bytes':f.stat().st_size,'mtime':f.stat().st_mtime}
  for f in (p/'checkpoint.pkl',p/'provisional.pkl') if f.is_file()}
 packet['runs'][tag]=e
for name,args in {
 'queue':['squeue','-u','yx789','-o','%.18i %.14P %.34j %.10T %.12M %.12l %.6D %R'],
 'accounting':['sacct','-S','2026-10-28','-X','-j',
  '748163,748164,735301,763119,763120,763121,763122,763123,763124,763125,763126',
  '--format=JobID,JobName%28,State,Elapsed,Timelimit,ExitCode','-P'],
}.items():
 r=subprocess.run(['/usr/local/slurm/current/bin/'+args[0],*args[1:]],capture_output=True,text=True,timeout=20)
 packet[name]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
packet['logs']={}
for name,path in {
 'ct6_j0.out':'/home/yx789/nqs-ct6-adaptive-production-20261028/logs/748163.out',
 'ct6_j0.err':'/home/yx789/nqs-ct6-adaptive-production-20261028/logs/748163.err',
 'ct6_j0125.out':'/home/yx789/nqs-ct6-adaptive-production-20261028/logs/748164.out',
 'ct6_j0125.err':'/home/yx789/nqs-ct6-adaptive-production-20261028/logs/748164.err',
 'ct12.out':'/home/yx789/nqs-ct12-production32-20261028/logs/production-735301.out',
 'ct12.err':'/home/yx789/nqs-ct12-production32-20261028/logs/production-735301.err',
}.items():
 p=Path(path)
 if p.is_file():
  with p.open('rb') as stream:
   stream.seek(max(0,p.stat().st_size-14000));tail=stream.read()
  packet['logs'][name]={'bytes':p.stat().st_size,'tail':tail.decode(errors='replace')}
for job in (763119,763120,763121,763122,763123,763124,763125,763126):
 for suffix in ('out','err'):
  p=Path(f'/home/yx789/nqs-ct6-adaptive-campaign-20261029-v2/logs/{job}.{suffix}')
  if p.is_file():
   with p.open('rb') as stream:
    stream.seek(max(0,p.stat().st_size-14000));tail=stream.read()
   packet['logs'][f'{job}.{suffix}']={'bytes':p.stat().st_size,'tail':tail.decode(errors='replace')}
print(json.dumps(packet,allow_nan=False))
'''


def run(output: Path) -> dict:
    """Save an immutable local snapshot of already-existing remote products."""
    if output.exists():
        raise FileExistsError('new snapshot directory required')
    reply = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
                            'unicorn','python3 -'], input=REMOTE, text=True,
                           capture_output=True, timeout=90, check=True)
    packet = json.loads(reply.stdout)
    output.mkdir(parents=True)
    atomic_json(output/'snapshot.json',packet)
    return packet


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); packet=run(args.output)
    print(json.dumps({'output':str(args.output),'checked_at_UTC':packet['checked_at_UTC'],
        'steps':{k:len(v['steps']) for k,v in packet['runs'].items()},
        'monitors':{k:len(v['monitors']) for k,v in packet['runs'].items()}}))
