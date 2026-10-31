"""Bounded read-only SSH collection and audited notebook maintenance.

No job submission, cancellation, training, or wavefunction modification occurs
in this monitor. A plan fixes its directories, exact deadline and cadence.
Only complete CT6 endpoints with independently audited, verified checkpoint
parameters can replace the notebook's corresponding scalar data arrays.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]

# Executed on the login node. This reads only explicitly registered project
# directories. It imports JAX on CPU only when a finished CT6 checkpoint needs
# a parameter seal, never evaluates H or creates a GPU/Slurm allocation.
COLLECTOR = r'''
import os,sys,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_ENABLE_X64']='true'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
plan=json.loads(sys.stdin.read())
packet={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'entries':[]}
for item in plan['sources']:
 p=Path(item['directory'])
 if not str(p).startswith('/home/yx789/nqs-wavefunctions/nqs/'):
  raise ValueError('outside declared wavefunction project directory')
 row={'id':item['id'],'kind':item['kind'],'directory':str(p),'documents':{},'file_sha256':{}}
 for name in ('registration.json','progress.json','results.json','terminal_audit.json','energy.json','process_replay.json'):
  f=p/name
  if f.is_file():
   raw=f.read_bytes();row['documents'][name]=json.loads(raw)
   row['file_sha256'][name]=hashlib.sha256(raw).hexdigest()
 try:
  result=row['documents'].get('results.json',{})
  if item['kind']=='ct6' and result.get('status')=='fixed_steps_complete' and 'terminal_audit.json' in row['documents']:
   sys.path.insert(0,plan['checkpoint_reader_workspace'])
   from su2_nqs.vmc.checkpoint import load_checkpoint
   from su2_nqs.frontier.artifacts import pytree_sha256
   cp=p/'checkpoint.pkl';before=hashlib.sha256(cp.read_bytes()).hexdigest()
   state=load_checkpoint(cp)
   after=hashlib.sha256(cp.read_bytes()).hexdigest()
   if before!=after:raise ValueError('checkpoint changed during collection')
   row['checkpoint']={'file_sha256':after,'params_sha256':pytree_sha256(state['params']),
      'registration_sha256':state['metadata']['registration_sha256'],
      'completed_steps':state['metadata']['completed_steps']}
 except Exception as error:row['verification_error']=repr(error)
 packet['entries'].append(row)
queue=subprocess.run(['/usr/local/slurm/current/bin/squeue','-u','yx789','-h','-o','%i|%j|%T|%M|%R'],text=True,capture_output=True,check=True)
packet['queue']=queue.stdout.splitlines()
print(json.dumps(packet,allow_nan=False))
'''


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verified_ct6(entry: dict, expected: dict) -> dict | None:
    """Reject incomplete/mismatched results; never promote a training mean."""
    docs = entry['documents']
    result, audit, reg = (docs.get(n, {}) for n in
        ('results.json', 'terminal_audit.json', 'registration.json'))
    if result.get('status') != 'fixed_steps_complete' or not audit:
        return None
    if entry.get('verification_error') or 'checkpoint' not in entry:
        raise ValueError('finished CT6 lacks a verified frozen checkpoint')
    digest = reg.get('sha256')
    if digest != canonical_sha256({k: v for k, v in reg.items() if k != 'sha256'}):
        raise ValueError('registration seal differs')
    config = reg['config']; cp = entry['checkpoint']
    if (config['model']['length'] != 6 or config['arm'] != expected['arm']
            or config['j2_over_j1'] != expected['j2_over_j1']
            or config['model']['multiplicity_cap'] != 16 or config['model']['maximum_q'] != 10):
        raise ValueError('wrong Hamiltonian, size, arity or retained widths')
    if (result.get('registration_sha256') != digest
            or audit.get('registration_sha256') != digest
            or cp['registration_sha256'] != digest
            or audit.get('params_sha256') != cp['params_sha256']
            or result.get('terminal_audit') != audit
            or result['completed_steps'] != cp['completed_steps']
            or result['completed_steps'] != config['steps']):
        raise ValueError('endpoint/audit/checkpoint identity differs')
    if (audit.get('independent_from_training') is not True
            or audit.get('samples', 0) < 4096 or audit.get('chains', 0) < 32
            or not result.get('checkpoint_rng_sampler_replay_exact')
            or not result.get('checkpoint_optimizer_exact')):
        raise ValueError('independent audit and complete checkpoint replay required')
    import math
    for key in ('energy_per_site', 'energy_chain_SE', 'variance_per_site'):
        if not math.isfinite(audit[key]):
            raise ValueError('nonfinite independent audit')
    if audit['energy_chain_SE'] < 0 or audit['variance_per_site'] < 0:
        raise ValueError('invalid independent uncertainty/variance')
    history = result.get('history', [])
    if len(history) != result['completed_steps']:
        raise ValueError('full work history required')
    return dict(id=entry['id'], arm=config['arm'], j2_over_j1=config['j2_over_j1'],
        steps=result['completed_steps'], energy=audit['energy_per_site'],
        SE=audit['energy_chain_SE'], variance=audit['variance_per_site'],
        H_centers=sum(x['full_H_calls'] for x in history),
        training_seconds=sum(x['seconds'] for x in history),
        checkpoint_sha256=cp['file_sha256'], params_sha256=cp['params_sha256'],
        directory=entry['directory'], mixing_certified=False)


def replace_assignment(source: str, name: str, value: str) -> str:
    """Change one existing editable data assignment, preserving plotting code."""
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
             and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id == name]
    if len(nodes) != 1:
        raise ValueError('expected exactly one notebook assignment: '+name)
    node = nodes[0]; lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+name+' = '+value+'\n'+''.join(lines[node.end_lineno:])


def _array_rows(source: str, name: str) -> list[list[float]]:
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)
    # Unmeasured accounting is explicitly np.nan, never a fabricated zero.
    # Accept only that literal attribute; do not eval arbitrary notebook code.
    return [[float('nan') if isinstance(value, ast.Attribute)
             and isinstance(value.value, ast.Name) and value.value.id == 'np'
             and value.attr == 'nan' else float(ast.literal_eval(value))
             for value in call.args[0].elts] for call in node.value.elts]


def _array_literal(rows: list[list[float]]) -> str:
    return '['+',\n    '.join('np.array(['+', '.join(
        'np.nan' if value != value else repr(value) for value in row)+'])' for row in rows)+']'


def update_notebook(document: dict, packet: dict, candidates: list[dict], baseline: dict) -> bool:
    """Mechanical inline-array update only; retain user omissions and styling."""
    changed = False
    data = next(c for c in document['cells'] if c.get('id') == 'tri-size-data')
    source = ''.join(data['source'])
    recorded = data.setdefault('metadata', {}).setdefault('nqs_monitor_endpoints', {})
    for row in candidates:
        if row['id'] not in baseline or 'energy' not in baseline[row['id']]:
            continue
        index = 0 if row['j2_over_j1'] == 0 else 1
        prefix = 'tri_binary' if row['arm'] == 'binary_truncated' else 'tri_joint'
        current = _array_rows(source, prefix+'_E')[index][0]
        expected = recorded.get(row['id'], baseline[row['id']])
        if current != expected['energy']:
            raise ValueError('manual energy edit detected; preserve user data: '+row['id'])
        if row['steps'] < expected['steps']:
            raise ValueError('monitor cannot replace a newer notebook endpoint')
    accounting = next(c for c in document['cells'] if c.get('id') == 'tri-gcnn-accounting-plot')
    account_source = ''.join(accounting['source'])
    notes = []
    for prefix, arm in (('tri_binary', 'binary_truncated'), ('tri_joint', 'ternary_joint')):
        for field, key in (('E', 'energy'), ('SE', 'SE')):
            name = prefix+'_'+field
            rows = _array_rows(source, name)
            original_rows = [row.copy() for row in rows]
            if len(rows) != 2 or any(len(row) != 1 for row in rows):
                raise ValueError('user changed CT array layout; leave notebook untouched')
            for row in candidates:
                if row['arm'] == arm and row['steps'] >= baseline[row['id']]['steps']:
                    index = 0 if row['j2_over_j1'] == 0 else 1
                    changed |= rows[index][0] != row[key]
                    rows[index][0] = row[key]
            if rows != original_rows:
                source = replace_assignment(source, name, _array_literal(rows))
    data['source'] = source.splitlines(keepends=True)
    for row in candidates:
        recorded[row['id']] = dict(steps=row['steps'], energy=row['energy'], SE=row['SE'])
    for name, key in (('complexity_H', 'H_centers'), ('complexity_E', 'energy'), ('complexity_SE', 'SE')):
        rows = _array_rows(account_source, name)
        original_rows = [row.copy() for row in rows]
        if len(rows) != 2 or any(len(row) != 4 for row in rows):
            raise ValueError('user changed accounting layout; leave notebook untouched')
        for row in candidates:
            if row['steps'] >= baseline[row['id']]['steps']:
                i = 0 if row['j2_over_j1'] == 0 else 1
                k = 2 if row['arm'] == 'binary_truncated' else 3
                rows[i][k] = row[key]
        if rows != original_rows:
            account_source = replace_assignment(account_source, name, _array_literal(rows))
    accounting['source'] = account_source.splitlines(keepends=True)
    for row in candidates:
        notes.append(f"{row['id']}: step{row['steps']}, E/N={row['energy']:.9f} ± {row['SE']:.9f}, variance/N={row['variance']:.7f}")
    status = next((c for c in document['cells'] if c.get('id') == 'tri-bounded-monitor-status'), None)
    if status is None:
        status = dict(cell_type='markdown', id='tri-bounded-monitor-status', metadata={}, source=[])
        document['cells'].insert(1, status)
    lines = ['### 有界远端监控\n', '\n', '检查时间：'+packet['checked_at_utc']+'（UTC）。\n',
        '只将冻结终态、参数身份与独立审计一致的数据更新到能量数组；排队、训练中和小样本12×12工程测试不进入主图。\n', '\n']
    lines += ['- '+note+'\n' for note in notes]
    if not notes:
        lines += ['本次尚无新的合格CT6终态审计，已有能量数组保持不变。\n']
    lines += ['\n其他任务状态（以下不是主图的新能量点）：\n', '\n']
    for entry in packet.get('entries', []):
        docs = entry['documents']; result = docs.get('results.json', {})
        progress = docs.get('progress.json', {})
        state = result.get('status', progress.get('status', '尚未生成结果'))
        step = progress.get('completed_steps', result.get('completed_steps'))
        suffix = '' if step is None else f', step={step}'
        lines.append(f"- {entry['id']}: {state}{suffix}\n")
        if entry['kind'] == 'attention' and result.get('arms'):
            for arm, value in result['arms'].items():
                audit = value.get('terminal_audit', {})
                if audit:
                    lines.append(f"  - {arm}: 独立工程审计 E/N={audit.get('energy_per_site')}, "
                        f"SE={audit.get('energy_chain_SE')}, variance/N={audit.get('variance_per_site')}; "
                        f"全比较/replay状态={state}/{docs.get('process_replay.json', {}).get('status', '待完成')}\n")
        if entry['kind'] == 'ct12_train' and result.get('arms'):
            for arm, value in result['arms'].items():
                audit = value.get('terminal_audit', {})
                if audit:
                    lines.append(f"  - {arm}: 小样本工程审计 E/N={audit.get('energy_per_site')}, "
                        f"SE={audit.get('energy_chain_SE')}; 不作为竞争性精度或相图证据\n")
    lines += ['\n全部12×12／attention状态及原始元数据见 `projects/clebsch_tree_nqs/results/monitor_20261019/`。\n']
    status['source'] = lines
    # Do not leave old output images paired with changed numeric inputs.
    if changed:
        for cell in document['cells']:
            if cell['cell_type'] == 'code':
                cell['outputs'] = []; cell['execution_count'] = None
    return changed


def run_once(plan: dict, *, root: Path = ROOT) -> dict:
    """Collect one bounded snapshot, verify evidence and refresh the notebook."""
    destination = root/plan['output']; destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    command = 'env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true '+shlex.quote(plan['remote_python'])+' -c '+shlex.quote(COLLECTOR)
    reply = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
        '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2', plan['host'], command],
        input=json.dumps(plan), text=True, capture_output=True, timeout=180, check=True)
    packet = json.loads(reply.stdout)
    candidates, rejected = [], []
    sources = {s['id']: s for s in plan['sources']}
    for entry in packet['entries']:
        if entry['kind'] != 'ct6':
            continue
        try:
            row = verified_ct6(entry, sources[entry['id']])
            if row is not None:
                candidates.append(row)
        except (ValueError, KeyError) as error:
            rejected.append(dict(id=entry['id'], reason=str(error)))
    snapshot = dict(packet, verified_CT6=candidates, rejected_CT6=rejected,
        no_training_or_remote_mutation=True)
    for row in candidates:
        reference = plan['baseline'][row['id']]
        row['energy_change_vs_previous_audit'] = row['energy']-reference['energy']
        row['combined_SE_unpaired'] = (row['SE']**2+reference['SE']**2)**.5
        row['variance_change_vs_previous_audit'] = row['variance']-reference['variance']
        row['energy_lower_by_two_combined_SE'] = row['energy_change_vs_previous_audit'] < -2*row['combined_SE_unpaired']
    (destination/(stamp+'.json')).write_text(json.dumps(snapshot, indent=2)+'\n')
    (destination/'latest.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    path = root/plan['notebook']; before = path.read_bytes()
    document = json.loads(before)
    changed = update_notebook(document, packet, candidates, plan['baseline'])
    if path.read_bytes() != before:
        raise RuntimeError('notebook changed during collection; preserve user edits')
    temporary = path.with_suffix('.monitor.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=1)+'\n')
    temporary.replace(path)
    if changed and plan.get('execute_notebook', False):
        updated_bytes = path.read_bytes()
        with tempfile.TemporaryDirectory(prefix='nqs-monitor-notebook-') as build:
            environment = dict(os.environ, IPYTHONDIR=str(Path(build)/'ipython'),
                MPLCONFIGDIR=str(Path(build)/'matplotlib'))
            run = subprocess.run([plan['local_python'], '-m', 'jupyter', 'nbconvert',
                '--to', 'notebook', '--execute', '--ExecutePreprocessor.kernel_name=nqs',
                '--ExecutePreprocessor.timeout=300', '--output-dir', build,
                '--output', path.name, str(path)], cwd=root, env=environment,
                capture_output=True, text=True, timeout=600)
            rendered = Path(build)/path.name
            if run.returncode != 0:
                (destination/(stamp+'_render_error.txt')).write_text(run.stderr)
            elif path.read_bytes() == updated_bytes:
                path.write_bytes(rendered.read_bytes())
            else:
                (destination/(stamp+'_rendered_conflict.ipynb')).write_bytes(rendered.read_bytes())
    summary = dict(checked_at_utc=packet['checked_at_utc'], promoted=candidates,
        rejected=rejected, energy_arrays_changed=changed)
    (destination/'last_update.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def watch(plan: dict) -> None:
    """Run at fixed offsets and stop at the explicit eight-hour deadline."""
    start = datetime.fromisoformat(plan['start_utc']).timestamp()
    end = datetime.fromisoformat(plan['stop_utc']).timestamp()
    if not 0 < end-start <= 8*3600 or plan['interval_seconds'] != 7200:
        raise ValueError('monitor must be bounded to eight hours/two-hour cadence')
    destination = ROOT/plan['output']; destination.mkdir(parents=True, exist_ok=True)
    launch = dict(status='running', pid=os.getpid(), start_utc=plan['start_utc'],
        stop_utc=plan['stop_utc'], interval_seconds=plan['interval_seconds'],
        launched_at_utc=datetime.now(timezone.utc).isoformat())
    (destination/'launch_status.json').write_text(json.dumps(launch, indent=2)+'\n')
    for offset in range(0, int(end-start)+1, plan['interval_seconds']):
        scheduled = start+offset
        if time.time() > scheduled+600:
            continue
        while time.time() < scheduled:
            time.sleep(max(0., min(30., scheduled-time.time())))
        try:
            print(json.dumps(run_once(plan)), flush=True)
        except Exception as error:
            record = dict(scheduled_utc=datetime.fromtimestamp(scheduled, timezone.utc).isoformat(),
                status='failed_snapshot_no_fabricated_update', error=repr(error))
            with (destination/'errors.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
    launch.update(status='completed_fixed_window', finished_at_utc=datetime.now(timezone.utc).isoformat())
    (destination/'launch_status.json').write_text(json.dumps(launch, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args(); specification = json.loads(args.plan.read_text())
    if args.watch:
        watch(specification)
    else:
        print(json.dumps(run_once(specification), indent=2))
