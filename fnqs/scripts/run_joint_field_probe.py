"""Matched total-work sequential versus simultaneous coupling SR probe.

Only update grouping changes. Joint five-field batches use 120 updates,
each with five tasks; sequential controls use 600 one-task updates. Thus
field order, exposure, 64 draws/task and total H/Born work match. Per-update
batch noise, conditioning and number of parameter updates differ together.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import jax
from fnqs.artifacts import write_json,load_checkpoint
from fnqs.pilots.engine import Options,init_model,train,model_signature
from fnqs.pilots.factorial import points,PRIMARY,SECONDARY,RESERVED,SEEDS,endpoint_audit,metrics
from fnqs.pilots.lineage import ROOT,sha,sources,verify_sources,read_node,cost,check_cost,save_node,validate_splits
from fnqs.variants.size_features import SizeFeatureConfig


def declare(output: Path, data: dict) -> None:
    """Seal this orchestration and all imported numerical sources."""
    if output.exists():raise FileExistsError('use a fresh directory')
    paths=['fnqs/scripts/run_joint_field_probe.py','fnqs/scripts/run_coupling_factorial.py','fnqs/scripts/run_factorial_solution.py']
    manifest={**sources(),**{p:sha(ROOT/p) for p in paths}}
    output.mkdir(parents=True)
    for name in manifest:
        target=output/'runtime'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write_json(output/'protocol.json',{**data,'source_sha256':manifest})


def prepare(output: Path,factorial: Path) -> None:
    """Declare three joint trajectories and three reused matched controls."""
    old=json.loads((factorial/'protocol.json').read_text());verify_sources(old['source_sha256'])
    runs=[]
    for seed in SEEDS:
        t=next(t for t in old['trajectories'] if t['window']=='wide' and t['seed']==seed)
        control=factorial/'trajectories'/t['name']/'step_600/node.json';read_node(control)
        schedule=[t['train_points'] for _ in range(120)]
        expected=cost(schedule,64);controlcost=cost(t['schedule'],64)
        assert all(expected[k]==controlcost[k] for k in ('task_evaluations','born_samples','hamiltonian_actions','source_actions'))
        assert [g for step in schedule for g in step]==[g for step in t['schedule'] for g in step]
        runs.append(dict(name=f'joint_seed_{seed}',seed=seed,train_points=t['train_points'],schedule=schedule,cost=expected,
            primary_points=old['primary_points'],secondary_points=old['secondary_points'],
            control_node=str(control.resolve()),control_node_sha256=sha(control),
            control_audit=str(control.parent/'audit.json'),control_audit_sha256=sha(control.parent/'audit.json')))
    declare(output,dict(kind='matched_joint_field_probe_v13',n=12,seeds=SEEDS,model=old['model'],options=old['options'],runs=runs,
        input_bindings={str(factorial/'review.json'):sha(factorial/'review.json')},
        grouping_is_only_implementation_change=True,per_task_samples=64,per_update_samples_joint=320,per_update_samples_sequential=64,
        total_queries_and_draws_matched=True,number_of_updates_matched=False,
        criterion='ALL three paired chi worst errors improve >= .05, and none worsens full-C error by > .005.',
        conditional_family_retest_maximum_tasks=10800,conditional_family_joint_updates=120,
        reserved_confirmation_measured=False,formal_scientific_gate=False,universality_established=False))


def execute(output: Path) -> dict:
    """Run already declared trajectories; preserve failed/incomplete output."""
    protocol=json.loads((output/'protocol.json').read_text());verify_sources(protocol['source_sha256'])
    for name,digest in protocol['input_bindings'].items():
        if sha(Path(name))!=digest:raise ValueError('selection input changed')
    config=SizeFeatureConfig(**protocol['model']);options=Options(**protocol['options']);began=time.perf_counter()
    for r in protocol['runs']:
        folder=output/'runs'/r['name'];node=folder/'node.json'
        if node.exists():read_node(node);continue
        if (folder/'started.json').exists():raise RuntimeError('unfinished run preserved')
        write_json(folder/'started.json',dict(protocol_sha256=sha(output/'protocol.json')))
        p=init_model(jax.random.PRNGKey(r['seed']),config)
        def callback(p,state,key,history):
            if state['step']%40==0:print(r['name'],'step',state['step'],flush=True)
        p,state,key,history=train(p,jax.random.PRNGKey(r['seed']+1000),12,r['schedule'],config,options,callback)
        check_cost(state['work'],r['cost'])
        save_node(folder,p,state,key,signature=model_signature(config,12),protocol_path=output/'protocol.json',
                  train_points=r['train_points'],history=history,parents=[],stage='joint_cold_parameter_updates')
    write_json(output/'endpoint_seal.json',dict(nodes={r['name']:sha(output/'runs'/r['name']/'node.json') for r in protocol['runs']},reference_opened=False))
    rows=[]
    for r in protocol['runs']:
        folder=output/'runs'/r['name'];audited=endpoint_audit(output,folder,protocol,r['primary_points'],r['secondary_points'])
        row=dict(run=r['name'],seed=r['seed'],primary=metrics(audited['points']),secondary=metrics(audited['points'],'secondary'))
        if 'control_node' in r:
            node=Path(r['control_node']);a=Path(r['control_audit'])
            if sha(node)!=r['control_node_sha256'] or sha(a)!=r['control_audit_sha256']:raise ValueError('sequential control changed')
            read_node(node);control=json.loads(a.read_text())
            row['sequential_primary']=metrics(control['points']);row['sequential_secondary']=metrics(control['points'],'secondary')
            row['chi_improvement']=row['sequential_primary']['chi']-row['primary']['chi']
            row['correlation_change']=row['primary']['correlation']-row['sequential_primary']['correlation']
        if 'families' in r:
            row['families']={}
            for family in r['families']:
                c=next(c for c in protocol['cuts'] if c['family']==family)
                point_rows=[p for p in audited['points'] if p['couplings'] in c['primary']+c['secondary']]
                row['families'][family]=dict(primary=metrics(point_rows),secondary=metrics(point_rows,'secondary'))
        rows.append(row)
    consistent=(all(r['chi_improvement']>=.05 and r['correlation_change']<=.005 for r in rows) if 'control_node' in protocol['runs'][0] else None)
    point_count=sum(len(r['primary_points'])+len(r['secondary_points']) for r in protocol['runs'])
    result=dict(rows=rows,joint_grouping_consistent_improvement=consistent,
        exclusive_training_cost=cost([task for r in protocol['runs'] for task in r['schedule']],64),
        audit_cost=dict(born_samples=0,basis_rows=point_count*4096,hamiltonian_actions=2*point_count,source_actions=point_count),
        protocol_sha256=sha(output/'protocol.json'),elapsed_seconds=time.perf_counter()-began,
        reserved_confirmation_measured=False,formal_scientific_gate=False,universality_established=False)
    write_json(output/'review.json',result);print('joint grouping consistent improvement',consistent,flush=True);return result


def prepare_families(output: Path,probe: Path) -> None:
    """Apply the grouping remedy to cold ABC/shared and family-wise controls."""
    review=json.loads((probe/'review.json').read_text())
    if not review['joint_grouping_consistent_improvement']:raise ValueError('joint-update remedy not supported')
    oldpath=ROOT/'fnqs/results/fixed_size_pilots_v11/protocol.json';old=json.loads(oldpath.read_text())
    previous=json.loads((probe/'protocol.json').read_text());verify_sources(previous['source_sha256'])
    cuts=[]
    for c in old['cuts']:
        center=c['finite_size_peak'];j2,jx=c['train'][0][1:3]
        tr=points(center,[-.15,-.075,0,.075,.15],j2,jx)
        primary,secondary,reserved=[points(center,x,j2,jx) for x in (PRIMARY,SECONDARY,RESERVED)]
        validate_splits(tr,primary+secondary,reserved)
        cuts.append(dict(family=c['family'],train=tr,primary=primary,secondary=secondary))
    runs=[]
    for seed in SEEDS:
        for name,chosen in [('ABC',cuts)]+[(c['family'],[c]) for c in cuts]:
            fields=sum((c['train'] for c in chosen),[])
            schedule=[fields for _ in range(120)]
            runs.append(dict(name=f'{name}_seed_{seed}',seed=seed,families=[c['family'] for c in chosen],train_points=fields,
                primary_points=sum((c['primary'] for c in chosen),[]),secondary_points=sum((c['secondary'] for c in chosen),[]),
                schedule=schedule,cost=cost(schedule,64)))
    declare(output,dict(kind='conditional_joint_family_solution_v14',n=12,seeds=SEEDS,model=previous['model'],options=previous['options'],
        runs=runs,cuts=cuts,input_bindings={str(probe/'review.json'):sha(probe/'review.json'),str(oldpath):sha(oldpath)},
        selected_method='simultaneous multi-field SR, wide window',
        per_family_total_tasks=600,total_comparison_tasks=10800,A_controls_retrained_and_cost_counted=True,
        capacity_limit='Shared ABC one model versus three family models; query/draw matching does not match aggregate parameters.',
        reserved_confirmation_measured=False,formal_scientific_gate=False,universality_established=False))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','prepare-families'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--parent',type=Path)
    args=parser.parse_args()
    if args.command=='run':execute(args.output)
    elif args.parent is None:parser.error('--parent required for declaration')
    elif args.command=='prepare':prepare(args.output,args.parent)
    else:prepare_families(args.output,args.parent)

if __name__=='__main__':main()
