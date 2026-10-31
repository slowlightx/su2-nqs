"""Conditional cold A/B/C retest of the identified coverage/budget remedy."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax
from fnqs.artifacts import load_checkpoint,write_json
from fnqs.variants.size_features import SizeFeatureConfig
from .engine import Options,init_model,train,model_signature
from .factorial import points,PRIMARY,SECONDARY,RESERVED,SEEDS,seal,endpoint_audit,metrics
from .lineage import ROOT,sha,verify_sources,validate_splits,cost,check_cost,save_node,read_node


def prepare_followup(directory: Path, factorial: Path) -> None:
    """Choose only by preregistered central development effects, then seal."""
    report=json.loads((factorial/'review.json').read_text());decision=report['decision']
    if not decision['identified_under_declared_budget']:
        raise ValueError('no consistent identified factor; conditional training is not triggered')
    prior=ROOT/'fnqs/results/fixed_size_pilots_v11/protocol.json';old=json.loads(prior.read_text())
    source_protocol=json.loads((factorial/'protocol.json').read_text());verify_sources(source_protocol['source_sha256'])
    width=.15 if decision['selected_window']=='wide' else .04
    cuts=[]
    for c in old['cuts']:
        center=c['finite_size_peak'];j2,jx=c['train'][0][1:3]
        tr=points(center,[-width,-width/2,0,width/2,width],j2,jx)
        primary,secondary,reserved=[points(center,x,j2,jx) for x in (PRIMARY,SECONDARY,RESERVED)]
        validate_splits(tr,primary+secondary,reserved)
        cuts.append(dict(family=c['family'],train=tr,primary=primary,secondary=secondary,reserved=reserved))
    runs=[]
    for seed in SEEDS:
        for name,chosen in [('ABC',cuts)]+[(c['family'],[c]) for c in cuts]:
            schedule=[[c['train'][i%5] for c in chosen] for i in range(600)]
            reused=(factorial/'trajectories'/f"{decision['selected_window']}_seed_{seed}"/'step_600/node.json') if name=='A' else None
            if reused is not None:read_node(reused)
            runs.append(dict(name=f'{name}_seed_{seed}',seed=seed,families=[c['family'] for c in chosen],
                train_points=sum((c['train'] for c in chosen),[]),primary_points=sum((c['primary'] for c in chosen),[]),
                secondary_points=sum((c['secondary'] for c in chosen),[]),schedule=schedule,cost=cost(schedule,64),
                reused_node=str(reused.resolve()) if reused else None,reused_node_sha256=sha(reused) if reused else None))
    seal(directory,dict(kind='identified_factor_solution_family_retest_v13',n=12,seeds=SEEDS,
        model=asdict(SizeFeatureConfig(8)),options=asdict(Options(max_step=.1)),runs=runs,cuts=cuts,
        selected_window=decision['selected_window'],selected_steps=600,
        original_input={str(factorial/'review.json'):sha(factorial/'review.json'),str(prior):sha(prior)},
        method_selection_is_development_only=True,cold_parameter_initialization=True,
        A_controls_reused_from_matching_terminal_trajectory=True,
        exclusive_new_training_cost=cost([x for r in runs if not r['reused_node'] for x in r['schedule']],64),
        aggregate_comparison_training_cost=cost([x for r in runs for x in r['schedule']],64),
        historical_factorial_method_selection_cost=report['actual_training_cost'],
        capacity_limit='ABC has one 546-parameter model versus three family-wise models; equal per-family queries/draws, different update and capacity totals.',
        unchanged_observable_limits=source_protocol['unchanged_observable_limits'],
        reserved_confirmation_measured=False,phase0_scientific_gate=False,universality_established=False,
        no_automatic_budget_extension=True))


def run_followup(directory: Path) -> dict:
    """Retest the remedy, without changing width, SR or per-task samples."""
    protocol=json.loads((directory/'protocol.json').read_text());verify_sources(protocol['source_sha256'])
    for name,digest in protocol['original_input'].items():
        if sha(Path(name))!=digest:raise ValueError('method-selection input changed')
    config=SizeFeatureConfig(**protocol['model']);options=Options(**protocol['options']);began=time.perf_counter()
    nodes={}
    for r in protocol['runs']:
        folder=directory/'runs'/r['name']
        if r['reused_node']:
            node=Path(r['reused_node'])
            if sha(node)!=r['reused_node_sha256']:raise ValueError('reused A control changed')
            data=read_node(node)[node.resolve()]
            check_cost(data['exclusive_work'],r['cost'])
            if data['train_points']!=r['train_points']:raise ValueError('A control schedule mismatch')
            nodes[r['name']]=node;continue
        node=folder/'node.json'
        if node.exists():read_node(node);nodes[r['name']]=node;continue
        if (folder/'started.json').exists():raise RuntimeError('unfinished solution trajectory preserved')
        write_json(folder/'started.json',dict(protocol_sha256=sha(directory/'protocol.json')))
        p=init_model(jax.random.PRNGKey(r['seed']),config)
        def callback(p,state,key,history):
            if state['step']%150==0:print(r['name'],'step',state['step'],flush=True)
        p,state,key,history=train(p,jax.random.PRNGKey(r['seed']+1000),12,r['schedule'],config,options,callback)
        check_cost(state['work'],r['cost'])
        save_node(folder,p,state,key,signature=model_signature(config,12),protocol_path=directory/'protocol.json',
                  train_points=r['train_points'],history=history,parents=[],stage='conditional_cold_family_solution')
        nodes[r['name']]=node
    write_json(directory/'endpoint_seal.json',dict(nodes={str(p):sha(p) for p in nodes.values()},reference_opened=False))
    rows=[]
    for r in protocol['runs']:
        # Reused A weights are imported into an audit-only copy with their
        # original ancestry intact; audit binding uses the original node.
        node=nodes[r['name']];folder=node.parent
        if r['reused_node']:
            p,_,_,_=load_checkpoint(folder/'checkpoint.npz')
            from .factorial import get_reference,audit
            points_out=[];out=directory/'runs'/r['name'];out.mkdir(parents=True,exist_ok=True)
            for split,point_list in [('primary',r['primary_points']),('secondary',r['secondary_points'])]:
                for g in point_list:
                    ref,digest=get_reference(directory,12,g)
                    result=audit(p,12,g,config,ref,out/f'point_{len(points_out)}.npz')
                    points_out.append({**result,'split':split,'reference_sha256':digest})
            write_json(out/'audit.json',dict(points=points_out,reused_node_sha256=sha(node),
                                           protocol_sha256=sha(directory/'protocol.json')))
        else:
            result=endpoint_audit(directory,folder,protocol,r['primary_points'],r['secondary_points'])
            points_out=result['points']
        for family in r['families']:
            cut=next(c for c in protocol['cuts'] if c['family']==family)
            chosen=[p for p in points_out if p['couplings'] in cut['primary']+cut['secondary']]
            primary=metrics(chosen);secondary=metrics(chosen,'secondary')
            limits=protocol['unchanged_observable_limits']
            passed=(primary['chi']<=limits['chi_relative_error'] and
                    primary['correlation']<=limits['max_correlation_absolute_error'] and
                    primary['energy']<=limits['energy_relative_error'])
            rows.append(dict(run=r['name'],seed=r['seed'],family=family,primary=primary,secondary=secondary,
                             small_size_development_observables_passed=passed))
    result=dict(rows=rows,all_central_development_observables_passed=all(r['small_size_development_observables_passed'] for r in rows),
        exclusive_new_training_cost=protocol['exclusive_new_training_cost'],
        aggregate_comparison_training_cost=protocol['aggregate_comparison_training_cost'],
        historical_factorial_method_selection_cost=protocol['historical_factorial_method_selection_cost'],
        audit_cost=dict(born_samples=0,basis_rows=324*4096,hamiltonian_actions=648,source_actions=324),
        reserved_confirmation_measured=False,phase0_scientific_gate=False,universality_established=False,
        protocol_sha256=sha(directory/'protocol.json'),elapsed_seconds=time.perf_counter()-began)
    write_json(directory/'review.json',result);return result
