"""Executable three-fold transfer engineering canary with transitive auditing.

This small n=6 budget is a runner test, not a Phase 2 science campaign.
Every optimized arm has eight task evaluations/256 draws. Few-shot and
scratch match both task schedule and initialization seed; only ancestry
differs. Pointwise controls train directly at four audit Hamiltonians and
therefore are reported as supervised per-Hamiltonian controls.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax

from fnqs.artifacts import load_checkpoint, write_json
from fnqs.pilots.engine import Options, init_model, train, model_signature
from fnqs.pilots.lineage import (seal_protocol, cost, check_cost, verify_sources, verify_holdout,
    inclusive_cost, save_node, read_node, sha, validate_splits)
from fnqs.pilots.measurements import reference, exact_audit
from fnqs.variants.size_features import SizeFeatureConfig, parameter_counts


def prepare(output: Path):
    cfg = SizeFeatureConfig(4)
    cuts = {"A":(0.,0.,.96),"B":(.1,0.,1.115),"C":(0.,.1,.8)}
    catalog = {family:[[1.,j2,jx,round(h+d,8)] for d in (-.07,-.05,-.03,-.01,.01,.03,.05,.07)]
               for family,(j2,jx,h) in cuts.items()}
    tests = {family:[[1.,j2,jx,round(h+d,8)] for d in (-.06,-.02,.02,.06)]
             for family,(j2,jx,h) in cuts.items()}
    folds = []
    for held in "ABC":
        seen = [f for f in "ABC" if f!=held]
        pretrain = [[catalog[f][i] for f in seen] for i in range(4)]
        pool = catalog[held]
        validate_splits(pool,tests[held],[])
        arms = []
        for k in (1,2,4,8):
            indices = {1:[4],2:[0,7],4:[0,2,5,7],8:list(range(8))}[k]
            selected = [pool[i] for i in indices]
            schedule = [selected for _ in range(8//k)]
            for prefix in ("fewshot","scratch"):
                arms.append(dict(name=f"{prefix}_{k}",kind=prefix,shots=k,
                                 train_points=selected,schedule=schedule,cost=cost(schedule,32)))
        folds.append(dict(held_family=held,seen_families=seen,pretrain_schedule=pretrain,
            pretrain_points=[g for row in pretrain for g in row],pretrain_cost=cost(pretrain,32),
            held_audit_points=tests[held],arms=arms,
            pointwise=[dict(train_points=[g],schedule=[[g],[g]],cost=cost([[g],[g]],32)) for g in tests[held]]))
    seal_protocol(output,dict(kind="phase2_runner_engineering_canary_v11",n=6,seeds=[151],
        model=asdict(cfg),options=asdict(Options(samples=32,max_step=.1)),folds=folds,
        fewshot_counts=[1,2,4,8],scientific_transfer_campaign_started=False,
        universality_established=False,large_size_BC_class_confirmed=False,
        all_family_phase1_parent_allowed=False,holdout_checked_transitively=True,
        budget_contract="8 tasks / 16 H actions / 256 Born draws per optimized arm; pointwise aggregates four two-task models.",
        parameter_count_per_model=parameter_counts(cfg),
        pointwise_policy="Training directly at audit Hamiltonians; supervised per-point control, no held-field claim.",
        costs_include_unique_parameter_ancestors=True,held_reference_labels_only_in_final_audit=True,
        no_model_selection_no_automatic_budget_extension=True))
    print("sealed",output/"protocol.json",flush=True)


def execute_node(output, directory, cfg, options, n, seed, schedule, train_points, parents, stage, expected):
    if (directory/"node.json").exists():
        read_node(directory/"node.json")
        return directory/"node.json"
    if (directory/"started.json").exists():
        raise RuntimeError("failed/incomplete run preserved; declare a new protocol")
    write_json(directory/"started.json",dict(stage=stage,seed=seed))
    params = (load_checkpoint(Path(parents[0]).parent/"checkpoint.npz")[0] if parents else
              init_model(jax.random.PRNGKey(seed),cfg))
    params,state,key,history = train(params,jax.random.PRNGKey(seed+1000),n,schedule,cfg,options)
    check_cost(state["work"],expected)
    return save_node(directory,params,state,key,signature=model_signature(cfg,n),
        protocol_path=output/"protocol.json",train_points=train_points,history=history,
        parents=parents,stage=stage)


def run(output: Path):
    protocol = json.loads((output/"protocol.json").read_text())
    verify_sources(protocol["source_sha256"])
    cfg,options = SizeFeatureConfig(**protocol["model"]),Options(**protocol["options"])
    n = protocol["n"]
    summary, all_endpoints = [], []
    began = time.perf_counter()
    for seed in protocol["seeds"]:
        for fold in protocol["folds"]:
            held = fold["held_family"]
            base = output/f"held_{held}"/f"seed_{seed}"
            parent = execute_node(output,base/"pretrain",cfg,options,n,seed,
                fold["pretrain_schedule"],fold["pretrain_points"],[],"pretrain",fold["pretrain_cost"])
            verify_holdout([parent],held)
            # Seal ALL endpoints before exposing any held reference or metric.
            endpoints = [("frozen_zero_shot",parent,fold["held_audit_points"])]
            for arm in fold["arms"]:
                parents = [parent] if arm["kind"]=="fewshot" else []
                verify_holdout(parents,held)
                validate_splits(arm["train_points"],fold["held_audit_points"],[])
                node = execute_node(output,base/arm["name"],cfg,options,n,seed,arm["schedule"],
                    arm["train_points"],parents,"adaptation" if parents else "scratch",arm["cost"])
                endpoints.append((arm["name"],node,fold["held_audit_points"]))
            pointwise_nodes = []
            for i,arm in enumerate(fold["pointwise"]):
                node = execute_node(output,base/f"pointwise_{i}",cfg,options,n,seed,arm["schedule"],
                    arm["train_points"],[],"pointwise",arm["cost"])
                pointwise_nodes.append(node)
                endpoints.append((f"pointwise_{i}",node,arm["train_points"]))
            seal = {str(node):sha(node) for _,node,_ in endpoints}
            write_json(base/"endpoint_seal.json",dict(nodes=seal,held_reference_opened=False))
            ref = {tuple(g):reference(n,g) for g in fold["held_audit_points"]}
            for name,node,points in endpoints:
                if sha(node)!=seal[str(node)]:
                    raise ValueError("endpoint changed before audit")
                params,_,_,_ = load_checkpoint(node.parent/"checkpoint.npz")
                audits = [exact_audit(params,n,g,cfg,ref[tuple(g)]) for g in points]
                ledger = inclusive_cost([node])
                write_json(base/f"{name}_audit.json",dict(points=audits,inclusive_training_cost=ledger,
                    node_sha256=sha(node),held_reference_opened_after_endpoint_seal=True))
                summary.append(dict(held_family=held,seed=seed,arm=name,
                    worst_chi_relative_error=max(x["chi_relative_error"] for x in audits),
                    worst_energy_per_site_error=max(x["energy_per_site_error"] for x in audits),
                    inclusive_training_cost=ledger))
                all_endpoints.append(node)
            check_cost(inclusive_cost(pointwise_nodes)["work"],cost([[0]]*8,32))
            print("audited fold",held,flush=True)
    write_json(output/"summary.json",dict(rows=summary,protocol_sha256=sha(output/"protocol.json"),
        aggregate_training_cost=inclusive_cost(all_endpoints),
        audit_cost=dict(born_samples=0,basis_rows=sum(64*(1 if r["arm"].startswith("pointwise") else 4) for r in summary),
                        hamiltonian_actions=sum(2*(1 if r["arm"].startswith("pointwise") else 4) for r in summary)),
        engineering_canary_complete=True,scientific_transfer_campaign_started=False,universality_established=False,
        elapsed_seconds=time.perf_counter()-began))
    print("canary complete",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run"))
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    (prepare if args.command=="prepare" else run)(args.output)


if __name__=="__main__":
    main()
