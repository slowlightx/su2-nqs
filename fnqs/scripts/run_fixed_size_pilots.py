"""Seal and run bounded fixed-L coupling pilots and matched response repair."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax
import numpy as np

from fnqs.artifacts import load_checkpoint, write_json, source_hashes
from fnqs.exact import TFIMReference
from fnqs.pilots.engine import Options, init_model, train, model_signature
from fnqs.pilots.lineage import (sha, seal_protocol, verify_sources, cost, check_cost,
                               save_node, read_node, validate_splits, ROOT)
from fnqs.pilots.measurements import reference, exact_audit, born_audit
from fnqs.variants.size_features import SizeFeatureConfig, parameter_counts


def points(j2, jx, center, offsets):
    return [[1.,j2,jx,round(center+o,8)] for o in offsets]


def prepare(output: Path) -> None:
    refpath = ROOT/"fnqs/results/phase1_parity_reference_v5/reference.json"
    oldpath = ROOT/"fnqs/results/size_capacity_controls_v8/protocol.json"
    references, old = json.loads(refpath.read_text()), json.loads(oldpath.read_text())
    cuts = []
    for name in ("A","B_0.1","C_0.1"):
        cut = next(c for c in references["cuts"] if c["name"]==name and c["n"]==12)
        center = max(cut["rows"],key=lambda r:r["chi_total"])["h"]
        tr = points(cut["j2"],cut["jx"],center,[-.04,-.02,0,.02,.04])
        dev = points(cut["j2"],cut["jx"],center,[-.03,-.01,.01,.03])
        reserved = points(cut["j2"],cut["jx"],center,[-.035,-.015,.015,.035])
        validate_splits(tr,dev,reserved)
        cuts.append(dict(name=name,family=name[0],finite_size_peak=center,
                         train=tr,development=dev,reserved=reserved))
    runs = []
    for seed in (113,127,139):
        for label, chosen in [("ABC",cuts)]+[(c["family"],[c]) for c in cuts]:
            schedule = [[c["train"][step%5] for c in chosen] for step in range(150)]
            runs.append(dict(track="phase1",name=f"{label}_seed_{seed}",seed=seed,n=12,
                model=asdict(SizeFeatureConfig(8)),options=asdict(Options(max_step=.1)),
                schedule=schedule,train_points=sum((c["train"] for c in chosen),[]),
                audit_points=[dict(couplings=g,split=split,family=c["family"]) for c in chosen
                              for split in ("train","development") for g in c[split]],
                cost=cost(schedule,64),parents=[]))
    parent_receipts = []
    for seed in (71,83,97):
        folder = ROOT/f"fnqs/results/size_capacity_controls_v8/runs/fixed16_L16/seed_{seed}"
        cp = folder/"step_1000/checkpoint.npz"
        complete = folder/"complete.json"
        parent_receipts.append(dict(seed=seed,checkpoint=str(cp),checkpoint_sha256=sha(cp),
                                   complete=str(complete),complete_sha256=sha(complete)))
        train_points = [[1.,0.,0.,h] for h in old["train_fields"]]
        audit = [[1.,0.,0.,h] for h in (.952,.977,1.002,1.027,1.047)]
        for label,weight in (("energy",0.),("response",.02)):
            schedule = [[train_points[(2*step)%21],train_points[(2*step+1)%21]] for step in range(50)]
            runs.append(dict(track="phase0",name=f"{label}_seed_{seed}",seed=seed,n=16,
                model=asdict(SizeFeatureConfig(16,size_features="legacy")),
                options=asdict(Options(response_weight=weight)),schedule=schedule,
                train_points=train_points,audit_points=[dict(couplings=g,split="development",family="A") for g in audit],
                cost=cost(schedule,64),parents=[seed]))
    seal_protocol(output,dict(kind="bounded_fixed_size_pilots_v11",cuts=cuts,runs=runs,
        parent_receipts=parent_receipts,inputs={str(p):sha(p) for p in (refpath,oldpath)},
        phase0_parent_protocol=str(oldpath),phase0_parent_source_sha256=old["source_sha256"],
        phase0_parent_train_points=[[1.,0.,0.,h] for h in old["train_fields"]],
        phase0_parent_selection_points=[[1.,0.,0.,h] for h in old["development_fields"]],
        phase0_reserved_confirmation_fields=old["reserved_confirmation_fields"],
        phase0_audit_samples=4096,phase0_baseline_audit_points=[r for r in runs if r["track"]=="phase0"][0]["audit_points"],
        no_automatic_budget_extension=True,reference_labels_in_training=False,
        phase1_fairness="Same 150 per-family tasks, 64 draws/task; ABC has one 546-weight model, controls three. Updates differ: 150 versus aggregate 450.",
        phase0_fairness="Same parent, reset SR state, same key/schedule/draws/max step; only response_weight differs. Both compute response diagnostic.",
        parameter_counts=parameter_counts(SizeFeatureConfig(8)),
        phase0_penalty="rho_response squared; Born-measure derivative by frozen-proposal self-normalized importance weights; finite-batch biased.",
        predeclared_response_weight=.02,universality_established=False,scientific_gate_passed=False,
        reserved_confirmation_measured=False,large_size_BC_class_confirmed=False))
    print("sealed",output/"protocol.json",flush=True)


def import_parent(output: Path, receipt: dict, protocol: dict) -> Path:
    directory = output/"parents"/f"seed_{receipt['seed']}"
    if (directory/"node.json").exists():
        read_node(directory/"node.json")
        return directory/"node.json"
    cp, complete = Path(receipt["checkpoint"]),Path(receipt["complete"])
    if sha(cp)!=receipt["checkpoint_sha256"] or sha(complete)!=receipt["complete_sha256"]:
        raise ValueError("legacy parent receipt mismatch")
    p,state,key,m = load_checkpoint(cp)
    completed = json.loads(complete.read_text())
    if source_hashes()!=protocol["phase0_parent_source_sha256"] or m["source_sha256"]!=source_hashes():
        raise ValueError("legacy runtime mismatch")
    if m["model"]!={"hidden_dim":16,"size_scale":64} or m["seed"]!=receipt["seed"] or state["step"]!=1000:
        raise ValueError("wrong legacy model/seed/endpoint")
    if completed["signature"]!=m["signature"] or completed["kind"]!="completed_control_trajectory":
        raise ValueError("legacy completion mismatch")
    if state["work"]["hamiltonian_evaluations"]!=2000 or state["work"]["born_samples"]!=256000:
        raise ValueError("legacy cost mismatch")
    legacy_history = m["history"]
    if len(legacy_history)!=1000:
        raise ValueError("legacy history incomplete")
    state["work"] = dict(updates=1000,task_evaluations=2000,hamiltonian_actions=2000,
        source_actions=0,born_samples=256000,basis_rows=0,
        sr_correction_solves=state["work"].get("sr_refinement_solves",0))
    return save_node(directory,p,state,key,signature=model_signature(SizeFeatureConfig(16,size_features="legacy"),16),
        protocol_path=output/"protocol.json",train_points=protocol["phase0_parent_train_points"],history=legacy_history,
        parents=[],stage="verified_v8_import",selection_points=protocol["phase0_parent_selection_points"])


def run(output: Path, track: str) -> None:
    started = time.perf_counter()
    protocol_path = output/"protocol.json"
    protocol = json.loads(protocol_path.read_text())
    verify_sources(protocol["source_sha256"])
    for name,digest in protocol["inputs"].items():
        if sha(Path(name))!=digest:
            raise ValueError("input artifact identity mismatch")
    cache = {}
    def get_ref(n,g):
        label = (n,*g)
        if label not in cache:
            if n==16:
                ref = TFIMReference(n,g[3])
                cache[label] = dict(energy_total=ref.energy(),chi_total=ref.fidelity_susceptibility(),c_half=ref.correlation(n//2))
            else:
                cache[label] = reference(n,g)
        return cache[label]
    chosen = [r for r in protocol["runs"] if track=="both" or r["track"]==track]
    for run_config in chosen:
        name,seed,n = run_config["name"],run_config["seed"],run_config["n"]
        directory = output/"runs"/name
        if (directory/"audit.json").exists():
            read_node(directory/"node.json")
            continue
        cfg = SizeFeatureConfig(**run_config["model"])
        parents = []
        if run_config["track"]=="phase0":
            receipt = next(r for r in protocol["parent_receipts"] if r["seed"]==seed)
            parent = import_parent(output,receipt,protocol)
            parents = [parent]
            p,_,_,_ = load_checkpoint(parent.parent/"checkpoint.npz")
            baseline_path = output/"parents"/f"seed_{seed}"/"audit.json"
            if not baseline_path.exists():
                baseline = [born_audit(p,n,x["couplings"],cfg,seed*10000+i,4096,get_ref(n,x["couplings"]),
                            baseline_path.parent/f"point_{i}_rows.npz")
                            for i,x in enumerate(run_config["audit_points"])]
                write_json(baseline_path,dict(points=baseline,kind="development_baseline",checkpoint_sha256=sha(parent.parent/"checkpoint.npz")))
        else:
            p = init_model(jax.random.PRNGKey(seed),cfg)
        if not (directory/"node.json").exists():
            if (directory/"started.json").exists():
                raise RuntimeError("unfinished trajectory preserved; use a newly sealed protocol for a retry")
            write_json(directory/"started.json",dict(protocol_sha256=sha(protocol_path),run=name))
            begin = time.perf_counter()
            print("start",name,flush=True)
            def callback(p,state,key,history):
                if state["step"]%25==0:
                    print(name,"step",state["step"],"rho",history[-1]["true_sr_residual"],flush=True)
            p,state,key,history = train(p,jax.random.PRNGKey(seed+1000),n,run_config["schedule"],cfg,
                Options(**run_config["options"]),callback)
            check_cost(state["work"],run_config["cost"])
            save_node(directory,p,state,key,signature=model_signature(cfg,n),protocol_path=protocol_path,
                train_points=run_config["train_points"],history=history,parents=parents,stage=run_config["track"])
            write_json(directory/"timing.json",dict(training_seconds=time.perf_counter()-begin))
        else:
            read_node(directory/"node.json")
            p,_,_,_ = load_checkpoint(directory/"checkpoint.npz")
        points_out = []
        for i,x in enumerate(run_config["audit_points"]):
            g = x["couplings"]
            result = (exact_audit(p,n,g,cfg,get_ref(n,g)) if n==12 else
                      born_audit(p,n,g,cfg,seed*10000+i,4096,get_ref(n,g),directory/f"point_{i}_rows.npz"))
            points_out.append({**result, "split":x["split"],"family":x["family"]})
        write_json(directory/"audit.json",dict(points=points_out,checkpoint_sha256=sha(directory/"checkpoint.npz"),
                                                protocol_sha256=sha(protocol_path)))
        print("done",name,flush=True)
    summarize(output)
    print("elapsed",time.perf_counter()-started,flush=True)


def summarize(output: Path) -> None:
    protocol = json.loads((output/"protocol.json").read_text())
    rows = []
    for r in protocol["runs"]:
        directory = output/"runs"/r["name"]
        if not (directory/"audit.json").exists():
            continue
        read_node(directory/"node.json")
        points_out = json.loads((directory/"audit.json").read_text())["points"]
        for family in sorted({p["family"] for p in points_out}):
            p = [x for x in points_out if x["family"]==family and x["split"]=="development"]
            rows.append(dict(run=r["name"],track=r["track"],family=family,seed=r["seed"],
                worst_chi_relative_error=max(x["chi_relative_error"] for x in p),
                worst_chi_relative_error_plus_2se=max(x.get("chi_relative_error_plus_2se",x["chi_relative_error"]) for x in p),
                worst_energy_per_site_error=max(x["energy_per_site_error"] for x in p),
                worst_c_half_error=max(x.get("c_half_absolute_error_plus_2se",x.get("c_half_absolute_error",0)) for x in p),
                worst_response_relative_residual=max(x["response_relative_residual"] for x in p)))
    write_json(output/"summary.json",dict(rows=rows,complete_runs=len({r["run"] for r in rows}),
        expected_runs=len(protocol["runs"]),scientific_gate_passed=False,universality_established=False,
        reserved_confirmation_measured=False,protocol_sha256=sha(output/"protocol.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run","summarize"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--track",choices=("phase0","phase1","both"),default="both")
    args = parser.parse_args()
    {"prepare":lambda:prepare(args.output),"run":lambda:run(args.output,args.track),
     "summarize":lambda:summarize(args.output)}[args.command]()


if __name__=="__main__":
    main()
