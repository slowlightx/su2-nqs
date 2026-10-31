"""Fail-closed checkpoint ancestry, physical holdouts and additive work."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

from fnqs.artifacts import load_checkpoint, save_checkpoint, write_json
from fnqs.hamiltonians import IsingCouplings
from fnqs.staging import candidate_family
from fnqs.training import Task

ROOT = Path(__file__).resolve().parents[2]
WORK_KEYS = ("updates", "task_evaluations", "hamiltonian_actions", "source_actions",
             "born_samples", "basis_rows", "sr_correction_solves")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources() -> dict[str, str]:
    """Include numerical imports and launchers, preserving old source seals."""
    paths = list((ROOT/"fnqs").glob("*.py"))
    for sub in ("pilots", "variants", "diagnostics"):
        paths += list((ROOT/"fnqs"/sub).glob("*.py"))
    paths += [ROOT/"fnqs/scripts/run_fixed_size_pilots.py", ROOT/"fnqs/scripts/run_transfer_pilot.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def verify_sources(expected: dict) -> None:
    if not expected or any(not (ROOT/p).is_file() or sha(ROOT/p) != digest for p,digest in expected.items()):
        raise ValueError("runtime source identity mismatch")


def seal_protocol(directory: Path, protocol: dict) -> dict:
    """Refuse overwrite; snapshot sources before any training or auditing."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("use a fresh output directory")
    protocol = {**protocol, "source_sha256": sources()}
    directory.mkdir(parents=True)
    for name in protocol["source_sha256"]:
        destination = directory/"runtime"/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    write_json(directory/"protocol.json",protocol)
    return protocol


def cost(schedule: list, samples: int) -> dict:
    tasks = sum(map(len,schedule))
    return dict(updates=len(schedule),task_evaluations=tasks,hamiltonian_actions=2*tasks,
                source_actions=tasks,born_samples=samples*tasks,basis_rows=0)


def check_cost(actual: dict, expected: dict) -> None:
    if any(actual.get(k) != v for k,v in expected.items()):
        raise ValueError("declared exclusive work mismatch")


def save_node(directory: Path, params: dict, state: dict, key, *, signature: dict,
              protocol_path: Path, train_points: list, history: list, parents: list,
              stage: str, selection_points: list | None = None) -> Path:
    """Seal an accepted final checkpoint, optimizer/PRNG, tasks and parents."""
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/"node.json").exists():
        raise FileExistsError("node already sealed")
    protocol = json.loads(protocol_path.read_text())
    verify_sources(protocol["source_sha256"])
    metadata = dict(signature=signature, protocol_path=str(protocol_path.resolve()),
                    protocol_sha256=sha(protocol_path), source_sha256=protocol["source_sha256"],
                    train_points=train_points, selection_points=selection_points or [],
                    parents=[dict(path=str(Path(p).resolve()),sha256=sha(p)) for p in parents],
                    stage=stage, history=history)
    checkpoint = directory/"checkpoint.npz"
    save_checkpoint(checkpoint,params,state,key,metadata)
    node = {**metadata, "checkpoint_path":str(checkpoint.resolve()),
            "checkpoint_sha256":sha(checkpoint), "exclusive_work":state["work"],
            "complete":True}
    write_json(directory/"node.json",node)
    return directory/"node.json"


def read_node(path: Path, active: set | None = None, found: dict | None = None) -> dict:
    """Verify transitive hashes, source identity, metadata and feature policy."""
    active, found = (set() if active is None else active), ({} if found is None else found)
    path = Path(path).resolve()
    if path in active:
        raise ValueError("ancestry cycle")
    if path in found:
        return found
    active.add(path)
    node = json.loads(path.read_text())
    if node.get("complete") is not True or not node.get("source_sha256"):
        raise ValueError("incomplete or unsigned ancestor")
    verify_sources(node["source_sha256"])
    cp, protocol = Path(node["checkpoint_path"]), Path(node["protocol_path"])
    if sha(cp) != node["checkpoint_sha256"] or sha(protocol) != node["protocol_sha256"]:
        raise ValueError("checkpoint/protocol hash mismatch")
    params, state, _, metadata = load_checkpoint(cp)
    for k in ("signature", "source_sha256", "train_points", "selection_points", "parents", "stage",
              "protocol_path", "protocol_sha256", "history"):
        if metadata.get(k) != node.get(k):
            raise ValueError(f"checkpoint metadata mismatch: {k}")
    if state["work"] != node["exclusive_work"] or state["step"] != len(node["history"]):
        raise ValueError("optimizer/work/history mismatch")
    if node["stage"] == "verified_v8_import":
        declaration = json.loads(protocol.read_text())
        receipts = declaration.get("parent_receipts",[])
        receipt = next((r for r in receipts if path.parent.name==f"seed_{r['seed']}"),None)
        if receipt is None or sha(Path(receipt["checkpoint"]))!=receipt["checkpoint_sha256"] or sha(Path(receipt["complete"]))!=receipt["complete_sha256"]:
            raise ValueError("legacy ancestry receipt mismatch")
        old_params,old_state,_,old_meta=load_checkpoint(Path(receipt["checkpoint"]))
        import numpy as np
        if old_meta["history"]!=node["history"] or any(not np.array_equal(params[k],old_params[k]) for k in params):
            raise ValueError("imported parameters/history differ from legacy ancestor")
        if node["train_points"]!=declaration["phase0_parent_train_points"] or node["selection_points"]!=declaration["phase0_parent_selection_points"]:
            raise ValueError("legacy family declarations mismatch")
    else:
        declared = {tuple(g) for g in node["train_points"]}
        actual = {tuple(g) for row in node["history"] for g in row["tasks"]}
        if actual != declared or [row["step"] for row in node["history"]] != list(range(1,state["step"]+1)):
            raise ValueError("declared points disagree with actual task history")
        if node["exclusive_work"].get("updates") != state["step"]:
            raise ValueError("update count mismatch")
        tasks = sum(len(row["tasks"]) for row in node["history"])
        if node["exclusive_work"].get("task_evaluations") != tasks:
            raise ValueError("task count mismatch")
    for value in node["exclusive_work"].values():
        if isinstance(value,bool) or not isinstance(value,int) or value < 0:
            raise ValueError("work must contain nonnegative integer counts")
    for parent in node["parents"]:
        p = Path(parent["path"])
        if sha(p) != parent["sha256"]:
            raise ValueError("ancestor manifest hash mismatch")
        read_node(p,active,found)
        if found[p.resolve()]["signature"] != node["signature"]:
            raise ValueError("ancestor feature/size signature mismatch")
    found[path] = node
    active.remove(path)
    return found


def verify_holdout(parents: list[Path], held_family: str) -> dict:
    """Every parameter ancestor must exclude the entire held physical family."""
    if held_family not in ("A","B","C"):
        raise ValueError("unknown held family")
    nodes = {}
    for parent in parents:
        read_node(parent,found=nodes)
    for node in nodes.values():
        n = node["signature"]["n"]
        for g in node["train_points"]+node["selection_points"]:
            if candidate_family(Task(n,IsingCouplings(*g))) == held_family:
                raise ValueError("held physical family occurs in a parameter ancestor")
    return nodes


def inclusive_cost(endpoints: list[Path]) -> dict:
    """Deduplicate shared ancestors by content identity, not by display label."""
    nodes = {}
    for path in endpoints:
        read_node(path,found=nodes)
    unique = {node["checkpoint_sha256"]:node for node in nodes.values()}
    return {"unique_checkpoints":len(unique), "work":{
        k:sum(node["exclusive_work"].get(k,0) for node in unique.values()) for k in WORK_KEYS}}


def validate_splits(train: list, development: list, reserved: list) -> None:
    """Validate physical task points and reject overlap, including aliases."""
    groups = []
    for points in (train,development,reserved):
        labels = set()
        for g in points:
            Task(4,IsingCouplings(*g))
            candidate_family(Task(4,IsingCouplings(*g)))
            label = tuple(round(float(x),12) for x in g)
            if label in labels:
                raise ValueError("duplicate physical coupling point")
            labels.add(label)
        groups.append(labels)
    if any(groups[i]&groups[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError("physical field leakage")
