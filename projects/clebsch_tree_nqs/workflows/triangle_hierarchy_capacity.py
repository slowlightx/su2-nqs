"""Isolated physical-only upper-capacity screen after the frozen τ4 run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from .triangle_hierarchy_initialization import initialize_arm, native_bank_from_state
from .triangle_sector_transfer import load_source


def run(output: Path, active: int = 8, seconds_cap: float = 900., *, resume: bool = False) -> dict:
    """Retain all q0..10 with τ8 below root, within one fixed CPU deadline."""
    if active != 8 or not 60 <= seconds_cap <= 1200:
        raise ValueError("this follow-up registers width8 and a60..1200 second wallcap")
    output.mkdir(parents=True, exist_ok=True)
    registration = dict(task="full_target_CT12_physical_upper_capacity_followup",
        active_upper_width=active, couplings=(.125, 0.), arms=("physical",),
        model=dict(length=12, multiplicity_cap=20, maximum_q=10),
        workflow_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initializer_sha256=hashlib.sha256(Path(__file__).with_name("triangle_hierarchy_initialization.py").read_bytes()).hexdigest(),
        wall_seconds=seconds_cap, max_node_seconds=300, no_training=True, no_remote_submission=True)
    if (output/"registration.json").exists():
        if not resume or json.loads((output/"registration.json").read_text()) != registration:
            raise FileExistsError("an existing run needs an identical explicit resume")
    else:
        if any(output.iterdir()):
            raise FileExistsError("refusing to overwrite existing artifacts")
        atomic_json(output/"registration.json", registration)
    started = perf_counter(); reports = []
    for j2 in (.125, 0.):
        tag = "j0125" if j2 else "j0"
        source, provenance = load_source(j2)
        atomic_json(output/tag/"source.json", provenance)
        folder = output/tag/"physical"
        report = initialize_arm(source, j2, "physical", active, folder,
                                started+seconds_cap, resume=resume)
        if report["status"] == "complete":
            state = load_checkpoint(folder/"state.pkl")["params"]
            bank, embedding = native_bank_from_state(source, state)
            metadata = dict(model=registration["model"], j2=j2, arm="physical", source=provenance,
                raw_tree_energy_per_site=report["raw_tree_energy_per_site"], embedding=embedding,
                full_A1_energy_not_yet_audited=True, optimizer_reset_required=True)
            save_checkpoint(folder/"initialized_bank.pkl", bank, optimizer_state=None, metadata=metadata)
            atomic_json(folder/"native_metadata.json", metadata)
        reports.append(report)
        atomic_json(output/"results.json", dict(reports=reports, seconds=perf_counter()-started,
            physical_variance_not_from_compressed_H=True))
        if perf_counter() > started+seconds_cap-30:
            break
    return dict(reports=reports)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds-cap", type=float, default=900.)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(); run(a.output, seconds_cap=a.seconds_cap, resume=a.resume)
