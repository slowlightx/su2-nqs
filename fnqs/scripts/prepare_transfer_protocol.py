"""Seal Phase 2 fold definitions and readiness without launching transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fnqs.artifacts import write_json
from fnqs.experiments import provenance
from fnqs.hamiltonians import IsingCouplings
from fnqs.staging import assess_stages, candidate_family, validate_family_holdout
from fnqs.training import Task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "protocol.json").exists():
        raise FileExistsError("use a new protocol output directory")
    phase0 = json.loads(args.phase0.read_text())
    reference = json.loads(args.references.read_text())
    catalogue = []
    for name in sorted({cut["name"] for cut in reference["cuts"]}):
        cut = next(cut for cut in reference["cuts"] if cut["name"] == name)
        g = IsingCouplings(j2=cut["j2"], jx=cut["jx"])
        family = candidate_family(Task(16, g))
        catalogue.append({"name":name,"family":family,"j2":cut["j2"],"jx":cut["jx"]})
    fields = [round(.35 + .05 * i, 8) for i in range(31)]
    confirmation = [round(.375 + .05 * i, 8) for i in range(30)]
    assert not set(fields) & set(confirmation)
    sizes, held_sizes = [16, 24, 32, 48, 64], [40]
    folds = []
    for held in ["A", "B", "C"]:
        train_cuts = [cut for cut in catalogue if cut["family"] != held]
        held_cuts = [cut for cut in catalogue if cut["family"] == held]
        train = [Task(n, IsingCouplings(j2=cut["j2"], jx=cut["jx"], h=h), cut["family"])
                 for cut in train_cuts for n in sizes for h in fields]
        selection = [Task(n, IsingCouplings(j2=cut["j2"], jx=cut["jx"], h=h), cut["family"])
                     for cut in train_cuts for n in sizes for h in confirmation]
        validate_family_holdout(train, selection, held)
        adaptation_pool = [dict(cut, h=h) for cut in held_cuts for h in fields]
        fewshot = {}
        for k in [1, 2, 4, 8]:
            indices = ([len(adaptation_pool) // 2] if k == 1 else
                       np.rint(np.linspace(0, len(adaptation_pool) - 1, k)).astype(int).tolist())
            fewshot[str(k)] = [adaptation_pool[index] for index in indices]
        folds.append({"held_family":held,"pretraining_families":sorted({t.family for t in train}),
            "pretraining_cuts":train_cuts,"held_cuts":held_cuts,
            "training_task_count":len(train),"model_selection_task_count":len(selection),
            "holdout_validated_from_physical_couplings":True,
            "all_family_phase1_checkpoint_allowed":False,
            "phase0_A_checkpoint_as_parent_allowed":held != "A",
            "ancestor_policy":"Every parameter ancestor must exclude the held family, including Phase 0 for held A.",
            "fewshot_total_distinct_coupling_points":fewshot})
    config = {"model":{"hidden_dim":16,"size_scale":64},"sizes":sizes,"held_sizes":held_sizes,
              "train_fields":fields,"confirmation_fields":confirmation,"seeds":[17,29,43],
              "fewshot_point_counts":[1,2,4,8]}
    protocol = {**provenance(config),"kind":"phase2_transfer_protocol_only","catalogue":catalogue,"folds":folds,
        "readiness":assess_stages(phase0),"transfer_experiments_started":False,
        "inputs":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.phase0,args.references]},
        "launcher_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_classes_confirmed":False,"large_L_independent_reference_available":False,
        "arms":["frozen_zero_shot","fewshot_1_2_4_8","same_architecture_scratch","pointwise_nqs"],
        "followup_cost_contract":{"gradient_hamiltonians":6000,"born_draws":768000,
            "applies_to":"each optimized comparison arm; zero-shot uses no optimization",
            "pretraining":"Report ancestral queries, exact enumerations and Born draws separately and in total.",
            "pointwise":"Sum training cost and parameter counts over all Hamiltonian-specific models.",
            "scheduler":"Must seal exact updates/minibatches to meet these totals before execution."},
        "fewshot_selection":"Fixed broad coupling-grid coverage, sealed before training; counts total points across the held family.",
        "confirmation_excludes_all_adaptation_fields":True,
        "model_selection_policy":"Use only pretraining families; held reference values are confined to final physics auditing.",
        "metrics":["energy","variance","correlations","chi_total","hc","nu","eta","fit_stability","cost_to_accuracy"],
        "execution_missing":["Phase 0 independent-initialization reproduction","Phase 1 reproduction",
            "large-L B/C references and class verification","training runner with ancestor and exact-cost validation"]}
    write_json(args.output / "protocol.json", protocol)
    print({"folds":len(folds),"readiness":protocol["readiness"],"training_started":False})


if __name__ == "__main__":
    main()
