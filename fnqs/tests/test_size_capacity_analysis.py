"""Independent checks of aggregate controls and signed paired contrasts."""

import numpy as np
import json
import shutil
import hashlib

from fnqs.exact import TFIMReference
from fnqs.artifacts import save_checkpoint, write_json
from fnqs.scripts.size_capacity_controls import declare as declare_training
from fnqs.scripts.analyze_size_capacity import grouped_summary, row_metrics, contrast, declare, analyze


def reference_row(n, h, chi_scale=1., split="development"):
    ref = TFIMReference(n, h)
    return {"n": n, "h": h, "split": split,
        "chi_total": chi_scale * ref.fidelity_susceptibility(),
        "chi_reference": ref.fidelity_susceptibility(), "chi_se": 0.,
        "energy_total": ref.energy(), "energy_se": 0.,
        "correlations": [ref.correlation(r) for r in range(n)],
        "correlation_se": [0.] * n}


def test_fixed_aggregate_retains_all_costs_and_the_worst_size():
    endpoints = [{"parameter_count": 1474,
        "training_work": {"hamiltonian_evaluations": 2000, "born_samples": 256000},
        "audit_born_draws": 8192 * 2, "checkpoint_sha256": str(n),
        "arm": f"fixed16_L{n}", "convergence": {},
        "rows": [reference_row(n, 1., scale, split) for split in ["train", "development"]]}
        for n, scale in [(4,1.), (6,1.), (8,1.3)]]
    result = grouped_summary(endpoints, [4,6,8])
    assert result["parameter_count"] == 4422
    assert result["training_work"] == {"hamiltonian_evaluations": 6000, "born_samples": 768000}
    np.testing.assert_allclose(result["development"]["chi_max_relative"], .3, atol=1e-15)
    np.testing.assert_allclose(result["development"]["chi_rms_relative"], .3/np.sqrt(3), atol=1e-15)
    assert not result["development"]["all_point_engineering_margins_pass"]


def test_uncertainty_margin_and_signed_error_do_not_cancel():
    first = reference_row(4, 1., .8)
    second = reference_row(6, 1., 1.2)
    first["chi_se"] = .06 * first["chi_reference"]
    metrics = row_metrics([first, second])
    np.testing.assert_allclose(metrics["chi_mean_signed_relative"], 0., atol=1e-15)
    np.testing.assert_allclose(metrics["chi_rms_relative"], .2, atol=1e-15)
    np.testing.assert_allclose(metrics["chi_max_relative_plus_2se"], .32, atol=1e-15)
    assert not metrics["all_point_engineering_margins_pass"]


def test_paired_reductions_preserve_per_seed_direction():
    before = row_metrics([reference_row(4, 1., .7)])
    after = row_metrics([reference_row(4, 1., 1.1)])
    np.testing.assert_allclose(contrast(before, after)["chi_max_relative"], .2, atol=1e-15)
    np.testing.assert_allclose(contrast(after, before)["chi_max_relative"], -.2, atol=1e-15)


def test_complete_orchestration_counts_each_shared_prefix_once(tmp_path):
    """Synthetic analytic rows test accounting, never neural accuracy."""
    directory = tmp_path / "synthetic_accounting_only"
    declare_training(directory, [71,83,97], 2, 2)
    declare(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    protocol_hash = hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest()
    for seed in protocol["seeds"]:
        for name, arm in protocol["arms"].items():
            root = directory / "runs" / name / f"seed_{seed}"
            signature = {"arm": arm, "seed": seed, "training_options": protocol["training_options"],
                "protocol_sha256": protocol_hash, "schedule_sha256": protocol["schedule_sha256"][str(seed)]}
            history = [{"step": i, "field_schedule_epoch": i-1,
                "tasks_evaluated": arm["tasks_per_epoch"], "sr_relative_residual": 1e-8,
                "gradient_norm": 0., "variance_per_site": 0., "max_variance_per_site": 0.,
                "effective_learning_rate": .5} for i in range(1,max(arm["milestones"])+1)]
            for step in arm["milestones"]:
                work = {**arm["endpoint_costs"][str(step)], "basis_state_evaluations": 0,
                        "sr_refinement_solves": 0}
                metadata = {"signature": signature, "source_sha256": protocol["source_sha256"],
                    "launcher_sha256": protocol["launcher_sha256"], "warm_start": None, "history": history[:step]}
                endpoint = root / f"step_{step}"
                save_checkpoint(endpoint / "checkpoint.npz", {"fixture": np.zeros(arm["parameter_count"])},
                    {"m": {}, "v": {}, "step": step, "work": work}, np.array([0,seed],dtype=np.uint32), metadata)
                rows = []
                for n in arm["sizes"]:
                    for split, fields in [("train",protocol["train_fields"]),("development",protocol["development_fields"])]:
                        for h in fields:
                            row = reference_row(n,h,split=split)
                            row.update(audit_samples=2,audit_independent_of_training=True)
                            rows.append(row)
                write_json(endpoint / "audit.json", {"arm": name,"seed": seed,"step": step,"signature": signature,
                    "source_sha256": protocol["source_sha256"],"launcher_sha256": protocol["launcher_sha256"],
                    "checkpoint_sha256": hashlib.sha256((endpoint / "checkpoint.npz").read_bytes()).hexdigest(),
                    "parameter_count": arm["parameter_count"],"training_work": work,"rows": rows,
                    "audit_born_draws": len(rows)*2,"reserved_confirmation_measured": False,"universality_established": False})
            shutil.copyfile(endpoint / "checkpoint.npz",root / "accepted.npz")
            write_json(root / "complete.json", {"profile_only": False,"source_sha256": protocol["source_sha256"],
                "launcher_sha256": protocol["launcher_sha256"]})
    result = analyze(directory)
    assert result["validated_endpoints"] == 21
    assert result["unique_training_trajectories"] == 15
    assert result["total_unique_training_work"]["hamiltonian_evaluations"] == 180
    assert result["total_unique_training_work"]["born_samples"] == 23040
    assert result["total_audit_born_draws"] == 3690
    assert not result["phase0_scientific_gate_opened"]
