from __future__ import annotations

from pathlib import Path

from projects.clebsch_tree_nqs.run import _build_parser, main
from projects.clebsch_tree_nqs.workflows import spin_scale_first_carrier_single_seed


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "projects/clebsch_tree_nqs/configs/remote"
    / "spin_scale_first_carrier_single_seed.json"
)
SBATCH = (
    ROOT
    / "projects/clebsch_tree_nqs/remote"
    / "scale_first_carrier_single_seed_a100.sbatch"
)


def test_carrier_single_seed_cli_registers_bounded_call() -> None:
    parsed = _build_parser().parse_args(
        [
            "spin-scale-first-carrier-single-seed",
            "--config",
            str(CONFIG),
            "--case-id",
            "triangular_6x6_j2_0.125",
            "--arm-id",
            "exact_su2_balanced_schur_spatial_vit",
            "--output",
            "unused",
            "--maximum-steps-this-call",
            "1",
        ]
    )
    assert parsed.maximum_steps_this_call == 1


def test_carrier_single_seed_cli_forwards_bounded_call(monkeypatch, tmp_path) -> None:
    observed = {}

    def fake_run(config, output, **kwargs):
        observed.update(config=config, output=output, **kwargs)

    monkeypatch.setattr(
        spin_scale_first_carrier_single_seed,
        "run_spin_scale_first_carrier_single_seed",
        fake_run,
    )
    output = tmp_path / "slice"
    assert main(
        [
            "spin-scale-first-carrier-single-seed",
            "--config",
            str(CONFIG),
            "--case-id",
            "triangular_6x6_j2_0.125",
            "--arm-id",
            "fixed_sz_spatial_vit",
            "--output",
            str(output),
            "--maximum-steps-this-call",
            "1",
        ]
    ) == 0
    assert observed == {
        "config": CONFIG,
        "output": output,
        "case_id": "triangular_6x6_j2_0.125",
        "arm_id": "fixed_sz_spatial_vit",
        "maximum_steps_this_call": 1,
        "overwrite": False,
    }


def test_carrier_single_seed_job_has_cross_submission_resume_root() -> None:
    source = SBATCH.read_text()
    assert "${NQS_OUTPUT_ROOT:-" in source
    assert "--maximum-steps-this-call" in source
    assert "production_budget_released" in source
