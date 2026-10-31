"""Exercise the real Slurm launcher with a CPU-only, non-JAX Python stub."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "projects/clebsch_tree_nqs/configs/remote/triangle_quality_20261017/run.sbatch"


@pytest.mark.parametrize(
    ("kind", "phase", "stage"),
    [
        ("backend", "unused", "0"),
        ("spatial", "pilot", "0"),
        ("spatial", "replay", "0"),
        ("spatial", "train", "0"),
        ("spatial", "audit", "0"),
        ("spatial", "aggregate", "0"),
        ("clebsch", "unused", "0"),
        ("clebsch", "unused", "1"),
        ("couplings", "pilot", "0"),
        ("couplings", "continue", "0"),
    ],
)
def test_quality_runner_overrides_inherited_cpu_backend(
    tmp_path: Path, kind: str, phase: str, stage: str
) -> None:
    """Inspect launch-time state without importing JAX or touching a GPU.

    Only the test copy's fixed working directory and interpreter are changed.
    The production shell's environment handling and argument forwarding run
    unchanged, including the array index of an independent audit stream.
    """
    stub = tmp_path / "python_stub"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if len(sys.argv) > 2 and sys.argv[1] == '-':\n"
        "    print('stub_case ternary_joint'); sys.exit(0)\n"
        "print(json.dumps({\n"
        "    'argv': sys.argv[1:],\n"
        "    'platforms': os.environ.get('JAX_PLATFORMS'),\n"
        "    'platform_name': os.environ.get('JAX_PLATFORM_NAME'),\n"
        "    'array_index': os.environ.get('SLURM_ARRAY_TASK_ID'),\n"
        "    'sharding': os.environ.get('NETKET_SHARDING'),\n"
        "    'cwd': os.getcwd(),\n"
        "    'pythonpath': os.environ.get('PYTHONPATH'),\n"
        "}))\n"
    )
    stub.chmod(0o755)
    actual_runner = RUNNER if kind != "couplings" else ROOT / "projects/clebsch_tree_nqs/configs/remote/triangle_couplings_20261017/run.sbatch"
    source = actual_runner.read_text()
    workspace = "quality" if kind != "couplings" else "couplings"
    fixed_cd = f"cd /home/yx789/nqs-triangle-{workspace}-20261017"
    fixed_python = "NQS_PY=/home/yx789/nqs-vit-quality/.conda-netket/bin/python"
    assert source.count(fixed_cd) == source.count(fixed_python) == 1
    source = source.replace(fixed_cd, f"cd {shlex.quote(str(tmp_path))}")
    source = source.replace(fixed_python, f"NQS_PY={shlex.quote(str(stub))}")
    runner = tmp_path / "run.sbatch"
    runner.write_text(source)
    env = dict(os.environ)
    env.update(
        JAX_PLATFORMS="cpu",
        JAX_PLATFORM_NAME="cpu",
        NETKET_SHARDING="unexpected-submitter-value",
        NQS_KIND=kind,
        NQS_PHASE=phase,
        NQS_STAGE=stage,
        NQS_CASE="stub_case",
        NQS_PLAN_SHA256="0" * 64,
        NQS_SESSION_SECONDS="120",
        NQS_AUDIT_STEP="500",
        SLURM_ARRAY_TASK_ID="1",
        SLURM_CPUS_PER_TASK="4",
    )
    completed = subprocess.run(
        ["bash", str(runner)],
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    calls = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(calls) == (2 if phase == "train" or kind == "couplings" else 1)
    for call in calls:
        assert call["cwd"] == call["pythonpath"] == str(tmp_path)
        assert call["array_index"] == "1"
        if phase == "aggregate":
            assert call["platforms"] == "cpu"
            assert call["platform_name"] is None
        else:
            assert call["platforms"] == "cuda"
            assert call["platform_name"] == "gpu"
        expected_sharding = "0" if kind in ("clebsch", "backend", "couplings") or phase in ("audit", "aggregate") else "1"
        assert call["sharding"] == expected_sharding

    args = calls[-1]["argv"]
    if kind == "couplings":
        args = next(call["argv"] for call in calls if "-m" in call["argv"])
        assert args[:3] == ["-u", "-m", "projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train"]
        assert args[args.index("--arm") + 1] == "ternary_joint"
        if phase == "pilot":
            assert args[args.index("--until-step") + 1] == "2"
        else:
            assert args[args.index("--require-pilot-step") + 1] == "2"
    elif kind == "backend":
        assert args == ["-"]
    elif kind == "clebsch":
        assert args[:3] == [
            "-u", "-m", "projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_schedule"
        ]
        assert args[args.index("--stage") + 1] == stage
    elif phase in ("audit", "aggregate"):
        assert args[:4] == [
            "-u", "-m", "projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality", "audit-energy"
        ]
        if phase == "audit":
            assert args[args.index("--stream-index") + 1] == "1"
            assert "--aggregate-only" not in args
        else:
            assert "--aggregate-only" in args
    else:
        assert args[:3] == [
            "-u", "-m", "projects.clebsch_tree_nqs.workflows.spin_spatial_vit_netket"
        ]
        if phase == "replay":
            assert "--resume" in args
            assert args[args.index("--until-step") + 1] == "4"
        elif phase == "pilot":
            assert "--resume" not in args
            assert args[args.index("--until-step") + 1] == "2"
        else:
            assert "--resume" in args
            assert calls[0]["argv"] == ["-"]
