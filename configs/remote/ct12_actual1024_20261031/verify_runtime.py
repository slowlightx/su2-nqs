"""CPU-only package fingerprint check before any Slurm GPU Python process."""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path


def verify() -> dict[str,str]:
    """Compare all eight pinned runtime packages; never install or mutate them."""
    root=Path(__file__).resolve().parents[3]
    path=root/'configs/remote/ct12_w28_gpu_preflight_20261030/runtime_packages.json'
    expected=json.loads(path.read_text())
    actual={name:importlib.metadata.version(name) for name in expected}
    if actual!=expected:raise RuntimeError('shared Python package fingerprint changed')
    return actual


if __name__=='__main__':print(json.dumps(dict(passed=True,runtime_packages=verify()),sort_keys=True))
