"""Atomic pickle-free checkpoints with provenance and resumable PRNG/Adam."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def source_hashes() -> dict[str, str]:
    """Hash project Python sources to identify the actual numerical code."""
    directory = Path(__file__).parent
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.py"))}


def write_json(path: Path, data: dict) -> None:
    """Atomically write strict JSON; NaN/Inf cannot masquerade as results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path: Path, params: dict, state: dict, key, metadata: dict) -> None:
    """Store params, moments, optimizer step, key and complete config metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"params__{name}": np.asarray(value) for name, value in params.items()}
    for moment in ("m", "v"):
        arrays.update({f"{moment}__{name}": np.asarray(value) for name, value in state[moment].items()})
    arrays.update(step=np.asarray(state["step"]), key=np.asarray(key),
                  work=np.asarray(json.dumps(state.get("work", {}))),
                  metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False)))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> tuple[dict, dict, object, dict]:
    """Read a checkpoint with allow_pickle=False; caller validates its config."""
    with np.load(path, allow_pickle=False) as archive:
        trees = {prefix: {name.split("__", 1)[1]: jnp.asarray(archive[name])
                          for name in archive.files if name.startswith(prefix + "__")}
                 for prefix in ("params", "m", "v")}
        state = {"m": trees["m"], "v": trees["v"], "step": int(archive["step"]),
                 "work": json.loads(str(archive["work"])) if "work" in archive.files else {}}
        return trees["params"], state, jnp.asarray(archive["key"]), json.loads(str(archive["metadata"]))
