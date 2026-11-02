"""Atomic checkpoints for JAX parameter pytrees and optimizer state."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


def save_checkpoint(
    path: str | Path,
    params: object,
    *,
    optimizer_state: object | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Atomically save a checkpoint.  Only load files from trusted sources."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": params,
        "optimizer_state": optimizer_state,
        "metadata": {} if metadata is None else metadata,
    }
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name, dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def load_checkpoint(path: str | Path) -> dict[str, object]:
    """Load a trusted checkpoint written by :func:`save_checkpoint`."""

    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "params" not in payload:
        raise ValueError("invalid su2-nqs checkpoint")
    return payload

