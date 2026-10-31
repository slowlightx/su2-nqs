"""Explicit parameter warm starts, separate from strict optimizer resume."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import jax
import numpy as np

from .artifacts import load_checkpoint, source_hashes
from .models import GRUConfig, init_model
from .training import init_optimizer


def initialize_run(seed: int, model: GRUConfig,
                   checkpoint: Path | None = None) -> tuple[dict, dict, jax.Array, dict | None]:
    """Start a new run, optionally reusing model parameters with fresh moments.

    A warm start may change training code, tasks or optimizer, but must preserve
    the model, Hilbert-label conventions and Hamiltonian coupling semantics.
    Multiple seeds from one parent vary stochastic trajectories, not the parent
    initialization; this limitation is recorded explicitly.
    """
    key, init_key = jax.random.split(jax.random.PRNGKey(seed))
    template = init_model(init_key, model)
    if checkpoint is None:
        return template, init_optimizer(template), key, None
    params, parent_state, _, metadata = load_checkpoint(checkpoint)
    parent_config = metadata.get("signature", metadata.get("config", {}))
    if parent_config.get("model") != asdict(model):
        raise ValueError("warm-start model configuration differs")
    current = source_hashes()
    for name in ("models.py", "basis.py", "hamiltonians.py"):
        if metadata.get("source_sha256", {}).get(name) != current[name]:
            raise ValueError(f"warm-start parameter semantics changed: {name}")
    if params.keys() != template.keys():
        raise ValueError("warm-start parameter keys differ")
    for name, value in params.items():
        if value.shape != template[name].shape or value.dtype != template[name].dtype:
            raise ValueError(f"warm-start shape/dtype mismatch: {name}")
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"nonfinite warm-start parameter: {name}")
    origin = {"checkpoint": str(checkpoint),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "parent_source_sha256": metadata["source_sha256"],
              "parent_step": parent_state["step"], "parent_work": parent_state.get("work", {}),
              "parent_seed": metadata.get("seed"),
              "parent_origin": metadata.get("warm_start") or metadata.get("origin"),
              "optimizer_and_prng_restarted": True,
              "replica_semantics": "same pretrained parameters; seeds vary new stochastic trajectories"}
    return params, init_optimizer(params), key, origin
