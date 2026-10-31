"""Reusable exact-support Route-ViT training that retains final states.

Historical scan workflows intentionally keep their sealed JSON immutable and
discard trained parameters.  This module mirrors that registered protocol but
returns the final raw wavefunction and parameter pytree, allowing multiple
physics analyses to share one deterministic training run.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.support_aware_spatial_cg import spatial_cg_route_spec
from su2_nqs.models.spin_clebsch_tree import (
    init_spin_clebsch_tree,
    spin_clebsch_tree_amplitudes,
    spin_clebsch_tree_parameter_count,
)
from su2_nqs.models.su2_route_vit import (
    init_su2_route_vit,
    su2_route_vit_amplitudes,
    su2_route_vit_backbone_parameter_count,
    su2_route_vit_controls,
    su2_route_vit_parameter_count,
)
from su2_nqs.models.su2_route_vit_causal_controls import (
    su2_route_local_gate_amplitudes,
    su2_route_local_gate_controls,
)

from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit import (
    _loss_parts,
    _multiscale_spatial_config,
    _train,
)
from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_causal_scan import (
    _route_config,
    _route_diagnostics,
    _scrambled_spatial_config,
)


jax.config.update("jax_enable_x64", True)


def _trained_route_arm(
    *,
    arm_id: str,
    warmed: object,
    warm_raw: np.ndarray,
    basis: jax.Array,
    rayleigh: Callable,
    config: object,
    seed: int,
    steps: int,
    learning_rate: float,
    record_every: int,
    amplitudes_fn: Callable,
    controls_fn: Callable,
    route_mixer: str,
    token_alignment: str,
) -> dict[str, object]:
    initialized = init_su2_route_vit(jax.random.PRNGKey(seed), config)
    initialized = {**initialized, "backbone": warmed}

    def raw(candidate):
        return amplitudes_fn(candidate, basis, config)

    nesting_error = float(
        np.max(np.abs(np.asarray(raw(initialized)) - np.asarray(warm_raw)))
    )
    trained, trace = _train(
        initialized,
        raw,
        rayleigh,
        steps=steps,
        learning_rate=learning_rate,
        record_every=record_every,
    )
    return {
        "arm_id": arm_id,
        "architecture": "exact_su2_route_conditioned_clebsch_tree",
        "params": trained,
        "raw_state": np.asarray(raw(trained), dtype=np.complex128),
        "training": trace,
        "parameter_count": su2_route_vit_parameter_count(trained),
        "backbone_parameter_count": su2_route_vit_backbone_parameter_count(trained),
        "nested_warm_state_max_abs_error": nesting_error,
        "route_diagnostics": _route_diagnostics(
            trained,
            config,
            controls_fn,
            route_mixer=route_mixer,
            token_alignment=token_alignment,
        ),
    }


def train_exact_route_vit_trial(
    *,
    model: dict[str, object],
    optimization: dict[str, object],
    data: dict[str, object],
    seed: int,
    projection: str,
) -> dict[str, object]:
    """Train the six preregistered causal-scan arms and retain final states.

    The warm D=4 state is shared by the continued-tree and all Route-ViT arms.
    ``projection`` enters every loss evaluation from the first warmup update;
    it is not a post-training readout.
    """

    if projection not in data["tables"]:
        raise ValueError(f"projection {projection!r} is unavailable")
    geometry = str(data["geometry"])
    rows, columns = int(data["rows"]), int(data["columns"])
    if (rows, columns) != (4, 4):
        raise ValueError("exact-state Route-ViT retention is registered on 4x4")
    basis = jnp.asarray(data["basis"], dtype=jnp.int32)
    rayleigh = _loss_parts(data, projection)

    base_spatial, ledger = spatial_cg_route_spec(
        geometry,
        rows,
        columns,
        feature_dim=int(model["base_feature_dim"]),
        maximum_intermediate_q=int(model["maximum_intermediate_q"]),
        initialization_scale=float(model["initialization_scale"]),
    )
    base_spatial, token_ledger = _multiscale_spatial_config(base_spatial, geometry)
    scrambled_spatial, scramble_ledger = _scrambled_spatial_config(base_spatial)
    reference_order = jnp.asarray(
        base_spatial.route_orders[base_spatial.reference_route], dtype=jnp.int32
    )

    def base_raw(candidate):
        return spin_clebsch_tree_amplitudes(
            candidate, basis[:, reference_order], base_spatial.backbone
        )

    initial = init_spin_clebsch_tree(
        jax.random.PRNGKey(int(seed)), base_spatial.backbone
    )
    warmed, warm_trace = _train(
        initial,
        base_raw,
        rayleigh,
        steps=int(optimization["warmup_steps"]),
        learning_rate=float(optimization["learning_rate"]),
        record_every=int(optimization["record_every"]),
    )
    warm_raw = np.asarray(base_raw(warmed), dtype=np.complex128)
    continued, continued_trace = _train(
        warmed,
        base_raw,
        rayleigh,
        steps=int(optimization["comparison_steps"]),
        learning_rate=float(optimization["learning_rate"]),
        record_every=int(optimization["record_every"]),
    )
    arms: list[dict[str, object]] = [
        {
            "arm_id": "single_tree_d4",
            "architecture": "single_clebsch_tree",
            "params": continued,
            "raw_state": np.asarray(base_raw(continued), dtype=np.complex128),
            "training": continued_trace,
            "warm_training": warm_trace,
            "parameter_count": spin_clebsch_tree_parameter_count(continued),
        }
    ]

    rank = int(model["modulation_rank"])
    common = {
        "warmed": warmed,
        "warm_raw": warm_raw,
        "basis": basis,
        "rayleigh": rayleigh,
        "steps": int(optimization["comparison_steps"]),
        "learning_rate": float(optimization["route_learning_rate"]),
        "record_every": int(optimization["record_every"]),
        "seed": int(seed) + 1000 + rank,
    }
    coefficient_common = dict(common)
    coefficient_common["seed"] = int(seed) + 1000
    arms.append(
        _trained_route_arm(
            arm_id="route_vit_coefficients_d4",
            config=_route_config(base_spatial, model, 0),
            amplitudes_fn=su2_route_vit_amplitudes,
            controls_fn=su2_route_vit_controls,
            route_mixer="global_attention",
            token_alignment="physical",
            **coefficient_common,
        )
    )
    for arm_id, spatial, amplitudes_fn, controls_fn, mixer, alignment in (
        (
            "route_vit_conditioned_d4",
            base_spatial,
            su2_route_vit_amplitudes,
            su2_route_vit_controls,
            "global_attention",
            "physical",
        ),
        (
            "route_vit_scrambled_tokens_d4",
            scrambled_spatial,
            su2_route_vit_amplitudes,
            su2_route_vit_controls,
            "global_attention",
            "scrambled_multiset_preserving",
        ),
        (
            "route_local_conditioned_d4",
            base_spatial,
            su2_route_local_gate_amplitudes,
            su2_route_local_gate_controls,
            "route_local_gate",
            "physical",
        ),
    ):
        arms.append(
            _trained_route_arm(
                arm_id=arm_id,
                config=_route_config(spatial, model, rank),
                amplitudes_fn=amplitudes_fn,
                controls_fn=controls_fn,
                route_mixer=mixer,
                token_alignment=alignment,
                **common,
            )
        )

    matched_spatial, matched_ledger = spatial_cg_route_spec(
        geometry,
        rows,
        columns,
        feature_dim=int(model["matched_feature_dim"]),
        maximum_intermediate_q=int(model["maximum_intermediate_q"]),
        initialization_scale=float(model["initialization_scale"]),
    )
    matched_order = jnp.asarray(
        matched_spatial.route_orders[matched_spatial.reference_route], dtype=jnp.int32
    )

    def matched_raw(candidate):
        return spin_clebsch_tree_amplitudes(
            candidate, basis[:, matched_order], matched_spatial.backbone
        )

    matched = init_spin_clebsch_tree(
        jax.random.PRNGKey(int(seed)), matched_spatial.backbone
    )
    matched, matched_trace = _train(
        matched,
        matched_raw,
        rayleigh,
        steps=int(optimization["warmup_steps"])
        + int(optimization["comparison_steps"]),
        learning_rate=float(optimization["learning_rate"]),
        record_every=int(optimization["record_every"]),
    )
    arms.append(
        {
            "arm_id": "single_tree_matched_width",
            "architecture": "single_clebsch_tree",
            "params": matched,
            "raw_state": np.asarray(matched_raw(matched), dtype=np.complex128),
            "training": matched_trace,
            "parameter_count": spin_clebsch_tree_parameter_count(matched),
        }
    )
    return {
        "seed": int(seed),
        "projection": projection,
        "route_ledger": {
            **ledger,
            "route_vit_tokens": token_ledger,
            "scrambled_token_control": scramble_ledger,
            "matched_width_route_ledger": matched_ledger,
        },
        "arms": arms,
    }


__all__ = ["train_exact_route_vit_trial"]
