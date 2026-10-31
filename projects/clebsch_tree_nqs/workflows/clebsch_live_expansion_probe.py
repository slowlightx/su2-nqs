"""Replay a saved square ClebschTree through a quality-preserving width lift.

No optimization, new Born chain, ED solve, or new head is performed. The full
coherent A1 amplitude and complete physical Hamiltonian are evaluated before
and after the lift. Small-system exhaustive tests separately establish its
mathematical invariants and the live-versus-zero-padding tangent contrast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.models.clebsch_tree_live_expansion import initialize_live_clebsch_expansion
from su2_nqs.models.clebsch_tree_orbit_dag import make_orbit_dag_clebsch
from su2_nqs.models.clebsch_tree_transfer import transfer_clebsch_parameters
from su2_nqs.models.literature_clebsch_tree import (
    LiteratureClebschTreeConfig, clebsch_layers, make_clebsch_features,
)
from su2_nqs.su2.fusion_tree import Leaf
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from .spin_clebsch_tree_sampled import components

jax.config.update("jax_enable_x64", True)


def reduced_feature_diagnostics(
    config: LiteratureClebschTreeConfig, params: dict,
    states: np.ndarray, order: np.ndarray,
) -> list[dict]:
    """Coordinate RMS/rank on saved rows, not Born charge probabilities.

    Condition on magnetic reachability to avoid calling kinematic zeros a
    scale defect. Ranks are limited by the saved probe rows and are gauge
    dependent; this is not a physical Schmidt-spectrum measurement.
    """
    rows = np.asarray(states)
    values = jax.jit(make_clebsch_features(config))(params, jnp.asarray(rows[:, order]))

    def sites(node):
        return (node.index,) if isinstance(node, Leaf) else sites(node.left) + sites(node.right)

    report = []
    for index, (layer, value) in enumerate(zip(clebsch_layers(config), values, strict=True)):
        value = np.asarray(value)
        magnetic = np.stack([np.sum(2 * rows[:, np.asarray(order)[np.asarray(sites(node))]] - 1,
                                   axis=1) for node in layer.nodes], axis=1)
        channels = []
        for q, width in enumerate(layer.widths):
            if not width:
                continue
            valid = (abs(magnetic) <= q) & ((magnetic - q) % 2 == 0)
            matrix = value[:, :, q, :width][valid]
            if not len(matrix):
                channels.append(dict(q=q, width=width, reachable_probe_rows=0))
                continue
            singular = np.linalg.svd(matrix, compute_uv=False)
            rms = np.sqrt(np.mean(matrix ** 2, axis=0))
            active = rms[rms > 0]
            channels.append(dict(q=q, width=width, reachable_probe_rows=len(matrix),
                RMS=float(np.sqrt(np.mean(matrix ** 2))),
                feature_RMS_min=float(np.min(rms)), feature_RMS_max=float(np.max(rms)),
                feature_RMS_ratio=float(np.max(active) / np.min(active)) if len(active) else None,
                coordinate_rank_relative_1e_minus_8=int(np.count_nonzero(
                    singular > singular[0] * 1e-8)) if len(singular) and singular[0] else 0))
        report.append(dict(layer=index, channels=channels))
    return report


def probe(parent: Path, output: Path, *, target_cap: int = 20, seed: int = 2026101791) -> dict:
    """Evaluate saved native configurations; errors use unfloored relative norms."""
    record = json.loads((parent / "registration.json").read_text())
    if record["sha256"] != canonical_sha256({k: v for k, v in record.items() if k != "sha256"}):
        raise ValueError("saved registration content changed")
    saved = load_checkpoint(parent / "checkpoint.pkl")
    params, metadata = saved["params"], saved["metadata"]
    if (metadata["registration_sha256"] != record["sha256"]
            or metadata["params_sha256"] != pytree_sha256(params)):
        raise ValueError("parent parameter/registration identity mismatch")
    config, _, _, bonds, images, order = components(record["config"])
    wider = LiteratureClebschTreeConfig(config.tree, target_q=config.target_q,
        target_m2=config.target_m2, maximum_q=config.maximum_q,
        multiplicity_cap=target_cap, high_spin_cap=config.high_spin_cap,
        high_spin_threshold_q=config.high_spin_threshold_q)
    live, report = initialize_live_clebsch_expansion(config, params, wider, jax.random.PRNGKey(seed))
    old, old_plan = make_orbit_dag_clebsch(config, order, images)
    new, new_plan = make_orbit_dag_clebsch(wider, order, images)
    states = jnp.asarray(metadata["sampler_states"][:4])
    times = {}

    def measured(name, function, *args):
        evaluate = jax.jit(function)
        started = perf_counter()
        result = jax.block_until_ready(evaluate(*args))
        compile_and_first = perf_counter() - started
        started = perf_counter()
        result = jax.block_until_ready(evaluate(*args))
        times[name] = dict(compile_and_first_seconds=compile_and_first,
                           warm_seconds=perf_counter() - started)
        return result

    def relative(a, b):
        a, b = np.asarray(a), np.asarray(b)
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
        if not scale:
            return 0.
        return float(np.linalg.norm((a - b) / scale) / np.linalg.norm(a / scale))

    a = measured("old_full_A1_amplitude_4_saved_rows", old, params, states)
    b = measured("wide_full_A1_amplitude_4_saved_rows", new, live, states)
    errors = dict(full_A1_amplitude=relative(a, b))
    errors["full_A1_covariance"] = relative(b, new(live, states[:, images[13]]))
    direction = jax.tree.map(lambda x: jnp.ones_like(x) / np.sqrt(report["source_parameters"]), params)
    zero = jax.tree.map(jnp.zeros_like, live)
    inherited, _ = transfer_clebsch_parameters(config, direction, wider, zero, mode="embedding")
    old_jvp = measured("old_inherited_JVP", lambda p, d: jax.jvp(
        lambda z: old(z, states), (p,), (d,))[1], params, direction)
    new_jvp = measured("wide_inherited_JVP", lambda p, d: jax.jvp(
        lambda z: new(z, states), (p,), (d,))[1], live, inherited)
    errors["inherited_parameter_JVP"] = relative(old_jvp, new_jvp)
    open_direction = dict(merges=zero["merges"],
                          readout=jnp.zeros_like(live["readout"]).at[-1].set(1.))
    new_tangent = measured("new_readout_JVP", lambda p, d: jax.jvp(
        lambda z: new(z, states), (p,), (d,))[1], live, open_direction)
    tangent_ratio = float(jnp.linalg.norm(new_tangent) / jnp.linalg.norm(b))
    orthogonal = new_tangent - b * jnp.vdot(b, new_tangent) / jnp.vdot(b, b)
    orthogonal_ratio = float(jnp.linalg.norm(orthogonal) / jnp.linalg.norm(b))
    score = new_tangent / b
    sampled_projective_speed = float(jnp.sqrt(jnp.mean(jnp.abs(score - jnp.mean(score)) ** 2)))
    first = make_amplitude_heisenberg_action(lambda p, x: old(p, x[None])[0], bonds,
                                             n_sites=config.tree.n_leaves)
    second = make_amplitude_heisenberg_action(lambda p, x: new(p, x[None])[0], bonds,
                                              n_sites=config.tree.n_leaves)
    original_action = measured("old_complete_H_one_saved_row", first, params, states[:1])
    lifted_action = measured("wide_complete_H_one_saved_row", second, live, states[:1])
    errors["complete_H_product"] = relative(original_action[1], lifted_action[1])
    if (max(errors.values()) > 2e-11 or not np.isfinite(tangent_ratio)
            or tangent_ratio <= 0 or not np.isfinite(orthogonal_ratio) or orthogonal_ratio <= 1e-12):
        raise ArithmeticError("state-preserving live width lift failed")
    output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output / "initialized_only.pkl", live, metadata=dict(
        parent_checkpoint_sha256=file_sha256(parent / "checkpoint.pkl"),
        source_step=metadata["completed_steps"], trained_updates=0,
        optimizer_requires_new_registration=True, sampler_states=np.asarray(states)))
    result = dict(status="passed", parent=str(parent), source_step=metadata["completed_steps"],
        checkpoint_sha256=file_sha256(parent / "checkpoint.pkl"),
        parent_parameters_sha256=pytree_sha256(params),
        initialized_checkpoint_sha256=file_sha256(output / "initialized_only.pkl"),
        initializer=report, maximum_relative_errors=errors,
        new_readout_tangent_norm_relative_to_parent=tangent_ratio,
        new_readout_tangent_orthogonal_norm_relative_to_parent_on_saved_rows=orthogonal_ratio,
        new_readout_sampled_Born_projective_speed_on_saved_rows=sampled_projective_speed,
        direction_is_not_only_global_scale_on_saved_rows=True,
        saved_row_projective_probe_is_not_exhaustive_N64_FS_metric=True,
        independence_from_entire_old_tangent_not_established=True,
        times=times, old_projection=old_plan, new_projection=new_plan,
        parent_coordinate_features=reduced_feature_diagnostics(
            config, params, metadata["sampler_states"], np.asarray(order)),
        feature_diagnostics_are_not_sector_weights_or_Schmidt_values=True,
        backend=jax.default_backend(), new_training_updates=0, new_Born_samples=0,
        energy_improvement_claim=False, larger_GPU_memory_not_measured=True,
        original_optimizer_and_checkpoints_unchanged=True)
    atomic_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-cap", default=20, type=int)
    args = parser.parse_args()
    print(json.dumps(probe(args.parent, args.output, target_cap=args.target_cap), indent=2))
