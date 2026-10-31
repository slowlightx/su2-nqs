"""Local N=36 sampler, full-H, line-search, and resume audit for Route-ViT."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import numerical_runtime_identity, transitive_local_source_manifest
from su2_nqs.frontier.spin_dimer_product_init import init_spin_clebsch_tree_dimer_product
from su2_nqs.lattices.pair_balanced_tree import pair_balanced_fusion_tree
from su2_nqs.lattices.physical_route_banks import triangular_dimer_complete_route_orders
from su2_nqs.lattices.route_vit_features import multiscale_bond_route_features
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_translation_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector, spatial_projected_amplitude
from su2_nqs.models.spin_clebsch_tree import SpinClebschTreeConfig, init_spin_clebsch_tree
from su2_nqs.models.su2_route_vit import SU2RouteViTConfig, init_su2_route_vit, su2_route_vit_amplitudes, su2_route_vit_parameter_count
from su2_nqs.models.support_aware_spatial_cg import SupportAwareSpatialCGConfig
from su2_nqs.vmc.physical_exchange import run_exchange_sweeps
from su2_nqs.vmc.resumable_checkpoint import load_latest_resumable_checkpoint, save_resumable_checkpoint_generation
from su2_nqs.vmc.singlet_diagonal_force import sampled_singlet_diagonal_value_and_grad, three_hz_local_energies
from su2_nqs.vmc.spin_physical import padded_spin_connectivity, spin_physical_hpsi

from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_n36_smoke import _dimer_chains


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "exact_su2_route_vit_n36_local_audit"


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (1, "spin", TASK):
        raise ValueError("invalid N36 audit identity")
    case, model, sampler, audit = (spec.get(name) for name in ("case", "model", "sampler", "audit"))
    if not all(isinstance(value, dict) for value in (case, model, sampler, audit)):
        raise TypeError("case, model, sampler, and audit must be objects")
    if (case.get("geometry"), case.get("rows"), case.get("columns")) != ("triangular", 6, 6):
        raise ValueError("the local audit is triangular 6x6")
    if not isinstance(sampler.get("arms"), list) or not sampler["arms"]:
        raise ValueError("sampler.arms must be nonempty")
    for arm in sampler["arms"]:
        if arm.get("edge_bank") not in {"row", "dimer_complete"}:
            raise ValueError("sampler edge_bank must be row or dimer_complete")
        if not 0.0 <= float(arm.get("proposal_probability", -1.0)) <= 1.0:
            raise ValueError("proposal_probability must lie in [0,1]")


def _configuration(spec: dict[str, object]):
    model = spec["model"]
    routes = triangular_dimer_complete_route_orders(6, 6)
    tree = pair_balanced_fusion_tree(36)
    j1 = triangular_j1j2_bonds(6, 6, j1=1.0, j2=0.0, periodic=True)
    j2 = triangular_j1j2_bonds(6, 6, j1=0.0, j2=1.0, periodic=True)
    features = multiscale_bond_route_features(tree, routes, {"J1": j1, "J2": j2})
    names = tuple(routes)
    spatial = SupportAwareSpatialCGConfig(
        backbone=SpinClebschTreeConfig(
            tree,
            target_q=0,
            target_m2=0,
            feature_dim=int(model["feature_dim"]),
            maximum_intermediate_q=int(model["maximum_intermediate_q"]),
            initialization_scale=float(model["initialization_scale"]),
        ),
        route_names=names,
        route_orders=tuple(routes.values()),
        route_features=features.standardized,
        feature_names=features.labels,
        reference_route=names.index("row_dimers"),
    )
    config = SU2RouteViTConfig(
        spatial=spatial,
        embedding_dim=int(model["embedding_dim"]),
        head_count=int(model["head_count"]),
        depth=int(model["depth"]),
        mlp_ratio=int(model["mlp_ratio"]),
        modulation_rank=int(model["modulation_rank"]),
        modulation_scale=float(model["modulation_scale"]),
        initialization_scale=float(model["initialization_scale"]),
    )
    bonds = triangular_j1j2_bonds(
        6, 6, j1=1.0, j2=float(spec["case"]["j2_over_j1"]), periodic=True
    )
    return config, tuple(bonds), {
        "route_names": list(names),
        "route_count": len(names),
        "feature_names": list(features.labels),
        "tree": "pair_balanced_over_adjacent_dimers",
        "physical_extension": "three_orientation_dimer_complete",
    }


def _initialized_runtime(spec: dict[str, object]):
    config, bonds, ledger = _configuration(spec)
    seed = int(spec["seed"])
    params = init_su2_route_vit(jax.random.PRNGKey(seed), config)
    dimer = init_spin_clebsch_tree_dimer_product(config.spatial.backbone)
    floor = float(spec["model"]["backbone_support_floor"])
    if floor:
        random = init_spin_clebsch_tree(jax.random.PRNGKey(seed + 100), config.spatial.backbone)
        dimer = jax.tree_util.tree_map(lambda left, right: left + floor * right, dimer, random)
    params = {**params, "backbone": dimer}
    head_scale = float(spec["model"]["support_floor_head_scale"])
    if head_scale:
        head = params["transformer"]["modulation_head"]
        perturbation = head_scale * jax.random.normal(
            jax.random.PRNGKey(seed + 101), head.shape, dtype=head.dtype
        ) / np.sqrt(max(head.shape[0], 1))
        params = {**params, "transformer": {**params["transformer"], "modulation_head": perturbation}}
    projector = PhysicalSpatialProjector(
        triangular_hnf_translation_group(((6, 0), (0, 6))),
        orbit_chunk_size=int(spec["model"]["translation_orbit_chunk_size"]),
        rematerialize_orbit=True,
    )

    def raw(candidate, state, _config):
        return su2_route_vit_amplitudes(candidate, state[None, :], config)[0]

    def amplitude(candidate, state):
        return spatial_projected_amplitude(candidate, state, config, projector, amplitude_fn=raw)

    forward = jax.jit(lambda candidate, states: jax.vmap(lambda state: amplitude(candidate, state))(states))
    return config, bonds, ledger, params, amplitude, forward


def _edge_bank(config: SU2RouteViTConfig, mode: str) -> tuple[tuple[int, int], ...]:
    names = config.spatial.route_names
    selected = ("row_dimers",) if mode == "row" else ("row_dimers", "column_dimers", "diagonal_dimers")
    edges: set[tuple[int, int]] = set()
    for name in selected:
        order = config.spatial.route_orders[names.index(name)]
        edges.update(tuple(sorted(order[index : index + 2])) for index in range(0, 36, 2))
    return tuple(sorted(edges))


def _autocorrelation_report(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError("autocorrelation input must be snapshots by chains")
    centered = array - np.mean(array, axis=0, keepdims=True)
    denominator = float(np.sum(centered**2))
    correlations = [1.0]
    if denominator > 0.0:
        for lag in range(1, array.shape[0] // 2 + 1):
            numerator = float(np.sum(centered[:-lag] * centered[lag:]))
            correction = array.shape[0] / (array.shape[0] - lag)
            correlations.append(correction * numerator / denominator)
    tau = 1.0
    for value in correlations[1:]:
        if value <= 0.0:
            break
        tau += 2.0 * value
    tau = max(tau, 1.0)
    total = array.size
    return {
        "autocorrelation_by_lag": [float(value) for value in correlations],
        "integrated_autocorrelation_snapshots": float(tau),
        "effective_sample_size": float(total / tau),
        "effective_sample_size_fraction": float(1.0 / tau),
    }


def _sample_arm(spec, config, params, amplitude, arm, *, key_seed: int):
    sampler = spec["sampler"]
    chains_count = int(sampler["chains"])
    reference_order = config.spatial.route_orders[config.spatial.reference_route]
    chains = _dimer_chains(jax.random.PRNGKey(key_seed), chains=chains_count, order=reference_order)
    edges = _edge_bank(config, str(arm["edge_bank"]))
    sweeps = jax.jit(
        lambda candidate, key, states, count: run_exchange_sweeps(
            candidate,
            key,
            states,
            amplitude_fn=amplitude,
            particle_kind="spin",
            sweeps=count,
            exchange_edges=edges,
            bond_proposal_probability=float(arm["proposal_probability"]),
            lazy_probability=0.05,
        ),
        static_argnums=(3,),
    )
    key = jax.random.PRNGKey(key_seed + 1)
    started = time.perf_counter()
    chains, amplitudes, key, burn = sweeps(params, key, chains, int(sampler["burn_sweeps"]))
    jax.block_until_ready((chains, amplitudes))
    snapshots, diagnostics = [], []
    for _ in range(int(sampler["snapshots"])):
        chains, amplitudes, key, row = sweeps(params, key, chains, int(sampler["sweeps_between"]))
        jax.block_until_ready((chains, amplitudes))
        snapshots.append(np.asarray(chains, dtype=np.int32))
        diagnostics.append({name: float(np.asarray(value)) for name, value in row.items()})
    samples = np.concatenate(snapshots, axis=0)
    local = np.asarray(three_hz_local_energies(samples, spec["_bonds"]), dtype=np.float64)
    shaped = local.reshape(int(sampler["snapshots"]), chains_count)
    chain_means = np.mean(shaped, axis=0)
    return samples, local, {
        "arm_id": str(arm["arm_id"]),
        "edge_bank": str(arm["edge_bank"]),
        "edge_count": len(edges),
        "proposal_probability": float(arm["proposal_probability"]),
        "sampling_seconds": time.perf_counter() - started,
        "sample_count": int(len(samples)),
        "unique_sample_fraction": float(len({tuple(row) for row in samples}) / len(samples)),
        "energy_total": float(np.mean(local)),
        "energy_per_site": float(np.mean(local) / 36.0),
        "chain_mean_standard_error_total": float(np.std(chain_means, ddof=1) / np.sqrt(chains_count)),
        "burn": {name: float(np.asarray(value)) for name, value in burn.items()},
        "mean_measurement_accepted_move_fraction": float(np.mean([row["accepted_move_fraction"] for row in diagnostics])),
        **_autocorrelation_report(shaped),
    }


def _full_h_audit(params, samples, config, bonds, amplitude, *, count: int):
    selected = np.asarray(samples[:count], dtype=np.int32)
    connected, elements, valid, _widths = padded_spin_connectivity(selected, bonds)
    full = []
    started = time.perf_counter()
    for index in range(count):
        center, action = spin_physical_hpsi(
            params,
            jnp.asarray(selected[index : index + 1]),
            jnp.asarray(connected[index : index + 1]),
            jnp.asarray(elements[index : index + 1]),
            jnp.asarray(valid[index : index + 1]),
            config,
            amplitude_fn=lambda candidate, state, _config: amplitude(candidate, state),
        )
        jax.block_until_ready((center, action))
        full.append(complex(np.asarray(action / center)[0]))
    full = np.asarray(full, dtype=np.complex128)
    diagonal = np.asarray(three_hz_local_energies(selected, bonds), dtype=np.float64)
    difference = full.real - diagonal
    return {
        "sample_count": count,
        "seconds": time.perf_counter() - started,
        "full_h_energy_total": float(np.mean(full.real)),
        "diagonal_3hz_energy_total": float(np.mean(diagonal)),
        "paired_mean_difference": float(np.mean(difference)),
        "paired_difference_standard_error": float(np.std(difference, ddof=1) / np.sqrt(count)),
        "full_h_variance_per_site": float(np.mean(np.abs(full - np.mean(full)) ** 2) / 36.0),
        "maximum_imaginary_local_energy": float(np.max(np.abs(full.imag))),
    }


def _line_search(spec, params, samples, config, bonds, forward):
    split = len(samples) // 2
    training = np.asarray(samples[:split], dtype=np.int32)
    validation = np.asarray(samples[split:], dtype=np.int32)
    batches = tuple(jnp.asarray(training[start : start + 8]) for start in range(0, len(training), 8))
    gradient = sampled_singlet_diagonal_value_and_grad(
        params,
        batches,
        center_fn=lambda candidate, batch: forward(candidate, jnp.asarray(batch, dtype=jnp.int32)),
        bonds=bonds,
    )
    reference = np.asarray(forward(params, jnp.asarray(validation)))
    diagonal = np.asarray(three_hz_local_energies(validation, bonds), dtype=np.float64)
    reference_energy = float(np.mean(diagonal))
    rows = []
    accepted = None
    accepted_params = params
    for scale in spec["audit"]["line_search_scales"]:
        scale = float(scale)
        candidate = jax.tree_util.tree_map(lambda value, direction: value - scale * direction, params, gradient.gradient)
        candidate_values = np.asarray(forward(candidate, jnp.asarray(validation)))
        weights = np.abs(candidate_values / reference) ** 2
        normalized = weights / np.sum(weights)
        energy = float(np.sum(normalized * diagonal))
        ess = float(np.sum(weights) ** 2 / np.sum(weights**2))
        shaped_weights = weights.reshape(-1, int(spec["sampler"]["chains"]))
        shaped_local = diagonal.reshape(shaped_weights.shape)
        chain_candidate = np.sum(shaped_weights * shaped_local, axis=0) / np.sum(shaped_weights, axis=0)
        chain_reference = np.mean(shaped_local, axis=0)
        differences = chain_candidate - chain_reference
        standard_error = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
        passes = bool(
            energy + 2.0 * standard_error < reference_energy
            and ess / len(validation) >= 0.5
        )
        rows.append({
            "step_scale": scale,
            "candidate_energy_total": energy,
            "reference_energy_total": reference_energy,
            "candidate_minus_reference": energy - reference_energy,
            "paired_chain_standard_error": standard_error,
            "importance_ess": ess,
            "importance_ess_fraction": ess / len(validation),
            "accepted": passes,
        })
        if passes and (accepted is None or energy < accepted[0]):
            accepted = (energy, scale)
            accepted_params = candidate
    return accepted_params, {
        "training_sample_count": len(training),
        "validation_sample_count": len(validation),
        "gradient_energy_total": gradient.energy,
        "gradient_norm": gradient.gradient_norm,
        "attempts": rows,
        "accepted_scale": None if accepted is None else accepted[1],
    }


def run_spin_su2_route_vit_n36_audit(config_path: Path, output: Path, *, overwrite: bool = False):
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    config, bonds, ledger, params, amplitude, forward = _initialized_runtime(spec)
    runtime_spec = {**spec, "_bonds": bonds}
    forward(params, _dimer_chains(jax.random.PRNGKey(int(spec["seed"]) + 500), chains=2, order=config.spatial.route_orders[config.spatial.reference_route])).block_until_ready()
    arm_rows, samples_by_arm = [], {}
    for index, arm in enumerate(spec["sampler"]["arms"]):
        samples, _local, row = _sample_arm(runtime_spec, config, params, amplitude, arm, key_seed=int(spec["seed"]) + 1000 * (index + 1))
        arm_rows.append(row)
        samples_by_arm[row["arm_id"]] = samples
        print(row["arm_id"], f"ESS={row['effective_sample_size_fraction']:.3f}", f"move={row['mean_measurement_accepted_move_fraction']:.3f}", flush=True)
    for row in arm_rows:
        row["selection_score"] = min(
            row["effective_sample_size_fraction"],
            row["mean_measurement_accepted_move_fraction"],
        )
    best = max(
        arm_rows,
        key=lambda row: (row["selection_score"], row["unique_sample_fraction"]),
    )
    best_samples = samples_by_arm[best["arm_id"]]
    full_h = _full_h_audit(params, best_samples, config, bonds, amplitude, count=int(spec["audit"]["full_h_samples"]))
    candidate, line = _line_search(spec, params, best_samples, config, bonds, forward)
    accepted = line["accepted_scale"] is not None
    run_identity = {
        "task": TASK,
        "config_sha256": canonical_sha256(spec),
        "parameter_count": su2_route_vit_parameter_count(params),
        "selected_sampler_arm": best["arm_id"],
    }
    history = [] if not accepted else [{"step": 1, "scale": line["accepted_scale"]}]
    generation = save_resumable_checkpoint_generation(
        output / "checkpoints",
        candidate,
        optimizer_state={"kind": "center_only_gradient_line_search"},
        completed_steps=int(accepted),
        next_sample_index=len(best_samples),
        run_identity=run_identity,
        history=history,
        metadata={"exact_su2": True, "translation_projected": True},
    )
    restored = load_latest_resumable_checkpoint(output / "checkpoints", expected_run_identity=run_identity)
    resume_error = max(
        float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
        for left, right in zip(jax.tree_util.tree_leaves(candidate), jax.tree_util.tree_leaves(restored["params"]), strict=True)
    )
    manifest = transitive_local_source_manifest((Path(__file__).resolve(),), root=ROOT, package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"))
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": now,
        "completed_utc": now,
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_manifest": manifest,
        "source_manifest_sha256": canonical_sha256(manifest),
        "runtime_identity": numerical_runtime_identity(),
        "hamiltonian_diagonalization_performed": False,
        "model": {**ledger, "parameter_count": su2_route_vit_parameter_count(params), "exact_su2_by_construction": True, "translation_group_order": 36},
        "sampler_arms": arm_rows,
        "selected_sampler_arm": best["arm_id"],
        "sampler_cross_arm_energy_spread_total": float(
            max(row["energy_total"] for row in arm_rows)
            - min(row["energy_total"] for row in arm_rows)
        ),
        "full_h_audit": full_h,
        "one_step_line_search": line,
        "checkpoint": {"generation": str(generation.relative_to(output)), "completed_steps": int(accepted), "resume_max_abs_error": resume_error},
        "claim_boundary": "Local N36 sampler/force/full-H/checkpoint readiness only; no converged energy, phase, scaling, or variational win is claimed.",
    }
    payload["runtime_identity_sha256"] = canonical_sha256(payload["runtime_identity"])
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_su2_route_vit_n36_audit"]
