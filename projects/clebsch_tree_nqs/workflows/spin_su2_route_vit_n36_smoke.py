"""Local N=36 runtime and singlet-energy smoke for SU(2) Route-ViT.

The state starts from a translation-projected nearest-neighbour dimer product.
No ED and no optimizer update is performed.  The smoke establishes that the
exact-SU(2), translation-projected Route-ViT can be sampled locally and that a
center-only singlet VMC force is finite before a remote optimization campaign.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.frontier.spin_dimer_product_init import (
    init_spin_clebsch_tree_dimer_product,
)
from su2_nqs.lattices.pair_balanced_tree import pair_balanced_fusion_tree
from su2_nqs.lattices.route_vit_features import multiscale_bond_route_features
from su2_nqs.lattices.spatial_routes import rectangular_support_route_orders
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_translation_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector,
    spatial_projected_amplitude,
)
from su2_nqs.models.spin_clebsch_tree import (
    SpinClebschTreeConfig,
    init_spin_clebsch_tree,
)
from su2_nqs.models.su2_route_vit import (
    SU2RouteViTConfig,
    init_su2_route_vit,
    su2_route_vit_amplitudes,
    su2_route_vit_parameter_count,
)
from su2_nqs.models.support_aware_spatial_cg import SupportAwareSpatialCGConfig
from su2_nqs.vmc.physical_exchange import run_exchange_sweeps
from su2_nqs.vmc.singlet_diagonal_force import (
    sampled_singlet_diagonal_value_and_grad,
    three_hz_local_energies,
)


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "exact_su2_route_vit_n36_local_smoke"


def _validate(spec: dict[str, object]) -> None:
    if (
        spec.get("schema_version") != 1
        or spec.get("domain") != "spin"
        or spec.get("task") != TASK
    ):
        raise ValueError("invalid SU2 Route-ViT N36 smoke identity")
    case = spec.get("case")
    model = spec.get("model")
    sampler = spec.get("sampler")
    if not isinstance(case, dict) or case != {
        "case_id": "triangular_6x6_j2_0p125_su2_route_vit_smoke",
        "geometry": "triangular",
        "rows": 6,
        "columns": 6,
        "j2_over_j1": 0.125,
        "reference_route": "row_dimers",
    }:
        raise ValueError("the smoke requires the registered triangular 6x6 case")
    if not isinstance(model, dict):
        raise TypeError("model must be an object")
    for field in (
        "feature_dim",
        "maximum_intermediate_q",
        "embedding_dim",
        "head_count",
        "depth",
        "mlp_ratio",
        "modulation_rank",
        "translation_orbit_chunk_size",
    ):
        if isinstance(model.get(field), bool) or int(model.get(field, 0)) < 1:
            raise ValueError(f"model.{field} must be a positive integer")
    if model["embedding_dim"] % model["head_count"]:
        raise ValueError("embedding_dim must be divisible by head_count")
    for field in ("initialization_scale", "modulation_scale"):
        if float(model.get(field, 0.0)) <= 0.0:
            raise ValueError(f"model.{field} must be positive")
    floor = float(model.get("support_floor_head_scale", 0.0))
    if not 0.0 <= floor <= 0.5:
        raise ValueError("model.support_floor_head_scale must lie in [0,0.5]")
    backbone_floor = float(model.get("backbone_support_floor", 0.0))
    if not 0.0 <= backbone_floor <= 0.5:
        raise ValueError("model.backbone_support_floor must lie in [0,0.5]")
    if not isinstance(sampler, dict):
        raise TypeError("sampler must be an object")
    for field in ("chains", "burn_sweeps", "snapshots", "sweeps_between"):
        if isinstance(sampler.get(field), bool) or int(sampler.get(field, 0)) < 1:
            raise ValueError(f"sampler.{field} must be positive")
    probability = float(sampler.get("reference_dimer_proposal_probability", -1.0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("reference_dimer_proposal_probability must lie in [0,1]")


def _configuration(
    spec: dict[str, object],
) -> tuple[SU2RouteViTConfig, dict[str, object], tuple[tuple[int, int, float], ...]]:
    case = spec["case"]
    model = spec["model"]
    rows, columns = int(case["rows"]), int(case["columns"])
    n_sites = rows * columns
    routes = rectangular_support_route_orders(
        "triangular", rows, columns, drop_duplicates=True
    )
    tree = pair_balanced_fusion_tree(n_sites)
    j1 = triangular_j1j2_bonds(
        rows, columns, j1=1.0, j2=0.0, periodic=True
    )
    j2 = triangular_j1j2_bonds(
        rows, columns, j1=0.0, j2=1.0, periodic=True
    )
    features = multiscale_bond_route_features(
        tree, routes, {"J1": j1, "J2": j2}
    )
    reference = tuple(routes).index(str(case["reference_route"]))
    spatial = SupportAwareSpatialCGConfig(
        backbone=SpinClebschTreeConfig(
            tree,
            target_q=0,
            target_m2=0,
            feature_dim=int(model["feature_dim"]),
            maximum_intermediate_q=int(model["maximum_intermediate_q"]),
            initialization_scale=float(model["initialization_scale"]),
        ),
        route_names=tuple(routes),
        route_orders=tuple(routes.values()),
        route_features=features.standardized,
        feature_names=features.labels,
        reference_route=reference,
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
        rows,
        columns,
        j1=1.0,
        j2=float(case["j2_over_j1"]),
        periodic=True,
    )
    ledger = {
        "route_names": list(spatial.route_names),
        "route_count": spatial.n_routes,
        "reference_route": spatial.route_names[spatial.reference_route],
        "feature_names": list(features.labels),
        "unique_token_count": len(set(features.standardized)),
        "tree": "pair_balanced_over_adjacent_dimers",
    }
    return config, ledger, tuple(bonds)


def _dimer_chains(
    key: jax.Array,
    *,
    chains: int,
    order: tuple[int, ...],
) -> jax.Array:
    orientations = jax.random.bernoulli(
        key, shape=(chains, len(order) // 2)
    ).astype(jnp.int32)
    tree_states = jnp.stack((orientations, 1 - orientations), axis=-1).reshape(
        chains, len(order)
    )
    physical = jnp.zeros_like(tree_states)
    return physical.at[:, jnp.asarray(order, dtype=jnp.int32)].set(tree_states)


def _seconds(callable_, *args):
    started = time.perf_counter()
    value = callable_(*args)
    jax.block_until_ready(value)
    return value, time.perf_counter() - started


def _tree_l2_norm(tree: object) -> float:
    return float(
        np.sqrt(
            sum(
                np.sum(np.asarray(leaf, dtype=np.float64) ** 2)
                for leaf in jax.tree_util.tree_leaves(tree)
            )
        )
    )


def run_spin_su2_route_vit_n36_smoke(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run the N36 projected dimer-state sampling and force smoke."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    config, ledger, bonds = _configuration(spec)
    params = init_su2_route_vit(jax.random.PRNGKey(int(spec["seed"])), config)
    dimer_backbone = init_spin_clebsch_tree_dimer_product(
        config.spatial.backbone
    )
    backbone_floor = float(spec["model"]["backbone_support_floor"])
    if backbone_floor > 0.0:
        random_backbone = init_spin_clebsch_tree(
            jax.random.PRNGKey(int(spec["seed"]) + 100),
            config.spatial.backbone,
        )
        dimer_backbone = jax.tree_util.tree_map(
            lambda dimer, random: dimer + backbone_floor * random,
            dimer_backbone,
            random_backbone,
        )
    params = {
        **params,
        "backbone": dimer_backbone,
    }
    floor_scale = float(spec["model"]["support_floor_head_scale"])
    if floor_scale > 0.0:
        head = params["transformer"]["modulation_head"]
        floor = floor_scale * jax.random.normal(
            jax.random.PRNGKey(int(spec["seed"]) + 101),
            head.shape,
            dtype=head.dtype,
        ) / np.sqrt(max(head.shape[0], 1))
        params = {
            **params,
            "transformer": {
                **params["transformer"],
                "modulation_head": floor,
            },
        }
    projector = PhysicalSpatialProjector(
        triangular_hnf_translation_group(((6, 0), (0, 6))),
        orbit_chunk_size=int(spec["model"]["translation_orbit_chunk_size"]),
        rematerialize_orbit=True,
    )

    def raw_amplitude(candidate, state, _config):
        return su2_route_vit_amplitudes(
            candidate, state[None, :], config
        )[0]

    def amplitude(candidate, state):
        return spatial_projected_amplitude(
            candidate,
            state,
            config,
            projector,
            amplitude_fn=raw_amplitude,
        )

    sampler = spec["sampler"]
    order = config.spatial.route_orders[config.spatial.reference_route]
    chains = _dimer_chains(
        jax.random.PRNGKey(int(spec["seed"]) + 1),
        chains=int(sampler["chains"]),
        order=order,
    )
    compiled_forward = jax.jit(
        lambda candidate, states: jax.vmap(
            lambda state: amplitude(candidate, state)
        )(states)
    )
    initial_values, forward_cold = _seconds(compiled_forward, params, chains)
    _replay, forward_warm = _seconds(compiled_forward, params, chains)
    if np.any(~np.isfinite(np.asarray(initial_values))) or np.any(
        np.abs(np.asarray(initial_values)) == 0.0
    ):
        raise RuntimeError("projected dimer initialization has invalid chain amplitudes")

    compiled_sweeps = jax.jit(
        lambda candidate, key, states, sweeps: run_exchange_sweeps(
            candidate,
            key,
            states,
            amplitude_fn=amplitude,
            particle_kind="spin",
            sweeps=sweeps,
            exchange_edges=tuple(
                tuple(sorted((order[site], order[site + 1])))
                for site in range(0, len(order), 2)
            ),
            bond_proposal_probability=float(
                sampler["reference_dimer_proposal_probability"]
            ),
            lazy_probability=0.05,
        ),
        static_argnums=(3,),
    )
    sampling_key = jax.random.PRNGKey(int(spec["seed"]) + 2)
    sampling_started = time.perf_counter()
    chains, amplitudes, sampling_key, burn_diagnostics = compiled_sweeps(
        params,
        sampling_key,
        chains,
        int(sampler["burn_sweeps"]),
    )
    jax.block_until_ready((chains, amplitudes))
    snapshots = []
    diagnostics = []
    for _ in range(int(sampler["snapshots"])):
        chains, amplitudes, sampling_key, row = compiled_sweeps(
            params,
            sampling_key,
            chains,
            int(sampler["sweeps_between"]),
        )
        jax.block_until_ready((chains, amplitudes))
        snapshots.append(np.asarray(chains, dtype=np.int32))
        diagnostics.append(
            {name: float(np.asarray(value)) for name, value in row.items()}
        )
    sampling_seconds = time.perf_counter() - sampling_started
    samples = np.concatenate(snapshots, axis=0)
    local = np.asarray(three_hz_local_energies(samples, bonds), dtype=np.float64)
    local_by_snapshot_chain = local.reshape(
        int(sampler["snapshots"]), int(sampler["chains"])
    )
    chain_means = np.mean(local_by_snapshot_chain, axis=0)

    def center_fn(candidate, batch):
        return compiled_forward(candidate, jnp.asarray(batch, dtype=jnp.int32))

    gradient_started = time.perf_counter()
    sample_batches = tuple(
        jnp.asarray(samples[start : start + 8])
        for start in range(0, len(samples), 8)
    )
    gradient = sampled_singlet_diagonal_value_and_grad(
        params,
        sample_batches,
        center_fn=center_fn,
        bonds=bonds,
    )
    gradient_seconds = time.perf_counter() - gradient_started
    if not np.isfinite(gradient.energy) or gradient.gradient_norm <= 0.0:
        raise RuntimeError("N36 sampled singlet force is invalid")

    energy = float(np.mean(local))
    standard_error = float(np.std(local, ddof=1) / np.sqrt(local.size))
    chain_standard_error = float(
        np.std(chain_means, ddof=1) / np.sqrt(chain_means.size)
    )
    unique_sample_fraction = float(
        len({tuple(row) for row in samples}) / len(samples)
    )
    manifest = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    now = datetime.now(timezone.utc).isoformat()
    payload: dict[str, object] = {
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
        "training_performed": False,
        "model": {
            **ledger,
            "parameter_count": su2_route_vit_parameter_count(params),
            "translation_group_order": projector.group_order,
            "exact_su2_by_construction": True,
            "initial_state": (
                "translation_projected_row_dimer_product"
                if floor_scale == 0.0
                else "translation_projected_dimer_plus_exact_su2_modulation_floor"
            ),
            "support_floor_head_scale": floor_scale,
            "backbone_support_floor": backbone_floor,
        },
        "runtime": {
            "projected_forward_batch_size": int(sampler["chains"]),
            "projected_forward_cold_seconds": forward_cold,
            "projected_forward_warm_seconds": forward_warm,
            "sampling_seconds": sampling_seconds,
            "sampled_force_seconds": gradient_seconds,
        },
        "sampler": {
            "sample_count": int(samples.shape[0]),
            "unique_sample_fraction": unique_sample_fraction,
            "burn": {
                name: float(np.asarray(value))
                for name, value in burn_diagnostics.items()
            },
            "measurement_diagnostics": diagnostics,
        },
        "singlet_diagonal_energy": {
            "total": energy,
            "per_site": energy / config.n_sites,
            "naive_standard_error": standard_error,
            "standard_error_from_chain_means": chain_standard_error,
            "chain_means": chain_means.tolist(),
            "warning": (
                "The naive error ignores Markov autocorrelation and Var(3Hz) "
                "is not the physical Hamiltonian variance."
            ),
        },
        "sampled_force": {
            "energy": gradient.energy,
            "sample_count": gradient.sample_count,
            "gradient_norm": gradient.gradient_norm,
            "independent_tree_l2_norm": _tree_l2_norm(gradient.gradient),
            "minimum_center_magnitude": gradient.minimum_center_magnitude,
            "maximum_local_energy_magnitude": gradient.maximum_local_energy_magnitude,
        },
        "claim_boundary": (
            "Local triangular-6x6 projected-forward, MCMC, and finite-force "
            "readiness from a dimer product only. No optimized energy, physical "
            "variance, phase, QSL, scaling, efficiency, or PRL win is claimed."
        ),
    }
    payload["runtime_identity_sha256"] = canonical_sha256(
        payload["runtime_identity"]
    )
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_su2_route_vit_n36_smoke"]
