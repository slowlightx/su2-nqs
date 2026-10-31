"""N=36 common-sample matched-work carrier and ViT comparison.

This workflow consumes the already validated four-basin Route-ViT checkpoint;
it does not repeat ED or a long burn-in.  All six trainable arms see exactly
the same configurations and the same number of full-H actions.  Fixed-Sz
controls are Hamiltonian-blind tangent spaces attached residually to the same
SU(2) base, after which only a six-scalar adapter is optimized.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_n36_audit import (
    _initialized_runtime,
)
from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_n36_basin_gate import (
    _nematic_diagonal_series,
)
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256, pytree_sha256
from su2_nqs.lattices.physical_route_banks import triangular_dimer_complete_route_orders
from su2_nqs.lattices.route_carrier_templates import triangular_route_carrier_templates
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_translation_group
from su2_nqs.models.fixed_sz_generic_vit import (
    FixedSzGenericViTConfig,
    init_fixed_sz_generic_vit,
)
from su2_nqs.models.fixed_sz_generic_vit_adapter import (
    FixedSzGenericViTAdapterConfig,
    fixed_sz_generic_vit_adapter_amplitude,
    init_fixed_sz_generic_vit_adapter,
    seeded_fixed_sz_generic_vit_adapter_directions,
)
from su2_nqs.models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig,
    init_fixed_sz_spatial_vit,
)
from su2_nqs.models.fixed_sz_vit_adapter import (
    FixedSzViTAdapterConfig,
    fixed_sz_vit_adapter_amplitude,
    init_fixed_sz_vit_adapter,
    seeded_fixed_sz_vit_adapter_directions,
)
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector,
    spatial_projected_amplitude,
)
from su2_nqs.models.su2_route_carrier_vit import (
    SU2RouteCarrierViTConfig,
    init_su2_route_carrier_head,
    su2_route_carrier_vit_amplitudes,
)
from su2_nqs.models.su2_route_vit import su2_route_vit_amplitudes
from su2_nqs.vmc.common_sample_vmc import common_sample_vmc_value_and_grad
from su2_nqs.vmc.resumable_checkpoint import (
    load_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)
from su2_nqs.vmc.spin_physical import padded_spin_connectivity, spin_physical_hpsi
from su2_nqs.vmc.spin_symmetry_mixture import d6_complete_dimer_matchings


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_n36_carrier_vit_matched_work"
ARM_IDS = (
    "su2_a1_3",
    "su2_a1_e2_6",
    "su2_generic_3",
    "su2_generic_6",
    "fixed_sz_spatial_vit_6",
    "symmetry_unaware_vit_6",
)


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N36 carrier/ViT matched-work identity")
    for name in ("reference", "common_sample", "vit_control", "training", "gates"):
        if not isinstance(spec.get(name), dict):
            raise TypeError(f"{name} must be an object")
    training = spec["training"]
    if int(training.get("steps", 0)) < 1:
        raise ValueError("training.steps must be positive")
    common = spec["common_sample"]
    for name in ("calibration_count", "training_count", "audit_count"):
        if int(common.get(name, 0)) < 1:
            raise ValueError(f"common_sample.{name} must be positive")
    bridge = (
        float(common.get("reference_dimer_probability", 0.0)),
        float(common.get("exploratory_dimer_probability", 0.0)),
    )
    if min(bridge) <= 0.0 or sum(bridge) >= 1.0:
        raise ValueError("common_sample bridge probabilities are invalid")


def _matched_residual_control(
    *,
    zero_params,
    vit_amplitude_fn,
    base_amplitude_fn,
    calibration_states: np.ndarray,
    proposal_density: np.ndarray,
):
    """Whiten a six-dimensional ViT tangent on a common SU(2) base state."""

    count = int(np.asarray(zero_params["coefficients"]).shape[0])
    states = jnp.asarray(calibration_states, dtype=jnp.int32)

    def batch(candidate):
        return jax.vmap(lambda row: vit_amplitude_fn(candidate, row))(states)

    columns = []
    for index in range(count):
        tangent = {
            "coefficients": jnp.eye(count, dtype=jnp.float64)[index]
        }
        _, column = jax.jvp(batch, (zero_params,), (tangent,))
        columns.append(np.asarray(column, dtype=np.complex128))
    jacobian = np.column_stack(columns)
    base_values = np.asarray(
        jax.vmap(lambda row: base_amplitude_fn(row))(states),
        dtype=np.complex128,
    )
    zero_vit_values = np.asarray(batch(zero_params), dtype=np.complex128)
    base_norm = float(
        np.sqrt(np.sum(np.abs(base_values) ** 2 / proposal_density))
    )
    minimum_zero_vit_magnitude = float(np.min(np.abs(zero_vit_values)))
    if (
        minimum_zero_vit_magnitude <= 0.0
        or np.any(~np.isfinite(zero_vit_values))
    ):
        raise FloatingPointError("ViT correlator zero state has invalid centers")
    # Tangents of psi0 * psi_vit(c) / psi_vit(0).  This per-configuration
    # nesting is stable even when the two unnormalised models use vastly
    # different amplitude scales.
    correlator_jacobian = (
        base_values[:, None] * jacobian / zero_vit_values[:, None]
    )
    weighted = correlator_jacobian / np.sqrt(proposal_density)[:, None]
    gram = np.real(weighted.conj().T @ weighted)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = max(float(np.max(eigenvalues)) * 1.0e-10, 1.0e-20)
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    if rank != count:
        raise RuntimeError("ViT residual tangent bank is rank deficient")
    transform = (
        eigenvectors
        @ np.diag(base_norm / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )
    transform_jax = jnp.asarray(transform, dtype=jnp.float64)

    def amplitude(candidate, state):
        inner = {
            "coefficients": transform_jax
            @ jnp.asarray(candidate["coefficients"], dtype=jnp.float64)
        }
        return base_amplitude_fn(state) * (
            vit_amplitude_fn(inner, state)
            / vit_amplitude_fn(zero_params, state)
        )

    zero_values = np.asarray(
        jax.vmap(lambda row: amplitude(zero_params, row))(states)
    )
    return zero_params, amplitude, {
        "tangent_rank": rank,
        "tangent_count": count,
        "tangent_gram_condition_number": float(
            np.max(eigenvalues) / np.min(eigenvalues)
        ),
        "tangent_gram_eigenvalues": eigenvalues.tolist(),
        "zero_head_max_abs_common_base_error": float(
            np.max(np.abs(zero_values - base_values))
        ),
        "minimum_zero_vit_calibration_magnitude": minimum_zero_vit_magnitude,
        "hamiltonian_or_energy_calls": 0,
        "metric": "Re[J^dagger diag(1/q) J] whitened to common-base norm",
    }


def _load_reference(spec: dict[str, object]):
    config_path = ROOT / str(spec["reference"]["basin_config"])
    checkpoint_directory = ROOT / str(spec["reference"]["checkpoint_directory"])
    basin_spec = json.loads(config_path.read_text())
    config, bonds, ledger, _runtime_params, _amplitude, _forward = _initialized_runtime(
        basin_spec
    )
    manifest = json.loads((checkpoint_directory / "LATEST.json").read_text())
    generations = manifest["generations"]
    if len(generations) < 2:
        raise ValueError("reference basin gate must contain burn and final generations")
    burn = load_resumable_checkpoint(checkpoint_directory / generations[0]["file"])
    final = load_resumable_checkpoint(checkpoint_directory / generations[-1]["file"])
    if pytree_sha256(burn["params"]) != pytree_sha256(final["params"]):
        raise ValueError("reference checkpoint changed parameters during sampling")
    for payload in (burn, final):
        metadata = payload["metadata"]
        if "chain_states" not in metadata or "cached_amplitudes" not in metadata:
            raise ValueError("reference checkpoint lacks common-sample state")
    return basin_spec, config, tuple(bonds), ledger, burn, final


def _known_mixture_samples(
    *,
    seed: int,
    count: int,
    reference_dimer_probability: float,
    exploratory_dimer_probability: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Sample a known-density reference/exploratory/uniform bridge."""

    reference_probability = float(reference_dimer_probability)
    exploratory_probability = float(exploratory_dimer_probability)
    uniform_probability = 1.0 - reference_probability - exploratory_probability
    if (
        count < 1
        or reference_probability <= 0.0
        or exploratory_probability <= 0.0
        or uniform_probability <= 0.0
    ):
        raise ValueError("known bridge requires three positive probabilities")
    matchings = np.asarray(d6_complete_dimer_matchings(6, 6), dtype=np.int32)
    row_order = triangular_dimer_complete_route_orders(6, 6)["row_dimers"]
    row_matching = {
        tuple(sorted(row_order[index : index + 2]))
        for index in range(0, 36, 2)
    }
    reference_orbit = set()
    for image in triangular_hnf_translation_group(((6, 0), (0, 6))):
        reference_orbit.add(
            tuple(
                sorted(
                    tuple(sorted((image[first], image[second])))
                    for first, second in row_matching
                )
            )
        )
    matching_keys = [
        tuple(sorted(tuple(sorted(map(int, pair))) for pair in matching))
        for matching in matchings
    ]
    reference_indices = tuple(
        index for index, key in enumerate(matching_keys) if key in reference_orbit
    )
    exploratory_indices = tuple(
        index for index in range(len(matchings)) if index not in reference_indices
    )
    if not reference_indices or not exploratory_indices:
        raise RuntimeError("could not separate reference and exploratory dimer families")
    matching_weights = np.zeros((len(matchings),), dtype=np.float64)
    matching_weights[list(reference_indices)] = (
        reference_probability / len(reference_indices)
    )
    matching_weights[list(exploratory_indices)] = (
        exploratory_probability / len(exploratory_indices)
    )
    rng = np.random.default_rng(int(seed))
    rows = []
    origins = []
    for _ in range(int(count)):
        if rng.random() < reference_probability + exploratory_probability:
            matching_index = int(
                rng.choice(
                    len(matchings), p=matching_weights / np.sum(matching_weights)
                )
            )
            matching = matchings[matching_index]
            state = np.zeros((36,), dtype=np.int32)
            orientation = rng.integers(0, 2, size=len(matching))
            for bit, (first, second) in zip(orientation, matching, strict=True):
                state[int(first)] = int(bit)
                state[int(second)] = 1 - int(bit)
            origins.append(
                "reference_dimer"
                if matching_index in reference_indices
                else "exploratory_dimer"
            )
        else:
            state = np.zeros((36,), dtype=np.int32)
            state[rng.choice(36, size=18, replace=False)] = 1
            origins.append("uniform_sz0")
        rows.append(state)
    states = np.stack(rows, axis=0)
    compatible = np.asarray(
        [
            sum(
                bool(np.all(row[matching[:, 0]] + row[matching[:, 1]] == 1))
                for matching in matchings
            )
            for row in states
        ],
        dtype=np.int32,
    )
    compatibility_matrix = np.asarray(
        [
            [
                bool(np.all(row[matching[:, 0]] + row[matching[:, 1]] == 1))
                for matching in matchings
            ]
            for row in states
        ],
        dtype=np.float64,
    )
    uniform_density = 1.0 / math.comb(36, 18)
    proposal = compatibility_matrix @ matching_weights / 2.0**18
    proposal += uniform_probability * uniform_density
    if np.any(proposal <= 0.0) or np.any(~np.isfinite(proposal)):
        raise RuntimeError("known common-sample proposal density is invalid")
    return states, proposal, {
        "reference_dimer_origin_count": origins.count("reference_dimer"),
        "exploratory_dimer_origin_count": origins.count("exploratory_dimer"),
        "uniform_origin_count": origins.count("uniform_sz0"),
        "reference_matching_count": len(reference_indices),
        "exploratory_matching_count": len(exploratory_indices),
        "compatible_d6_matching_count_minimum": int(np.min(compatible)),
        "compatible_d6_matching_count_maximum": int(np.max(compatible)),
    }


def _batch_hpsi(params, states, connected, elements, valid, config, amplitude_fn):
    centers, actions = [], []
    for index in range(len(states)):
        center, action = spin_physical_hpsi(
            params,
            jnp.asarray(states[index : index + 1]),
            jnp.asarray(connected[index : index + 1]),
            jnp.asarray(elements[index : index + 1]),
            jnp.asarray(valid[index : index + 1]),
            config,
            amplitude_fn=lambda candidate, row, _config: amplitude_fn(candidate, row),
        )
        centers.append(center)
        actions.append(action)
    return jnp.concatenate(centers), jnp.concatenate(actions)


def _common_energy(
    params,
    states,
    reference_centers,
    connected,
    elements,
    valid,
    config,
    amplitude_fn,
) -> tuple[dict[str, float], np.ndarray]:
    centers, actions = _batch_hpsi(
        params, states, connected, elements, valid, config, amplitude_fn
    )
    centers_host = np.asarray(centers, dtype=np.complex128)
    local = np.asarray(actions / centers, dtype=np.complex128)
    weights = np.abs(centers_host / np.asarray(reference_centers)) ** 2
    weights /= np.sum(weights)
    mean = complex(np.sum(weights * local))
    ess = float(1.0 / np.sum(weights**2))
    standard_error = float(
        np.sqrt(np.sum(weights * np.abs(local - mean) ** 2).real / ess)
    )
    q1, q2 = _nematic_diagonal_series(states)
    return {
        "energy_total": float(mean.real),
        "energy_per_site": float(mean.real / states.shape[1]),
        "energy_standard_error_total": standard_error,
        "variance_per_site": float(
            np.sum(weights * np.abs(local - mean) ** 2).real / states.shape[1]
        ),
        "importance_ess": ess,
        "importance_ess_fraction": ess / len(states),
        "maximum_normalized_weight": float(np.max(weights)),
        "maximum_imaginary_local_energy": float(np.max(np.abs(local.imag))),
        "diagonal_nematic_q_squared": float(np.sum(weights * (q1 * q1 + q2 * q2))),
    }, centers_host


def _normalized_gradient_step(params, gradient, learning_rate: float):
    norm = float(
        np.sqrt(
            sum(
                float(np.sum(np.asarray(leaf, dtype=np.float64) ** 2))
                for leaf in jax.tree_util.tree_leaves(gradient)
            )
        )
    )
    scale = 0.0 if norm <= 1.0e-12 else float(learning_rate) / norm
    return jax.tree_util.tree_map(
        lambda value, direction: value - scale * direction, params, gradient
    ), norm, scale


def _train_arm(
    *,
    arm_id: str,
    params,
    amplitude_fn,
    config,
    train_states,
    train_reference,
    train_connectivity,
    audit_states,
    audit_reference,
    audit_connectivity,
    steps: int,
    learning_rate: float,
    output: Path,
    run_identity: dict[str, object],
) -> dict[str, object]:
    train_connected, train_elements, train_valid = train_connectivity
    audit_connected, audit_elements, audit_valid = audit_connectivity
    started = time.perf_counter()
    initial, _ = _common_energy(
        params,
        audit_states,
        audit_reference,
        audit_connected,
        audit_elements,
        audit_valid,
        config,
        amplitude_fn,
    )
    history = []
    for step in range(1, int(steps) + 1):
        gradient = common_sample_vmc_value_and_grad(
            params,
            jnp.asarray(train_states),
            jnp.asarray(train_reference),
            hpsi_fn=lambda candidate, states: _batch_hpsi(
                candidate,
                states,
                train_connected,
                train_elements,
                train_valid,
                config,
                amplitude_fn,
            ),
            center_fn=lambda candidate, states: jax.vmap(
                lambda row: amplitude_fn(candidate, row)
            )(states),
        )
        params, raw_gradient_norm, applied_scale = _normalized_gradient_step(
            params, gradient.gradient, learning_rate
        )
        row = {
            "step": step,
            "training_energy_total": gradient.energy,
            "training_variance": gradient.variance,
            "training_importance_ess_fraction": gradient.ess_fraction,
            "gradient_norm": raw_gradient_norm,
            "applied_gradient_scale": applied_scale,
        }
        history.append(row)
        save_resumable_checkpoint_generation(
            output / "checkpoints" / arm_id,
            params,
            optimizer_state={"kind": "normalized_gradient_descent"},
            completed_steps=step,
            next_sample_index=step * len(train_states),
            run_identity={**run_identity, "arm_id": arm_id},
            history=history,
            metadata={
                "shared_training_sample_count": len(train_states),
                "shared_audit_sample_count": len(audit_states),
            },
        )
    final, _ = _common_energy(
        params,
        audit_states,
        audit_reference,
        audit_connected,
        audit_elements,
        audit_valid,
        config,
        amplitude_fn,
    )
    return {
        "arm_id": arm_id,
        "trainable_parameter_count": int(
            sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
        ),
        "initial_audit": initial,
        "training_history": history,
        "final_audit": final,
        "wall_seconds": time.perf_counter() - started,
        "work_ledger": {
            "training_steps": int(steps),
            "shared_samples_per_step": len(train_states),
            "full_h_configuration_calls_training": int(steps) * len(train_states),
            "full_h_configuration_calls_audit": 2 * len(audit_states),
            "full_h_configuration_calls_total": int(steps) * len(train_states)
            + 2 * len(audit_states),
            "center_only_configuration_calls_training": int(steps)
            * len(train_states),
        },
    }


def run_spin_n36_carrier_vit_matched_work(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")
    basin_spec, base_config, bonds, route_ledger, burn, final = _load_reference(spec)
    frozen_su2 = final["params"]
    common = spec["common_sample"]
    calibration_states, calibration_density, calibration_ledger = _known_mixture_samples(
        seed=int(spec["seed"]) + 101,
        count=int(common["calibration_count"]),
        reference_dimer_probability=float(common["reference_dimer_probability"]),
        exploratory_dimer_probability=float(common["exploratory_dimer_probability"]),
    )
    train_states, train_density, train_ledger = _known_mixture_samples(
        seed=int(spec["seed"]) + 102,
        count=int(common["training_count"]),
        reference_dimer_probability=float(common["reference_dimer_probability"]),
        exploratory_dimer_probability=float(common["exploratory_dimer_probability"]),
    )
    audit_states, audit_density, audit_ledger = _known_mixture_samples(
        seed=int(spec["seed"]) + 103,
        count=int(common["audit_count"]),
        reference_dimer_probability=float(common["reference_dimer_probability"]),
        exploratory_dimer_probability=float(common["exploratory_dimer_probability"]),
    )
    training = spec["training"]
    train_reference = np.sqrt(train_density).astype(np.complex128)
    audit_reference = np.sqrt(audit_density).astype(np.complex128)
    train_raw = padded_spin_connectivity(train_states, bonds)
    audit_raw = padded_spin_connectivity(audit_states, bonds)
    train_connectivity = tuple(jnp.asarray(value) for value in train_raw[:3])
    audit_connectivity = tuple(jnp.asarray(value) for value in audit_raw[:3])

    projector = PhysicalSpatialProjector(
        triangular_hnf_translation_group(((6, 0), (0, 6))),
        orbit_chunk_size=int(basin_spec["model"]["translation_orbit_chunk_size"]),
        rematerialize_orbit=True,
    )
    routes = triangular_dimer_complete_route_orders(6, 6)
    templates = triangular_route_carrier_templates(
        base_config.spatial.backbone.tree,
        routes,
        ((6, 0), (0, 6)),
        generic_seed=int(spec["seed"]) + 10,
    )
    physical = (*templates.a1, *templates.e2)

    def base_raw(candidate, state, _config):
        return su2_route_vit_amplitudes(
            candidate, state[None, :], base_config
        )[0]

    def base_amplitude(candidate, state):
        return spatial_projected_amplitude(
            candidate,
            state,
            base_config,
            projector,
            amplitude_fn=base_raw,
        )

    carrier_specs = {
        "su2_a1_3": (templates.a1, templates.physical_labels[:3]),
        "su2_a1_e2_6": (physical, templates.physical_labels),
        "su2_generic_3": (templates.generic[:3], tuple(f"generic-{i}" for i in range(3))),
        "su2_generic_6": (templates.generic, tuple(f"generic-{i}" for i in range(6))),
    }

    arm_runtime = {}
    for arm_id, (directions, names) in carrier_specs.items():
        carrier_config = SU2RouteCarrierViTConfig(
            base=base_config,
            templates=tuple(directions),
            labels=tuple(names),
            carrier_scale=float(training["head_scale"]),
        )
        params = init_su2_route_carrier_head(carrier_config)

        def raw(candidate, state, _config, *, local_config=carrier_config):
            return su2_route_carrier_vit_amplitudes(
                candidate, frozen_su2, state[None, :], local_config
            )[0]

        def amplitude(candidate, state, *, local_config=carrier_config, raw_fn=raw):
            return spatial_projected_amplitude(
                candidate,
                state,
                local_config,
                projector,
                amplitude_fn=raw_fn,
            )

        arm_runtime[arm_id] = (params, amplitude, carrier_config)

    vit_control = spec["vit_control"]
    spatial_base = FixedSzSpatialViTConfig(
        geometry="triangular",
        rows=6,
        columns=6,
        patch_rows=2,
        patch_columns=2,
        embedding_dim=int(vit_control["embedding_dim"]),
        head_count=int(vit_control["head_count"]),
        depth=int(vit_control["depth"]),
        mlp_ratio=int(vit_control["mlp_ratio"]),
        target_m2=0,
        initialization_scale=float(vit_control["initialization_scale"]),
        use_distance_envelope=True,
    )
    initial_spatial = init_fixed_sz_spatial_vit(
        jax.random.PRNGKey(int(spec["seed"]) + 20), spatial_base
    )
    initial_spatial = {
        **initial_spatial,
        "log_cosh_head": {
            **initial_spatial["log_cosh_head"],
            "weight_imag": jnp.zeros_like(
                initial_spatial["log_cosh_head"]["weight_imag"]
            ),
            "bias_imag": jnp.zeros_like(
                initial_spatial["log_cosh_head"]["bias_imag"]
            ),
        },
    }

    generic_base = FixedSzGenericViTConfig(
        n_sites=36,
        embedding_dim=int(vit_control["embedding_dim"]),
        head_count=int(vit_control["head_count"]),
        depth=int(vit_control["depth"]),
        mlp_ratio=int(vit_control["mlp_ratio"]),
        target_m2=0,
        initialization_scale=float(vit_control["initialization_scale"]),
    )
    initial_generic = init_fixed_sz_generic_vit(
        jax.random.PRNGKey(int(spec["seed"]) + 30), generic_base
    )
    initial_generic = {
        **initial_generic,
        "log_cosh_head": {
            **initial_generic["log_cosh_head"],
            "weight_imag": jnp.zeros_like(
                initial_generic["log_cosh_head"]["weight_imag"]
            ),
            "bias_imag": jnp.zeros_like(
                initial_generic["log_cosh_head"]["bias_imag"]
            ),
        },
    }
    adapter_directions = seeded_fixed_sz_vit_adapter_directions(
        int(vit_control["embedding_dim"]),
        6,
        seed=int(spec["seed"]) + 40,
        real_only=True,
    )
    adapter_labels = tuple(f"adapter-{index}" for index in range(6))
    spatial_adapter_config = FixedSzViTAdapterConfig(
        base=spatial_base,
        directions=adapter_directions,
        labels=adapter_labels,
        adapter_scale=float(training["head_scale"]),
    )
    spatial_adapter = init_fixed_sz_vit_adapter(spatial_adapter_config)

    def spatial_adapter_raw(candidate, state, _config):
        return fixed_sz_vit_adapter_amplitude(
            candidate, initial_spatial, state, spatial_adapter_config
        )

    def spatial_adapter_amplitude(candidate, state):
        return spatial_projected_amplitude(
            candidate,
            state,
            spatial_adapter_config,
            projector,
            amplitude_fn=spatial_adapter_raw,
        )

    spatial_adapter, spatial_residual_amplitude, spatial_registration = (
        _matched_residual_control(
            zero_params=spatial_adapter,
            vit_amplitude_fn=spatial_adapter_amplitude,
            base_amplitude_fn=lambda state: base_amplitude(frozen_su2, state),
            calibration_states=calibration_states,
            proposal_density=calibration_density,
        )
    )
    arm_runtime["fixed_sz_spatial_vit_6"] = (
        spatial_adapter,
        spatial_residual_amplitude,
        spatial_adapter_config,
    )
    generic_adapter_config = FixedSzGenericViTAdapterConfig(
        base=generic_base,
        directions=seeded_fixed_sz_generic_vit_adapter_directions(
            generic_base, 6, seed=int(spec["seed"]) + 41
        ),
        labels=adapter_labels,
        adapter_scale=float(training["head_scale"]),
    )
    generic_adapter = init_fixed_sz_generic_vit_adapter(generic_adapter_config)
    generic_adapter, generic_residual_amplitude, generic_registration = (
        _matched_residual_control(
            zero_params=generic_adapter,
            vit_amplitude_fn=lambda candidate, state: fixed_sz_generic_vit_adapter_amplitude(
                candidate, initial_generic, state, generic_adapter_config
            ),
            base_amplitude_fn=lambda state: base_amplitude(frozen_su2, state),
            calibration_states=calibration_states,
            proposal_density=calibration_density,
        )
    )
    arm_runtime["symmetry_unaware_vit_6"] = (
        generic_adapter,
        generic_residual_amplitude,
        generic_adapter_config,
    )

    run_identity = {
        "task": TASK,
        "workflow_revision": "real_ground_state_common_base_correlator_v2",
        "config_sha256": canonical_sha256(spec),
        "reference_params_sha256": pytree_sha256(frozen_su2),
        "shared_train_states_sha256": canonical_sha256(train_states.tolist()),
        "shared_audit_states_sha256": canonical_sha256(audit_states.tolist()),
    }
    arms = []
    for arm_id in ARM_IDS:
        params, amplitude_fn, model_config = arm_runtime[arm_id]
        arm_path = output / "arm_results" / f"{arm_id}.json"
        cached = None
        if arm_path.is_file():
            candidate = json.loads(arm_path.read_text())
            if candidate.get("run_identity") == run_identity:
                cached = candidate.get("row")
        if cached is None:
            row = _train_arm(
                arm_id=arm_id,
                params=params,
                amplitude_fn=amplitude_fn,
                config=model_config,
                train_states=train_states,
                train_reference=train_reference,
                train_connectivity=train_connectivity,
                audit_states=audit_states,
                audit_reference=audit_reference,
                audit_connectivity=audit_connectivity,
                steps=int(training["steps"]),
                learning_rate=float(training["learning_rate"]),
                output=output,
                run_identity=run_identity,
            )
            atomic_json(
                arm_path, {"run_identity": run_identity, "row": row}
            )
        else:
            row = cached
        arms.append(row)
        print(
            arm_id,
            f"E/N={row['final_audit']['energy_per_site']:.8f}",
            f"ESS={row['final_audit']['importance_ess_fraction']:.3f}",
            flush=True,
        )
    by_id = {row["arm_id"]: row for row in arms}
    comparisons = {
        "a1_minus_generic3_energy_per_site": by_id["su2_a1_3"]["final_audit"]["energy_per_site"]
        - by_id["su2_generic_3"]["final_audit"]["energy_per_site"],
        "a1_e2_minus_generic6_energy_per_site": by_id["su2_a1_e2_6"]["final_audit"]["energy_per_site"]
        - by_id["su2_generic_6"]["final_audit"]["energy_per_site"],
        "e2_increment_energy_per_site": by_id["su2_a1_e2_6"]["final_audit"]["energy_per_site"]
        - by_id["su2_a1_3"]["final_audit"]["energy_per_site"],
        "a1_e2_minus_fixed_sz_spatial_vit_energy_per_site": by_id["su2_a1_e2_6"]["final_audit"]["energy_per_site"]
        - by_id["fixed_sz_spatial_vit_6"]["final_audit"]["energy_per_site"],
        "a1_e2_minus_symmetry_unaware_vit_energy_per_site": by_id["su2_a1_e2_6"]["final_audit"]["energy_per_site"]
        - by_id["symmetry_unaware_vit_6"]["final_audit"]["energy_per_site"],
    }
    gates = spec["gates"]
    minimum_ess = min(
        row["final_audit"]["importance_ess_fraction"] for row in arms
    )
    maximum_control_condition = max(
        spatial_registration["tangent_gram_condition_number"],
        generic_registration["tangent_gram_condition_number"],
    )
    maximum_imaginary_local_energy = max(
        row["final_audit"]["maximum_imaginary_local_energy"] for row in arms
    )
    result = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "reference": {
            "basin_task": basin_spec["task"],
            "case": basin_spec["case"],
            "params_sha256": pytree_sha256(frozen_su2),
            "route_ledger": route_ledger,
            "no_ed_repeated": True,
        },
        "common_samples": {
            "calibration_count": len(calibration_states),
            "training_count": len(train_states),
            "audit_count": len(audit_states),
            "proposal": "known-density D6-dimer plus uniform-Sz0 independent mixture",
            "reference_dimer_probability": float(
                common["reference_dimer_probability"]
            ),
            "exploratory_dimer_probability": float(
                common["exploratory_dimer_probability"]
            ),
            "calibration_ledger": calibration_ledger,
            "training_ledger": train_ledger,
            "audit_ledger": audit_ledger,
            "exact_same_training_configurations_every_arm": True,
            "exact_same_audit_configurations_every_arm": True,
        },
        "carrier_registration": {
            "physical_labels": list(templates.physical_labels),
            "a1_rank": 3,
            "e2_response_rank": 3,
            "joint_physical_rank": 6,
            "matched_generic_rank": 6,
            "generic_orthogonal_to_physical_span": True,
            "e2_channel": "E2 x E2 -> A1 scalar response; no explicit nematic orientation selected",
        },
        "vit_residual_controls": {
            "fixed_sz_spatial_vit": spatial_registration,
            "symmetry_unaware_vit": generic_registration,
            "purpose": "Hamiltonian-blind tangent whitening on an exactly common SU2 base",
            "real_ground_state_correlators": True,
        },
        "arms": arms,
        "comparisons": comparisons,
        "gates": {
            "minimum_final_importance_ess_fraction": minimum_ess,
            "maximum_control_tangent_condition_number": maximum_control_condition,
            "maximum_final_imaginary_local_energy": maximum_imaginary_local_energy,
            "ess_gate_passed": minimum_ess
            >= float(gates["importance_ess_fraction_minimum"]),
            "control_tangent_gate_passed": maximum_control_condition
            <= float(gates["control_tangent_condition_maximum"]),
            "real_local_energy_gate_passed": maximum_imaginary_local_energy
            <= 1.0e-10,
        },
        "interpretation_boundary": {
            "full_gamma_a1_projected": False,
            "translation_projected_su2_and_spatial_vit": True,
            "symmetry_unaware_control_still_uses_common_sz0_domain": True,
            "pilot_not_publication_energy_claim": True,
        },
    }
    result["gates"]["all_comparison_gates_passed"] = bool(
        result["gates"]["ess_gate_passed"]
        and result["gates"]["control_tangent_gate_passed"]
        and result["gates"]["real_local_energy_gate_passed"]
    )
    atomic_json(result_path, result)
    return result


__all__ = ["ARM_IDS", "TASK", "run_spin_n36_carrier_vit_matched_work"]
