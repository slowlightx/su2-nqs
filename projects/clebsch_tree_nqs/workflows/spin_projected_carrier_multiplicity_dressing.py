"""Exact-small gate for a projected carrier plus multiplicity-space dressing.

The workflow never rebuilds the many-body Hamiltonian.  It consumes a sealed
14-dimensional ``S=0, Gamma-A1`` representation of triangular 4x4 and uses
that complete basis as an exact acceleration of the physical Reynolds
projector during optimization.  A direct physical-projector replay on held
configurations checks the acceleration before any result is accepted.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from su2_nqs.benchmarks.four_cell_schur import four_cell_schur_basis_matrix
from su2_nqs.benchmarks.many_cell_replay import TRIANGULAR_N16_SUPPORTS
from su2_nqs.benchmarks.support_aware_spatial_cg import spatial_cg_route_spec
from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.lattices.multiplicity_spatial_features import (
    multiplicity_spatial_features,
)
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.projected_carrier_multiplicity_dressing import (
    ProjectedCarrierMultiplicityDressingConfig,
    init_projected_carrier_multiplicity_dressing,
    projected_carrier_multiplicity_dressing_amplitude,
)
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.models.su2_multiplicity_spatial_attention import (
    SU2MultiplicitySpatialAttentionConfig,
    su2_multiplicity_spatial_attention_amplitudes,
)
from su2_nqs.models.su2_route_vit import SU2RouteViTConfig, init_su2_route_vit
from su2_nqs.vmc.physical_exchange import run_exchange_sweeps


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_projected_carrier_multiplicity_dressing"


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid projected-carrier dressing identity")
    if spec.get("geometry") != "triangular_4x4_periodic":
        raise ValueError("this exact-small gate is fixed to triangular 4x4")
    if spec.get("preregistered_before_run") is not True:
        raise ValueError("the fresh seeds must be preregistered")
    if tuple(float(value) for value in spec.get("couplings", ())) != (0.0, 0.125):
        raise ValueError("couplings must be the registered pair (0, 0.125)")
    seeds = tuple(int(value) for value in spec.get("seeds", ()))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct fresh seeds are required")
    for name in ("source", "filter_source", "architecture", "training", "audit", "gates"):
        if not isinstance(spec.get(name), dict):
            raise TypeError(f"{name} must be an object")
    if int(spec["training"].get("steps", 0)) < 1:
        raise ValueError("training.steps must be positive")
    if float(spec["training"].get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("training.learning_rate must be positive")


def _load_sealed(source: dict[str, object]) -> tuple[dict[str, object], Path]:
    result_path = ROOT / str(source["result"])
    bundle_path = ROOT / str(source["bundle"])
    result = json.loads(result_path.read_text())
    seal = str(source["result_sha256"])
    if result.get("result_sha256") != seal:
        raise RuntimeError(f"source result seal changed: {result_path}")
    if canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    ) != seal:
        raise RuntimeError(f"source result self-seal is invalid: {result_path}")
    if file_sha256(bundle_path) != str(source["bundle_sha256"]):
        raise RuntimeError(f"source bundle seal changed: {bundle_path}")
    return result, bundle_path


def _model_config(spec: dict[str, object]):
    architecture = spec["architecture"]
    spatial, ledger = spatial_cg_route_spec(
        "triangular",
        4,
        4,
        feature_dim=int(architecture["feature_dim"]),
        maximum_intermediate_q=int(architecture["maximum_intermediate_q"]),
        initialization_scale=float(architecture["initialization_scale"]),
    )
    route = SU2RouteViTConfig(
        spatial=spatial,
        embedding_dim=int(architecture["embedding_dim"]),
        head_count=int(architecture["head_count"]),
        depth=int(architecture["depth"]),
        mlp_ratio=int(architecture["mlp_ratio"]),
        modulation_rank=int(architecture["modulation_rank"]),
        modulation_scale=float(architecture["modulation_scale"]),
        initialization_scale=float(architecture["initialization_scale"]),
    )
    features = multiplicity_spatial_features(
        spatial.backbone.tree,
        spatial.route_orders,
        triangular_j1j2_bonds(4, 4, j1=1.0, j2=0.0, periodic=True),
        triangular_j1j2_bonds(4, 4, j1=0.0, j2=1.0, periodic=True),
    )
    multiplicity = SU2MultiplicitySpatialAttentionConfig(
        base=route,
        features=features,
        embedding_dim=int(architecture["embedding_dim"]),
        head_count=int(architecture["head_count"]),
        modulation_scale=float(architecture["modulation_scale"]),
        initialization_scale=float(architecture["initialization_scale"]),
    )
    group = triangular_hnf_gamma_a1_group(((4, 0), (0, 4)))
    dressing = ProjectedCarrierMultiplicityDressingConfig(
        multiplicity=multiplicity,
        projector=PhysicalSpatialProjector(
            group,
            orbit_chunk_size=24,
            rematerialize_orbit=True,
        ),
    )
    return dressing, {**ledger, "full_gamma_a1_group_order": len(group)}


def _metrics(state: jax.Array, hamiltonian: jax.Array) -> dict[str, float]:
    normalized = state / jnp.linalg.norm(state)
    action = hamiltonian @ normalized
    energy = jnp.real(jnp.vdot(normalized, action))
    residual = action - energy * normalized
    return {
        "energy_total": float(energy),
        "energy_per_site": float(energy / 16.0),
        "variance_total": float(jnp.real(jnp.vdot(residual, residual))),
        "variance_per_site": float(jnp.real(jnp.vdot(residual, residual)) / 16.0),
    }


def _parameter_count(params: object) -> int:
    return int(sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params)))


def _configuration_integers(states: np.ndarray) -> np.ndarray:
    weights = 1 << np.arange(states.shape[1] - 1, -1, -1, dtype=np.int64)
    return np.asarray(states, dtype=np.int64) @ weights


def run_spin_projected_carrier_multiplicity_dressing(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Train and replay the nonleaking N16 projected dressing gate."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    state_path = output / "states.npz"
    if (result_path.exists() or state_path.exists()) and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {output}")
    source_result, source_bundle = _load_sealed(spec["source"])
    filter_result, filter_bundle = _load_sealed(spec["filter_source"])
    dressing_config, route_ledger = _model_config(spec)
    _, configurations = four_cell_schur_basis_matrix(
        "square_plaquette", supports=TRIANGULAR_N16_SUPPORTS
    )
    states = jnp.asarray(configurations, dtype=jnp.int32)
    with np.load(source_bundle, allow_pickle=False) as arrays:
        physical_a1_basis = np.asarray(arrays["physical_a1_basis"], dtype=np.float64)
        carrier = np.asarray(arrays["carrier_a1"], dtype=np.float64)
        hamiltonians = {
            coupling: np.asarray(
                arrays[f"coupling_{coupling:g}_h_a1"], dtype=np.float64
            )
            for coupling in (0.0, 0.125)
        }
        two_lanczos = {
            coupling: np.asarray(
                arrays[f"coupling_{coupling:g}_two_lanczos_a1"], dtype=np.float64
            )
            for coupling in (0.0, 0.125)
        }
    with np.load(filter_bundle, allow_pickle=False) as arrays:
        filter_states = {
            coupling: np.asarray(
                arrays[f"coupling_{coupling:g}_physical_best_a1"],
                dtype=np.complex128,
            )
            for coupling in (0.0, 0.125)
        }
    carrier = carrier / np.linalg.norm(carrier)
    basis_jax = jnp.asarray(physical_a1_basis, dtype=jnp.float64)
    carrier_jax = jnp.asarray(carrier, dtype=jnp.float64)
    training = spec["training"]
    architecture = spec["architecture"]
    steps = int(training["steps"])
    stride = int(training["history_stride"])
    learning_rate = float(training["learning_rate"])

    def reduced_residual(frozen, dressing_params):
        physical = su2_multiplicity_spatial_attention_amplitudes(
            dressing_params["multiplicity"],
            frozen,
            states,
            dressing_config.multiplicity,
        )
        return basis_jax.T @ physical

    def objective(joint, hamiltonian):
        residual = reduced_residual(joint["frozen"], joint["dressing"])
        candidate = (
            carrier_jax
            + joint["dressing"]["residual_coefficient"] * residual
        )
        normalized = candidate / jnp.linalg.norm(candidate)
        action = hamiltonian @ normalized
        return jnp.real(jnp.vdot(normalized, action))

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    runs: dict[float, list[dict[str, object]]] = {0.0: [], 0.125: []}
    best: dict[float, tuple[float, object, np.ndarray]] = {}
    saved: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for coupling in (0.0, 0.125):
        hamiltonian = jnp.asarray(hamiltonians[coupling], dtype=jnp.float64)
        exact_values, exact_vectors = np.linalg.eigh(hamiltonians[coupling])
        exact = exact_vectors[:, 0]
        for seed in (int(value) for value in spec["seeds"]):
            frozen = init_su2_route_vit(
                jax.random.PRNGKey(seed), dressing_config.multiplicity.base
            )
            dressing_params = init_projected_carrier_multiplicity_dressing(
                jax.random.PRNGKey(seed + 100),
                frozen,
                dressing_config,
                initial_residual_coefficient=float(
                    architecture["initial_residual_coefficient"]
                ),
            )
            head = dressing_params["multiplicity"]
            head = {
                **head,
                "control_head": float(architecture["multiplicity_control_scale"])
                * jax.random.normal(
                    jax.random.PRNGKey(seed + 200),
                    head["control_head"].shape,
                    dtype=jnp.float64,
                ),
            }
            dressing_params = {**dressing_params, "multiplicity": head}
            initial_reduced = np.asarray(
                reduced_residual(frozen, dressing_params)
            )
            initial_norm = float(np.linalg.norm(initial_reduced))
            if not np.isfinite(initial_norm) or initial_norm <= 1.0e-14:
                raise RuntimeError("initial projected residual is numerically empty")
            # A fixed gauge choice for conditioning only.  The ClebschTree
            # amplitude is linear in its readout, so this changes no manifold.
            frozen = {
                **frozen,
                "backbone": {
                    **frozen["backbone"],
                    "readout": frozen["backbone"]["readout"] / initial_norm,
                },
            }
            joint = {"frozen": frozen, "dressing": dressing_params}
            optimizer = optax.adam(learning_rate)
            optimizer_state = optimizer.init(joint)
            history = []
            run_started = time.perf_counter()
            for step in range(steps + 1):
                energy, gradient = value_and_grad(joint, hamiltonian)
                if step % stride == 0 or step == steps:
                    history.append(
                        {
                            "step": step,
                            "energy_total": float(energy),
                            "energy_per_site": float(energy / 16.0),
                        }
                    )
                if step == steps:
                    break
                updates, optimizer_state = optimizer.update(
                    gradient, optimizer_state, joint
                )
                joint = optax.apply_updates(joint, updates)
            residual = np.asarray(
                reduced_residual(joint["frozen"], joint["dressing"])
            )
            state = carrier + float(
                joint["dressing"]["residual_coefficient"]
            ) * residual
            state /= np.linalg.norm(state)
            metrics = _metrics(jnp.asarray(state), hamiltonian)
            row = {
                "seed": seed,
                "metrics": metrics,
                "energy_error_to_exact_total": metrics["energy_total"]
                - float(exact_values[0]),
                "fidelity_to_exact": float(abs(np.vdot(exact, state)) ** 2),
                "residual_coefficient": float(
                    joint["dressing"]["residual_coefficient"]
                ),
                "initial_projected_residual_norm": initial_norm,
                "trainable_parameter_count": _parameter_count(joint),
                "history": history,
                "wall_seconds": time.perf_counter() - run_started,
            }
            runs[coupling].append(row)
            if coupling not in best or metrics["energy_total"] < best[coupling][0]:
                best[coupling] = (metrics["energy_total"], joint, state)

    baselines = {}
    summaries = {}
    gates = spec["gates"]
    for coupling in (0.0, 0.125):
        hamiltonian = jnp.asarray(hamiltonians[coupling])
        baseline = {
            "carrier": _metrics(carrier_jax, hamiltonian),
            "two_lanczos": _metrics(jnp.asarray(two_lanczos[coupling]), hamiltonian),
            "entanglement_filter": _metrics(
                jnp.asarray(filter_states[coupling]), hamiltonian
            ),
            "exact_a1": _metrics(
                jnp.asarray(np.linalg.eigh(hamiltonians[coupling])[1][:, 0]),
                hamiltonian,
            ),
        }
        baselines[coupling] = baseline
        energy_wins = sum(
            row["metrics"]["energy_total"]
            < baseline["two_lanczos"]["energy_total"]
            for row in runs[coupling]
        )
        variance_wins = sum(
            row["metrics"]["variance_total"]
            < baseline["two_lanczos"]["variance_total"]
            for row in runs[coupling]
        )
        summaries[coupling] = {
            "best": min(runs[coupling], key=lambda row: row["metrics"]["energy_total"]),
            "seed_count": len(runs[coupling]),
            "seeds_beating_two_lanczos_energy": int(energy_wins),
            "seeds_beating_two_lanczos_variance": int(variance_wins),
            "energy_gate_passed": energy_wins
            >= int(gates["minimum_seeds_beating_two_lanczos_energy"]),
            "variance_gate_passed": variance_wins
            >= int(gates["minimum_seeds_beating_two_lanczos_variance"]),
        }
        saved[f"coupling_{coupling:g}_best_a1"] = np.asarray(best[coupling][2])

    # Directly replay the best trained states with the same physical Reynolds
    # projector used by production sampling.  The lookup represents only the
    # fixed exact carrier; the learned residual remains evaluated on demand.
    configuration_array = np.asarray(configurations, dtype=np.int8)
    carrier_physical = physical_a1_basis @ carrier
    lookup = np.zeros((1 << 16,), dtype=np.complex128)
    lookup[_configuration_integers(configuration_array)] = carrier_physical
    lookup_jax = jnp.asarray(lookup)
    binary_weights = jnp.asarray(1 << np.arange(15, -1, -1), dtype=jnp.int32)

    def carrier_amplitude(state):
        return lookup_jax[jnp.dot(state.astype(jnp.int32), binary_weights)]

    direct_rows = {}
    direct_count = int(spec["audit"]["direct_projector_configuration_count"])
    selected_indices = np.linspace(
        0, len(configuration_array) - 1, direct_count, dtype=np.int32
    )
    selected = jnp.asarray(configuration_array[selected_indices])
    for coupling in (0.0, 0.125):
        joint = best[coupling][1]

        def amplitude(local_joint, state):
            return projected_carrier_multiplicity_dressing_amplitude(
                local_joint["dressing"],
                local_joint["frozen"],
                state,
                dressing_config,
                carrier_amplitude_fn=carrier_amplitude,
            )

        evaluate = jax.jit(
            lambda local_joint, batch: jax.vmap(
                lambda row: amplitude(local_joint, row)
            )(batch)
        )
        direct = np.asarray(evaluate(joint, selected))
        expected_physical = physical_a1_basis @ (
            carrier
            + float(joint["dressing"]["residual_coefficient"])
            * np.asarray(reduced_residual(joint["frozen"], joint["dressing"]))
        )
        expected = expected_physical[selected_indices]
        denominator = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
        relative_error = float(np.linalg.norm(direct - expected) / denominator)
        invariance_error = 0.0
        for image in dressing_config.projector.site_permutations[:4]:
            transformed = selected[:, jnp.asarray(image, dtype=jnp.int32)]
            values = np.asarray(evaluate(joint, transformed))
            invariance_error = max(
                invariance_error,
                float(
                    np.max(np.abs(values - direct))
                    / max(float(np.max(np.abs(direct))), np.finfo(float).tiny)
                ),
            )
        direct_rows[coupling] = {
            "configuration_indices": selected_indices.tolist(),
            "direct_projector_relative_error": relative_error,
            "a1_invariance_relative_error": invariance_error,
            "projector_group_order": dressing_config.projector.group_order,
            "passed": relative_error
            <= float(gates["maximum_direct_projector_relative_error"])
            and invariance_error
            <= float(gates["maximum_a1_invariance_error"]),
        }

    # One short QSL-point integration smoke proves that Metropolis never calls
    # an unprojected surrogate.  This is not an energy estimate.
    qsl_joint = best[0.125][1]

    def qsl_amplitude(local_joint, state):
        return projected_carrier_multiplicity_dressing_amplitude(
            local_joint["dressing"],
            local_joint["frozen"],
            state,
            dressing_config,
            carrier_amplitude_fn=carrier_amplitude,
        )

    sampling_chains = int(spec["audit"]["sampling_chains"])
    sampling_initial = jnp.asarray(configuration_array[selected_indices[:sampling_chains]])
    sample = jax.jit(
        lambda key, initial: run_exchange_sweeps(
            qsl_joint,
            key,
            initial,
            amplitude_fn=qsl_amplitude,
            particle_kind="spin",
            sweeps=int(spec["audit"]["sampling_sweeps"]),
            exchange_edges=tuple(
                (int(first), int(second))
                for first, second, _weight in triangular_j1j2_bonds(
                    4, 4, j1=1.0, j2=0.125, periodic=True
                )
            ),
            bond_proposal_probability=0.7,
            lazy_probability=0.2,
        )
    )
    sampled, cached, _, diagnostics = sample(
        jax.random.PRNGKey(int(spec["audit"]["sampling_seed"])), sampling_initial
    )
    replayed = jax.vmap(lambda row: qsl_amplitude(qsl_joint, row))(sampled)
    sampling_audit = {
        key: float(value) for key, value in diagnostics.items()
    }
    sampling_audit.update(
        {
            "fixed_m2_preserved": bool(
                np.all(np.sum(np.asarray(sampled), axis=1) == 8)
            ),
            "cached_projected_amplitude_max_error": float(
                np.max(np.abs(np.asarray(cached - replayed)))
            ),
            "unprojected_amplitude_path_available_to_sampler": False,
        }
    )
    saved["direct_audit_configuration_indices"] = selected_indices
    saved["sampling_final_configurations"] = np.asarray(sampled)
    _atomic_npz(state_path, saved)

    all_passed = all(
        summary["energy_gate_passed"] and summary["variance_gate_passed"]
        for summary in summaries.values()
    ) and all(row["passed"] for row in direct_rows.values())
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "architecture": {
            "carrier": "fixed exact-S0 pi/0-flux physical amplitude",
            "residual": "configuration-dependent physical ClebschTree amplitude with reduced multiplicity spatial attention",
            "spatial_projection": "full physical Gamma-A1 Reynolds projector on every amplitude call",
            "combination": "additive carrier plus trainable residual",
            "route_ledger": route_ledger,
        },
        "baselines": {str(key): value for key, value in baselines.items()},
        "runs": {str(key): value for key, value in runs.items()},
        "summaries": {str(key): value for key, value in summaries.items()},
        "direct_common_representation_audit": {
            str(key): value for key, value in direct_rows.items()
        },
        "sampling_audit": sampling_audit,
        "gates": {
            "passed": bool(all_passed),
            "variational_energy_and_variance": bool(
                all(
                    summary["energy_gate_passed"]
                    and summary["variance_gate_passed"]
                    for summary in summaries.values()
                )
            ),
            "direct_projector_replay": bool(
                all(row["passed"] for row in direct_rows.values())
            ),
        },
        "decision": (
            "license_n36_sampled_preflight_not_production"
            if all_passed
            else "revise_state_manifold_before_n36"
        ),
        "wall_seconds": time.perf_counter() - started,
        "repeated_ed_performed": False,
        "sealed_small_representation_reused": True,
        "source_artifacts": {
            "carrier_a1": source_result["result_sha256"],
            "entanglement_filter": filter_result["result_sha256"],
        },
        "state_bundle": {
            "path": str(state_path.resolve().relative_to(ROOT.resolve())),
            "sha256": file_sha256(state_path),
        },
        "claim_boundary": spec["claim_boundary"],
        "runtime_identity": numerical_runtime_identity(),
    }
    payload["source_manifest"] = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    payload["source_manifest_sha256"] = canonical_sha256(payload["source_manifest"])
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["TASK", "run_spin_projected_carrier_multiplicity_dressing"]
