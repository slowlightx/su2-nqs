"""Nested low-spin intercluster-resonance profile on the two ``6x6`` tori.

This is a deliberately narrow causal canary.  The exact open-``2x3`` local
multiplet carriers and every local patch tensor are frozen.  Four arms share
one baseline Born sample:

``A0``
    frozen physical-multiplet backbone, including its declared ``0.05``
    support floor (and hence small pre-existing ``q>=4`` support);
``A1``
    ten first-order upper-tree coordinates that include mixing with the local
    first-excited singlet and multiscale ``q=0`` recoupling;
``A2``
    the eight additional multiscale ``q=2``-related tangent coordinates with
    the ``q=0`` arm frozen;
``A12``
    all eighteen nested ``q=0,2`` coordinates.

One sampled-SR direction is formed per active arm.  Cheap singlet ``3 Hz``
reweighting preselects one of four declared step scales, an independent
reference-Born sample performs a full-H correlated confirmation, and a third
independent stream measures physical observables.  The workflow therefore
tests whether missing energy is associated with singlet resonance, virtual
cluster triplets, or destruction of the short-range singlet core; it is not a
converged-energy or phase claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import resource
import shutil
import time
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    pytree_sha256,
)
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.frontier.triangle_patch_physical_init import (
    sample_triangle_patch_cluster_product_chains,
    triangle_patch_cluster_reference,
)
from su2_nqs.frontier.triangle_patch_physical_multiplet_init import (
    init_triangle_patch_cluster_multiplets,
    triangle_patch_cluster_multiplet_reference,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import (
    rectangular_translation_group,
    triangular_hnf_translation_group,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector,
    spatial_projected_amplitude,
)
from su2_nqs.models.triangle_patch_clebsch_tree import (
    triangle_patch_6x6_canary_plan,
    triangle_patch_amplitude,
    triangle_patch_parameter_count,
)
from su2_nqs.vmc.active_subspace import (
    ActiveParameterSubspace,
    triangle_patch_low_spin_upper_tree_mask,
    triangle_patch_low_spin_upper_tree_subspace,
)
from su2_nqs.vmc.cluster_observables import (
    cluster_spin_sector_local_values,
    cluster_spin_squared_from_sector_values,
    cluster_state_projector_local_values,
    prepare_cluster_spin_sectors,
    prepare_cluster_state_projector,
    prepare_spin_pair_observables,
    prepare_translated_cluster_bond_decomposition,
    spin_pair_observable_local_values,
)
from su2_nqs.vmc.physical_exchange import run_exchange_sweeps
from su2_nqs.vmc.physical_sr_controller import correlated_physical_energy_difference
from su2_nqs.vmc.resumable_checkpoint import (
    load_latest_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)
from su2_nqs.vmc.singlet_diagonal_force import three_hz_local_energies
from su2_nqs.vmc.spin_physical import (
    padded_spin_connectivity,
    spin_physical_hpsi,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
TASK = "triangle_patch_n36_low_spin_resonance_profile"
CLAIM_BOUNDARY = (
    "One-step nested A0/A1/A2/A12 frozen-core causal profile on each 6x6 "
    "torus; energies and physical diagnostics are preliminary "
    "sampler-conditioned evidence, not a converged, phase, QSL, scaling, "
    "efficiency, or PRL claim."
)

SINGLE_CARRIER_CAPACITY_NOTE = (
    "A0 already contains the declared 0.05 support floor, including q=2 and "
    "q>=4 weight. A2 releases q=2-related multiscale tangents coupled to one "
    "lowest-triplet local carrier; it is not an on/off triplet comparison. "
    "Independent open-2x6 perturbative guides show that this single carrier "
    "captures only part of the second-order energy, so a weak A2 result does "
    "not by itself rule out an S=1 mechanism."
)


@dataclass(frozen=True)
class _Runtime:
    plan: object
    params: object
    projector: PhysicalSpatialProjector
    amplitude: Callable[[object, jax.Array], jax.Array]


@dataclass(frozen=True)
class _AuditKernels:
    """Case-static observable specs and JIT callables shared by all arms."""

    bond_spec: object
    bond_kernel: Callable[[object, jax.Array], jax.Array]
    sector_spec: object
    sector_kernel: Callable[[object, jax.Array], jax.Array]
    core_kernel: Callable[[object, jax.Array], jax.Array]
    direction_spec: object
    direction_kernel: Callable[[object, jax.Array], jax.Array]
    structure: tuple[tuple[str, np.ndarray], ...]


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return value if value > 10**8 else value * 1024


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _positive_finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _validate(spec: dict[str, object]) -> None:
    """Fail closed around the registered causal profile."""

    if (
        spec.get("schema_version") != 1
        or spec.get("domain") != "spin"
        or spec.get("task") != TASK
        or spec.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("invalid low-spin-resonance profile identity")
    if spec.get("model") != {
        "family": "strict_triangle_patch_clebsch_tree",
        "feature_dim": 2,
        "maximum_intermediate_q": 10,
        "parameter_count": 1252,
    }:
        raise ValueError("model must remain the D2/qmax10/P1252 strict tree")
    initialization = spec.get("initialization")
    if not isinstance(initialization, dict) or initialization != {
        "kind": "cluster_lowest_multiplets",
        "pre_projection_background_coefficient": 0.05,
        "route_gate_scale": 0.05,
        "local_q0_channel_0": "open_2x3_ground_singlet",
        "local_q0_channel_1": "open_2x3_first_excited_singlet",
        "local_q2_channel_1": "open_2x3_lowest_triplet",
    }:
        raise ValueError("the exact local-multiplet initialization changed")
    if spec.get("projection") != {
        "kind": "translation_k0",
        "group_order": 36,
        "orbit_chunk_size": 6,
        "rematerialize_orbit": True,
    }:
        raise ValueError("the exact 36-translation projector is required")

    expected_cases = {
        "square_6x6_j2_0p5": ("square", 0.5, -0.503810, -0.503765),
        "triangular_6x6_j2_0p125": (
            "triangular",
            0.125,
            -0.515564,
            -0.515386,
        ),
    }
    cases = spec.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError("the square and triangular N36 cases are required")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("each case must be an object")
        case_id = str(case.get("case_id", ""))
        if case_id not in expected_cases or case_id in seen:
            raise ValueError("case ids must be the unique registered pair")
        seen.add(case_id)
        geometry, ratio, exact, literature = expected_cases[case_id]
        if (
            case.get("geometry") != geometry
            or float(case.get("j2_over_j1", np.nan)) != ratio
            or float(case.get("literature_reference_energy_per_site", np.nan))
            != exact
            or float(case.get("exact_energy_per_site", np.nan)) != exact
            or case.get("reference_energy_provenance")
            != "external_literature_guide_not_locally_recomputed_n36_ed"
            or float(case.get("raw_literature_energy_per_site", np.nan))
            != literature
        ):
            raise ValueError("case coupling or comparison energy changed")
        for seed_name in (
            "model_seed",
            "train_seed",
            "confirmation_seed",
            "final_audit_seed",
        ):
            _integer(case.get(seed_name), f"case.{seed_name}")

    sampler = spec.get("sampler")
    if not isinstance(sampler, dict) or (
        sampler.get("chains"),
        sampler.get("initial_chain_kind"),
        sampler.get("burn_proposals_per_chain"),
        sampler.get("snapshots"),
        sampler.get("proposals_between_snapshots"),
        sampler.get("chain_major_output"),
    ) != (
        32,
        "cluster_product_born_random_translation",
        576,
        8,
        72,
        True,
    ):
        raise ValueError("the shared training sampler contract changed")
    probability = float(sampler.get("bond_proposal_probability", np.nan))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("bond proposal probability must lie in [0,1]")

    optimizer = spec.get("optimizer")
    expected_arms = [
        {"arm_id": "A0_frozen", "active_q": None, "expected_dimension": 0},
        {"arm_id": "A1_q0", "active_q": 0, "expected_dimension": 10},
        {
            "arm_id": "A2_q2_increment",
            "active_q": 2,
            "difference_from_q0": True,
            "expected_dimension": 8,
        },
        {"arm_id": "A12_q0_q2", "active_q": 2, "expected_dimension": 18},
    ]
    if not isinstance(optimizer, dict) or optimizer.get("arms") != expected_arms:
        raise ValueError("optimizer arms must remain the nested A0/A1/A2/A12 set")
    if (
        optimizer.get("name"),
        optimizer.get("active_stage"),
        optimizer.get("qgt_center_microbatch_size"),
        optimizer.get("persistent_solver_parity_in_profile"),
    ) != ("explicit_small_active_qgt_sr", "first_order", 2, False):
        raise ValueError("active SR execution contract changed")
    for name in (
        "damping",
        "step_size",
        "fisher_trust_radius",
    ):
        _positive_finite(optimizer.get(name), f"optimizer.{name}")

    line = spec.get("line_search")
    if not isinstance(line, dict):
        raise TypeError("line_search must be an object")
    scales = tuple(float(value) for value in line.get("training_step_scales", ()))
    if scales != (1.0, 0.5, 0.25, 0.125):
        raise ValueError("the declared training scales changed")
    if (
        line.get("require_training_mean_decrease") is not True
        or line.get("strict_acceptance_requires_negative_upper_bound") is not True
        or line.get("unresolved_negative_mean_is_recorded_not_discarded") is not True
    ):
        raise ValueError("the conservative line-search decisions changed")
    for name in (
        "minimum_training_ess_fraction",
        "confirmation_confidence_level",
        "minimum_confirmation_ess_fraction",
        "maximum_confirmation_variance_ratio",
    ):
        _positive_finite(line.get(name), f"line_search.{name}")

    audit = spec.get("final_audit")
    if not isinstance(audit, dict) or (
        audit.get("chains"),
        audit.get("burn_proposals_per_chain"),
        audit.get("snapshots"),
        audit.get("proposals_between_snapshots"),
        audit.get("full_h_microbatch_size"),
        audit.get("physics_observable_samples"),
        audit.get("representative_cluster_sites"),
    ) != (32, 576, 4, 72, 2, 32, [0, 1, 2, 6, 7, 8]):
        raise ValueError("the independent physical-audit contract changed")


def _geometry(case: Mapping[str, object]):
    ratio = float(case["j2_over_j1"])
    if case["geometry"] == "square":
        return (
            square_j1j2_bonds(6, 6, j1=1.0, j2=ratio, periodic=True),
            rectangular_translation_group(6, 6),
        )
    return (
        triangular_j1j2_bonds(6, 6, j1=1.0, j2=ratio, periodic=True),
        triangular_hnf_translation_group(((6, 0), (0, 6))),
    )


def _runtime(case: Mapping[str, object], spec: Mapping[str, object]) -> _Runtime:
    plan = triangle_patch_6x6_canary_plan(str(case["geometry"]))
    initialization = spec["initialization"]
    params = init_triangle_patch_cluster_multiplets(
        jax.random.PRNGKey(int(case["model_seed"])),
        plan,
        j2_over_j1=float(case["j2_over_j1"]),
        background_scale=float(
            initialization["pre_projection_background_coefficient"]
        ),
        route_gate_scale=float(initialization["route_gate_scale"]),
    )
    if triangle_patch_parameter_count(params) != 1252:
        raise RuntimeError("the registered parameter count changed")
    _, group = _geometry(case)
    projection = spec["projection"]
    projector = PhysicalSpatialProjector(
        group,
        orbit_chunk_size=int(projection["orbit_chunk_size"]),
        rematerialize_orbit=bool(projection["rematerialize_orbit"]),
    )

    def amplitude(candidate, state):
        return spatial_projected_amplitude(
            candidate,
            state,
            plan,
            projector,
            amplitude_fn=triangle_patch_amplitude,
        )

    return _Runtime(plan=plan, params=params, projector=projector, amplitude=amplitude)


def _arm_subspaces(
    params: object,
    plan: object,
) -> dict[str, ActiveParameterSubspace | None]:
    q0 = triangle_patch_low_spin_upper_tree_subspace(
        params, plan, stage="first_order", maximum_active_q=0
    )
    q02 = triangle_patch_low_spin_upper_tree_subspace(
        params, plan, stage="first_order", maximum_active_q=2
    )
    q0_mask = triangle_patch_low_spin_upper_tree_mask(
        params, plan, stage="first_order", maximum_active_q=0
    )
    q02_mask = triangle_patch_low_spin_upper_tree_mask(
        params, plan, stage="first_order", maximum_active_q=2
    )
    q2_mask = jax.tree_util.tree_map(
        lambda complete, singlet: np.asarray(complete, dtype=bool)
        & ~np.asarray(singlet, dtype=bool),
        q02_mask,
        q0_mask,
    )
    q2 = ActiveParameterSubspace.from_mask(params, q2_mask)
    result = {
        "A0_frozen": None,
        "A1_q0": q0,
        "A2_q2_increment": q2,
        "A12_q0_q2": q02,
    }
    dimensions = {name: 0 if value is None else value.dimension for name, value in result.items()}
    if dimensions != {
        "A0_frozen": 0,
        "A1_q0": 10,
        "A2_q2_increment": 8,
        "A12_q0_q2": 18,
    }:
        raise RuntimeError(f"low-spin active dimensions changed: {dimensions}")
    return result


def _kernels(runtime: _Runtime, bonds):
    edges = tuple((first, second) for first, second, _ in bonds)

    def uncached(params, key, states, sweeps, probability):
        return run_exchange_sweeps(
            params,
            key,
            states,
            amplitude_fn=runtime.amplitude,
            particle_kind="spin",
            sweeps=sweeps,
            amplitudes=None,
            exchange_edges=edges,
            bond_proposal_probability=probability,
        )

    def cached(params, key, states, amplitudes, sweeps, probability):
        return run_exchange_sweeps(
            params,
            key,
            states,
            amplitude_fn=runtime.amplitude,
            particle_kind="spin",
            sweeps=sweeps,
            amplitudes=amplitudes,
            exchange_edges=edges,
            bond_proposal_probability=probability,
        )

    def centers(params, states):
        return jax.vmap(lambda state: runtime.amplitude(params, state))(states)

    def hpsi(params, batch):
        samples, connected, elements, valid = batch
        return spin_physical_hpsi(
            params,
            samples,
            connected,
            elements,
            valid,
            runtime.plan,
            amplitude_fn=lambda candidate, state, _plan: runtime.amplitude(
                candidate, state
            ),
        )

    return (
        jax.jit(uncached, static_argnums=(3, 4)),
        jax.jit(cached, static_argnums=(4, 5)),
        jax.jit(centers),
        jax.jit(hpsi),
    )


def _float_sampler_row(row: Mapping[str, object]) -> dict[str, float]:
    return {str(name): float(value) for name, value in row.items()}


def _mean_sampler_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    return {f"{key}_mean": float(np.mean([row[key] for row in rows])) for key in keys}


def _sample_reference_born(
    params: object,
    key: jax.Array,
    runtime: _Runtime,
    *,
    ratio: float,
    chains: int,
    burn: int,
    snapshots: int,
    separation: int,
    probability: float,
    uncached,
    cached,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return samples in chain-major order plus a sampler ledger."""

    start_key, mh_key = jax.random.split(key)
    states = sample_triangle_patch_cluster_product_chains(
        start_key,
        runtime.plan,
        j2_over_j1=ratio,
        n_chains=chains,
        translate=True,
    )
    states, amplitudes, mh_key, burn_row = uncached(
        params, mh_key, states, burn, probability
    )
    jax.block_until_ready((states, amplitudes, mh_key, burn_row))
    rows = [_float_sampler_row(burn_row)]
    collected = []
    for _ in range(snapshots):
        states, amplitudes, mh_key, row = cached(
            params, mh_key, states, amplitudes, separation, probability
        )
        jax.block_until_ready((states, amplitudes, mh_key, row))
        rows.append(_float_sampler_row(row))
        collected.append(np.asarray(states, dtype=np.int32).copy())
    snapshots_first = np.stack(collected, axis=0)
    chain_major = np.transpose(snapshots_first, (1, 0, 2)).reshape(
        chains * snapshots, runtime.plan.n_sites
    )
    if np.any(np.sum(chain_major, axis=1) != runtime.plan.n_sites // 2):
        raise RuntimeError("sampler left the fixed-M=0 sector")
    return chain_major, {
        "sample_count": int(len(chain_major)),
        "chain_count": chains,
        "snapshots_per_chain": snapshots,
        "chain_major": True,
        "states_sha256": array_sha256(chain_major),
        "burn": rows[0],
        "post_burn_mean": _mean_sampler_rows(rows[1:]),
    }


def _score_matrix(
    center_fn: Callable[[jax.Array, jax.Array], jax.Array],
    active: jax.Array,
    samples: np.ndarray,
    *,
    microbatch_size: int,
) -> tuple[np.ndarray, float]:
    """Materialize only the small active score matrix for diagnostics."""

    size = int(microbatch_size)
    if len(samples) % size:
        raise ValueError("score sample count must divide the microbatch size")

    def log_centers(values, states):
        centers = center_fn(values, states).astype(jnp.complex128)
        return jnp.log(centers)

    kernel = jax.jit(jax.jacfwd(log_centers, argnums=0))
    rows = []
    started = time.perf_counter()
    for start in range(0, len(samples), size):
        block = kernel(active, jnp.asarray(samples[start : start + size]))
        jax.block_until_ready(block)
        rows.append(np.asarray(block, dtype=np.complex128))
    matrix = np.concatenate(rows, axis=0)
    if matrix.shape != (len(samples), len(active)) or np.any(~np.isfinite(matrix)):
        raise FloatingPointError("active score matrix is non-finite or malformed")
    return matrix, time.perf_counter() - started


def _qgt_and_force(
    scores: np.ndarray,
    local_energies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centered_scores = scores - np.mean(scores, axis=0, keepdims=True)
    centered_energy = local_energies - np.mean(local_energies)
    qgt = 2.0 * np.real(centered_scores.conj().T @ centered_scores) / len(scores)
    qgt = 0.5 * (qgt + qgt.T)
    gradient = 2.0 * np.real(
        np.mean(np.conj(centered_scores) * centered_energy[:, None], axis=0)
    )
    return qgt, gradient


def _qgt_summary(qgt: np.ndarray, damping: float) -> dict[str, object]:
    matrix = np.asarray(qgt, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("qgt must be square")
    if matrix.size == 0:
        return {
            "eigenvalues": [],
            "numerical_rank": 0,
            "stable_rank": 0.0,
            "condition_number_positive_subspace": None,
            "damped_condition_number": 1.0,
        }
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    tolerance = max(matrix.shape) * np.finfo(float).eps * scale
    positive = eigenvalues[eigenvalues > tolerance]
    spectral = float(np.max(np.abs(eigenvalues)))
    stable_rank = (
        float(np.sum(eigenvalues**2) / spectral**2) if spectral > 0.0 else 0.0
    )
    return {
        "eigenvalues": [float(value) for value in eigenvalues],
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "numerical_rank": int(len(positive)),
        "rank_tolerance": tolerance,
        "stable_rank": stable_rank,
        "condition_number_positive_subspace": (
            float(positive[-1] / positive[0]) if len(positive) else None
        ),
        "damped_condition_number": float(
            (eigenvalues[-1] + damping) / (eigenvalues[0] + damping)
        ),
    }


def _explicit_active_sr_direction(
    qgt: np.ndarray,
    gradient: np.ndarray,
    *,
    damping: float,
    step_size: float,
    fisher_trust_radius: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve the already-materialized 8/10/18 dimensional SR system."""

    matrix = np.asarray(qgt, dtype=np.float64)
    force = np.asarray(gradient, dtype=np.float64)
    if matrix.shape != (len(force), len(force)) or force.ndim != 1:
        raise ValueError("qgt and gradient dimensions disagree")
    damped = matrix + float(damping) * np.eye(len(force))
    raw = np.linalg.solve(damped, force)
    residual = damped @ raw - force
    raw_fisher_norm = math.sqrt(max(float(raw @ matrix @ raw), 0.0))
    proposed_fisher = float(step_size) * raw_fisher_norm
    trust_scale = (
        min(1.0, float(fisher_trust_radius) / proposed_fisher)
        if proposed_fisher > 0.0
        else 1.0
    )
    direction = float(step_size) * trust_scale * raw
    if np.any(~np.isfinite(direction)):
        raise FloatingPointError("explicit active SR produced a non-finite direction")
    diagnostics = {
        "solver": "dense_symmetric_solve_on_materialized_active_qgt",
        "dimension": len(force),
        "damping": float(damping),
        "step_size": float(step_size),
        "fisher_trust_radius": float(fisher_trust_radius),
        "gradient_norm": float(np.linalg.norm(force)),
        "raw_direction_norm": float(np.linalg.norm(raw)),
        "raw_fisher_norm": raw_fisher_norm,
        "trust_scale": trust_scale,
        "direction_norm": float(np.linalg.norm(direction)),
        "fisher_norm": float(step_size) * trust_scale * raw_fisher_norm,
        "linear_residual_norm": float(np.linalg.norm(residual)),
        "relative_linear_residual": float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(force), np.finfo(np.float64).tiny)
        ),
        "persistent_matrix_free_parity_skipped_by_profile": True,
        "persistent_solver_covered_by_independent_unit_tests": True,
    }
    return direction, diagnostics


def _cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 0.0 else None


def _split_half_diagnostics(
    scores: np.ndarray,
    energies: np.ndarray,
    *,
    chains: int,
    snapshots: int,
    damping: float,
) -> dict[str, object]:
    shaped_scores = scores.reshape(chains, snapshots, -1)
    shaped_energies = energies.reshape(chains, snapshots)
    half = chains // 2
    first_scores = shaped_scores[:half].reshape(-1, scores.shape[1])
    second_scores = shaped_scores[half:].reshape(-1, scores.shape[1])
    first_energies = shaped_energies[:half].reshape(-1)
    second_energies = shaped_energies[half:].reshape(-1)
    qgt_first, gradient_first = _qgt_and_force(first_scores, first_energies)
    qgt_second, gradient_second = _qgt_and_force(second_scores, second_energies)
    identity = np.eye(scores.shape[1])
    direction_first = np.linalg.solve(qgt_first + damping * identity, gradient_first)
    direction_second = np.linalg.solve(qgt_second + damping * identity, gradient_second)
    return {
        "gradient_cosine": _cosine(gradient_first, gradient_second),
        "damped_direction_cosine": _cosine(direction_first, direction_second),
        "first_gradient_norm": float(np.linalg.norm(gradient_first)),
        "second_gradient_norm": float(np.linalg.norm(gradient_second)),
    }


def _training_preselect(
    reference_logs: np.ndarray,
    candidate_logs_by_scale: Mapping[float, np.ndarray],
    diagonal_local_energies: np.ndarray,
    *,
    minimum_ess_fraction: float,
) -> tuple[float | None, list[dict[str, object]]]:
    """Select the largest declared scale using only the training sample."""

    reference = np.asarray(reference_logs, dtype=np.complex128)
    local = np.asarray(diagonal_local_energies, dtype=np.float64)
    reference_energy = float(np.mean(local))
    attempts = []
    selected = None
    for scale, raw_candidate in candidate_logs_by_scale.items():
        candidate = np.asarray(raw_candidate, dtype=np.complex128)
        log_weight = 2.0 * np.real(candidate - reference)
        log_weight -= np.max(log_weight)
        weights = np.exp(log_weight)
        weights /= np.sum(weights)
        energy = float(np.sum(weights * local))
        ess = float(1.0 / np.sum(weights**2))
        reasons = []
        if energy >= reference_energy:
            reasons.append("nonnegative_training_mean_change")
        if ess / len(local) < minimum_ess_fraction:
            reasons.append("training_importance_ess")
        accepted = not reasons
        attempts.append(
            {
                "step_scale": float(scale),
                "energy": energy,
                "reference_energy": reference_energy,
                "energy_difference": energy - reference_energy,
                "effective_sample_size": ess,
                "effective_sample_size_fraction": ess / len(local),
                "accepted_for_independent_confirmation": accepted,
                "rejection_reasons": reasons,
            }
        )
        if accepted:
            selected = float(scale)
            break
    return selected, attempts


def _full_local_energies(
    params: object,
    samples: np.ndarray,
    bonds,
    hpsi,
    *,
    microbatch_size: int,
) -> np.ndarray:
    connected, elements, valid, _ = padded_spin_connectivity(
        samples, bonds, padded_width=len(bonds) + 1
    )
    rows = []
    for start in range(0, len(samples), microbatch_size):
        stop = min(start + microbatch_size, len(samples))
        batch = tuple(
            jnp.asarray(value[start:stop])
            for value in (samples, connected, elements, valid)
        )
        centers, actions = hpsi(params, batch)
        jax.block_until_ready((centers, actions))
        rows.append(np.asarray(actions) / np.asarray(centers))
    result = np.concatenate(rows).astype(np.complex128)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("full-H local energies are non-finite")
    return result


def _directional_bond_modes(geometry: str, bonds):
    """Group Hamiltonian bonds by unoriented periodic coordinate direction."""

    grouped: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for first, second, coupling in bonds:
        first_rc = divmod(first, 6)
        second_rc = divmod(second, 6)
        raw_dr = second_rc[0] - first_rc[0]
        raw_dc = second_rc[1] - first_rc[1]
        candidates = [
            (raw_dr + row_shift, raw_dc + col_shift)
            for row_shift in (-6, 0, 6)
            for col_shift in (-6, 0, 6)
        ]
        if geometry == "square":
            metric = lambda item: item[0] ** 2 + item[1] ** 2
        else:
            metric = lambda item: item[0] ** 2 + item[1] ** 2 - item[0] * item[1]
        dr, dc = min(candidates, key=lambda item: (metric(item), abs(item[0]) + abs(item[1]), item))
        if dr < 0 or (dr == 0 and dc < 0):
            dr, dc = -dr, -dc
        grouped.setdefault((dr, dc), []).append((first, second, coupling))
    return tuple(
        (f"direction_{dr}_{dc}", 0.0, tuple(grouped[(dr, dc)]))
        for dr, dc in sorted(grouped)
    )


def _structure_factor_specs(geometry: str):
    coordinates = tuple((row, column) for row in range(6) for column in range(6))
    if geometry == "square":
        modes = (
            ("neel_pi_pi", (math.pi, math.pi)),
            ("stripe_pi_0", (math.pi, 0.0)),
            ("stripe_0_pi", (0.0, math.pi)),
        )
    else:
        k = 2.0 * math.pi / 3.0
        modes = (
            ("120_K", (k, k)),
            ("stripe_M_pi_0", (math.pi, 0.0)),
            ("stripe_M_0_pi", (0.0, math.pi)),
            ("stripe_M_pi_pi", (math.pi, math.pi)),
        )
    points = np.asarray(coordinates, dtype=np.float64)
    return tuple(
        (
            label,
            np.exp(1.0j * (points @ np.asarray(wavevector, dtype=np.float64))),
        )
        for label, wavevector in modes
    )


def _singlet_diagonal_structure_factors(
    samples: np.ndarray,
    modes: tuple[tuple[str, np.ndarray], ...],
) -> np.ndarray:
    """Unbiased full-spin ``S(q)`` estimators for an exact global singlet.

    Rotational invariance gives ``<S.S>=3<Sz Sz>``.  Consequently no
    connected model amplitudes are needed: each Born sample contributes
    ``3/N |sum_i w_i s_i^z|^2``.  This identity is valid for every arm here
    because the Clebsch root fixes total ``q=0`` exactly.
    """

    spins = np.asarray(samples, dtype=np.float64) - 0.5
    if spins.ndim != 2 or not modes:
        raise ValueError("samples and structure-factor modes must be nonempty")
    weights = np.stack([np.asarray(weight, dtype=np.complex128) for _, weight in modes])
    if weights.shape[1] != spins.shape[1]:
        raise ValueError("structure-factor weights do not match sample sites")
    fourier = spins @ weights.T
    return 3.0 * np.abs(fourier) ** 2 / spins.shape[1]


def _build_audit_kernels(
    runtime: _Runtime,
    case: Mapping[str, object],
    spec: Mapping[str, object],
    bonds,
) -> _AuditKernels:
    """Construct each expensive JIT wrapper once per geometry/case."""

    bond_spec = prepare_translated_cluster_bond_decomposition(6, 6, bonds)
    bond_kernel = jax.jit(
        lambda candidate, states: spin_pair_observable_local_values(
            candidate,
            states,
            bond_spec.observables,
            amplitude_fn=runtime.amplitude,
        )
    )
    representative = (
        tuple(int(site) for site in spec["final_audit"]["representative_cluster_sites"]),
    )
    sector_spec = prepare_cluster_spin_sectors(36, representative)
    sector_kernel = jax.jit(
        lambda candidate, states: cluster_spin_sector_local_values(
            candidate, states, sector_spec, amplitude_fn=runtime.amplitude
        )
    )
    local_ground = triangle_patch_cluster_reference(
        str(case["geometry"]), float(case["j2_over_j1"])
    ).computational_state
    core_spec = prepare_cluster_state_projector(36, representative, local_ground)
    core_kernel = jax.jit(
        lambda candidate, states: cluster_state_projector_local_values(
            candidate, states, core_spec, amplitude_fn=runtime.amplitude
        )
    )
    direction_spec = prepare_spin_pair_observables(
        36, _directional_bond_modes(str(case["geometry"]), bonds)
    )
    direction_kernel = jax.jit(
        lambda candidate, states: spin_pair_observable_local_values(
            candidate, states, direction_spec, amplitude_fn=runtime.amplitude
        )
    )
    return _AuditKernels(
        bond_spec=bond_spec,
        bond_kernel=bond_kernel,
        sector_spec=sector_spec,
        sector_kernel=sector_kernel,
        core_kernel=core_kernel,
        direction_spec=direction_spec,
        direction_kernel=direction_kernel,
        structure=_structure_factor_specs(str(case["geometry"])),
    )


def _batched_observable(kernel, params, samples: np.ndarray, *, size: int = 2):
    rows = []
    for start in range(0, len(samples), size):
        values = kernel(params, jnp.asarray(samples[start : start + size]))
        jax.block_until_ready(values)
        rows.append(np.asarray(values))
    result = np.concatenate(rows)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("a physical observable produced non-finite values")
    return result


def _real_mcse(values: np.ndarray, *, scale: float = 1.0) -> float:
    """Across-independent-chain standard error for one audit time slice."""

    concrete = np.real(np.asarray(values, dtype=np.complex128)).reshape(-1)
    if concrete.size < 2:
        raise ValueError("an across-chain MCSE requires at least two chains")
    return float(np.std(concrete, ddof=1) / math.sqrt(concrete.size) * scale)


def _physical_audit(
    params: object,
    key: jax.Array,
    runtime: _Runtime,
    case: Mapping[str, object],
    spec: Mapping[str, object],
    bonds,
    uncached,
    cached,
    hpsi,
    audit_kernels: _AuditKernels,
) -> dict[str, object]:
    audit = spec["final_audit"]
    probability = float(spec["sampler"]["bond_proposal_probability"])
    samples, sampler = _sample_reference_born(
        params,
        key,
        runtime,
        ratio=float(case["j2_over_j1"]),
        chains=int(audit["chains"]),
        burn=int(audit["burn_proposals_per_chain"]),
        snapshots=int(audit["snapshots"]),
        separation=int(audit["proposals_between_snapshots"]),
        probability=probability,
        uncached=uncached,
        cached=cached,
    )
    full = _full_local_energies(
        params,
        samples,
        bonds,
        hpsi,
        microbatch_size=int(audit["full_h_microbatch_size"]),
    )
    mean = complex(np.mean(full))
    shaped = full.real.reshape(int(audit["chains"]), int(audit["snapshots"]))
    chain_means = np.mean(shaped, axis=1) / 36.0
    energy_mcse = float(np.std(chain_means, ddof=1) / math.sqrt(len(chain_means)))
    variance = float(np.mean(np.abs(full - mean) ** 2).real / 36.0)

    # Use one common time slice per chain for the more expensive diagnostics.
    physical_count = int(audit["physics_observable_samples"])
    physical = samples.reshape(int(audit["chains"]), int(audit["snapshots"]), 36)[
        :physical_count, 0
    ]
    bond_spec = audit_kernels.bond_spec
    bond_values = _batched_observable(audit_kernels.bond_kernel, params, physical)
    bond_mean = np.mean(bond_values, axis=0)
    bond_mcse = np.asarray(
        [_real_mcse(bond_values[:, index], scale=1.0 / 36.0) for index in range(bond_values.shape[1])]
    )

    representative = (tuple(int(site) for site in audit["representative_cluster_sites"]),)
    sector_spec = audit_kernels.sector_spec
    sectors = _batched_observable(audit_kernels.sector_kernel, params, physical)[:, 0, :]
    sector_mean = np.mean(sectors, axis=0)
    sector_mcse = np.asarray(
        [_real_mcse(sectors[:, index]) for index in range(sectors.shape[1])]
    )
    s2_values = cluster_spin_squared_from_sector_values(
        jnp.asarray(sectors), sector_spec.q_values
    )

    core_values = _batched_observable(audit_kernels.core_kernel, params, physical)[:, 0]

    direction_spec = audit_kernels.direction_spec
    direction_values = _batched_observable(
        audit_kernels.direction_kernel, params, physical
    )
    direction_mcse = np.asarray(
        [
            _real_mcse(direction_values[:, index], scale=1.0 / 36.0)
            for index in range(direction_values.shape[1])
        ]
    )

    structure = audit_kernels.structure
    structure_values = _singlet_diagonal_structure_factors(physical, structure)

    external_reference = float(case["literature_reference_energy_per_site"])
    literature = float(case["raw_literature_energy_per_site"])
    energy = float(mean.real / 36.0)
    result = {
        "sampler": sampler,
        "params_sha256": pytree_sha256(params),
        "energy_per_site": energy,
        "energy_per_site_chain_mcse": energy_mcse,
        "mean_imaginary_energy_per_site": float(mean.imag / 36.0),
        "physical_hamiltonian_variance_per_site": variance,
        "gap_to_external_reference_energy_per_site": energy - external_reference,
        "external_reference_not_locally_verified_by_n36_ed": True,
        "gap_to_raw_literature_energy_per_site": energy - literature,
        "cluster_bond_energy_per_site": {
            "translation_average_intra": float(
                np.real(bond_mean[bond_spec.average_intra_index]) / 36.0
            ),
            "translation_average_intra_mcse": float(
                bond_mcse[bond_spec.average_intra_index]
            ),
            "translation_average_inter": float(
                np.real(bond_mean[bond_spec.average_inter_index]) / 36.0
            ),
            "translation_average_inter_mcse": float(
                bond_mcse[bond_spec.average_inter_index]
            ),
            "total": float(np.real(bond_mean[bond_spec.total_index]) / 36.0),
            "total_mcse": float(bond_mcse[bond_spec.total_index]),
            "partition_residual": float(
                np.real(
                    bond_mean[bond_spec.average_intra_index]
                    + bond_mean[bond_spec.average_inter_index]
                    - bond_mean[bond_spec.total_index]
                )
                / 36.0
            ),
        },
        "representative_cluster": {
            "sites": list(representative[0]),
            "q_values": list(sector_spec.q_values),
            "P_S0": float(np.real(sector_mean[0])),
            "P_S0_mcse": float(sector_mcse[0]),
            "P_S1": float(np.real(sector_mean[1])),
            "P_S1_mcse": float(sector_mcse[1]),
            "P_S_ge_2": float(np.real(np.sum(sector_mean[2:]))),
            "P_S_ge_2_mcse": _real_mcse(np.sum(sectors[:, 2:], axis=1)),
            "probability_sum_residual": float(np.real(np.sum(sector_mean) - 1.0)),
            "S_C_squared": float(np.real(np.mean(np.asarray(s2_values)))),
            "S_C_squared_mcse": _real_mcse(np.asarray(s2_values)),
            "rank1_ground_singlet_fidelity": float(np.real(np.mean(core_values))),
            "rank1_ground_singlet_fidelity_mcse": _real_mcse(core_values),
            "maximum_mean_imaginary_component": float(
                max(
                    np.max(np.abs(np.imag(sector_mean))),
                    abs(np.imag(np.mean(core_values))),
                )
            ),
        },
        "directional_bond_energy_per_site": {
            label: float(value)
            for label, value in zip(
                direction_spec.labels,
                np.real(np.mean(direction_values, axis=0)) / 36.0,
                strict=True,
            )
        },
        "directional_bond_energy_per_site_mcse": {
            label: float(value)
            for label, value in zip(
                direction_spec.labels, direction_mcse, strict=True
            )
        },
        "structure_factors": {
            label: float(value)
            for (label, _), value in zip(
                structure,
                np.real(np.mean(structure_values, axis=0)),
                strict=True,
            )
        },
        "structure_factor_mcse": {
            label: _real_mcse(structure_values[:, index])
            for index, (label, _) in enumerate(structure)
        },
        "structure_factor_estimator": (
            "exact_global_singlet_identity_3_over_N_times_diagonal_Sz_Fourier_norm"
        ),
        "physics_observable_sample_count": len(physical),
        "samples_are_autocorrelated_within_chain": True,
        "physical_interpretation_is_preliminary": True,
    }
    return result


def _arm_indices(
    complete: ActiveParameterSubspace,
    arm: ActiveParameterSubspace,
) -> np.ndarray:
    lookup = {index: position for position, index in enumerate(complete.active_flat_indices)}
    try:
        return np.asarray([lookup[index] for index in arm.active_flat_indices], dtype=np.int32)
    except KeyError as error:
        raise RuntimeError("an arm is not nested in A12") from error


def _run_active_arm(
    arm_id: str,
    subspace: ActiveParameterSubspace,
    score_rows: np.ndarray,
    train_energies: np.ndarray,
    train_samples: np.ndarray,
    runtime: _Runtime,
    case: Mapping[str, object],
    spec: Mapping[str, object],
    bonds,
    uncached,
    cached,
    centers,
    hpsi,
    audit_kernels: _AuditKernels,
    *,
    arm_index: int,
) -> tuple[dict[str, object], object]:
    optimizer = spec["optimizer"]
    line = spec["line_search"]
    active0 = subspace.pack(runtime.params)
    qgt, gradient = _qgt_and_force(score_rows, train_energies)

    def active_centers(active, samples):
        return centers(subspace.inject(active), samples)

    direction, explicit_sr = _explicit_active_sr_direction(
        qgt,
        gradient,
        damping=float(optimizer["damping"]),
        step_size=float(optimizer["step_size"]),
        fisher_trust_radius=float(optimizer["fisher_trust_radius"]),
    )
    direction = jnp.asarray(direction, dtype=active0.dtype)

    reference_logs = np.log(
        np.asarray(active_centers(active0, jnp.asarray(train_samples))).astype(np.complex128)
    )
    candidates: dict[float, jax.Array] = {}
    candidate_logs: dict[float, np.ndarray] = {}
    for scale in tuple(float(value) for value in line["training_step_scales"]):
        candidate = active0 - scale * direction
        candidates[scale] = candidate
        candidate_logs[scale] = np.log(
            np.asarray(active_centers(candidate, jnp.asarray(train_samples))).astype(
                np.complex128
            )
        )
    selected_scale, training_attempts = _training_preselect(
        reference_logs,
        candidate_logs,
        train_energies,
        minimum_ess_fraction=float(line["minimum_training_ess_fraction"]),
    )

    confirmation = None
    strict_accepted = False
    proposed_active = active0
    confirmation_sampler = None
    if selected_scale is not None:
        proposed_active = candidates[selected_scale]
        confirmation_key = jax.random.fold_in(
            jax.random.PRNGKey(int(case["confirmation_seed"])), arm_index
        )
        heldout, confirmation_sampler = _sample_reference_born(
            runtime.params,
            confirmation_key,
            runtime,
            ratio=float(case["j2_over_j1"]),
            chains=int(spec["sampler"]["chains"]),
            burn=int(spec["sampler"]["burn_proposals_per_chain"]),
            snapshots=int(spec["sampler"]["snapshots"]),
            separation=int(spec["sampler"]["proposals_between_snapshots"]),
            probability=float(spec["sampler"]["bond_proposal_probability"]),
            uncached=uncached,
            cached=cached,
        )

        def active_full_local(active, samples):
            return jnp.asarray(
                _full_local_energies(
                    subspace.inject(active),
                    np.asarray(samples, dtype=np.int32),
                    bonds,
                    hpsi,
                    microbatch_size=int(spec["final_audit"]["full_h_microbatch_size"]),
                )
            )

        familywise_confidence = float(line["confirmation_confidence_level"])
        per_arm_confidence = 1.0 - (1.0 - familywise_confidence) / 3.0
        compared = correlated_physical_energy_difference(
            active0,
            proposed_active,
            jnp.asarray(heldout),
            local_energy_fn=active_full_local,
            center_fn=active_centers,
            batch_count=int(spec["sampler"]["chains"]),
            standard_error_method="max",
            confidence_level=per_arm_confidence,
        )
        reasons = []
        if compared.upper_confidence_bound >= 0.0:
            reasons.append("nonnegative_correlated_upper_bound")
        if (
            compared.candidate.effective_sample_size_fraction
            < float(line["minimum_confirmation_ess_fraction"])
        ):
            reasons.append("confirmation_importance_ess")
        if compared.variance_ratio > float(line["maximum_confirmation_variance_ratio"]):
            reasons.append("physical_variance_ratio")
        strict_accepted = not reasons
        confirmation = {
            "comparison": asdict(compared),
            "familywise_confidence_across_three_active_arms": familywise_confidence,
            "per_arm_bonferroni_confidence": per_arm_confidence,
            "strict_accepted": strict_accepted,
            "rejection_reasons": reasons,
            "negative_mean_but_statistically_unresolved": bool(
                compared.energy_difference < 0.0
                and compared.upper_confidence_bound >= 0.0
            ),
        }

    # A rejected proposal remains scientifically informative, so the third
    # stream audits it while the decision state rolls back bit-for-bit.
    audited_active = proposed_active
    audited_params = subspace.inject(audited_active)
    decision_active = proposed_active if strict_accepted else active0
    decision_params = subspace.inject(decision_active)
    audit_key = jax.random.fold_in(
        jax.random.PRNGKey(int(case["final_audit_seed"])), arm_index
    )
    audit_started = time.perf_counter()
    audit = _physical_audit(
        audited_params,
        audit_key,
        runtime,
        case,
        spec,
        bonds,
        uncached,
        cached,
        hpsi,
        audit_kernels,
    )
    audit_seconds = time.perf_counter() - audit_started
    return (
        {
            "arm_id": arm_id,
            "active_dimension": subspace.dimension,
            "active_flat_indices": list(subspace.active_flat_indices),
            "qgt": _qgt_summary(qgt, float(optimizer["damping"])),
            "split_half": _split_half_diagnostics(
                score_rows,
                train_energies,
                chains=int(spec["sampler"]["chains"]),
                snapshots=int(spec["sampler"]["snapshots"]),
                damping=float(optimizer["damping"]),
            ),
            "energy_gradient_norm": float(np.linalg.norm(gradient)),
            "energy_gradient": [float(value) for value in gradient],
            "explicit_active_sr": explicit_sr,
            "training_line_search": {
                "selected_scale": selected_scale,
                "attempts": training_attempts,
            },
            "confirmation_sampler": confirmation_sampler,
            "independent_full_h_confirmation": confirmation,
            "strictly_accepted": strict_accepted,
            "decision_rolled_back": not strict_accepted,
            "third_stream_audits_proposal_even_if_unresolved": True,
            "proposal_params_sha256": pytree_sha256(audited_params),
            "decision_params_sha256": pytree_sha256(decision_params),
            "physical_audit": audit,
            "physical_audit_seconds": audit_seconds,
        },
        decision_params,
    )


def _run_case(
    case: Mapping[str, object],
    spec: Mapping[str, object],
    output: Path,
    *,
    overwrite: bool,
    source_manifest_sha256: str,
    runtime_identity_sha256: str,
) -> dict[str, object]:
    started = time.perf_counter()
    runtime = _runtime(case, spec)
    bonds, _ = _geometry(case)
    uncached, cached, centers, hpsi = _kernels(runtime, bonds)
    audit_kernels = _build_audit_kernels(runtime, case, spec, bonds)
    subspaces = _arm_subspaces(runtime.params, runtime.plan)
    train_samples, train_sampler = _sample_reference_born(
        runtime.params,
        jax.random.PRNGKey(int(case["train_seed"])),
        runtime,
        ratio=float(case["j2_over_j1"]),
        chains=int(spec["sampler"]["chains"]),
        burn=int(spec["sampler"]["burn_proposals_per_chain"]),
        snapshots=int(spec["sampler"]["snapshots"]),
        separation=int(spec["sampler"]["proposals_between_snapshots"]),
        probability=float(spec["sampler"]["bond_proposal_probability"]),
        uncached=uncached,
        cached=cached,
    )
    train_energies = np.asarray(three_hz_local_energies(train_samples, bonds))
    complete = subspaces["A12_q0_q2"]
    assert complete is not None

    def complete_centers(active, states):
        return centers(complete.inject(active), states)

    complete_active = complete.pack(runtime.params)
    score, score_seconds = _score_matrix(
        complete_centers,
        complete_active,
        train_samples,
        microbatch_size=int(spec["optimizer"]["qgt_center_microbatch_size"]),
    )

    initialization = {
        "params_sha256": pytree_sha256(runtime.params),
        "local_multiplets": {
            f"q{q}_level{1 if q == 0 else 0}": {
                "energy": reference.energy,
                "gap_above": reference.gap_above,
                "multiplicity_dimension": reference.multiplicity_dimension,
                "eigen_residual": reference.eigen_residual,
                "spin_squared_residual": reference.spin_squared_residual,
            }
            for q in (0, 2, 4, 6)
            for reference in (
                triangle_patch_cluster_multiplet_reference(
                    str(case["geometry"]),
                    float(case["j2_over_j1"]),
                    q,
                    eigen_index=1 if q == 0 else 0,
                ),
            )
        },
        "local_ground_singlet": {
            "energy": triangle_patch_cluster_reference(
                str(case["geometry"]), float(case["j2_over_j1"])
            ).energy,
            "rank1_fidelity_is_distinct_from_P_S0": True,
        },
    }
    identity = {
        "task": TASK,
        "case_id": case["case_id"],
        "config_sha256": canonical_sha256(spec),
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "initial_params_sha256": initialization["params_sha256"],
        "train_states_sha256": train_sampler["states_sha256"],
    }
    checkpoint_dir = output / "checkpoints" / str(case["case_id"])
    if overwrite and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint = None
    if checkpoint_dir.exists() and not overwrite:
        checkpoint = load_latest_resumable_checkpoint(
            checkpoint_dir, expected_run_identity=identity
        )
    if checkpoint is None:
        completed = 0
        history: list[dict[str, object]] = []
        arm_params: dict[str, object] = {}
    else:
        completed = int(checkpoint["completed_steps"])
        history = list(checkpoint["history"])
        arm_params = dict(checkpoint["params"])

    arm_order = ("A0_frozen", "A1_q0", "A2_q2_increment", "A12_q0_q2")
    for arm_index, arm_id in enumerate(arm_order[completed:], start=completed):
        if arm_id == "A0_frozen":
            audit_key = jax.random.fold_in(
                jax.random.PRNGKey(int(case["final_audit_seed"])), arm_index
            )
            audit_started = time.perf_counter()
            audit = _physical_audit(
                runtime.params,
                audit_key,
                runtime,
                case,
                spec,
                bonds,
                uncached,
                cached,
                hpsi,
                audit_kernels,
            )
            row = {
                "arm_id": arm_id,
                "active_dimension": 0,
                "qgt": _qgt_summary(np.empty((0, 0)), float(spec["optimizer"]["damping"])),
                "split_half": None,
                "training_line_search": None,
                "independent_full_h_confirmation": None,
                "strictly_accepted": True,
                "decision_rolled_back": False,
                "proposal_params_sha256": pytree_sha256(runtime.params),
                "decision_params_sha256": pytree_sha256(runtime.params),
                "physical_audit": audit,
                "physical_audit_seconds": time.perf_counter() - audit_started,
            }
            decision_params = runtime.params
        else:
            subspace = subspaces[arm_id]
            assert subspace is not None
            columns = _arm_indices(complete, subspace)
            row, decision_params = _run_active_arm(
                arm_id,
                subspace,
                score[:, columns],
                train_energies,
                train_samples,
                runtime,
                case,
                spec,
                bonds,
                uncached,
                cached,
                centers,
                hpsi,
                audit_kernels,
                arm_index=arm_index,
            )
        history.append(row)
        arm_params[arm_id] = decision_params
        completed = arm_index + 1
        save_resumable_checkpoint_generation(
            checkpoint_dir,
            arm_params,
            optimizer_state={
                "kind": "one_step_explicit_small_active_qgt_sr",
                "completed_arms": completed,
            },
            completed_steps=completed,
            next_sample_index=completed * len(train_samples),
            run_identity=identity,
            history=history,
            metadata={
                "arm_order": list(arm_order),
                "shared_train_sampler": train_sampler,
                "shared_score_matrix_sha256": array_sha256(score),
            },
        )

    baseline_energy = history[0]["physical_audit"]["energy_per_site"]
    reference_energy = float(case["literature_reference_energy_per_site"])
    denominator = baseline_energy - reference_energy
    baseline_audit = history[0]["physical_audit"]
    for row in history:
        is_baseline = row["arm_id"] == "A0_frozen"
        current = row["physical_audit"]["energy_per_site"]
        row["recovered_gap_fraction_against_external_literature_guide"] = (
            float((baseline_energy - current) / denominator)
            if denominator != 0.0
            else None
        )
        audit = row["physical_audit"]
        def difference(value, base_value, standard_error, base_standard_error):
            delta = float(value - base_value)
            combined = (
                0.0
                if is_baseline
                else float(math.hypot(standard_error, base_standard_error))
            )
            return {
                "difference": delta,
                "independent_stream_combined_mcse": combined,
                "z_like_difference_over_combined_mcse": (
                    delta / combined if combined > 0.0 else None
                ),
            }

        row["physical_change_from_A0"] = {
            "energy_per_site": float(current - baseline_energy),
            "intra_energy_per_site": float(
                audit["cluster_bond_energy_per_site"]["translation_average_intra"]
                - baseline_audit["cluster_bond_energy_per_site"][
                    "translation_average_intra"
                ]
            ),
            "inter_energy_per_site": float(
                audit["cluster_bond_energy_per_site"]["translation_average_inter"]
                - baseline_audit["cluster_bond_energy_per_site"][
                    "translation_average_inter"
                ]
            ),
            "P_S0": float(
                audit["representative_cluster"]["P_S0"]
                - baseline_audit["representative_cluster"]["P_S0"]
            ),
            "P_S1": float(
                audit["representative_cluster"]["P_S1"]
                - baseline_audit["representative_cluster"]["P_S1"]
            ),
            "P_S_ge_2": float(
                audit["representative_cluster"]["P_S_ge_2"]
                - baseline_audit["representative_cluster"]["P_S_ge_2"]
            ),
            "rank1_ground_singlet_fidelity": float(
                audit["representative_cluster"]["rank1_ground_singlet_fidelity"]
                - baseline_audit["representative_cluster"][
                    "rank1_ground_singlet_fidelity"
                ]
            ),
        }
        bond = audit["cluster_bond_energy_per_site"]
        base_bond = baseline_audit["cluster_bond_energy_per_site"]
        cluster = audit["representative_cluster"]
        base_cluster = baseline_audit["representative_cluster"]
        row["physical_change_from_A0_with_independent_chain_uncertainty"] = {
            "energy_per_site": difference(
                audit["energy_per_site"],
                baseline_audit["energy_per_site"],
                audit["energy_per_site_chain_mcse"],
                baseline_audit["energy_per_site_chain_mcse"],
            ),
            "intra_energy_per_site": difference(
                bond["translation_average_intra"],
                base_bond["translation_average_intra"],
                bond["translation_average_intra_mcse"],
                base_bond["translation_average_intra_mcse"],
            ),
            "inter_energy_per_site": difference(
                bond["translation_average_inter"],
                base_bond["translation_average_inter"],
                bond["translation_average_inter_mcse"],
                base_bond["translation_average_inter_mcse"],
            ),
            **{
                name: difference(
                    cluster[name],
                    base_cluster[name],
                    cluster[f"{name}_mcse"],
                    base_cluster[f"{name}_mcse"],
                )
                for name in (
                    "P_S0",
                    "P_S1",
                    "P_S_ge_2",
                    "S_C_squared",
                    "rank1_ground_singlet_fidelity",
                )
            },
            "directional_bond_energy_per_site": {
                label: difference(
                    value,
                    baseline_audit["directional_bond_energy_per_site"][label],
                    audit["directional_bond_energy_per_site_mcse"][label],
                    baseline_audit["directional_bond_energy_per_site_mcse"][label],
                )
                for label, value in audit["directional_bond_energy_per_site"].items()
            },
            "structure_factors": {
                label: difference(
                    value,
                    baseline_audit["structure_factors"][label],
                    audit["structure_factor_mcse"][label],
                    baseline_audit["structure_factor_mcse"][label],
                )
                for label, value in audit["structure_factors"].items()
            },
        }
        row["arm_physical_interpretation"] = {
            "A0_frozen": "frozen_support_floor_baseline_not_pure_cluster_product",
            "A1_q0": "local_excited_singlet_mixing_and_multiscale_q0_recoupling_marginal_response",
            "A2_q2_increment": "multiscale_q2_related_recoupling_marginal_response_not_triplet_on_off",
            "A12_q0_q2": "joint_low_spin_multiscale_recoupling_response",
        }[row["arm_id"]]
    return {
        "case_id": case["case_id"],
        "geometry": case["geometry"],
        "status": "complete",
        "initialization": initialization,
        "translation_group_order": runtime.projector.group_order,
        "shared_training_sampler": train_sampler,
        "shared_train_three_hz_energy_per_site": float(np.mean(train_energies) / 36.0),
        "shared_score_matrix_shape": list(score.shape),
        "shared_score_matrix_sha256": array_sha256(score),
        "shared_score_matrix_seconds": score_seconds,
        "external_reference_energy_per_site": reference_energy,
        "external_reference_not_locally_verified_by_n36_ed": True,
        "baseline_contains_background_scale_0p05_support_floor": True,
        "single_carrier_capacity_note": SINGLE_CARRIER_CAPACITY_NOTE,
        "active_direction_solver": "explicit_8_10_18_dimensional_damped_qgt_solve",
        "audit_kernels_constructed_once_and_reused_across_arms": True,
        "arms": history,
        "checkpoint_directory": str(checkpoint_dir.relative_to(ROOT)),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def run_triangle_patch_n36_low_spin_resonance(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
    case_id: str | None = None,
    smoke: bool = False,
    local_pilot: bool = False,
) -> dict[str, object]:
    """Run or resume the square and triangular nested causal profiles."""

    config_path = config_path.resolve()
    output = output.resolve()
    registered_spec = json.loads(config_path.read_text())
    _validate(registered_spec)
    if smoke and local_pilot:
        raise ValueError("smoke and local_pilot are mutually exclusive")
    # Smoke mode changes only Monte Carlo work.  The registered and executed
    # specifications are separately sealed so it cannot mimic the profile.
    spec = json.loads(json.dumps(registered_spec))
    if smoke:
        spec["sampler"].update(
            {
                "chains": 4,
                "burn_proposals_per_chain": 4,
                "snapshots": 2,
                "proposals_between_snapshots": 2,
            }
        )
        spec["final_audit"].update(
            {
                "chains": 4,
                "burn_proposals_per_chain": 4,
                "snapshots": 1,
                "proposals_between_snapshots": 2,
                "physics_observable_samples": 4,
            }
        )
    elif local_pilot:
        # This intermediate tier is large enough to expose gross direction
        # instability while remaining practical on a laptop.  It deliberately
        # does not inherit the registered-profile label: both specifications
        # and the execution mode are sealed in the result.
        spec["sampler"].update(
            {
                "chains": 16,
                "burn_proposals_per_chain": 64,
                "snapshots": 4,
                "proposals_between_snapshots": 18,
            }
        )
        spec["final_audit"].update(
            {
                "chains": 16,
                "burn_proposals_per_chain": 64,
                "snapshots": 2,
                "proposals_between_snapshots": 18,
                "physics_observable_samples": 32,
            }
        )
    selected_cases = list(spec["cases"])
    if case_id is not None:
        selected_cases = [
            case for case in selected_cases if case["case_id"] == case_id
        ]
        if not selected_cases:
            raise ValueError(f"unknown case_id: {case_id}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result exists; pass --overwrite: {result_path}")
    source_manifest = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    source_manifest_sha256 = canonical_sha256(source_manifest)
    runtime_identity = numerical_runtime_identity()
    runtime_identity_sha256 = canonical_sha256(runtime_identity)
    started = time.perf_counter()
    cases = [
        _run_case(
            case,
            spec,
            output,
            overwrite=overwrite,
            source_manifest_sha256=source_manifest_sha256,
            runtime_identity_sha256=runtime_identity_sha256,
        )
        for case in selected_cases
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config": registered_spec,
        "config_sha256": canonical_sha256(registered_spec),
        "execution_spec": spec,
        "execution_spec_sha256": canonical_sha256(spec),
        "selected_case_id": case_id,
        "smoke_mode": smoke,
        "local_pilot_mode": local_pilot,
        "execution_tier": (
            "smoke" if smoke else "local_pilot" if local_pilot else "registered"
        ),
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_identity_sha256,
        "cases": cases,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "hamiltonian_diagonalization_on_n36_performed": False,
        "local_six_site_diagonalization_only": True,
        "causal_question": (
            "Does frozen-core q=0 resonance and/or q=2 virtual-triplet "
            "recoupling recover intercluster energy without destroying the "
            "rank-one short-range singlet core?"
        ),
        "baseline_contains_frozen_support_floor": True,
        "single_carrier_capacity_note": SINGLE_CARRIER_CAPACITY_NOTE,
        "external_local_capacity_guides_not_measured_by_this_workflow": {
            "square_single_lowest_triplet_second_order_energy_fraction": 0.347,
            "triangular_single_lowest_triplet_second_order_energy_fraction": 0.165,
            "square_leading_boundary_response_schmidt_norm_fraction": 0.87,
            "triangular_leading_boundary_response_schmidt_norm_fraction": 0.82,
            "boundary_response_rank_guide": "2/3",
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = [
    "CLAIM_BOUNDARY",
    "TASK",
    "run_triangle_patch_n36_low_spin_resonance",
]
