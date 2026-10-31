"""Two-update projected matrix-free-SR canary on the ``6x6`` spin tori.

The workflow is intentionally a numerical and memory gate, not an energy
benchmark.  Every training sample is drawn from the translation-projected
exact-singlet state.  The inexpensive exact-singlet ``3 Hz`` estimator
provides the force, while independent frozen-parameter full-H audits at
updates zero and two are the only reported physical energies and variances.

Runtime construction, projected exchange kernels, and the paired full-H
audit are reused from :mod:`spin_triangle_patch_n36_energy_smoke`; this file
owns only the four-snapshot sample collection, matrix-free SR step, and its
resumable trajectory contract.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

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
from su2_nqs.models.triangle_patch_clebsch_tree import (
    init_triangle_patch_clebsch_tree,
    triangle_patch_parameter_count,
)
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.physical_sampled_sr import sampled_physical_sr_direction
from su2_nqs.vmc.resumable_checkpoint import (
    load_latest_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)
from su2_nqs.vmc.singlet_diagonal_force import (
    sampled_singlet_diagonal_value_and_grad,
    three_hz_local_energies,
)

from . import spin_triangle_patch_n36_energy_smoke as energy_smoke
from .spin_triangle_patch_n36_runtime import _peak_rss_bytes


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
TASK = "triangle_patch_n36_sr_pilot"
CLAIM_BOUNDARY = (
    "Two-update projected matrix-free-SR optimization and memory canary only; "
    "no converged energy, efficiency, phase, QSL, scaling, or PRL advantage "
    "is claimed."
)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _validate(spec: dict[str, object]) -> None:
    """Fail closed around the registered two-update local SR canary."""

    if (
        spec.get("schema_version") != 1
        or spec.get("domain") != "spin"
        or spec.get("task") != TASK
        or spec.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("invalid N36 SR-pilot identity or claim boundary")
    model = spec.get("model")
    if not isinstance(model, dict) or model != {
        "family": "strict_triangle_patch_clebsch_tree",
        "feature_dim": 2,
        "maximum_intermediate_q": 10,
        "parameter_count": 1252,
        "initialization_scale": 0.6,
        "route_gate_scale": 0.05,
    }:
        raise ValueError("model must be the registered D2/qmax10/P1252 canary")
    initialization = spec.get("initialization")
    if not isinstance(initialization, dict) or set(initialization) != {
        "kind",
        "pre_projection_background_coefficient",
    }:
        raise ValueError(
            "initialization must register the pre-projection background coefficient"
        )
    if initialization.get("kind") != "cluster_eigenvector":
        raise ValueError("the SR pilot must start from the physical cluster eigenvector")
    background_scale = float(
        initialization.get("pre_projection_background_coefficient", np.nan)
    )
    if not np.isfinite(background_scale) or background_scale != 0.05:
        raise ValueError(
            "initialization.pre_projection_background_coefficient must be the "
            "N36-validated value 0.05"
        )

    projection = spec.get("projection")
    if not isinstance(projection, dict) or projection != {
        "kind": "translation_k0",
        "group_order": 36,
        "orbit_chunk_size": 6,
        "rematerialize_orbit": True,
        "applied_from_update": 0,
    }:
        raise ValueError("the exact 36-translation projector must apply from update zero")

    expected = {
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
        raise ValueError("the SR pilot requires exactly two N36 cases")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("every case must be an object")
        case_id = str(case.get("case_id", ""))
        if case_id not in expected or case_id in seen:
            raise ValueError("case ids must be the unique square/triangular pair")
        seen.add(case_id)
        geometry, ratio, exact, literature = expected[case_id]
        if (
            case.get("geometry"),
            case.get("rows"),
            case.get("columns"),
            case.get("periodic"),
            case.get("ed"),
        ) != (geometry, 6, 6, True, False):
            raise ValueError("each case must be a periodic non-ED 6x6 torus")
        if (
            float(case.get("j2_over_j1", np.nan)) != ratio
            or float(case.get("exact_energy_per_site", np.nan)) != exact
            or float(case.get("raw_literature_energy_per_site", np.nan))
            != literature
        ):
            raise ValueError("case coupling or raw literature target changed")
        _integer(case.get("model_seed"), "case.model_seed")
        _integer(case.get("sampler_seed"), "case.sampler_seed")

    sampler = spec.get("sampler")
    if not isinstance(sampler, dict) or (
        sampler.get("chains"),
        sampler.get("initial_chain_kind"),
        sampler.get("random_control_kind"),
        sampler.get("initial_chain_probe_proposals_per_chain"),
        sampler.get("initial_burn_proposals_per_chain"),
        sampler.get("snapshots_per_update"),
        sampler.get("proposals_between_snapshots"),
    ) != (
        16,
        "cluster_product_born_random_translation",
        "uniform_fixed_sz0",
        36,
        288,
        4,
        36,
    ):
        raise ValueError(
            "sampler must use physical cluster-product starts, a fixed-Sz "
            "random control, and four 36-proposal snapshots"
        )
    probability = float(sampler.get("bond_proposal_probability", np.nan))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("bond_proposal_probability must lie in [0,1]")

    force = spec.get("force")
    if not isinstance(force, dict) or force != {
        "estimator": "exact_singlet_3hz",
        "microbatch_size": 2,
    }:
        raise ValueError("training force must be exact-singlet 3Hz in batches of two")
    optimizer = spec.get("optimizer")
    expected_optimizer = {
        "name": "matrix_free_sampled_physical_sr",
        "updates": 2,
        "damping": 0.01,
        "cg_tolerance": 0.001,
        "cg_max_iterations": 20,
        "maximum_relative_parameter_step": 0.02,
        "fisher_trust_radius": 0.05,
        "step_size": 0.05,
        "qgt_center_microbatch_size": 2,
        "qgt_rematerialize_center": True,
        "checkpoint_every": 1,
    }
    if not isinstance(optimizer, dict) or optimizer != expected_optimizer:
        raise ValueError("optimizer is not the registered two-update SR gate")
    audit = spec.get("audit")
    if not isinstance(audit, dict) or (
        audit.get("updates"),
        audit.get("snapshots"),
        audit.get("full_h_microbatch_size"),
    ) != ([0, 2], 2, 2):
        raise ValueError("independent paired full-H audits are required at 0 and 2")
    for name in (
        "seed",
        "burn_proposals_per_chain",
        "proposals_between_snapshots",
    ):
        _integer(audit.get(name), f"audit.{name}", minimum=1)


def _initialize_params(runtime: object, case: dict[str, object], spec: dict[str, object]):
    """Apply the registered physical initializer without changing the model."""

    initialization = spec["initialization"]
    key = jax.random.PRNGKey(int(case["model_seed"]))
    if initialization["kind"] == "cluster_eigenvector":
        # Imported lazily so config/artifact tests remain independent of the
        # Hamiltonian-informed frontier helper's optional construction cost.
        from su2_nqs.frontier.triangle_patch_physical_init import (
            init_triangle_patch_cluster_singlet,
        )

        params = init_triangle_patch_cluster_singlet(
            key,
            runtime.plan,
            j2_over_j1=float(case["j2_over_j1"]),
            background_scale=float(
                initialization["pre_projection_background_coefficient"]
            ),
            route_gate_scale=float(spec["model"]["route_gate_scale"]),
        )
    else:  # Kept as a fail-closed defensive branch; _validate rejects it.
        params = init_triangle_patch_clebsch_tree(
            key,
            runtime.plan,
            scale=float(spec["model"]["initialization_scale"]),
            route_gate_scale=float(spec["model"]["route_gate_scale"]),
        )
    if triangle_patch_parameter_count(params) != int(spec["model"]["parameter_count"]):
        raise RuntimeError("physical initializer changed the registered parameter count")
    return replace(runtime, params=params)


def _cluster_branch_params(
    runtime: object,
    case: dict[str, object],
    spec: dict[str, object],
):
    """Return the same deterministic initializer with its support floor off."""

    from su2_nqs.frontier.triangle_patch_physical_init import (
        init_triangle_patch_cluster_singlet,
    )

    params = init_triangle_patch_cluster_singlet(
        jax.random.PRNGKey(int(case["model_seed"])),
        runtime.plan,
        j2_over_j1=float(case["j2_over_j1"]),
        background_scale=0.0,
        route_gate_scale=float(spec["model"]["route_gate_scale"]),
    )
    if triangle_patch_parameter_count(params) != int(spec["model"]["parameter_count"]):
        raise RuntimeError("cluster branch changed the registered parameter count")
    return params


def _center_support_summary(values: object) -> dict[str, float | int | None]:
    """Summarize projected support without interpreting tiny values as energy."""

    centers = np.asarray(values)
    if centers.ndim != 1 or centers.size < 1:
        raise ValueError("one projected center is required per initial chain")
    magnitudes = np.abs(centers)
    if np.any(~np.isfinite(magnitudes)):
        raise FloatingPointError("initial projected centers contain non-finite values")
    exact_nonzero = magnitudes > 0.0
    above_threshold = magnitudes > 1.0e-14
    nonzero_log_magnitudes = np.log(magnitudes[exact_nonzero])
    return {
        "chain_count": int(magnitudes.size),
        "exact_zero_count": int(np.sum(~exact_nonzero)),
        "exact_nonzero_fraction": float(np.mean(exact_nonzero)),
        "above_absolute_1e-14_fraction": float(np.mean(above_threshold)),
        "magnitude_minimum": float(np.min(magnitudes)),
        "magnitude_median": float(np.median(magnitudes)),
        "magnitude_maximum": float(np.max(magnitudes)),
        "log_magnitude_mean_over_nonzero": (
            float(np.mean(nonzero_log_magnitudes))
            if nonzero_log_magnitudes.size
            else None
        ),
        "log_magnitude_median_over_nonzero": (
            float(np.median(nonzero_log_magnitudes))
            if nonzero_log_magnitudes.size
            else None
        ),
    }


def _initial_chain_ensembles(
    key: jax.Array,
    runtime: object,
    case: dict[str, object],
    spec: dict[str, object],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build the physical training starts and independent random control."""

    (
        physics_chain_key,
        random_control_key,
        physics_probe_key,
        random_probe_key,
        sampling_key,
    ) = jax.random.split(key, 5)
    from su2_nqs.frontier.triangle_patch_physical_init import (
        sample_triangle_patch_cluster_product_chains,
    )

    physical_states = sample_triangle_patch_cluster_product_chains(
        physics_chain_key,
        runtime.plan,
        j2_over_j1=float(case["j2_over_j1"]),
        n_chains=int(spec["sampler"]["chains"]),
        translate=True,
    )
    random_states = initialize_spin_exchange_chains(
        random_control_key,
        n_chains=int(spec["sampler"]["chains"]),
        n_sites=36,
        n_up=18,
    )
    return (
        physical_states,
        random_states,
        physics_probe_key,
        random_probe_key,
        sampling_key,
    )


def _initial_chain_diagnostic(
    params: object,
    cluster_branch_params: object,
    states: jax.Array,
    probe_key: jax.Array,
    *,
    chain_kind: str,
    spec: dict[str, object],
    centers,
    uncached,
) -> dict[str, object]:
    """Compare mixed-state and main-branch support plus a short MH probe.

    The advanced states are deliberately discarded.  Physics-seeded and
    uniformly random fixed-``Sz`` starts therefore receive the same diagnostic
    work without allowing the random control to enter the training trajectory.
    """

    mixed_centers = centers(params, states)
    main_centers = centers(cluster_branch_params, states)
    advanced, amplitudes, _, row = uncached(
        params,
        probe_key,
        states,
        int(spec["sampler"]["initial_chain_probe_proposals_per_chain"]),
        float(spec["sampler"]["bond_proposal_probability"]),
    )
    jax.block_until_ready((mixed_centers, main_centers, advanced, amplitudes, row))
    host_states = np.asarray(states, dtype=np.int32)
    host_advanced = np.asarray(advanced, dtype=np.int32)
    return {
        "chain_kind": chain_kind,
        "states_sha256": array_sha256(host_states),
        "fixed_sz0_verified": bool(np.all(np.sum(host_states, axis=1) == 18)),
        "translation_randomized": chain_kind
        == "cluster_product_born_random_translation",
        "mixed_projected_support": _center_support_summary(mixed_centers),
        "cluster_product_main_branch_support": _center_support_summary(main_centers),
        "probe_proposals_per_chain": int(
            spec["sampler"]["initial_chain_probe_proposals_per_chain"]
        ),
        "probe_sampler": energy_smoke._float_row(row),
        "probe_chain_snapshot_move_fraction": float(
            np.mean(np.any(host_advanced != host_states, axis=1))
        ),
        "probe_final_states_sha256": array_sha256(host_advanced),
        "probe_final_mixed_support": _center_support_summary(amplitudes),
        "diagnostic_only_not_used_for_training": True,
    }


def _chain_block_ess(values: np.ndarray) -> dict[str, float | int | str]:
    """Return a conservative chain-block ESS diagnostic for four snapshots."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("ESS values must have shape (snapshots>=2, chains>=2)")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("ESS values contain non-finite entries")
    sample_count = int(array.size)
    # Use matching population normalizations: mixing ``N-1`` corrections for
    # all observations and for the shorter vector of chain means would bias
    # the diagnostic even for four perfectly persistent copies.
    marginal_variance = float(np.var(array, ddof=0))
    chain_means = np.mean(array, axis=0)
    chain_mean_variance = float(np.var(chain_means, ddof=0))
    if marginal_variance == 0.0:
        estimate = float(sample_count)
    elif chain_mean_variance == 0.0:
        estimate = float(sample_count)
    else:
        estimate = float(
            np.clip(
                array.shape[1] * marginal_variance / chain_mean_variance,
                1.0,
                sample_count,
            )
        )
    return {
        "method": "between_chain_means_short_trajectory_diagnostic",
        "sample_count": sample_count,
        "chain_count": int(array.shape[1]),
        "snapshots_per_chain": int(array.shape[0]),
        "effective_sample_size": estimate,
        "effective_sample_size_fraction": estimate / sample_count,
        "short_trajectory_not_publication_ess": True,
    }


def _collect_training_samples(
    params: object,
    states: jax.Array,
    sampling_key: jax.Array,
    *,
    spec: dict[str, object],
    uncached,
    cached,
) -> tuple[jax.Array, jax.Array, np.ndarray, list[dict[str, float]]]:
    sampler = spec["sampler"]
    snapshots: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []
    amplitudes = None
    for snapshot in range(int(sampler["snapshots_per_update"])):
        if snapshot == 0:
            states, amplitudes, sampling_key, row = uncached(
                params,
                sampling_key,
                states,
                int(sampler["proposals_between_snapshots"]),
                float(sampler["bond_proposal_probability"]),
            )
        else:
            states, amplitudes, sampling_key, row = cached(
                params,
                sampling_key,
                states,
                amplitudes,
                int(sampler["proposals_between_snapshots"]),
                float(sampler["bond_proposal_probability"]),
            )
        jax.block_until_ready((states, amplitudes, sampling_key, row))
        concrete_row = energy_smoke._float_row(row)
        if concrete_row["zero_amplitude_chain_fraction"] != 0.0:
            raise FloatingPointError("projected SR chain reached a zero amplitude")
        diagnostics.append(concrete_row)
        snapshots.append(np.asarray(states, dtype=np.int32).copy())
    samples = np.concatenate(snapshots, axis=0)
    if samples.shape != (64, 36) or np.any(np.sum(samples, axis=1) != 18):
        raise RuntimeError("SR samples escaped the registered fixed-M2=0 shape")
    return states, sampling_key, samples, diagnostics


def _mean_sampler_diagnostics(rows: list[dict[str, float]]) -> dict[str, float]:
    names = (
        "acceptance_fraction_including_null",
        "accepted_move_fraction",
        "conditional_move_acceptance",
        "nontrivial_proposal_fraction",
        "zero_amplitude_chain_fraction",
    )
    return {
        f"{name}_mean": float(np.mean([row[name] for row in rows]))
        for name in names
    }


def _rematerialized_center_microbatch_fn(
    center_fn,
    *,
    microbatch_size: int,
    n_sites: int = 36,
):
    """Wrap a batched center kernel in sequential rematerialized microbatches.

    The returned callback retains the ``center_fn(params, states)`` interface
    required by sampled SR.  It prevents a 64-state translation-projected
    VJP/JVP from retaining every orbit image simultaneously.
    """

    checked_size = _integer(
        microbatch_size, "qgt_center_microbatch_size", minimum=1
    )
    checked_sites = _integer(n_sites, "n_sites", minimum=1)

    def chunk_center(candidate, chunk):
        return center_fn(candidate, chunk)

    rematerialized = jax.checkpoint(chunk_center)

    def wrapped(candidate, states):
        array = jnp.asarray(states)
        if array.ndim != 2 or array.shape[1] != checked_sites:
            raise ValueError("QGT center states have the wrong physical shape")
        if array.shape[0] % checked_size:
            raise ValueError("QGT center sample count must divide the microbatch size")
        chunks = array.reshape((-1, checked_size, checked_sites))
        values = jax.lax.map(
            lambda chunk: rematerialized(candidate, chunk), chunks
        )
        return values.reshape((array.shape[0],))

    return wrapped


def _checkpoint_metadata(
    states: jax.Array,
    key: jax.Array,
    audits: list[dict[str, object]],
    initial: dict[str, object],
    work: dict[str, object],
    resume_count: int,
    *,
    completed_steps: int,
) -> dict[str, object]:
    metadata = energy_smoke._checkpoint_metadata(
        states, key, audits, initial, work, resume_count
    )
    metadata.update(
        {
            "optimizer": "matrix_free_sampled_physical_sr",
            "optimizer_completed_steps": completed_steps,
            "samples_per_update": 64,
            "amplitude_cache_present": False,
        }
    )
    return metadata


def _optimizer_state(completed_steps: int) -> dict[str, object]:
    return {
        "name": "matrix_free_sampled_physical_sr",
        "completed_steps": int(completed_steps),
        "has_momentum_state": False,
    }


def _audit_at(
    update: int,
    params: object,
    states: jax.Array,
    *,
    case: dict[str, object],
    bonds,
    spec: dict[str, object],
    uncached,
    cached,
    hpsi,
) -> tuple[dict[str, object], dict[str, object]]:
    audit_spec = dict(spec)
    audit_spec["audit"] = {**spec["audit"], "current_update": int(update)}
    return energy_smoke._audit(
        params,
        np.asarray(states),
        case=case,
        bonds=bonds,
        spec=audit_spec,
        uncached=uncached,
        cached=cached,
        hpsi=hpsi,
    )


def _run_case(
    case: dict[str, object],
    spec: dict[str, object],
    output: Path,
    *,
    overwrite: bool,
    source_manifest_sha256: str,
    runtime_identity_sha256: str,
) -> dict[str, object]:
    base_runtime = energy_smoke._runtime(case, spec)
    runtime = _initialize_params(base_runtime, case, spec)
    bonds, group = energy_smoke._geometry(case)
    uncached, cached, centers, hpsi = energy_smoke._kernels(runtime, bonds)
    sr_centers = _rematerialized_center_microbatch_fn(
        centers,
        microbatch_size=int(spec["optimizer"]["qgt_center_microbatch_size"]),
        n_sites=36,
    )
    initial_key = jax.random.PRNGKey(int(case["sampler_seed"]))
    (
        initial_states,
        random_control_states,
        physics_probe_key,
        random_probe_key,
        sampling_key,
    ) = _initial_chain_ensembles(
        initial_key,
        runtime,
        case,
        spec,
    )
    main_branch_params = _cluster_branch_params(runtime, case, spec)
    physics_diagnostic = _initial_chain_diagnostic(
        runtime.params,
        main_branch_params,
        initial_states,
        physics_probe_key,
        chain_kind=str(spec["sampler"]["initial_chain_kind"]),
        spec=spec,
        centers=centers,
        uncached=uncached,
    )
    random_diagnostic = _initial_chain_diagnostic(
        runtime.params,
        main_branch_params,
        random_control_states,
        random_probe_key,
        chain_kind=str(spec["sampler"]["random_control_kind"]),
        spec=spec,
        centers=centers,
        uncached=uncached,
    )
    if (
        physics_diagnostic["mixed_projected_support"]["exact_zero_count"] != 0
        or physics_diagnostic["cluster_product_main_branch_support"][
            "exact_zero_count"
        ]
        != 0
    ):
        raise FloatingPointError(
            "physics-seeded chains must start on the projected cluster branch"
        )
    initial = {
        "kind": spec["initialization"]["kind"],
        "pre_projection_background_coefficient": float(
            spec["initialization"]["pre_projection_background_coefficient"]
        ),
        "projected_background_norm_ratio_not_assumed": True,
        "params_sha256": pytree_sha256(runtime.params),
        "cluster_product_main_branch_params_sha256": pytree_sha256(
            main_branch_params
        ),
        "training_chains": physics_diagnostic,
        "uniform_fixed_sz0_control": random_diagnostic,
        "random_control_enters_training": False,
        "either_initial_ensemble_claimed_equilibrated": False,
        "equilibration_promotion_requirement": (
            "independent physics-seeded and uniform-fixed-Sz0 trajectories must "
            "agree in log-amplitude and physical observables within their "
            "combined Monte Carlo uncertainty"
        ),
    }
    identity = {
        "task": TASK,
        "config_sha256": canonical_sha256(spec),
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "case_id": case["case_id"],
        "bonds_sha256": canonical_sha256(bonds),
        "group_sha256": canonical_sha256(group),
        "initial": initial,
        "force": "exact_singlet_3hz_center_only_v1",
        "optimizer": dict(spec["optimizer"]),
        "samples_per_update": 64,
    }
    checkpoint_dir = output / "checkpoints" / str(case["case_id"])
    if overwrite and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint = None
    if not overwrite and checkpoint_dir.exists():
        checkpoint = load_latest_resumable_checkpoint(
            checkpoint_dir, expected_run_identity=identity
        )

    started = time.perf_counter()
    rss_before = _peak_rss_bytes()
    if checkpoint is None:
        params = runtime.params
        states, amplitudes, sampling_key, burn = uncached(
            params,
            sampling_key,
            initial_states,
            int(spec["sampler"]["initial_burn_proposals_per_chain"]),
            float(spec["sampler"]["bond_proposal_probability"]),
        )
        jax.block_until_ready((states, amplitudes, sampling_key, burn))
        if float(burn["zero_amplitude_chain_fraction"]) != 0.0:
            raise FloatingPointError("post-burn projected chain has a zero amplitude")
        del amplitudes
        audit_started = time.perf_counter()
        audit0, audit_work = _audit_at(
            0,
            params,
            states,
            case=case,
            bonds=bonds,
            spec=spec,
            uncached=uncached,
            cached=cached,
            hpsi=hpsi,
        )
        audit0["wall_seconds"] = time.perf_counter() - audit_started
        audits = [audit0]
        history: list[dict[str, object]] = []
        completed = 0
        resume_count = 0
        work: dict[str, object] = {
            "initial_chain_probe_sampler_proposals": 2
            * int(spec["sampler"]["chains"])
            * int(spec["sampler"]["initial_chain_probe_proposals_per_chain"]),
            "initial_burn_sampler_proposals": int(spec["sampler"]["chains"])
            * int(spec["sampler"]["initial_burn_proposals_per_chain"]),
            "initial_burn_sampler_diagnostics": energy_smoke._float_row(burn),
            **audit_work,
        }
        save_resumable_checkpoint_generation(
            checkpoint_dir,
            params,
            optimizer_state=_optimizer_state(0),
            completed_steps=0,
            next_sample_index=0,
            run_identity=identity,
            history=history,
            metadata=_checkpoint_metadata(
                states,
                sampling_key,
                audits,
                initial,
                work,
                resume_count,
                completed_steps=0,
            ),
        )
    else:
        params = checkpoint["params"]
        completed = int(checkpoint["completed_steps"])
        state = checkpoint["optimizer_state"]
        if state != _optimizer_state(completed):
            raise RuntimeError("SR checkpoint optimizer state is inconsistent")
        history = list(checkpoint["history"])
        metadata = checkpoint["metadata"]
        if metadata.get("optimizer_completed_steps") != completed:
            raise RuntimeError("SR checkpoint metadata step is inconsistent")
        states_host = np.asarray(metadata["chain_states"], dtype=np.int32)
        key_host = np.asarray(metadata["sampling_key"], dtype=np.uint32)
        if (
            array_sha256(states_host) != metadata.get("chain_states_sha256")
            or array_sha256(key_host) != metadata.get("sampling_key_sha256")
            or states_host.shape != (16, 36)
            or np.any(np.sum(states_host, axis=1) != 18)
        ):
            raise RuntimeError("SR checkpoint chain/key integrity failed")
        states = jnp.asarray(states_host)
        sampling_key = jnp.asarray(key_host)
        audits = list(metadata["audits"])
        initial = dict(metadata["initial"])
        work = dict(metadata["work"])
        resume_count = int(metadata["resume_count"]) + 1

    hard_failure = None
    requested_updates = int(spec["optimizer"]["updates"])
    for update in range(completed + 1, requested_updates + 1):
        params_before = params
        states_before = np.asarray(states).copy()
        key_before = np.asarray(sampling_key).copy()
        completed_before = completed
        history_before = list(history)
        audits_before = list(audits)
        work_before = dict(work)
        try:
            sampling_started = time.perf_counter()
            states, sampling_key, samples, sampler_rows = _collect_training_samples(
                params,
                states,
                sampling_key,
                spec=spec,
                uncached=uncached,
                cached=cached,
            )
            sampling_seconds = time.perf_counter() - sampling_started
            diagonal = np.asarray(three_hz_local_energies(samples, bonds)).reshape(
                int(spec["sampler"]["snapshots_per_update"]),
                int(spec["sampler"]["chains"]),
            )
            ess = _chain_block_ess(diagonal)

            microbatch = int(spec["force"]["microbatch_size"])
            batches = tuple(
                jnp.asarray(samples[start : start + microbatch])
                for start in range(0, len(samples), microbatch)
            )
            force_started = time.perf_counter()
            force = sampled_singlet_diagonal_value_and_grad(
                params, batches, center_fn=centers, bonds=bonds
            )
            force_seconds = time.perf_counter() - force_started

            sr_started = time.perf_counter()
            sr = sampled_physical_sr_direction(
                params,
                jnp.asarray(samples),
                energy_gradient=force.gradient,
                center_fn=sr_centers,
                damping=float(spec["optimizer"]["damping"]),
                cg_tolerance=float(spec["optimizer"]["cg_tolerance"]),
                cg_max_iterations=int(spec["optimizer"]["cg_max_iterations"]),
                step_size=float(spec["optimizer"]["step_size"]),
                maximum_relative_parameter_step=float(
                    spec["optimizer"]["maximum_relative_parameter_step"]
                ),
                fisher_trust_radius=float(
                    spec["optimizer"]["fisher_trust_radius"]
                ),
            )
            candidate = jax.tree_util.tree_map(
                lambda value, direction: value - direction,
                params,
                sr.direction,
            )
            jax.block_until_ready(candidate)
            if not all(
                np.all(np.isfinite(np.asarray(leaf)))
                for leaf in jax.tree_util.tree_leaves(candidate)
            ):
                raise FloatingPointError("SR candidate parameters are non-finite")
            sr_seconds = time.perf_counter() - sr_started
            params = candidate

            audit_row = None
            audit_work: dict[str, object] = {}
            audit_seconds = 0.0
            if update in spec["audit"]["updates"]:
                audit_started = time.perf_counter()
                audit_row, audit_work = _audit_at(
                    update,
                    params,
                    states,
                    case=case,
                    bonds=bonds,
                    spec=spec,
                    uncached=uncached,
                    cached=cached,
                    hpsi=hpsi,
                )
                audit_seconds = time.perf_counter() - audit_started
                audit_row["wall_seconds"] = audit_seconds

            sr_diagnostics = asdict(sr.diagnostics)
            proposals = (
                int(spec["sampler"]["chains"])
                * int(spec["sampler"]["snapshots_per_update"])
                * int(spec["sampler"]["proposals_between_snapshots"])
            )
            step_work = {
                "sampler_proposals": proposals,
                "physical_samples": len(samples),
                "three_hz_projected_center_evaluations_proxy": 2
                * len(samples),
                "sr_qgt_matvecs": int(sr.diagnostics.qgt_matvec_count),
                "sr_base_translation_orbit_images_proxy": 36
                * len(samples)
                * (1 + 2 * sr.diagnostics.qgt_matvec_count),
                **{
                    name: int(value)
                    for name, value in audit_work.items()
                },
            }
            row = {
                "update": update,
                "params_before_sha256": pytree_sha256(params_before),
                "params_after_sha256": pytree_sha256(params),
                "sample_states_sha256": array_sha256(samples),
                "sample_count": len(samples),
                "sampler": _mean_sampler_diagnostics(sampler_rows),
                "three_hz_chain_block_ess": ess,
                "three_hz_energy_per_site_training_estimator": force.energy / 36,
                "three_hz_sample_variance_per_site_estimator_diagnostic_only": (
                    force.variance / 36
                ),
                "three_hz_variance_is_not_physical_h_variance": True,
                "energy_gradient_norm": force.gradient_norm,
                "energy_gradient_sha256": pytree_sha256(force.gradient),
                "minimum_center_magnitude": force.minimum_center_magnitude,
                "sr": sr_diagnostics,
                "sr_qgt_center_execution": {
                    "microbatch_size": int(
                        spec["optimizer"]["qgt_center_microbatch_size"]
                    ),
                    "rematerialized": bool(
                        spec["optimizer"]["qgt_rematerialize_center"]
                    ),
                    "materialized_score_matrix": False,
                    "materialized_parameter_square_qgt": False,
                },
                "sr_raw_direction_sha256": pytree_sha256(sr.raw_direction),
                "sr_applied_direction_sha256": pytree_sha256(sr.direction),
                "sr_true_residual_sha256": pytree_sha256(sr.linear_residual),
                "full_h_audit": audit_row,
                "work": step_work,
                "timing": {
                    "sampling_seconds": sampling_seconds,
                    "three_hz_force_seconds": force_seconds,
                    "sr_solve_seconds": sr_seconds,
                    "full_h_audit_seconds": audit_seconds,
                },
                "peak_rss_bytes_process_high_water": _peak_rss_bytes(),
            }
            history.append(row)
            completed = update
            if audit_row is not None:
                audits.append(audit_row)
            work["training_sampler_proposals"] = int(
                work.get("training_sampler_proposals", 0)
            ) + proposals
            work["training_physical_samples"] = int(
                work.get("training_physical_samples", 0)
            ) + len(samples)
            work["three_hz_projected_center_evaluations_proxy"] = int(
                work.get("three_hz_projected_center_evaluations_proxy", 0)
            ) + 2 * len(samples)
            work["sr_qgt_matvecs"] = int(work.get("sr_qgt_matvecs", 0)) + int(
                sr.diagnostics.qgt_matvec_count
            )
            work["sr_base_translation_orbit_images_proxy"] = int(
                work.get("sr_base_translation_orbit_images_proxy", 0)
            ) + 36 * len(samples) * (1 + 2 * sr.diagnostics.qgt_matvec_count)
            for name, value in audit_work.items():
                work[name] = int(work.get(name, 0)) + int(value)

            save_resumable_checkpoint_generation(
                checkpoint_dir,
                params,
                optimizer_state=_optimizer_state(completed),
                completed_steps=completed,
                next_sample_index=completed * 64,
                run_identity=identity,
                history=history,
                metadata=_checkpoint_metadata(
                    states,
                    sampling_key,
                    audits,
                    initial,
                    work,
                    resume_count,
                    completed_steps=completed,
                ),
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            params = params_before
            states = jnp.asarray(states_before)
            sampling_key = jnp.asarray(key_before)
            completed = completed_before
            history = history_before
            audits = audits_before
            work = work_before
            hard_failure = {
                "attempted_update": update,
                "type": type(error).__name__,
                "message": str(error),
            }
            save_resumable_checkpoint_generation(
                checkpoint_dir,
                params,
                optimizer_state=_optimizer_state(completed),
                completed_steps=completed,
                next_sample_index=completed * 64,
                run_identity=identity,
                history=history,
                metadata=_checkpoint_metadata(
                    states,
                    sampling_key,
                    audits,
                    initial,
                    work,
                    resume_count,
                    completed_steps=completed,
                ),
            )
            break

    final_audit = audits[-1]
    return {
        "case_id": case["case_id"],
        "geometry": case["geometry"],
        "status": (
            "complete"
            if completed == requested_updates and hard_failure is None
            else "hard_failed"
        ),
        "completed_updates": completed,
        "requested_updates": requested_updates,
        "parameter_count": 1252,
        "initialization": dict(spec["initialization"]),
        "translation_group_order": runtime.projector.group_order,
        "translation_projection_applied_from_update_zero": True,
        "samples_per_update": 64,
        "resume_count": resume_count,
        "initial": initial,
        "history": history,
        "full_h_paired_audits": audits,
        "final_full_h_energy_per_site": final_audit["energy_per_site"],
        "final_gap_to_exact_energy_per_site": final_audit[
            "gap_to_exact_energy_per_site"
        ],
        "final_gap_to_raw_literature_energy_per_site": final_audit[
            "gap_to_raw_literature_energy_per_site"
        ],
        "work_ledger": work,
        "hard_failure": hard_failure,
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "peak_rss_bytes_before_case": rss_before,
        "peak_rss_bytes_after_case": _peak_rss_bytes(),
        "checkpoint_directory": str(checkpoint_dir.relative_to(ROOT)),
    }


def run_triangle_patch_n36_sr_pilot(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run or deterministically resume both registered N36 SR cases."""

    # The CLI accepts relative paths, whereas checkpoint provenance is
    # recorded relative to the absolute repository root.  Resolve both paths
    # once at the public boundary so final path formatting cannot invalidate
    # an otherwise completed calculation.
    config_path = config_path.resolve()
    output = output.resolve()
    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result exists; pass --overwrite: {result_path}")
    started = time.perf_counter()
    source_manifest = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    source_manifest_sha256 = canonical_sha256(source_manifest)
    runtime_identity = numerical_runtime_identity()
    runtime_identity_sha256 = canonical_sha256(runtime_identity)
    cases = [
        _run_case(
            case,
            spec,
            output,
            overwrite=overwrite,
            source_manifest_sha256=source_manifest_sha256,
            runtime_identity_sha256=runtime_identity_sha256,
        )
        for case in spec["cases"]
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": (
            "complete"
            if all(case["status"] == "complete" for case in cases)
            else "hard_failed"
        ),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_identity_sha256,
        "cases": cases,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes_process_high_water": _peak_rss_bytes(),
        "hamiltonian_diagonalization_performed": False,
        "translation_projection_applied_from_update_zero": True,
        "training_force": "exact_singlet_3hz_center_only",
        "optimizer": "matrix_free_sampled_physical_sr",
        "physical_energy_and_variance_source": (
            "independent_full_h_paired_audits_only"
        ),
        "uniform_fixed_sz0_initial_control_is_diagnostic_only": True,
        "either_initial_ensemble_claimed_equilibrated": False,
        "equilibration_promotion_requirement": (
            "independent physics-seeded and uniform-fixed-Sz0 trajectories must "
            "agree in log-amplitude and physical observables within their "
            "combined Monte Carlo uncertainty"
        ),
        "promotion_decision": "local_optimization_and_memory_gate_only",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = [
    "CLAIM_BOUNDARY",
    "TASK",
    "run_triangle_patch_n36_sr_pilot",
]
