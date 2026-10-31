"""Reproducible zero-update readout for the N=36 physical cluster seed.

This workflow intentionally performs no optimization.  It measures projected
support and the exact-singlet diagonal ``3 Hz`` energy estimator after a
frozen MCMC protocol.  Its sampled variance is not a physical-Hamiltonian
variance and is therefore neither computed nor reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.artifacts import atomic_json, canonical_sha256
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.frontier.triangle_patch_physical_init import (
    init_triangle_patch_cluster_singlet,
    sample_triangle_patch_cluster_product_chains,
    triangle_patch_cluster_reference,
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
    spatial_projected_amplitudes,
)
from su2_nqs.models.triangle_patch_clebsch_tree import (
    triangle_patch_6x6_canary_plan,
    triangle_patch_amplitude,
    triangle_patch_parameter_count,
)
from su2_nqs.vmc.physical_exchange import run_exchange_sweeps
from su2_nqs.vmc.singlet_diagonal_force import three_hz_local_energies


jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[3]
TASK = "triangle_patch_n36_physical_init_readout"
CLAIM_BOUNDARY = (
    "Zero-update single-seed projected 3Hz initialization readout only; no "
    "optimized or converged energy, physical Hamiltonian variance, efficiency, "
    "phase, QSL, scaling, topology, or PRL advantage is claimed."
)


@dataclass(frozen=True)
class _Runtime:
    plan: object
    params: object
    projector: PhysicalSpatialProjector
    amplitude: Callable[[object, jax.Array], jax.Array]
    bonds: tuple[tuple[int, int, float], ...]


def _validate(spec: dict[str, object]) -> None:
    if (
        spec.get("schema_version") != 1
        or spec.get("domain") != "spin"
        or spec.get("task") != TASK
        or spec.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("invalid physical-initialization readout identity")
    if spec.get("model") != {
        "family": "strict_triangle_patch_clebsch_tree",
        "feature_dim": 2,
        "maximum_intermediate_q": 10,
        "parameter_count": 1252,
        "route_gate_scale": 0.05,
    }:
        raise ValueError("model must be the registered D2/qmax10/P1252 tree")
    if spec.get("initialization") != {
        "kind": "finite_torus_short_range_rvb_cluster_plus_orthogonal_support",
        "pre_projection_background_coefficient": 0.05,
        "projected_branch_norm_ratio_not_assumed": True,
        "not_a_topological_or_ipeps_ansatz_claim": True,
    }:
        raise ValueError("physical initialization contract changed")
    if spec.get("projection") != {
        "kind": "translation_k0",
        "group_order": 36,
        "orbit_chunk_size": 6,
        "rematerialize_orbit": True,
    }:
        raise ValueError("readout requires the exact 36-translation projector")
    if spec.get("support_probe") != {
        "seed": 17017,
        "uniform_fixed_m0_count": 2048,
        "relative_near_zero_threshold": 1.0e-14,
        "arms": [0.0, 0.05],
    }:
        raise ValueError("support-probe identity changed")
    if spec.get("sampler") != {
        "chains": 32,
        "initial_chain_kind": "cluster_product_born_random_translation",
        "burn_proposals_per_chain": 500,
        "snapshots": 20,
        "proposals_between_snapshots": 100,
        "bond_proposal_probability": 0.8,
    }:
        raise ValueError("frozen MCMC readout protocol changed")
    if spec.get("estimator") != {
        "kind": "exact_singlet_3hz_diagonal",
        "energy_units": "per_site",
        "mcse": (
            "standard_error_across_32_chain_time_means_each_averaging_20_"
            "autocorrelated_snapshots"
        ),
        "sample_variance_is_physical_h_variance": False,
        "full_h_audit_performed": False,
    }:
        raise ValueError("3Hz estimator or uncertainty convention changed")
    expected = {
        "square_6x6_j2_0p5": ("square", 0.5, 1001, 1002, 1003),
        "triangular_6x6_j2_0p125": (
            "triangular",
            0.125,
            1002,
            1003,
            1004,
        ),
    }
    cases = spec.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError("exactly two registered cases are required")
    for case in cases:
        if not isinstance(case, dict) or case.get("case_id") not in expected:
            raise ValueError("unregistered readout case")
        geometry, ratio, model_seed, chain_seed, markov_seed = expected[case["case_id"]]
        if (
            case.get("geometry"),
            case.get("rows"),
            case.get("columns"),
            case.get("periodic"),
            float(case.get("j2_over_j1", np.nan)),
            case.get("model_seed"),
            case.get("chain_seed"),
            case.get("markov_seed"),
        ) != (geometry, 6, 6, True, ratio, model_seed, chain_seed, markov_seed):
            raise ValueError("case geometry, coupling, or seed changed")


def _runtime(case: dict[str, object], spec: dict[str, object]) -> _Runtime:
    geometry = str(case["geometry"])
    ratio = float(case["j2_over_j1"])
    plan = triangle_patch_6x6_canary_plan(geometry)
    params = init_triangle_patch_cluster_singlet(
        jax.random.PRNGKey(int(case["model_seed"])),
        plan,
        j2_over_j1=ratio,
        background_scale=float(
            spec["initialization"]["pre_projection_background_coefficient"]
        ),
        route_gate_scale=float(spec["model"]["route_gate_scale"]),
    )
    if triangle_patch_parameter_count(params) != 1252:
        raise RuntimeError("physical initializer changed the parameter count")
    if geometry == "square":
        group = rectangular_translation_group(6, 6)
        bonds = square_j1j2_bonds(6, 6, j1=1.0, j2=ratio, periodic=True)
    else:
        group = triangular_hnf_translation_group(((6, 0), (0, 6)))
        bonds = triangular_j1j2_bonds(6, 6, j1=1.0, j2=ratio, periodic=True)
    projector = PhysicalSpatialProjector(
        group, orbit_chunk_size=6, rematerialize_orbit=True
    )

    def amplitude(candidate, state):
        return spatial_projected_amplitude(
            candidate,
            state,
            plan,
            projector,
            amplitude_fn=triangle_patch_amplitude,
        )

    return _Runtime(plan, params, projector, amplitude, tuple(bonds))


def _uniform_fixed_m0(seed: int, count: int) -> jax.Array:
    rng = np.random.default_rng(seed)
    configurations = []
    for _ in range(count):
        row = np.zeros(36, dtype=np.int32)
        row[rng.choice(36, size=18, replace=False)] = 1
        configurations.append(row)
    return jnp.asarray(configurations)


def _support_summary(values: np.ndarray, threshold: float) -> dict[str, object]:
    magnitudes = np.abs(np.asarray(values))
    rms = float(np.sqrt(np.mean(np.square(magnitudes))))
    return {
        "sample_count": int(magnitudes.size),
        "exact_zero_count": int(np.count_nonzero(magnitudes == 0.0)),
        "below_relative_rms_threshold_count": int(
            np.count_nonzero(magnitudes < threshold * rms)
        ),
        "minimum_magnitude": float(np.min(magnitudes)),
        "p01_magnitude": float(np.quantile(magnitudes, 0.01)),
        "median_magnitude": float(np.median(magnitudes)),
        "rms_magnitude": rms,
        "maximum_magnitude": float(np.max(magnitudes)),
    }


def _support_probe(
    runtime: _Runtime,
    case: dict[str, object],
    spec: dict[str, object],
    states: jax.Array,
) -> list[dict[str, object]]:
    rows = []
    for coefficient in spec["support_probe"]["arms"]:
        params = init_triangle_patch_cluster_singlet(
            jax.random.PRNGKey(int(case["model_seed"])),
            runtime.plan,
            j2_over_j1=float(case["j2_over_j1"]),
            background_scale=float(coefficient),
            route_gate_scale=float(spec["model"]["route_gate_scale"]),
        )
        kernel = jax.jit(
            lambda candidate, batch: spatial_projected_amplitudes(
                candidate,
                batch,
                runtime.plan,
                runtime.projector,
                amplitude_fn=triangle_patch_amplitude,
            )
        )
        chunks = []
        for start in range(0, len(states), 128):
            chunks.append(np.asarray(kernel(params, states[start : start + 128])))
        summary = _support_summary(
            np.concatenate(chunks),
            float(spec["support_probe"]["relative_near_zero_threshold"]),
        )
        rows.append(
            {"pre_projection_background_coefficient": float(coefficient), **summary}
        )
    return rows


def _run_case(
    case: dict[str, object],
    spec: dict[str, object],
    support_states: jax.Array,
) -> dict[str, object]:
    runtime = _runtime(case, spec)
    sampler = spec["sampler"]
    chains = sample_triangle_patch_cluster_product_chains(
        jax.random.PRNGKey(int(case["chain_seed"])),
        runtime.plan,
        j2_over_j1=float(case["j2_over_j1"]),
        n_chains=int(sampler["chains"]),
        translate=True,
    )
    key = jax.random.PRNGKey(int(case["markov_seed"]))
    edges = tuple((first, second) for first, second, _ in runtime.bonds)
    probability = float(sampler["bond_proposal_probability"])
    burn_attempts = int(sampler["burn_proposals_per_chain"])
    between = int(sampler["proposals_between_snapshots"])

    def uncached(candidate, random_key, states):
        return run_exchange_sweeps(
            candidate,
            random_key,
            states,
            amplitude_fn=runtime.amplitude,
            particle_kind="spin",
            sweeps=burn_attempts,
            amplitudes=None,
            exchange_edges=edges,
            bond_proposal_probability=probability,
        )

    def cached(candidate, random_key, states, amplitudes):
        return run_exchange_sweeps(
            candidate,
            random_key,
            states,
            amplitude_fn=runtime.amplitude,
            particle_kind="spin",
            sweeps=between,
            amplitudes=amplitudes,
            exchange_edges=edges,
            bond_proposal_probability=probability,
        )

    burn_kernel = jax.jit(uncached)
    step_kernel = jax.jit(cached)
    chains, amplitudes, key, burn = burn_kernel(runtime.params, key, chains)
    jax.block_until_ready((chains, amplitudes))
    energy_rows = []
    diagnostics = []
    for _ in range(int(sampler["snapshots"])):
        chains, amplitudes, key, row = step_kernel(
            runtime.params, key, chains, amplitudes
        )
        jax.block_until_ready((chains, amplitudes))
        energy_rows.append(
            np.asarray(three_hz_local_energies(chains, runtime.bonds))
        )
        diagnostics.append({name: float(value) for name, value in row.items()})
    energies = np.stack(energy_rows)
    chain_means = np.mean(energies, axis=0) / 36.0
    reference = triangle_patch_cluster_reference(
        str(case["geometry"]), float(case["j2_over_j1"])
    )
    return {
        "case_id": case["case_id"],
        "geometry": case["geometry"],
        "parameter_count": triangle_patch_parameter_count(runtime.params),
        "pre_projection_unmixed_cluster_energy_per_site": reference.energy / 6.0,
        "support_probe_arms": _support_probe(runtime, case, spec, support_states),
        "three_hz_energy_per_site": float(np.mean(energies) / 36.0),
        "three_hz_energy_per_site_chain_mcse": float(
            np.std(chain_means, ddof=1) / np.sqrt(len(chain_means))
        ),
        "three_hz_sample_count": int(energies.size),
        "chains": int(energies.shape[1]),
        "snapshots_per_chain": int(energies.shape[0]),
        "samples_are_temporally_autocorrelated": True,
        "mcse_is_across_chain_time_averages": True,
        "three_hz_sample_variance_is_not_physical_h_variance": True,
        "full_h_audit_performed": False,
        "snapshot_energy_per_site_first_five": [
            float(value) for value in np.mean(energies, axis=1)[:5] / 36.0
        ],
        "snapshot_energy_per_site_last_five": [
            float(value) for value in np.mean(energies, axis=1)[-5:] / 36.0
        ],
        "burn_sampler": {name: float(value) for name, value in burn.items()},
        "post_burn_accepted_move_fraction_mean": float(
            np.mean([row["accepted_move_fraction"] for row in diagnostics])
        ),
        "post_burn_conditional_move_acceptance_mean": float(
            np.mean([row["conditional_move_acceptance"] for row in diagnostics])
        ),
        "post_burn_zero_amplitude_chain_fraction_maximum": float(
            max(row["zero_amplitude_chain_fraction"] for row in diagnostics)
        ),
        "final_minimum_center_magnitude": float(
            np.min(np.abs(np.asarray(amplitudes)))
        ),
    }


def run_triangle_patch_n36_physical_init_readout(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run and seal the frozen square/triangular zero-update readout."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result exists; pass --overwrite: {result_path}")
    manifest = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    support_states = _uniform_fixed_m0(
        int(spec["support_probe"]["seed"]),
        int(spec["support_probe"]["uniform_fixed_m0_count"]),
    )
    started = time.perf_counter()
    cases = [_run_case(case, spec, support_states) for case in spec["cases"]]
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_manifest": manifest,
        "source_manifest_sha256": canonical_sha256(manifest),
        "runtime_identity": numerical_runtime_identity(),
        "cases": cases,
        "total_wall_seconds": time.perf_counter() - started,
        "optimization_performed": False,
        "hamiltonian_diagonalization_performed_for_n36": False,
        "translation_projection_applied": True,
        "energy_estimator": "exact_singlet_3hz_diagonal",
        "physical_hamiltonian_variance_reported": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = [
    "CLAIM_BOUNDARY",
    "TASK",
    "run_triangle_patch_n36_physical_init_readout",
]
