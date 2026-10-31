"""Cross-basin sampling gate for the triangular N=36 SU(2) Route-ViT."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_sampled_runtime import (
    _autocorrelation_summary,
    _ordering_structure_factor_series,
)
from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_n36_audit import (
    _initialized_runtime,
)
from projects.clebsch_tree_nqs.workflows.spin_su2_route_vit_n36_smoke import (
    _dimer_chains,
)
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
from su2_nqs.lattices.triangular_e2 import E2_ORIENTATION_FRAME
from su2_nqs.lattices.triangular_hnf import J1_POSITIVE
from su2_nqs.models.su2_route_vit import su2_route_vit_parameter_count
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.resumable_checkpoint import (
    load_latest_resumable_checkpoint,
    save_resumable_checkpoint_generation,
)
from su2_nqs.vmc.singlet_diagonal_force import three_hz_local_energies
from su2_nqs.vmc.spin_physical import padded_spin_connectivity, spin_physical_hpsi
from su2_nqs.vmc.spin_symmetry_mixture import (
    d6_complete_dimer_matchings,
    d6_point_group_images,
    run_spin_symmetry_mixture_sweeps,
)


jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[3]
TASK = "exact_su2_route_vit_n36_basin_gate"
STRATA = ("row_dimer", "column_dimer", "diagonal_dimer", "uniform_sz0")


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid N36 basin-gate identity")
    case, model, sampler, gates = (
        spec.get(name) for name in ("case", "model", "sampler", "gates")
    )
    if not all(isinstance(value, dict) for value in (case, model, sampler, gates)):
        raise TypeError("case, model, sampler, and gates must be objects")
    if (case.get("geometry"), case.get("rows"), case.get("columns")) != (
        "triangular",
        6,
        6,
    ):
        raise ValueError("the cross-basin gate is registered for triangular 6x6")
    if int(sampler.get("chains_per_stratum", 0)) < 2:
        raise ValueError("chains_per_stratum must be at least two")
    for field in ("burn_sweeps", "snapshots", "sweeps_between", "full_h_snapshots"):
        if int(sampler.get(field, 0)) < 1:
            raise ValueError(f"sampler.{field} must be positive")
    weights = tuple(float(value) for value in sampler.get("kernel_weights", ()))
    if len(weights) != 3 or any(value < 0.0 for value in weights) or abs(sum(weights) - 1.0) > 1.0e-12:
        raise ValueError("kernel_weights must be three probabilities summing to one")
    if not 0.0 < float(sampler.get("uniform_floor", 0.0)) <= 1.0:
        raise ValueError("uniform_floor must lie in (0,1]")


def _initial_chains(spec: dict[str, object], config) -> tuple[jax.Array, np.ndarray]:
    count = int(spec["sampler"]["chains_per_stratum"])
    names = config.spatial.route_names
    orders = config.spatial.route_orders
    seed = int(spec["seed"])
    rows = []
    for offset, route_name in enumerate(
        ("row_dimers", "column_dimers", "diagonal_dimers"), start=1
    ):
        rows.append(
            _dimer_chains(
                jax.random.PRNGKey(seed + 100 * offset),
                chains=count,
                order=orders[names.index(route_name)],
            )
        )
    rows.append(
        initialize_spin_exchange_chains(
            jax.random.PRNGKey(seed + 400),
            n_chains=count,
            n_sites=36,
            n_up=18,
        )
    )
    labels = np.repeat(np.arange(len(STRATA), dtype=np.int32), count)
    return jnp.concatenate(rows, axis=0), labels


def _nematic_diagonal_series(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    concrete = np.asarray(states, dtype=np.float64)
    spins = concrete - 0.5
    orientation_sums = []
    for dx, dy in J1_POSITIVE:
        values = np.zeros(concrete.shape[0], dtype=np.float64)
        for y in range(6):
            for x in range(6):
                first = y * 6 + x
                second = ((y + dy) % 6) * 6 + ((x + dx) % 6)
                values += spins[:, first] * spins[:, second]
        orientation_sums.append(values)
    doublet = E2_ORIENTATION_FRAME @ np.asarray(orientation_sums)
    doublet /= np.sqrt(36.0)
    return np.asarray(doublet[0]), np.asarray(doublet[1])


def _snapshot_observables(
    states: np.ndarray,
    amplitudes: np.ndarray,
    bonds,
) -> dict[str, np.ndarray]:
    q1, q2 = _nematic_diagonal_series(states)
    ordering = _ordering_structure_factor_series(
        states, geometry="triangular", rows=6, columns=6
    )
    magnitude = np.abs(np.asarray(amplitudes, dtype=np.complex128))
    return {
        "diagonal_3hz_energy": np.asarray(
            three_hz_local_energies(states, bonds), dtype=np.float64
        ),
        "log_born_weight": 2.0 * np.log(np.maximum(magnitude, np.finfo(float).tiny)),
        "nematic_q1": q1,
        "nematic_q2": q2,
        "nematic_q_squared": q1**2 + q2**2,
        **ordering,
    }


def _full_h_values(params, states, config, bonds, amplitude) -> tuple[np.ndarray, float]:
    selected = np.asarray(states, dtype=np.int32)
    connected, elements, valid, _widths = padded_spin_connectivity(selected, bonds)
    values = []
    started = time.perf_counter()
    for index in range(len(selected)):
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
        values.append(complex(np.asarray(action / center)[0]))
    return np.asarray(values, dtype=np.complex128), time.perf_counter() - started


def _checkpoint_metadata(
    *,
    states,
    amplitudes,
    master_key,
    proposal_index: int,
    labels,
    params_hash: str,
    images_hash: str,
    matchings_hash: str,
    spec: dict[str, object],
) -> dict[str, object]:
    return {
        "chain_states": np.asarray(states, dtype=np.int32),
        "cached_amplitudes": np.asarray(amplitudes, dtype=np.complex128),
        "master_sampling_key": np.asarray(master_key, dtype=np.uint32),
        "next_global_proposal_index": int(proposal_index),
        "chain_stratum_labels": np.asarray(labels, dtype=np.int32),
        "params_sha256": params_hash,
        "d6_images_sha256": images_hash,
        "dimer_matchings_sha256": matchings_hash,
        "kernel_weights": list(spec["sampler"]["kernel_weights"]),
        "uniform_floor": float(spec["sampler"]["uniform_floor"]),
    }


def _summary_with_fraction(values: np.ndarray) -> dict[str, object]:
    summary = _autocorrelation_summary(values)
    summary["ess_fraction"] = float(summary["ess"] / values.size)
    return summary


def run_spin_su2_route_vit_n36_basin_gate(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    """Run the fixed-parameter, no-ED N36 cross-basin sampling audit."""

    spec = json.loads(config_path.read_text())
    _validate(spec)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"result already exists; pass --overwrite: {result_path}")

    config, bonds, ledger, params, amplitude, forward = _initialized_runtime(spec)
    chains, labels = _initial_chains(spec, config)
    amplitudes = forward(params, chains)
    jax.block_until_ready(amplitudes)
    if np.any(np.abs(np.asarray(amplitudes)) == 0.0):
        raise RuntimeError("initial cross-basin chains contain zero-amplitude states")

    images = np.asarray(d6_point_group_images(((6, 0), (0, 6))), dtype=np.int32)
    matchings = np.asarray(d6_complete_dimer_matchings(6, 6), dtype=np.int32)
    exchange_edges = tuple(
        sorted({tuple(sorted((first, second))) for first, second, _weight in bonds})
    )
    sampler = spec["sampler"]
    master_key = jax.random.PRNGKey(int(spec["seed"]) + 500)
    proposal_index = 0
    advance_kwargs = {
        "amplitude_fn": amplitude,
        "exchange_edges": exchange_edges,
        "kernel_weights": tuple(sampler["kernel_weights"]),
        "d6_images": images,
        "dimer_matchings": matchings,
        "uniform_floor": float(sampler["uniform_floor"]),
        "bond_proposal_probability": float(sampler["bond_proposal_probability"]),
    }
    started = time.perf_counter()
    chains, amplitudes, proposal_index, burn_diagnostics = run_spin_symmetry_mixture_sweeps(
        params,
        master_key,
        proposal_index,
        chains,
        amplitudes=amplitudes,
        sweeps=int(sampler["burn_sweeps"]),
        **advance_kwargs,
    )
    jax.block_until_ready((chains, amplitudes))

    params_hash = pytree_sha256(params)
    images_hash = array_sha256(images)
    matchings_hash = array_sha256(matchings)
    run_identity = {
        "task": TASK,
        "config_sha256": canonical_sha256(spec),
        "params_sha256": params_hash,
        "d6_images_sha256": images_hash,
        "dimer_matchings_sha256": matchings_hash,
    }
    checkpoint_directory = output / "checkpoints"
    first_generation = save_resumable_checkpoint_generation(
        checkpoint_directory,
        params,
        optimizer_state={"kind": "fixed_parameter_sampling_gate"},
        completed_steps=0,
        next_sample_index=proposal_index,
        run_identity=run_identity,
        history=[],
        metadata=_checkpoint_metadata(
            states=chains,
            amplitudes=amplitudes,
            master_key=master_key,
            proposal_index=proposal_index,
            labels=labels,
            params_hash=params_hash,
            images_hash=images_hash,
            matchings_hash=matchings_hash,
            spec=spec,
        ),
    )
    restored = load_latest_resumable_checkpoint(
        checkpoint_directory, expected_run_identity=run_identity
    )
    metadata = restored["metadata"]
    replay_a = run_spin_symmetry_mixture_sweeps(
        params,
        master_key,
        proposal_index,
        chains,
        amplitudes=amplitudes,
        sweeps=int(sampler["sweeps_between"]),
        **advance_kwargs,
    )
    replay_b = run_spin_symmetry_mixture_sweeps(
        restored["params"],
        jnp.asarray(metadata["master_sampling_key"], dtype=jnp.uint32),
        int(metadata["next_global_proposal_index"]),
        jnp.asarray(metadata["chain_states"], dtype=jnp.int32),
        amplitudes=jnp.asarray(metadata["cached_amplitudes"], dtype=jnp.complex128),
        sweeps=int(sampler["sweeps_between"]),
        **advance_kwargs,
    )
    replay_state_equal = bool(np.array_equal(np.asarray(replay_a[0]), np.asarray(replay_b[0])))
    replay_amplitude_error = float(
        np.max(np.abs(np.asarray(replay_a[1]) - np.asarray(replay_b[1])))
    )
    replay_index_equal = bool(replay_a[2] == replay_b[2])

    chains, amplitudes, proposal_index, first_diagnostics = replay_a
    observable_rows = [
        _snapshot_observables(np.asarray(chains), np.asarray(amplitudes), bonds)
    ]
    sample_rows = [np.asarray(chains, dtype=np.int32)]
    diagnostics = [first_diagnostics]
    for _ in range(1, int(sampler["snapshots"])):
        chains, amplitudes, proposal_index, row = run_spin_symmetry_mixture_sweeps(
            params,
            master_key,
            proposal_index,
            chains,
            amplitudes=amplitudes,
            sweeps=int(sampler["sweeps_between"]),
            **advance_kwargs,
        )
        jax.block_until_ready((chains, amplitudes))
        observable_rows.append(
            _snapshot_observables(np.asarray(chains), np.asarray(amplitudes), bonds)
        )
        sample_rows.append(np.asarray(chains, dtype=np.int32))
        diagnostics.append(row)

    series = {
        name: np.stack([row[name] for row in observable_rows], axis=0)
        for name in observable_rows[0]
    }
    correlations = {name: _summary_with_fraction(values) for name, values in series.items()}
    by_stratum = {
        stratum: {
            name: _summary_with_fraction(values[:, labels == index])
            for name, values in series.items()
        }
        for index, stratum in enumerate(STRATA)
    }

    full_h_snapshot_count = int(sampler["full_h_snapshots"])
    full_states = np.concatenate(sample_rows[-full_h_snapshot_count:], axis=0)
    full_labels = np.tile(labels, full_h_snapshot_count)
    full_h, full_h_seconds = _full_h_values(
        params, full_states, config, bonds, amplitude
    )
    full_h_by_stratum: dict[str, dict[str, float]] = {}
    for index, stratum in enumerate(STRATA):
        values = full_h.real[full_labels == index]
        full_h_by_stratum[stratum] = {
            "sample_count": int(len(values)),
            "mean_total": float(np.mean(values)),
            "standard_error_total": float(np.std(values, ddof=1) / np.sqrt(len(values))),
            "variance_per_site": float(np.var(values, ddof=1) / 36.0),
        }
    pairwise = []
    for first in range(len(STRATA)):
        for second in range(first + 1, len(STRATA)):
            left = full_h_by_stratum[STRATA[first]]
            right = full_h_by_stratum[STRATA[second]]
            difference = abs(left["mean_total"] - right["mean_total"])
            combined = float(
                np.sqrt(left["standard_error_total"] ** 2 + right["standard_error_total"] ** 2)
            )
            pairwise.append(
                {
                    "first": STRATA[first],
                    "second": STRATA[second],
                    "absolute_mean_difference": difference,
                    "combined_standard_error": combined,
                    "within_two_combined_standard_errors": bool(difference <= 2.0 * combined),
                }
            )

    proposed = np.sum(
        [np.asarray(row["proposed_by_kernel"], dtype=np.int64) for row in diagnostics], axis=0
    )
    nontrivial = np.sum(
        [np.asarray(row["nontrivial_by_kernel"], dtype=np.int64) for row in diagnostics], axis=0
    )
    accepted = np.sum(
        [np.asarray(row["accepted_moves_by_kernel"], dtype=np.int64) for row in diagnostics], axis=0
    )
    kernel_names = ("exchange", "d6_orbit", "dimer_independence")
    kernel_report = {
        name: {
            "proposed": int(proposed[index]),
            "nontrivial": int(nontrivial[index]),
            "accepted_nontrivial": int(accepted[index]),
            "conditional_acceptance": float(accepted[index] / max(nontrivial[index], 1)),
        }
        for index, name in enumerate(kernel_names)
    }

    gates_spec = spec["gates"]
    gates = {
        "all_split_rhat_below_threshold": all(
            row["split_rhat"] is not None
            and float(row["split_rhat"]) < float(gates_spec["split_rhat_maximum"])
            for row in correlations.values()
        ),
        "all_strata_ess_fraction_above_threshold": all(
            float(row["ess_fraction"]) >= float(gates_spec["stratum_ess_fraction_minimum"])
            for stratum in by_stratum.values()
            for row in stratum.values()
        ),
        "full_h_strata_agree_within_two_sigma": all(
            row["within_two_combined_standard_errors"] for row in pairwise
        ),
        "d6_orbit_has_accepted_nontrivial_moves": kernel_report["d6_orbit"]["accepted_nontrivial"] > 0,
        "independence_has_accepted_nontrivial_moves": kernel_report["dimer_independence"]["accepted_nontrivial"] > 0,
        "checkpoint_replay_exact": replay_state_equal
        and replay_index_equal
        and replay_amplitude_error == 0.0,
        "no_zero_amplitudes": all(
            float(np.asarray(row["zero_amplitude_chain_fraction"])) == 0.0
            for row in diagnostics
        ),
        "finite_full_h": bool(np.all(np.isfinite(full_h))),
    }
    all_gates_passed = bool(all(gates.values()))
    final_generation = save_resumable_checkpoint_generation(
        checkpoint_directory,
        params,
        optimizer_state={"kind": "fixed_parameter_sampling_gate"},
        completed_steps=0,
        next_sample_index=proposal_index,
        run_identity=run_identity,
        history=[],
        metadata=_checkpoint_metadata(
            states=chains,
            amplitudes=amplitudes,
            master_key=master_key,
            proposal_index=proposal_index,
            labels=labels,
            params_hash=params_hash,
            images_hash=images_hash,
            matchings_hash=matchings_hash,
            spec=spec,
        ),
    )

    now = datetime.now(timezone.utc).isoformat()
    source = transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )
    payload = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "status": "complete",
        "created_utc": now,
        "completed_utc": now,
        "config": spec,
        "config_sha256": canonical_sha256(spec),
        "source_manifest": source,
        "source_manifest_sha256": canonical_sha256(source),
        "runtime_identity": numerical_runtime_identity(),
        "hamiltonian_diagonalization_performed": False,
        "model": {
            **ledger,
            "parameter_count": su2_route_vit_parameter_count(params),
            "exact_su2_by_construction": True,
            "spatial_projection": "translations_only",
        },
        "sampler": {
            "strata": list(STRATA),
            "chain_count": int(len(labels)),
            "d6_group_order": int(len(images)),
            "space_group_complete_dimer_covering_count": int(len(matchings)),
            "d6_images_sha256": images_hash,
            "dimer_matchings_sha256": matchings_hash,
            "burn": {
                name: np.asarray(value).tolist()
                for name, value in burn_diagnostics.items()
            },
            "measurement_kernels": kernel_report,
            "wall_seconds": time.perf_counter() - started,
        },
        "correlations": correlations,
        "correlations_by_initialization_stratum": by_stratum,
        "full_h": {
            "seconds": full_h_seconds,
            "sample_count": int(len(full_h)),
            "maximum_imaginary_local_energy": float(np.max(np.abs(full_h.imag))),
            "by_initialization_stratum": full_h_by_stratum,
            "pairwise_stratum_checks": pairwise,
        },
        "checkpoint": {
            "burn_generation": str(first_generation.relative_to(output)),
            "final_generation": str(final_generation.relative_to(output)),
            "replay_state_equal": replay_state_equal,
            "replay_index_equal": replay_index_equal,
            "replay_max_abs_amplitude_error": replay_amplitude_error,
        },
        "gates": gates,
        "all_gates_passed": all_gates_passed,
        "claim_boundary": (
            "Fixed-parameter translation-projected N36 sampling repair only. "
            "It is not a trained energy, full-Gamma-A1, phase, scaling, or "
            "carrier variational-win result."
        ),
    }
    payload["runtime_identity_sha256"] = canonical_sha256(payload["runtime_identity"])
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_json(result_path, payload)
    return payload


__all__ = ["STRATA", "TASK", "run_spin_su2_route_vit_n36_basin_gate"]
