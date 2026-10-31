"""Sealed deterministic replay of sampled-training fixed-parameter blocks.

The training workflow intentionally stores compact summaries rather than raw
local energies.  This independent analysis reloads the registered step
generations, reconstructs the exact fixed-parameter measurement streams, and
adds chain-level Monte Carlo uncertainty for ``E/N``, ``Var(H)/N``, and full
spin structure factors.  It never changes parameters or training chains.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows import spin_sampled_runtime as runtime_workflow
from projects.clebsch_tree_nqs.workflows import spin_sampled_training as training_workflow
from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    embedded_trial_manifest,
    pytree_sha256,
    trial_payload_sha256,
)
from su2_nqs.frontier.provenance import (
    file_sha256,
    numerical_runtime_identity,
    transitive_local_source_manifest,
)
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains
from su2_nqs.vmc.resumable_checkpoint import (
    canonical_json_sha256,
    load_resumable_checkpoint,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
POST_ANALYSIS_RESULT_SCHEMA_VERSION = 1
POST_ANALYSIS_CLAIM_BOUNDARY = (
    "Deterministic fixed-parameter replay and Monte Carlo resolution audit "
    "only. Single-seed short trajectories do not establish convergence, an "
    "efficiency advantage, magnetic order, a phase, QSL physics, or "
    "statistical/physical significance."
)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return f"external/{resolved.name}"


def _repository_path(relative: str) -> Path:
    candidate = (ROOT / relative).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("post-analysis input must remain inside the repository") from error
    return candidate


def _source_manifest() -> dict[str, str]:
    return transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )


def _validate_spec(spec: dict[str, object]) -> None:
    if spec.get("schema_version") != 1 or spec.get("domain") != "spin":
        raise ValueError("post-analysis requires schema_version=1 and domain='spin'")
    if spec.get("task") != "physical_sampled_training_post_analysis":
        raise ValueError("config task is not physical_sampled_training_post_analysis")
    path = spec.get("input_result")
    if not isinstance(path, str) or not path.endswith("/results.json"):
        raise ValueError("input_result must be a repository-relative results.json")
    if spec.get("expected_campaign_stage") != "pilot":
        raise ValueError("post-analysis is registered for the pilot campaign")
    baselines = spec.get("runtime_baselines")
    if not isinstance(baselines, list) or len(baselines) != 2:
        raise ValueError("post-analysis requires square and triangular runtime baselines")
    if {str(entry.get("geometry")) for entry in baselines} != {
        "square",
        "triangular",
    }:
        raise ValueError("runtime baselines must cover square and triangular geometry")
    for entry in baselines:
        if not isinstance(entry, dict):
            raise TypeError("every runtime baseline must be an object")
        path = entry.get("result")
        if not isinstance(path, str) or not path.endswith("/results.json"):
            raise ValueError("runtime baseline result must be a results.json path")
        if not str(entry.get("expected_case_id", "")):
            raise ValueError("runtime baseline expected_case_id must be nonempty")
    updates = spec.get("updates")
    if (
        not isinstance(updates, list)
        or not updates
        or updates != sorted(set(updates))
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in updates)
    ):
        raise ValueError("updates must be unique increasing positive integers")
    if spec.get("energy_normalization") != "E/N":
        raise ValueError("energy normalization must be E/N")
    if spec.get("variance_normalization") != "Var(H)/N":
        raise ValueError("variance normalization must be Var(H)/N")
    if spec.get("variance_primary_mcse") != (
        "independent_chain_means_over_registered_fixed_parameter_block"
    ):
        raise ValueError("variance primary MCSE contract changed")
    block = spec.get("naive_time_block_length_sweeps")
    if isinstance(block, bool) or not isinstance(block, int) or block < 1:
        raise ValueError("naive_time_block_length_sweeps must be positive")
    tolerance = float(spec.get("replay_absolute_tolerance", np.nan))
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("replay_absolute_tolerance must be positive and finite")
    if spec.get("claim_boundary") != POST_ANALYSIS_CLAIM_BOUNDARY:
        raise ValueError("post-analysis claim boundary differs from enforced code boundary")


def _load_training_result(
    spec: dict[str, object],
) -> tuple[dict[str, object], Path]:
    path = _repository_path(str(spec["input_result"]))
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != training_workflow.TRAINING_RESULT_SCHEMA_VERSION
        or payload.get("domain") != "spin"
        or payload.get("task") != "physical_sampled_training"
        or payload.get("campaign_stage") != spec["expected_campaign_stage"]
        or payload.get("status") != "complete"
    ):
        raise RuntimeError("post-analysis input is not a complete sampled-training pilot")
    if payload.get("config_sha256") != canonical_sha256(payload.get("config")):
        raise RuntimeError("training config seal failed")
    source = payload.get("source_sha256")
    if payload.get("source_manifest_sha256") != canonical_sha256(source):
        raise RuntimeError("training source seal failed")
    current_training_source = training_workflow._source_manifest()
    if source != current_training_source:
        raise RuntimeError("training input is stale relative to current training source")
    if payload.get("runtime_identity_sha256") != canonical_sha256(
        payload.get("runtime_identity")
    ):
        raise RuntimeError("training runtime identity seal failed")
    if payload.get("runtime_identity") != numerical_runtime_identity():
        raise RuntimeError("post-analysis must use the training numerical runtime")
    if payload.get("trial_manifest_sha256") != canonical_sha256(
        embedded_trial_manifest(payload)
    ):
        raise RuntimeError("training trial manifest seal failed")
    if payload.get("route_boundary", {}).get("direct_fusion_path_used") is not False:
        raise RuntimeError("training input violates the active route boundary")
    return payload, path


def _load_runtime_baselines(
    spec: dict[str, object],
) -> dict[str, tuple[dict[str, object], dict[str, object], Path]]:
    result: dict[str, tuple[dict[str, object], dict[str, object], Path]] = {}
    current_source = runtime_workflow._source_manifest()
    current_runtime = numerical_runtime_identity()
    for entry in spec["runtime_baselines"]:
        path = _repository_path(str(entry["result"]))
        payload = json.loads(path.read_text())
        if (
            payload.get("schema_version") != runtime_workflow.RUNTIME_RESULT_SCHEMA_VERSION
            or payload.get("task") != "physical_mcmc_runtime_smoke"
            or payload.get("status") != "complete"
        ):
            raise RuntimeError("runtime baseline is not a complete v2 smoke")
        if payload.get("config_sha256") != canonical_sha256(payload.get("config")):
            raise RuntimeError("runtime baseline config seal failed")
        if payload.get("source_sha256") != current_source or payload.get(
            "source_manifest_sha256"
        ) != canonical_sha256(current_source):
            raise RuntimeError("runtime baseline source seal failed or is stale")
        if payload.get("runtime_identity") != current_runtime or payload.get(
            "runtime_identity_sha256"
        ) != canonical_sha256(current_runtime):
            raise RuntimeError("runtime baseline numerical identity differs")
        if payload.get("trial_manifest_sha256") != canonical_sha256(
            embedded_trial_manifest(payload)
        ):
            raise RuntimeError("runtime baseline trial seal failed")
        cases = payload.get("cases")
        if not isinstance(cases, list) or len(cases) != 1:
            raise RuntimeError("runtime baseline must contain one case")
        case = cases[0]
        geometry = str(entry["geometry"])
        if (
            case.get("case_id") != entry["expected_case_id"]
            or case.get("geometry") != geometry
            or int(case.get("n_sites", -1)) != 36
        ):
            raise RuntimeError("runtime baseline case identity mismatch")
        if {trial.get("backbone") for trial in case.get("trials", [])} != {
            "u1",
            "su2",
        }:
            raise RuntimeError("runtime baseline must contain U1 and SU2")
        result[geometry] = (payload, case, path)
    return result


def _select_generation(
    directory: Path,
    *,
    completed_step: int,
    identity: dict[str, object],
) -> tuple[dict[str, object], Path, dict[str, object]]:
    manifest_path = directory / "LATEST.json"
    manifest = json.loads(manifest_path.read_text())
    expected_identity_hash = canonical_json_sha256(identity)
    if (
        manifest.get("kind") != "su2_nqs_checkpoint_series"
        or manifest.get("run_identity_sha256") != expected_identity_hash
        or not isinstance(manifest.get("generations"), list)
    ):
        raise RuntimeError("checkpoint-series manifest identity failed")
    candidates = [
        dict(entry)
        for entry in manifest["generations"]
        if isinstance(entry, dict)
        and entry.get("completed_steps") == completed_step
    ]
    candidates.sort(
        key=lambda entry: (int(entry.get("commit_sequence", -1)), str(entry.get("file"))),
        reverse=True,
    )
    failures = []
    for entry in candidates:
        path = (directory / str(entry["file"])).resolve()
        try:
            path.relative_to(directory.resolve())
        except ValueError:
            failures.append("generation path escapes checkpoint directory")
            continue
        if file_sha256(path) != entry.get("sha256"):
            failures.append(f"{path.name}: file checksum mismatch")
            continue
        try:
            payload = load_resumable_checkpoint(
                path, expected_run_identity=identity
            )
        except ValueError as error:
            failures.append(f"{path.name}: {error}")
            continue
        if int(payload["completed_steps"]) != completed_step:
            failures.append(f"{path.name}: completed-step mismatch")
            continue
        return payload, path, entry
    detail = "; ".join(failures) if failures else "no registered generation"
    raise RuntimeError(f"no valid step-{completed_step} generation: {detail}")


def _complex_pairs(values: np.ndarray) -> list:
    concrete = np.asarray(values, dtype=np.complex128)
    return np.stack((concrete.real, concrete.imag), axis=-1).tolist()


def _chain_summary(values: np.ndarray) -> dict[str, float | int]:
    concrete = np.asarray(values, dtype=np.complex128)
    if concrete.ndim != 2 or min(concrete.shape) < 2:
        raise ValueError("chain summary requires shape (sweeps>=2, chains>=2)")
    chain_means = np.mean(concrete.real, axis=0)
    return {
        "mean_real": float(np.mean(concrete.real)),
        "mean_imaginary": float(np.mean(concrete.imag)),
        "maximum_abs_imaginary_local": float(np.max(np.abs(concrete.imag))),
        "chain_mean_mcse": float(
            np.std(chain_means, ddof=1) / np.sqrt(len(chain_means))
        ),
        "sample_count": int(concrete.size),
    }


def _variance_summary(
    local_values: np.ndarray,
    *,
    n_sites: int,
    block_length: int,
) -> dict[str, object]:
    concrete = np.asarray(local_values, dtype=np.complex128)
    mean_energy = np.mean(concrete)
    contributions = np.abs(concrete - mean_energy) ** 2
    variance = float(np.mean(contributions))
    chain_means = np.mean(contributions, axis=0)
    chain_mcse = float(
        np.std(chain_means, ddof=1) / np.sqrt(len(chain_means))
    )
    usable = (concrete.shape[0] // block_length) * block_length
    blocked = contributions[:usable].reshape(
        usable // block_length, block_length, concrete.shape[1]
    ).mean(axis=1)
    naive_block_mcse = float(
        np.std(blocked.reshape(-1), ddof=1) / np.sqrt(blocked.size)
    )
    return {
        "variance_H": variance,
        "variance_H_chain_mcse": chain_mcse,
        "variance_H_per_site": variance / n_sites,
        "variance_H_per_site_chain_mcse": chain_mcse / n_sites,
        "primary_mcse_contract": (
            "standard error over independent-chain means of local variance "
            "contributions within one fixed-parameter block"
        ),
        "naive_time_block_length_sweeps": block_length,
        "naive_time_block_mcse_H_per_site": naive_block_mcse / n_sites,
        "naive_time_block_mcse_is_primary": False,
        "variance_contributions_sha256": array_sha256(
            contributions.astype(np.float64)
        ),
    }


def _replay_block(
    *,
    runtime: runtime_workflow._ArmRuntime,
    params: object,
    training_states: np.ndarray,
    case: dict[str, object],
    bonds: tuple[tuple[int, int, float], ...],
    names: tuple[str, ...],
    training_spec: dict[str, object],
    update: int,
    advance_uncached: Callable,
    advance_cached: Callable,
    hpsi_fn: Callable,
    observable_fn: Callable,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float]:
    sampler = dict(training_spec["sampler"])
    measurement = dict(training_spec["measurement"])
    chains = int(sampler["chains"])
    proposals = int(sampler["proposals_per_sweep"])
    bond_probability = float(sampler["bond_proposal_probability"])
    padded_width = len(bonds) + 1
    microbatch = int(training_spec["optimizer"]["microbatch_size"])
    observable_batch = int(measurement["observable_batch_size"])
    fixed_params_hash = pytree_sha256(params)
    training_states_hash = array_sha256(training_states)

    fold_value = int(
        canonical_sha256(
            {
                "measurement_seed": int(measurement["seed"]),
                "case_id": case["case_id"],
                "arm_id": runtime.arm_id,
                "update": update,
            }
        )[:8],
        16,
    )
    key = jax.random.fold_in(
        jax.random.PRNGKey(int(measurement["seed"])), fold_value
    )
    states = jnp.asarray(np.array(training_states, copy=True), dtype=jnp.int32)
    burn_attempts = int(measurement["burn_sweeps"]) * proposals
    states, amplitudes, key, burn_diagnostics = advance_uncached(
        params, key, states, burn_attempts, bond_probability
    )
    jax.block_until_ready((states, amplitudes, key, burn_diagnostics))
    diagnostics = [training_workflow._float_diagnostics(burn_diagnostics)]
    locals_: list[np.ndarray] = []
    modes_: list[np.ndarray] = []
    started = time.perf_counter()
    for _ in range(int(measurement["measurement_sweeps"])):
        states, amplitudes, key, diagnostic = advance_cached(
            params, key, states, amplitudes, proposals, bond_probability
        )
        jax.block_until_ready((states, amplitudes, key, diagnostic))
        diagnostics.append(training_workflow._float_diagnostics(diagnostic))
        host_states = np.asarray(states, dtype=np.int32)
        local, _host_seconds, _device_seconds, nonfinite = (
            training_workflow._local_energy_chunks(
                params,
                host_states,
                bonds,
                padded_width=padded_width,
                microbatch_size=microbatch,
                hpsi_fn=hpsi_fn,
            )
        )
        if nonfinite:
            raise FloatingPointError("replay produced non-finite local energy")
        locals_.append(local)
        chunks = []
        for start in range(0, chains, observable_batch):
            values = observable_fn(
                params, jnp.asarray(host_states[start : start + observable_batch])
            )
            jax.block_until_ready(values)
            chunks.append(np.asarray(values, dtype=np.complex128))
        modes_.append(np.concatenate(chunks, axis=0))
    elapsed = time.perf_counter() - started
    if fixed_params_hash != pytree_sha256(params):
        raise RuntimeError("replay mutated parameters")
    if training_states_hash != array_sha256(training_states):
        raise RuntimeError("replay mutated checkpoint training chains")
    local_values = np.stack(locals_, axis=0)
    mode_values = np.stack(modes_, axis=0)
    if np.any(~np.isfinite(local_values)) or np.any(~np.isfinite(mode_values)):
        raise FloatingPointError("replay produced non-finite raw arrays")
    return (
        local_values,
        mode_values,
        runtime_workflow._summarize_diagnostics(diagnostics),
        elapsed,
    )


def _replay_discrepancies(
    *,
    local_values: np.ndarray,
    mode_values: np.ndarray,
    names: tuple[str, ...],
    stored: dict[str, object],
    sampler_diagnostics: dict[str, float],
    n_sites: int,
) -> dict[str, float]:
    flat = local_values.reshape(-1)
    energy = complex(np.mean(flat))
    variance = float(np.mean(np.abs(flat - energy) ** 2).real)
    chain_means = np.mean(local_values.real, axis=0)
    energy_mcse = float(
        np.std(chain_means, ddof=1) / np.sqrt(len(chain_means))
    )
    discrepancies = {
        "energy_mean": abs(float(energy.real) - float(stored["energy_mean"])),
        "energy_imaginary_abs": abs(
            float(abs(energy.imag)) - float(stored["energy_imaginary_abs"])
        ),
        "energy_variance": abs(variance - float(stored["energy_variance"])),
        "energy_chain_mean_mcse": abs(
            energy_mcse - float(stored["energy_chain_mean_mcse"])
        ),
        "energy_per_site": abs(
            float(energy.real) / n_sites - float(stored["energy_per_site"])
        ),
        "q0_maximum_abs_local": abs(
            float(np.max(np.abs(mode_values[:, :, names.index("q0")])))
            - float(stored["full_spin_structure_factor"]["q0_maximum_abs_local"])
        ),
    }
    stored_modes = stored["full_spin_structure_factor"]["mode_summaries"]
    for index, name in enumerate(names):
        summary = _chain_summary(mode_values[:, :, index])
        for field in (
            "mean_real",
            "mean_imaginary",
            "maximum_abs_imaginary_local",
            "chain_mean_mcse",
        ):
            discrepancies[f"{name}.{field}"] = abs(
                float(summary[field]) - float(stored_modes[name][field])
            )
        discrepancies[f"{name}.sample_count"] = abs(
            int(summary["sample_count"]) - int(stored_modes[name]["sample_count"])
        )
    for field, value in sampler_diagnostics.items():
        discrepancies[f"sampler.{field}"] = abs(
            float(value) - float(stored["sampler_diagnostics"][field])
        )
    return discrepancies


def _resolution_label(ratio: float) -> str:
    if ratio < 1.0:
        return "unresolved_within_MCSE"
    if ratio < 2.0:
        return "suggestive_exploratory"
    return "MC_resolved_exploratory_not_physical_significance"


def _trend(
    updates: list[int], values: list[float], errors: list[float]
) -> dict[str, object]:
    x = np.asarray(updates, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    sigma = np.asarray(errors, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(sigma)):
        raise ValueError("trend inputs must be finite")
    combined = float(np.hypot(sigma[0], sigma[-1]))
    delta = float(y[-1] - y[0])
    endpoint_ratio = abs(delta) / combined if combined > 0.0 else math.inf
    safe_sigma = np.maximum(sigma, np.finfo(np.float64).tiny)
    weights = 1.0 / safe_sigma**2
    x_bar = float(np.sum(weights * x) / np.sum(weights))
    y_bar = float(np.sum(weights * y) / np.sum(weights))
    denominator = float(np.sum(weights * (x - x_bar) ** 2))
    slope = float(np.sum(weights * (x - x_bar) * (y - y_bar)) / denominator)
    slope_mcse = float(np.sqrt(1.0 / denominator))
    slope_ratio = abs(slope) / slope_mcse if slope_mcse > 0.0 else math.inf
    return {
        "first_update": int(updates[0]),
        "last_update": int(updates[-1]),
        "first_value": float(values[0]),
        "last_value": float(values[-1]),
        "endpoint_delta_last_minus_first": delta,
        "endpoint_combined_chain_mcse": combined,
        "endpoint_abs_delta_over_combined_mcse": endpoint_ratio,
        "endpoint_resolution": _resolution_label(endpoint_ratio),
        "weighted_slope_per_update": slope,
        "weighted_slope_chain_mcse": slope_mcse,
        "weighted_slope_abs_over_mcse": slope_ratio,
        "weighted_slope_resolution": _resolution_label(slope_ratio),
        "mcse_model_boundary": (
            "chain-level Monte Carlo resolution only; it excludes optimizer, "
            "finite-size, ansatz, and cross-update covariance uncertainty"
        ),
    }


def _paired_chain_trend(
    updates: list[int], chain_values: list[np.ndarray]
) -> dict[str, object]:
    """Resolve change using preserved chain lineage across checkpoints.

    The sampled-training workflow never reorders its leading chain axis.  The
    replay stream copies those checkpoint chains and applies independent
    update-registered random streams without changing the axis order.  Taking
    differences before the standard error therefore retains the observed
    cross-checkpoint covariance for each chain lineage.
    """

    concrete = np.asarray(chain_values, dtype=np.float64)
    if (
        concrete.ndim != 2
        or concrete.shape[0] != len(updates)
        or concrete.shape[1] < 2
        or np.any(~np.isfinite(concrete))
    ):
        raise ValueError("paired trend needs one finite value per update and chain")
    differences = concrete[-1] - concrete[0]
    delta = float(np.mean(differences))
    endpoint_mcse = float(
        np.std(differences, ddof=1) / np.sqrt(differences.size)
    )
    endpoint_ratio = abs(delta) / endpoint_mcse if endpoint_mcse > 0.0 else math.inf

    x = np.asarray(updates, dtype=np.float64)
    centered_x = x - np.mean(x)
    denominator = float(np.sum(centered_x**2))
    per_chain_slopes = np.sum(centered_x[:, None] * concrete, axis=0) / denominator
    slope = float(np.mean(per_chain_slopes))
    slope_mcse = float(
        np.std(per_chain_slopes, ddof=1) / np.sqrt(per_chain_slopes.size)
    )
    slope_ratio = abs(slope) / slope_mcse if slope_mcse > 0.0 else math.inf
    return {
        "chain_lineage_index_preserved": True,
        "first_update": int(updates[0]),
        "last_update": int(updates[-1]),
        "paired_endpoint_delta_last_minus_first": delta,
        "paired_endpoint_chain_mcse": endpoint_mcse,
        "paired_endpoint_abs_delta_over_mcse": endpoint_ratio,
        "paired_endpoint_resolution": _resolution_label(endpoint_ratio),
        "paired_unweighted_slope_per_update": slope,
        "paired_unweighted_slope_chain_mcse": slope_mcse,
        "paired_unweighted_slope_abs_over_mcse": slope_ratio,
        "paired_unweighted_slope_resolution": _resolution_label(slope_ratio),
        "primary_endpoint_resolution_uses_paired_chain_difference": True,
        "boundary": (
            "paired-chain Monte Carlo resolution only; chain lineages are "
            "not independent optimization replicas and this is not physical "
            "or statistical significance"
        ),
    }


def _difference_with_mcse(
    left: float,
    left_error: float,
    right: float,
    right_error: float,
    *,
    label: str,
) -> dict[str, object]:
    delta = float(left - right)
    combined = float(np.hypot(left_error, right_error))
    ratio = abs(delta) / combined if combined > 0.0 else math.inf
    return {
        "contrast": label,
        "difference": delta,
        "combined_chain_mcse": combined,
        "abs_difference_over_combined_mcse": ratio,
        "resolution": _resolution_label(ratio),
    }


def _find_by(items: list[dict[str, object]], field: str, value: object) -> dict[str, object]:
    matches = [item for item in items if item.get(field) == value]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {field}={value!r} record")
    return matches[0]


def _derived_mode_arrays(
    geometry: str,
    mode_values: np.ndarray,
    names: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    by_name = {name: mode_values[:, :, index] for index, name in enumerate(names)}
    if geometry == "square":
        stripe = np.stack((by_name["stripe_x"], by_name["stripe_y"]), axis=-1)
        arrays = {
            "neel": by_name["neel"],
            "stripe_star_mean": np.mean(stripe, axis=-1),
            "stripe_star_max_secondary": np.max(stripe.real, axis=-1).astype(
                np.complex128
            ),
        }
        contract = {
            "primary": ["neel", "stripe_star_mean"],
            "secondary": ["stripe_star_max_secondary"],
            "star_mean_formula": "(S(pi,0)+S(0,pi))/2 per sample",
        }
    elif geometry == "triangular":
        m_star = np.stack(
            (by_name["m1"], by_name["m2"], by_name["m3"]), axis=-1
        )
        arrays = {
            "k": by_name["k"],
            "M_star_mean": np.mean(m_star, axis=-1),
            "M_star_max_secondary": np.max(m_star.real, axis=-1).astype(
                np.complex128
            ),
        }
        contract = {
            "primary": ["k", "M_star_mean"],
            "secondary": ["M_star_max_secondary"],
            "star_mean_formula": "(S(M1)+S(M2)+S(M3))/3 per sample",
        }
    else:  # pragma: no cover - guarded by the training config
        raise ValueError(f"unsupported geometry: {geometry!r}")
    contract.update(
        {
            "star_mean_mcse_uses_per_sample_component_average": True,
            "component_covariance_retained_in_chain_means": True,
            "max_star_has_selection_bias_and_is_not_primary": True,
            "q0_role": "spin-sector diagnostic only",
        }
    )
    return arrays, contract


def _validate_generation(
    checkpoint: dict[str, object],
    *,
    update: int,
    params_hash: str,
    chains: int,
    n_sites: int,
) -> tuple[dict[str, object], np.ndarray, dict[str, object]]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("checkpoint metadata is missing")
    states = np.asarray(metadata.get("chain_states"), dtype=np.int32)
    history = checkpoint.get("history")
    if (
        int(checkpoint.get("completed_steps", -1)) != update
        or int(checkpoint.get("next_sample_index", -1)) != update * chains
        or not isinstance(history, list)
        or len(history) != update
        or states.shape != (chains, n_sites)
        or np.any(np.sum(states, axis=1) != n_sites // 2)
    ):
        raise RuntimeError("checkpoint counters, history, or chain shape failed")
    if metadata.get("amplitude_cache_present") is not False or "amplitudes" in metadata:
        raise RuntimeError("post-analysis refuses a checkpoint with amplitude cache")
    if metadata.get("params_sha256") != params_hash:
        raise RuntimeError("checkpoint parameter hash failed")
    if metadata.get("chain_states_sha256") != array_sha256(states):
        raise RuntimeError("checkpoint chain hash failed")
    if metadata.get("optimizer_state_sha256") != pytree_sha256(
        checkpoint["optimizer_state"]
    ):
        raise RuntimeError("checkpoint optimizer hash failed")
    row = history[-1]
    if (
        not isinstance(row, dict)
        or int(row.get("update", -1)) != update
        or row.get("params_after_sha256") != params_hash
        or row.get("chain_states_sha256") != metadata["chain_states_sha256"]
        or row.get("post_update_amplitude_cache")
        != "discarded_and_not_checkpointed"
    ):
        raise RuntimeError("checkpoint final history row failed")
    stored = row.get("fixed_parameter_measurement")
    if (
        not isinstance(stored, dict)
        or int(stored.get("update", -1)) != update
        or stored.get("fixed_params_sha256") != params_hash
    ):
        raise RuntimeError("checkpoint lacks the registered fixed block")
    return metadata, states, stored


def _analyze_trial(
    *,
    training_payload: dict[str, object],
    training_path: Path,
    case_payload: dict[str, object],
    case_spec: dict[str, object],
    arm_spec: dict[str, object],
    training_trial: dict[str, object],
    runtime_baseline: tuple[dict[str, object], dict[str, object], Path],
    analysis_spec: dict[str, object],
) -> dict[str, object]:
    training_spec = training_payload["config"]
    projection = dict(training_spec["projection"])
    bonds, group, projector = training_workflow._geometry(case_spec, projection)
    runtime = runtime_workflow._initialize_arm(
        arm_spec,
        rows=int(case_spec["rows"]),
        columns=int(case_spec["columns"]),
        projector=projector,
    )
    n_sites = int(case_spec["rows"]) * int(case_spec["columns"])
    chains = int(training_spec["sampler"]["chains"])
    if (
        case_payload.get("bonds_sha256") != canonical_sha256(bonds)
        or case_payload.get("translation_group_sha256") != canonical_sha256(group)
        or runtime.arm_id != training_trial.get("arm_id")
        or runtime.backbone != training_trial.get("backbone")
        or runtime.parameter_count != int(training_trial.get("parameter_count", -1))
        or pytree_sha256(runtime.params) != training_trial.get("initial_params_sha256")
    ):
        raise RuntimeError("training case/model reconstruction failed")

    seed_key = jax.random.PRNGKey(int(training_spec["sampler"]["seed"]))
    chain_key, _sampling_key = jax.random.split(seed_key)
    initial_states = np.asarray(
        initialize_spin_exchange_chains(
            chain_key,
            n_chains=chains,
            n_sites=n_sites,
            n_up=n_sites // 2,
        ),
        dtype=np.int32,
    )
    initial_chain_hash = array_sha256(initial_states)
    if initial_chain_hash != training_trial.get("initial_chain_states_sha256"):
        raise RuntimeError("training initial-chain reconstruction failed")
    identity = training_workflow._run_identity(
        config_hash=str(training_payload["config_sha256"]),
        source_hash=str(training_payload["source_manifest_sha256"]),
        runtime_hash=str(training_payload["runtime_identity_sha256"]),
        case=case_spec,
        arm=arm_spec,
        runtime=runtime,
        bonds=bonds,
        group=group,
        initial_chain_states_sha256=initial_chain_hash,
        spec=training_spec,
    )
    identity_hash = canonical_json_sha256(identity)
    if identity_hash != training_trial["checkpoint"]["run_identity_sha256"]:
        raise RuntimeError("reconstructed training run identity failed")
    checkpoint_directory = (
        training_path.parent / str(training_trial["checkpoint"]["series_directory"])
    ).resolve()
    checkpoint_directory.relative_to(training_path.parent.resolve())

    baseline_payload, baseline_case, baseline_path = runtime_baseline
    baseline_trial = _find_by(
        list(baseline_case["trials"]), "arm_id", runtime.arm_id
    )
    baseline_arm = _find_by(
        list(baseline_payload["config"]["arms"]), "arm_id", runtime.arm_id
    )
    if (
        baseline_case.get("geometry") != case_payload.get("geometry")
        or int(baseline_case.get("rows", -1)) != int(case_payload.get("rows", -2))
        or int(baseline_case.get("columns", -1))
        != int(case_payload.get("columns", -2))
        or float(baseline_case.get("j2_over_j1", np.nan))
        != float(case_payload.get("j2_over_j1", np.inf))
        or baseline_case.get("bonds_sha256") != case_payload.get("bonds_sha256")
        or baseline_case.get("translation_group_sha256")
        != case_payload.get("translation_group_sha256")
        or baseline_payload["config"]["projection"] != projection
        or baseline_arm != arm_spec
        or baseline_trial.get("backbone") != runtime.backbone
        or int(baseline_trial.get("parameter_count", -1)) != runtime.parameter_count
        or baseline_trial.get("initial_params_sha256")
        != training_trial.get("initial_params_sha256")
    ):
        raise RuntimeError("runtime update-0 baseline is not model/geometry matched")
    baseline_observables = baseline_trial["observables"]
    baseline_total = float(baseline_observables["energy_mean_random_state"])
    baseline_per_site = float(baseline_observables["energy_per_site_random_state"])
    baseline_total_mcse = float(
        baseline_observables["energy_mcmc_standard_error"]
    )
    if abs(baseline_total / n_sites - baseline_per_site) > 1e-12:
        raise RuntimeError("runtime baseline energy normalization failed")

    names, observable_specs = training_workflow._mode_specs(case_spec)
    advance_uncached, advance_cached, hpsi_fn, observable_fn = (
        training_workflow._compiled_kernels(runtime, bonds, observable_specs)
    )
    updates = [int(value) for value in analysis_spec["updates"]]
    tolerance = float(analysis_spec["replay_absolute_tolerance"])
    block_length = int(analysis_spec["naive_time_block_length_sweeps"])
    rows: list[dict[str, object]] = []
    chain_series: dict[str, list[np.ndarray]] = {
        "energy_E_per_site": [],
        "variance_H_per_site": [],
    }
    value_series: dict[str, list[float]] = {
        "energy_E_per_site": [],
        "variance_H_per_site": [],
    }
    error_series: dict[str, list[float]] = {
        "energy_E_per_site": [],
        "variance_H_per_site": [],
    }
    observable_contract: dict[str, object] | None = None
    generation_records: list[dict[str, object]] = []
    largest_discrepancy = 0.0

    for update in updates:
        checkpoint, generation_path, generation_entry = _select_generation(
            checkpoint_directory,
            completed_step=update,
            identity=identity,
        )
        params = checkpoint["params"]
        params_hash = pytree_sha256(params)
        metadata, training_states, stored = _validate_generation(
            checkpoint,
            update=update,
            params_hash=params_hash,
            chains=chains,
            n_sites=n_sites,
        )
        if (
            metadata.get("initial_chain_states_sha256") != initial_chain_hash
            or metadata.get("initial_params_sha256")
            != training_trial.get("initial_params_sha256")
            or canonical_sha256(checkpoint["history"])
            != canonical_sha256(training_trial["history"][:update])
        ):
            raise RuntimeError("checkpoint lineage/history seal failed")
        local_values, mode_values, replay_diagnostics, elapsed = _replay_block(
            runtime=runtime,
            params=params,
            training_states=training_states,
            case=case_spec,
            bonds=bonds,
            names=names,
            training_spec=training_spec,
            update=update,
            advance_uncached=advance_uncached,
            advance_cached=advance_cached,
            hpsi_fn=hpsi_fn,
            observable_fn=observable_fn,
        )
        discrepancies = _replay_discrepancies(
            local_values=local_values,
            mode_values=mode_values,
            names=names,
            stored=stored,
            sampler_diagnostics=replay_diagnostics,
            n_sites=n_sites,
        )
        row_maximum = max(discrepancies.values(), default=0.0)
        largest_discrepancy = max(largest_discrepancy, row_maximum)
        if row_maximum > tolerance:
            raise RuntimeError(
                f"step-{update} replay differs by {row_maximum:.3e} > {tolerance:.3e}"
            )

        energy_per_site_values = local_values / n_sites
        energy_summary = _chain_summary(energy_per_site_values)
        variance = _variance_summary(
            local_values, n_sites=n_sites, block_length=block_length
        )
        energy_chain_values = np.mean(energy_per_site_values.real, axis=0)
        energy_mean = complex(np.mean(local_values))
        variance_contributions = np.abs(local_values - energy_mean) ** 2 / n_sites
        variance_chain_values = np.mean(variance_contributions, axis=0)
        chain_series["energy_E_per_site"].append(energy_chain_values)
        chain_series["variance_H_per_site"].append(variance_chain_values)
        value_series["energy_E_per_site"].append(float(energy_summary["mean_real"]))
        error_series["energy_E_per_site"].append(
            float(energy_summary["chain_mean_mcse"])
        )
        value_series["variance_H_per_site"].append(
            float(variance["variance_H_per_site"])
        )
        error_series["variance_H_per_site"].append(
            float(variance["variance_H_per_site_chain_mcse"])
        )

        mode_summaries = {
            name: _chain_summary(mode_values[:, :, index])
            for index, name in enumerate(names)
        }
        derived_arrays, contract = _derived_mode_arrays(
            str(case_spec["geometry"]), mode_values, names
        )
        if observable_contract is None:
            observable_contract = contract
        elif observable_contract != contract:  # pragma: no cover - static geometry
            raise RuntimeError("observable contract changed between checkpoints")
        derived_summaries = {
            name: _chain_summary(values) for name, values in derived_arrays.items()
        }
        for name, values in derived_arrays.items():
            chain_series.setdefault(name, []).append(np.mean(values.real, axis=0))
            value_series.setdefault(name, []).append(
                float(derived_summaries[name]["mean_real"])
            )
            error_series.setdefault(name, []).append(
                float(derived_summaries[name]["chain_mean_mcse"])
            )

        generation_record = {
            "completed_update": update,
            "file": generation_path.name,
            "file_sha256": file_sha256(generation_path),
            "registered_file_sha256": generation_entry["sha256"],
            "commit_sequence": int(generation_entry["commit_sequence"]),
            "run_identity_sha256": str(checkpoint["run_identity_sha256"]),
            "params_sha256": params_hash,
            "optimizer_state_sha256": str(metadata["optimizer_state_sha256"]),
            "chain_states_sha256": str(metadata["chain_states_sha256"]),
            "initial_params_sha256": str(metadata["initial_params_sha256"]),
            "initial_chain_states_sha256": str(
                metadata["initial_chain_states_sha256"]
            ),
            "resume_count_at_generation": int(metadata["resume_count"]),
            "amplitude_cache_present": bool(metadata["amplitude_cache_present"]),
            "generation_and_checkpoint_seals_passed": True,
        }
        generation_records.append(generation_record)
        rows.append(
            {
                "update": update,
                "checkpoint_generation": generation_record,
                "energy_E_per_site": energy_summary,
                "variance_H_per_site": variance,
                "full_SU2_structure_factor_mode_summaries": mode_summaries,
                "physics_observables": derived_summaries,
                "raw_replay": {
                    "shape_energy_sweep_chain": list(local_values.shape),
                    "shape_modes_sweep_chain_mode": list(mode_values.shape),
                    "mode_order": list(names),
                    "total_H_local_values_complex_pairs": _complex_pairs(local_values),
                    "full_Sq_local_values_complex_pairs": _complex_pairs(mode_values),
                    "total_H_local_values_sha256": array_sha256(local_values),
                    "full_Sq_local_values_sha256": array_sha256(mode_values),
                },
                "registered_summary_replay_discrepancies": discrepancies,
                "maximum_registered_summary_replay_discrepancy": row_maximum,
                "registered_summary_replay_within_tolerance": True,
                "sampler_diagnostics": replay_diagnostics,
                "replay_wall_seconds_diagnostic_only": elapsed,
            }
        )

    trends = {
        name: {
            "paired_chain_primary": _paired_chain_trend(
                updates, chain_series[name]
            ),
            "independent_endpoint_and_weighted_fit_diagnostic": _trend(
                updates, value_series[name], error_series[name]
            ),
        }
        for name in value_series
    }
    final_energy = value_series["energy_E_per_site"][-1]
    final_energy_mcse = error_series["energy_E_per_site"][-1]
    baseline_mcse_per_site = baseline_total_mcse / n_sites
    baseline_comparison = _difference_with_mcse(
        final_energy,
        final_energy_mcse,
        baseline_per_site,
        baseline_mcse_per_site,
        label="pilot_step20_minus_runtime_random_state_update0_E_per_site",
    )
    baseline_comparison.update(
        {
            "energy_lowering_update0_minus_step20": baseline_per_site - final_energy,
            "signed_energy_lowering_z": (
                (baseline_per_site - final_energy)
                / float(baseline_comparison["combined_chain_mcse"])
                if float(baseline_comparison["combined_chain_mcse"]) > 0.0
                else math.inf
            ),
            "runtime_and_training_measurement_streams_independent": True,
        }
    )
    gates = {
        "all_registered_generations_verified": all(
            bool(row["generation_and_checkpoint_seals_passed"])
            for row in generation_records
        ),
        "all_fixed_blocks_replayed_within_tolerance": largest_discrepancy <= tolerance,
        "initial_model_params_match_runtime_update0": (
            baseline_trial["initial_params_sha256"]
            == training_trial["initial_params_sha256"]
        ),
        "geometry_and_projection_match_runtime_update0": True,
        "raw_arrays_finite": all(
            np.isfinite(np.asarray(chain_series[name])).all()
            for name in chain_series
        ),
        "chain_lineage_preserved_for_paired_differences": True,
        "star_mean_mcse_retains_channel_covariance": True,
    }
    baseline_sample_count = int(baseline_payload["config"]["sampler"]["chains"]) * int(
        baseline_payload["config"]["sampler"]["measurement_sweeps"]
    )
    return {
        "arm_id": runtime.arm_id,
        "backbone": runtime.backbone,
        "projection": "translation",
        "feature_dim": runtime.feature_dim,
        "maximum_intermediate_q": runtime.maximum_intermediate_q,
        "seed": int(arm_spec["seed"]),
        "parameter_count": runtime.parameter_count,
        "initial_params_sha256": str(training_trial["initial_params_sha256"]),
        "initial_chain_states_sha256": initial_chain_hash,
        "checkpoint_run_identity_sha256": identity_hash,
        "runtime_random_state_update0": {
            "source_result": _portable_path(baseline_path),
            "source_result_sha256": file_sha256(baseline_path),
            "source_case_id": baseline_case["case_id"],
            "energy_E_per_site": baseline_per_site,
            "energy_E_per_site_mcse": baseline_mcse_per_site,
            "reported_total_energy": baseline_total,
            "reported_total_energy_mcse": baseline_total_mcse,
            "sample_count": baseline_sample_count,
            "mcse_contract": (
                "runtime energy_mcmc_standard_error is a total-H, "
                "autocorrelation-adjusted MCSE; it is divided by N here"
            ),
            "matched_initial_params": True,
            "initial_chain_states_are_not_required_to_match": True,
        },
        "replay_updates": rows,
        "observable_contract": observable_contract,
        "trends": trends,
        "runtime_update0_to_pilot_step20_energy_comparison": baseline_comparison,
        "largest_registered_summary_replay_discrepancy": largest_discrepancy,
        "replay_absolute_tolerance": tolerance,
        "generation_manifest": generation_records,
        "training_resume_count": int(training_trial["timing"]["resume_count"]),
        "training_timing_claim_eligible": bool(
            training_trial["timing"]["timing_claim_eligible"]
        ),
        "post_analysis_timing_claim_eligible": False,
        "post_analysis_timing_boundary": (
            "replay timings include compilation/order effects and are diagnostic only"
        ),
        "audit_gates": gates,
        "all_post_analysis_gates_passed": all(gates.values()),
        "claim_boundary": POST_ANALYSIS_CLAIM_BOUNDARY,
    }


def _case_contrast(case: dict[str, object], update: int) -> dict[str, object]:
    by_backbone = {str(trial["backbone"]): trial for trial in case["trials"]}
    if set(by_backbone) != {"u1", "su2"}:
        raise RuntimeError("case contrast requires one U1 and one SU2 trial")

    def final_row(trial: dict[str, object]) -> dict[str, object]:
        return _find_by(list(trial["replay_updates"]), "update", update)

    u1 = final_row(by_backbone["u1"])
    su2 = final_row(by_backbone["su2"])
    energy = _difference_with_mcse(
        float(su2["energy_E_per_site"]["mean_real"]),
        float(su2["energy_E_per_site"]["chain_mean_mcse"]),
        float(u1["energy_E_per_site"]["mean_real"]),
        float(u1["energy_E_per_site"]["chain_mean_mcse"]),
        label="SU2_minus_U1_step20_E_per_site",
    )
    variance = _difference_with_mcse(
        float(su2["variance_H_per_site"]["variance_H_per_site"]),
        float(su2["variance_H_per_site"]["variance_H_per_site_chain_mcse"]),
        float(u1["variance_H_per_site"]["variance_H_per_site"]),
        float(u1["variance_H_per_site"]["variance_H_per_site_chain_mcse"]),
        label="SU2_minus_U1_step20_Var_H_per_site",
    )
    primary_names = list(by_backbone["u1"]["observable_contract"]["primary"])
    observables = {
        name: _difference_with_mcse(
            float(su2["physics_observables"][name]["mean_real"]),
            float(su2["physics_observables"][name]["chain_mean_mcse"]),
            float(u1["physics_observables"][name]["mean_real"]),
            float(u1["physics_observables"][name]["chain_mean_mcse"]),
            label=f"SU2_minus_U1_step20_{name}",
        )
        for name in primary_names
    }
    return {
        "update": update,
        "energy": energy,
        "variance": variance,
        "primary_physics_observables": observables,
        "matched_contract": {
            "same_backbone_tree_topology_feature_dim_cutoff_projection": True,
            "same_samples_chains_proposals_updates_and_schedule": True,
            "parameter_count_matched": False,
            "FLOP_matched": False,
            "single_seed": True,
        },
        "timing_comparison_eligible": all(
            bool(trial["training_timing_claim_eligible"])
            for trial in by_backbone.values()
        ),
        "claim_boundary": (
            "descriptive same-schedule single-seed contrast only; it is not an "
            "energy, efficiency, phase, order, or QSL win"
        ),
    }


def run_spin_sampled_training_post_analysis(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Replay and seal the registered pilot's fixed-parameter measurements."""

    spec = json.loads(config_path.read_text())
    _validate_spec(spec)
    training_payload, training_path = _load_training_result(spec)
    baselines = _load_runtime_baselines(spec)
    config_hash = canonical_sha256(spec)
    source_manifest = _source_manifest()
    source_hash = canonical_sha256(source_manifest)
    runtime_identity = numerical_runtime_identity()
    runtime_hash = canonical_sha256(runtime_identity)
    input_hashes = {
        "sampled_training_pilot": file_sha256(training_path),
        **{
            f"runtime_{geometry}": file_sha256(item[2])
            for geometry, item in sorted(baselines.items())
        },
    }
    input_hash = canonical_sha256(input_hashes)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        payload = json.loads(result_path.read_text())
        if (
            payload.get("status") != "complete"
            or payload.get("schema_version") != POST_ANALYSIS_RESULT_SCHEMA_VERSION
            or payload.get("config_sha256") != config_hash
            or payload.get("source_manifest_sha256") != source_hash
            or payload.get("runtime_identity_sha256") != runtime_hash
            or payload.get("input_results_manifest_sha256") != input_hash
            or payload.get("trial_manifest_sha256")
            != canonical_sha256(embedded_trial_manifest(payload))
        ):
            raise RuntimeError(f"refusing stale post-analysis result: {result_path}")
        print(f"reused immutable complete post-analysis: {result_path}", flush=True)
        return payload
    if result_path.exists():
        result_path.unlink()
    payload: dict[str, object] = {
        "schema_version": POST_ANALYSIS_RESULT_SCHEMA_VERSION,
        "domain": "spin",
        "task": "physical_sampled_training_post_analysis",
        "status": "running",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": _portable_path(config_path),
        "config": spec,
        "config_sha256": config_hash,
        "source_sha256": source_manifest,
        "source_manifest_sha256": source_hash,
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_hash,
        "input_results_manifest": input_hashes,
        "input_results_manifest_sha256": input_hash,
        "route_boundary": {
            "direct_fusion_path_used": False,
            "parameter_update_performed": False,
            "training_chain_mutated": False,
            "large_ED_performed": False,
            "sealed_checkpoint_replay_only": True,
            "energy_or_physics_win_claimed": False,
        },
        "normalization_contract": {
            "energy": "E/N",
            "energy_mcse": "total-H MCSE divided by N",
            "variance": "Var(H)/N",
            "variance_mcse": "total-H variance-contribution MCSE divided by N",
            "structure_factor": "N^-1 sum_ij <S_i dot S_j> exp(iq(ri-rj))",
        },
        "cases": [],
        "claim_boundary": POST_ANALYSIS_CLAIM_BOUNDARY,
    }
    atomic_json(result_path, payload)
    try:
        training_config = training_payload["config"]
        case_specs = {
            str(case["case_id"]): case for case in training_config["cases"]
        }
        arm_specs = {
            str(arm["arm_id"]): arm for arm in training_config["arms"]
        }
        last_update = int(spec["updates"][-1])
        for case_payload in training_payload["cases"]:
            case_spec = case_specs[str(case_payload["case_id"])]
            geometry = str(case_payload["geometry"])
            analyzed_case: dict[str, object] = {
                "case_id": case_payload["case_id"],
                "geometry": geometry,
                "rows": int(case_payload["rows"]),
                "columns": int(case_payload["columns"]),
                "n_sites": int(case_payload["n_sites"]),
                "j2_over_j1": float(case_payload["j2_over_j1"]),
                "bonds_sha256": case_payload["bonds_sha256"],
                "translation_group_sha256": case_payload[
                    "translation_group_sha256"
                ],
                "trials": [],
            }
            for training_trial in case_payload["trials"]:
                arm_spec = arm_specs[str(training_trial["arm_id"])]
                analyzed = _analyze_trial(
                    training_payload=training_payload,
                    training_path=training_path,
                    case_payload=case_payload,
                    case_spec=case_spec,
                    arm_spec=arm_spec,
                    training_trial=training_trial,
                    runtime_baseline=baselines[geometry],
                    analysis_spec=spec,
                )
                analyzed["trial_payload_sha256"] = trial_payload_sha256(analyzed)
                analyzed_case["trials"].append(analyzed)
                atomic_json(result_path, payload)
                final = analyzed["replay_updates"][-1]
                print(
                    f"post-analysis {case_payload['case_id']} {analyzed['arm_id']}: "
                    f"E/N={final['energy_E_per_site']['mean_real']:.8f} "
                    f"Var(H)/N={final['variance_H_per_site']['variance_H_per_site']:.6f} "
                    f"replay_max={analyzed['largest_registered_summary_replay_discrepancy']:.2e}",
                    flush=True,
                )
            analyzed_case["same_schedule_SU2_vs_U1_step20"] = _case_contrast(
                analyzed_case, last_update
            )
            payload["cases"].append(analyzed_case)
            atomic_json(result_path, payload)
        manifest = embedded_trial_manifest(payload)
        payload["trial_manifest"] = manifest
        payload["trial_manifest_sha256"] = canonical_sha256(manifest)
        payload["all_post_analysis_gates_passed"] = all(
            bool(trial["all_post_analysis_gates_passed"])
            for case in payload["cases"]
            for trial in case["trials"]
        )
        payload["status"] = "complete"
        payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(result_path, payload)
        return payload
    except BaseException as error:
        payload["status"] = "failed"
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(result_path, payload)
        raise


__all__ = ["run_spin_sampled_training_post_analysis"]
