"""Exact-Born estimator preflight for physical-input spin Clebsch trees.

This is an oracle test, not scalable VMC.  For ``N <= 16`` it enumerates the
common fixed-``M2=0`` support, draws i.i.d. samples from the exact variational
Born distribution, and compares sampled energy and VMC-gradient estimates to
their exhaustive references.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import coo_matvec, reynolds_project_state
from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
    embedded_trial_manifest,
    paired_initialization_audit,
    pytree_sha256,
    trial_payload_sha256,
)
from su2_nqs.frontier.provenance import (
    numerical_runtime_identity,
    transitive_local_source_manifest,
)

from projects.clebsch_tree_nqs.workflows import spin


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
ALLOWED_BACKBONES = {"u1", "su2"}
ALLOWED_PROJECTIONS = {"none", "translation"}


def _validate_spec(spec: dict[str, object]) -> None:
    """Fail closed before allocating an exact support."""

    if spec.get("schema_version") != 2 or spec.get("domain") != "spin":
        raise ValueError("sampled preflight requires schema_version=2 and domain='spin'")
    if spec.get("task") != "exact_born_sampled_estimator_preflight":
        raise ValueError("config task is not the exact-Born sampled estimator preflight")
    sampling = spec.get("sampling")
    if not isinstance(sampling, dict):
        raise TypeError("sampling must be an object")
    counts = sampling.get("sample_counts")
    if (
        not isinstance(counts, list)
        or not counts
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in counts
        )
        or counts != sorted(set(counts))
    ):
        raise ValueError("sample_counts must be unique increasing integers >= 2")
    replicates = sampling.get("replicates")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("sampling.replicates must be an integer at least two")
    if isinstance(sampling.get("seed"), bool) or not isinstance(sampling.get("seed"), int):
        raise TypeError("sampling.seed must be an integer")
    model_seeds = spec.get("model_seeds")
    if (
        not isinstance(model_seeds, list)
        or not model_seeds
        or any(isinstance(value, bool) or not isinstance(value, int) for value in model_seeds)
    ):
        raise ValueError("model_seeds must be a nonempty integer list")
    arms = spec.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("arms must be a nonempty list")
    arm_ids: set[str] = set()
    for arm in arms:
        if not isinstance(arm, dict):
            raise TypeError("every arm must be an object")
        arm_id = str(arm.get("arm_id", ""))
        if not arm_id or arm_id in arm_ids:
            raise ValueError("arm_id values must be nonempty and unique")
        arm_ids.add(arm_id)
        if str(arm.get("backbone")) not in ALLOWED_BACKBONES:
            raise ValueError("preflight backbones are restricted to u1 and su2")
        if str(arm.get("projection")) not in ALLOWED_PROJECTIONS:
            raise ValueError("preflight projections are restricted to none and translation")
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("every case must be an object")
        n_sites = int(case["rows"]) * int(case["columns"])
        if n_sites > 16:
            raise ValueError("exact-Born preflight is restricted to N <= 16")
        if n_sites % 2 or str(case.get("support")) != "fixed_m0":
            raise ValueError("preflight cases require even N and fixed_m0 support")
        selected = case.get("arm_ids")
        if not isinstance(selected, list) or not selected:
            raise ValueError("each case needs a nonempty arm_ids list")
        unknown = set(map(str, selected)) - arm_ids
        if unknown:
            raise ValueError(f"case references unknown arms: {sorted(unknown)}")


def _source_manifest() -> dict[str, str]:
    return transitive_local_source_manifest(
        (Path(__file__).resolve(),),
        root=ROOT,
        package_prefixes=("su2_nqs", "projects.clebsch_tree_nqs"),
    )


def _portable_config_path(config_path: Path) -> str:
    resolved = config_path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return f"external/{resolved.name}"


def _replicate_seed(
    base_seed: int,
    *,
    case_id: str,
    model_seed: int,
    sample_count: int,
    replicate: int,
) -> int:
    digest = canonical_sha256(
        {
            "base_seed": int(base_seed),
            "case_id": case_id,
            "model_seed": int(model_seed),
            "sample_count": int(sample_count),
            "replicate": int(replicate),
        }
    )
    return int(digest[:16], 16)


def _flat_gradient(gradient: object) -> np.ndarray:
    flat, _ = ravel_pytree(gradient)
    return np.asarray(flat, dtype=np.float64)


def _summarize_replicates(
    records: list[dict[str, object]],
    gradients: list[np.ndarray],
    *,
    exact_energy: float,
    exact_variance: float,
    exact_gradient: np.ndarray,
    sample_count: int,
    n_sites: int,
) -> dict[str, object]:
    energies = np.asarray([row["energy"] for row in records], dtype=np.float64)
    errors = energies - exact_energy
    exact_norm = float(np.linalg.norm(exact_gradient))
    stacked = np.stack(gradients)
    mean_gradient = np.mean(stacked, axis=0)
    debiased_mean_gradient = mean_gradient * sample_count / (sample_count - 1)
    relative_errors = np.linalg.norm(stacked - exact_gradient[None, :], axis=1) / exact_norm
    cosines = np.asarray([row["gradient_cosine"] for row in records], dtype=np.float64)
    predicted_rmse = float(np.sqrt(max(exact_variance, 0.0) / sample_count))
    observed_rmse = float(np.sqrt(np.mean(errors**2)))
    return {
        "sample_count": int(sample_count),
        "replicate_count": len(records),
        "energy_bias": float(np.mean(errors)),
        "energy_bias_per_site": float(np.mean(errors) / n_sites),
        "energy_rmse": observed_rmse,
        "energy_rmse_per_site": observed_rmse / n_sites,
        "iid_energy_rmse_prediction": predicted_rmse,
        "energy_rmse_over_iid_prediction": (
            None if predicted_rmse == 0.0 else observed_rmse / predicted_rmse
        ),
        "gradient_mean_bias_relative_norm": float(
            np.linalg.norm(mean_gradient - exact_gradient) / exact_norm
        ),
        "gradient_mean_debiased_bias_relative_norm": float(
            np.linalg.norm(debiased_mean_gradient - exact_gradient) / exact_norm
        ),
        "gradient_rmse_relative_norm": float(np.sqrt(np.mean(relative_errors**2))),
        "gradient_cosine_mean": float(np.mean(cosines)),
        "gradient_cosine_minimum": float(np.min(cosines)),
        "finite_sample_covariance_shrinkage": 1.0 - 1.0 / sample_count,
        "replicates": records,
    }


def _run_trial(
    arm: dict[str, object],
    *,
    model_seed: int,
    spec: dict[str, object],
    data: dict[str, object],
    case_id: str,
) -> dict[str, object]:
    backbone = str(arm["backbone"])
    projection = str(arm["projection"])
    if backbone not in ALLOWED_BACKBONES or projection not in ALLOWED_PROJECTIONS:
        raise ValueError("unsupported preflight arm")
    if projection not in data["groups"]:
        raise ValueError(f"projection {projection!r} is absent on this finite graph")
    model_spec = dict(spec["model"])
    model_spec.update(dict(arm.get("model", {})))
    config, params, amplitude_fn, parameter_count = spin._initialize(
        backbone, seed=model_seed, model=model_spec, data=data
    )
    basis = jnp.asarray(data["ordered_basis"], dtype=jnp.int32)
    projector_indices = jnp.asarray(data["tables"][projection], dtype=jnp.int32)
    h_rows = jnp.asarray(data["h_rows"], dtype=jnp.int32)
    h_columns = jnp.asarray(data["h_columns"], dtype=jnp.int32)
    h_values = jnp.asarray(data["h_values"], dtype=jnp.float64)

    def amplitudes(candidate):
        raw = amplitude_fn(candidate, basis, config)
        return reynolds_project_state(raw, projector_indices)

    def state_action(candidate):
        state = amplitudes(candidate)
        return state, coo_matvec(h_rows, h_columns, h_values, state)

    def exact_loss(candidate):
        state, action = state_action(candidate)
        return jnp.real(jnp.vdot(state, action) / jnp.vdot(state, state))

    exact_value_and_gradient = jax.jit(jax.value_and_grad(exact_loss))
    started = time.perf_counter()
    exact_energy_device, exact_gradient_tree = exact_value_and_gradient(params)
    jax.block_until_ready(exact_energy_device)
    exact_reference_seconds = time.perf_counter() - started
    exact_energy = float(exact_energy_device)
    exact_gradient = _flat_gradient(exact_gradient_tree)
    exact_gradient_norm = float(np.linalg.norm(exact_gradient))
    if not np.isfinite(exact_gradient_norm) or exact_gradient_norm <= 1.0e-14:
        raise RuntimeError("exhaustive gradient is numerically zero; choose another preflight state")

    state_device, action_device = jax.jit(state_action)(params)
    state = np.asarray(state_device, dtype=np.complex128)
    action = np.asarray(action_device, dtype=np.complex128)
    norm = float(np.vdot(state, state).real)
    if not np.isfinite(norm) or norm <= 1.0e-28:
        raise FloatingPointError("preflight state has zero or nonfinite norm")
    probabilities = np.abs(state) ** 2 / norm
    probabilities = probabilities / probabilities.sum()
    if np.any(~np.isfinite(probabilities)) or abs(float(probabilities.sum()) - 1.0) > 1.0e-12:
        raise FloatingPointError("preflight Born probabilities are invalid")
    residual = action - exact_energy * state
    exact_variance = float(np.vdot(residual, residual).real / norm)
    positive = probabilities > 0.0
    local_energies = np.zeros_like(state)
    local_energies[positive] = action[positive] / state[positive]
    imaginary_maximum = float(np.max(np.abs(local_energies[positive].imag)))
    born_reconstructed_energy = float(
        np.sum(probabilities[positive] * local_energies[positive]).real
    )
    if abs(born_reconstructed_energy - exact_energy) > 2.0e-11:
        raise RuntimeError("Born local-energy mean disagrees with exhaustive energy")

    def sampled_objective(candidate, sample_indices):
        candidate_state, candidate_action = state_action(candidate)
        centers = candidate_state[sample_indices]
        local = candidate_action[sample_indices] / centers
        mean_energy = jnp.mean(local)
        centered = jax.lax.stop_gradient(local - mean_energy)
        logs = jnp.log(centers.astype(jnp.complex128))
        surrogate = 2.0 * jnp.real(jnp.mean(jnp.conj(logs) * centered))
        loss = (
            jax.lax.stop_gradient(jnp.real(mean_energy))
            + surrogate
            - jax.lax.stop_gradient(surrogate)
        )
        auxiliary = jnp.stack(
            (jnp.real(mean_energy), jnp.max(jnp.abs(jnp.imag(local))))
        )
        return loss, auxiliary

    sample_summaries = []
    sampling_spec = spec["sampling"]
    for sample_count_value in sampling_spec["sample_counts"]:
        sample_count = int(sample_count_value)
        sampled_value_and_gradient = jax.jit(
            jax.value_and_grad(sampled_objective, has_aux=True)
        )
        records: list[dict[str, object]] = []
        gradients: list[np.ndarray] = []
        count_started = time.perf_counter()
        for replicate in range(int(sampling_spec["replicates"])):
            sampling_seed = _replicate_seed(
                int(sampling_spec["seed"]),
                case_id=case_id,
                model_seed=model_seed,
                sample_count=sample_count,
                replicate=replicate,
            )
            rng = np.random.default_rng(sampling_seed)
            sample_indices = rng.choice(
                len(probabilities), size=sample_count, replace=True, p=probabilities
            ).astype(np.int32)
            (_, auxiliary), gradient_tree = sampled_value_and_gradient(
                params, jnp.asarray(sample_indices)
            )
            jax.block_until_ready(auxiliary)
            gradient = _flat_gradient(gradient_tree)
            gradient_norm = float(np.linalg.norm(gradient))
            cosine = float(
                np.dot(gradient, exact_gradient)
                / max(gradient_norm * exact_gradient_norm, np.finfo(np.float64).tiny)
            )
            energy = float(auxiliary[0])
            record = {
                "replicate": replicate,
                "sampling_seed": sampling_seed,
                "energy": energy,
                "energy_error": energy - exact_energy,
                "maximum_local_energy_imaginary_abs": float(auxiliary[1]),
                "gradient_cosine": cosine,
                "gradient_relative_error": float(
                    np.linalg.norm(gradient - exact_gradient) / exact_gradient_norm
                ),
                "gradient_norm": gradient_norm,
            }
            if not all(
                np.isfinite(float(record[name]))
                for name in (
                    "energy",
                    "energy_error",
                    "maximum_local_energy_imaginary_abs",
                    "gradient_cosine",
                    "gradient_relative_error",
                    "gradient_norm",
                )
            ):
                raise FloatingPointError(f"nonfinite sampled preflight record: {record}")
            records.append(record)
            gradients.append(gradient)
        summary = _summarize_replicates(
            records,
            gradients,
            exact_energy=exact_energy,
            exact_variance=exact_variance,
            exact_gradient=exact_gradient,
            sample_count=sample_count,
            n_sites=int(data["n_sites"]),
        )
        summary["compile_and_replicates_seconds"] = time.perf_counter() - count_started
        sample_summaries.append(summary)

    raw_initial = np.asarray(amplitude_fn(params, basis, config), dtype=np.float64)
    initialization_pair_id = canonical_sha256(
        {
            "backbone": backbone,
            "seed": int(model_seed),
            "model": model_spec,
            "basis_sha256": str(data["basis_sha256"]),
            "site_order": list(data["site_order"]),
        }
    )
    return {
        "arm_id": str(arm["arm_id"]),
        "backbone": backbone,
        "projection": projection,
        "state_stage": "random_initialization",
        "seed": int(model_seed),
        "feature_dim": int(model_spec["feature_dim"]),
        "parameter_count": parameter_count,
        "group_order": len(data["groups"][projection]),
        "initial_params_sha256": pytree_sha256(params),
        "initial_raw_amplitudes_sha256": array_sha256(raw_initial),
        "initialization_pair_id": initialization_pair_id,
        "metrics": {
            "exact_variational_energy": exact_energy,
            "exact_variational_energy_per_site": exact_energy / int(data["n_sites"]),
            "exact_local_energy_variance": exact_variance,
            "born_reconstructed_energy_error": born_reconstructed_energy - exact_energy,
            "exact_gradient_norm": exact_gradient_norm,
            "exact_local_energy_imaginary_abs_maximum": imaginary_maximum,
            "born_support_size": int(np.count_nonzero(positive)),
            "born_effective_support": float(1.0 / np.sum(probabilities**2)),
            "sample_summaries": sample_summaries,
        },
        "exact_reference_compile_and_evaluate_seconds": exact_reference_seconds,
        "estimator_contract": {
            "sampling": "iid_with_replacement_from_exact_variational_Born_distribution",
            "energy": "sample_mean_local_energy",
            "gradient": "standard_real_parameter_centered_log_derivative_surrogate",
            "gradient_finite_sample_bias": (
                "sample covariance uses divisor M; expected multiplicative shrinkage 1-1/M"
            ),
            "mcmc_or_autoregressive_sampler_tested": False,
        },
    }


def run_spin_sampled_preflight(
    config_path: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run or reuse one immutable exact-Born spin estimator preflight."""

    spec = json.loads(config_path.read_text())
    _validate_spec(spec)
    config_hash = canonical_sha256(spec)
    manifest = _source_manifest()
    manifest_hash = canonical_sha256(manifest)
    runtime_identity = numerical_runtime_identity()
    runtime_hash = canonical_sha256(runtime_identity)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists() and not overwrite:
        existing = json.loads(result_path.read_text())
        expected = (config_hash, manifest_hash, runtime_hash, "complete")
        observed = (
            existing.get("config_sha256"),
            existing.get("source_manifest_sha256"),
            existing.get("runtime_identity_sha256"),
            existing.get("status"),
        )
        if observed != expected:
            raise RuntimeError(f"refusing stale or incomplete preflight: {result_path}")
        if existing.get("trial_manifest_sha256") != canonical_sha256(
            embedded_trial_manifest(existing)
        ):
            raise RuntimeError(f"refusing preflight with broken trial seal: {result_path}")
        print(f"reused immutable complete preflight: {result_path}", flush=True)
        return existing

    payload: dict[str, Any] = {
        "schema_version": 1,
        "domain": "spin",
        "task": "exact_born_sampled_estimator_preflight",
        "status": "running",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": _portable_config_path(config_path),
        "config_sha256": config_hash,
        "config": spec,
        "source_sha256": manifest,
        "source_manifest_sha256": manifest_hash,
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_hash,
        "jax_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "route_boundary": {
            "historical_project_imports": False,
            "direct_fusion_path_used": False,
            "hamiltonian_diagonalization_performed": False,
            "exact_born_oracle_used": True,
            "scalable_sampler_claim": False,
        },
        "cases": [],
    }
    arms_by_id = {str(arm["arm_id"]): arm for arm in spec["arms"]}
    for case_spec in spec["cases"]:
        data = spin._prepare_case(case_spec, "fixed_m0")
        case = {
            "case_id": str(case_spec["case_id"]),
            "geometry": data["geometry"],
            "rows": data["rows"],
            "columns": data["columns"],
            "n_sites": data["n_sites"],
            "j2_over_j1": data["j2_over_j1"],
            "basis_size": len(data["basis"]),
            "hamiltonian_nnz": int(data["hamiltonian"].nnz),
            "site_order": list(data["site_order"]),
            "group_orders": {name: len(group) for name, group in data["groups"].items()},
            "exact_energy_anchor": data["exact_energy"],
            "anchor_provenance": data["anchor_provenance"],
            "ground_anchor_used_by_estimator": False,
            "basis_sha256": data["basis_sha256"],
            "bonds_sha256": data["bonds_sha256"],
            "hamiltonian_coo_sha256": data["hamiltonian_coo_sha256"],
            "group_sha256": data["group_sha256"],
            "anchor_sha256": data["anchor_sha256"],
            "trials": [],
        }
        for arm_id in case_spec["arm_ids"]:
            arm = arms_by_id[str(arm_id)]
            for model_seed in spec["model_seeds"]:
                trial = _run_trial(
                    arm,
                    model_seed=int(model_seed),
                    spec=spec,
                    data=data,
                    case_id=str(case_spec["case_id"]),
                )
                trial["trial_payload_sha256"] = trial_payload_sha256(trial)
                case["trials"].append(trial)
                print(
                    f"preflight {case['case_id']} {trial['arm_id']} seed={model_seed}: "
                    f"E={trial['metrics']['exact_variational_energy']:.10f}",
                    flush=True,
                )
                atomic_json(result_path, payload)
        case["paired_initialization_audit"] = paired_initialization_audit(case["trials"])
        payload["cases"].append(case)
        atomic_json(result_path, payload)
    manifest_rows = embedded_trial_manifest(payload)
    payload["trial_manifest"] = manifest_rows
    payload["trial_manifest_sha256"] = canonical_sha256(manifest_rows)
    payload["status"] = "complete"
    atomic_json(result_path, payload)
    return payload


__all__ = ["run_spin_sampled_preflight"]
