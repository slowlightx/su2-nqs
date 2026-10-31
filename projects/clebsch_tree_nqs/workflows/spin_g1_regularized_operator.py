"""Two-stage N=36 cross-fitted scalar-operator experiment.

Development selects one ridge without creating an audit sample.  The separate
audit tier can only load a sealed development artifact whose preregistered
cross-fit gate passed; it then evaluates frozen coefficients on a newly seeded
sample exactly once.  Both tiers use the same reusable JAX/GPU scalar-word
kernel and an exactly nested ``I,H,H^2`` two-Lanczos control.
"""

from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time

import jax
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_g1_operator_vit import (
    _paired_chain_difference,
)
from su2_nqs.benchmarks.crossfit_regularized_operator import (
    crossfit_select_ridge,
    krylov_control_fit,
)
from su2_nqs.benchmarks.operator_vit_sampled import (
    carrier_standardization_for_labels,
    centered_scalar_component_matrices,
    krylov_control_basis,
    scalar_component_words,
)
from su2_nqs.benchmarks.operator_word_jax import (
    build_triangular_a1_word_action_plan,
    evaluate_fixed_capacity_scalar_word_kernel,
    make_fixed_capacity_scalar_word_kernel,
)
from su2_nqs.benchmarks.triangular_induced_parton import (
    build_n36_pi_zero_flux_orbit,
    projected_pi_zero_flux_a1_amplitudes,
    projected_pi_zero_flux_a1_amplitudes_jax,
)
from su2_nqs.frontier.artifacts import (
    array_sha256,
    atomic_json,
    canonical_sha256,
)
from su2_nqs.frontier.provenance import numerical_runtime_identity
from su2_nqs.operators.triangular_a1_scalar import (
    TRIANGULAR_A1_SCALAR_LABELS,
)
from su2_nqs.vmc.fixed_wavefunction_mcmc import sample_fixed_wavefunction_exchange


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
TASK = "spin_g1_regularized_operator"
LABELS = TRIANGULAR_A1_SCALAR_LABELS[:3]
RIDGE_GRID = (
    0.0,
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    3.0e-1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1.0e3,
    3.0e3,
    1.0e4,
)


def _ratio_summary(
    state: np.ndarray,
    action: np.ndarray,
    *,
    chain_count: int,
    n_sites: int,
) -> dict[str, object]:
    """Self-normalized energy/variance with delete-one-chain uncertainty."""

    indices = np.arange(len(state), dtype=np.int64)

    def energy(selected: np.ndarray) -> float:
        numerator = np.sum(np.conj(state[selected]) * action[selected])
        denominator = np.sum(np.abs(state[selected]) ** 2)
        return float(np.real(numerator / denominator) / n_sites)

    estimate = energy(indices)
    leave_one_out = []
    for chain in range(chain_count):
        keep = indices[indices % chain_count != chain]
        leave_one_out.append(energy(keep))
    center = float(np.mean(leave_one_out))
    standard_error = float(
        np.sqrt(
            (chain_count - 1.0)
            / chain_count
            * np.sum((np.asarray(leave_one_out) - center) ** 2)
        )
    )
    probability = np.abs(state) ** 2
    normalized = probability / np.sum(probability)
    total_energy = estimate * n_sites
    variance_total = float(
        np.sum(np.abs(action - total_energy * state) ** 2) / np.sum(probability)
    )
    return {
        "energy_per_site": estimate,
        "jackknife_standard_error_per_site": standard_error,
        "variance_total": variance_total,
        "importance_ess_fraction": float(
            1.0 / np.sum(normalized**2) / len(normalized)
        ),
        "maximum_normalized_weight": float(np.max(normalized)),
        "leave_one_chain_out_energy_per_site": leave_one_out,
    }


def _words() -> tuple[tuple[str, ...], ...]:
    components = scalar_component_words(LABELS)
    return tuple(
        dict.fromkeys(
            (
                *components,
                *((label, *word) for label in LABELS[:2] for word in components),
            )
        )
    )


def _validate(spec: dict[str, object]) -> None:
    if (spec.get("schema_version"), spec.get("domain"), spec.get("task")) != (
        1,
        "spin",
        TASK,
    ):
        raise ValueError("invalid regularized G1 identity")
    if spec.get("phase") not in {"development", "untouched_audit"}:
        raise ValueError("phase must be development or untouched_audit")
    if not np.isfinite(float(spec.get("j2_over_j1", np.nan))):
        raise ValueError("j2_over_j1 must be finite")
    sampling = spec.get("sampling")
    kernel = spec.get("kernel")
    if not isinstance(sampling, dict) or not isinstance(kernel, dict):
        raise TypeError("sampling and kernel must be objects")
    for name in (
        "chain_count",
        "burn_sweeps",
        "measurement_blocks",
        "decorrelation_sweeps",
        "amplitude_batch_size",
    ):
        if int(sampling.get(name, 0)) < 1:
            raise ValueError(f"sampling.{name} must be positive")
    for name in ("terminal_capacity", "entry_capacity"):
        if int(kernel.get(name, 0)) < 1:
            raise ValueError(f"kernel.{name} must be positive")
    if int(kernel["terminal_capacity"]) % int(sampling["amplitude_batch_size"]):
        raise ValueError("terminal_capacity must be a batch-size multiple")
    if spec["phase"] == "untouched_audit" and not isinstance(
        spec.get("development_artifact"), str
    ):
        raise ValueError("untouched_audit requires development_artifact")


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _build_plan_worker(
    state: np.ndarray, words: tuple[tuple[str, ...], ...]
):
    """Spawn-safe host plan builder used to keep the GPU feed queue full."""

    return build_triangular_a1_word_action_plan(
        np.asarray(state, dtype=np.int8)[None, :], ((6, 0), (0, 6)), words
    )


def _raw_dataset(
    states: np.ndarray,
    *,
    output: Path,
    prefix: str,
    words: tuple[tuple[str, ...], ...],
    amplitude_batch_size: int,
    terminal_capacity: int,
    entry_capacity: int,
    orbit,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    def amplitude(configurations):
        return projected_pi_zero_flux_a1_amplitudes_jax(
            configurations, orbit=orbit
        )

    kernel = make_fixed_capacity_scalar_word_kernel(
        amplitude,
        n_sites=36,
        amplitude_batch_size=amplitude_batch_size,
        terminal_capacity=terminal_capacity,
        entry_capacity=entry_capacity,
        output_count=len(words),
    )
    rows: list[np.ndarray] = []
    ledger: list[dict[str, object]] = []
    checked_states = np.asarray(states, dtype=np.int8)
    worker_count = int(os.environ.get("NQS_PLAN_WORKERS", "1"))
    if worker_count < 1:
        raise ValueError("NQS_PLAN_WORKERS must be positive")
    futures = {}
    executor = None
    if worker_count > 1:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
        )
        for index, state in enumerate(checked_states):
            cache = output / "component_cache" / f"{prefix}_{index:05d}.npz"
            if not cache.is_file():
                futures[index] = executor.submit(_build_plan_worker, state, words)
    for index, state in enumerate(checked_states):
        cache = output / "component_cache" / f"{prefix}_{index:05d}.npz"
        state_hash = array_sha256(state)
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as stored:
                cached_state = np.asarray(stored["state"], dtype=np.int8)
                actions = np.asarray(stored["actions"], dtype=np.complex128)
            if array_sha256(cached_state) != state_hash or actions.shape != (
                1,
                len(words),
            ):
                raise RuntimeError(f"invalid component cache: {cache}")
            row = {"cache_reused": True}
        else:
            started = time.perf_counter()
            if executor is None:
                plan, plan_row = _build_plan_worker(state, words)
            else:
                plan, plan_row = futures[index].result()
            actions, kernel_row = evaluate_fixed_capacity_scalar_word_kernel(
                kernel, plan
            )
            _atomic_npz(cache, state=state, actions=actions)
            row = {
                "cache_reused": False,
                "wall_seconds": time.perf_counter() - started,
                **plan_row,
                **kernel_row,
            }
        rows.append(actions[0])
        ledger.append({"sample_index": index, "state_sha256": state_hash, **row})
        print(
            f"regularized G1 {prefix} {index + 1}/{len(states)} "
            f"cache={row['cache_reused']}",
            flush=True,
        )
    if executor is not None:
        executor.shutdown(wait=True)
    return np.asarray(rows), ledger


def _matrices(
    raw: np.ndarray,
    words: tuple[tuple[str, ...], ...],
    *,
    coupling: float,
    means: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    components, action = centered_scalar_component_matrices(
        raw,
        words,
        labels=LABELS,
        hamiltonian_weights=(1.0, coupling, 0.0),
        means=means,
        scales=scales,
    )
    centers = raw[:, words.index(())]
    return components / centers[:, None], action / centers[:, None], centers


def _sample_and_actions(
    spec: dict[str, object], output: Path, *, prefix: str
) -> tuple[object, np.ndarray, list[dict[str, object]], tuple[tuple[str, ...], ...]]:
    sampling = dict(spec["sampling"])
    kernel = dict(spec["kernel"])
    orbit = build_n36_pi_zero_flux_orbit()

    def carrier(states: np.ndarray) -> np.ndarray:
        return projected_pi_zero_flux_a1_amplitudes(
            states,
            orbit=orbit,
            microbatch_size=int(sampling["amplitude_batch_size"]),
        ).amplitudes

    samples = sample_fixed_wavefunction_exchange(
        carrier,
        n_sites=36,
        n_up=18,
        chain_count=int(sampling["chain_count"]),
        burn_sweeps=int(sampling["burn_sweeps"]),
        measurement_blocks=int(sampling["measurement_blocks"]),
        decorrelation_sweeps=int(sampling["decorrelation_sweeps"]),
        seed=int(spec["seed"]),
    )
    words = _words()
    raw, ledger = _raw_dataset(
        samples.configurations,
        output=output,
        prefix=prefix,
        words=words,
        amplitude_batch_size=int(sampling["amplitude_batch_size"]),
        terminal_capacity=int(kernel["terminal_capacity"]),
        entry_capacity=int(kernel["entry_capacity"]),
        orbit=orbit,
    )
    return samples, raw, ledger, words


def _development(spec: dict[str, object], output: Path) -> dict[str, object]:
    samples, raw, ledger, words = _sample_and_actions(spec, output, prefix="develop")
    coupling = float(spec["j2_over_j1"])
    means, scales, standardization = carrier_standardization_for_labels(
        raw, words, LABELS
    )
    components, action, _ = _matrices(
        raw, words, coupling=coupling, means=means, scales=scales
    )
    control_basis = krylov_control_basis(
        hamiltonian_weights=(1.0, coupling, 0.0),
        means=means,
        scales=scales,
    )
    chain_count = int(spec["sampling"]["chain_count"])
    fit, crossfit = crossfit_select_ridge(
        components,
        action,
        chain_count=chain_count,
        fold_count=int(spec["crossfit_folds"]),
        ridge_grid=RIDGE_GRID,
        control_basis=control_basis,
    )
    control = krylov_control_fit(
        components, action, control_basis=control_basis
    )
    trial_state, trial_action = components @ fit.coefficients, action @ fit.coefficients
    control_state = components @ control.coefficients
    control_action = action @ control.coefficients
    paired = _paired_chain_difference(
        trial_state,
        trial_action,
        control_state,
        control_action,
        chain_count=chain_count,
    )
    selected = next(
        row for row in crossfit["grid"] if row["ridge"] == crossfit["selected_ridge"]
    )
    # This gate reads only grouped held-fold predictions.  The in-sample
    # summaries below are diagnostics and cannot license the audit.
    audit_licensed = bool(
        float(selected["mean_energy_difference"])
        + 2.0 * float(selected["fold_standard_error"])
        < 0.0
        and float(crossfit["selected_ridge"]) < max(RIDGE_GRID)
        and fit.resonance_norm_squared > 1.0e-8
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "phase": "development",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "model_case": {
            "hamiltonian": "triangular J1-J2 Heisenberg",
            "boundary": "6x6 periodic",
            "target_sector": "exact S=0, Gamma-A1",
            "carrier": "projected pi/0-flux",
            "tokens": list(LABELS),
        },
        "words": [list(word) for word in words],
        "standardization": {
            "means": means.tolist(),
            "scales": scales.tolist(),
            **standardization,
        },
        "control_basis": control_basis.tolist(),
        "crossfit": crossfit,
        "frozen_candidate": {
            "coefficients": fit.coefficients.tolist(),
            "ridge": crossfit["selected_ridge"],
            "non_krylov_norm_fraction": fit.resonance_norm_squared,
        },
        "frozen_two_lanczos": {"coefficients": control.coefficients.tolist()},
        "development_candidate": _ratio_summary(
            trial_state, trial_action, chain_count=chain_count, n_sites=36
        ),
        "development_two_lanczos": _ratio_summary(
            control_state, control_action, chain_count=chain_count, n_sites=36
        ),
        "development_paired_candidate_minus_two_lanczos": paired,
        "sample_identity": {
            "configurations_sha256": array_sha256(samples.configurations),
            "count": len(samples.configurations),
            "chain_count": chain_count,
            "measurement_blocks": int(spec["sampling"]["measurement_blocks"]),
        },
        "sampling_diagnostics": {
            "acceptance_fraction": samples.acceptance_fraction,
            "burn_acceptance_fraction": samples.burn_acceptance_fraction,
        },
        "execution": {
            "host_plan_worker_count": int(os.environ.get("NQS_PLAN_WORKERS", "1")),
            "jax_backend": jax.default_backend(),
        },
        "work_ledger": ledger,
        "decision": {
            "audit_licensed": audit_licensed,
            "criterion": (
                "selected_grouped_crossfit_mean_plus_2se_below_two_lanczos; "
                "finite_nontrivial_ridge"
            ),
            "untouched_audit_created": False,
        },
        "runtime_identity": numerical_runtime_identity(),
    }
    payload["result_sha256"] = canonical_sha256(payload)
    return payload


def _audit(spec: dict[str, object], output: Path) -> dict[str, object]:
    development_path = ROOT / str(spec["development_artifact"])
    development = json.loads(development_path.read_text())
    sealed = dict(development)
    recorded_hash = sealed.pop("result_sha256", None)
    if canonical_sha256(sealed) != recorded_hash:
        raise ValueError("development artifact hash mismatch")
    if (
        development.get("task") != TASK
        or development.get("phase") != "development"
        or not development["decision"]["audit_licensed"]
    ):
        raise RuntimeError("sealed development result does not license an audit")
    if float(development["config"]["j2_over_j1"]) != float(spec["j2_over_j1"]):
        raise ValueError("audit coupling differs from development")
    samples, raw, ledger, words = _sample_and_actions(spec, output, prefix="audit")
    if [list(word) for word in words] != development["words"]:
        raise RuntimeError("audit scalar-word basis changed")
    means = np.asarray(development["standardization"]["means"], dtype=np.float64)
    scales = np.asarray(development["standardization"]["scales"], dtype=np.float64)
    components, action, _ = _matrices(
        raw,
        words,
        coupling=float(spec["j2_over_j1"]),
        means=means,
        scales=scales,
    )
    candidate = np.asarray(
        development["frozen_candidate"]["coefficients"], dtype=np.float64
    )
    control = np.asarray(
        development["frozen_two_lanczos"]["coefficients"], dtype=np.float64
    )
    trial_state, trial_action = components @ candidate, action @ candidate
    control_state, control_action = components @ control, action @ control
    chain_count = int(spec["sampling"]["chain_count"])
    trial = _ratio_summary(
        trial_state, trial_action, chain_count=chain_count, n_sites=36
    )
    baseline = _ratio_summary(
        control_state, control_action, chain_count=chain_count, n_sites=36
    )
    paired = _paired_chain_difference(
        trial_state,
        trial_action,
        control_state,
        control_action,
        chain_count=chain_count,
    )
    resolved = bool(
        paired["difference_per_site"]
        + 2.0 * paired["chain_standard_error_per_site"]
        < 0.0
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain": "spin",
        "task": TASK,
        "phase": "untouched_audit",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": spec,
        "development_result_sha256": recorded_hash,
        "audit_candidate": trial,
        "audit_two_lanczos": baseline,
        "paired_candidate_minus_two_lanczos": paired,
        "sample_identity": {
            "configurations_sha256": array_sha256(samples.configurations),
            "count": len(samples.configurations),
            "chain_count": chain_count,
            "measurement_blocks": int(spec["sampling"]["measurement_blocks"]),
        },
        "sampling_diagnostics": {
            "acceptance_fraction": samples.acceptance_fraction,
            "burn_acceptance_fraction": samples.burn_acceptance_fraction,
        },
        "execution": {
            "host_plan_worker_count": int(os.environ.get("NQS_PLAN_WORKERS", "1")),
            "jax_backend": jax.default_backend(),
        },
        "work_ledger": ledger,
        "decision": {
            "energy_below_two_lanczos": paired["difference_per_site"] < 0.0,
            "energy_below_two_lanczos_at_2se": resolved,
            "variance_below_two_lanczos": trial["variance_total"]
            < baseline["variance_total"],
            "claim_licensed": resolved
            and trial["variance_total"] < baseline["variance_total"],
        },
        "runtime_identity": numerical_runtime_identity(),
    }
    payload["result_sha256"] = canonical_sha256(payload)
    return payload


def run_spin_g1_regularized_operator(
    config_path: Path, output: Path, *, overwrite: bool = False
) -> dict[str, object]:
    spec = json.loads(config_path.read_text())
    _validate(spec)
    if overwrite and output.is_dir():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = (
        _development(spec, output)
        if spec["phase"] == "development"
        else _audit(spec, output)
    )
    atomic_json(output / "results.json", payload)
    return payload


__all__ = ["TASK", "run_spin_g1_regularized_operator"]
