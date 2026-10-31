"""Post-seal analysis for the local spin PRL Pareto campaigns.

This module is deliberately outside the experiment workflow dependency graph.
It reads immutable result payloads, verifies their seals and registered design,
and computes resource frontiers without importing a model, Hamiltonian, or JAX.

The main campaign contains 384 trials. A companion U(1)-D6 campaign adds 32
longer trajectories as a horizon-extension diagnostic; an optional 32-trial
SU(2)-D6 campaign completes a time horizon that the first audit found to be
censored. Long runs replace only their corresponding D6 measured-time curves.
The parameter frontier always uses the preregistered 300-update main campaign.
Timing promotion is fail-closed unless every input artifact records a verified
machine-isolation audit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


BASE_TRIAL_COUNT = 384
LONG_TRIAL_COUNT = 32
TRANSLATION_WIDTHS = (2, 3, 4, 5, 6)
ENERGY_FLOOR = 1.0e-12
NEGATIVE_ENERGY_TOLERANCE = 1.0e-9
IDENTITY_TOLERANCE = 1.0e-10
DEFAULT_TIME_GRID_SIZE = 201


class AnalysisValidationError(ValueError):
    """Raised when a result cannot enter the paper-facing analysis."""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _trial_payload_sha256(trial: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {key: value for key, value in trial.items() if key != "trial_payload_sha256"}
    )


def _as_payload(source: str | Path | Mapping[str, object]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    return json.loads(Path(source).read_text())


def _finite(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise AnalysisValidationError(f"{name} must be finite, got {value!r}")
    return number


def _safe_log_error(value: object, *, name: str) -> tuple[float, bool]:
    error = _finite(value, name)
    if error < -NEGATIVE_ENERGY_TOLERANCE:
        raise AnalysisValidationError(
            f"{name} lies below the variational anchor: {error:.6e}"
        )
    censored = error <= ENERGY_FLOOR
    return math.log10(max(error, ENERGY_FLOOR)), censored


def _physical_key(case: Mapping[str, object]) -> tuple[object, ...]:
    return (
        str(case["geometry"]),
        int(case["rows"]),
        int(case["columns"]),
        float(case["j2_over_j1"]),
        str(case["support"]),
    )


def _case_map(payload: Mapping[str, object]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): dict(case) for case in payload["cases"]}  # type: ignore[index]


def _physical_case_map(
    payload: Mapping[str, object],
) -> dict[tuple[object, ...], dict[str, Any]]:
    result: dict[tuple[object, ...], dict[str, Any]] = {}
    for case in payload["cases"]:  # type: ignore[index]
        key = _physical_key(case)
        if key in result:
            raise AnalysisValidationError(f"duplicate physical case key: {key}")
        result[key] = dict(case)
    return result


def _validate_result_seal(
    payload: Mapping[str, object], *, expected_trials: int, label: str
) -> dict[str, object]:
    if payload.get("status") != "complete":
        raise AnalysisValidationError(
            f"{label} result is not sealed complete: {payload.get('status')!r}"
        )
    if payload.get("domain") != "spin":
        raise AnalysisValidationError(f"{label} result is not a spin campaign")
    if int(payload.get("schema_version", -1)) != 4:
        raise AnalysisValidationError(f"{label} result must use schema version 4")
    if int(payload.get("timing_schema_version", -1)) != 2:
        raise AnalysisValidationError(f"{label} result must use timing schema 2")

    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise AnalysisValidationError(f"{label} result has no embedded config")
    if payload.get("config_sha256") != _canonical_sha256(config):
        raise AnalysisValidationError(f"{label} config seal failed")
    source = payload.get("source_sha256")
    if payload.get("source_manifest_sha256") != _canonical_sha256(source):
        raise AnalysisValidationError(f"{label} source-manifest seal failed")
    runtime = payload.get("runtime_identity")
    if payload.get("runtime_identity_sha256") != _canonical_sha256(runtime):
        raise AnalysisValidationError(f"{label} runtime-identity seal failed")

    embedded: dict[str, str] = {}
    trial_count = 0
    for case in payload.get("cases", []):
        case_id = str(case["case_id"])
        for trial in case.get("trials", []):
            trial_count += 1
            stored = str(trial.get("trial_payload_sha256", ""))
            if stored != _trial_payload_sha256(trial):
                raise AnalysisValidationError(
                    f"{label} trial seal failed: {case_id}/{trial.get('arm_id')}/"
                    f"seed{trial.get('seed')}"
                )
            if trial.get("config_sha256") != payload.get("config_sha256"):
                raise AnalysisValidationError(f"{label} trial has wrong config seal")
            if trial.get("source_manifest_sha256") != payload.get(
                "source_manifest_sha256"
            ):
                raise AnalysisValidationError(f"{label} trial has wrong source seal")
            if trial.get("runtime_identity_sha256") != payload.get(
                "runtime_identity_sha256"
            ):
                raise AnalysisValidationError(f"{label} trial has wrong runtime seal")
            key = f"{case_id}/{trial['arm_id']}/seed{trial['seed']}"
            if key in embedded:
                raise AnalysisValidationError(f"duplicate trial identity: {key}")
            embedded[key] = stored

    if trial_count != expected_trials:
        raise AnalysisValidationError(
            f"{label} trial count is {trial_count}, expected {expected_trials}"
        )
    embedded = dict(sorted(embedded.items()))
    if payload.get("trial_manifest") != embedded:
        raise AnalysisValidationError(f"{label} embedded trial manifest differs")
    if payload.get("trial_manifest_sha256") != _canonical_sha256(embedded):
        raise AnalysisValidationError(f"{label} trial-manifest seal failed")
    return {"trial_count": trial_count, "trial_manifest_sha256": _canonical_sha256(embedded)}


def _validate_registered_grid(
    payload: Mapping[str, object], *, expected_trials: int, label: str
) -> None:
    config = payload["config"]  # type: ignore[index]
    seeds = tuple(int(seed) for seed in config["optimization"]["seeds"])
    if seeds != tuple(range(21, 29)):
        raise AnalysisValidationError(f"{label} seeds differ from 21..28")
    arms = {str(arm["arm_id"]): arm for arm in config["arms"]}
    cases = _case_map(payload)
    configured_cases = {str(case["case_id"]): case for case in config["cases"]}
    if set(cases) != set(configured_cases):
        raise AnalysisValidationError(f"{label} case ids differ from its config")

    observed: set[tuple[str, str, int]] = set()
    expected: set[tuple[str, str, int]] = set()
    for case_id, spec in configured_cases.items():
        case = cases[case_id]
        for field in (
            "geometry",
            "rows",
            "columns",
            "j2_over_j1",
            "support",
            "exact_energy_anchor",
        ):
            if case[field] != spec[field]:
                raise AnalysisValidationError(
                    f"{label} case {case_id} disagrees with config field {field}"
                )
        for arm_id in spec["arm_ids"]:
            if arm_id not in arms:
                raise AnalysisValidationError(f"{label} config references {arm_id!r}")
            for seed in seeds:
                expected.add((case_id, str(arm_id), seed))
        for trial in case["trials"]:
            key = (case_id, str(trial["arm_id"]), int(trial["seed"]))
            if key in observed:
                raise AnalysisValidationError(f"duplicate configured trial: {key}")
            observed.add(key)
            if key not in expected:
                raise AnalysisValidationError(f"unregistered trial: {key}")
            arm = arms[key[1]]
            if str(trial["backbone"]) != str(arm["backbone"]):
                raise AnalysisValidationError(f"backbone mismatch for {key}")
            if str(trial["projection"]) != str(arm["projection"]):
                raise AnalysisValidationError(f"projection mismatch for {key}")
            width = int(arm.get("model", {}).get("feature_dim", config["model"]["feature_dim"]))
            if int(trial["feature_dim"]) != width:
                raise AnalysisValidationError(f"feature-dimension mismatch for {key}")
    if observed != expected or len(expected) != expected_trials:
        missing = sorted(expected - observed)[:3]
        extra = sorted(observed - expected)[:3]
        raise AnalysisValidationError(
            f"{label} registered grid mismatch; missing={missing}, extra={extra}"
        )


def _validate_physics(
    payloads: Sequence[Mapping[str, object]], *, tolerance: float
) -> dict[str, object]:
    maximum = {
        "structure_factor_sum_error": 0.0,
        "uniform_s2_error": 0.0,
        "momentum_inversion_error": 0.0,
        "observable_alias_error": 0.0,
        "projection_closure_error": 0.0,
        "su2_spin_squared_expectation": 0.0,
        "su2_spin_squared_residual": 0.0,
        "energy_identity_error": 0.0,
    }
    minimum_structure_factor = math.inf
    checked = 0
    for payload in payloads:
        for case in payload["cases"]:  # type: ignore[index]
            rows = int(case["rows"])
            columns = int(case["columns"])
            n_sites = int(case["n_sites"])
            anchor = float(case["exact_energy_anchor"])
            geometry = str(case["geometry"])
            for trial in case["trials"]:
                checked += 1
                metrics = trial["metrics"]
                energy = _finite(metrics["energy"], "energy")
                error = _finite(metrics["energy_error_per_site"], "energy error/site")
                if error < -NEGATIVE_ENERGY_TOLERANCE:
                    raise AnalysisValidationError("variational energy lies below anchor")
                expected_error = (energy - anchor) / n_sites
                maximum["energy_identity_error"] = max(
                    maximum["energy_identity_error"], abs(error - expected_error)
                )
                if _finite(metrics["variance_per_site"], "variance/site") < -tolerance:
                    raise AnalysisValidationError("negative energy variance")
                closure = abs(_finite(metrics["projection_closure_error"], "closure"))
                maximum["projection_closure_error"] = max(
                    maximum["projection_closure_error"], closure
                )
                selected_weight = _finite(
                    metrics["selected_weight_of_raw"], "selected projected weight"
                )
                if not (-tolerance <= selected_weight <= 1.0 + tolerance):
                    raise AnalysisValidationError("projected weight lies outside [0,1]")

                s2 = _finite(metrics["spin_squared_expectation"], "<S^2>")
                s2_residual = _finite(metrics["spin_squared_residual"], "S2 residual")
                if s2 < -tolerance or s2_residual < -tolerance:
                    raise AnalysisValidationError("negative spin diagnostic")
                if str(trial["backbone"]) == "su2":
                    maximum["su2_spin_squared_expectation"] = max(
                        maximum["su2_spin_squared_expectation"], abs(s2)
                    )
                    maximum["su2_spin_squared_residual"] = max(
                        maximum["su2_spin_squared_residual"], abs(s2_residual)
                    )

                observables = metrics["physical_observables"]
                grid = np.asarray(observables["spin_structure_factor_grid"], dtype=float)
                if grid.shape != (rows, columns) or not np.all(np.isfinite(grid)):
                    raise AnalysisValidationError("invalid structure-factor grid")
                minimum_structure_factor = min(minimum_structure_factor, float(grid.min()))
                maximum["structure_factor_sum_error"] = max(
                    maximum["structure_factor_sum_error"],
                    abs(float(grid.sum()) - 0.75 * n_sites),
                )
                maximum["uniform_s2_error"] = max(
                    maximum["uniform_s2_error"], abs(float(grid[0, 0]) - s2 / n_sites)
                )
                inverted = np.empty_like(grid)
                for ky in range(rows):
                    for kx in range(columns):
                        inverted[ky, kx] = grid[(-ky) % rows, (-kx) % columns]
                maximum["momentum_inversion_error"] = max(
                    maximum["momentum_inversion_error"],
                    float(np.max(np.abs(grid - inverted))),
                )
                maximum["observable_alias_error"] = max(
                    maximum["observable_alias_error"],
                    abs(float(observables["uniform_structure_factor"]) - float(grid[0, 0])),
                )
                peak_index = tuple(int(value) for value in observables["peak_nonzero_momentum_index"])
                if len(peak_index) != 2 or not (
                    0 <= peak_index[0] < rows and 0 <= peak_index[1] < columns
                ):
                    raise AnalysisValidationError("invalid peak momentum index")
                nonzero = grid.copy()
                nonzero[0, 0] = -np.inf
                peak_value = float(observables["peak_nonzero_structure_factor"])
                maximum["observable_alias_error"] = max(
                    maximum["observable_alias_error"],
                    abs(peak_value - float(grid[peak_index])),
                    abs(peak_value - float(np.max(nonzero))),
                )
                if geometry == "square":
                    neel = float(grid[rows // 2, columns // 2])
                    stripe = float(max(grid[rows // 2, 0], grid[0, columns // 2]))
                    maximum["observable_alias_error"] = max(
                        maximum["observable_alias_error"],
                        abs(float(observables["neel_structure_factor"]) - neel),
                        abs(float(observables["stripe_structure_factor_max"]) - stripe),
                    )
                elif geometry == "triangular":
                    m_values = [grid[rows // 2, 0], grid[0, columns // 2], grid[rows // 2, columns // 2]]
                    maximum["observable_alias_error"] = max(
                        maximum["observable_alias_error"],
                        abs(float(observables["stripe_M_structure_factor_max"]) - float(max(m_values))),
                    )
                    if bool(observables["k_120_commensurate"]) or observables[
                        "order_120_K_structure_factor_max"
                    ] is not None:
                        raise AnalysisValidationError("4x4 triangular result falsely exposes K")

    for name, error in maximum.items():
        if error > tolerance:
            raise AnalysisValidationError(
                f"physical identity {name} failed: {error:.6e} > {tolerance:.1e}"
            )
    if minimum_structure_factor < -1.0e-12:
        raise AnalysisValidationError(
            f"negative structure factor: {minimum_structure_factor:.6e}"
        )
    return {
        "checked_trials": checked,
        "maximum_errors": {key: float(value) for key, value in maximum.items()},
        "minimum_structure_factor": float(minimum_structure_factor),
        "passed": True,
    }


def _trial_lookup(case: Mapping[str, object]) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for trial in case["trials"]:  # type: ignore[index]
        key = (str(trial["arm_id"]), int(trial["seed"]))
        if key in result:
            raise AnalysisValidationError(f"duplicate trial in case {case['case_id']}: {key}")
        result[key] = dict(trial)
    return result


def _validate_initializations(
    base: Mapping[str, object],
    long: Mapping[str, object],
    su2_long: Mapping[str, object] | None = None,
) -> dict[str, object]:
    projection_groups = 0
    cross_backbone_groups = 0
    for case in base["cases"]:  # type: ignore[index]
        grouped: dict[tuple[str, int, int], list[Mapping[str, object]]] = {}
        cross: dict[tuple[int, int], list[Mapping[str, object]]] = {}
        for trial in case["trials"]:
            grouped.setdefault(
                (str(trial["backbone"]), int(trial["feature_dim"]), int(trial["seed"])), []
            ).append(trial)
            if str(trial["projection"]) == "translation":
                cross.setdefault((int(trial["feature_dim"]), int(trial["seed"])), []).append(trial)
        for key, trials in grouped.items():
            projections = {str(trial["projection"]) for trial in trials}
            if len(projections) < 2:
                continue
            projection_groups += 1
            for field in (
                "initialization_pair_id",
                "initial_params_sha256",
                "initial_raw_amplitudes_sha256",
            ):
                values = {str(trial[field]) for trial in trials}
                if len(values) != 1:
                    raise AnalysisValidationError(
                        f"projection-only initialization mismatch {case['case_id']} {key}: {field}"
                    )
        for key, trials in cross.items():
            if {str(trial["backbone"]) for trial in trials} != {"u1", "su2"}:
                raise AnalysisValidationError(f"missing translation backbone at {key}")
            cross_backbone_groups += 1
            if len({str(trial["initialization_pair_id"]) for trial in trials}) != 2:
                raise AnalysisValidationError(
                    "U1/SU2 unexpectedly share an initialization-pair id"
                )

    base_by_physics = _physical_case_map(base)

    def validate_long_prefix(
        candidate: Mapping[str, object],
        *,
        base_arm: str,
        long_arm: str,
        label: str,
        allow_anchor_provenance_drift: bool = False,
    ) -> tuple[int, list[dict[str, object]]]:
        candidate_by_physics = _physical_case_map(candidate)
        if set(base_by_physics) != set(candidate_by_physics):
            raise AnalysisValidationError(
                f"base and {label} physical case grids differ"
            )
        matched = 0
        anchor_drifts: list[dict[str, object]] = []
        for key in sorted(base_by_physics):
            base_case = base_by_physics[key]
            candidate_case = candidate_by_physics[key]
            for field in (
                "n_sites",
                "basis_sha256",
                "bonds_sha256",
                "hamiltonian_coo_sha256",
            ):
                if base_case[field] != candidate_case[field]:
                    raise AnalysisValidationError(
                        f"base/{label} case mismatch for {field}: {key}"
                    )
            if base_case["anchor_sha256"] != candidate_case["anchor_sha256"]:
                if not allow_anchor_provenance_drift or float(
                    base_case["exact_energy_anchor"]
                ) != float(candidate_case["exact_energy_anchor"]):
                    raise AnalysisValidationError(
                        f"base/{label} anchor mismatch: {key}"
                    )
                anchor_drifts.append(
                    {
                        "physical_key": list(key),
                        "exact_energy_anchor": float(
                            base_case["exact_energy_anchor"]
                        ),
                        "base_anchor_sha256": str(base_case["anchor_sha256"]),
                        "long_anchor_sha256": str(candidate_case["anchor_sha256"]),
                        "base_provenance": str(base_case["anchor_provenance"]),
                        "long_provenance": str(candidate_case["anchor_provenance"]),
                        "numerical_anchor_and_hamiltonian_match": True,
                    }
                )
            base_lookup = _trial_lookup(base_case)
            candidate_lookup = _trial_lookup(candidate_case)
            for seed in range(21, 29):
                base_trial = base_lookup[(base_arm, seed)]
                long_trial = candidate_lookup[(long_arm, seed)]
                matched += 1
                for field in (
                    "parameter_count",
                    "initialization_pair_id",
                    "initial_params_sha256",
                    "initial_raw_amplitudes_sha256",
                ):
                    if base_trial[field] != long_trial[field]:
                        raise AnalysisValidationError(
                            f"base/{label} D6 initialization mismatch for "
                            f"{key}, seed {seed}: {field}"
                        )
                if not math.isclose(
                    float(base_trial["training"]["initial_loss"]),
                    float(long_trial["training"]["initial_loss"]),
                    rel_tol=1.0e-13,
                    abs_tol=1.0e-13,
                ):
                    raise AnalysisValidationError(
                        f"base/{label} initial losses differ"
                    )
                long_history = {
                    int(row["step"]): float(row["loss_before_update"])
                    for row in long_trial["training"]["history"]
                }
                for row in base_trial["training"]["history"]:
                    step = int(row["step"])
                    if step not in long_history or not math.isclose(
                        float(row["loss_before_update"]),
                        long_history[step],
                        rel_tol=1.0e-13,
                        abs_tol=1.0e-13,
                    ):
                        raise AnalysisValidationError(
                            f"{label} D6 trajectory does not reproduce base "
                            f"prefix at step {step}"
                        )
        return matched, anchor_drifts

    matched_long_trials, u1_anchor_drifts = validate_long_prefix(
        long,
        base_arm="u1_D6_translation",
        long_arm="u1_D6_translation_long",
        label="u1-long",
    )
    matched_su2_long_trials = 0
    if su2_long is not None:
        matched_su2_long_trials, su2_anchor_drifts = validate_long_prefix(
            su2_long,
            base_arm="su2_D6_translation",
            long_arm="su2_D6_translation_long",
            label="su2-long",
            allow_anchor_provenance_drift=True,
        )
    else:
        su2_anchor_drifts = []
    return {
        "projection_matched_initialization_groups": projection_groups,
        "cross_backbone_seed_label_groups": cross_backbone_groups,
        "base_long_matched_initializations": matched_long_trials,
        "base_su2_long_matched_initializations": matched_su2_long_trials,
        "u1_long_anchor_metadata_drifts": u1_anchor_drifts,
        "su2_long_anchor_metadata_drifts": su2_anchor_drifts,
        "u1_su2_initial_functions_are_paired": False,
        "passed": True,
    }


def validate_prl_results(
    base_result: str | Path | Mapping[str, object],
    long_result: str | Path | Mapping[str, object],
    su2_long_result: str | Path | Mapping[str, object] | None = None,
    *,
    identity_tolerance: float = IDENTITY_TOLERANCE,
) -> dict[str, object]:
    """Validate sealed Pareto/horizon campaigns and exact identities."""

    base = _as_payload(base_result)
    long = _as_payload(long_result)
    su2_long = None if su2_long_result is None else _as_payload(su2_long_result)
    base_seal = _validate_result_seal(base, expected_trials=BASE_TRIAL_COUNT, label="base")
    long_seal = _validate_result_seal(long, expected_trials=LONG_TRIAL_COUNT, label="long")
    _validate_registered_grid(base, expected_trials=BASE_TRIAL_COUNT, label="base")
    _validate_registered_grid(long, expected_trials=LONG_TRIAL_COUNT, label="long")
    su2_long_seal = None
    if su2_long is not None:
        su2_long_seal = _validate_result_seal(
            su2_long, expected_trials=LONG_TRIAL_COUNT, label="su2-long"
        )
        _validate_registered_grid(
            su2_long, expected_trials=LONG_TRIAL_COUNT, label="su2-long"
        )
    if base["source_manifest_sha256"] != long["source_manifest_sha256"]:
        raise AnalysisValidationError("base and long use different source manifests")
    if base["runtime_identity_sha256"] != long["runtime_identity_sha256"]:
        raise AnalysisValidationError("base and long use different numerical runtimes")
    if base.get("jax_backend") != long.get("jax_backend") or base.get(
        "devices"
    ) != long.get("devices"):
        raise AnalysisValidationError("base and long use different JAX devices")
    if base.get("timing_scope") != long.get("timing_scope"):
        raise AnalysisValidationError("base and long use different timing scopes")
    if su2_long is not None:
        for field in ("source_manifest_sha256", "runtime_identity_sha256"):
            if base[field] != su2_long[field]:
                raise AnalysisValidationError(
                    f"base and su2-long use different {field}"
                )
        if base.get("jax_backend") != su2_long.get("jax_backend") or base.get(
            "devices"
        ) != su2_long.get("devices"):
            raise AnalysisValidationError(
                "base and su2-long use different JAX devices"
            )
        if base.get("timing_scope") != su2_long.get("timing_scope"):
            raise AnalysisValidationError(
                "base and su2-long use different timing scopes"
            )
    base_config = base["config"]
    long_config = long["config"]
    for field in ("learning_rate", "record_every", "seeds"):
        if base_config["optimization"][field] != long_config["optimization"][field]:
            raise AnalysisValidationError(
                f"base and long optimizer contracts differ for {field}"
            )
    for field in ("maximum_intermediate_q", "initialization_scale"):
        if base_config["model"][field] != long_config["model"][field]:
            raise AnalysisValidationError(
                f"base and long model contracts differ for {field}"
            )
    if int(base_config["optimization"]["steps"]) != 300 or int(
        long_config["optimization"]["steps"]
    ) != 1200:
        raise AnalysisValidationError("base/long update budgets are not 300/1200")
    if su2_long is not None:
        su2_long_config = su2_long["config"]
        for field in ("learning_rate", "record_every", "seeds"):
            if base_config["optimization"][field] != su2_long_config["optimization"][field]:
                raise AnalysisValidationError(
                    f"base and su2-long optimizer contracts differ for {field}"
                )
        for field in ("maximum_intermediate_q", "initialization_scale"):
            if base_config["model"][field] != su2_long_config["model"][field]:
                raise AnalysisValidationError(
                    f"base and su2-long model contracts differ for {field}"
                )
        if int(su2_long_config["optimization"]["steps"]) != 600:
            raise AnalysisValidationError("su2-long update budget is not 600")
    payloads = (base, long) if su2_long is None else (base, long, su2_long)
    physics = _validate_physics(payloads, tolerance=identity_tolerance)
    initialization = _validate_initializations(base, long, su2_long)
    return {
        "passed": True,
        "base": base_seal,
        "long": long_seal,
        "su2_long": su2_long_seal,
        "physics": physics,
        "initialization": initialization,
    }


def _translation_by_width(
    case: Mapping[str, object], backbone: str
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for trial in case["trials"]:  # type: ignore[index]
        if str(trial["backbone"]) == backbone and str(trial["projection"]) == "translation":
            grouped.setdefault(int(trial["feature_dim"]), []).append(dict(trial))
    if set(grouped) != set(TRANSLATION_WIDTHS):
        raise AnalysisValidationError(
            f"{case['case_id']} {backbone} translation widths are {sorted(grouped)}"
        )
    for width, trials in grouped.items():
        trials.sort(key=lambda trial: int(trial["seed"]))
        if [int(trial["seed"]) for trial in trials] != list(range(21, 29)):
            raise AnalysisValidationError(
                f"{case['case_id']} {backbone} D{width} has wrong seeds"
            )
    return grouped


def _integration_weights(axis: np.ndarray) -> np.ndarray:
    if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0.0):
        raise AnalysisValidationError("integration axis must be strictly increasing")
    intervals = np.diff(axis)
    weights = np.zeros_like(axis)
    weights[0] = intervals[0] / 2.0
    weights[-1] = intervals[-1] / 2.0
    if axis.size > 2:
        weights[1:-1] = (intervals[:-1] + intervals[1:]) / 2.0
    return weights / float(axis[-1] - axis[0])


def _resource_effect(
    u1_values: np.ndarray,
    su2_values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    u1_frontier = np.min(np.median(u1_values, axis=1), axis=0)
    su2_frontier = np.min(np.median(su2_values, axis=1), axis=0)
    if not np.all(np.isfinite(u1_frontier)) or not np.all(np.isfinite(su2_frontier)):
        raise AnalysisValidationError("resource frontier has no available arm")
    difference = u1_frontier - su2_frontier
    average = float(np.dot(difference, weights))
    worst = float(np.min(difference))
    return u1_frontier, su2_frontier, average, worst


def exact_two_sample_permutation(
    u1_scores: Sequence[float], su2_scores: Sequence[float]
) -> dict[str, object]:
    """Exact one-sided 8+8 permutation test using the pairwise U statistic.

    Larger scores mean worse performance.  The alternative is therefore that
    a randomly drawn U(1) score exceeds a randomly drawn SU(2) score.
    """

    u1 = np.asarray(u1_scores, dtype=float)
    su2 = np.asarray(su2_scores, dtype=float)
    if u1.shape != (8,) or su2.shape != (8,) or not (
        np.all(np.isfinite(u1)) and np.all(np.isfinite(su2))
    ):
        raise ValueError("exact test requires two finite length-eight samples")

    def statistic(first: np.ndarray, second: np.ndarray) -> float:
        delta = first[:, None] - second[None, :]
        return float(np.count_nonzero(delta > 0.0) + 0.5 * np.count_nonzero(delta == 0.0))

    observed = statistic(u1, su2)
    combined = np.concatenate((u1, su2))
    exceedances = 0
    total = 0
    all_indices = np.arange(16)
    for selected_tuple in combinations(range(16), 8):
        selected = np.fromiter(selected_tuple, dtype=int, count=8)
        mask = np.ones(16, dtype=bool)
        mask[selected] = False
        candidate = statistic(combined[selected], combined[all_indices[mask]])
        exceedances += candidate >= observed - 1.0e-15
        total += 1
    return {
        "u_statistic": observed,
        "common_language_probability": observed / 64.0,
        "one_sided_p_value": exceedances / total,
        "enumerated_allocations": total,
        "alternative": "U1 integrated log-error score is larger than SU2",
    }


def _bootstrap_resource(
    u1_values: np.ndarray,
    su2_values: np.ndarray,
    weights: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    if resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    rng = np.random.default_rng(int(seed))
    averages = np.empty(resamples, dtype=float)
    worst = np.empty(resamples, dtype=float)
    cursor = 0
    while cursor < resamples:
        size = min(512, resamples - cursor)
        u_indices = rng.integers(0, 8, size=(size, 8))
        s_indices = rng.integers(0, 8, size=(size, 8))
        u_draws = np.take(u1_values, u_indices, axis=1)
        s_draws = np.take(su2_values, s_indices, axis=1)
        u_frontier = np.min(np.median(u_draws, axis=2), axis=0)
        s_frontier = np.min(np.median(s_draws, axis=2), axis=0)
        difference = u_frontier - s_frontier
        averages[cursor : cursor + size] = difference @ weights
        worst[cursor : cursor + size] = np.min(difference, axis=1)
        cursor += size

    def interval(values: np.ndarray) -> dict[str, object]:
        low, high = np.quantile(values, (0.025, 0.975))
        return {
            "percentile_95": [float(low), float(high)],
            "one_sided_95_lower": float(np.quantile(values, 0.05)),
        }

    return {
        "method": "independent_backbone_seed_label_cluster_within_case_percentile",
        "resamples": int(resamples),
        "seed": int(seed),
        "average_log10_advantage": interval(averages),
        "worst_log10_advantage": interval(worst),
    }


def _seed_scores(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    individual_frontier = np.min(values, axis=0)
    if not np.all(np.isfinite(individual_frontier)):
        raise AnalysisValidationError("a seed-specific frontier is unavailable")
    return individual_frontier @ weights


def _summary_from_resource(
    u1_values: np.ndarray,
    su2_values: np.ndarray,
    *,
    display_axis: np.ndarray,
    integration_axis: np.ndarray,
    weights: np.ndarray,
    u1_labels: Sequence[str],
    su2_labels: Sequence[str],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    u_frontier, s_frontier, average, worst = _resource_effect(
        u1_values, su2_values, weights
    )
    u_scores = _seed_scores(u1_values, weights)
    s_scores = _seed_scores(su2_values, weights)
    return {
        "budget_axis": [float(value) for value in display_axis],
        "integration_axis_log10": [float(value) for value in integration_axis],
        "u1_frontier_log10_error_per_site": [float(value) for value in u_frontier],
        "su2_frontier_log10_error_per_site": [float(value) for value in s_frontier],
        "u1_arm_labels": list(u1_labels),
        "su2_arm_labels": list(su2_labels),
        "average_log10_advantage": average,
        "geometric_error_advantage_factor": float(10.0**average),
        "worst_log10_advantage": worst,
        "seed_integrated_scores": {
            "u1": [float(value) for value in u_scores],
            "su2": [float(value) for value in s_scores],
        },
        "exact_8_plus_8_permutation": exact_two_sample_permutation(u_scores, s_scores),
        "independent_cluster_bootstrap": _bootstrap_resource(
            u1_values,
            su2_values,
            weights,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
    }


def parameter_pareto_summary(
    case: Mapping[str, object],
    *,
    bootstrap_resamples: int = 100_000,
    bootstrap_seed: int = 910_241,
) -> dict[str, object]:
    """Return the terminal translation-sector raw-parameter frontier."""

    grouped = {backbone: _translation_by_width(case, backbone) for backbone in ("u1", "su2")}
    counts: dict[str, dict[int, int]] = {}
    for backbone, by_width in grouped.items():
        counts[backbone] = {}
        for width, trials in by_width.items():
            values = {int(trial["parameter_count"]) for trial in trials}
            if len(values) != 1:
                raise AnalysisValidationError("parameter count varies over seeds")
            counts[backbone][width] = values.pop()

    lower = max(min(counts["u1"].values()), min(counts["su2"].values()))
    upper = min(max(counts["u1"].values()), max(counts["su2"].values()))
    if upper <= lower:
        raise AnalysisValidationError("parameter frontiers have no common budget interval")
    log_knots = sorted(
        {
            math.log10(lower),
            math.log10(upper),
            *(
                math.log10(value)
                for backbone in ("u1", "su2")
                for value in counts[backbone].values()
                if lower < value < upper
            ),
        }
    )
    left = np.asarray(log_knots[:-1], dtype=float)
    right = np.asarray(log_knots[1:], dtype=float)
    log_midpoints = 0.5 * (left + right)
    budgets = 10.0**log_midpoints
    weights = (right - left) / float(right[-1] - left[0])

    matrices: dict[str, np.ndarray] = {}
    labels: dict[str, list[str]] = {}
    for backbone in ("u1", "su2"):
        rows = []
        labels[backbone] = []
        for width in TRANSLATION_WIDTHS:
            arm_values = []
            for trial in grouped[backbone][width]:
                value, _ = _safe_log_error(
                    trial["metrics"]["energy_error_per_site"],
                    name=f"{case['case_id']} {trial['arm_id']} final error",
                )
                arm_values.append(value)
            row = np.repeat(np.asarray(arm_values)[:, None], budgets.size, axis=1)
            row[:, budgets < counts[backbone][width]] = np.inf
            rows.append(row)
            labels[backbone].append(f"{backbone}_D{width}:P={counts[backbone][width]}")
        matrices[backbone] = np.stack(rows)

    summary = _summary_from_resource(
        matrices["u1"],
        matrices["su2"],
        display_axis=budgets,
        integration_axis=log_midpoints,
        weights=weights,
        u1_labels=labels["u1"],
        su2_labels=labels["su2"],
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    summary.update(
        {
            "resource": "raw_parameter_count",
            "common_budget_interval": [int(lower), int(upper)],
            "frontier_semantics": "median over seeds, then reachable-width lower envelope",
        }
    )
    return summary


def _trial_curve(
    case: Mapping[str, object], trial: Mapping[str, object], *, clock: str
) -> tuple[np.ndarray, np.ndarray]:
    training = trial["training"]
    training_seconds = _finite(training["training_seconds"], "training seconds")
    compile_plus = _finite(
        trial["compile_plus_optimizer_seconds"], "compile plus optimizer seconds"
    )
    offset = compile_plus - training_seconds
    if offset < -1.0e-9:
        raise AnalysisValidationError("compile-plus time is shorter than training time")
    offset = max(offset, 0.0)
    if clock == "warm":
        time_offset = 0.0
        final_time = training_seconds
    elif clock == "cold":
        time_offset = offset
        final_time = compile_plus
    else:
        raise ValueError("clock must be 'warm' or 'cold'")

    anchor = float(case["exact_energy_anchor"])
    n_sites = int(case["n_sites"])
    times = [time_offset]
    energies = [float(training["initial_loss"])]
    previous_step = 0
    previous_time = -math.inf
    for row in training["history"]:
        step = int(row["step"])
        timestamp = _finite(row["cumulative_training_seconds"], "checkpoint time")
        if step <= previous_step or timestamp <= previous_time:
            raise AnalysisValidationError("training history is not strictly increasing")
        previous_step = step
        previous_time = timestamp
        times.append(time_offset + timestamp)
        energies.append(float(row["loss_before_update"]))
    times.append(final_time)
    energies.append(float(trial["metrics"]["energy"]))
    time_array = np.asarray(times, dtype=float)
    if np.any(np.diff(time_array) < -1.0e-9):
        raise AnalysisValidationError("final timing precedes a checkpoint")
    # Resolve sub-nanosecond duplicate timestamps by retaining the better state.
    unique_times: list[float] = []
    unique_errors: list[float] = []
    for timestamp, energy in zip(time_array, energies):
        log_error, _ = _safe_log_error(
            (energy - anchor) / n_sites,
            name=f"{case['case_id']} {trial['arm_id']} trajectory error",
        )
        if unique_times and timestamp <= unique_times[-1] + 1.0e-15:
            unique_errors[-1] = min(unique_errors[-1], log_error)
        else:
            unique_times.append(float(timestamp))
            unique_errors.append(log_error)
    best = np.minimum.accumulate(np.asarray(unique_errors, dtype=float))
    return np.asarray(unique_times, dtype=float), best


def matched_time_pareto_summary(
    base_case: Mapping[str, object],
    long_case: Mapping[str, object],
    su2_long_case: Mapping[str, object] | None = None,
    *,
    clock: str,
    bootstrap_resamples: int = 100_000,
    bootstrap_seed: int = 910_242,
    grid_size: int = DEFAULT_TIME_GRID_SIZE,
) -> dict[str, object]:
    """Return the anytime translation frontier on measured warm/cold seconds."""

    if _physical_key(base_case) != _physical_key(long_case):
        raise AnalysisValidationError("base and long time cases describe different physics")
    if grid_size < 3:
        raise ValueError("grid_size must be at least three")
    base_grouped = {
        backbone: _translation_by_width(base_case, backbone)
        for backbone in ("u1", "su2")
    }
    long_trials = sorted(long_case["trials"], key=lambda trial: int(trial["seed"]))
    if len(long_trials) != 8 or any(
        str(trial["arm_id"]) != "u1_D6_translation_long"
        or str(trial["backbone"]) != "u1"
        or str(trial["projection"]) != "translation"
        or int(trial["feature_dim"]) != 6
        for trial in long_trials
    ) or [int(trial["seed"]) for trial in long_trials] != list(range(21, 29)):
        raise AnalysisValidationError("long case is not an eight-seed U1-D6 arm")
    base_grouped["u1"][6] = [dict(trial) for trial in long_trials]
    if su2_long_case is not None:
        if _physical_key(base_case) != _physical_key(su2_long_case):
            raise AnalysisValidationError(
                "base and SU2-long time cases describe different physics"
            )
        su2_long_trials = sorted(
            su2_long_case["trials"], key=lambda trial: int(trial["seed"])
        )
        if len(su2_long_trials) != 8 or any(
            str(trial["arm_id"]) != "su2_D6_translation_long"
            or str(trial["backbone"]) != "su2"
            or str(trial["projection"]) != "translation"
            or int(trial["feature_dim"]) != 6
            for trial in su2_long_trials
        ) or [int(trial["seed"]) for trial in su2_long_trials] != list(range(21, 29)):
            raise AnalysisValidationError(
                "SU2-long case is not an eight-seed SU2-D6 arm"
            )
        base_grouped["su2"][6] = [dict(trial) for trial in su2_long_trials]

    curves: dict[str, dict[int, list[tuple[np.ndarray, np.ndarray]]]] = {
        "u1": {},
        "su2": {},
    }
    starts: dict[str, dict[int, float]] = {"u1": {}, "su2": {}}
    horizons: dict[str, dict[int, float]] = {"u1": {}, "su2": {}}
    for backbone in ("u1", "su2"):
        for width in TRANSLATION_WIDTHS:
            width_curves = [
                _trial_curve(base_case, trial, clock=clock)
                for trial in base_grouped[backbone][width]
            ]
            curves[backbone][width] = width_curves
            positive_starts = []
            for times, _ in width_curves:
                positive = times[times > 0.0]
                if positive.size == 0:
                    raise AnalysisValidationError("time curve has no positive timestamp")
                positive_starts.append(float(positive[0]))
            starts[backbone][width] = max(positive_starts)
            horizons[backbone][width] = min(float(times[-1]) for times, _ in width_curves)

    lower = max(min(starts["u1"].values()), min(starts["su2"].values()))
    upper = min(max(horizons["u1"].values()), max(horizons["su2"].values()))
    if not (0.0 < lower < upper):
        raise AnalysisValidationError(
            f"{clock} time frontiers have no common observed support: {lower}, {upper}"
        )
    grid = np.geomspace(lower, upper, grid_size)
    log_grid = np.log10(grid)
    weights = _integration_weights(log_grid)

    matrices: dict[str, np.ndarray] = {}
    labels: dict[str, list[str]] = {"u1": [], "su2": []}
    for backbone in ("u1", "su2"):
        width_rows = []
        for width in TRANSLATION_WIDTHS:
            seed_rows = []
            for times, log_errors in curves[backbone][width]:
                positive = times > 0.0
                interpolated = np.interp(
                    log_grid,
                    np.log10(times[positive]),
                    log_errors[positive],
                    left=np.inf,
                    right=float(log_errors[positive][-1]),
                )
                seed_rows.append(interpolated)
            row = np.asarray(seed_rows, dtype=float)
            row[:, grid < starts[backbone][width]] = np.inf
            width_rows.append(row)
            suffix = (
                ":long"
                if width == 6
                and (backbone == "u1" or su2_long_case is not None)
                else ""
            )
            labels[backbone].append(f"{backbone}_D{width}{suffix}")
        matrices[backbone] = np.stack(width_rows)

    summary = _summary_from_resource(
        matrices["u1"],
        matrices["su2"],
        display_axis=grid,
        integration_axis=log_grid,
        weights=weights,
        u1_labels=labels["u1"],
        su2_labels=labels["su2"],
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    summary.update(
        {
            "resource": f"{clock}_training_seconds",
            "clock": clock,
            "common_observed_time_interval_seconds": [float(lower), float(upper)],
            "checkpoint_timestamp_semantics": (
                "loss_before_update is assigned the later recorded cumulative timestamp; "
                "completed trajectories persist without extrapolated improvement"
            ),
            "u1_D6_source": "sealed 1200-update long falsification trajectory",
            "su2_D6_source": (
                "sealed 600-update horizon-completion trajectory"
                if su2_long_case is not None
                else "sealed 300-update base trajectory"
            ),
        }
    )
    return summary


_SECTOR_RANKS = {
    "square": {
        "m0": {"none": 12870, "translation": 822, "gamma_a1": 153},
        "singlet": {"none": 1430, "translation": 107, "gamma_a1": 31},
    },
    "triangular": {
        "m0": {"none": 12870, "translation": 822, "gamma_a1": 96},
        "singlet": {"none": 1430, "translation": 107, "gamma_a1": 14},
    },
}


def a1_floor_diagnosis(
    case: Mapping[str, object],
    *,
    floors: Sequence[float] = (1.0e-12, 1.0e-10, 1.0e-8),
    tight_target: float = 1.0e-5,
) -> dict[str, object] | None:
    """Diagnose D4 A1 small-sector and numerical-floor saturation."""

    selected: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for trial in case["trials"]:  # type: ignore[index]
        if int(trial["feature_dim"]) == 4 and str(trial["projection"]) in {
            "translation",
            "gamma_a1",
        }:
            selected.setdefault((str(trial["backbone"]), str(trial["projection"])), []).append(trial)
    required = {
        (backbone, projection)
        for backbone in ("u1", "su2")
        for projection in ("translation", "gamma_a1")
    }
    if not required.issubset(selected):
        return None
    for trials in selected.values():
        trials.sort(key=lambda trial: int(trial["seed"]))
        if len(trials) != 8:
            raise AnalysisValidationError("A1 diagnostic requires eight seeds per arm")

    errors = {
        key: np.asarray(
            [max(float(trial["metrics"]["energy_error_per_site"]), 0.0) for trial in trials],
            dtype=float,
        )
        for key, trials in selected.items()
    }
    floor_ratios: dict[str, float] = {}
    for floor in floors:
        paired_log_ratio = np.log10(np.maximum(errors[("u1", "gamma_a1")], floor)) - np.log10(
            np.maximum(errors[("su2", "gamma_a1")], floor)
        )
        floor_ratios[f"{floor:.0e}"] = float(np.median(paired_log_ratio))
    swing = max(floor_ratios.values()) - min(floor_ratios.values())
    floor_counts = {
        backbone: {
            f"{floor:.0e}": int(np.count_nonzero(errors[(backbone, "gamma_a1")] <= floor))
            for floor in floors
        }
        for backbone in ("u1", "su2")
    }
    geometry = str(case["geometry"])
    ranks = _SECTOR_RANKS[geometry]
    parameter_counts = {
        backbone: int(selected[(backbone, "gamma_a1")][0]["parameter_count"])
        for backbone in ("u1", "su2")
    }
    dof = {"u1": ranks["m0"]["gamma_a1"] - 1, "su2": ranks["singlet"]["gamma_a1"] - 1}
    overparameterization = {
        backbone: parameter_counts[backbone] / dof[backbone]
        for backbone in ("u1", "su2")
    }
    target_hits = {
        backbone: int(np.count_nonzero(errors[(backbone, "gamma_a1")] <= tight_target))
        for backbone in ("u1", "su2")
    }
    projection_gain = {
        backbone: float(
            np.median(
                np.log10(np.maximum(errors[(backbone, "translation")], ENERGY_FLOOR))
                - np.log10(np.maximum(errors[(backbone, "gamma_a1")], ENERGY_FLOOR))
            )
        )
        for backbone in ("u1", "su2")
    }
    floor_dominated = (
        floor_counts["u1"].get("1e-10", 0) >= 2
        or floor_counts["su2"].get("1e-10", 0) >= 2
        or swing > 0.25
    )
    saturated = (
        all(value > 10.0 for value in overparameterization.values())
        and all(value >= 6 for value in target_hits.values())
        and all(value > 0.0 for value in projection_gain.values())
    )
    return {
        "case_id": str(case["case_id"]),
        "sector_ranks": ranks,
        "parameter_counts": parameter_counts,
        "parameter_to_real_projective_dof": overparameterization,
        "tight_target_energy_error_per_site": float(tight_target),
        "tight_target_hits_of_8": target_hits,
        "median_translation_to_a1_log10_gain": projection_gain,
        "a1_censored_counts": floor_counts,
        "median_seed_label_log10_ratio_by_floor": floor_ratios,
        "floor_sensitivity_range_dex": float(swing),
        "floor_dominated": bool(floor_dominated),
        "small_sector_saturated": bool(saturated),
        "claim_use": "secondary mechanism control only; never satisfies promotion",
    }


def _derived_seed(seed: int, *labels: str) -> int:
    digest = hashlib.sha256(
        (str(int(seed)) + "\0" + "\0".join(labels)).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _holm_adjusted(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm step-down family-wise adjusted p values."""

    ordered = sorted(
        ((str(label), float(value)) for label, value in p_values.items()),
        key=lambda item: (item[1], item[0]),
    )
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (label, value) in enumerate(ordered):
        running = max(running, (total - rank) * value)
        adjusted[label] = min(1.0, running)
    return adjusted


def promotion_gate_summary(
    cases: Mapping[str, object],
    *,
    timing_claim_eligible: bool = False,
) -> dict[str, object]:
    """Apply the common four-case PRL promotion gate to each resource.

    The gate demands a useful average gain, a positive independently
    resampled lower bound, no material loss anywhere on the common frontier,
    and a family-wise significant exact permutation result.  The tiny A1
    sectors are intentionally absent from this decision.
    """

    minimum_average = math.log10(1.25)
    minimum_worst = -math.log10(1.1)
    alpha = 0.05
    resources = {
        "parameter": lambda case: case["parameter_pareto"],
        "warm": lambda case: case["matched_time_pareto"]["warm"],
        "cold": lambda case: case["matched_time_pareto"]["cold"],
    }
    families: dict[str, object] = {}
    for resource, select in resources.items():
        claim_eligible = resource == "parameter" or timing_claim_eligible
        selected = {case_id: select(case) for case_id, case in cases.items()}
        adjusted = _holm_adjusted(
            {
                case_id: float(
                    summary["exact_8_plus_8_permutation"]["one_sided_p_value"]
                )
                for case_id, summary in selected.items()
            }
        )
        decisions: dict[str, object] = {}
        for case_id, summary in selected.items():
            average = float(summary["average_log10_advantage"])
            worst = float(summary["worst_log10_advantage"])
            lower = float(
                summary["independent_cluster_bootstrap"][
                    "average_log10_advantage"
                ]["one_sided_95_lower"]
            )
            checks = {
                "average_gain": average >= minimum_average,
                "positive_bootstrap_lower": lower > 0.0,
                "worst_budget_noninferiority": worst >= minimum_worst,
                "holm_familywise_significance": adjusted[case_id] <= alpha,
            }
            numerical_passed = bool(all(checks.values()))
            decisions[case_id] = {
                "passed": bool(numerical_passed and claim_eligible),
                "numerical_checks_passed": numerical_passed,
                "claim_eligible": claim_eligible,
                "checks": checks,
                "average_log10_advantage": average,
                "geometric_error_advantage_factor": float(10.0**average),
                "average_one_sided_95_lower": lower,
                "worst_log10_advantage": worst,
                "raw_permutation_p": float(
                    summary["exact_8_plus_8_permutation"]["one_sided_p_value"]
                ),
                "holm_adjusted_p": adjusted[case_id],
            }
        families[resource] = {
            "claim_eligible": claim_eligible,
            "claim_eligibility_reason": (
                "raw-parameter counts do not require machine-load isolation"
                if resource == "parameter"
                else (
                    "all timing inputs carry a verified machine-isolation audit"
                    if timing_claim_eligible
                    else (
                        "timing artifacts do not record verified machine-load "
                        "isolation; numerical frontiers are exploratory"
                    )
                )
            ),
            "all_four_cases_pass": bool(
                len(decisions) == 4
                and all(bool(value["passed"]) for value in decisions.values())
            ),
            "passed_case_count": int(
                sum(bool(value["passed"]) for value in decisions.values())
            ),
            "numerical_passed_case_count": int(
                sum(
                    bool(value["numerical_checks_passed"])
                    for value in decisions.values()
                )
            ),
            "cases": decisions,
        }
    return {
        "thresholds": {
            "minimum_average_log10_advantage": minimum_average,
            "minimum_geometric_error_advantage_factor": 1.25,
            "average_bootstrap_one_sided_95_lower_must_exceed": 0.0,
            "minimum_worst_log10_advantage": minimum_worst,
            "maximum_worst_budget_error_disadvantage_factor": 1.1,
            "holm_familywise_alpha": alpha,
        },
        "families": families,
        "a1_excluded": True,
    }


def analyze_prl_results(
    base_result: str | Path | Mapping[str, object],
    long_result: str | Path | Mapping[str, object],
    su2_long_result: str | Path | Mapping[str, object] | None = None,
    *,
    bootstrap_resamples: int = 100_000,
    bootstrap_seed: int = 910_240,
    time_grid_size: int = DEFAULT_TIME_GRID_SIZE,
) -> dict[str, object]:
    """Validate and summarize the sealed 384+32 local PRL campaigns.

    The returned object contains only ordinary Python scalars, lists, and
    dictionaries and can therefore be passed directly to :mod:`json`.
    """

    base = _as_payload(base_result)
    long = _as_payload(long_result)
    su2_long = None if su2_long_result is None else _as_payload(su2_long_result)
    validation = validate_prl_results(base, long, su2_long)
    timing_payloads = (base, long) if su2_long is None else (base, long, su2_long)
    timing_claim_eligible = all(
        isinstance(payload.get("timing_environment_audit"), Mapping)
        and payload["timing_environment_audit"].get(
            "machine_isolation_verified"
        )
        is True
        for payload in timing_payloads
    )
    base_cases = _physical_case_map(base)
    long_cases = _physical_case_map(long)
    su2_long_cases = None if su2_long is None else _physical_case_map(su2_long)
    cases: dict[str, object] = {}
    a1: dict[str, object] = {}
    for key, base_case in sorted(base_cases.items(), key=lambda item: str(item[1]["case_id"])):
        case_id = str(base_case["case_id"])
        long_case = long_cases[key]
        su2_long_case = None if su2_long_cases is None else su2_long_cases[key]
        parameter_seed = _derived_seed(bootstrap_seed, case_id, "parameter")
        warm_seed = _derived_seed(bootstrap_seed, case_id, "warm")
        cold_seed = _derived_seed(bootstrap_seed, case_id, "cold")
        cases[case_id] = {
            "physical_key": list(key),
            "parameter_pareto": parameter_pareto_summary(
                base_case,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=parameter_seed,
            ),
            "matched_time_pareto": {
                "warm": matched_time_pareto_summary(
                    base_case,
                    long_case,
                    su2_long_case,
                    clock="warm",
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=warm_seed,
                    grid_size=time_grid_size,
                ),
                "cold": matched_time_pareto_summary(
                    base_case,
                    long_case,
                    su2_long_case,
                    clock="cold",
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=cold_seed,
                    grid_size=time_grid_size,
                ),
            },
        }
        diagnosis = a1_floor_diagnosis(base_case)
        if diagnosis is not None:
            a1[case_id] = diagnosis
    source_path = Path(__file__).resolve()
    source_manifest = {
        "projects/clebsch_tree_nqs/analysis.py": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
    }
    result = {
        "schema_version": 1,
        "analysis": "spin_prl_local_parameter_and_measured_time_pareto",
        "analysis_provenance": {
            "source_sha256": source_manifest,
            "source_manifest_sha256": _canonical_sha256(source_manifest),
        },
        "inputs": {
            "base_config_sha256": str(base["config_sha256"]),
            "base_trial_manifest_sha256": str(base["trial_manifest_sha256"]),
            "long_config_sha256": str(long["config_sha256"]),
            "long_trial_manifest_sha256": str(long["trial_manifest_sha256"]),
            "su2_long_config_sha256": (
                None if su2_long is None else str(su2_long["config_sha256"])
            ),
            "su2_long_trial_manifest_sha256": (
                None
                if su2_long is None
                else str(su2_long["trial_manifest_sha256"])
            ),
            "bootstrap_resamples": int(bootstrap_resamples),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "validation": validation,
        "cases": cases,
        "promotion_gates": promotion_gate_summary(
            cases, timing_claim_eligible=timing_claim_eligible
        ),
        "a1_floor_diagnostics": a1,
        "inference_boundary": {
            "primary_bootstrap": "independent U1 and SU2 seed-label clusters",
            "seed_pairing": "same integer labels are not matched initial functions",
            "long_campaign": (
                "targeted U1-D6 horizon extension plus optional SU2-D6 horizon "
                "completion; neither is independent confirmation"
            ),
            "timing_claim": (
                "publication-eligible machine-isolated timing"
                if timing_claim_eligible
                else (
                    "exploratory conservative timing only: the sealed inputs do "
                    "not record verified machine-load isolation"
                )
            ),
            "observable_claim": "exact identities are audited; no independent observable anchor is inferred",
        },
    }
    result["summary_payload_sha256"] = _canonical_sha256(result)
    return result


__all__ = [
    "AnalysisValidationError",
    "a1_floor_diagnosis",
    "analyze_prl_results",
    "exact_two_sample_permutation",
    "matched_time_pareto_summary",
    "parameter_pareto_summary",
    "promotion_gate_summary",
    "validate_prl_results",
]
