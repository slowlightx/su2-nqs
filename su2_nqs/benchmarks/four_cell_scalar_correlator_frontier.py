"""Matched frontier for a configuration-dependent SU(2)-scalar correlator."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from ..models.su2_four_cell_sector_vit import su2_four_cell_sector_log_amplitudes
from ..models.su2_scalar_correlator import (
    init_schur_scalar_correlator,
    schur_scalar_features,
    schur_scalar_log_correction,
)
from ..vmc.adam import train_exact_adam
from .clebsch_tree_exact import coo_matvec, reynolds_project_state
from .four_cell_additive_carrier import _case_data
from .four_cell_matched_frontier import _metrics, _parameter_count


jax.config.update("jax_enable_x64", True)


ARM_IDS = (
    "no_scalar_correlator",
    "physical_su2_scalar_correlator",
    "scrambled_scalar_correlator",
)


def _init_params(key: jax.Array, data: dict[str, object], *, correlated: bool):
    from ..models.su2_four_cell_sector_vit import init_su2_four_cell_sector_vit

    params: dict[str, object] = {
        "backbone": init_su2_four_cell_sector_vit(key, data["config"])
    }
    if correlated:
        params["scalar_correlator"] = init_schur_scalar_correlator(
            int(data["physical_features"].shape[1])
        )
    return params


def _state(params: object, data: dict[str, object], features) -> jax.Array:
    logs = su2_four_cell_sector_log_amplitudes(
        params["backbone"], data["actions"], data["config"]
    )
    if features is not None:
        logs = logs + schur_scalar_log_correction(
            params["scalar_correlator"], features
        )
    shift = jax.lax.stop_gradient(jnp.max(jnp.real(logs)))
    coefficients = jnp.exp(logs - shift)
    physical = data["basis"] @ coefficients
    return reynolds_project_state(physical, data["indices"])


def _correlator_case_data(name: str) -> dict[str, object]:
    data = _case_data(name)
    features = schur_scalar_features(
        np.asarray(data["actions"]), data["config"]
    )
    scramble_seed = 2026092211 if name.startswith("triangular") else 2026092212
    permutation = np.random.default_rng(scramble_seed).permutation(
        len(features.values)
    )
    data = dict(data)
    data.update(
        {
            "physical_features": jnp.asarray(features.values),
            "scrambled_features": jnp.asarray(features.values[permutation]),
            "feature_labels": features.labels,
            "scramble_seed": scramble_seed,
        }
    )
    return data


def scalar_correlator_frontier(
    *,
    seeds: tuple[int, ...],
    steps: int = 1000,
) -> dict[str, object]:
    """Compare physical and row-scrambled scalar data at matched work."""

    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct seeds are required")
    if steps < 1:
        raise ValueError("steps must be positive")
    cases = {}
    for case_name in ("triangular_n12_j2_0125", "square_n16_j2_05"):
        data = _correlator_case_data(case_name)
        runs = {arm: [] for arm in ARM_IDS}
        for seed in seeds:
            initial_states = []
            for arm_id, features in (
                (ARM_IDS[0], None),
                (ARM_IDS[1], data["physical_features"]),
                (ARM_IDS[2], data["scrambled_features"]),
            ):
                params = _init_params(
                    jax.random.PRNGKey(seed),
                    data,
                    correlated=features is not None,
                )

                def state_fn(candidate):
                    return _state(candidate, data, features)

                def loss(candidate):
                    state = state_fn(candidate)
                    action = coo_matvec(
                        data["rows"], data["columns"], data["values"], state
                    )
                    return jnp.vdot(state, action).real / jnp.vdot(state, state).real

                initial_state = np.asarray(state_fn(params))
                initial_states.append(initial_state / np.linalg.norm(initial_state))
                initial = float(loss(params))
                started = time.perf_counter()
                trained, optimizer = train_exact_adam(
                    params,
                    loss_fn=loss,
                    steps=steps,
                    learning_rate=1.0e-2,
                    record_every=max(1, steps // 10),
                )
                state = np.array(state_fn(trained), copy=True)
                state /= np.linalg.norm(state)
                row = {
                    "arm_id": arm_id,
                    "seed": int(seed),
                    "parameter_count": _parameter_count(trained),
                    "initial_energy_total": initial,
                    "wall_seconds": time.perf_counter() - started,
                    "full_h_calls": int(steps + 2),
                    "optimizer": optimizer,
                    "metrics": _metrics(
                        state,
                        data["hamiltonian"],
                        data["exact_state"],
                        data["singlet_basis"],
                        data["indices_numpy"],
                    ),
                }
                if features is not None:
                    scalar = trained["scalar_correlator"]
                    row["scalar_log_magnitude_norm"] = float(
                        np.linalg.norm(np.asarray(scalar["log_magnitude"]))
                    )
                    row["scalar_phase_norm"] = float(
                        np.linalg.norm(np.asarray(scalar["phase"]))
                    )
                runs[arm_id].append(row)
            for state in initial_states[1:]:
                if np.max(np.abs(state - initial_states[0])) > 2.0e-13:
                    raise RuntimeError("matched scalar arms do not share an initial state")

        exact = float(data["exact_energy_total"])
        summary = {}
        for arm_id, rows in runs.items():
            errors = np.asarray(
                [row["metrics"]["energy_total"] - exact for row in rows]
            )
            summary[arm_id] = {
                "median_energy_error_total": float(np.median(errors)),
                "maximum_energy_error_total": float(np.max(errors)),
                "all_seed_energy_errors_total": errors.tolist(),
                "median_variance_total": float(
                    np.median([row["metrics"]["variance_total"] for row in rows])
                ),
                "median_wall_seconds": float(
                    np.median([row["wall_seconds"] for row in rows])
                ),
            }
        physical = summary[ARM_IDS[1]]
        raw = summary[ARM_IDS[0]]
        scrambled = summary[ARM_IDS[2]]
        paired_wins = [
            physical_error < scrambled_error
            for physical_error, scrambled_error in zip(
                physical["all_seed_energy_errors_total"],
                scrambled["all_seed_energy_errors_total"],
                strict=True,
            )
        ]
        gates = {
            "physical_improves_raw_median": physical[
                "median_energy_error_total"
            ]
            < raw["median_energy_error_total"],
            "physical_improves_raw_worst_seed": physical[
                "maximum_energy_error_total"
            ]
            < raw["maximum_energy_error_total"],
            "physical_beats_scrambled_median": physical[
                "median_energy_error_total"
            ]
            < scrambled["median_energy_error_total"],
            "physical_beats_scrambled_every_paired_seed": bool(all(paired_wins)),
            "physical_all_seeds_below_1e-5": physical[
                "maximum_energy_error_total"
            ]
            < 1.0e-5,
        }
        cases[case_name] = {
            "n_sites": data["n_sites"],
            "exact_energy_total": exact,
            "steps": int(steps),
            "feature_count": int(data["physical_features"].shape[1]),
            "feature_labels": list(data["feature_labels"]),
            "scramble_seed": int(data["scramble_seed"]),
            "runs": runs,
            "summary": summary,
            "paired_physical_win_flags": paired_wins,
            "gates": gates,
            "gates_passed": bool(all(gates.values())),
        }
    return {
        "seeds": list(seeds),
        "cases": cases,
        "all_cases_passed": bool(
            all(case["gates_passed"] for case in cases.values())
        ),
        "claim_boundary": (
            "The correlator acts only on configuration-dependent reduced "
            "Schur q,a and inter-cell-q scalars. The scrambled arm has the "
            "same feature distribution, parameters, initial state, updates, "
            "and full-H calls, but breaks the configuration-to-physics map."
        ),
    }


__all__ = ["ARM_IDS", "scalar_correlator_frontier"]
