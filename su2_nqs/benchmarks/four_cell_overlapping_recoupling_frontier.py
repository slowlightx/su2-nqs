"""Matched exact-small frontier for overlapping local-Schur recoupling views."""

from __future__ import annotations

from functools import lru_cache
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..models.su2_four_cell_sector_vit import (
    init_su2_four_cell_sector_vit,
    su2_four_cell_sector_log_amplitudes,
)
from ..vmc.adam import train_exact_adam
from .clebsch_tree_exact import coo_matvec, fixed_m_spin_basis
from .four_cell_additive_carrier import _case_data as _base_case_data
from .four_cell_matched_frontier import _metrics, _parameter_count
from .overlapping_schur_recoupling import build_overlapping_schur_a1_representation


jax.config.update("jax_enable_x64", True)


ARM_IDS = (
    "single_tiling_backbone",
    "physical_overlapping_recoupling",
    "a1_scrambled_recoupling",
)
TARGET_ERRORS = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5)


@lru_cache(maxsize=None)
def _case_data(case_name: str) -> dict[str, object]:
    data = dict(_base_case_data(case_name))
    configurations = fixed_m_spin_basis(int(data["n_sites"]), 0)
    representation = build_overlapping_schur_a1_representation(
        case_name,
        np.asarray(data["singlet_basis"]),
        configurations,
        np.asarray(data["indices_numpy"]),
        maximum_operator_count=8,
    )
    data.update(
        {
            "coefficient_a1_basis": jnp.asarray(
                representation.coefficient_a1_basis
            ),
            "physical_a1_basis": jnp.asarray(representation.physical_a1_basis),
            "physical_operators": jnp.asarray(representation.physical_operators),
            "scrambled_operators": jnp.asarray(
                representation.scrambled_operators
            ),
            "operator_labels": representation.operator_labels,
            "representation_diagnostics": representation.diagnostics,
            "tilings": representation.tilings,
        }
    )
    return data


def _init_params(key: jax.Array, data: dict[str, object], *, correlated: bool):
    params: dict[str, object] = {
        "backbone": init_su2_four_cell_sector_vit(key, data["config"])
    }
    if correlated:
        count = len(data["operator_labels"])
        params["recoupling_correlator"] = {
            "real": jnp.zeros((count,), dtype=jnp.float64),
            "imag": jnp.zeros((count,), dtype=jnp.float64),
        }
    return params


def _state(params: object, data: dict[str, object], operators) -> jax.Array:
    logs = su2_four_cell_sector_log_amplitudes(
        params["backbone"], data["actions"], data["config"]
    )
    shift = jax.lax.stop_gradient(jnp.max(jnp.real(logs)))
    coefficients = jnp.exp(logs - shift)
    reduced_a1 = data["coefficient_a1_basis"].T @ coefficients
    if operators is not None:
        channels = jnp.einsum("kij,j->ki", operators, reduced_a1)
        values = params["recoupling_correlator"]
        weights = values["real"] + 1.0j * values["imag"]
        reduced_a1 = reduced_a1 + jnp.einsum("k,ki->i", weights, channels)
    return data["physical_a1_basis"] @ reduced_a1


def _steps_to_targets(
    initial_energy: float,
    optimizer: dict[str, object],
    exact_energy: float,
) -> dict[str, int | None]:
    curve = [(0, max(initial_energy - exact_energy, 0.0))]
    curve.extend(
        (
            int(record["step"]),
            max(float(record["loss_before_update"]) - exact_energy, 0.0),
        )
        for record in optimizer["history"]
    )
    result = {}
    for target in TARGET_ERRORS:
        crossing = next((step for step, error in curve if error < target), None)
        result[f"steps_below_{target:.0e}"] = crossing
    return result


def overlapping_recoupling_frontier(
    *,
    seeds: tuple[int, ...],
    steps: int,
    learning_rate: float = 1.0e-2,
) -> dict[str, object]:
    """Compare physical overlapping views with an in-A1 scrambled control."""

    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct seeds are required")
    if steps < 1:
        raise ValueError("steps must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    cases = {}
    for case_name in ("triangular_n12_j2_0125", "square_n16_j2_05"):
        data = _case_data(case_name)
        runs = {arm: [] for arm in ARM_IDS}
        for seed in seeds:
            initial_states = []
            for arm_id, operators in (
                (ARM_IDS[0], None),
                (ARM_IDS[1], data["physical_operators"]),
                (ARM_IDS[2], data["scrambled_operators"]),
            ):
                params = _init_params(
                    jax.random.PRNGKey(seed),
                    data,
                    correlated=operators is not None,
                )

                def state_fn(candidate):
                    return _state(candidate, data, operators)

                def loss(candidate):
                    state = state_fn(candidate)
                    action = coo_matvec(
                        data["rows"], data["columns"], data["values"], state
                    )
                    return jnp.vdot(state, action).real / jnp.vdot(state, state).real

                initial_state = np.array(state_fn(params), copy=True)
                initial_state /= np.linalg.norm(initial_state)
                initial_states.append(initial_state)
                initial_energy = float(loss(params))
                started = time.perf_counter()
                trained, optimizer = train_exact_adam(
                    params,
                    loss_fn=loss,
                    steps=steps,
                    learning_rate=learning_rate,
                    record_every=max(1, steps // 20),
                )
                state = np.array(state_fn(trained), copy=True)
                state /= np.linalg.norm(state)
                row = {
                    "arm_id": arm_id,
                    "seed": int(seed),
                    "parameter_count": _parameter_count(trained),
                    "initial_energy_total": initial_energy,
                    "wall_seconds": time.perf_counter() - started,
                    "outer_full_h_calls": int(steps + 2),
                    "internal_a1_operator_actions_per_state": (
                        0 if operators is None else len(data["operator_labels"])
                    ),
                    "optimizer": optimizer,
                    "steps_to_targets": _steps_to_targets(
                        initial_energy,
                        optimizer,
                        float(data["exact_energy_total"]),
                    ),
                    "metrics": _metrics(
                        state,
                        data["hamiltonian"],
                        data["exact_state"],
                        data["singlet_basis"],
                        data["indices_numpy"],
                    ),
                }
                if operators is not None:
                    values = trained["recoupling_correlator"]
                    row["operator_weights_real"] = np.asarray(
                        values["real"]
                    ).tolist()
                    row["operator_weights_imag"] = np.asarray(
                        values["imag"]
                    ).tolist()
                runs[arm_id].append(row)
            for state in initial_states[1:]:
                if np.max(np.abs(state - initial_states[0])) > 3.0e-13:
                    raise RuntimeError("matched recoupling arms start differently")

        exact = float(data["exact_energy_total"])
        summary = {}
        for arm_id, rows in runs.items():
            errors = np.asarray(
                [row["metrics"]["energy_total"] - exact for row in rows]
            )
            target_steps = {}
            for target in TARGET_ERRORS:
                key = f"steps_below_{target:.0e}"
                values = [
                    row["steps_to_targets"][key]
                    for row in rows
                    if row["steps_to_targets"][key] is not None
                ]
                target_steps[key] = (
                    float(np.median(values)) if len(values) == len(rows) else None
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
                "median_steps_to_targets": target_steps,
            }
        raw = summary[ARM_IDS[0]]
        physical = summary[ARM_IDS[1]]
        scrambled = summary[ARM_IDS[2]]
        paired_wins = [
            left < right
            for left, right in zip(
                physical["all_seed_energy_errors_total"],
                scrambled["all_seed_energy_errors_total"],
                strict=True,
            )
        ]
        if case_name == "triangular_n12_j2_0125":
            gates = {
                "physical_numerically_noninferior_below_1e-10": physical[
                    "maximum_energy_error_total"
                ]
                < 1.0e-10,
            }
        else:
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
                "physical_beats_scrambled_every_paired_seed": bool(
                    all(paired_wins)
                ),
                "physical_beats_scrambled_median_variance": physical[
                    "median_variance_total"
                ]
                < scrambled["median_variance_total"],
                "physical_all_seeds_below_1e-5": physical[
                    "maximum_energy_error_total"
                ]
                < 1.0e-5,
            }
        cases[case_name] = {
            "n_sites": int(data["n_sites"]),
            "exact_energy_total": exact,
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "operator_labels": list(data["operator_labels"]),
            "representation_diagnostics": data["representation_diagnostics"],
            "tilings": [
                [list(support) for support in tiling]
                for tiling in data["tilings"]
            ],
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
            "Every arm uses the same complete singlet backbone and exact "
            "Gamma-A1 coordinates. The physical bank averages local reduced-"
            "label scalars over translated overlapping Schur tilings using "
            "exact basis overlaps. The scrambled bank is orthogonally "
            "conjugated inside the identical A1 space, preserving every "
            "operator spectrum, parameter count, initialization, and action "
            "count. Raw-backbone cost is lower and reported separately."
        ),
    }


__all__ = [
    "ARM_IDS",
    "TARGET_ERRORS",
    "overlapping_recoupling_frontier",
]
