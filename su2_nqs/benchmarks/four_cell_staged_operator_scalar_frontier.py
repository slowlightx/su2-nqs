"""Staged matched test of shell-resolved SU(2)-scalar operator words."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from ..vmc.adam import train_exact_adam
from .clebsch_tree_exact import coo_matvec
from .four_cell_matched_frontier import _metrics, _parameter_count
from .four_cell_operator_scalar_frontier import (
    ARM_IDS,
    OPERATOR_LABELS,
    OPERATOR_SHELL_ACTION_COUNTS,
    _case_data,
    _init_params,
    _state,
)


jax.config.update("jax_enable_x64", True)


def _loss(params: object, data: dict[str, object], operators) -> jax.Array:
    state = _state(params, data, operators)
    action = coo_matvec(
        data["rows"], data["columns"], data["values"], state
    )
    return jnp.vdot(state, action).real / jnp.vdot(state, state).real


def staged_operator_scalar_frontier(
    *,
    seeds: tuple[int, ...],
    warmup_steps: int,
    branch_steps: int,
    learning_rate: float = 1.0e-2,
) -> dict[str, object]:
    """Warm one shared backbone, then branch into raw/physical/scrambled arms."""

    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct seeds are required")
    if min(warmup_steps, branch_steps) < 1:
        raise ValueError("warmup_steps and branch_steps must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    cases = {}
    for case_name in ("triangular_n12_j2_0125", "square_n16_j2_05"):
        data = _case_data(case_name)
        runs = {arm: [] for arm in ARM_IDS}
        warmups = []
        for seed in seeds:
            initial = _init_params(
                jax.random.PRNGKey(seed), data, correlated=False
            )
            warm_started = time.perf_counter()
            warmed, warm_optimizer = train_exact_adam(
                initial,
                loss_fn=lambda candidate: _loss(candidate, data, None),
                steps=warmup_steps,
                learning_rate=learning_rate,
                record_every=max(1, warmup_steps // 5),
            )
            warm_wall = time.perf_counter() - warm_started
            warm_state = np.array(_state(warmed, data, None), copy=True)
            warm_state /= np.linalg.norm(warm_state)
            warmups.append(
                {
                    "seed": int(seed),
                    "wall_seconds": warm_wall,
                    "optimizer": warm_optimizer,
                    "metrics": _metrics(
                        warm_state,
                        data["hamiltonian"],
                        data["exact_state"],
                        data["singlet_basis"],
                        data["indices_numpy"],
                    ),
                }
            )
            initial_states = []
            for arm_id, operators in (
                (ARM_IDS[0], None),
                (ARM_IDS[1], data["physical_operators"]),
                (ARM_IDS[2], data["scrambled_operators"]),
            ):
                if operators is None:
                    params = {"backbone": warmed["backbone"]}
                else:
                    params = _init_params(
                        jax.random.PRNGKey(seed), data, correlated=True
                    )
                    params["backbone"] = warmed["backbone"]
                state_before = np.asarray(_state(params, data, operators))
                initial_states.append(state_before / np.linalg.norm(state_before))
                started = time.perf_counter()
                trained, optimizer = train_exact_adam(
                    params,
                    loss_fn=lambda candidate, op=operators: _loss(
                        candidate, data, op
                    ),
                    steps=branch_steps,
                    learning_rate=learning_rate,
                    record_every=max(1, branch_steps // 5),
                )
                state = np.array(_state(trained, data, operators), copy=True)
                state /= np.linalg.norm(state)
                row = {
                    "arm_id": arm_id,
                    "seed": int(seed),
                    "parameter_count": _parameter_count(trained),
                    "shared_warmup_steps": int(warmup_steps),
                    "branch_steps": int(branch_steps),
                    "shared_warmup_wall_seconds": warm_wall,
                    "branch_wall_seconds": time.perf_counter() - started,
                    "outer_full_h_calls_per_arm_including_shared_warmup": int(
                        warmup_steps + branch_steps + 4
                    ),
                    "internal_shell_actions_per_branch_state": (
                        0 if operators is None else sum(OPERATOR_SHELL_ACTION_COUNTS)
                    ),
                    "optimizer": optimizer,
                    "metrics": _metrics(
                        state,
                        data["hamiltonian"],
                        data["exact_state"],
                        data["singlet_basis"],
                        data["indices_numpy"],
                    ),
                }
                if operators is not None:
                    values = trained["operator_correlator"]
                    row["operator_weights_real"] = np.asarray(values["real"]).tolist()
                    row["operator_weights_imag"] = np.asarray(values["imag"]).tolist()
                runs[arm_id].append(row)
            for state in initial_states[1:]:
                if np.max(np.abs(state - initial_states[0])) > 2.0e-13:
                    raise RuntimeError("staged arms do not share the branch state")

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
                "median_total_wall_seconds": float(
                    np.median(
                        [
                            row["shared_warmup_wall_seconds"]
                            + row["branch_wall_seconds"]
                            for row in rows
                        ]
                    )
                ),
            }
        physical = summary[ARM_IDS[1]]
        raw = summary[ARM_IDS[0]]
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
                "physical_beats_scrambled_every_paired_seed": bool(all(paired_wins)),
                "physical_all_seeds_below_1e-5": physical[
                    "maximum_energy_error_total"
                ]
                < 1.0e-5,
            }
        cases[case_name] = {
            "n_sites": data["n_sites"],
            "exact_energy_total": exact,
            "warmup_steps": int(warmup_steps),
            "branch_steps": int(branch_steps),
            "learning_rate": float(learning_rate),
            "operator_labels": list(OPERATOR_LABELS),
            "operator_shell_action_counts": list(OPERATOR_SHELL_ACTION_COUNTS),
            "scramble_seed": int(data["scramble_seed"]),
            "warmups": warmups,
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
            "Every seed first trains one shared SU2 Schur-ViT. Raw, physical "
            "and isospectral-scrambled arms then start from the identical "
            "branch state. Correlated arms have equal internal operator work; "
            "raw has lower cost and is not an efficiency-matched control."
        ),
    }


__all__ = ["staged_operator_scalar_frontier"]
