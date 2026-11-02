"""Shared-backbone Ritz calibration of overlapping recoupling directions.

This benchmark separates variational subspace quality from joint nonlinear
optimization.  A raw SU(2)/Gamma-A1 backbone is trained once per seed and
frozen.  Physical and scrambled operator heads are then solved exactly in the
matched spans ``{psi, O_k psi}``.  The cheaper ``{psi, H psi}`` one-Lanczos
span is reported explicitly so a recoupling win cannot be mistaken for a new
mechanism when it is only Hamiltonian alignment.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from ..vmc.adam import train_exact_adam
from .clebsch_tree_exact import coo_matvec
from .four_cell_matched_frontier import _metrics, _parameter_count
from .four_cell_overlapping_recoupling_frontier import (
    _case_data,
    _init_params,
    _state,
)


jax.config.update("jax_enable_x64", True)


ARM_IDS = (
    "shared_raw_backbone",
    "one_lanczos_ritz",
    "physical_overlapping_ritz",
    "a1_scrambled_ritz",
)


def _loss(params: object, data: dict[str, object]) -> jax.Array:
    state = _state(params, data, None)
    action = coo_matvec(
        data["rows"], data["columns"], data["values"], state
    )
    return jnp.vdot(state, action).real / jnp.vdot(state, state).real


def _ritz_state(
    reduced_state: np.ndarray,
    hamiltonian_a1: np.ndarray,
    operators: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    state = np.asarray(reduced_state, dtype=np.complex128)
    columns = [state]
    if operators is None:
        columns.append(hamiltonian_a1 @ state)
    else:
        columns.extend(operator @ state for operator in operators)
    trial = np.column_stack(columns)
    left, singular, _right = np.linalg.svd(trial, full_matrices=False)
    tolerance = max(float(singular[0]), 1.0) * 2.0e-12
    rank = int(np.count_nonzero(singular > tolerance))
    basis = left[:, :rank]
    projected = basis.conj().T @ (hamiltonian_a1 @ basis)
    projected = 0.5 * (projected + projected.conj().T)
    _values, vectors = np.linalg.eigh(projected)
    result = basis @ vectors[:, 0]
    result /= np.linalg.norm(result)
    return result, rank


def _operator_span_fraction(
    operators: np.ndarray,
    hamiltonian_a1: np.ndarray,
) -> float:
    dimension = hamiltonian_a1.shape[0]
    target = 0.5 * (hamiltonian_a1 + hamiltonian_a1.conj().T)
    target -= np.trace(target) / dimension * np.eye(dimension)
    target /= np.linalg.norm(target)
    flattened = np.asarray(operators).reshape(len(operators), -1).T
    basis, _singular, _right = np.linalg.svd(flattened, full_matrices=False)
    projection = basis @ (basis.conj().T @ target.reshape(-1))
    return float(np.linalg.norm(projection) ** 2)


def recoupling_ritz_frontier(
    *,
    seeds: tuple[int, ...],
    warmup_steps: int,
    learning_rate: float = 1.0e-2,
) -> dict[str, object]:
    """Compare matched physical/scrambled Ritz spans on shared backbones."""

    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct seeds are required")
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    cases = {}
    for case_name in ("triangular_n12_j2_0125", "square_n16_j2_05"):
        data = _case_data(case_name)
        physical_a1 = np.asarray(data["physical_a1_basis"])
        hamiltonian_a1 = physical_a1.T @ (data["hamiltonian"] @ physical_a1)
        hamiltonian_a1 = 0.5 * (hamiltonian_a1 + hamiltonian_a1.T)
        physical_operators = np.asarray(data["physical_operators"])
        scrambled_operators = np.asarray(data["scrambled_operators"])
        runs = {arm: [] for arm in ARM_IDS}
        for seed in seeds:
            initial = _init_params(
                jax.random.PRNGKey(seed), data, correlated=False
            )
            started = time.perf_counter()
            warmed, optimizer = train_exact_adam(
                initial,
                loss_fn=lambda candidate: _loss(candidate, data),
                steps=warmup_steps,
                learning_rate=learning_rate,
                record_every=max(1, warmup_steps // 10),
            )
            warmup_seconds = time.perf_counter() - started
            raw_physical = np.array(_state(warmed, data, None), copy=True)
            raw_physical /= np.linalg.norm(raw_physical)
            reduced = physical_a1.T @ raw_physical
            reduced /= np.linalg.norm(reduced)
            reduced_states = {
                ARM_IDS[0]: (reduced, 1),
                ARM_IDS[1]: _ritz_state(reduced, hamiltonian_a1, None),
                ARM_IDS[2]: _ritz_state(
                    reduced, hamiltonian_a1, physical_operators
                ),
                ARM_IDS[3]: _ritz_state(
                    reduced, hamiltonian_a1, scrambled_operators
                ),
            }
            for arm_id, (reduced_result, rank) in reduced_states.items():
                state = physical_a1 @ reduced_result
                row = {
                    "arm_id": arm_id,
                    "seed": int(seed),
                    "shared_backbone_parameter_count": _parameter_count(warmed),
                    "shared_warmup_steps": int(warmup_steps),
                    "shared_warmup_full_h_calls": int(warmup_steps + 2),
                    "shared_warmup_seconds": warmup_seconds,
                    "ritz_rank": int(rank),
                    "additional_operator_actions": (
                        0
                        if arm_id == ARM_IDS[0]
                        else 1
                        if arm_id == ARM_IDS[1]
                        else len(data["operator_labels"])
                    ),
                    "optimizer": optimizer if arm_id == ARM_IDS[0] else None,
                    "metrics": _metrics(
                        state,
                        data["hamiltonian"],
                        data["exact_state"],
                        data["singlet_basis"],
                        data["indices_numpy"],
                    ),
                }
                runs[arm_id].append(row)
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
                "median_ritz_rank": float(
                    np.median([row["ritz_rank"] for row in rows])
                ),
            }
        raw = summary[ARM_IDS[0]]
        lanczos = summary[ARM_IDS[1]]
        physical = summary[ARM_IDS[2]]
        scrambled = summary[ARM_IDS[3]]
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
            "warmup_steps": int(warmup_steps),
            "learning_rate": float(learning_rate),
            "operator_count": len(data["operator_labels"]),
            "physical_hamiltonian_span_fraction": _operator_span_fraction(
                physical_operators, hamiltonian_a1
            ),
            "scrambled_hamiltonian_span_fraction": _operator_span_fraction(
                scrambled_operators, hamiltonian_a1
            ),
            "runs": runs,
            "summary": summary,
            "paired_physical_win_flags": paired_wins,
            "physical_median_gain_beyond_one_lanczos": float(
                lanczos["median_energy_error_total"]
                - physical["median_energy_error_total"]
            ),
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
            "This is a frozen-backbone variational-subspace calibration, not "
            "a trained scalable ansatz and not an untouched audit. Physical "
            "and scrambled heads use eight matched A1 operator actions and "
            "the same exact Ritz solve. The one-Lanczos arm uses only Hpsi "
            "and must be reported before attributing a gain to overlapping "
            "recoupling beyond standard Krylov improvement."
        ),
    }


__all__ = ["ARM_IDS", "recoupling_ritz_frontier"]
