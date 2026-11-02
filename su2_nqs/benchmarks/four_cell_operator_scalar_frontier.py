"""Shell-resolved SU(2)-scalar operator correlator on Schur-ViT states."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.j1j2 import square_j1j2_bonds
from ..lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from ..models.su2_four_cell_sector_vit import (
    init_su2_four_cell_sector_vit,
    su2_four_cell_sector_log_amplitudes,
)
from ..vmc.adam import train_exact_adam
from .clebsch_tree_exact import coo_matvec, reynolds_project_state
from .four_cell_additive_carrier import _case_data as _base_case_data
from .four_cell_matched_frontier import _metrics, _parameter_count
from .four_cell_schur import four_cell_schur_sector_basis_matrix
from .four_cell_schur_frontier import (
    SQUARE_N16_SUPPORTS,
    TRIANGULAR_N12_SUPERCELL,
    TRIANGULAR_N12_SUPPORTS,
    _coo_sparse,
)


jax.config.update("jax_enable_x64", True)


ARM_IDS = (
    "no_operator_correlator",
    "physical_su2_operator_correlator",
    "scrambled_operator_correlator",
)
OPERATOR_LABELS = ("J1", "J2", "J1_squared", "J2_squared", "J1_J2_anticommutator")
OPERATOR_SHELL_ACTION_COUNTS = (1, 1, 2, 2, 4)


def _reduced_shell_operators(name: str, data: dict[str, object]) -> np.ndarray:
    """Return normalized J1/J2 matrices without using ground-state data."""

    if name == "triangular_n12_j2_0125":
        kind = "triangular_triangle"
        supports = TRIANGULAR_N12_SUPPORTS
        site_count = 12
        shells = (
            triangular_hnf_j1j2_bonds(
                TRIANGULAR_N12_SUPERCELL, j1=1.0, j2=0.0
            ),
            triangular_hnf_j1j2_bonds(
                TRIANGULAR_N12_SUPERCELL, j1=0.0, j2=1.0
            ),
        )
    elif name == "square_n16_j2_05":
        kind = "square_plaquette"
        supports = SQUARE_N16_SUPPORTS
        site_count = 16
        shells = (
            square_j1j2_bonds(4, 4, j1=1.0, j2=0.0, periodic=True),
            square_j1j2_bonds(4, 4, j1=0.0, j2=1.0, periodic=True),
        )
    else:
        raise ValueError(f"unknown operator-correlator case: {name}")
    basis, configurations = four_cell_schur_sector_basis_matrix(
        kind, 0, target_m2=0, supports=supports
    )
    if not np.array_equal(basis, data["singlet_basis"]):
        raise RuntimeError("operator-correlator and common singlet bases disagree")
    reduced = []
    for bonds in shells:
        _rows, _columns, _values, checked, hamiltonian = _coo_sparse(
            site_count, bonds
        )
        if checked != configurations:
            raise RuntimeError("operator-correlator fixed-M bases disagree")
        matrix = np.asarray(
            basis.conj().T @ (hamiltonian @ basis), dtype=np.complex128
        )
        scale = float(np.max(np.sum(np.abs(matrix), axis=1)))
        if not np.isfinite(scale) or scale <= 1.0e-12:
            raise RuntimeError("a reduced shell operator has zero scale")
        reduced.append(matrix / scale)
    first, second = reduced
    return np.stack(
        (
            first,
            second,
            first @ first,
            second @ second,
            0.5 * (first @ second + second @ first),
        )
    )


def _case_data(name: str) -> dict[str, object]:
    data = dict(_base_case_data(name))
    operators = _reduced_shell_operators(name, data)
    scramble_seed = 2026092231 if name.startswith("triangular") else 2026092232
    permutation = np.random.default_rng(scramble_seed).permutation(
        operators.shape[1]
    )
    scrambled = operators[:, permutation][:, :, permutation]
    data.update(
        {
            "physical_operators": jnp.asarray(operators),
            "scrambled_operators": jnp.asarray(scrambled),
            "scramble_seed": scramble_seed,
        }
    )
    return data


def _init_params(key: jax.Array, data: dict[str, object], *, correlated: bool):
    params: dict[str, object] = {
        "backbone": init_su2_four_cell_sector_vit(key, data["config"])
    }
    if correlated:
        params["operator_correlator"] = {
            "real": jnp.zeros((len(OPERATOR_LABELS),), dtype=jnp.float64),
            "imag": jnp.zeros((len(OPERATOR_LABELS),), dtype=jnp.float64),
        }
    return params


def _operator_channels(coefficients: jax.Array, operators: jax.Array) -> jax.Array:
    """Apply the five registered scalar words to one reduced state."""

    return jnp.einsum("kij,j->ki", operators, coefficients)


def _state(params: object, data: dict[str, object], operators) -> jax.Array:
    logs = su2_four_cell_sector_log_amplitudes(
        params["backbone"], data["actions"], data["config"]
    )
    shift = jax.lax.stop_gradient(jnp.max(jnp.real(logs)))
    coefficients = jnp.exp(logs - shift)
    if operators is not None:
        channels = _operator_channels(coefficients, operators)
        values = params["operator_correlator"]
        weights = values["real"] + 1.0j * values["imag"]
        coefficients = coefficients + jnp.einsum("k,ki->i", weights, channels)
    return reynolds_project_state(data["basis"] @ coefficients, data["indices"])


def operator_scalar_frontier(
    *,
    seeds: tuple[int, ...],
    steps: int,
    learning_rate: float = 1.0e-2,
) -> dict[str, object]:
    """Compare physical and isospectral-scrambled scalar operator words."""

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
                    jax.random.PRNGKey(seed), data, correlated=operators is not None
                )

                def state_fn(candidate):
                    return _state(candidate, data, operators)

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
                    learning_rate=learning_rate,
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
                    "outer_full_h_calls": int(steps + 2),
                    "internal_shell_actions_per_state": (
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
                    raise RuntimeError("matched operator arms start differently")

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
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "operator_labels": list(OPERATOR_LABELS),
            "operator_shell_action_counts": list(OPERATOR_SHELL_ACTION_COUNTS),
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
            "The physical arm applies shell-resolved SU2-scalar J1/J2 words "
            "to configuration-dependent multiplicity coefficients. The "
            "scrambled arm is related by a coefficient-basis permutation and "
            "has identical spectra, parameters, initialization, and internal "
            "operator-action count. Raw-backbone cost is lower and is reported "
            "separately; an energy win is not yet an efficiency win."
        ),
    }


__all__ = ["ARM_IDS", "OPERATOR_LABELS", "operator_scalar_frontier"]
