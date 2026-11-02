"""Seed-robust exact-small frontier for symmetry-labelled response loops."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from ..models.su2_four_cell_sector_vit import init_su2_four_cell_sector_vit
from ..vmc.adam import train_exact_adam
from .clebsch_tree_exact import fixed_m_spin_basis
from .four_cell_matched_frontier import _metrics, _parameter_count
from .four_cell_overlapping_recoupling_frontier import _case_data
from .four_cell_recoupling_variable_projection import (
    _variable_projection_energy_state,
)
from .symmetry_labelled_response import build_symmetry_labelled_response_a1


jax.config.update("jax_enable_x64", True)


ARM_IDS = (
    "physical_overlapping_baseline",
    "B1_J1_response_augmented",
    "B2_J2_response_augmented",
    "A2_chirality_response_augmented",
    "all_labelled_response_augmented",
    "all_scrambled_response_augmented",
    "one_lanczos_control",
)


def _moment_panel(
    reduced: np.ndarray,
    moment_operators: np.ndarray,
    labels: tuple[str, ...],
    exact_values: dict[str, float],
) -> dict[str, object]:
    state = np.array(reduced, dtype=np.complex128, copy=True)
    state /= np.linalg.norm(state)
    values = {
        label: float(np.vdot(state, operator @ state).real)
        for label, operator in zip(labels, moment_operators, strict=True)
    }
    absolute = {
        label: abs(values[label] - exact_values[label]) for label in labels
    }
    relative = {
        label: absolute[label] / max(abs(exact_values[label]), 1.0e-12)
        for label in labels
    }
    return {
        "m0": values,
        "m0_absolute_errors": absolute,
        "m0_relative_errors": relative,
        "median_m0_relative_error": float(np.median(tuple(relative.values()))),
        "maximum_m0_relative_error": float(np.max(tuple(relative.values()))),
    }


def _summary(rows: list[dict[str, object]], exact_energy: float) -> dict[str, object]:
    errors = np.asarray(
        [float(row["metrics"]["energy_total"]) - exact_energy for row in rows]
    )
    return {
        "median_energy_error_total": float(np.median(errors)),
        "maximum_energy_error_total": float(np.max(errors)),
        "minimum_energy_error_total": float(np.min(errors)),
        "seed_range_energy_error_total": float(np.max(errors) - np.min(errors)),
        "seed_std_energy_error_total": float(np.std(errors)),
        "all_seed_energy_errors_total": errors.tolist(),
        "median_variance_total": float(
            np.median([float(row["metrics"]["variance_total"]) for row in rows])
        ),
        "median_response_m0_relative_error": float(
            np.median(
                [
                    float(row["response_moments"]["median_m0_relative_error"])
                    for row in rows
                ]
            )
        ),
        "maximum_response_m0_relative_error": float(
            np.max(
                [
                    float(row["response_moments"]["maximum_m0_relative_error"])
                    for row in rows
                ]
            )
        ),
        "median_wall_seconds": float(
            np.median([float(row["wall_seconds"]) for row in rows])
        ),
    }


def symmetry_response_operator_frontier(
    *,
    seeds: tuple[int, ...],
    steps: int,
    learning_rate: float,
) -> dict[str, object]:
    """Train the registered square response ablations and exact certificates."""

    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("at least five distinct seeds are required")
    if steps < 1:
        raise ValueError("steps must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")

    certificates = {}
    representations = {}
    for case_name in ("triangular_n12_j2_0125", "square_n16_j2_05"):
        data = _case_data(case_name)
        response = build_symmetry_labelled_response_a1(
            case_name,
            np.asarray(data["physical_a1_basis"]),
            np.asarray(data["singlet_basis"]),
            fixed_m_spin_basis(int(data["n_sites"]), 0),
        )
        representations[case_name] = response
        certificates[case_name] = {
            "labels": list(response.labels),
            "diagnostics": response.diagnostics,
        }

    case_name = "square_n16_j2_05"
    data = _case_data(case_name)
    response = representations[case_name]
    physical_a1 = np.asarray(data["physical_a1_basis"])
    hamiltonian_numpy = physical_a1.T @ (data["hamiltonian"] @ physical_a1)
    hamiltonian_numpy = 0.5 * (hamiltonian_numpy + hamiltonian_numpy.T)
    hamiltonian_a1 = jnp.asarray(hamiltonian_numpy)
    base = np.asarray(data["physical_operators"])
    labelled = np.asarray(response.operators)
    scrambled = np.asarray(response.scrambled_operators)
    operator_banks: dict[str, np.ndarray | None] = {
        ARM_IDS[0]: base,
        ARM_IDS[1]: np.concatenate((base, labelled[0:1])),
        ARM_IDS[2]: np.concatenate((base, labelled[1:2])),
        ARM_IDS[3]: np.concatenate((base, labelled[2:3])),
        ARM_IDS[4]: np.concatenate((base, labelled)),
        ARM_IDS[5]: np.concatenate((base, scrambled)),
        ARM_IDS[6]: None,
    }
    exact_reduced = physical_a1.T @ np.asarray(data["exact_state"])
    exact_reduced /= np.linalg.norm(exact_reduced)
    exact_moments = {
        label: float(np.vdot(exact_reduced, operator @ exact_reduced).real)
        for label, operator in zip(
            response.labels, response.moment_operators, strict=True
        )
    }
    runs: dict[str, list[dict[str, object]]] = {arm: [] for arm in ARM_IDS}
    for seed in seeds:
        initial_params = init_su2_four_cell_sector_vit(
            jax.random.PRNGKey(seed), data["config"]
        )
        initial_hash_reference = None
        for arm_id in ARM_IDS:
            operators = operator_banks[arm_id]
            params = jax.tree_util.tree_map(lambda value: value, initial_params)
            initial_hash = tuple(
                np.asarray(leaf).tobytes()
                for leaf in jax.tree_util.tree_leaves(params)
            )
            if initial_hash_reference is None:
                initial_hash_reference = initial_hash
            elif initial_hash != initial_hash_reference:
                raise RuntimeError("response arms did not start identically")

            def energy_state(candidate):
                return _variable_projection_energy_state(
                    candidate, data, hamiltonian_a1, operators
                )

            def loss(candidate):
                return energy_state(candidate)[0]

            initial_energy = float(loss(params))
            started = time.perf_counter()
            trained, optimizer = train_exact_adam(
                params,
                loss_fn=loss,
                steps=steps,
                learning_rate=learning_rate,
                record_every=max(1, steps // 20),
            )
            energy, reduced = energy_state(trained)
            jax.block_until_ready(energy)
            reduced_numpy = np.asarray(reduced)
            state = physical_a1 @ reduced_numpy
            state /= np.linalg.norm(state)
            operator_count = 1 if operators is None else len(operators)
            runs[arm_id].append(
                {
                    "arm_id": arm_id,
                    "seed": int(seed),
                    "parameter_count": _parameter_count(trained),
                    "initial_energy_total": initial_energy,
                    "wall_seconds": time.perf_counter() - started,
                    "outer_objective_calls": int(steps + 2),
                    "operator_directions_per_objective": int(operator_count),
                    "optimizer": optimizer,
                    "metrics": _metrics(
                        state,
                        data["hamiltonian"],
                        data["exact_state"],
                        data["singlet_basis"],
                        data["indices_numpy"],
                    ),
                    "response_moments": _moment_panel(
                        reduced_numpy,
                        response.moment_operators,
                        response.labels,
                        exact_moments,
                    ),
                }
            )
    exact_energy = float(data["exact_energy_total"])
    summaries = {
        arm: _summary(rows, exact_energy) for arm, rows in runs.items()
    }
    baseline = summaries[ARM_IDS[0]]
    labelled_summary = summaries[ARM_IDS[4]]
    scrambled_summary = summaries[ARM_IDS[5]]
    paired_baseline = [
        left < right
        for left, right in zip(
            labelled_summary["all_seed_energy_errors_total"],
            baseline["all_seed_energy_errors_total"],
            strict=True,
        )
    ]
    paired_scrambled = [
        left < right
        for left, right in zip(
            labelled_summary["all_seed_energy_errors_total"],
            scrambled_summary["all_seed_energy_errors_total"],
            strict=True,
        )
    ]
    gates = {
        "labelled_improves_baseline_every_paired_seed": bool(all(paired_baseline)),
        "labelled_improves_baseline_worst_seed": labelled_summary[
            "maximum_energy_error_total"
        ]
        < baseline["maximum_energy_error_total"],
        "labelled_reduces_seed_range": labelled_summary[
            "seed_range_energy_error_total"
        ]
        < baseline["seed_range_energy_error_total"],
        "labelled_beats_scrambled_every_paired_seed": bool(all(paired_scrambled)),
        "labelled_beats_scrambled_median_variance": labelled_summary[
            "median_variance_total"
        ]
        < scrambled_summary["median_variance_total"],
        "labelled_improves_baseline_response_moments": labelled_summary[
            "median_response_m0_relative_error"
        ]
        < baseline["median_response_m0_relative_error"],
        "labelled_all_seeds_below_1e-5": labelled_summary[
            "maximum_energy_error_total"
        ]
        < 1.0e-5,
    }
    return {
        "seeds": list(seeds),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "certificates": certificates,
        "square_n16_j2_05": {
            "n_sites": 16,
            "exact_energy_total": exact_energy,
            "response_labels": list(response.labels),
            "exact_response_m0": exact_moments,
            "runs": runs,
            "summary": summaries,
            "paired_labelled_vs_baseline_win_flags": paired_baseline,
            "paired_labelled_vs_scrambled_win_flags": paired_scrambled,
            "gates": gates,
            "gates_passed": bool(all(gates.values())),
        },
        "all_gates_passed": bool(all(gates.values())),
        "claim_boundary": (
            "The response loops are exact SU2 scalars and return to Gamma-A1 "
            "through registered nontrivial point-group sectors. The square "
            "N16 comparison uses identical backbone initialization, samples "
            "(exhaustive), steps, and full-H objective calls. Labelled and "
            "scrambled augmented banks have the same dimension and individual "
            "operator spectra. This is an exact-small seed-robustness test, "
            "not evidence for N36 scaling or a thermodynamic QSL claim."
        ),
    }


__all__ = ["ARM_IDS", "symmetry_response_operator_frontier"]
