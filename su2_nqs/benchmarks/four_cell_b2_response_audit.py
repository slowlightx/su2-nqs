"""Untouched-seed audit of the development-selected square B2 response loop."""

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
from .four_cell_response_operator_frontier import _moment_panel, _summary
from .symmetry_labelled_response import build_symmetry_labelled_response_a1


jax.config.update("jax_enable_x64", True)


ARM_IDS = (
    "physical_overlapping_baseline",
    "B2_J2_response_augmented",
    "scrambled_B2_response_augmented",
    "one_lanczos_control",
)


def square_b2_response_audit(
    *,
    seeds: tuple[int, ...],
    steps: int,
    learning_rate: float,
) -> dict[str, object]:
    """Audit one frozen B2-return hypothesis on untouched seeds."""

    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("the B2 audit requires exactly five distinct seeds")
    if steps != 1000 or learning_rate != 0.01:
        raise ValueError("the B2 audit hyperparameters are frozen")
    case_name = "square_n16_j2_05"
    data = _case_data(case_name)
    physical_a1 = np.asarray(data["physical_a1_basis"])
    hamiltonian_numpy = physical_a1.T @ (data["hamiltonian"] @ physical_a1)
    hamiltonian_numpy = 0.5 * (hamiltonian_numpy + hamiltonian_numpy.T)
    hamiltonian_a1 = jnp.asarray(hamiltonian_numpy)
    response = build_symmetry_labelled_response_a1(
        case_name,
        physical_a1,
        np.asarray(data["singlet_basis"]),
        fixed_m_spin_basis(16, 0),
    )
    b2_index = response.labels.index("B2_J2_nematic_return")
    base = np.asarray(data["physical_operators"])
    banks: dict[str, np.ndarray | None] = {
        ARM_IDS[0]: base,
        ARM_IDS[1]: np.concatenate(
            (base, response.operators[b2_index : b2_index + 1])
        ),
        ARM_IDS[2]: np.concatenate(
            (base, response.scrambled_operators[b2_index : b2_index + 1])
        ),
        ARM_IDS[3]: None,
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
            operators = banks[arm_id]
            params = jax.tree_util.tree_map(lambda value: value, initial_params)
            initial_hash = tuple(
                np.asarray(leaf).tobytes()
                for leaf in jax.tree_util.tree_leaves(params)
            )
            if initial_hash_reference is None:
                initial_hash_reference = initial_hash
            elif initial_hash != initial_hash_reference:
                raise RuntimeError("B2 audit arms did not start identically")

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
            runs[arm_id].append(
                {
                    "arm_id": arm_id,
                    "seed": int(seed),
                    "parameter_count": _parameter_count(trained),
                    "initial_energy_total": initial_energy,
                    "wall_seconds": time.perf_counter() - started,
                    "outer_objective_calls": int(steps + 2),
                    "operator_directions_per_objective": int(
                        1 if operators is None else len(operators)
                    ),
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
    physical = summaries[ARM_IDS[1]]
    scrambled = summaries[ARM_IDS[2]]
    paired_baseline = [
        left < right
        for left, right in zip(
            physical["all_seed_energy_errors_total"],
            baseline["all_seed_energy_errors_total"],
            strict=True,
        )
    ]
    paired_scrambled = [
        left < right
        for left, right in zip(
            physical["all_seed_energy_errors_total"],
            scrambled["all_seed_energy_errors_total"],
            strict=True,
        )
    ]
    gates = {
        "B2_at_least_four_of_five_paired_baseline_wins": sum(paired_baseline) >= 4,
        "B2_improves_baseline_worst_seed": physical[
            "maximum_energy_error_total"
        ]
        < baseline["maximum_energy_error_total"],
        "B2_reduces_baseline_seed_range": physical[
            "seed_range_energy_error_total"
        ]
        < baseline["seed_range_energy_error_total"],
        "B2_at_least_four_of_five_paired_scrambled_wins": sum(paired_scrambled) >= 4,
        "B2_beats_scrambled_median_variance": physical[
            "median_variance_total"
        ]
        < scrambled["median_variance_total"],
        "B2_beats_baseline_response_moments": physical[
            "median_response_m0_relative_error"
        ]
        < baseline["median_response_m0_relative_error"],
        "B2_beats_scrambled_response_moments": physical[
            "median_response_m0_relative_error"
        ]
        < scrambled["median_response_m0_relative_error"],
        "B2_worst_seed_below_1e-6": physical["maximum_energy_error_total"]
        < 1.0e-6,
    }
    return {
        "seeds": list(seeds),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "n_sites": 16,
        "case": case_name,
        "exact_energy_total": exact_energy,
        "exact_response_m0": exact_moments,
        "response_certificate": response.diagnostics,
        "runs": runs,
        "summary": summaries,
        "paired_B2_vs_baseline_win_flags": paired_baseline,
        "paired_B2_vs_scrambled_win_flags": paired_scrambled,
        "gates": gates,
        "gates_passed": bool(all(gates.values())),
        "claim_boundary": (
            "B2 was selected post hoc from the frozen 1111-1115 development "
            "artifact, then frozen before these untouched seeds. Physical and "
            "scrambled B2 arms add one isospectral direction to the identical "
            "eight-operator overlapping-Schur bank. A pass supports an N16 "
            "seed-robust high-energy error-filter interpretation only; it does "
            "not establish large-system or thermodynamic physics."
        ),
    }


__all__ = ["ARM_IDS", "square_b2_response_audit"]
