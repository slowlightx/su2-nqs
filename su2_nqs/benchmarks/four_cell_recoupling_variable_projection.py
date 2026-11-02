"""Trainable variable-projection form of overlapping Schur recoupling.

The backbone supplies one Gamma-A1 reduced vector.  At every objective
evaluation the linear coefficients in ``{psi, O_k psi}`` are eliminated by a
small Rayleigh--Ritz solve, and gradients flow only through the shared neural
backbone.  This removes the poorly conditioned joint Adam head diagnosed by
the frozen-backbone calibration while retaining a strict variational state.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from ..models.su2_four_cell_sector_vit import (
    init_su2_four_cell_sector_vit,
    su2_four_cell_sector_log_amplitudes,
)
from ..vmc.adam import train_exact_adam
from .four_cell_matched_frontier import _metrics, _parameter_count
from .four_cell_overlapping_recoupling_frontier import _case_data


jax.config.update("jax_enable_x64", True)


ARM_IDS = (
    "raw_a1_backbone",
    "one_lanczos_variable_projection",
    "physical_overlapping_variable_projection",
    "a1_scrambled_variable_projection",
)


def _backbone_reduced(
    params: object,
    data: dict[str, object],
) -> jax.Array:
    logs = su2_four_cell_sector_log_amplitudes(
        params, data["actions"], data["config"]
    )
    shift = jax.lax.stop_gradient(jnp.max(jnp.real(logs)))
    coefficients = jnp.exp(logs - shift)
    return data["coefficient_a1_basis"].T @ coefficients


def _raw_energy_state(
    params: object,
    data: dict[str, object],
    hamiltonian_a1: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    reduced = _backbone_reduced(params, data)
    norm = jnp.vdot(reduced, reduced).real
    energy = jnp.vdot(reduced, hamiltonian_a1 @ reduced).real / norm
    return energy, reduced / jnp.sqrt(norm)


def _variable_projection_energy_state(
    params: object,
    data: dict[str, object],
    hamiltonian_a1: jax.Array,
    operators: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    reduced = _backbone_reduced(params, data)
    if operators is None:
        channels = (reduced, hamiltonian_a1 @ reduced)
    else:
        dimension = int(hamiltonian_a1.shape[0])
        count = min(int(operators.shape[0]), dimension - 1)
        action = jnp.einsum("kij,j->ki", operators[:count], reduced)
        channels = (reduced, *(action[index] for index in range(count)))
    trial = jnp.column_stack(channels)
    basis, _triangular = jnp.linalg.qr(trial, mode="reduced")
    projected = basis.conj().T @ (hamiltonian_a1 @ basis)
    projected = 0.5 * (projected + projected.conj().T)
    values, vectors = jnp.linalg.eigh(projected)
    state = basis @ vectors[:, 0]
    return values[0].real, state / jnp.linalg.norm(state)


def recoupling_variable_projection_frontier(
    *,
    seeds: tuple[int, ...],
    steps: int,
    learning_rate: float = 1.0e-2,
) -> dict[str, object]:
    """Train raw, one-Lanczos, physical, and scrambled VP objectives."""

    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct seeds are required")
    if steps < 1:
        raise ValueError("steps must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    cases = {}
    for case_name in ("triangular_n12_j2_0125", "square_n16_j2_05"):
        data = _case_data(case_name)
        physical_a1 = np.asarray(data["physical_a1_basis"])
        hamiltonian_numpy = physical_a1.T @ (data["hamiltonian"] @ physical_a1)
        hamiltonian_numpy = 0.5 * (hamiltonian_numpy + hamiltonian_numpy.T)
        hamiltonian_a1 = jnp.asarray(hamiltonian_numpy)
        effective_operator_count = min(
            len(data["operator_labels"]), hamiltonian_numpy.shape[0] - 1
        )
        runs = {arm: [] for arm in ARM_IDS}
        for seed in seeds:
            initial_params = init_su2_four_cell_sector_vit(
                jax.random.PRNGKey(seed), data["config"]
            )
            initial_hash_reference = None
            for arm_id, operators in (
                (ARM_IDS[0], "raw"),
                (ARM_IDS[1], None),
                (ARM_IDS[2], data["physical_operators"]),
                (ARM_IDS[3], data["scrambled_operators"]),
            ):
                params = jax.tree_util.tree_map(lambda value: value, initial_params)
                leaves = jax.tree_util.tree_leaves(params)
                initial_hash = tuple(np.asarray(leaf).tobytes() for leaf in leaves)
                if initial_hash_reference is None:
                    initial_hash_reference = initial_hash
                elif initial_hash != initial_hash_reference:
                    raise RuntimeError("variable-projection arms start differently")

                def energy_state(candidate):
                    if arm_id == ARM_IDS[0]:
                        return _raw_energy_state(
                            candidate, data, hamiltonian_a1
                        )
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
                state = physical_a1 @ np.asarray(reduced)
                state /= np.linalg.norm(state)
                row = {
                    "arm_id": arm_id,
                    "seed": int(seed),
                    "parameter_count": _parameter_count(trained),
                    "initial_energy_total": initial_energy,
                    "wall_seconds": time.perf_counter() - started,
                    "outer_objective_calls": int(steps + 2),
                    "operator_directions_per_objective": (
                        0
                        if arm_id == ARM_IDS[0]
                        else 1
                        if arm_id == ARM_IDS[1]
                        else effective_operator_count
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
                "median_wall_seconds": float(
                    np.median([row["wall_seconds"] for row in rows])
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
                "physical_beats_one_lanczos_median": physical[
                    "median_energy_error_total"
                ]
                < lanczos["median_energy_error_total"],
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
            "effective_operator_count": int(effective_operator_count),
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
            "The low-dimensional linear coefficients are solved by a "
            "differentiable Ritz projection at every step; only the common "
            "SU2/Gamma-A1 backbone is trained. Physical and scrambled arms "
            "have identical dimensions and objective calls. This exact-small "
            "calibration uses a dense reduced Hamiltonian and is not yet a "
            "scalable sampled-VMC implementation."
        ),
    }


__all__ = [
    "ARM_IDS",
    "recoupling_variable_projection_frontier",
]
