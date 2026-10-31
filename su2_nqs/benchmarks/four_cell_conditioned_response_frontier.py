"""Exact-small frontier for configuration-conditioned SU(2) response gates."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from ..models.su2_conditioned_response_gate import (
    init_su2_conditioned_response_gate,
    su2_conditioned_response_operators,
    su2_conditioned_response_parameter_count,
)
from ..models.su2_four_cell_sector_vit import (
    su2_four_cell_sector_log_amplitudes,
)
from ..vmc.adam import train_exact_adam
from .clebsch_tree_exact import fixed_m_spin_basis
from .four_cell_matched_frontier import _metrics
from .four_cell_overlapping_recoupling_frontier import _case_data
from .four_cell_response_operator_frontier import _moment_panel, _summary
from .symmetry_labelled_response import build_symmetry_labelled_response_a1


jax.config.update("jax_enable_x64", True)


ARM_IDS = (
    "physical_overlapping_baseline",
    "fixed_labelled_response",
    "conditioned_labelled_response",
    "conditioned_isospectral_scramble",
)


def _haar_orthogonal(dimension: int, seed: int) -> np.ndarray:
    raw = np.random.default_rng(seed).normal(size=(dimension, dimension))
    orthogonal, triangular = np.linalg.qr(raw)
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    return orthogonal * signs[None, :]


def _reduced_backbone(params: object, data: dict[str, object]) -> jax.Array:
    logs = su2_four_cell_sector_log_amplitudes(
        params["backbone"], data["actions"], data["config"]
    )
    shift = jax.lax.stop_gradient(jnp.max(jnp.real(logs)))
    coefficients = jnp.exp(logs - shift)
    return data["coefficient_a1_basis"].T @ coefficients


def _conditioned_operators(
    params: object,
    data: dict[str, object],
    response_operators: jax.Array,
    rotation: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    return su2_conditioned_response_operators(
        params,
        data["actions"],
        data["config"],
        data["coefficient_a1_basis"],
        response_operators,
        scramble_rotation=rotation,
    )


def _energy_state(
    params: object,
    data: dict[str, object],
    hamiltonian_a1: jax.Array,
    base_operators: jax.Array,
    response_operators: jax.Array,
    rotation: jax.Array,
    arm_id: str,
) -> tuple[jax.Array, jax.Array]:
    reduced = _reduced_backbone(params, data)
    if arm_id == ARM_IDS[0]:
        operators = base_operators
    elif arm_id == ARM_IDS[1]:
        operators = jnp.concatenate((base_operators, response_operators), axis=0)
    elif arm_id in ARM_IDS[2:]:
        conditioned, _gates = _conditioned_operators(
            params,
            data,
            response_operators,
            None if arm_id == ARM_IDS[2] else rotation,
        )
        operators = jnp.concatenate((base_operators, conditioned), axis=0)
    else:
        raise ValueError(f"unknown conditioned-response arm: {arm_id}")
    count = min(int(operators.shape[0]), int(hamiltonian_a1.shape[0]) - 1)
    action = jnp.einsum("kij,j->ki", operators[:count], reduced)
    trial = jnp.column_stack(
        (reduced, *(action[index] for index in range(count)))
    )
    basis, _triangular = jnp.linalg.qr(trial, mode="reduced")
    projected = basis.conj().T @ (hamiltonian_a1 @ basis)
    projected = 0.5 * (projected + projected.conj().T)
    values, vectors = jnp.linalg.eigh(projected)
    state = basis @ vectors[:, 0]
    return values[0].real, state / jnp.linalg.norm(state)


def _gate_diagnostics(
    params: object,
    data: dict[str, object],
    response_operators: jax.Array,
    rotation: jax.Array,
    labels: tuple[str, ...],
) -> dict[str, object]:
    physical, gates = _conditioned_operators(
        params, data, response_operators, None
    )
    scrambled, scrambled_gates = _conditioned_operators(
        params, data, response_operators, rotation
    )
    physical_numpy = np.asarray(physical)
    scrambled_numpy = np.asarray(scrambled)
    gate_numpy = np.asarray(gates)
    spectrum_error = max(
        float(
            np.max(
                np.abs(
                    np.linalg.eigvalsh(left) - np.linalg.eigvalsh(right)
                )
            )
        )
        for left, right in zip(physical_numpy, scrambled_numpy, strict=True)
    )
    fixed = np.asarray(response_operators)
    overlaps = {
        label: float(abs(np.vdot(fixed[index], physical_numpy[index])))
        for index, label in enumerate(labels)
    }
    return {
        "gate_mean": {
            label: float(np.mean(gate_numpy[:, index]))
            for index, label in enumerate(labels)
        },
        "gate_standard_deviation": {
            label: float(np.std(gate_numpy[:, index]))
            for index, label in enumerate(labels)
        },
        "gate_minimum": {
            label: float(np.min(gate_numpy[:, index]))
            for index, label in enumerate(labels)
        },
        "gate_maximum": {
            label: float(np.max(gate_numpy[:, index]))
            for index, label in enumerate(labels)
        },
        "conditioned_vs_fixed_frobenius_overlap": overlaps,
        "conditioned_scrambled_gate_value_error": float(
            np.max(np.abs(gate_numpy - np.asarray(scrambled_gates)))
        ),
        "conditioned_scrambled_spectrum_error": spectrum_error,
    }


def conditioned_response_frontier(
    *,
    seeds: tuple[int, ...],
    steps: int,
    learning_rate: float,
) -> dict[str, object]:
    """Train matched fixed, conditioned, and isospectral-control arms."""

    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("the conditioned-response frontier requires five seeds")
    if steps != 1000 or learning_rate != 0.01:
        raise ValueError("conditioned-response hyperparameters are frozen")

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
        parameters = init_su2_conditioned_response_gate(
            jax.random.PRNGKey(2026092601),
            data["config"],
            len(response.labels),
        )
        rotation = jnp.asarray(
            _haar_orthogonal(
                int(data["physical_a1_basis"].shape[1]),
                2026092602 if case_name.startswith("square") else 2026092603,
            )
        )
        diagnostics = _gate_diagnostics(
            parameters,
            data,
            jnp.asarray(response.operators),
            rotation,
            response.labels,
        )
        probabilities = np.exp(
            2.0
            * np.real(
                np.asarray(
                    su2_four_cell_sector_log_amplitudes(
                        parameters["backbone"], data["actions"], data["config"]
                    )
                )
            )
        )
        certificates[case_name] = {
            "response_labels": list(response.labels),
            "response_certificate": response.diagnostics,
            "conditioned_gate_diagnostics": diagnostics,
            "backbone_exhaustive_normalization_error": float(
                abs(np.sum(probabilities) - 1.0)
            ),
            "parameter_count": su2_conditioned_response_parameter_count(
                parameters
            ),
        }
        representations[case_name] = response

    case_name = "square_n16_j2_05"
    data = _case_data(case_name)
    response = representations[case_name]
    physical_a1 = np.asarray(data["physical_a1_basis"])
    hamiltonian_numpy = physical_a1.T @ (data["hamiltonian"] @ physical_a1)
    hamiltonian_numpy = 0.5 * (hamiltonian_numpy + hamiltonian_numpy.T)
    hamiltonian_a1 = jnp.asarray(hamiltonian_numpy)
    base = jnp.asarray(data["physical_operators"])
    response_operators = jnp.asarray(response.operators)
    rotation = jnp.asarray(_haar_orthogonal(physical_a1.shape[1], 2026092602))
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
        initial_params = init_su2_conditioned_response_gate(
            jax.random.PRNGKey(seed), data["config"], len(response.labels)
        )
        initial_hash_reference = None
        for arm_id in ARM_IDS:
            params = jax.tree_util.tree_map(lambda value: value, initial_params)
            initial_hash = tuple(
                np.asarray(leaf).tobytes()
                for leaf in jax.tree_util.tree_leaves(params)
            )
            if initial_hash_reference is None:
                initial_hash_reference = initial_hash
            elif initial_hash != initial_hash_reference:
                raise RuntimeError("conditioned response arms start differently")

            def energy_state(candidate):
                return _energy_state(
                    candidate,
                    data,
                    hamiltonian_a1,
                    base,
                    response_operators,
                    rotation,
                    arm_id,
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
            row = {
                "arm_id": arm_id,
                "seed": int(seed),
                "parameter_count": su2_conditioned_response_parameter_count(
                    trained
                ),
                "initial_energy_total": initial_energy,
                "wall_seconds": time.perf_counter() - started,
                "outer_objective_calls": int(steps + 2),
                "operator_directions_per_objective": (
                    len(base) if arm_id == ARM_IDS[0] else len(base) + len(response.labels)
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
            if arm_id in ARM_IDS[2:]:
                row["conditioned_gate"] = _gate_diagnostics(
                    trained,
                    data,
                    response_operators,
                    rotation,
                    response.labels,
                )
            runs[arm_id].append(row)
    exact_energy = float(data["exact_energy_total"])
    summaries = {
        arm: _summary(rows, exact_energy) for arm, rows in runs.items()
    }
    conditioned = summaries[ARM_IDS[2]]
    baseline = summaries[ARM_IDS[0]]
    fixed = summaries[ARM_IDS[1]]
    scrambled = summaries[ARM_IDS[3]]
    paired_scrambled = [
        left < right
        for left, right in zip(
            conditioned["all_seed_energy_errors_total"],
            scrambled["all_seed_energy_errors_total"],
            strict=True,
        )
    ]
    gate_standard_deviations = [
        min(row["conditioned_gate"]["gate_standard_deviation"].values())
        for row in runs[ARM_IDS[2]]
    ]
    gates = {
        "conditioned_improves_fixed_median": conditioned[
            "median_energy_error_total"
        ]
        < fixed["median_energy_error_total"],
        "conditioned_improves_fixed_worst_seed": conditioned[
            "maximum_energy_error_total"
        ]
        < fixed["maximum_energy_error_total"],
        "conditioned_improves_baseline_worst_seed": conditioned[
            "maximum_energy_error_total"
        ]
        < baseline["maximum_energy_error_total"],
        "conditioned_at_least_four_of_five_scrambled_wins": sum(
            paired_scrambled
        )
        >= 4,
        "conditioned_beats_scrambled_median_variance": conditioned[
            "median_variance_total"
        ]
        < scrambled["median_variance_total"],
        "conditioned_beats_scrambled_response_moments": conditioned[
            "median_response_m0_relative_error"
        ]
        < scrambled["median_response_m0_relative_error"],
        "conditioned_all_seeds_below_1e-5": conditioned[
            "maximum_energy_error_total"
        ]
        < 1.0e-5,
        "conditioned_gate_is_nonconstant": float(
            np.median(gate_standard_deviations)
        )
        > 0.02,
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
            "paired_conditioned_vs_scrambled_win_flags": paired_scrambled,
            "minimum_gate_standard_deviation_by_seed": gate_standard_deviations,
            "gates": gates,
            "gates_passed": bool(all(gates.values())),
        },
        "all_gates_passed": bool(all(gates.values())),
        "claim_boundary": (
            "The learned gates read only reduced q,a and spatial ViT tokens. "
            "Projection through the exact coefficient A1 basis and Hermitian "
            "coupling to prevalidated response-return operators keep every "
            "state and parameter tangent in the SU2-singlet Gamma-A1 space. "
            "The control orthogonally conjugates each conditioned operator at "
            "every parameter value, preserving its spectrum and parameter "
            "count. This is exact-small development, not N36 evidence."
        ),
    }


__all__ = [
    "ARM_IDS",
    "conditioned_response_frontier",
]
