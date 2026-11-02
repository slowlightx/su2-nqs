"""Fair exact-support QSL frontier built on one projected pi-flux carrier.

The triangular ``4x4`` benchmark removes sampling noise while comparing the
three variational families that can later be evaluated by VMC:

* an optimized two-shell spin-Jastrow carrier followed by zero, one, or two
  Lanczos steps;
* the same carrier multiplied by a positive fixed-``Sz`` Spatial-ViT
  correlator and projected to full ``Gamma-A1``; and
* a two-layer exact-SU(2), ``Gamma-A1`` Operator-ViT acting through scalar
  bond and resonance tokens.

No full-Hamiltonian diagonalization is performed here.  The exact energy is a
stored validation anchor, and every reported variational energy is evaluated
directly on the common fixed-``M=0`` support.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from ..lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from ..models.fixed_sz_spatial_vit import (
    FixedSzSpatialViTConfig,
    fixed_sz_spatial_vit_log_amplitudes,
    fixed_sz_spatial_vit_parameter_count,
    init_fixed_sz_spatial_vit,
)
from ..models.su2_a1_operator_vit import (
    SU2A1OperatorViTConfig,
    init_su2_a1_operator_vit,
    su2_a1_operator_vit_amplitudes,
    su2_a1_operator_vit_coefficients,
    su2_a1_operator_vit_component_weights,
    su2_a1_operator_vit_parameter_count,
)
from ..operators.triangular_a1_scalar import (
    triangular_a1_scalar_connectivity,
)
from ..vmc.adam import train_exact_adam
from .clebsch_tree_exact import (
    coo_matvec,
    physical_permutation_index_table,
)
from .scalar_operator_frontier import triangular_a1_resonance_operators
from .xdiag_physical_seed_krylov import (
    SMALL_CASES,
    configuration_from_repository_integer,
    fixed_sz_sparse_hamiltonian,
    physical_seed_amplitudes,
    spin_raising_relative_norm,
)


jax.config.update("jax_enable_x64", True)


TRIANGULAR_4X4_J2_0125_EXACT_ENERGY = -8.553311087208755


@dataclass(frozen=True)
class OperatorComponentData:
    """Prevalidated response columns for the two-layer Operator-ViT."""

    configurations: np.ndarray
    integer_basis: tuple[int, ...]
    carrier: np.ndarray
    component_matrix: np.ndarray
    hamiltonian: object
    labels: tuple[str, ...]
    descriptors: tuple[tuple[float, ...], ...]
    centering_means: tuple[float, ...]
    response_scales: tuple[float, ...]
    hamiltonian_nnz: int
    scalar_operator_union_nnz: int


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("state has zero or nonfinite norm")
    return np.asarray(vector) / norm


def _energy_variance(state: np.ndarray, hamiltonian) -> tuple[float, float]:
    normalized = _normalize(state)
    action = np.asarray(hamiltonian @ normalized)
    energy = float(np.vdot(normalized, action).real)
    residual = action - energy * normalized
    return energy, float(np.vdot(residual, residual).real)


def triangular_pi_flux_operator_components(
    coupling: float = 0.125,
    *,
    carrier_override: np.ndarray | None = None,
) -> OperatorComponentData:
    """Build centered, normalized scalar words through depth two.

    ``carrier_override`` supports carrier/backbone factorial tests on the
    identical operator vocabulary.  The caller is responsible for certifying
    its symmetry sector; this routine validates only support, finiteness, and
    norm.
    """

    if not np.isfinite(coupling) or coupling < 0.0:
        raise ValueError("coupling must be finite and nonnegative")
    case = SMALL_CASES["square4"]
    hamiltonian, integer_basis = fixed_sz_sparse_hamiltonian(case, coupling)
    h1, basis_h1 = fixed_sz_sparse_hamiltonian(case, 0.0)
    h_at_one, basis_h_one = fixed_sz_sparse_hamiltonian(case, 1.0)
    if integer_basis != basis_h1 or integer_basis != basis_h_one:
        raise RuntimeError("Hamiltonian shell bases are inconsistent")
    h2 = h_at_one - h1
    configurations = np.asarray(
        [
            configuration_from_repository_integer(value, case.n_sites)
            for value in integer_basis
        ],
        dtype=np.int8,
    )
    if carrier_override is None:
        carrier = np.asarray(
            physical_seed_amplitudes(configurations, case)[:, 0], dtype=np.float64
        )
    else:
        carrier = np.asarray(carrier_override)
        if carrier.shape != (len(configurations),):
            raise ValueError("carrier_override has the wrong exact-support shape")
        if np.any(~np.isfinite(carrier.real)) or np.any(~np.isfinite(carrier.imag)):
            raise ValueError("carrier_override must be finite")
    carrier = _normalize(carrier)
    resonance = triangular_a1_resonance_operators(case, integer_basis)
    labels = ("J1_shell", "J2_shell", *resonance.keys())
    operators = (h1, h2, *resonance.values())
    union = operators[0].copy()
    for operator in operators[1:]:
        union = union + operator
    union = union.tocsr()
    union.eliminate_zeros()
    means = tuple(float(np.vdot(carrier, operator @ carrier).real) for operator in operators)
    scales = tuple(
        float(np.linalg.norm(operator @ carrier - mean * carrier))
        for operator, mean in zip(operators, means, strict=True)
    )
    if any(scale <= 1.0e-12 or not np.isfinite(scale) for scale in scales):
        raise RuntimeError("a scalar operator has no carrier response")

    def action(index: int, vector: np.ndarray) -> np.ndarray:
        return np.asarray(
            (operators[index] @ vector - means[index] * vector) / scales[index]
        )

    first = [action(index, carrier) for index in range(len(operators))]
    second = [
        action(second_index, first[first_index])
        for second_index in range(len(operators))
        for first_index in range(len(operators))
    ]
    components = np.column_stack((carrier, *first, *second))
    # Descriptors: interaction range, support size, permutation character,
    # singlet-projector character, and explicit J2-shell flag.  Their role is
    # parameter sharing, not symmetry enforcement.
    descriptors = (
        (1.0, 2.0 / 6.0, 0.0, 0.0, 0.0),
        (np.sqrt(3.0), 2.0 / 6.0, 0.0, 0.0, 1.0),
        (1.0, 4.0 / 6.0, 1.0, 0.0, 0.0),
        (1.0, 4.0 / 6.0, 0.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 0.0, 0.0),
    )
    return OperatorComponentData(
        configurations=configurations,
        integer_basis=integer_basis,
        carrier=carrier,
        component_matrix=components,
        hamiltonian=hamiltonian,
        labels=labels,
        descriptors=descriptors,
        centering_means=means,
        response_scales=scales,
        hamiltonian_nnz=int(hamiltonian.nnz),
        scalar_operator_union_nnz=int(union.nnz),
    )


def _ritz_state(
    seed: np.ndarray,
    hamiltonian,
    degree: int,
) -> tuple[np.ndarray, float, float, int]:
    columns = [np.asarray(seed)]
    for _ in range(degree):
        columns.append(np.asarray(hamiltonian @ columns[-1]))
    matrix = np.column_stack(columns)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(1.0e-14, float(singular_values[0]) * 1.0e-12)
    rank = int(np.count_nonzero(singular_values > tolerance))
    basis = left[:, :rank]
    projected = basis.conj().T @ (hamiltonian @ basis)
    energies, vectors = np.linalg.eigh((projected + projected.conj().T) / 2.0)
    state = _normalize(basis @ vectors[:, 0])
    energy, variance = _energy_variance(state, hamiltonian)
    return state, energy, variance, rank


def _full_a1_pair_jastrow_features(
    configurations: np.ndarray,
    carrier: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return every inequivalent full-D6 pair pseudopotential on the torus."""

    group = triangular_hnf_gamma_a1_group(((4, 0), (0, 4)))
    remaining = {
        (first, second)
        for first in range(16)
        for second in range(first + 1, 16)
    }
    orbits: list[tuple[tuple[int, int], ...]] = []
    while remaining:
        representative = min(remaining)
        orbit = tuple(
            sorted(
                {
                    tuple(
                        sorted(
                            (
                                int(image[representative[0]]),
                                int(image[representative[1]]),
                            )
                        )
                    )
                    for image in group
                }
            )
        )
        orbits.append(orbit)
        remaining.difference_update(orbit)
    spins = configurations.astype(np.float64) - 0.5
    raw = np.column_stack(
        [
            sum(spins[:, first] * spins[:, second] for first, second in orbit)
            for orbit in orbits
        ]
    )
    probabilities = np.abs(carrier) ** 2
    means = probabilities @ raw
    centered = raw - means[None, :]
    scales = np.sqrt(probabilities @ (centered**2))
    if np.any(scales <= 1.0e-12):
        raise RuntimeError("Jastrow shell feature has zero carrier variance")
    return centered / scales[None, :], tuple(len(orbit) for orbit in orbits)


def _optimized_jastrow_lanczos_rows(
    data: OperatorComponentData,
    *,
    seeds: tuple[int, ...],
    retain_state: bool = False,
) -> tuple[dict[str, object], ...]:
    features, orbit_sizes = _full_a1_pair_jastrow_features(
        data.configurations, data.carrier
    )
    rows = []
    for degree in range(3):
        evaluations = 0

        def state(parameters: np.ndarray) -> np.ndarray:
            log_weight = features @ parameters
            log_weight -= np.max(log_weight)
            return _normalize(data.carrier * np.exp(log_weight))

        def objective(parameters: np.ndarray) -> float:
            nonlocal evaluations
            evaluations += 1
            return _ritz_state(
                state(np.asarray(parameters)), data.hamiltonian, degree
            )[1]

        starts = [np.zeros((features.shape[1],), dtype=np.float64)]
        starts.extend(
            np.random.default_rng(seed).normal(
                0.0, 0.08, size=features.shape[1]
            )
            for seed in seeds
        )
        started = time.perf_counter()
        candidates = [
            minimize(
                objective,
                initial,
                method="L-BFGS-B",
                bounds=tuple((-3.0, 3.0) for _ in range(features.shape[1])),
                options={"maxiter": 400, "ftol": 1.0e-14, "gtol": 1.0e-10},
            )
            for initial in starts
        ]
        best = min(candidates, key=lambda value: float(value.fun))
        final_state, energy, variance, rank = _ritz_state(
            state(np.asarray(best.x)), data.hamiltonian, degree
        )
        row: dict[str, object] = {
                "arm_id": f"optimized_pi_flux_jastrow_{degree}ls",
                "family": "pi_flux_full_a1_pair_jastrow_lanczos",
                "lanczos_degree": degree,
                "variational_parameter_count": features.shape[1] + degree,
                "jastrow_pair_orbit_sizes": orbit_sizes,
                "optimized_jastrow_parameters": np.asarray(best.x).tolist(),
                "energy_total": energy,
                "energy_per_site": energy / 16.0,
                "variance_total": variance,
                "variance_per_site": variance / 16.0,
                "krylov_rank": rank,
                "optimizer_success": bool(best.success),
                "objective_evaluations": evaluations,
                "full_h_matvecs_during_objectives": evaluations
                * (2 * degree + 2),
                "wall_seconds": time.perf_counter() - started,
                "splus_relative_norm": float(
                    spin_raising_relative_norm(
                        final_state[:, None], data.integer_basis, 16
                    )[0]
                ),
            }
        if retain_state:
            row["state"] = final_state
        rows.append(row)
    return tuple(rows)


def _operator_vit_trials(
    data: OperatorComponentData,
    *,
    seeds: tuple[int, ...],
    steps: int,
    learning_rate: float,
    retain_state: bool = False,
) -> tuple[dict[str, object], ...]:
    config = SU2A1OperatorViTConfig(
        operator_labels=data.labels,
        operator_descriptors=data.descriptors,
        embedding_dim=8,
        depth=2,
        coefficient_scale=0.5,
        initialization_scale=0.04,
    )
    components = jnp.asarray(data.component_matrix, dtype=jnp.float64)
    gram = jnp.asarray(data.component_matrix.T @ data.component_matrix)
    hamiltonian_gram = jnp.asarray(
        data.component_matrix.T @ (data.hamiltonian @ data.component_matrix)
    )
    trials = []
    for seed in seeds:
        params = init_su2_a1_operator_vit(jax.random.PRNGKey(seed), config)

        def loss(candidate):
            weights = su2_a1_operator_vit_component_weights(candidate, config)
            return (weights @ hamiltonian_gram @ weights) / (
                weights @ gram @ weights
            )

        trained, training = train_exact_adam(
            params,
            loss_fn=loss,
            steps=steps,
            learning_rate=learning_rate,
            record_every=max(1, steps // 20),
        )
        state = np.asarray(
            su2_a1_operator_vit_amplitudes(trained, components, config)
        )
        energy, variance = _energy_variance(state, data.hamiltonian)
        row: dict[str, object] = {
                "arm_id": "pi_flux_exact_su2_a1_operator_vit",
                "seed": seed,
                "family": "two_layer_scalar_operator_token_attention",
                "parameter_count": su2_a1_operator_vit_parameter_count(trained),
                "effective_layer_coefficient_count": 2 * len(data.labels),
                "energy_total": energy,
                "energy_per_site": energy / 16.0,
                "variance_total": variance,
                "variance_per_site": variance / 16.0,
                "splus_relative_norm": float(
                    spin_raising_relative_norm(
                        _normalize(state)[:, None], data.integer_basis, 16
                    )[0]
                ),
                "operator_coefficients": np.asarray(
                    su2_a1_operator_vit_coefficients(trained, config)
                ).tolist(),
                "training": training,
                "scalable_full_h_call_equivalent": 2 * steps,
                "exact_small_precompute_h_matvecs": config.component_count,
                "serial_scalar_operator_depth": 2,
                "parallel_operator_branches_per_depth": len(data.labels),
            }
        if retain_state:
            row["state"] = _normalize(state)
        trials.append(row)
    return tuple(trials)


def _fixed_sz_vit_trials(
    data: OperatorComponentData,
    *,
    seeds: tuple[int, ...],
    steps: int,
    learning_rate: float,
    retain_state: bool = False,
) -> tuple[dict[str, object], ...]:
    case = SMALL_CASES["square4"]
    group = triangular_hnf_gamma_a1_group(case.supercell)
    basis = tuple(tuple(int(bit) for bit in row) for row in data.configurations)
    indices = jnp.asarray(
        physical_permutation_index_table(basis, group), dtype=jnp.int32
    )
    configurations = jnp.asarray(data.configurations, dtype=jnp.int32)
    carrier = jnp.asarray(data.carrier, dtype=jnp.float64)
    coo = data.hamiltonian.tocoo()
    rows = jnp.asarray(coo.row, dtype=jnp.int32)
    columns = jnp.asarray(coo.col, dtype=jnp.int32)
    values = jnp.asarray(coo.data, dtype=jnp.float64)
    config = FixedSzSpatialViTConfig(
        geometry="triangular",
        rows=4,
        columns=4,
        patch_rows=2,
        patch_columns=2,
        embedding_dim=8,
        head_count=2,
        depth=1,
        mlp_ratio=2,
        target_m2=0,
        initialization_scale=0.08,
        initial_distance_decay=1.0,
        use_distance_envelope=True,
    )
    trials = []
    for seed in seeds:
        params = init_fixed_sz_spatial_vit(jax.random.PRNGKey(seed), config)
        reference_logs = jax.lax.stop_gradient(
            jnp.real(
                fixed_sz_spatial_vit_log_amplitudes(
                    params, configurations, config
                )
            )
        )

        def state(candidate):
            logs = jnp.real(
                fixed_sz_spatial_vit_log_amplitudes(
                    candidate, configurations, config
                )
            )
            correlator = logs - reference_logs
            # Averaging log-correlators gives an invariant positive diagonal
            # correlator while evaluating the ViT only once per configuration.
            correlator = jnp.mean(correlator[indices], axis=0)
            correlator -= jax.lax.stop_gradient(jnp.max(correlator))
            return carrier * jnp.exp(correlator)

        def loss(candidate):
            vector = state(candidate)
            action = coo_matvec(rows, columns, values, vector)
            return jnp.vdot(vector, action).real / jnp.vdot(vector, vector).real

        trained, training = train_exact_adam(
            params,
            loss_fn=loss,
            steps=steps,
            learning_rate=learning_rate,
            record_every=max(1, steps // 20),
        )
        final_state = np.asarray(state(trained))
        energy, variance = _energy_variance(final_state, data.hamiltonian)
        row: dict[str, object] = {
                "arm_id": "pi_flux_fixed_sz_gamma_a1_spatial_vit",
                "seed": seed,
                "family": "positive_diagonal_spatial_vit_correlator",
                "parameter_count": fixed_sz_spatial_vit_parameter_count(trained),
                "energy_total": energy,
                "energy_per_site": energy / 16.0,
                "variance_total": variance,
                "variance_per_site": variance / 16.0,
                "splus_relative_norm": float(
                    spin_raising_relative_norm(
                        _normalize(final_state)[:, None], data.integer_basis, 16
                    )[0]
                ),
                "training": training,
                "scalable_full_h_call_equivalent": 2 * steps,
                "gamma_a1_group_order": len(group),
                "serial_attention_depth": config.depth,
                "attention_tokens": config.token_count,
            }
        if retain_state:
            row["state"] = _normalize(final_state)
        trials.append(row)
    return tuple(trials)


def triangular_qsl_three_arm_state_panel(
    coupling: float = 0.125,
    *,
    jastrow_seeds: tuple[int, ...] = (0, 1, 2),
    vit_seeds: tuple[int, ...] = (0, 1, 2),
    operator_seeds: tuple[int, ...] = (0, 1, 2),
    vit_steps: int = 160,
    operator_steps: int = 160,
) -> tuple[OperatorComponentData, dict[str, np.ndarray], dict[str, dict[str, object]]]:
    """Return the best normalized state from every common-carrier arm.

    This is the non-serializable companion of
    :func:`triangular_qsl_three_arm_frontier`.  It exists so physics-facing
    diagnostics can act on the trained states without duplicating model or
    optimizer code.  Selection uses only the variational energy on the exact
    common support; no exact ground state enters training or model selection.
    """

    if abs(float(coupling) - 0.125) > 1.0e-14:
        raise ValueError("the preregistered state panel is J2/J1=0.125")
    if vit_steps < 1 or operator_steps < 1:
        raise ValueError("training steps must be positive")
    data = triangular_pi_flux_operator_components(coupling)
    jastrow = _optimized_jastrow_lanczos_rows(
        data, seeds=jastrow_seeds, retain_state=True
    )
    fixed_sz = _fixed_sz_vit_trials(
        data,
        seeds=vit_seeds,
        steps=vit_steps,
        learning_rate=0.01,
        retain_state=True,
    )
    operator = _operator_vit_trials(
        data,
        seeds=operator_seeds,
        steps=operator_steps,
        learning_rate=0.015,
        retain_state=True,
    )

    selected_rows = {
        "optimized_pi_flux_jastrow_two_lanczos": jastrow[-1],
        "pi_flux_fixed_sz_gamma_a1_spatial_vit": min(
            fixed_sz, key=lambda row: float(row["energy_per_site"])
        ),
        "pi_flux_exact_su2_a1_operator_vit": min(
            operator, key=lambda row: float(row["energy_per_site"])
        ),
    }
    states = {"bare_pi_flux": data.carrier.copy()}
    metadata: dict[str, dict[str, object]] = {
        "bare_pi_flux": {
            "arm_id": "bare_pi_flux",
            "selection": "fixed common carrier",
        }
    }
    for name, raw_row in selected_rows.items():
        row = dict(raw_row)
        states[name] = _normalize(np.asarray(row.pop("state")))
        row["selection"] = "minimum variational energy within registered seeds"
        metadata[name] = row
    return data, states, metadata


def _aggregate_trials(trials: tuple[dict[str, object], ...]) -> dict[str, object]:
    energies = np.asarray([row["energy_per_site"] for row in trials], dtype=float)
    return {
        "trial_count": len(trials),
        "mean_energy_per_site": float(np.mean(energies)),
        "best_energy_per_site": float(np.min(energies)),
        "replicate_standard_error_per_site": (
            0.0
            if len(energies) == 1
            else float(np.std(energies, ddof=1) / np.sqrt(len(energies)))
        ),
        "individual_energy_per_site": energies.tolist(),
    }


def triangular_qsl_three_arm_frontier(
    coupling: float = 0.125,
    *,
    jastrow_seeds: tuple[int, ...] = (0, 1, 2),
    vit_seeds: tuple[int, ...] = (0, 1, 2),
    operator_seeds: tuple[int, ...] = (0, 1, 2),
    vit_steps: int = 160,
    operator_steps: int = 160,
) -> dict[str, object]:
    """Run the common-carrier three-arm exact-support comparison."""

    if abs(float(coupling) - 0.125) > 1.0e-14:
        raise ValueError("the preregistered first comparison is J2/J1=0.125")
    if vit_steps < 1 or operator_steps < 1:
        raise ValueError("training steps must be positive")
    started = time.perf_counter()
    data = triangular_pi_flux_operator_components(coupling)
    bare_energy, bare_variance = _energy_variance(
        data.carrier, data.hamiltonian
    )
    jastrow = _optimized_jastrow_lanczos_rows(data, seeds=jastrow_seeds)
    operator = _operator_vit_trials(
        data,
        seeds=operator_seeds,
        steps=operator_steps,
        learning_rate=0.015,
    )
    fixed_sz = _fixed_sz_vit_trials(
        data,
        seeds=vit_seeds,
        steps=vit_steps,
        learning_rate=0.01,
    )
    exact_per_site = TRIANGULAR_4X4_J2_0125_EXACT_ENERGY / 16.0
    operator_summary = _aggregate_trials(operator)
    fixed_summary = _aggregate_trials(fixed_sz)
    strongest_baseline = min(row["energy_per_site"] for row in jastrow)
    group_indices = physical_permutation_index_table(
        tuple(tuple(int(bit) for bit in row) for row in data.configurations),
        triangular_hnf_gamma_a1_group(SMALL_CASES["square4"].supercell),
    )
    projected_components = np.mean(
        data.component_matrix[group_indices], axis=0
    )
    component_norms = np.linalg.norm(data.component_matrix, axis=0)
    maximum_a1_residual = float(
        np.max(
            np.linalg.norm(
                projected_components - data.component_matrix, axis=0
            )
            / component_norms
        )
    )
    operator_mean = float(operator_summary["mean_energy_per_site"])
    fixed_mean = float(fixed_summary["mean_energy_per_site"])
    bare_per_site = bare_energy / 16.0
    denominator = bare_per_site - exact_per_site
    return {
        "schema_version": 1,
        "task": "triangular_qsl_pi_flux_three_arm_frontier",
        "geometry": "triangular_4x4_periodic",
        "j2_over_j1": coupling,
        "support": "common_fixed_m0_exact_sum",
        "hamiltonian_diagonalization_performed": False,
        "exact_energy_anchor_total": TRIANGULAR_4X4_J2_0125_EXACT_ENERGY,
        "exact_energy_anchor_per_site": exact_per_site,
        "common_carrier": "projected pi/0-flux S=0 Gamma-A1",
        "bare_carrier": {
            "energy_total": bare_energy,
            "energy_per_site": bare_energy / 16.0,
            "variance_total": bare_variance,
            "variance_per_site": bare_variance / 16.0,
        },
        "operator_tokens": {
            "labels": data.labels,
            "descriptors": data.descriptors,
            "centering_means": data.centering_means,
            "response_scales": data.response_scales,
            "component_count": data.component_matrix.shape[1],
            "component_rank": int(np.linalg.matrix_rank(data.component_matrix)),
            "maximum_component_splus_relative_norm": float(
                np.max(
                    spin_raising_relative_norm(
                        data.component_matrix, data.integer_basis, 16
                    )
                )
            ),
            "maximum_component_gamma_a1_relative_residual": maximum_a1_residual,
            "hamiltonian_nnz": data.hamiltonian_nnz,
            "scalar_operator_union_nnz": data.scalar_operator_union_nnz,
            "per_layer_union_nnz_over_full_h_nnz": (
                data.scalar_operator_union_nnz / data.hamiltonian_nnz
            ),
            "two_operator_layers_over_two_lanczos_sparse_action_ratio": (
                data.scalar_operator_union_nnz / data.hamiltonian_nnz
            ),
        },
        "optimized_pi_flux_jastrow_lanczos": jastrow,
        "pi_flux_fixed_sz_vit_trials": fixed_sz,
        "pi_flux_exact_su2_a1_operator_vit_trials": operator,
        "aggregates": {
            "fixed_sz_vit": fixed_summary,
            "exact_su2_a1_operator_vit": operator_summary,
        },
        "comparisons": {
            "operator_vit_mean_minus_jastrow_two_lanczos_per_site": (
                operator_mean - strongest_baseline
            ),
            "operator_vit_mean_minus_fixed_sz_vit_mean_per_site": (
                operator_mean - fixed_mean
            ),
            "operator_vit_mean_gap_above_exact_per_site": (
                operator_mean - exact_per_site
            ),
            "fixed_sz_vit_mean_gap_above_exact_per_site": (
                fixed_mean - exact_per_site
            ),
            "operator_vit_fraction_of_bare_energy_error_removed": (
                (bare_per_site - operator_mean) / denominator
            ),
            "fixed_sz_vit_fraction_of_bare_energy_error_removed": (
                (bare_per_site - fixed_mean) / denominator
            ),
            "matched_neural_steps": vit_steps == operator_steps,
            "fixed_sz_vit_parameter_count": fixed_sz[0]["parameter_count"],
            "operator_vit_parameter_count": operator[0]["parameter_count"],
        },
        "gates": {
            "operator_vit_beats_optimized_jastrow_two_lanczos": (
                operator_summary["best_energy_per_site"]
                < strongest_baseline - 1.0e-10
            ),
            "operator_vit_beats_fixed_sz_vit_mean": (
                operator_mean < fixed_mean - 1.0e-10
            ),
            "every_operator_seed_beats_optimized_jastrow_two_lanczos": all(
                row["energy_per_site"] < strongest_baseline - 1.0e-10
                for row in operator
            ),
            "two_standard_error_operator_win_over_jastrow_two_lanczos": (
                operator_mean
                + 2.0
                * float(
                    operator_summary["replicate_standard_error_per_site"]
                )
                < strongest_baseline - 1.0e-10
            ),
            "operator_vit_exact_singlet": all(
                row["splus_relative_norm"] < 3.0e-11 for row in operator
            ),
            "fixed_sz_vit_spin_contamination_detected": any(
                row["splus_relative_norm"] > 1.0e-8 for row in fixed_sz
            ),
        },
        "wall_seconds": time.perf_counter() - started,
        "interpretation_contract": (
            "A finite-cluster variational win requires the trained exact-SU2/A1 "
            "arm to beat the jointly optimized full-D6 pair Jastrow plus two-Lanczos "
            "baseline, not merely the bare projected pi-flux carrier. N=36 is "
            "licensed only after this gate and independent-seed stability pass."
        ),
    }


def triangular_operator_vit_phase_scan(
    exact_energy_anchors: dict[float, float],
    *,
    seed: int = 0,
    steps: int = 160,
) -> dict[str, object]:
    """Test whether the variational gain is specific to the QSL window."""

    if not exact_energy_anchors:
        raise ValueError("exact_energy_anchors must be nonempty")
    rows = []
    for coupling, exact_total in sorted(exact_energy_anchors.items()):
        data = triangular_pi_flux_operator_components(float(coupling))
        jastrow = _optimized_jastrow_lanczos_rows(data, seeds=(0, 1))[-1]
        operator = _operator_vit_trials(
            data,
            seeds=(int(seed),),
            steps=steps,
            learning_rate=0.015,
        )[0]
        rows.append(
            {
                "j2_over_j1": float(coupling),
                "exact_energy_anchor_per_site": float(exact_total) / 16.0,
                "jastrow_two_lanczos_energy_per_site": jastrow[
                    "energy_per_site"
                ],
                "operator_vit_energy_per_site": operator["energy_per_site"],
                "operator_minus_jastrow_two_lanczos_per_site": (
                    operator["energy_per_site"] - jastrow["energy_per_site"]
                ),
                "operator_gap_above_exact_per_site": (
                    operator["energy_per_site"] - float(exact_total) / 16.0
                ),
                "operator_variance_per_site": operator["variance_per_site"],
            }
        )
    gains = np.asarray(
        [row["operator_minus_jastrow_two_lanczos_per_site"] for row in rows]
    )
    qsl_rows = [
        row for row in rows if 0.075 <= row["j2_over_j1"] <= 0.15
    ]
    outside_rows = [
        row for row in rows if not 0.075 <= row["j2_over_j1"] <= 0.15
    ]
    return {
        "rows": tuple(rows),
        "seed": int(seed),
        "steps": int(steps),
        "operator_wins_at_every_coupling": bool(np.all(gains < -1.0e-10)),
        "mean_gain_per_site_in_declared_qsl_window": float(
            np.mean(
                [
                    row["operator_minus_jastrow_two_lanczos_per_site"]
                    for row in qsl_rows
                ]
            )
        ),
        "mean_gain_per_site_outside_declared_qsl_window": float(
            np.mean(
                [
                    row["operator_minus_jastrow_two_lanczos_per_site"]
                    for row in outside_rows
                ]
            )
        ),
        "qsl_specific_enhancement_gate": bool(
            abs(
                np.mean(
                    [
                        row["operator_minus_jastrow_two_lanczos_per_site"]
                        for row in qsl_rows
                    ]
                )
            )
            > abs(
                np.mean(
                    [
                        row["operator_minus_jastrow_two_lanczos_per_site"]
                        for row in outside_rows
                    ]
                )
            )
        ),
        "interpretation": (
            "A gain throughout the scan establishes a compact symmetry-preserving "
            "correction, but only a gain enhanced in the frustrated window would "
            "support a QSL-specific finite-size claim."
        ),
    }


def triangular_n36_operator_connectivity_preflight(
    *, seeds: tuple[int, ...] = (0, 1, 2)
) -> dict[str, object]:
    """Count one- and two-layer sampled supports without evaluating amplitudes."""

    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    supercell = ((6, 0), (0, 6))
    hamiltonian_weights = (1.0, 0.125, 0.0, 0.0, 0.0)
    operator_union_weights = (1.0, 1.0, 1.0, 1.0, 1.0)
    rows = []
    for seed in seeds:
        configuration = np.concatenate(
            (np.ones((18,), dtype=np.int8), np.zeros((18,), dtype=np.int8))
        )
        np.random.default_rng(seed).shuffle(configuration)
        row: dict[str, object] = {"seed": int(seed)}
        for label, weights in (
            ("two_lanczos_H", hamiltonian_weights),
            ("two_layer_operator_union", operator_union_weights),
        ):
            started = time.perf_counter()
            first_states, _ = triangular_a1_scalar_connectivity(
                configuration, supercell, weights
            )
            second_states: set[tuple[int, ...]] = set()
            raw_second_entries = 0
            for first_state in first_states:
                connected, _ = triangular_a1_scalar_connectivity(
                    first_state, supercell, weights
                )
                second_states.update(connected)
                raw_second_entries += len(connected)
            row[label] = {
                "one_layer_unique_configurations": len(first_states),
                "two_layer_unique_configurations": len(second_states),
                "raw_second_layer_connectivity_entries": raw_second_entries,
                "python_connectivity_wall_seconds": time.perf_counter()
                - started,
            }
        h_row = row["two_lanczos_H"]
        o_row = row["two_layer_operator_union"]
        row["ratios_operator_over_h"] = {
            "one_layer_unique": (
                o_row["one_layer_unique_configurations"]
                / h_row["one_layer_unique_configurations"]
            ),
            "two_layer_unique": (
                o_row["two_layer_unique_configurations"]
                / h_row["two_layer_unique_configurations"]
            ),
            "raw_second_layer_entries": (
                o_row["raw_second_layer_connectivity_entries"]
                / h_row["raw_second_layer_connectivity_entries"]
            ),
        }
        rows.append(row)
    ratio_keys = tuple(rows[0]["ratios_operator_over_h"])
    return {
        "geometry": "triangular_6x6_periodic",
        "j2_over_j1": 0.125,
        "rows": tuple(rows),
        "mean_ratios_operator_over_h": {
            key: float(
                np.mean([row["ratios_operator_over_h"][key] for row in rows])
            )
            for key in ratio_keys
        },
        "amplitudes_evaluated": False,
        "claim_boundary": (
            "Counts establish a finite sampled support and batching target. "
            "Python timings exclude projected-parton determinant evaluation and "
            "are not accelerator-efficiency evidence."
        ),
    }


__all__ = [
    "OperatorComponentData",
    "TRIANGULAR_4X4_J2_0125_EXACT_ENERGY",
    "triangular_pi_flux_operator_components",
    "triangular_operator_vit_phase_scan",
    "triangular_n36_operator_connectivity_preflight",
    "triangular_qsl_three_arm_frontier",
    "triangular_qsl_three_arm_state_panel",
]
