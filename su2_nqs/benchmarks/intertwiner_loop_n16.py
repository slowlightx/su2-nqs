"""Exact-N16 gate for a coherent SU(2) intertwiner-loop Spatial-ViT.

The neural residual is generated in the complete four-cell singlet
multiplicity basis, projected into a sealed ``Gamma-A1`` basis, and added to a
jointly optimized two-Lanczos carrier.  A deterministic orthogonal rotation
inside the same ``Gamma-A1`` space is the matched spatial-scramble control.
No magnetic index is visible to the neural model.
"""

from __future__ import annotations

import time
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..models.intertwiner_loop_vit import (
    DEFAULT_PAIRING_FRAMES,
    IntertwinerLoopViTConfig,
    init_intertwiner_loop_vit,
    intertwiner_loop_canonical_coefficients,
    intertwiner_loop_parameter_count,
    project_intertwiner_loop_coefficients,
)
from ..su2.local_schur_fusion import four_cell_schur_states
from ..vmc.adam import train_exact_adam
from .spectral_error_tomography import (
    krylov_columns,
    lowest_subspace_state,
    normalized_state,
    orthogonal_weight,
    orthonormal_column_space,
    state_metrics,
)


jax.config.update("jax_enable_x64", True)


ARM_PHYSICAL = "coherent_intertwiner_loop_vit"
ARM_SINGLE = "single_frame_intertwiner_loop_vit"
ARM_SCRAMBLED = "spatial_scrambled_intertwiner_loop_vit"


def _haar_orthogonal(dimension: int, seed: int) -> np.ndarray:
    raw = np.random.default_rng(seed).normal(size=(dimension, dimension))
    orthogonal, triangular = np.linalg.qr(raw)
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    result = orthogonal * signs[None, :]
    result.setflags(write=False)
    return result


def _normalized_jax(vector: jax.Array) -> jax.Array:
    norm = jnp.sqrt(jnp.maximum(jnp.vdot(vector, vector).real, 1.0e-28))
    return vector / norm


def _joint_state(
    params: dict[str, object],
    config: IntertwinerLoopViTConfig,
    coefficient_a1_basis: jax.Array,
    carrier_basis: jax.Array,
    scramble: jax.Array | None,
) -> jax.Array:
    loop = project_intertwiner_loop_coefficients(
        intertwiner_loop_canonical_coefficients(params["model"], config),
        coefficient_a1_basis,
    )
    if scramble is not None:
        loop = scramble @ loop
    loop = _normalized_jax(loop)
    carrier = carrier_basis @ params["carrier_coordinates"]
    state = carrier + params["residual_scale"] * loop
    return _normalized_jax(state)


def _metrics_with_per_site(
    state: Sequence[complex], hamiltonian: np.ndarray, n_sites: int
) -> dict[str, float]:
    values = state_metrics(state, hamiltonian)
    return {
        **values,
        "energy_per_site": values["energy_total"] / n_sites,
        "variance_per_site": values["variance_total"] / n_sites,
    }


def _tangent_rank(
    params: object,
    config: IntertwinerLoopViTConfig,
    coefficient_a1_basis: np.ndarray | None,
) -> tuple[int, float]:
    def output(candidate):
        canonical = intertwiner_loop_canonical_coefficients(candidate, config)
        if coefficient_a1_basis is None:
            return canonical
        return project_intertwiner_loop_coefficients(
            canonical, jnp.asarray(coefficient_a1_basis)
        )

    jacobian = jax.jacrev(output)(params)
    rows = config.coefficient_dimension if coefficient_a1_basis is None else coefficient_a1_basis.shape[1]
    packed = np.concatenate(
        [
            np.asarray(leaf).reshape(rows, -1)
            for leaf in jax.tree_util.tree_leaves(jacobian)
        ],
        axis=1,
    )
    singular = np.linalg.svd(packed, compute_uv=False)
    tolerance = max(float(singular[0]) * 1.0e-10, 1.0e-13)
    rank = int(np.count_nonzero(singular > tolerance))
    smallest_kept = float(singular[rank - 1]) if rank else 0.0
    return rank, smallest_kept


def _train_arm(
    *,
    seed: int,
    config: IntertwinerLoopViTConfig,
    coefficient_a1_basis: np.ndarray,
    carrier_basis: np.ndarray,
    initial_carrier_coordinates: np.ndarray,
    hamiltonian: np.ndarray,
    scramble: np.ndarray | None,
    steps: int,
    learning_rate: float,
    initial_residual_scale: float,
) -> tuple[dict[str, object], np.ndarray]:
    model = init_intertwiner_loop_vit(jax.random.PRNGKey(seed), config)
    params = {
        "model": model,
        "carrier_coordinates": jnp.asarray(initial_carrier_coordinates),
        "residual_scale": jnp.asarray(initial_residual_scale, dtype=jnp.float64),
    }
    a1 = jnp.asarray(coefficient_a1_basis)
    carrier = jnp.asarray(carrier_basis)
    matrix = jnp.asarray(hamiltonian)
    scramble_jax = None if scramble is None else jnp.asarray(scramble)

    def state(candidate):
        return _joint_state(candidate, config, a1, carrier, scramble_jax)

    def loss(candidate):
        vector = state(candidate)
        return jnp.vdot(vector, matrix @ vector).real

    initial_energy = float(loss(params))
    started = time.perf_counter()
    trained, optimizer = train_exact_adam(
        params,
        loss_fn=loss,
        steps=steps,
        learning_rate=learning_rate,
        record_every=max(1, steps // 8),
    )
    wall_seconds = time.perf_counter() - started
    final = np.asarray(state(trained), dtype=np.complex128)
    final = normalized_state(final)
    return (
        {
            "seed": int(seed),
            "parameter_count": int(
                intertwiner_loop_parameter_count(trained["model"])
                + np.size(np.asarray(trained["carrier_coordinates"]))
                + 1
            ),
            "initial_energy_total": initial_energy,
            "residual_scale": float(np.asarray(trained["residual_scale"])),
            "wall_seconds": wall_seconds,
            "full_h_calls": int(steps + 2),
            "optimizer": optimizer,
            "metrics": _metrics_with_per_site(final, hamiltonian, 16),
        },
        final,
    )


def intertwiner_loop_n16_comparison(
    *,
    coefficient_a1_basis: np.ndarray,
    physical_a1_basis: np.ndarray,
    carrier: Sequence[complex],
    two_lanczos: Sequence[complex],
    hamiltonian: np.ndarray,
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 320,
    learning_rate: float = 3.0e-3,
    initial_residual_scale: float = 2.0e-2,
    scramble_seed: int = 2026092901,
    embedding_dim: int = 4,
    head_count: int = 1,
    depth: int = 1,
    mlp_ratio: int = 2,
) -> dict[str, object]:
    """Run the exact-small coherent/single/scrambled architecture gate."""

    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if steps < 1 or learning_rate <= 0.0 or initial_residual_scale <= 0.0:
        raise ValueError("training controls must be positive")
    a1 = np.asarray(coefficient_a1_basis, dtype=np.float64)
    physical = np.asarray(physical_a1_basis, dtype=np.float64)
    matrix = np.asarray(hamiltonian, dtype=np.float64)
    if a1.shape[0] != len(four_cell_schur_states("square_plaquette")):
        raise ValueError("coefficient basis is not the four-plaquette singlet basis")
    if physical.shape[1] != a1.shape[1] or matrix.shape != (a1.shape[1],) * 2:
        raise ValueError("Gamma-A1 basis and Hamiltonian shapes disagree")
    a1_gram_error = float(np.max(np.abs(a1.T @ a1 - np.eye(a1.shape[1]))))
    physical_gram_error = float(
        np.max(np.abs(physical.T @ physical - np.eye(physical.shape[1])))
    )
    if max(a1_gram_error, physical_gram_error) > 2.0e-11:
        raise ValueError("Gamma-A1 bases must be orthonormal")
    if not np.allclose(matrix, matrix.T, atol=2.0e-11, rtol=0.0):
        raise ValueError("Hamiltonian must be real symmetric")

    carrier_vector = normalized_state(carrier)
    two_lanczos_vector = normalized_state(two_lanczos)
    k2_columns = krylov_columns(carrier_vector, matrix, degree=2)
    carrier_basis, carrier_rank, _ = orthonormal_column_space(k2_columns)
    initial_coordinates = carrier_basis.conj().T @ two_lanczos_vector
    replay = normalized_state(carrier_basis @ initial_coordinates)
    two_lanczos_replay_error = float(
        min(
            np.linalg.norm(replay - two_lanczos_vector),
            np.linalg.norm(replay + two_lanczos_vector),
        )
    )
    if two_lanczos_replay_error > 2.0e-11:
        raise RuntimeError("two-Lanczos carrier is outside its registered K2 space")

    coherent_config = IntertwinerLoopViTConfig(
        "square_plaquette",
        ((0.5, 0.5), (2.5, 0.5), (0.5, 2.5), (2.5, 2.5)),
        frame_orders=DEFAULT_PAIRING_FRAMES,
        embedding_dim=embedding_dim,
        head_count=head_count,
        depth=depth,
        mlp_ratio=mlp_ratio,
    )
    single_config = IntertwinerLoopViTConfig(
        "square_plaquette",
        coherent_config.cell_coordinates,
        frame_orders=(DEFAULT_PAIRING_FRAMES[0],),
        embedding_dim=embedding_dim,
        head_count=head_count,
        depth=depth,
        mlp_ratio=mlp_ratio,
    )
    reference_model = init_intertwiner_loop_vit(
        jax.random.PRNGKey(seeds[0]), coherent_config
    )
    canonical_rank, canonical_smallest = _tangent_rank(
        reference_model, coherent_config, None
    )
    a1_rank, a1_smallest = _tangent_rank(reference_model, coherent_config, a1)

    coherent_canonical = np.asarray(
        intertwiner_loop_canonical_coefficients(reference_model, coherent_config)
    )
    single_canonical = np.asarray(
        intertwiner_loop_canonical_coefficients(reference_model, single_config)
    )
    pair_q = np.asarray(
        [state.pair_q for state in four_cell_schur_states("square_plaquette")]
    )

    def positive_q_weight(values: np.ndarray) -> float:
        norm = float(np.vdot(values, values).real)
        return float(np.vdot(values[pair_q > 0], values[pair_q > 0]).real / norm)

    coherent_positive_q = positive_q_weight(coherent_canonical)
    single_positive_q = positive_q_weight(single_canonical)
    scramble = _haar_orthogonal(a1.shape[1], scramble_seed)
    runs = {ARM_PHYSICAL: [], ARM_SINGLE: [], ARM_SCRAMBLED: []}
    states: dict[str, list[np.ndarray]] = {arm: [] for arm in runs}
    for seed in seeds:
        for arm, config, rotation in (
            (ARM_PHYSICAL, coherent_config, None),
            (ARM_SINGLE, single_config, None),
            (ARM_SCRAMBLED, coherent_config, scramble),
        ):
            report, final = _train_arm(
                seed=seed,
                config=config,
                coefficient_a1_basis=a1,
                carrier_basis=carrier_basis,
                initial_carrier_coordinates=initial_coordinates,
                hamiltonian=matrix,
                scramble=rotation,
                steps=steps,
                learning_rate=learning_rate,
                initial_residual_scale=initial_residual_scale,
            )
            report["arm_id"] = arm
            report["weight_outside_degree_four_krylov"] = orthogonal_weight(
                final, krylov_columns(carrier_vector, matrix, degree=4)
            )
            runs[arm].append(report)
            states[arm].append(final)

    baselines = {}
    baseline_states = {}
    for degree in (2, 4, 5, 6):
        state, rank = lowest_subspace_state(
            krylov_columns(carrier_vector, matrix, degree=degree), matrix
        )
        name = f"degree_{degree}_krylov"
        baseline_states[name] = state
        baselines[name] = {
            **_metrics_with_per_site(state, matrix, 16),
            "rank": int(rank),
        }
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    exact = normalized_state(eigenvectors[:, 0])
    baselines["exact_gamma_a1"] = _metrics_with_per_site(exact, matrix, 16)
    baselines["projected_vmc"] = _metrics_with_per_site(carrier_vector, matrix, 16)
    baselines["two_lanczos"] = _metrics_with_per_site(
        two_lanczos_vector, matrix, 16
    )

    summaries = {}
    for arm, rows in runs.items():
        summaries[arm] = {
            "median_energy_total": float(
                np.median([row["metrics"]["energy_total"] for row in rows])
            ),
            "median_variance_total": float(
                np.median([row["metrics"]["variance_total"] for row in rows])
            ),
            "maximum_energy_error_total": float(
                max(row["metrics"]["energy_total"] - eigenvalues[0] for row in rows)
            ),
            "paired_energy_totals": [
                float(row["metrics"]["energy_total"]) for row in rows
            ],
            "paired_variance_totals": [
                float(row["metrics"]["variance_total"]) for row in rows
            ],
            "median_weight_outside_degree_four_krylov": float(
                np.median(
                    [row["weight_outside_degree_four_krylov"] for row in rows]
                )
            ),
        }

    physical_summary = summaries[ARM_PHYSICAL]
    scrambled_summary = summaries[ARM_SCRAMBLED]
    k4 = baselines["degree_4_krylov"]
    paired_energy_wins = sum(
        physical["metrics"]["energy_total"]
        < scrambled["metrics"]["energy_total"] - 1.0e-12
        for physical, scrambled in zip(
            runs[ARM_PHYSICAL], runs[ARM_SCRAMBLED], strict=True
        )
    )
    gates = {
        "recoupling_creates_nonzero_intercell_q_support": bool(
            coherent_positive_q > single_positive_q + 1.0e-10
            and single_positive_q < 1.0e-20
        ),
        "canonical_tangent_is_non_saturated": bool(
            canonical_rank < coherent_config.coefficient_dimension
        ),
        "coherent_median_energy_beats_degree_four_krylov": bool(
            physical_summary["median_energy_total"]
            < k4["energy_total"] - 1.0e-12
        ),
        "coherent_median_variance_beats_degree_four_krylov": bool(
            physical_summary["median_variance_total"]
            < k4["variance_total"] - 1.0e-12
        ),
        "coherent_beats_scramble_median_energy": bool(
            physical_summary["median_energy_total"]
            < scrambled_summary["median_energy_total"] - 1.0e-12
        ),
        "coherent_beats_scramble_in_at_least_two_seeds": bool(
            paired_energy_wins >= 2
        ),
    }
    promotion = bool(all(gates.values()))
    return {
        "system": {
            "n_sites": 16,
            "complete_singlet_dimension": int(a1.shape[0]),
            "gamma_a1_dimension": int(a1.shape[1]),
        },
        "representation_audit": {
            "coefficient_a1_gram_max_abs": a1_gram_error,
            "physical_a1_gram_max_abs": physical_gram_error,
            "two_lanczos_coordinate_replay_l2_error": two_lanczos_replay_error,
            "native_pair_singlet_configuration_count_per_frame": int(
                coherent_config.loop_configuration_count
            ),
            "pairing_frame_count": len(coherent_config.frame_orders),
            "shared_model_parameter_count": intertwiner_loop_parameter_count(
                reference_model
            ),
            "canonical_tangent_rank": canonical_rank,
            "canonical_tangent_smallest_kept_singular_value": canonical_smallest,
            "gamma_a1_tangent_rank": a1_rank,
            "gamma_a1_tangent_smallest_kept_singular_value": a1_smallest,
            "single_frame_intercell_q_positive_weight": single_positive_q,
            "coherent_frame_intercell_q_positive_weight": coherent_positive_q,
            "magnetic_indices_visible_to_network": False,
        },
        "baselines": baselines,
        "runs": runs,
        "summary": summaries,
        "gates": gates,
        "promotion_gate_passed": promotion,
        "decision": (
            "license_many_cell_sampled_prototype"
            if promotion
            else "keep_many_cell_gpu_production_locked"
        ),
        "claim_boundary": (
            "Exact N16 architecture gate. Gamma-A1 is a small codomain, so even "
            "a pass would license only a many-cell sampled prototype, not a PRL "
            "energy or scaling claim."
        ),
    }


__all__ = [
    "ARM_PHYSICAL",
    "ARM_SCRAMBLED",
    "ARM_SINGLE",
    "intertwiner_loop_n16_comparison",
]
