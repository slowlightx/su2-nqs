"""Exact-small validation and triangular-6x6 RVB throughput canary."""

from __future__ import annotations

import time
from itertools import combinations, product

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix

from ..lattices.j1j2 import square_j1j2_bonds
from ..lattices.spatial_symmetry import (
    square_gamma_a1_group,
    triangular_hnf_gamma_a1_group,
)
from ..lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from ..models.rvb_covering_transformer import (
    RVBCoveringTransformerConfig,
    init_rvb_covering_transformer,
    rvb_covering_transformer_coefficients,
    rvb_covering_transformer_parameter_count,
)
from .clebsch_tree_exact import (
    graph_hamiltonian_coo,
    physical_permutation_index_table,
)
from .rvb_covering_matrix_free import (
    apply_two_dimer_flip,
    compile_projected_loop_kernel,
    enumerate_two_dimer_flips,
    flippable_two_dimer_move_indices,
    matching_partner_array,
    partner_array_matching,
    projected_valence_bond_matrix_elements,
    sample_sequential_matchings,
    sequential_matching_log_weight,
    valence_bond_hamiltonian_element,
    valence_bond_overlap,
)
from .rvb_covering_frontier import prepare_rvb_frontier_case
from .rvb_covering_oracle import (
    covering_relation_features,
    enumerate_perfect_matchings,
    fixed_m_dimer_amplitudes,
    transform_oriented_matching,
)


jax.config.update("jax_enable_x64", True)


TRIANGULAR_N36_SUPERCELL = ((6, 0), (0, 6))


def _physical_columns(configurations, matchings) -> np.ndarray:
    return np.column_stack(
        [
            fixed_m_dimer_amplitudes(configurations, matching)
            for matching in matchings
        ]
    )


def _overlap_sign_frustration(gram: np.ndarray) -> dict[str, object]:
    """Find the best real covering gauge and gauge-invariant bad triangles."""

    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be square")
    if matrix.shape[0] > 20:
        raise ValueError("exhaustive gauge audit is restricted to <=20 columns")
    edge_mask = np.triu(np.abs(matrix) > 1.0e-12, 1)
    best_negative = int(np.count_nonzero(edge_mask)) + 1
    best_gauge = None
    for tail in product((-1.0, 1.0), repeat=matrix.shape[0] - 1):
        gauge = np.asarray((1.0,) + tail)
        transformed = gauge[:, None] * matrix * gauge[None, :]
        negative = int(np.count_nonzero(edge_mask & (transformed < -1.0e-12)))
        if negative < best_negative:
            best_negative = negative
            best_gauge = gauge
    frustrated_triangles = []
    for first, second, third in combinations(range(matrix.shape[0]), 3):
        values = (
            matrix[first, second],
            matrix[second, third],
            matrix[third, first],
        )
        if min(abs(value) for value in values) > 1.0e-12 and np.prod(values) < 0.0:
            frustrated_triangles.append((first, second, third))
    assert best_gauge is not None
    return {
        "orbit_count": int(matrix.shape[0]),
        "overlap_graph_edge_count": int(np.count_nonzero(edge_mask)),
        "raw_negative_edge_count": int(
            np.count_nonzero(edge_mask & (matrix < -1.0e-12))
        ),
        "minimum_negative_edge_count_over_real_gauges": best_negative,
        "best_real_gauge": best_gauge.astype(int).tolist(),
        "gauge_invariant_frustrated_triangle_count": len(frustrated_triangles),
        "gauge_invariant_frustrated_triangles": [
            list(triangle) for triangle in frustrated_triangles
        ],
        "stoquastic_covering_gauge_exists": best_negative == 0,
    }


def exact_small_overlap_sign_frustration() -> dict[str, object]:
    """Contrast intrinsic triangular signs with the gaugeable square case."""

    result = {}
    for key, case in (
        ("triangular_n12", "triangular_n12_j2_0.125"),
        ("square_n16", "square_n16_j2_0.5"),
    ):
        data = prepare_rvb_frontier_case(case)
        columns = np.asarray(data.orbit_columns.columns)
        result[key] = _overlap_sign_frustration(columns.T @ columns)
    result["physical_conclusion"] = (
        "The square NN-RVB orbit Gram matrix admits an all-positive real "
        "gauge, whereas the triangular Gram matrix contains gauge-invariant "
        "negative cycles. Triangular sampling therefore requires a phase-aware "
        "carrier and/or absolute-overlap loop-gas sampling; a sign convention "
        "alone cannot remove the obstruction."
    )
    return result


def exact_small_transition_loop_gate() -> dict[str, object]:
    """Validate loop rules, A1 projection, JAX, and importance algebra."""

    # N=8 contains nontrivial transition loops but remains independently
    # contractible in the fixed-M physical basis.
    matching_bonds = square_j1j2_bonds(
        4, 2, j1=1.0, j2=0.0, periodic=True
    )
    hamiltonian_bonds = square_j1j2_bonds(
        4, 2, j1=1.0, j2=0.3, periodic=True
    )
    matchings = enumerate_perfect_matchings(8, matching_bonds)
    rows, columns, values, configurations = graph_hamiltonian_coo(
        8, 0, hamiltonian_bonds
    )
    hamiltonian = coo_matrix(
        (values, (rows, columns)),
        shape=(len(configurations), len(configurations)),
    ).toarray()
    physical = _physical_columns(configurations, matchings)
    partners = np.stack(
        [matching_partner_array(matching) for matching in matchings]
    )
    overlap_reference = physical.T @ physical
    hamiltonian_reference = physical.T @ hamiltonian @ physical
    overlap_loop = np.asarray(
        [
            [valence_bond_overlap(left, right) for right in partners]
            for left in partners
        ]
    )
    hamiltonian_loop = np.asarray(
        [
            [
                valence_bond_hamiltonian_element(
                    left, right, hamiltonian_bonds
                )
                for right in partners
            ]
            for left in partners
        ]
    )

    # N=4 independently checks the reduction of two A1 projectors to one
    # relative group sum, including oriented-singlet gauge signs.
    n4_bonds = square_j1j2_bonds(2, 2, j1=1.0, j2=0.0, periodic=True)
    n4_group = square_gamma_a1_group(2)
    n4_matchings = enumerate_perfect_matchings(4, n4_bonds)
    n4_rows, n4_columns, n4_values, n4_configurations = graph_hamiltonian_coo(
        4, 0, n4_bonds
    )
    n4_hamiltonian = coo_matrix(
        (n4_values, (n4_rows, n4_columns)),
        shape=(len(n4_configurations), len(n4_configurations)),
    ).toarray()
    n4_physical = _physical_columns(n4_configurations, n4_matchings)
    indices = physical_permutation_index_table(n4_configurations, n4_group)
    n4_projected = np.mean(n4_physical[indices, :], axis=0)
    projected_overlap_reference = n4_projected.T @ n4_projected
    projected_h_reference = n4_projected.T @ n4_hamiltonian @ n4_projected
    n4_partners = np.stack(
        [matching_partner_array(matching) for matching in n4_matchings]
    )
    pair_left = []
    pair_right = []
    projected_overlap = np.empty_like(projected_overlap_reference)
    projected_h = np.empty_like(projected_h_reference)
    for left in range(len(n4_matchings)):
        for right in range(len(n4_matchings)):
            pair_left.append(n4_partners[left])
            pair_right.append(n4_partners[right])
            projected_overlap[left, right], projected_h[left, right] = (
                projected_valence_bond_matrix_elements(
                    n4_partners[left],
                    n4_partners[right],
                    n4_bonds,
                    n4_group,
                )
            )
    kernel = compile_projected_loop_kernel(n4_bonds, n4_group, n_sites=4)
    jax_overlap, jax_h = kernel(
        jnp.asarray(np.stack(pair_left)), jnp.asarray(np.stack(pair_right))
    )

    coefficients = np.asarray(
        [complex(0.7 - 0.2 * index, 0.15 + 0.1 * index) for index in range(len(n4_matchings))]
    )
    proposal = np.exp(
        np.asarray(
            [
                sequential_matching_log_weight(partner, n4_bonds)
                for partner in n4_partners
            ]
        )
    )
    exact_energy = np.vdot(coefficients, projected_h @ coefficients) / np.vdot(
        coefficients, projected_overlap @ coefficients
    )
    normalized = proposal / proposal.sum()
    importance_numerator = 0.0j
    importance_denominator = 0.0j
    for left in range(len(n4_matchings)):
        for right in range(len(n4_matchings)):
            probability = normalized[left] * normalized[right]
            factor = (
                probability
                * coefficients[left].conjugate()
                * coefficients[right]
                / (proposal[left] * proposal[right])
            )
            importance_numerator += factor * projected_h[left, right]
            importance_denominator += factor * projected_overlap[left, right]
    importance_energy = importance_numerator / importance_denominator

    errors = {
        "n8_overlap_max_abs": float(
            np.max(np.abs(overlap_loop - overlap_reference))
        ),
        "n8_hamiltonian_max_abs": float(
            np.max(np.abs(hamiltonian_loop - hamiltonian_reference))
        ),
        "n4_projected_overlap_max_abs": float(
            np.max(np.abs(projected_overlap - projected_overlap_reference))
        ),
        "n4_projected_hamiltonian_max_abs": float(
            np.max(np.abs(projected_h - projected_h_reference))
        ),
        "n4_jax_overlap_max_abs": float(
            np.max(np.abs(np.asarray(jax_overlap).reshape(projected_overlap.shape) - projected_overlap))
        ),
        "n4_jax_hamiltonian_max_abs": float(
            np.max(np.abs(np.asarray(jax_h).reshape(projected_h.shape) - projected_h))
        ),
        "importance_ratio_abs_error": float(abs(importance_energy - exact_energy)),
    }
    return {
        "n8_matching_count": len(matchings),
        "n4_matching_count": len(n4_matchings),
        "errors": errors,
        "maximum_error": max(errors.values()),
        "passed": max(errors.values()) < 1.0e-12,
        "interpretation": (
            "The numerator and denominator estimators are unbiased under the "
            "accepted sequential proposal; their finite-sample ratio is "
            "consistent and is audited with independent blocks."
        ),
    }


def _gradient_norm(tree: object) -> float:
    return float(
        np.sqrt(
            sum(
                float(jnp.vdot(leaf, leaf).real)
                for leaf in jax.tree_util.tree_leaves(tree)
            )
        )
    )


def _triangular_2x2_edge_parity(partner: np.ndarray) -> np.ndarray:
    """Return mod-two dimer counts for 3 directions x 2x2 source parity."""

    result = np.zeros(12, dtype=np.int8)
    directions = ((1, 0), (0, 1), (1, 1))
    for first, second in partner_array_matching(partner):
        first_coordinate = (first % 6, first // 6)
        second_coordinate = (second % 6, second // 6)
        found = None
        for source, target in (
            (first_coordinate, second_coordinate),
            (second_coordinate, first_coordinate),
        ):
            for direction_index, direction in enumerate(directions):
                image = (
                    (source[0] + direction[0]) % 6,
                    (source[1] + direction[1]) % 6,
                )
                if image == target:
                    found = (
                        direction_index,
                        source[0] % 2,
                        source[1] % 2,
                    )
                    break
            if found is not None:
                break
        if found is None:
            raise RuntimeError("a triangular NN dimer has no positive direction")
        direction_index, parity_x, parity_y = found
        feature = 4 * direction_index + 2 * parity_y + parity_x
        result[feature] ^= 1
    return result


def _z2_phase_screen(
    signs: np.ndarray,
    numerators: np.ndarray,
    left_features: np.ndarray,
    right_features: np.ndarray,
) -> dict[str, object]:
    """Select a tiny bond-flux sign carrier and audit it on later samples."""

    if signs.shape != numerators.shape or signs.shape[:2] != left_features.shape[:2]:
        raise ValueError("phase-screen samples must share time/chain axes")
    split = signs.shape[0] // 2
    if split < 1 or split == signs.shape[0]:
        raise ValueError("phase screen requires at least two time samples")
    delta = np.bitwise_xor(left_features, right_features)

    def select(feature_delta: np.ndarray, width: int):
        best_score = -1.0
        best_bits = None
        best_phase = None
        # Fix the final bit: flipping every edge sign is redundant for the
        # even number (18) of dimers in every N=36 covering.
        for encoded in range(1 << (width - 1)):
            bits = np.asarray(
                [(encoded >> index) & 1 for index in range(width - 1)] + [0],
                dtype=np.int8,
            )
            phase = 1.0 - 2.0 * ((feature_delta @ bits) % 2)
            score = float(abs(np.mean(signs[:split] * phase[:split])))
            if score > best_score:
                best_score = score
                best_bits = bits.copy()
                best_phase = phase
        assert best_bits is not None and best_phase is not None
        audit_sign = signs[split:] * best_phase[split:]
        audit_numerator = numerators[split:] * best_phase[split:]
        denominator = float(np.mean(audit_sign))
        energy = float(np.mean(audit_numerator) / denominator)
        return {
            "selected_bits": best_bits.tolist(),
            "development_absolute_average_sign": best_score,
            "audit_average_sign": denominator,
            "audit_absolute_average_sign": abs(denominator),
            "audit_energy_per_site": energy / 36.0,
        }

    direction_delta = np.stack(
        [
            np.bitwise_xor.reduce(delta[..., 4 * direction : 4 * direction + 4], axis=-1)
            for direction in range(3)
        ],
        axis=-1,
    )
    return {
        "selection_protocol": "first_half_development_second_half_audit",
        "canonical_audit_absolute_average_sign": float(
            abs(np.mean(signs[split:]))
        ),
        "direction_only_z2": select(direction_delta, 3),
        "two_by_two_cell_z2": select(delta, 12),
        "claim_boundary": (
            "This is a low-dimensional post-sampling phase screen. The audit "
            "half was not used to select the bits, but time-adjacent samples "
            "from the same chains are not an untouched production audit."
        ),
    }


def triangular_n36_rvb_throughput_canary(
    *,
    pair_count: int = 512,
    block_count: int = 8,
    seed_left: int = 2026092301,
    seed_right: int = 2026092302,
    model_seed: int = 2026092303,
    residual_scale: float = 0.1,
) -> dict[str, object]:
    """Run one no-training N=36 loop-kernel and coefficient-gradient canary."""

    if pair_count < 1 or block_count < 2 or pair_count % block_count:
        raise ValueError("pair_count must be positive and divisible by block_count >= 2")
    if not np.isfinite(residual_scale) or residual_scale <= 0.0:
        raise ValueError("residual_scale must be positive and finite")
    n_sites = 36
    matching_bonds = triangular_hnf_j1j2_bonds(
        TRIANGULAR_N36_SUPERCELL, j1=1.0, j2=0.0
    )
    hamiltonian_bonds = triangular_hnf_j1j2_bonds(
        TRIANGULAR_N36_SUPERCELL, j1=1.0, j2=0.125
    )
    group = triangular_hnf_gamma_a1_group(TRIANGULAR_N36_SUPERCELL)

    sampling_started = time.perf_counter()
    left = sample_sequential_matchings(
        n_sites, matching_bonds, count=pair_count, seed=seed_left
    )
    right = sample_sequential_matchings(
        n_sites, matching_bonds, count=pair_count, seed=seed_right
    )
    sampling_seconds = time.perf_counter() - sampling_started

    loop_kernel = compile_projected_loop_kernel(
        hamiltonian_bonds, group, n_sites=n_sites
    )
    left_jax = jnp.asarray(left.partners)
    right_jax = jnp.asarray(right.partners)
    cold_started = time.perf_counter()
    overlaps, hamiltonian = loop_kernel(left_jax, right_jax)
    jax.block_until_ready(hamiltonian)
    loop_cold_seconds = time.perf_counter() - cold_started
    warm_started = time.perf_counter()
    overlaps_warm, hamiltonian_warm = loop_kernel(left_jax, right_jax)
    jax.block_until_ready(hamiltonian_warm)
    loop_warm_seconds = time.perf_counter() - warm_started

    matchings = tuple(
        partner_array_matching(partner)
        for partner in np.concatenate((left.partners, right.partners), axis=0)
    )
    features_started = time.perf_counter()
    features = covering_relation_features(matchings, matching_bonds, group)
    feature_compilation_seconds = time.perf_counter() - features_started
    config = RVBCoveringTransformerConfig(
        token_count=n_sites // 2,
        bond_class_count=features.bond_class_count,
        relation_class_count=features.relation_class_count,
        embedding_dim=4,
        head_count=1,
        depth=2,
        mlp_ratio=2,
    )
    params = init_rvb_covering_transformer(
        jax.random.PRNGKey(model_seed), config
    )
    nodes = jnp.asarray(features.bond_classes)
    relations = jnp.asarray(features.relation_classes)
    inverse_log_proposal = jnp.asarray(
        -left.log_unnormalized_probabilities
        - right.log_unnormalized_probabilities
    )
    importance = jnp.exp(inverse_log_proposal - jnp.max(inverse_log_proposal))

    def coefficients(candidate):
        residual = rvb_covering_transformer_coefficients(
            candidate, nodes, relations, config
        )
        return 1.0 + residual_scale * residual

    coefficient_forward = jax.jit(coefficients)
    coefficient_cold_started = time.perf_counter()
    initial_coefficients = coefficient_forward(params)
    jax.block_until_ready(initial_coefficients)
    coefficient_cold_seconds = time.perf_counter() - coefficient_cold_started
    coefficient_warm_started = time.perf_counter()
    initial_coefficients = coefficient_forward(params)
    jax.block_until_ready(initial_coefficients)
    coefficient_warm_seconds = time.perf_counter() - coefficient_warm_started

    def loss(candidate):
        values = coefficients(candidate)
        pair_coefficients = jnp.real(
            jnp.conj(values[:pair_count]) * values[pair_count:]
        )
        denominator = jnp.sum(importance * pair_coefficients * overlaps)
        numerator = jnp.sum(importance * pair_coefficients * hamiltonian)
        return numerator / denominator

    value_and_gradient = jax.jit(jax.value_and_grad(loss))
    gradient_cold_started = time.perf_counter()
    energy, gradient = value_and_gradient(params)
    jax.block_until_ready(energy)
    gradient_cold_seconds = time.perf_counter() - gradient_cold_started
    gradient_warm_started = time.perf_counter()
    energy, gradient = value_and_gradient(params)
    jax.block_until_ready(energy)
    gradient_warm_seconds = time.perf_counter() - gradient_warm_started

    values_np = np.asarray(initial_coefficients)
    pair_coefficients = np.real(
        np.conj(values_np[:pair_count]) * values_np[pair_count:]
    )
    importance_np = np.asarray(importance)
    norm_terms = importance_np * pair_coefficients * np.asarray(overlaps)
    energy_terms = importance_np * pair_coefficients * np.asarray(hamiltonian)
    block_size = pair_count // block_count
    norm_blocks = norm_terms.reshape(block_count, block_size).sum(axis=1)
    energy_blocks = energy_terms.reshape(block_count, block_size).sum(axis=1)
    block_energies = energy_blocks / norm_blocks
    block_signed_fractions = np.abs(norm_blocks) / np.abs(
        norm_terms.reshape(block_count, block_size)
    ).sum(axis=1)
    absolute_weights = np.abs(norm_terms)
    absolute_weight_ess_fraction = float(
        np.sum(absolute_weights) ** 2
        / (pair_count * np.sum(absolute_weights**2))
    )

    # Exact finite-group invariance and tangent invariance of the coefficient
    # network are checked on a complete orbit of one unseen covering.
    seed_matching = partner_array_matching(right.partners[0])
    orbit_matchings = tuple(
        transform_oriented_matching(seed_matching, image)[0] for image in group
    )
    orbit_features = covering_relation_features(
        orbit_matchings, matching_bonds, group
    )
    orbit_nodes = jnp.asarray(orbit_features.bond_classes)
    orbit_relations = jnp.asarray(orbit_features.relation_classes)
    tangent = jax.tree_util.tree_map(
        lambda value: jnp.full_like(value, 0.013), params
    )

    def orbit_coefficients(candidate):
        return rvb_covering_transformer_coefficients(
            candidate, orbit_nodes, orbit_relations, config
        )

    orbit_values, orbit_tangents = jax.jvp(
        orbit_coefficients, (params,), (tangent,)
    )
    orbit_values_np = np.asarray(orbit_values)
    orbit_tangents_np = np.asarray(orbit_tangents)
    value_invariance_error = float(
        np.max(np.abs(orbit_values_np - orbit_values_np[0]))
    )
    tangent_invariance_error = float(
        np.max(np.abs(orbit_tangents_np - orbit_tangents_np[0]))
    )

    finite = bool(
        np.isfinite(float(energy))
        and np.all(np.isfinite(block_energies))
        and np.isfinite(_gradient_norm(gradient))
    )
    signed_fraction = float(abs(np.sum(norm_terms)) / np.sum(np.abs(norm_terms)))
    gates = {
        "finite_energy_and_gradient": finite,
        "coefficient_value_invariance": value_invariance_error < 1.0e-11,
        "coefficient_tangent_invariance": tangent_invariance_error < 1.0e-10,
        "network_call_factor_independent_of_group_order": True,
        "warm_loop_kernel_under_5_seconds_local_cpu": loop_warm_seconds < 5.0,
        "warm_loss_gradient_under_1_second_local_cpu": gradient_warm_seconds < 1.0,
        "overall_signed_fraction_above_0.05": signed_fraction > 0.05,
        "every_block_signed_fraction_above_0.01": bool(
            np.min(block_signed_fractions) > 0.01
        ),
    }
    return {
        "case": "triangular_6x6_j2_0.125",
        "purpose": "no_training_matrix_free_forward_backward_canary",
        "n_sites": n_sites,
        "pair_count": pair_count,
        "block_count": block_count,
        "space_group_order": len(group),
        "hamiltonian_bond_count": len(hamiltonian_bonds),
        "architecture": {
            "basis": "nonorthogonal_oriented_nearest_neighbour_coverings",
            "carrier": "additive_equal_amplitude_short_range_RVB",
            "residual_scale": residual_scale,
            "embedding_dim": 4,
            "depth": 2,
            "parameter_count": rvb_covering_transformer_parameter_count(params),
            "bond_class_count": features.bond_class_count,
            "relation_class_count": features.relation_class_count,
            "neural_evaluations_per_loss": 2 * pair_count,
            "relative_group_loop_evaluations_per_loss": pair_count * len(group),
            "magnetic_indices_visible": False,
        },
        "proposal": {
            "left_acceptance_fraction": pair_count / left.proposal_attempts,
            "right_acceptance_fraction": pair_count / right.proposal_attempts,
            "left_rejected_dead_ends": left.rejected_dead_ends,
            "right_rejected_dead_ends": right.rejected_dead_ends,
            "log_probability_minimum": float(
                min(
                    np.min(left.log_unnormalized_probabilities),
                    np.min(right.log_unnormalized_probabilities),
                )
            ),
            "log_probability_maximum": float(
                max(
                    np.max(left.log_unnormalized_probabilities),
                    np.max(right.log_unnormalized_probabilities),
                )
            ),
        },
        "timing_seconds": {
            "sampling": sampling_seconds,
            "feature_compilation": feature_compilation_seconds,
            "projected_loop_cold": loop_cold_seconds,
            "projected_loop_warm": loop_warm_seconds,
            "coefficient_forward_cold": coefficient_cold_seconds,
            "coefficient_forward_warm": coefficient_warm_seconds,
            "loss_gradient_cold": gradient_cold_seconds,
            "loss_gradient_warm": gradient_warm_seconds,
        },
        "diagnostics": {
            "energy_total_ratio_estimate": float(energy),
            "energy_per_site_ratio_estimate": float(energy) / n_sites,
            "block_energy_total": block_energies.tolist(),
            "block_energy_standard_error_per_site": float(
                np.std(block_energies / n_sites, ddof=1) / np.sqrt(block_count)
            ),
            "gradient_norm": _gradient_norm(gradient),
            "projected_overlap_nonzero_fraction": float(
                np.count_nonzero(np.asarray(overlaps)) / pair_count
            ),
            "overall_signed_fraction": signed_fraction,
            "block_signed_fractions": block_signed_fractions.tolist(),
            "absolute_weight_ess_fraction": absolute_weight_ess_fraction,
            "coefficient_value_invariance_max_abs": value_invariance_error,
            "coefficient_tangent_invariance_max_abs": tangent_invariance_error,
            "loop_warm_replay_max_abs": float(
                max(
                    np.max(np.abs(np.asarray(overlaps_warm) - np.asarray(overlaps))),
                    np.max(
                        np.abs(
                            np.asarray(hamiltonian_warm) - np.asarray(hamiltonian)
                        )
                    ),
                )
            ),
        },
        "gates": gates,
        "passed": all(gates.values()),
        "claim_boundary": (
            "This is a no-training local throughput and estimator-stability "
            "canary. Its finite-sample energy is not a variational result, the "
            "carrier is short-range RVB rather than optimized Dirac/Z2, and no "
            "GPU-hour or QSL claim is licensed."
        ),
    }


def _n36_overlap_loop_gas_canary(
    *,
    geometry: str,
    chain_count: int = 32,
    burn_steps: int = 128,
    sample_steps: int = 128,
    thin: int = 2,
    seed: int = 2026092311,
) -> dict[str, object]:
    """Test an absolute-overlap loop-gas proposal on one 6x6 geometry.

    The target is ``|<C|P_A1|D>|`` for the equal-amplitude short-range RVB
    carrier.  Local two-dimer reconnections are chosen uniformly from the
    currently flippable set, and the Metropolis ratio includes the exact
    reverse/forward move-count correction.  This is an estimator-stability
    test, not the final neural sampler.
    """

    if min(chain_count, burn_steps, sample_steps, thin) < 1:
        raise ValueError("chain and schedule values must be positive")
    n_sites = 36
    if geometry == "triangular":
        matching_bonds = triangular_hnf_j1j2_bonds(
            TRIANGULAR_N36_SUPERCELL, j1=1.0, j2=0.0
        )
        hamiltonian_bonds = triangular_hnf_j1j2_bonds(
            TRIANGULAR_N36_SUPERCELL, j1=1.0, j2=0.125
        )
        group = triangular_hnf_gamma_a1_group(TRIANGULAR_N36_SUPERCELL)

        def covering_gauge(_partner):
            return 1.0

        gauge_name = "canonical_oriented_singlets"
    elif geometry == "square":
        matching_bonds = square_j1j2_bonds(
            6, 6, j1=1.0, j2=0.0, periodic=True
        )
        hamiltonian_bonds = square_j1j2_bonds(
            6, 6, j1=1.0, j2=0.5, periodic=True
        )
        group = square_gamma_a1_group(6)

        def covering_gauge(partner):
            # Reorient every canonical dimer from the even square sublattice
            # to the odd sublattice.  Reversing one singlet contributes -1.
            sign = 1.0
            for first, _second in partner_array_matching(partner):
                row, column = divmod(first, 6)
                if (row + column) % 2:
                    sign = -sign
            return sign

        gauge_name = "square_Marshall_A_to_B_dimer_orientation"
    else:
        raise ValueError("geometry must be 'triangular' or 'square'")
    flips = enumerate_two_dimer_flips(n_sites, matching_bonds)
    left = sample_sequential_matchings(
        n_sites, matching_bonds, count=chain_count, seed=seed
    ).partners.copy()
    right = sample_sequential_matchings(
        n_sites, matching_bonds, count=chain_count, seed=seed + 1
    ).partners.copy()
    kernel = compile_projected_loop_kernel(
        hamiltonian_bonds, group, n_sites=n_sites
    )
    overlaps_jax, hamiltonian_jax = kernel(
        jnp.asarray(left), jnp.asarray(right)
    )
    overlaps = np.array(overlaps_jax)
    hamiltonian = np.array(hamiltonian_jax)
    left_gauge = np.asarray([covering_gauge(row) for row in left])
    right_gauge = np.asarray([covering_gauge(row) for row in right])
    rng = np.random.default_rng(seed + 2)
    accepted = 0
    proposed = 0
    numerator_samples = []
    sign_samples = []
    unique_pairs = [set() for _ in range(chain_count)]
    left_feature_samples = []
    right_feature_samples = []
    started = time.perf_counter()
    total_steps = burn_steps + sample_steps * thin
    for step in range(total_steps):
        update_left = rng.random(chain_count) < 0.5
        proposed_left = left.copy()
        proposed_right = right.copy()
        current_move_counts = np.empty(chain_count, dtype=np.int32)
        proposed_move_counts = np.empty(chain_count, dtype=np.int32)
        for chain in range(chain_count):
            current = left[chain] if update_left[chain] else right[chain]
            available = flippable_two_dimer_move_indices(current, flips)
            if not available:
                current_move_counts[chain] = 0
                proposed_move_counts[chain] = 0
                continue
            index = int(available[int(rng.integers(len(available)))])
            moved = apply_two_dimer_flip(current, flips[index])
            current_move_counts[chain] = len(available)
            proposed_move_counts[chain] = len(
                flippable_two_dimer_move_indices(moved, flips)
            )
            if update_left[chain]:
                proposed_left[chain] = moved
            else:
                proposed_right[chain] = moved
        proposed_mask = current_move_counts > 0
        next_overlap_jax, next_hamiltonian_jax = kernel(
            jnp.asarray(proposed_left), jnp.asarray(proposed_right)
        )
        next_overlap = np.asarray(next_overlap_jax)
        next_hamiltonian = np.asarray(next_hamiltonian_jax)
        next_left_gauge = np.asarray(
            [covering_gauge(row) for row in proposed_left]
        )
        next_right_gauge = np.asarray(
            [covering_gauge(row) for row in proposed_right]
        )
        current_norm = left_gauge * right_gauge * overlaps
        next_norm = next_left_gauge * next_right_gauge * next_overlap
        ratio = np.zeros(chain_count, dtype=np.float64)
        nonzero = proposed_mask & (np.abs(current_norm) > 0.0)
        ratio[nonzero] = (
            np.abs(next_norm[nonzero])
            / np.abs(current_norm[nonzero])
            * current_move_counts[nonzero]
            / proposed_move_counts[nonzero]
        )
        zero_to_nonzero = (
            proposed_mask
            & (np.abs(current_norm) == 0.0)
            & (np.abs(next_norm) > 0.0)
        )
        ratio[zero_to_nonzero] = np.inf
        accept = proposed_mask & (rng.random(chain_count) < np.minimum(1.0, ratio))
        left[accept] = proposed_left[accept]
        right[accept] = proposed_right[accept]
        overlaps[accept] = next_overlap[accept]
        hamiltonian[accept] = next_hamiltonian[accept]
        left_gauge[accept] = next_left_gauge[accept]
        right_gauge[accept] = next_right_gauge[accept]
        proposed += int(np.count_nonzero(proposed_mask))
        accepted += int(np.count_nonzero(accept))

        if step >= burn_steps and (step - burn_steps) % thin == 0:
            norm = left_gauge * right_gauge * overlaps
            sign_samples.append(np.sign(norm).copy())
            numerator_samples.append(
                left_gauge
                * right_gauge
                * hamiltonian
                / np.where(np.abs(norm) > 0.0, np.abs(norm), 1.0)
            )
            if geometry == "triangular":
                left_feature_samples.append(
                    np.stack([_triangular_2x2_edge_parity(row) for row in left])
                )
                right_feature_samples.append(
                    np.stack([_triangular_2x2_edge_parity(row) for row in right])
                )
            for chain in range(chain_count):
                unique_pairs[chain].add(
                    (left[chain].tobytes(), right[chain].tobytes())
                )

    wall_seconds = time.perf_counter() - started
    signs = np.asarray(sign_samples)
    numerators = np.asarray(numerator_samples)
    mean_sign = float(np.mean(signs))
    mean_numerator = float(np.mean(numerators))
    energy_total = mean_numerator / mean_sign
    # Leave-one-chain-out jackknife retains time correlations inside each
    # chain and treats the independent initial chains as the uncertainty unit.
    chain_sign = np.mean(signs, axis=0)
    chain_numerator = np.mean(numerators, axis=0)
    jackknife = []
    for omitted in range(chain_count):
        retained = np.arange(chain_count) != omitted
        jackknife.append(
            np.mean(chain_numerator[retained]) / np.mean(chain_sign[retained])
        )
    jackknife = np.asarray(jackknife)
    jackknife_mean = float(np.mean(jackknife))
    jackknife_se = float(
        np.sqrt(
            (chain_count - 1)
            / chain_count
            * np.sum((jackknife - jackknife_mean) ** 2)
        )
    )
    gates = {
        "finite_ratio_and_uncertainty": bool(
            np.isfinite(energy_total) and np.isfinite(jackknife_se)
        ),
        "absolute_average_sign_above_0.05": abs(mean_sign) > 0.05,
        "accepted_move_fraction_above_0.05": (
            accepted / proposed if proposed else 0.0
        )
        > 0.05,
        "every_chain_moves": min(len(values) for values in unique_pairs) > 1,
    }
    phase_screen = None
    if geometry == "triangular":
        phase_screen = _z2_phase_screen(
            signs,
            numerators,
            np.asarray(left_feature_samples),
            np.asarray(right_feature_samples),
        )
    return {
        "geometry": geometry,
        "target": "absolute_projected_overlap_equal_amplitude_short_range_RVB",
        "covering_gauge": gauge_name,
        "chain_count": chain_count,
        "burn_steps": burn_steps,
        "sample_steps": sample_steps,
        "thin": thin,
        "two_dimer_flip_count": len(flips),
        "proposed_moves": proposed,
        "accepted_moves": accepted,
        "accepted_move_fraction": accepted / proposed if proposed else 0.0,
        "wall_seconds": wall_seconds,
        "average_sign": mean_sign,
        "absolute_average_sign": abs(mean_sign),
        "energy_total_ratio_estimate": energy_total,
        "energy_per_site_ratio_estimate": energy_total / n_sites,
        "jackknife_standard_error_per_site": jackknife_se / n_sites,
        "minimum_unique_pairs_per_chain": min(
            len(values) for values in unique_pairs
        ),
        "median_unique_pairs_per_chain": float(
            np.median([len(values) for values in unique_pairs])
        ),
        "z2_phase_carrier_screen": phase_screen,
        "gates": gates,
        "passed": all(gates.values()),
        "claim_boundary": (
            "Local two-dimer flips preserve winding sectors. Passing this gate "
            "only shows that absolute-overlap targeting controls the immediate "
            "sign/importance failure; nonlocal sector moves and a competitive "
            "physical carrier remain required before training."
        ),
    }


def triangular_n36_overlap_loop_gas_canary(**kwargs) -> dict[str, object]:
    """Run the non-bipartite triangular sign-stability control."""

    return _n36_overlap_loop_gas_canary(geometry="triangular", **kwargs)


def square_n36_overlap_loop_gas_canary(**kwargs) -> dict[str, object]:
    """Run the bipartite square Marshall-gauge estimator control."""

    return _n36_overlap_loop_gas_canary(geometry="square", **kwargs)


def rvb_covering_matrix_free_frontier() -> dict[str, object]:
    """Run the exact-small estimator gate before the N=36 canary."""

    exact_small = exact_small_transition_loop_gate()
    if not exact_small["passed"]:
        raise RuntimeError("exact-small RVB transition-loop validation failed")
    sign_frustration = exact_small_overlap_sign_frustration()
    n36 = triangular_n36_rvb_throughput_canary()
    loop_gas = triangular_n36_overlap_loop_gas_canary()
    square_loop_gas = square_n36_overlap_loop_gas_canary()
    return {
        "exact_small": exact_small,
        "overlap_sign_frustration": sign_frustration,
        "triangular_n36": n36,
        "triangular_n36_overlap_loop_gas": loop_gas,
        "square_n36_overlap_loop_gas_control": square_loop_gas,
        "throughput_gate_passed": bool(
            n36["gates"]["network_call_factor_independent_of_group_order"]
            and n36["gates"]["warm_loop_kernel_under_5_seconds_local_cpu"]
            and n36["gates"]["warm_loss_gradient_under_1_second_local_cpu"]
        ),
        "triangular_estimator_stability_gate_passed": bool(loop_gas["passed"]),
        "passed": bool(exact_small["passed"] and loop_gas["passed"]),
        "next_step": (
            "add nonlocal winding-sector moves and a phase-aware Dirac/Z2 "
            "covering carrier, then GPU-replay the fixed estimator"
            if loop_gas["passed"]
            else (
                "do not train the equal-amplitude triangular NN-RVB state; "
                "test a phase-aware projected Dirac/Z2 carrier or switch the "
                "triangular estimator back to spin configurations"
            )
        ),
    }


__all__ = [
    "TRIANGULAR_N36_SUPERCELL",
    "exact_small_transition_loop_gate",
    "exact_small_overlap_sign_frustration",
    "rvb_covering_matrix_free_frontier",
    "square_n36_overlap_loop_gas_canary",
    "triangular_n36_rvb_throughput_canary",
    "triangular_n36_overlap_loop_gas_canary",
]
