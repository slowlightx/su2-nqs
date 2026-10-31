from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix

from su2_nqs.benchmarks.clebsch_tree_exact import (
    graph_hamiltonian_coo,
    physical_permutation_index_table,
)
from su2_nqs.benchmarks.rvb_covering_matrix_free import (
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
from su2_nqs.benchmarks.rvb_covering_oracle import (
    enumerate_perfect_matchings,
    fixed_m_dimer_amplitudes,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import square_gamma_a1_group


def _square_n4_data():
    bonds = square_j1j2_bonds(2, 2, j1=1.0, j2=0.0, periodic=True)
    matchings = enumerate_perfect_matchings(4, bonds)
    rows, columns, values, configurations = graph_hamiltonian_coo(4, 0, bonds)
    hamiltonian = coo_matrix(
        (values, (rows, columns)),
        shape=(len(configurations), len(configurations)),
    ).toarray()
    physical = np.column_stack(
        [fixed_m_dimer_amplitudes(configurations, matching) for matching in matchings]
    )
    partners = np.stack([matching_partner_array(matching) for matching in matchings])
    return bonds, matchings, configurations, hamiltonian, physical, partners


def test_transition_loop_overlap_and_hamiltonian_match_physical_basis() -> None:
    bonds, _matchings, _configs, hamiltonian, physical, partners = _square_n4_data()
    overlap_reference = physical.T @ physical
    hamiltonian_reference = physical.T @ hamiltonian @ physical
    for left in range(len(partners)):
        for right in range(len(partners)):
            np.testing.assert_allclose(
                valence_bond_overlap(partners[left], partners[right]),
                overlap_reference[left, right],
                atol=2.0e-15,
            )
            np.testing.assert_allclose(
                valence_bond_hamiltonian_element(
                    partners[left], partners[right], bonds
                ),
                hamiltonian_reference[left, right],
                atol=3.0e-15,
            )


def test_n8_long_transition_loops_match_independent_physical_contractions() -> None:
    bonds = square_j1j2_bonds(4, 2, j1=1.0, j2=0.3, periodic=True)
    matching_bonds = square_j1j2_bonds(4, 2, j1=1.0, j2=0.0, periodic=True)
    matchings = enumerate_perfect_matchings(8, matching_bonds)
    rows, columns, values, configurations = graph_hamiltonian_coo(8, 0, bonds)
    hamiltonian = coo_matrix(
        (values, (rows, columns)),
        shape=(len(configurations), len(configurations)),
    ).toarray()
    physical = np.column_stack(
        [fixed_m_dimer_amplitudes(configurations, matching) for matching in matchings]
    )
    partners = np.stack([matching_partner_array(matching) for matching in matchings])
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
                valence_bond_hamiltonian_element(left, right, bonds)
                for right in partners
            ]
            for left in partners
        ]
    )
    np.testing.assert_allclose(overlap_loop, overlap_reference, atol=3.0e-15)
    np.testing.assert_allclose(
        hamiltonian_loop, hamiltonian_reference, atol=5.0e-15
    )


def test_projected_numpy_and_jax_loop_kernels_match_explicit_a1_columns() -> None:
    bonds, _matchings, configurations, hamiltonian, physical, partners = (
        _square_n4_data()
    )
    group = square_gamma_a1_group(2)
    indices = physical_permutation_index_table(configurations, group)
    projected = np.mean(physical[indices, :], axis=0)
    overlap_reference = projected.T @ projected
    hamiltonian_reference = projected.T @ hamiltonian @ projected

    pair_left = []
    pair_right = []
    numpy_overlap = []
    numpy_hamiltonian = []
    for left in range(len(partners)):
        for right in range(len(partners)):
            pair_left.append(partners[left])
            pair_right.append(partners[right])
            overlap, element = projected_valence_bond_matrix_elements(
                partners[left], partners[right], bonds, group
            )
            numpy_overlap.append(overlap)
            numpy_hamiltonian.append(element)
            np.testing.assert_allclose(
                overlap, overlap_reference[left, right], atol=3.0e-15
            )
            np.testing.assert_allclose(
                element, hamiltonian_reference[left, right], atol=4.0e-15
            )

    kernel = compile_projected_loop_kernel(bonds, group, n_sites=4)
    overlap_jax, hamiltonian_jax = kernel(
        jnp.asarray(np.stack(pair_left)), jnp.asarray(np.stack(pair_right))
    )
    np.testing.assert_allclose(overlap_jax, numpy_overlap, atol=3.0e-15)
    np.testing.assert_allclose(hamiltonian_jax, numpy_hamiltonian, atol=4.0e-15)


def test_rejection_proposal_weights_give_exact_importance_ratio() -> None:
    bonds, matchings, _configs, _hamiltonian, _physical, partners = _square_n4_data()
    group = square_gamma_a1_group(2)
    proposal = np.asarray(
        [np.exp(sequential_matching_log_weight(partner, bonds)) for partner in partners]
    )
    assert np.all(proposal > 0.0)
    overlap = np.empty((len(matchings), len(matchings)))
    hamiltonian = np.empty_like(overlap)
    for left in range(len(matchings)):
        for right in range(len(matchings)):
            overlap[left, right], hamiltonian[left, right] = (
                projected_valence_bond_matrix_elements(
                    partners[left], partners[right], bonds, group
                )
            )
    coefficients = np.asarray((1.0 + 0.2j, -0.3 + 0.7j))[: len(matchings)]
    exact = np.vdot(coefficients, hamiltonian @ coefficients) / np.vdot(
        coefficients, overlap @ coefficients
    )
    normalized = proposal / proposal.sum()
    numerator = 0.0j
    denominator = 0.0j
    for left in range(len(matchings)):
        for right in range(len(matchings)):
            sampling_probability = normalized[left] * normalized[right]
            inverse_weight = 1.0 / (proposal[left] * proposal[right])
            coefficient = coefficients[left].conjugate() * coefficients[right]
            numerator += (
                sampling_probability
                * inverse_weight
                * coefficient
                * hamiltonian[left, right]
            )
            denominator += (
                sampling_probability
                * inverse_weight
                * coefficient
                * overlap[left, right]
            )
    np.testing.assert_allclose(numerator / denominator, exact, atol=3.0e-15)


def test_sequential_sampler_returns_valid_matchings_and_logged_weights() -> None:
    bonds = square_j1j2_bonds(4, 2, j1=1.0, j2=0.0, periodic=True)
    sampled = sample_sequential_matchings(8, bonds, count=64, seed=811)
    assert sampled.partners.shape == (64, 8)
    assert sampled.proposal_attempts >= 64
    assert sampled.rejected_dead_ends == sampled.proposal_attempts - 64
    for partner, log_probability in zip(
        sampled.partners,
        sampled.log_unnormalized_probabilities,
        strict=True,
    ):
        assert len(partner_array_matching(partner)) == 4
        np.testing.assert_allclose(
            log_probability,
            sequential_matching_log_weight(partner, bonds),
            atol=1.0e-15,
        )


def test_two_dimer_moves_are_graph_supported_and_exactly_reversible() -> None:
    bonds = square_j1j2_bonds(4, 2, j1=1.0, j2=0.0, periodic=True)
    flips = enumerate_two_dimer_flips(8, bonds)
    assert flips
    sampled = sample_sequential_matchings(8, bonds, count=16, seed=912)
    checked = 0
    for partner in sampled.partners:
        for index in flippable_two_dimer_move_indices(partner, flips):
            moved = apply_two_dimer_flip(partner, flips[index])
            assert index in flippable_two_dimer_move_indices(moved, flips)
            restored = apply_two_dimer_flip(moved, flips[index])
            np.testing.assert_array_equal(restored, partner)
            checked += 1
    assert checked > 0
