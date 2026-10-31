from collections import Counter

import numpy as np

from su2_nqs.benchmarks.common_basis_gamma_a1 import (
    capped_permutation_row_profile,
    exhaustive_projected_born_state,
    gamma_a1_n36_cost_audit,
    sample_exhaustive_projected_born,
    triangular_hnf_gamma_a1_circuit,
)
from su2_nqs.benchmarks.triangular_hnf_space_group import (
    exact_gamma_a1_spin_multiplicity_dimension,
)
from su2_nqs.lattices.triangular_hnf import (
    hnf_point_group_permutations,
    hnf_site_permutation,
    hnf_translation_permutations,
    triangular_hnf_j1j2_bonds,
)
from su2_nqs.operators.permutation import (
    heisenberg_graph_fusion_matrix,
    site_permutation_fusion_matrix,
)
from su2_nqs.su2.fusion_path import enumerate_paths


def _direct_group_images(supercell):
    return tuple(
        tuple(point[translation[index]] for index in range(len(translation)))
        for _, translation in hnf_translation_permutations(supercell)
        for _, _, point in hnf_point_group_permutations(supercell)
    )


def test_factorized_images_are_the_full_abstract_hnf_space_group():
    supercell = ((3, 0), (0, 3))
    circuit = triangular_hnf_gamma_a1_circuit(supercell, target_q=1)
    expanded = circuit.group_images()
    direct = _direct_group_images(supercell)
    assert circuit.abstract_group_order == 108
    assert Counter(expanded) == Counter(direct)


def test_common_basis_projector_is_exhaustively_exact_and_commutes_with_h():
    supercell, target_q = ((3, 0), (0, 3)), 3
    circuit = triangular_hnf_gamma_a1_circuit(supercell, target_q=target_q)
    projector = circuit.dense_matrix()
    direct = sum(
        site_permutation_fusion_matrix(image, target_q=target_q).toarray()
        for image in _direct_group_images(supercell)
    ) / circuit.abstract_group_order
    np.testing.assert_allclose(projector, direct, atol=2.5e-13)
    np.testing.assert_allclose(projector, projector.T, atol=2.5e-13)
    np.testing.assert_allclose(projector @ projector, projector, atol=4.0e-13)

    expected_dimension = exact_gamma_a1_spin_multiplicity_dimension(
        supercell, target_q=target_q
    )["dimension"]
    assert np.count_nonzero(np.linalg.eigvalsh(projector) > 0.5) == expected_dimension

    hamiltonian = heisenberg_graph_fusion_matrix(
        9,
        triangular_hnf_j1j2_bonds(supercell, j1=1.0, j2=0.31),
        target_q=target_q,
    ).toarray()
    denominator = max(np.linalg.norm(hamiltonian), 1.0e-30)
    assert np.linalg.norm(hamiltonian @ projector - projector @ hamiltonian) / denominator < 3.0e-13


def test_direct_projected_amplitudes_match_factorized_common_basis_vector():
    circuit = triangular_hnf_gamma_a1_circuit(((2, 0), (0, 2)), target_q=0)
    paths = enumerate_paths(4, 0)
    rng = np.random.default_rng(2026091301)
    base = rng.normal(size=len(paths)) + 1.0j * rng.normal(size=len(paths))
    expected = circuit.apply(base)
    positions = {path: index for index, path in enumerate(paths)}
    actual = np.asarray(
        [
            circuit.projected_amplitude(
                path, lambda source: base[positions[source]]
            )
            for path in paths
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=2.0e-13)


def test_exhaustive_projected_sampler_has_normalized_correct_frequencies():
    circuit = triangular_hnf_gamma_a1_circuit(((3, 0), (0, 3)), target_q=3)
    rng = np.random.default_rng(2026091302)
    base = rng.normal(size=circuit.dimension) + 1.0j * rng.normal(
        size=circuit.dimension
    )
    state = exhaustive_projected_born_state(circuit, base)
    np.testing.assert_allclose(np.vdot(state.amplitudes, state.amplitudes), 1.0, atol=3.0e-13)
    np.testing.assert_allclose(np.sum(state.probabilities), 1.0, atol=3.0e-13)
    samples = sample_exhaustive_projected_born(state, rng, 100_000)
    empirical = np.bincount(samples, minlength=circuit.dimension) / len(samples)
    assert np.max(np.abs(empirical - state.probabilities)) < 7.5e-3


def test_n16_randomized_projector_idempotence_in_fixed_basis():
    """N=16 is probed without constructing a 1430-square dense projector."""

    circuit = triangular_hnf_gamma_a1_circuit(((4, 0), (0, 4)), target_q=0)
    rng = np.random.default_rng(2026091303)
    probes = rng.normal(size=(circuit.dimension, 2))
    once = circuit.apply(probes)
    twice = circuit.apply(once)
    relative = np.linalg.norm(twice - once) / max(np.linalg.norm(once), 1.0e-30)
    assert relative < 8.0e-13


def test_n36_cost_audit_fails_closed_and_reports_exact_dimensions():
    audit = gamma_a1_n36_cost_audit()
    assert audit["n_sites"] == 36
    assert audit["full_q_sector_dimension"] == 477_638_700
    assert audit["exact_gamma_a1_dimension"] == 1_109_658
    assert audit["abstract_space_group_order"] == 432
    assert audit["factorized_adjacent_swap_stages_per_full_vector"] == 3485
    assert audit["unfactorized_adjacent_swap_stages_per_full_vector"] == 136_080
    assert audit["factorization_stage_reduction"] > 39.0
    assert not audit["production_sampler_available"]
    assert not audit["formal_d1_unlock"]


def test_n36_capped_row_profile_is_exact_or_explicitly_aborted():
    path = tuple(1 if site % 2 == 0 else 0 for site in range(36))
    tx = hnf_site_permutation(((6, 0), (0, 6)), translation=(1, 0))
    result = capped_permutation_row_profile(
        path, tx, target_q=0, maximum_support=5000
    )
    assert result["completed"]
    assert result["word_length"] == 30
    assert result["terminal_support"] == 4096
    assert result["maximum_exact_support_seen"] == 4096
    assert result["row_norm_error"] < 2.0e-12
