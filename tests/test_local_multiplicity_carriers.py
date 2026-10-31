from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.local_multiplicity_carriers import (
    SQUARE_C4,
    SQUARE_MX,
    TRIANGLE_C3,
    TRIANGLE_MIRROR,
    apply_site_permutation,
    heisenberg_bond_in_carrier,
    heisenberg_bond_matrix,
    local_multiplicity_carrier_audit,
    site_permutation_in_carrier,
    square_plaquette_singlet_carrier,
    square_plaquette_spin_one_carrier,
    triangular_triangle_spin_half_carrier,
)
from su2_nqs.benchmarks.triangular_locality import physical_dimer_product
from su2_nqs.lattices.physics_guided_tree import square_d4_permutations
from su2_nqs.lattices.spatial_symmetry import permutation_group_closure


def test_three_four_spin_valence_bond_pairings_have_exact_rank_two() -> None:
    horizontal = physical_dimer_product(4, ((0, 1), (2, 3)))
    vertical = physical_dimer_product(4, ((0, 2), (1, 3)))
    diagonal = physical_dimer_product(4, ((0, 3), (1, 2)))
    states = np.column_stack((horizontal, vertical, diagonal))
    expected_gram = np.asarray(
        ((1.0, 0.5, -0.5), (0.5, 1.0, 0.5), (-0.5, 0.5, 1.0))
    )
    np.testing.assert_allclose(
        states.conj().T @ states, expected_gram, atol=2.0e-15, rtol=2.0e-15
    )
    assert np.linalg.matrix_rank(states, tol=1.0e-13) == 2
    np.testing.assert_allclose(horizontal - vertical + diagonal, 0.0, atol=0.0)

    carrier = square_plaquette_singlet_carrier()
    coordinates = carrier.vectors.conj().T @ states
    np.testing.assert_allclose(
        coordinates,
        np.asarray(
            ((np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 2.0, 0.0),
             (0.5, -0.5, -1.0))
        ),
        atol=3.0e-15,
        rtol=3.0e-15,
    )


def test_square_q0_carrier_is_orthonormal_A1_plus_B1_under_exact_d4() -> None:
    carrier = square_plaquette_singlet_carrier()
    assert carrier.target_q == 0
    assert carrier.target_m2 == 0
    assert carrier.multiplicity_rank == 2
    assert carrier.channel_labels == ("A1", "B1")
    assert carrier.irrep_decomposition == "A1_plus_B1"
    np.testing.assert_allclose(
        carrier.vectors.conj().T @ carrier.vectors, np.eye(2), atol=3.0e-15
    )
    np.testing.assert_allclose(
        carrier.generator_matrix("c4"), np.diag((1.0, -1.0)), atol=3.0e-15
    )
    np.testing.assert_allclose(
        carrier.generator_matrix("mx"), np.eye(2), atol=3.0e-15
    )
    for image in square_d4_permutations(2):
        matrix, residual = site_permutation_in_carrier(carrier, image)
        assert residual < 3.0e-15
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(2), atol=4.0e-15)
    assert not carrier.vectors.flags.writeable
    assert not carrier.generator_matrix("c4").flags.writeable
    with pytest.raises(KeyError):
        carrier.generator_matrix("not_a_generator")


def test_square_q0_shell_hamiltonian_resolves_the_two_physical_channels() -> None:
    carrier = square_plaquette_singlet_carrier()
    j1 = sum(
        heisenberg_bond_in_carrier(carrier, first, second)
        for first, second in ((0, 1), (2, 3), (0, 2), (1, 3))
    )
    j2 = sum(
        heisenberg_bond_in_carrier(carrier, first, second)
        for first, second in ((0, 3), (1, 2))
    )
    np.testing.assert_allclose(j1, np.diag((-2.0, 0.0)), atol=4.0e-15)
    np.testing.assert_allclose(j2, np.diag((0.5, -1.5)), atol=4.0e-15)


def test_square_q2_block_has_rank_three_and_d4_content_b2_plus_e() -> None:
    carrier = square_plaquette_spin_one_carrier()
    assert carrier.target_q == 2
    assert carrier.target_m2 == 2
    assert carrier.multiplicity_rank == 3
    assert carrier.irrep_decomposition == "B2_plus_E"
    c4 = carrier.generator_matrix("c4")
    mirror = carrier.generator_matrix("mx")
    assert np.trace(c4).real == pytest.approx(-1.0, abs=3.0e-15)
    assert np.trace(mirror).real == pytest.approx(-1.0, abs=3.0e-15)
    np.testing.assert_allclose(np.linalg.matrix_power(c4, 4), np.eye(3), atol=4.0e-15)
    np.testing.assert_allclose(mirror @ mirror, np.eye(3), atol=4.0e-15)
    np.testing.assert_allclose(mirror @ c4 @ mirror, c4.conj().T, atol=4.0e-15)


def test_triangle_q1_carrier_is_irreducible_d3_e_doublet() -> None:
    carrier = triangular_triangle_spin_half_carrier()
    assert carrier.target_q == 1
    assert carrier.target_m2 == 1
    assert carrier.multiplicity_rank == 2
    assert carrier.irrep_decomposition == "E"
    c3 = carrier.generator_matrix("c3")
    mirror = carrier.generator_matrix("mirror")
    np.testing.assert_allclose(
        c3,
        np.asarray(
            ((-0.5, -np.sqrt(3.0) / 2.0),
             (np.sqrt(3.0) / 2.0, -0.5))
        ),
        atol=3.0e-15,
    )
    np.testing.assert_allclose(
        mirror,
        np.asarray(
            ((0.5, np.sqrt(3.0) / 2.0),
             (np.sqrt(3.0) / 2.0, -0.5))
        ),
        atol=3.0e-15,
    )
    np.testing.assert_allclose(np.linalg.matrix_power(c3, 3), np.eye(2), atol=3.0e-15)
    np.testing.assert_allclose(mirror @ mirror, np.eye(2), atol=3.0e-15)
    np.testing.assert_allclose(mirror @ c3 @ mirror, c3.conj().T, atol=3.0e-15)
    assert np.trace(c3).real == pytest.approx(-1.0, abs=2.0e-15)
    assert np.trace(mirror).real == pytest.approx(0.0, abs=2.0e-15)

    group = permutation_group_closure((TRIANGLE_C3, TRIANGLE_MIRROR))
    assert len(group) == 6
    for image in group:
        matrix, residual = site_permutation_in_carrier(carrier, image)
        assert residual < 3.0e-15
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(2), atol=4.0e-15)


def test_triangle_bonds_are_three_pseudospin_axes_and_sum_to_scalar() -> None:
    carrier = triangular_triangle_spin_half_carrier()
    bonds = [
        heisenberg_bond_in_carrier(carrier, first, second)
        for first, second in ((0, 1), (1, 2), (0, 2))
    ]
    expected = (
        np.asarray(((-0.75, 0.0), (0.0, 0.25))),
        np.asarray(((0.0, np.sqrt(3.0) / 4.0),
                    (np.sqrt(3.0) / 4.0, -0.5))),
        np.asarray(((0.0, -np.sqrt(3.0) / 4.0),
                    (-np.sqrt(3.0) / 4.0, -0.5))),
    )
    for actual, wanted in zip(bonds, expected, strict=True):
        np.testing.assert_allclose(actual, wanted, atol=3.0e-15)
    np.testing.assert_allclose(sum(bonds), -0.75 * np.eye(2), atol=3.0e-15)
    # After removing the common trace, the three two-component Pauli vectors
    # have equal norm and mutual angle 120 degrees.
    traceless = [matrix - np.trace(matrix) * np.eye(2) / 2.0 for matrix in bonds]
    norms = [np.trace(matrix.conj().T @ matrix).real for matrix in traceless]
    assert max(norms) - min(norms) < 2.0e-15
    for first in range(3):
        for second in range(first + 1, 3):
            cosine = (
                np.trace(traceless[first].conj().T @ traceless[second]).real
                / np.sqrt(norms[first] * norms[second])
            )
            assert cosine == pytest.approx(-0.5, abs=3.0e-15)


def test_json_audit_records_shell_mismatch_rank_and_no_training_decision() -> None:
    audit = local_multiplicity_carrier_audit()
    valence = audit["literal_three_valence_bond_pairings"]
    assert valence["rank"] == 2
    assert valence["third_pairing_new_q0_rank"] == 0
    assert valence["null_relation_residual_norm"] == 0.0
    coverage = valence["shell_coverage"]
    assert coverage["square"] == {
        "first_two_J1_covered_over_total": [4, 4],
        "first_two_J2_covered_over_total": [0, 2],
        "all_three_J1_covered_over_total": [4, 4],
        "all_three_J2_covered_over_total": [2, 2],
    }
    assert coverage["triangular"] == {
        "first_two_J1_covered_over_total": [4, 5],
        "third_J1_covered_over_total": [1, 5],
        "third_non_J1_J2_pairs": 1,
        "all_three_J1_covered_over_total": [5, 5],
    }
    assert audit["square_plaquette"]["q0_multiplicity_rank"] == 2
    assert audit["square_plaquette"]["q2_multiplicity_rank"] == 3
    assert audit["triangular_triangle"]["q1_multiplicity_rank"] == 2
    assert audit["square_plaquette"]["d4_generator_relation_error"] < 3.0e-14
    assert audit["triangular_triangle"]["d3_generator_relation_error"] < 3.0e-14
    assert audit["decision"] == {
        "literal_third_four_site_pairing": "reject_as_redundant_q0_mechanism",
        "square_candidate": "carry_orthonormal_A1_B1_multiplicity_labels",
        "triangular_candidate": "carry_triangle_q1_E_doublet",
        "production_catalog_or_training_started": False,
    }


def test_carrier_builders_are_deterministic_and_validate_bond_inputs() -> None:
    for builder in (
        square_plaquette_singlet_carrier,
        square_plaquette_spin_one_carrier,
        triangular_triangle_spin_half_carrier,
    ):
        first = builder()
        second = builder()
        np.testing.assert_array_equal(first.vectors, second.vectors)
        for left, right in zip(
            first.generator_matrices, second.generator_matrices, strict=True
        ):
            np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError, match="distinct and in range"):
        heisenberg_bond_matrix(3, 1, 1)
    with pytest.raises(TypeError, match="integers"):
        heisenberg_bond_matrix(3, True, 1)
    with pytest.raises(ValueError, match="power of two"):
        apply_site_permutation(np.ones(6), (0, 1, 2))

