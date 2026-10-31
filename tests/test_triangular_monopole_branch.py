from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from su2_nqs.benchmarks.triangular_monopole_branch import (
    bond_center_orientation_operators,
    little_group_character_projector,
    maximum_low_subspace_source_fraction,
    projector_basis,
    source_spectrum,
)
from su2_nqs.lattices.triangular_hnf import (
    J1_POSITIVE,
    hnf_point_group_permutations,
)
from su2_nqs.operators.permutation import site_permutation_fusion_matrix


def test_many_body_gamma_a1_projector_has_exact_character_action() -> None:
    supercell = ((2, 0), (0, 2))
    characters = {
        name: 1 for name, _, _ in hnf_point_group_permutations(supercell)
    }
    projector = little_group_character_projector(
        supercell,
        (Fraction(0), Fraction(0)),
        characters,
        target_q=4,
    )
    basis = projector_basis(projector)
    assert basis.shape == (1, 1)
    np.testing.assert_allclose(projector, projector.conj().T, atol=2.0e-14)
    np.testing.assert_allclose(projector @ projector, projector, atol=2.0e-14)
    for _, _, image in hnf_point_group_permutations(supercell):
        action = site_permutation_fusion_matrix(image, target_q=4).toarray()
        np.testing.assert_allclose(action @ basis, basis, atol=2.0e-14)


def test_bond_center_operator_adjoint_is_opposite_momentum() -> None:
    supercell = ((6, 4), (0, 2))
    momentum = (Fraction(1, 6), Fraction(1, 6))
    # Keep the literal negative reciprocal coordinate.  Reducing it modulo a
    # reciprocal vector can add an orientation-dependent bond-center gauge.
    opposite = (-momentum[0], -momentum[1])
    forward = bond_center_orientation_operators(
        supercell, momentum, J1_POSITIVE, target_q=0
    )
    backward = bond_center_orientation_operators(
        supercell, opposite, J1_POSITIVE, target_q=0
    )
    assert min(np.linalg.norm(operator) for operator in forward) > 10.0
    for left, right in zip(forward, backward, strict=True):
        np.testing.assert_allclose(left.conj().T, right, atol=2.0e-14)


def test_low_subspace_source_optimization_and_moment_closure() -> None:
    source_columns = np.asarray(
        [[1.0, 0.0], [0.0, 0.2], [0.0, 1.0]], dtype=np.complex128
    )
    low = np.asarray([[1.0], [0.0], [0.0]], dtype=np.complex128)
    fraction, coefficients = maximum_low_subspace_source_fraction(
        source_columns, low
    )
    assert fraction == pytest.approx(1.0, abs=2.0e-14)
    assert abs(coefficients[0]) == pytest.approx(1.0, abs=2.0e-14)

    hamiltonian = np.diag((0.0, 1.0, 3.0))
    source = np.asarray([0.0, 2.0, 1.0], dtype=np.complex128)
    result = source_spectrum(
        hamiltonian,
        0.0,
        source,
        (0.0, 1.0, 3.0),
        np.eye(3, dtype=np.complex128),
    )
    assert result.m0 == pytest.approx(5.0)
    assert result.m1 == pytest.approx(7.0)
    assert result.mean_excitation == pytest.approx(1.4)
    assert result.lowest_pole_fraction == pytest.approx(0.0)
    assert result.m0_closure_error < 2.0e-14
    assert result.m1_closure_error < 2.0e-14


def test_character_projector_rejects_incomplete_little_group() -> None:
    with pytest.raises(ValueError, match="cover the exact little group"):
        little_group_character_projector(
            ((2, 0), (0, 2)),
            (Fraction(0), Fraction(0)),
            {"C6^0": 1},
            target_q=0,
        )
