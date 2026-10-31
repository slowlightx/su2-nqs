import numpy as np

from su2_nqs.benchmarks.su2_vit_architecture_oracle import (
    spin_half_su2_rotation,
    su2_vit_architecture_oracle,
    two_spin_basis_coefficients,
    two_spin_cg_singlet_tensor,
    two_spin_invariant_scalar,
    two_spin_wavefunction_rotation_error,
)


def test_spin_half_rotation_and_cg_tensor_use_repository_convention() -> None:
    tensor = two_spin_cg_singlet_tensor()
    root_two = np.sqrt(2.0)
    np.testing.assert_allclose(
        tensor,
        np.asarray(((0.0, -1.0 / root_two), (1.0 / root_two, 0.0))),
        atol=2.0e-15,
        rtol=0.0,
    )
    rotation = spin_half_su2_rotation((0.2, -0.7, 1.1), 0.83)
    np.testing.assert_allclose(
        rotation.conj().T @ rotation,
        np.eye(2),
        atol=2.0e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.linalg.det(rotation), 1.0, atol=2.0e-15, rtol=0.0
    )


def test_cg_scalar_and_its_square_are_both_classically_invariant() -> None:
    rng = np.random.default_rng(713)
    for _ in range(12):
        rotation = spin_half_su2_rotation(rng.normal(size=3), rng.normal())
        first = rng.normal(size=2) + 1.0j * rng.normal(size=2)
        second = rng.normal(size=2) + 1.0j * rng.normal(size=2)
        for power in (1, 2):
            expected = two_spin_invariant_scalar(first, second, power=power)
            actual = two_spin_invariant_scalar(
                rotation @ first,
                rotation @ second,
                power=power,
            )
            np.testing.assert_allclose(actual, expected, atol=8.0e-14, rtol=8.0e-14)


def test_invariant_scalar_nonlinearity_does_not_imply_singlet_coefficients() -> None:
    result = su2_vit_architecture_oracle()
    root_two = np.sqrt(2.0)
    np.testing.assert_allclose(
        result.singlet_coefficients,
        (0.0, -1.0 / root_two, 1.0 / root_two, 0.0),
        atol=2.0e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.squared_coefficients,
        (0.0, 0.5, 0.5, 0.0),
        atol=2.0e-15,
        rtol=0.0,
    )
    assert result.singlet_rotation_invariance_error < 2.0e-14
    assert result.squared_rotation_invariance_error < 5.0e-14
    assert result.singlet_wavefunction_rotation_error < 2.0e-14
    assert result.squared_wavefunction_rotation_error > 0.1
    assert result.singlet_spin_squared_residual_ratio < 2.0e-14
    assert result.singlet_weight > 1.0 - 2.0e-14
    np.testing.assert_allclose(
        result.squared_spin_squared_residual_ratio,
        2.0,
        atol=2.0e-14,
        rtol=0.0,
    )
    assert result.squared_singlet_weight < 2.0e-14


def test_multi_affinity_is_the_missing_architecture_condition() -> None:
    result = su2_vit_architecture_oracle()
    assert result.singlet_multi_affinity_defect < 2.0e-14
    assert result.squared_multi_affinity_defect > 0.1

    # The direct basis evaluator remains deterministic and read-only.
    first = two_spin_basis_coefficients(power=2)
    second = two_spin_basis_coefficients(power=2)
    np.testing.assert_array_equal(first, second)
    assert not first.flags.writeable


def test_coefficient_tensor_rotation_is_distinct_from_scalar_invariance() -> None:
    singlet = two_spin_basis_coefficients(power=1)
    nonlinear = two_spin_basis_coefficients(power=2)
    assert two_spin_wavefunction_rotation_error(singlet) < 2.0e-14
    assert two_spin_wavefunction_rotation_error(nonlinear) > 0.1
