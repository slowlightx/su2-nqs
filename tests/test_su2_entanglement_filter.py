from __future__ import annotations

import numpy as np
import pytest

from su2_nqs.benchmarks.su2_entanglement_filter import (
    apply_subsystem_filter,
    exponential_filter,
    scramble_generators,
    spin_pair_operator,
    spin_permutation_operator,
    subsystem_spin_squared,
)
from su2_nqs.su2.sz_basis import enumerate_sz_configs


def test_pair_operator_has_singlet_triplet_spectrum_and_commutes_with_s2() -> None:
    pair = spin_pair_operator(2, 0, 1)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(pair), (-0.75, 0.25, 0.25, 0.25), atol=2.0e-15
    )
    spin_squared = subsystem_spin_squared(2)
    np.testing.assert_allclose(pair @ spin_squared, spin_squared @ pair, atol=2.0e-15)


def test_common_site_scramble_preserves_generator_spectra_and_su2() -> None:
    generators = (
        spin_pair_operator(4, 0, 1) + spin_pair_operator(4, 2, 3),
        spin_pair_operator(4, 0, 2),
        subsystem_spin_squared(4),
    )
    scrambled = scramble_generators(generators, (3, 1, 0, 2))
    for physical, control in zip(generators, scrambled, strict=True):
        np.testing.assert_allclose(
            np.linalg.eigvalsh(physical), np.linalg.eigvalsh(control), atol=3.0e-15
        )
        np.testing.assert_allclose(
            control @ generators[-1], generators[-1] @ control, atol=3.0e-15
        )
    permutation = spin_permutation_operator((3, 1, 0, 2))
    np.testing.assert_allclose(permutation.T @ permutation, np.eye(16), atol=0.0)


def test_zero_filter_is_identity_on_fixed_m_state() -> None:
    basis = enumerate_sz_configs(4, 2)
    rng = np.random.default_rng(17)
    state = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))
    state /= np.linalg.norm(state)
    generators = (
        spin_pair_operator(2, 0, 1),
        subsystem_spin_squared(2),
    )
    identity = exponential_filter((0.0, 0.0), generators)
    actual = apply_subsystem_filter(state, basis, (0, 1), identity)
    np.testing.assert_allclose(actual, state, atol=3.0e-15)


def test_filter_input_validation() -> None:
    with pytest.raises(ValueError, match="distinct"):
        spin_pair_operator(4, 1, 1)
    with pytest.raises(ValueError, match="one finite"):
        exponential_filter((0.0,), (np.eye(2), np.eye(2)))
    with pytest.raises(ValueError, match="permutation"):
        spin_permutation_operator((0, 0))
