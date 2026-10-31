"""Independent physical H-row and node tests for the fixed scalar source."""
import numpy as np
import pytest

from projects.clebsch_tree_nqs.explorations.ct_scalar_X_H_probe_20261030 import (
    full_H_connections, stream_full_H,
)
from su2_nqs.benchmarks.spin_operator_information import spin_dot_matrix


def test_full_H_rows_reproduce_dense_bonds_and_nodes_without_eigenstate_assumption():
    n = 6
    all_rows = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    powers = 1 << np.arange(n-1, -1, -1)
    bonds = ((0, 1, 1.), (1, 0, .125), (2, 5, -.17), (1, 4, .41), (3, 4, .91))
    h = sum(c*spin_dot_matrix(n, a, b) for a, b, c in bonds)
    rng = np.random.default_rng(303031)
    psi = rng.normal(size=(2**n, 2))+1j*rng.normal(size=(2**n, 2))
    psi[21, 1] = 0.
    actual = []
    for row in all_rows:
        connected, coefficient = full_H_connections(row, bonds)
        assert np.all(connected.sum(1) == row.sum())
        center, hp, _ = stream_full_H(lambda x: psi[x@powers], connected, coefficient, batch_size=4)
        np.testing.assert_array_equal(center, psi[row@powers])
        actual.append(hp)
    np.testing.assert_allclose(actual, h@psi, atol=2e-15, rtol=0)
    assert abs(actual[21][1]) > .01


def test_invalid_rows_and_nonfinite_values_are_not_repaired():
    for bits in (np.array([0., 1.]), np.array([0, 2]), np.array([[0, 1]])):
        with pytest.raises(ValueError):
            full_H_connections(bits, ((0, 1, 1.),))
    rows, c = full_H_connections(np.array([0, 1]), ((0, 1, 1.),))
    with pytest.raises(ArithmeticError):
        stream_full_H(lambda x: np.full((len(x), 1), np.nan), rows, c)
