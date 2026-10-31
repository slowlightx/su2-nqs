from __future__ import annotations

from itertools import combinations

import numpy as np

from su2_nqs.benchmarks.local_carrier_product_subspace import (
    triangular_e_singlet_product_columns,
)
from su2_nqs.benchmarks.local_multiplicity_carriers import (
    square_plaquette_singlet_carrier,
)
from su2_nqs.benchmarks.local_multiplicity_observables import (
    local_multiplicity_density,
    local_multiplicity_density_json,
)


def _fixed_m_zero_basis(n_sites: int) -> np.ndarray:
    rows = []
    for occupied in combinations(range(n_sites), n_sites // 2):
        row = np.zeros(n_sites, dtype=np.int8)
        row[list(occupied)] = 1
        rows.append(row)
    return np.asarray(rows)


def _restrict_full_state(full: np.ndarray, basis: np.ndarray) -> np.ndarray:
    weights = 1 << np.arange(basis.shape[1] - 1, -1, -1, dtype=np.int64)
    return np.asarray(full)[basis @ weights]


def test_square_product_state_resolves_point_group_fixed_a1_b1_channels() -> None:
    carrier = square_plaquette_singlet_carrier()
    basis = _fixed_m_zero_basis(8)
    for channel in (0, 1):
        full = np.kron(carrier.vectors[:, channel], carrier.vectors[:, 0])
        amplitudes = _restrict_full_state(full, basis)
        density = local_multiplicity_density(
            amplitudes,
            basis,
            cell=(0, 1, 2, 3),
            kind="square_plaquette",
        )
        block = density.block(0)
        expected = np.zeros((2, 2))
        expected[channel, channel] = 1.0
        assert np.isclose(block.block_weight, 1.0, atol=4.0e-15)
        np.testing.assert_allclose(block.conditional_matrix, expected, atol=4.0e-15)
        np.testing.assert_allclose(block.eigenvalues, (1.0, 0.0), atol=4.0e-15)
        assert np.isclose(block.purity, 1.0, atol=4.0e-15)
        assert density.block(2).block_weight < 4.0e-15
        assert density.block(4).block_weight < 4.0e-15


def test_triangle_e_product_oracle_has_pure_local_multiplicity_label() -> None:
    basis = _fixed_m_zero_basis(12)
    cells = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
    columns, _, _ = triangular_e_singlet_product_columns(basis, cells)
    # Column ordering is E-label tuple first, then the two effective-spin
    # singlet fusion paths.  The first column has local E label zero in every
    # triangle.
    density = local_multiplicity_density(
        columns[:, 0], basis, cell=cells[0], kind="triangular_triangle"
    )
    block = density.block(1)
    assert np.isclose(block.block_weight, 1.0, atol=5.0e-15)
    np.testing.assert_allclose(
        block.conditional_matrix, np.diag((1.0, 0.0)), atol=5.0e-15
    )
    assert density.block(3).block_weight < 5.0e-15
    payload = local_multiplicity_density_json(density)
    assert payload["kind"] == "triangular_triangle"
    assert [row["q"] for row in payload["blocks"]] == [1, 3]


def test_local_density_is_invariant_to_state_scale_and_validates_cell() -> None:
    carrier = square_plaquette_singlet_carrier()
    basis = _fixed_m_zero_basis(8)
    full = np.kron(carrier.vectors[:, 0], carrier.vectors[:, 1])
    amplitudes = _restrict_full_state(full, basis)
    first = local_multiplicity_density(
        amplitudes, basis, cell=(0, 1, 2, 3), kind="square_plaquette"
    )
    second = local_multiplicity_density(
        -3.2j * amplitudes,
        basis,
        cell=(0, 1, 2, 3),
        kind="square_plaquette",
    )
    for left, right in zip(first.blocks, second.blocks, strict=True):
        np.testing.assert_allclose(left.matrix, right.matrix, atol=4.0e-15)
    with np.testing.assert_raises_regex(ValueError, "distinct"):
        local_multiplicity_density(
            amplitudes,
            basis,
            cell=(0, 1, 1, 3),
            kind="square_plaquette",
        )
