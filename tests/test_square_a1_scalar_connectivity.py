from __future__ import annotations

import numpy as np

from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.operators.spin_config import heisenberg_spin_connectivity
from su2_nqs.operators.square_a1_scalar import (
    SQUARE_A1_SCALAR_LABELS,
    square_a1_scalar_connectivity,
    square_a1_scalar_token_connectivity,
)


def _as_map(column):
    states, values = column
    return {state: float(value) for state, value in zip(states, values, strict=True)}


def _site(x: int, y: int, size: int = 4) -> int:
    return (y % size) * size + (x % size)


def _local_column(spins, sites, matrix):
    old_index = sum(spins[site] << (len(sites) - 1 - k) for k, site in enumerate(sites))
    result = {}
    for new_index, value in enumerate(matrix[:, old_index]):
        if abs(value) < 1.0e-14:
            continue
        state = list(spins)
        for k, site in enumerate(sites):
            state[site] = (new_index >> (len(sites) - 1 - k)) & 1
        key = tuple(state)
        result[key] = result.get(key, 0.0) + float(value)
    return result


def _permutation_matrix(length: int, mapping: tuple[int, ...]) -> np.ndarray:
    result = np.zeros((2**length, 2**length))
    for old in range(2**length):
        bits = tuple((old >> (length - 1 - k)) & 1 for k in range(length))
        new_bits = [0] * length
        for source, target in enumerate(mapping):
            new_bits[target] = bits[source]
        new = sum(bit << (length - 1 - k) for k, bit in enumerate(new_bits))
        result[new, old] = 1.0
    return result


def _singlet_projector(first: int, second: int) -> np.ndarray:
    result = np.zeros((16, 16))
    for old in range(16):
        bits = [(old >> (3 - k)) & 1 for k in range(4)]
        result[old, old] += 0.5 if bits[first] != bits[second] else 0.0
        if bits[first] != bits[second]:
            swapped = list(bits)
            swapped[first], swapped[second] = swapped[second], swapped[first]
            new = sum(bit << (3 - k) for k, bit in enumerate(swapped))
            result[new, old] -= 0.5
    return result


def _add(target, source):
    for state, value in source.items():
        target[state] = target.get(state, 0.0) + value


def test_square_four_by_four_columns_match_independent_local_matrices() -> None:
    rng = np.random.default_rng(20260929)
    spins = np.asarray([1] * 8 + [0] * 8, dtype=np.int8)
    rng.shuffle(spins)
    spins = tuple(int(value) for value in spins)
    actual = square_a1_scalar_token_connectivity(spins, 4)

    for shell, label in ((1, "J1_shell"), (2, "J2_shell")):
        bonds = tuple(
            (i, j)
            for i, j, value in square_j1j2_bonds(
                4,
                4,
                j1=1.0 if shell == 1 else 0.0,
                j2=1.0 if shell == 2 else 0.0,
                periodic=True,
            )
            if value
        )
        assert _as_map(actual[label]) == _as_map(
            heisenberg_spin_connectivity(spins, bonds)
        )

    cycle4 = _permutation_matrix(4, (1, 2, 3, 0))
    reverse4 = cycle4.T
    p01 = _singlet_projector(0, 1)
    p23 = _singlet_projector(2, 3)
    p12 = _singlet_projector(1, 2)
    p30 = _singlet_projector(3, 0)
    ring_expected = {}
    dimer_expected = {}
    rectangle_expected = {}
    cycle6 = _permutation_matrix(6, (1, 2, 3, 4, 5, 0))
    for y in range(4):
        for x in range(4):
            plaquette = (
                _site(x, y),
                _site(x + 1, y),
                _site(x + 1, y + 1),
                _site(x, y + 1),
            )
            _add(ring_expected, _local_column(spins, plaquette, cycle4 + reverse4))
            _add(
                dimer_expected,
                _local_column(spins, plaquette, p23 @ p01 + p30 @ p12),
            )
            for rectangle in (
                (
                    _site(x, y), _site(x + 1, y), _site(x + 2, y),
                    _site(x + 2, y + 1), _site(x + 1, y + 1), _site(x, y + 1),
                ),
                (
                    _site(x, y), _site(x, y + 1), _site(x, y + 2),
                    _site(x + 1, y + 2), _site(x + 1, y + 1), _site(x + 1, y),
                ),
            ):
                _add(
                    rectangle_expected,
                    _local_column(spins, rectangle, cycle6 + cycle6.T),
                )
    for expected, label in (
        (ring_expected, "plaquette_ring_exchange"),
        (dimer_expected, "plaquette_dimer_resonance"),
        (rectangle_expected, "six_site_rectangle_rearrangement"),
    ):
        expected = {key: value for key, value in expected.items() if abs(value) > 1e-14}
        assert set(expected) == set(_as_map(actual[label]))
        assert max(
            abs(expected[key] - _as_map(actual[label])[key]) for key in expected
        ) < 1.0e-13


def test_square_combined_token_column_is_exact_linear_combination() -> None:
    spins = tuple((3 * site + 1) % 2 for site in range(16))
    weights = np.asarray((0.2, -0.3, 0.4, -0.5, 0.6))
    tokens = square_a1_scalar_token_connectivity(spins, 4)
    expected = {}
    for weight, label in zip(weights, SQUARE_A1_SCALAR_LABELS, strict=True):
        for state, value in _as_map(tokens[label]).items():
            expected[state] = expected.get(state, 0.0) + weight * value
    expected = {key: value for key, value in expected.items() if abs(value) > 1e-14}
    actual = _as_map(square_a1_scalar_connectivity(spins, 4, weights))
    assert set(actual) == set(expected)
    assert max(abs(actual[key] - expected[key]) for key in expected) < 1.0e-13
    assert all(sum(state) == sum(spins) for state in actual)
