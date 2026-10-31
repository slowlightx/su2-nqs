import numpy as np
import pytest

from su2_nqs.operators.permutation import (
    adjacent_swap_matrix,
    adjacent_swap_path_connectivity,
    apply_site_permutation_fusion,
    evaluate_site_permutation_fusion_amplitude,
    heisenberg_graph_fusion_matrix,
    heisenberg_graph_path_connectivity,
    site_permutation_fusion_row,
    weighted_site_permutation_fusion_row,
)
from su2_nqs.su2.fusion_path import enumerate_paths


@pytest.mark.parametrize("target_q", (0, 2, 4))
def test_local_adjacent_swap_matches_sparse_operator(target_q: int) -> None:
    n = 6
    basis = enumerate_paths(n, target_q)
    positions = {path: index for index, path in enumerate(basis)}
    for bond in range(n - 1):
        operator = adjacent_swap_matrix(n, target_q, bond)
        for column, path in enumerate(basis):
            connected, values = adjacent_swap_path_connectivity(
                path, bond, target_q=target_q
            )
            actual = np.zeros(len(basis))
            for candidate, value in zip(connected, values, strict=True):
                actual[positions[candidate]] += value
            np.testing.assert_allclose(
                actual, operator.getcol(column).toarray().ravel(), atol=3.0e-14
            )


@pytest.mark.parametrize("n,target_q", ((6, 0), (6, 2), (8, 4)))
def test_matrix_free_permutation_rows_reconstruct_full_action(
    n: int, target_q: int
) -> None:
    rng = np.random.default_rng(20260910 + n + target_q)
    image = tuple(int(value) for value in rng.permutation(n))
    basis = enumerate_paths(n, target_q)
    positions = {path: index for index, path in enumerate(basis)}
    state = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))
    expected = apply_site_permutation_fusion(state, image, target_q=target_q)

    actual = np.asarray(
        [
            evaluate_site_permutation_fusion_amplitude(
                path,
                image,
                lambda source: state[positions[source]],
                target_q=target_q,
            )
            for path in basis
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=2.0e-13)

    for row, path in enumerate(basis):
        sources, coefficients = site_permutation_fusion_row(
            path, image, target_q=target_q
        )
        assert len(sources) == len(coefficients)
        assert np.linalg.norm(coefficients) == pytest.approx(1.0, abs=2.0e-13)
        reconstructed = sum(
            value * state[positions[source]]
            for source, value in zip(sources, coefficients, strict=True)
        )
        assert reconstructed == pytest.approx(expected[row], abs=2.0e-13)


def test_weighted_local_rows_match_cyclic_character_projection() -> None:
    n, target_q, momentum = 8, 2, 3
    basis = enumerate_paths(n, target_q)
    positions = {path: index for index, path in enumerate(basis)}
    rng = np.random.default_rng(260910)
    state = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))
    terms = tuple(
        (
            np.exp(2.0j * np.pi * momentum * shift / n) / n,
            tuple((site + shift) % n for site in range(n)),
        )
        for shift in range(n)
    )
    expected = sum(
        weight * apply_site_permutation_fusion(
            state, image, target_q=target_q
        )
        for weight, image in terms
    )
    actual = []
    for path in basis:
        sources, coefficients = weighted_site_permutation_fusion_row(
            path, terms, target_q=target_q
        )
        actual.append(
            sum(
                value * state[positions[source]]
                for source, value in zip(sources, coefficients, strict=True)
            )
        )
    np.testing.assert_allclose(actual, expected, atol=3.0e-13)


def test_local_row_validates_sector_and_orbit_weights() -> None:
    with pytest.raises(ValueError, match="requested sector"):
        site_permutation_fusion_row((1, 0, 1, 0), (0, 1, 2, 3), target_q=2)
    with pytest.raises(ValueError, match="nonempty"):
        weighted_site_permutation_fusion_row(
            (1, 0, 1, 0), (), target_q=0
        )
    with pytest.raises(ValueError, match="finite"):
        weighted_site_permutation_fusion_row(
            (1, 0, 1, 0), ((complex(np.nan), (0, 1, 2, 3)),), target_q=0
        )


@pytest.mark.parametrize("n,target_q", ((4, 0), (6, 0), (6, 2), (8, 4)))
def test_matrix_free_graph_connectivity_matches_every_exact_column(
    n: int, target_q: int
) -> None:
    bonds = tuple(
        (site, (site + 1) % n, 1.0 + 0.1 * site) for site in range(n)
    ) + ((0, n // 2, -0.37),)
    basis = enumerate_paths(n, target_q)
    positions = {path: index for index, path in enumerate(basis)}
    expected = heisenberg_graph_fusion_matrix(
        n, bonds, target_q=target_q
    ).tocsc()
    for column, path in enumerate(basis):
        connected, values = heisenberg_graph_path_connectivity(
            path, bonds, target_q=target_q
        )
        actual = np.zeros(len(basis), dtype=np.float64)
        for candidate, value in zip(connected, values, strict=True):
            actual[positions[candidate]] = value
        np.testing.assert_allclose(
            actual, expected.getcol(column).toarray().ravel(), atol=5.0e-13
        )
