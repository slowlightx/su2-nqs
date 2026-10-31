from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from su2_nqs.lattices.cut_chirality import (
    cut_chirality_coefficients,
    cut_chirality_gram,
    fusion_tree_cut_supports,
    hierarchy_chirality_signature,
)
from su2_nqs.su2.fusion_tree import FusionTree


_IDENTITY = np.eye(2, dtype=np.complex128)
_SPIN = (
    np.asarray(((0.0, 0.5), (0.5, 0.0)), dtype=np.complex128),
    np.asarray(((0.0, -0.5j), (0.5j, 0.0)), dtype=np.complex128),
    np.asarray(((0.5, 0.0), (0.0, -0.5)), dtype=np.complex128),
)


def _one_site(operator: np.ndarray, site: int, n_sites: int) -> np.ndarray:
    result = np.ones((1, 1), dtype=np.complex128)
    for index in range(n_sites):
        result = np.kron(result, operator if index == site else _IDENTITY)
    return result


def _bond(first: int, second: int, n_sites: int) -> np.ndarray:
    return sum(
        _one_site(operator, first, n_sites)
        @ _one_site(operator, second, n_sites)
        for operator in _SPIN
    )


def _chirality(first: int, second: int, third: int, n_sites: int) -> np.ndarray:
    parity = {
        (0, 1, 2): 1,
        (1, 2, 0): 1,
        (2, 0, 1): 1,
        (0, 2, 1): -1,
        (2, 1, 0): -1,
        (1, 0, 2): -1,
    }
    return sum(
        sign
        * _one_site(_SPIN[x], first, n_sites)
        @ _one_site(_SPIN[y], second, n_sites)
        @ _one_site(_SPIN[z], third, n_sites)
        for (x, y, z), sign in parity.items()
    )


def _dense_currents(
    n_sites: int,
    support: tuple[int, ...],
    edges: tuple[tuple[int, int, float], ...],
) -> tuple[np.ndarray, np.ndarray]:
    h_left, h_right, boundary = _dense_partition(n_sites, support, edges)
    return (
        h_left @ boundary - boundary @ h_left,
        h_right @ boundary - boundary @ h_right,
    )


def _dense_partition(
    n_sites: int,
    support: tuple[int, ...],
    edges: tuple[tuple[int, int, float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (2**n_sites, 2**n_sites)
    h_left = np.zeros(shape, dtype=np.complex128)
    h_right = np.zeros(shape, dtype=np.complex128)
    boundary = np.zeros(shape, dtype=np.complex128)
    inside = set(support)
    for first, second, coupling in edges:
        term = coupling * _bond(first, second, n_sites)
        if first in inside and second in inside:
            h_left += term
        elif first not in inside and second not in inside:
            h_right += term
        else:
            boundary += term
    return h_left, h_right, boundary


def _coefficient_operator(
    row: tuple[tuple[tuple[int, int, int], float], ...],
    *,
    n_sites: int,
) -> np.ndarray:
    result = np.zeros((2**n_sites, 2**n_sites), dtype=np.complex128)
    for (first, second, third), coefficient in row:
        result += 1j * coefficient * _chirality(
            first, second, third, n_sites
        )
    return result


def _gram_loads(
    left: np.ndarray, right: np.ndarray
) -> tuple[float, float, float]:
    count = left.shape[0]
    incoherent = float((np.trace(left) + np.trace(right)) / count)
    coherent = float((np.sum(left) + np.sum(right)) / count**2)
    ratio = coherent / incoherent if incoherent else 0.0
    return incoherent, coherent, ratio


def _dense_gram(
    currents: tuple[np.ndarray, ...], *, n_sites: int, j_scale: float
) -> np.ndarray:
    normalization = (32.0 / 3.0) / (2**n_sites * j_scale**4)
    return np.asarray(
        [
            [
                normalization
                * np.trace(left.conj().T @ right).real
                for right in currents
            ]
            for left in currents
        ]
    )


def test_three_spin_exchange_commutator_and_chirality_norm() -> None:
    h01 = _bond(0, 1, 3)
    h12 = _bond(1, 2, 3)
    h02 = _bond(0, 2, 3)
    chirality = _chirality(0, 1, 2, 3)

    assert np.linalg.norm(h01 @ h12 - h12 @ h01 + 1j * chirality) < 1.0e-14
    assert np.linalg.norm(h01 @ h02 - h02 @ h01 - 1j * chirality) < 1.0e-14
    observed_norm = np.trace(chirality.conj().T @ chirality).real / 8.0
    assert observed_norm == pytest.approx(3.0 / 32.0, abs=1.0e-15)


def test_scalene_triangle_formula_and_symmetric_cancellation() -> None:
    coupling01, coupling02, coupling12 = 0.7, -0.2, 1.3
    edges = (
        (0, 1, coupling01),
        (0, 2, coupling02),
        (1, 2, coupling12),
    )
    coefficients = cut_chirality_coefficients(3, (0, 1), edges)
    expected = coupling01 * (coupling02 - coupling12)
    assert coefficients.left == (((0, 1, 2), expected),)
    assert coefficients.right == ()

    current, _ = _dense_currents(3, (0, 1), edges)
    observed = np.trace(current.conj().T @ current).real / 8.0
    assert observed == pytest.approx((3.0 / 32.0) * expected**2, abs=2.0e-16)

    symmetric = ((0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0))
    cancelled = cut_chirality_coefficients(3, (0, 1), symmetric)
    assert cancelled.left == ()
    assert cancelled.right == ()


def test_arbitrary_graph_reconstructs_signed_currents_and_continuity() -> None:
    n_sites = 5
    support = (0, 2, 3)
    edges = (
        (0, 1, 0.7),
        (1, 0, -0.1),
        (0, 2, -0.3),
        (0, 3, 0.45),
        (0, 4, -0.55),
        (1, 2, 1.1),
        (1, 3, 0.4),
        (1, 4, -0.8),
        (2, 3, 0.9),
        (2, 4, 0.2),
        (3, 4, 0.65),
    )
    coefficients = cut_chirality_coefficients(n_sites, support, edges)
    dense_left, dense_right = _dense_currents(n_sites, support, edges)
    expanded_left = _coefficient_operator(
        coefficients.left, n_sites=n_sites
    )
    expanded_right = _coefficient_operator(
        coefficients.right, n_sites=n_sites
    )

    # This operator-level check fixes the common sign that a K^dagger K Gram
    # comparison alone cannot determine.
    assert np.linalg.norm(dense_left - expanded_left) < 2.0e-15
    assert np.linalg.norm(dense_right - expanded_right) < 2.0e-15

    h_left, h_right, boundary = _dense_partition(n_sites, support, edges)
    hamiltonian = h_left + h_right + boundary
    current_left = 1j * dense_left
    current_right = 1j * dense_right
    assert np.linalg.norm(current_left - current_left.conj().T) < 2.0e-15
    assert np.linalg.norm(current_right - current_right.conj().T) < 2.0e-15

    derivative_left = 1j * (
        hamiltonian @ h_left - h_left @ hamiltonian
    )
    derivative_right = 1j * (
        hamiltonian @ h_right - h_right @ hamiltonian
    )
    derivative_boundary = 1j * (
        hamiltonian @ boundary - boundary @ hamiltonian
    )
    assert np.linalg.norm(derivative_left + current_left) < 2.0e-15
    assert np.linalg.norm(derivative_right + current_right) < 2.0e-15
    assert (
        np.linalg.norm(derivative_boundary - current_left - current_right)
        < 2.0e-15
    )
    assert (
        np.linalg.norm(
            derivative_left + derivative_right + derivative_boundary
        )
        < 2.0e-15
    )


@pytest.mark.parametrize("n_sites", (3, 5, 8))
def test_graph_gram_matches_independent_dense_matrices_through_n8(
    n_sites: int,
) -> None:
    all_edges = tuple(
        (
            first,
            second,
            ((-1.0) ** (first + 2 * second))
            * (0.17 + 0.03 * first + 0.02 * second),
        )
        for first, second in combinations(range(n_sites), 2)
        if (3 * first + second) % 4 != 0
    )
    tree = FusionTree.balanced(n_sites)
    supports = fusion_tree_cut_supports(tree, tuple(range(n_sites)))
    # N=3 has one cut; larger systems exercise off-diagonal cut-cut coherence.
    gram = cut_chirality_gram(n_sites, supports, all_edges)
    dense = tuple(
        _dense_currents(n_sites, support, all_edges) for support in supports
    )
    j_scale = max(abs(coupling) for _, _, coupling in all_edges)
    expected_left = _dense_gram(
        tuple(row[0] for row in dense), n_sites=n_sites, j_scale=j_scale
    )
    expected_right = _dense_gram(
        tuple(row[1] for row in dense), n_sites=n_sites, j_scale=j_scale
    )

    assert np.max(np.abs(np.asarray(gram.left) - expected_left)) < 5.0e-12
    assert np.max(np.abs(np.asarray(gram.right) - expected_right)) < 5.0e-12


def test_signature_is_permutation_covariant_and_gram_is_psd() -> None:
    tree = FusionTree.balanced(8)
    order = (0, 3, 5, 1, 7, 2, 6, 4)
    edges = tuple(
        (first, second, 0.2 + 0.1 * ((first + second) % 5))
        for first, second in combinations(range(8), 2)
        if (first + 2 * second) % 3 == 0
    )
    permutation = (5, 2, 7, 0, 6, 3, 1, 4)
    transformed_edges = tuple(
        (permutation[first], permutation[second], coupling)
        for first, second, coupling in edges
    )
    transformed_order = tuple(permutation[site] for site in order)

    original = hierarchy_chirality_signature(tree, order, edges)
    transformed = hierarchy_chirality_signature(
        tree, transformed_order, transformed_edges
    )
    assert transformed.incoherent_load == pytest.approx(
        original.incoherent_load, abs=2.0e-12
    )
    assert transformed.coherent_load == pytest.approx(
        original.coherent_load, abs=2.0e-12
    )
    assert transformed.coherence_ratio == pytest.approx(
        original.coherence_ratio, abs=2.0e-12
    )
    assert np.asarray(transformed.left_gram) == pytest.approx(
        np.asarray(original.left_gram), abs=2.0e-12
    )
    assert np.asarray(transformed.right_gram) == pytest.approx(
        np.asarray(original.right_gram), abs=2.0e-12
    )

    for matrix in (original.left_gram, original.right_gram):
        assert np.min(np.linalg.eigvalsh(np.asarray(matrix))) >= -5.0e-12
    assert 0.0 <= original.coherence_ratio <= 1.0


def test_normalized_gram_is_invariant_to_nonzero_uniform_coupling_rescaling() -> None:
    n_sites = 6
    supports = fusion_tree_cut_supports(
        FusionTree.balanced(n_sites), tuple(range(n_sites))
    )
    edges = (
        (0, 1, 0.7),
        (0, 2, -0.3),
        (0, 5, 0.55),
        (1, 2, 1.1),
        (1, 3, 0.4),
        (2, 3, -0.8),
        (2, 4, 0.9),
        (3, 4, 0.2),
        (3, 5, -0.45),
        (4, 5, 0.65),
    )
    reference = cut_chirality_gram(n_sites, supports, edges)
    for scale in (2.0, -0.5):
        scaled = cut_chirality_gram(
            n_sites,
            supports,
            tuple(
                (first, second, scale * coupling)
                for first, second, coupling in edges
            ),
        )
        assert scaled.j_scale == pytest.approx(
            abs(scale) * reference.j_scale, abs=1.0e-15
        )
        assert np.asarray(scaled.left) == pytest.approx(
            np.asarray(reference.left), abs=2.0e-14
        )
        assert np.asarray(scaled.right) == pytest.approx(
            np.asarray(reference.right), abs=2.0e-14
        )


def test_cut_list_permutation_only_permutes_the_gram() -> None:
    n_sites = 6
    supports = fusion_tree_cut_supports(
        FusionTree.balanced(n_sites), tuple(range(n_sites))
    )
    edges = tuple(
        (
            first,
            second,
            (-1.0) ** (first + second) * (0.2 + 0.03 * first + 0.05 * second),
        )
        for first, second in combinations(range(n_sites), 2)
        if (first + 3 * second) % 4 != 0
    )
    permutation = (2, 0, 3, 1)
    reference = cut_chirality_gram(n_sites, supports, edges)
    permuted = cut_chirality_gram(
        n_sites, tuple(supports[index] for index in permutation), edges
    )
    reference_left = np.asarray(reference.left)
    reference_right = np.asarray(reference.right)
    expected_left = reference_left[np.ix_(permutation, permutation)]
    expected_right = reference_right[np.ix_(permutation, permutation)]
    assert np.asarray(permuted.left) == pytest.approx(
        expected_left, abs=1.0e-14
    )
    assert np.asarray(permuted.right) == pytest.approx(
        expected_right, abs=1.0e-14
    )
    assert _gram_loads(reference_left, reference_right) == pytest.approx(
        _gram_loads(np.asarray(permuted.left), np.asarray(permuted.right)),
        abs=1.0e-14,
    )


def test_global_cut_complement_is_invariant_but_selected_complement_is_not() -> None:
    n_sites = 5
    supports = ((0, 1), (0, 1, 2))
    edges = (
        (0, 1, 0.7),
        (0, 2, -0.3),
        (0, 4, -0.55),
        (1, 2, 1.1),
        (1, 3, 0.4),
        (1, 4, 0.65),
        (2, 3, -0.8),
        (2, 4, 0.9),
        (3, 4, 0.2),
    )

    def complement(support: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(site for site in range(n_sites) if site not in support)

    reference = cut_chirality_gram(n_sites, supports, edges)
    all_complemented = cut_chirality_gram(
        n_sites, tuple(complement(support) for support in supports), edges
    )
    reference_left = np.asarray(reference.left)
    reference_right = np.asarray(reference.right)
    complemented_left = np.asarray(all_complemented.left)
    complemented_right = np.asarray(all_complemented.right)
    assert complemented_left == pytest.approx(reference_right, abs=1.0e-14)
    assert complemented_right == pytest.approx(reference_left, abs=1.0e-14)
    assert _gram_loads(reference_left, reference_right) == pytest.approx(
        _gram_loads(complemented_left, complemented_right), abs=1.0e-14
    )

    # The hierarchy uses oriented/rooted subtree cuts.  Complementing only one
    # cut swaps its side labels without swapping those of the other cut, so
    # cross-cut coherent alignment is generally changed even though the mean
    # individual-cut norm remains invariant.
    selected_complement = cut_chirality_gram(
        n_sites, (complement(supports[0]), supports[1]), edges
    )
    selected_loads = _gram_loads(
        np.asarray(selected_complement.left),
        np.asarray(selected_complement.right),
    )
    reference_loads = _gram_loads(reference_left, reference_right)
    assert selected_loads[0] == pytest.approx(reference_loads[0], abs=1.0e-14)
    assert abs(selected_loads[1] - reference_loads[1]) > 0.1
    assert abs(selected_loads[2] - reference_loads[2]) > 0.1


def test_complement_swaps_left_and_right_currents() -> None:
    edges = ((0, 1, 0.7), (1, 2, -0.4), (2, 3, 1.1), (0, 3, 0.2))
    first = cut_chirality_gram(4, ((0, 1),), edges)
    complement = cut_chirality_gram(4, ((2, 3),), edges)
    assert np.asarray(complement.left) == pytest.approx(
        np.asarray(first.right), abs=1.0e-14
    )
    assert np.asarray(complement.right) == pytest.approx(
        np.asarray(first.left), abs=1.0e-14
    )


def test_disconnected_and_nonincident_boundary_limits_are_zero() -> None:
    disconnected = cut_chirality_gram(
        4, ((0, 1),), ((0, 1, 1.0), (2, 3, -0.8))
    )
    assert disconnected.left == ((0.0,),)
    assert disconnected.right == ((0.0,),)

    # A crossing exchange with no adjacent internal exchange cannot generate
    # a three-spin current on either side.
    isolated_boundary = cut_chirality_gram(4, ((0, 1),), ((0, 2, 1.0),))
    assert isolated_boundary.left == ((0.0,),)
    assert isolated_boundary.right == ((0.0,),)


def test_duplicate_edges_are_aggregated_algebraically() -> None:
    raw = (
        (0, 1, 0.8),
        (1, 0, 0.2),
        (0, 2, 0.5),
        (2, 0, -0.1),
        (1, 2, 0.7),
    )
    canonical = ((0, 1, 1.0), (0, 2, 0.4), (1, 2, 0.7))
    assert cut_chirality_gram(3, ((0, 1),), raw) == cut_chirality_gram(
        3, ((0, 1),), canonical
    )


def test_two_site_tree_and_zero_graph_have_zero_signature() -> None:
    tree = FusionTree.balanced(2)
    result = hierarchy_chirality_signature(tree, (0, 1), ())
    assert result.internal_cut_count == 0
    assert result.incoherent_load == 0.0
    assert result.coherent_load == 0.0
    assert result.coherence_ratio == 0.0


def test_invalid_labels_fail_closed() -> None:
    with pytest.raises(TypeError, match="n_sites"):
        cut_chirality_gram(4.0, ((0, 1),), ((0, 1, 1.0),))
    with pytest.raises(ValueError, match="nontrivial"):
        cut_chirality_gram(3, ((0, 1, 2),), ((0, 1, 1.0),))
    with pytest.raises(ValueError, match="duplicate"):
        cut_chirality_gram(3, ((0, 1), (1, 0)), ((0, 1, 1.0),))
    with pytest.raises(ValueError, match="self"):
        cut_chirality_gram(3, ((0, 1),), ((0, 0, 1.0),))
