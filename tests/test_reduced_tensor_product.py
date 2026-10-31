import numpy as np
import pytest

from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.reduced_tensor_product import (
    VECTOR_PRODUCT_RANKS,
    advance_reduced_vector_product,
    compile_weighted_spin_vector_jordan,
    compile_weighted_spin_vector_product,
    couple_reduced_vectors,
    couple_reduced_vectors_commutator,
    couple_reduced_vectors_jordan,
    contract_reduced_tensors_scalar,
    tensor_delta_q,
    tensor_spherical_component_coupled_matrix,
)
from su2_nqs.su2.reduced_vector import (
    advance_weighted_spin_vector,
    compile_weighted_spin_vector,
    spherical_component_coupled_matrix,
)


VECTOR_COMPONENTS = (-2, 0, 2)


def _vector_components(table) -> dict[int, np.ndarray]:
    return {
        component: spherical_component_coupled_matrix(table, component)[0]
        for component in VECTOR_COMPONENTS
    }


def _direct_coupled_product(
    left: dict[int, np.ndarray],
    right: dict[int, np.ndarray],
    q_op: int,
    component_q: int,
    *,
    bracket: str = "ordered",
) -> np.ndarray:
    result = np.zeros_like(left[0])
    for mu_q in VECTOR_COMPONENTS:
        nu_q = component_q - mu_q
        if nu_q not in VECTOR_COMPONENTS:
            continue
        coefficient = clebsch_gordan(
            2, mu_q, 2, nu_q, q_op, component_q
        )
        ordered = left[mu_q] @ right[nu_q]
        if bracket == "ordered":
            value = ordered
        elif bracket == "jordan":
            value = 0.5 * (ordered + right[nu_q] @ left[mu_q])
        elif bracket == "commutator":
            value = 0.5 * (ordered - right[nu_q] @ left[mu_q])
        else:
            raise ValueError("unknown bracket")
        result += coefficient * value
    return result


def _maximum_component_error(table, expected, bracket: str) -> float:
    error = 0.0
    for component_q in range(-table.q_op, table.q_op + 1, 2):
        actual, _ = tensor_spherical_component_coupled_matrix(
            table, component_q
        )
        direct = _direct_coupled_product(
            expected[0], expected[1], table.q_op, component_q, bracket=bracket
        )
        error = max(error, float(np.max(np.abs(actual - direct))))
    return error


@pytest.mark.parametrize("n_sites", range(1, 7))
def test_sparse_vector_products_match_direct_component_matrices(
    n_sites: int,
) -> None:
    rng = np.random.default_rng(9120 + n_sites)
    left = compile_weighted_spin_vector(
        rng.normal(size=n_sites) + 0.2j * rng.normal(size=n_sites)
    )
    right = compile_weighted_spin_vector(
        rng.normal(size=n_sites) - 0.3j * rng.normal(size=n_sites)
    )
    components = (_vector_components(left), _vector_components(right))
    for q_op in VECTOR_PRODUCT_RANKS:
        table = couple_reduced_vectors(left, right, q_op)
        assert table.delta_q == tensor_delta_q(q_op)
        assert _maximum_component_error(table, components, "ordered") < 8.0e-13


def test_q024_channels_reconstruct_every_ordered_component_product() -> None:
    left = compile_weighted_spin_vector([0.2, -0.7j, 1.1, 0.3j])
    right = compile_weighted_spin_vector([-0.4j, 0.9, -0.3, 0.5])
    left_components = _vector_components(left)
    right_components = _vector_components(right)
    products = {
        q_op: couple_reduced_vectors(left, right, q_op)
        for q_op in VECTOR_PRODUCT_RANKS
    }
    for mu_q in VECTOR_COMPONENTS:
        for nu_q in VECTOR_COMPONENTS:
            component_q = mu_q + nu_q
            reconstructed = np.zeros_like(left_components[0])
            for q_op, table in products.items():
                if abs(component_q) > q_op:
                    continue
                matrix, _ = tensor_spherical_component_coupled_matrix(
                    table, component_q
                )
                reconstructed += clebsch_gordan(
                    2, mu_q, 2, nu_q, q_op, component_q
                ) * matrix
            np.testing.assert_allclose(
                reconstructed,
                left_components[mu_q] @ right_components[nu_q],
                atol=2.0e-13,
            )


@pytest.mark.parametrize("q_op", VECTOR_PRODUCT_RANKS)
def test_jordan_and_commutator_tables_include_coupled_exchange_phase(
    q_op: int,
) -> None:
    left = compile_weighted_spin_vector([0.8, -0.2, 0.4j, 1.1])
    right = compile_weighted_spin_vector([0.3j, 0.7, -0.5, 0.2])
    components = (_vector_components(left), _vector_components(right))
    jordan = couple_reduced_vectors_jordan(left, right, q_op)
    commutator = couple_reduced_vectors_commutator(left, right, q_op)
    assert _maximum_component_error(jordan, components, "jordan") < 2.0e-13
    assert (
        _maximum_component_error(commutator, components, "commutator")
        < 2.0e-13
    )
    ordered = couple_reduced_vectors(left, right, q_op).entry_dict()
    reconstructed = {
        key: jordan.entry_dict().get(key, 0.0j)
        + commutator.entry_dict().get(key, 0.0j)
        for key in set(ordered)
        | set(jordan.entry_dict())
        | set(commutator.entry_dict())
    }
    assert max(
        abs(reconstructed.get(key, 0.0j) - ordered.get(key, 0.0j))
        for key in set(reconstructed) | set(ordered)
    ) < 2.0e-13


def test_identical_vector_separates_even_jordan_and_odd_commutator() -> None:
    vector = compile_weighted_spin_vector([0.2, -0.9, 0.7, 0.4])
    for q_op in (0, 4):
        commutator = couple_reduced_vectors_commutator(vector, vector, q_op)
        assert commutator.entry_count == 0
    jordan_vector = couple_reduced_vectors_jordan(vector, vector, 2)
    assert jordan_vector.entry_count == 0
    assert couple_reduced_vectors_commutator(
        vector, vector, 2
    ).component_frobenius_norm_squared > 0.0

    single_spin = compile_weighted_spin_vector([1.0])
    assert couple_reduced_vectors(single_spin, single_spin, 4).entry_count == 0


def test_reduced_norm_equals_sum_of_component_frobenius_norms() -> None:
    left = compile_weighted_spin_vector([0.2, -0.7, 1.1])
    right = compile_weighted_spin_vector([0.4, 0.9, -0.3])
    for q_op in VECTOR_PRODUCT_RANKS:
        table = couple_reduced_vectors(left, right, q_op)
        direct = 0.0
        for component_q in range(-q_op, q_op + 1, 2):
            matrix, _ = tensor_spherical_component_coupled_matrix(
                table, component_q
            )
            direct += float(np.vdot(matrix, matrix).real)
        assert table.component_frobenius_norm_squared == pytest.approx(
            direct, abs=2.0e-13
        )


def test_vector_product_api_rejects_invalid_ranks_and_mismatched_blocks() -> None:
    two = compile_weighted_spin_vector([1.0, -0.2])
    three = compile_weighted_spin_vector([1.0, -0.2, 0.4])
    with pytest.raises(ValueError, match="q_op=0, 2, or 4"):
        couple_reduced_vectors(two, two, 6)
    with pytest.raises(ValueError, match="same block basis"):
        couple_reduced_vectors(two, three, 0)
    scalar = couple_reduced_vectors(two, two, 0)
    with pytest.raises(ValueError, match="projection"):
        tensor_spherical_component_coupled_matrix(scalar, 2)


@pytest.mark.parametrize("target_q", (0, 2))
def test_three_tensor_ranks_close_product_of_two_scalar_contractions(
    target_q: int,
) -> None:
    """Exact boundary identity behind the proposed two-action carrier."""

    left_first = compile_weighted_spin_vector([0.2, -0.7, 1.1])
    left_second = compile_weighted_spin_vector([-0.4, 0.3, 0.8])
    right_first = compile_weighted_spin_vector([0.9, -0.2, 0.5])
    right_second = compile_weighted_spin_vector([0.1, 0.6, -0.3])
    from su2_nqs.su2.reduced_vector import contract_reduced_vector_scalar

    first_scalar, basis = contract_reduced_vector_scalar(
        left_first, right_first, target_q, total_m=0
    )
    second_scalar, second_basis = contract_reduced_vector_scalar(
        left_second, right_second, target_q, total_m=0
    )
    assert second_basis == basis
    exact = first_scalar @ second_scalar
    reconstructed = np.zeros_like(exact)
    rank_terms = {}
    for q_op in VECTOR_PRODUCT_RANKS:
        left_product = couple_reduced_vectors(
            left_first, left_second, q_op
        )
        right_product = couple_reduced_vectors(
            right_first, right_second, q_op
        )
        term, product_basis = contract_reduced_tensors_scalar(
            left_product, right_product, target_q, total_m=0
        )
        assert product_basis == basis
        phase = -1.0 if (q_op // 2) % 2 else 1.0
        rank_terms[q_op] = phase * term
        reconstructed += rank_terms[q_op]
    np.testing.assert_allclose(reconstructed, exact, atol=7.0e-13)
    assert all(np.linalg.norm(rank_terms[q_op]) > 1.0e-8 for q_op in (0, 2, 4))


def _maximum_table_difference(first, second) -> float:
    left = first.entry_dict()
    right = second.entry_dict()
    return max(
        (
            abs(left.get(key, 0.0j) - right.get(key, 0.0j))
            for key in set(left) | set(right)
        ),
        default=0.0,
    )


@pytest.mark.parametrize("n_sites", range(2, 9))
@pytest.mark.parametrize("q_op", VECTOR_PRODUCT_RANKS)
def test_recursive_product_compiler_matches_direct_sparse_product(
    n_sites: int, q_op: int
) -> None:
    rng = np.random.default_rng(48100 + 10 * n_sites + q_op)
    left_weights = rng.normal(size=n_sites) + 0.13j * rng.normal(size=n_sites)
    right_weights = rng.normal(size=n_sites) - 0.17j * rng.normal(size=n_sites)
    recursive = compile_weighted_spin_vector_product(
        left_weights, right_weights, q_op, zero_tolerance=1.0e-13
    )
    left = compile_weighted_spin_vector(left_weights, zero_tolerance=1.0e-13)
    right = compile_weighted_spin_vector(right_weights, zero_tolerance=1.0e-13)
    direct = couple_reduced_vectors(
        left, right, q_op, zero_tolerance=1.0e-13
    )
    assert recursive.paths == direct.paths
    assert _maximum_table_difference(recursive, direct) < 2.0e-12


@pytest.mark.parametrize("q_op", VECTOR_PRODUCT_RANKS)
def test_one_step_product_advance_matches_direct_advanced_vectors(q_op: int) -> None:
    left = compile_weighted_spin_vector([0.2, -0.7, 1.1, 0.4j])
    right = compile_weighted_spin_vector([0.9, -0.2j, 0.5, -0.3])
    product = couple_reduced_vectors(left, right, q_op)
    advanced = advance_reduced_vector_product(
        product, left, right, -0.6 + 0.2j, 0.8 - 0.1j
    )
    direct = couple_reduced_vectors(
        advance_weighted_spin_vector(left, -0.6 + 0.2j),
        advance_weighted_spin_vector(right, 0.8 - 0.1j),
        q_op,
    )
    assert _maximum_table_difference(advanced, direct) < 5.0e-13


@pytest.mark.parametrize("q_op", VECTOR_PRODUCT_RANKS)
def test_recursive_jordan_compiler_matches_direct_jordan(q_op: int) -> None:
    left_weights = [0.2, -0.7j, 1.1, 0.4, -0.2j, 0.8]
    right_weights = [0.9j, -0.2, 0.5, -0.3j, 0.7, 0.1]
    recursive = compile_weighted_spin_vector_jordan(
        left_weights, right_weights, q_op
    )
    direct = couple_reduced_vectors_jordan(
        compile_weighted_spin_vector(left_weights),
        compile_weighted_spin_vector(right_weights),
        q_op,
    )
    assert _maximum_table_difference(recursive, direct) < 8.0e-13
