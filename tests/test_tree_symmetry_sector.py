import numpy as np

from su2_nqs.benchmarks.tree_symmetry_sector import translation_character_basis
from su2_nqs.lattices.coset_tree import square_translation_coset_order
from su2_nqs.lattices.square_symmetry import square_site_generators
from su2_nqs.operators.tree_automorphism import compile_tree_automorphism
from su2_nqs.su2.fusion_tree import FusionTree


def _plan_matrix(plan, states):
    positions = {state.labels: index for index, state in enumerate(states)}
    result = np.zeros((len(states), len(states)), dtype=np.complex128)
    for column, state in enumerate(states):
        mapped, phase = plan.apply(state)
        result[positions[mapped.labels], column] = phase
    return result


def test_orbit_columns_equal_dense_translation_character_projector() -> None:
    linear_size, target_q = 2, 0
    tree = FusionTree.balanced(linear_size**2)
    order = square_translation_coset_order(linear_size)
    generators = square_site_generators(linear_size)
    tx = compile_tree_automorphism(tree, order, generators["tx"])
    ty = compile_tree_automorphism(tree, order, generators["ty"])
    states = tree.enumerate_states(target_q)
    tx_matrix = _plan_matrix(tx, states)
    ty_matrix = _plan_matrix(ty, states)

    projectors = []
    dimensions = 0
    for kx in range(linear_size):
        for ky in range(linear_size):
            sector = translation_character_basis(
                tree, target_q, tx, ty, linear_size, (kx, ky)
            )
            columns = sector.columns.toarray()
            projector_from_columns = columns @ columns.conj().T
            dense_projector = sum(
                np.exp(2.0j * np.pi * (kx * x + ky * y) / linear_size)
                * np.linalg.matrix_power(tx_matrix, x)
                @ np.linalg.matrix_power(ty_matrix, y)
                for x in range(linear_size)
                for y in range(linear_size)
            ) / linear_size**2
            np.testing.assert_allclose(
                projector_from_columns, dense_projector, atol=2.0e-14
            )
            assert sector.audit["orthonormality_error"] < 2.0e-14
            assert sector.audit["tx_eigen_residual"] < 2.0e-14
            assert sector.audit["ty_eigen_residual"] < 2.0e-14
            projectors.append(projector_from_columns)
            dimensions += sector.audit["sector_dimension"]
    np.testing.assert_allclose(sum(projectors), np.eye(len(states)), atol=2.0e-14)
    assert dimensions == len(states)


def test_square_q4_gamma_orbit_basis_is_sparse_and_orthonormal() -> None:
    linear_size, target_q = 4, 4
    tree = FusionTree.balanced(linear_size**2)
    order = square_translation_coset_order(linear_size)
    generators = square_site_generators(linear_size)
    sector = translation_character_basis(
        tree,
        target_q,
        compile_tree_automorphism(tree, order, generators["tx"]),
        compile_tree_automorphism(tree, order, generators["ty"]),
        linear_size,
        (0, 0),
    )
    assert sector.audit["full_dimension"] == 3640
    assert 0 < sector.audit["sector_dimension"] < 3640
    assert sector.audit["stored_nnz"] <= 16 * sector.audit["sector_dimension"]
    assert sector.audit["orthonormality_error"] < 2.0e-14
    assert sector.audit["tx_eigen_residual"] < 2.0e-14
    assert sector.audit["ty_eigen_residual"] < 2.0e-14
