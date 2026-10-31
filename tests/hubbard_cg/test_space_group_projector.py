import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from su2_nqs.benchmarks.hubbard_coupled_basis import (
    hubbard_coupled_basis_coo,
    hubbard_coupled_basis_matrix,
)

from su2_nqs.benchmarks.hubbard_space_group import (
    fermionic_permutation_index_sign_tables,
    fermionic_reynolds_project_state,
    rectangular_translation_group,
)
from su2_nqs.operators.hubbard_fock import (
    apply_site_permutation,
    enumerate_hubbard_tokens,
    hubbard_fock_sector_matrix,
    hubbard_fock_sector_sparse,
)
from su2_nqs.su2.hubbard_fusion import enumerate_hubbard_fusion_states


jax.config.update("jax_enable_x64", True)


def test_signed_gather_matches_dense_action_and_is_a_projector() -> None:
    basis = enumerate_hubbard_tokens(6, 6, total_m2=0)
    group = rectangular_translation_group(2, 3)
    indices, signs = fermionic_permutation_index_sign_tables(basis, group)
    positions = {state: index for index, state in enumerate(basis)}
    rng = np.random.default_rng(20260914)
    vector = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))

    expected = np.zeros_like(vector)
    for image in group:
        action = np.zeros((len(basis), len(basis)), dtype=np.complex128)
        for column, state in enumerate(basis):
            target, sign = apply_site_permutation(state, image)
            action[positions[target], column] = sign
        expected += action @ vector / len(group)
    actual = np.asarray(
        fermionic_reynolds_project_state(
            jnp.asarray(vector), indices, signs
        )
    )
    np.testing.assert_allclose(actual, expected, atol=5.0e-14)
    projected_twice = np.asarray(
        fermionic_reynolds_project_state(
            jnp.asarray(actual), indices, signs
        )
    )
    np.testing.assert_allclose(projected_twice, actual, atol=5.0e-14)


def test_rectangular_translation_group_is_deduplicated() -> None:
    group = rectangular_translation_group(2, 4)
    assert len(group) == 8
    assert len(set(group)) == 8


def test_sparse_fusion_to_fixed_m_transform_matches_dense_reference() -> None:
    states = enumerate_hubbard_fusion_states(4, 4, 0)
    physical_basis = enumerate_hubbard_tokens(4, 4, total_m2=0)
    rows, columns, values = hubbard_coupled_basis_coo(
        states, physical_basis, total_m=0
    )
    actual = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(physical_basis), len(states)),
    ).toarray()
    dense, fock_basis = hubbard_coupled_basis_matrix(states, total_m=0)
    positions = {mask: row for row, mask in enumerate(fock_basis)}
    selected = dense[
        [positions[sum(
            ((token in (1, 3)) << (2 * site))
            | ((token in (2, 3)) << (2 * site + 1))
            for site, token in enumerate(state)
        )] for state in physical_basis]
    ]
    np.testing.assert_allclose(actual, selected, atol=3.0e-15)
    np.testing.assert_allclose(actual.conj().T @ actual, np.eye(len(states)), atol=3.0e-14)


def test_sparse_physical_sector_constructor_matches_dense() -> None:
    bonds = ((0, 1), (1, 3), (3, 2), (2, 0))
    dense, dense_basis = hubbard_fock_sector_matrix(
        4, 4, 0, bonds, interaction=3.0
    )
    actual, sparse_basis = hubbard_fock_sector_sparse(
        4, 4, 0, bonds, interaction=3.0
    )
    assert sparse_basis == dense_basis
    np.testing.assert_allclose(actual.toarray(), dense, atol=0.0, rtol=0.0)
