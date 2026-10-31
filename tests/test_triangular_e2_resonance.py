from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import physical_permutation_index_table
from su2_nqs.benchmarks.physical_interpretation import FixedSzPairActions
from su2_nqs.lattices.triangular_e2 import (
    E2_ORIENTATION_FRAME,
    J2_E2_ORIENTATION_FRAME,
    e2_a1_clebsch_gordan,
    e2_orientation_representation,
)
from su2_nqs.lattices.triangular_hnf import (
    C6,
    MX,
    J1_POSITIVE,
    J2_POSITIVE,
    matrix_power,
    matrix_product,
)
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.operators.triangular_resonance import (
    connected_triangular_a1_pair_action,
    connected_triangular_e2_pair_action,
    triangular_a1_pair_action,
    triangular_e2_actions,
    triangular_e2_moments,
    triangular_e2_pair_action,
)
from su2_nqs.operators.sz_graph_sparse import heisenberg_sz_graph_sparse
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.su2.sz_basis import enumerate_sz_configs


def test_e2_frame_and_d6_representation_relations() -> None:
    np.testing.assert_allclose(
        E2_ORIENTATION_FRAME @ E2_ORIENTATION_FRAME.T,
        np.eye(2),
        atol=4.0e-16,
    )
    np.testing.assert_allclose(E2_ORIENTATION_FRAME.sum(axis=1), 0.0, atol=4.0e-16)
    rotation = e2_orientation_representation(C6)
    reflection = e2_orientation_representation(MX)
    np.testing.assert_allclose(
        np.linalg.matrix_power(rotation, 6), np.eye(2), atol=2.0e-15
    )
    np.testing.assert_allclose(reflection @ reflection, np.eye(2), atol=2.0e-15)
    np.testing.assert_allclose(
        reflection @ rotation @ reflection,
        np.linalg.inv(rotation),
        atol=2.0e-15,
    )
    cg = e2_a1_clebsch_gordan()
    for operation in (
        matrix_power(C6, 1),
        matrix_power(C6, 2),
        matrix_product(MX, matrix_power(C6, 1)),
    ):
        representation = e2_orientation_representation(operation)
        np.testing.assert_allclose(
            representation @ cg @ representation.T, cg, atol=2.0e-15
        )
        np.testing.assert_allclose(
            e2_orientation_representation(
                operation, axes=J2_POSITIVE, frame=J2_E2_ORIENTATION_FRAME
            ),
            e2_orientation_representation(
                operation, axes=J1_POSITIVE, frame=E2_ORIENTATION_FRAME
            ),
            atol=2.0e-15,
        )


def test_e2_actions_transform_covariantly_and_pair_is_a1() -> None:
    supercell = ((6, 4), (0, 2))
    basis = tuple(enumerate_sz_configs(12, 6))
    actions = FixedSzPairActions(basis)
    rng = np.random.default_rng(2026091801)
    state = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))
    state /= np.linalg.norm(state)

    for operation in (C6, MX):
        from su2_nqs.lattices.triangular_hnf import hnf_site_permutation

        site_image = hnf_site_permutation(supercell, operation=operation)
        indices = physical_permutation_index_table(basis, (site_image,))[0]
        transformed = state[indices]
        direct = np.stack(triangular_e2_actions(transformed, actions.pair, supercell))
        original = np.stack(triangular_e2_actions(state, actions.pair, supercell))
        pulled = original[:, indices]
        representation = e2_orientation_representation(operation)
        errors = (
            np.linalg.norm(direct - representation @ pulled),
            np.linalg.norm(direct - representation.T @ pulled),
        )
        assert min(errors) < 2.0e-13

        pair_direct = triangular_e2_pair_action(transformed, actions.pair, supercell)
        pair_pulled = triangular_e2_pair_action(state, actions.pair, supercell)[indices]
        np.testing.assert_allclose(pair_direct, pair_pulled, atol=2.0e-13)


def test_pair_action_is_hermitian_and_moments_match_direct_sum_rule() -> None:
    supercell = ((4, 0), (0, 2))
    basis = tuple(enumerate_sz_configs(8, 4))
    actions = FixedSzPairActions(basis)
    rng = np.random.default_rng(2026091802)
    first = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))
    second = rng.normal(size=len(basis)) + 1.0j * rng.normal(size=len(basis))
    k_first = triangular_e2_pair_action(first, actions.pair, supercell)
    k_second = triangular_e2_pair_action(second, actions.pair, supercell)
    np.testing.assert_allclose(
        np.vdot(first, k_second), np.vdot(k_first, second), atol=2.0e-13
    )
    cross_first = triangular_e2_pair_action(
        first,
        actions.pair,
        supercell,
        first_displacements=J1_POSITIVE,
        second_displacements=J2_POSITIVE,
    )
    cross_second = triangular_e2_pair_action(
        second,
        actions.pair,
        supercell,
        first_displacements=J1_POSITIVE,
        second_displacements=J2_POSITIVE,
    )
    np.testing.assert_allclose(
        np.vdot(first, cross_second), np.vdot(cross_first, second), atol=2.0e-13
    )
    a1_first = triangular_a1_pair_action(
        first,
        actions.pair,
        supercell,
        first_displacements=J1_POSITIVE,
        second_displacements=J2_POSITIVE,
    )
    a1_second = triangular_a1_pair_action(
        second,
        actions.pair,
        supercell,
        first_displacements=J1_POSITIVE,
        second_displacements=J2_POSITIVE,
    )
    np.testing.assert_allclose(
        np.vdot(first, a1_second), np.vdot(a1_first, second), atol=2.0e-13
    )

    state = first / np.linalg.norm(first)
    connected, mean = connected_triangular_e2_pair_action(
        state, actions.pair, supercell
    )
    assert abs(np.vdot(state, connected)) < 2.0e-13
    np.testing.assert_allclose(
        mean, np.vdot(state, triangular_e2_pair_action(state, actions.pair, supercell))
    )
    a1_connected, _ = connected_triangular_a1_pair_action(
        state,
        actions.pair,
        supercell,
        first_displacements=J1_POSITIVE,
        second_displacements=J2_POSITIVE,
    )
    assert abs(np.vdot(state, a1_connected)) < 2.0e-13

    bonds = triangular_hnf_j1j2_bonds(supercell, j1=1.0, j2=0.075)
    hamiltonian, h_basis = heisenberg_sz_graph_sparse(8, bonds, n_up=4)
    assert tuple(h_basis) == basis
    moments = triangular_e2_moments(
        state, actions.pair, supercell, lambda vector: hamiltonian @ vector
    )
    component_actions = triangular_e2_actions(state, actions.pair, supercell)
    direct_m0 = sum(
        np.vdot(image - np.vdot(state, image) * state, image - np.vdot(state, image) * state).real
        for image in component_actions
    )
    np.testing.assert_allclose(moments.m0, direct_m0, atol=2.0e-13)


def test_n36_full_gamma_a1_group_is_nontrivial() -> None:
    group = triangular_hnf_gamma_a1_group(((6, 0), (0, 6)))
    assert len(group) == 432
    assert all(len(permutation) == 36 for permutation in group)
