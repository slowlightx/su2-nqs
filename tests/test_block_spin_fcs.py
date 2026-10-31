import numpy as np

from su2_nqs.benchmarks.block_spin_fcs import (
    block_spin_fcs,
    block_spin_fcs_with_tangent,
    classical_fisher_information,
    doubled_spins_can_close_to_singlet,
    fcs_score_projection_diagnostics,
    grouped_fusion_tree,
    moment_fisher_lower_bound,
    moment_fisher_diagnostics,
    tree_block_spin_fcs_with_tangent,
)
from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.tree_rotations import recoupling_matrix_sparse


def test_two_spin_singlet_has_one_half_in_each_block():
    singlet = np.asarray([0.0, -1.0, 1.0, 0.0]) / np.sqrt(2.0)
    distribution = block_spin_fcs(singlet, ((0,), (1,)))
    assert distribution.keys() == {(1, 1)}
    np.testing.assert_allclose(distribution[(1, 1)], 1.0, atol=1e-14)


def test_computational_and_grouped_tree_fcs_agree_for_four_spin_singlet():
    paths = enumerate_paths(4, 0)
    source = FusionTree.left_associated(4)
    target, indices = grouped_fusion_tree((2, 2))
    recoupling, _ = recoupling_matrix_sparse(source, target, 0)
    amplitudes = np.asarray([1.0 + 0.2j, -0.3 + 0.7j])
    amplitudes /= np.linalg.norm(amplitudes)
    tangent = np.asarray([0.1 - 0.4j, 0.2 + 0.05j])
    tangent -= amplitudes * np.vdot(amplitudes, tangent)
    full = coupled_basis_matrix(paths, 0, 0) @ amplitudes
    full_tangent = coupled_basis_matrix(paths, 0, 0) @ tangent
    computational = block_spin_fcs_with_tangent(full, full_tangent, ((0, 1), (2, 3)))
    tree = tree_block_spin_fcs_with_tangent(
        recoupling @ amplitudes,
        recoupling @ tangent,
        target.enumerate_states(0),
        indices,
    )
    for left, right in zip(computational, tree, strict=True):
        keys = sorted(left.keys() | right.keys())
        np.testing.assert_allclose(
            [left.get(key, 0.0) for key in keys],
            [right.get(key, 0.0) for key in keys],
            atol=1e-12,
        )


def _permuted_left_coordinates(full, order, n, target_q=0):
    reordered = full.reshape((2,) * n).transpose(order).reshape(-1)
    basis = coupled_basis_matrix(enumerate_paths(n, target_q), target_q, 0)
    return basis.conj().T @ reordered


def test_noncontiguous_physical_groups_match_leaf_permutation_and_recoupling():
    rng = np.random.default_rng(19)
    n = 6
    paths = enumerate_paths(n, 0)
    transform = coupled_basis_matrix(paths, 0, 0)
    amplitudes = rng.normal(size=len(paths)) + 1j * rng.normal(size=len(paths))
    amplitudes /= np.linalg.norm(amplitudes)
    tangent = rng.normal(size=len(paths)) + 1j * rng.normal(size=len(paths))
    tangent -= amplitudes * np.vdot(amplitudes, tangent)
    full, full_tangent = transform @ amplitudes, transform @ tangent
    for groups in (
        ((0, 2, 4), (1, 3, 5)),
        ((0, 3), (1, 4), (2, 5)),
    ):
        expected = block_spin_fcs_with_tangent(full, full_tangent, groups)
        order = tuple(site for group in groups for site in group)
        permuted = _permuted_left_coordinates(full, order, n)
        permuted_tangent = _permuted_left_coordinates(full_tangent, order, n)
        source = FusionTree.left_associated(n)
        for association in ("left", "right"):
            target, indices = grouped_fusion_tree(
                tuple(len(group) for group in groups), association=association
            )
            recoupling, _ = recoupling_matrix_sparse(source, target, 0)
            actual = tree_block_spin_fcs_with_tangent(
                recoupling @ permuted,
                recoupling @ permuted_tangent,
                target.enumerate_states(0),
                indices,
            )
            for left, right in zip(expected, actual, strict=True):
                keys = sorted(left.keys() | right.keys())
                np.testing.assert_allclose(
                    [left.get(key, 0.0) for key in keys],
                    [right.get(key, 0.0) for key in keys],
                    atol=2e-12,
                )


def test_fcs_is_invariant_to_internal_site_order_and_obeys_information_bound():
    rng = np.random.default_rng(8)
    paths = enumerate_paths(6, 0)
    amplitudes = rng.normal(size=len(paths)) + 1j * rng.normal(size=len(paths))
    amplitudes /= np.linalg.norm(amplitudes)
    tangent = rng.normal(size=len(paths)) + 1j * rng.normal(size=len(paths))
    tangent -= amplitudes * np.vdot(amplitudes, tangent)
    transform = coupled_basis_matrix(paths, 0, 0)
    full = transform @ amplitudes
    full_tangent = transform @ tangent
    first = block_spin_fcs_with_tangent(full, full_tangent, ((0, 2, 4), (1, 3, 5)))
    second = block_spin_fcs_with_tangent(full, full_tangent, ((4, 2, 0), (5, 3, 1)))
    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(tuple(left.values()), tuple(right.values()), atol=1e-12)

    information, dropped = classical_fisher_information(*first)
    quantum = 4.0 * float(np.real(np.vdot(full_tangent, full_tangent)))
    assert dropped < 1e-14
    assert information <= quantum + 1e-10
    assert moment_fisher_lower_bound(*first) <= information + 1e-10
    for key, probability in first[0].items():
        if not doubled_spins_can_close_to_singlet(key):
            assert probability < 1e-12


def test_three_group_tree_associations_have_the_same_physical_label_support():
    left, left_indices = grouped_fusion_tree((2, 2, 2), association="left")
    right, right_indices = grouped_fusion_tree((2, 2, 2), association="right")
    assert len(left.enumerate_states(0)) == len(right.enumerate_states(0)) == 5
    assert all(index is not None for index in left_indices + right_indices)
    for tree, indices in ((left, left_indices), (right, right_indices)):
        for state in tree.enumerate_states(0):
            labels = tuple(state.labels[index] for index in indices if index is not None)
            assert doubled_spins_can_close_to_singlet(labels)


def test_moment_information_handles_redundant_singlet_features():
    probabilities = {(0, 0): 0.25, (2, 2): 0.75}
    derivatives = {(0, 0): -0.3, (2, 2): 0.3}
    full, _ = classical_fisher_information(probabilities, derivatives)
    diagnostics = moment_fisher_diagnostics(probabilities, derivatives)
    np.testing.assert_allclose(diagnostics["information"], full, atol=1e-12)
    assert diagnostics["covariance_rank"] == 1


def test_score_projection_resolves_casimir_and_distribution_shape_information():
    probabilities = {(0,): 0.2, (2,): 0.5, (4,): 0.3}
    derivatives = {(0,): 0.1, (2,): -0.2, (4,): 0.1}
    diagnostics = fcs_score_projection_diagnostics(probabilities, derivatives)
    full, _ = classical_fisher_information(probabilities, derivatives)
    moment = moment_fisher_lower_bound(probabilities, derivatives)
    np.testing.assert_allclose(diagnostics["full_information"], full, atol=1e-14)
    np.testing.assert_allclose(diagnostics["casimir_information"], moment, atol=1e-14)
    np.testing.assert_allclose(
        diagnostics["shape_information"], full - moment, atol=1e-14
    )
    np.testing.assert_allclose(
        sum(diagnostics["outcome_shape_information"].values()),
        diagnostics["shape_information"],
        atol=1e-14,
    )
    assert diagnostics["shape_information"] > 0.0
    assert diagnostics["pythagorean_closure_error"] < 1e-14
    assert abs(diagnostics["score_projection_cross_term"]) < 1e-14


def test_score_projection_vanishes_when_one_casimir_coordinate_is_sufficient():
    probabilities = {(0, 0): 0.25, (2, 2): 0.75}
    derivatives = {(0, 0): -0.3, (2, 2): 0.3}
    diagnostics = fcs_score_projection_diagnostics(probabilities, derivatives)
    assert diagnostics["covariance_rank"] == 1
    assert diagnostics["shape_information"] < 1e-14
    assert diagnostics["pythagorean_closure_error"] < 1e-14


def test_fcs_rejects_a_tangent_that_changes_normalization():
    singlet = np.asarray([0.0, -1.0, 1.0, 0.0]) / np.sqrt(2.0)
    with np.testing.assert_raises_regex(ValueError, "preserve wavefunction normalization"):
        block_spin_fcs_with_tangent(singlet, singlet, ((0,), (1,)))
