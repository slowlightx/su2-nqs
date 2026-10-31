import numpy as np
import pytest
from scipy import sparse

from su2_nqs.lattices.coset_tree import (
    cyclic_bit_reversed_order,
    square_translation_coset_order,
    square_translation_quotient_frame_order,
    translation_quotient_tree,
)
from su2_nqs.lattices.square_symmetry import square_site_generators
from su2_nqs.operators.permutation import site_permutation_fusion_matrix
from su2_nqs.operators.tree_automorphism import (
    compile_tree_automorphism,
    compile_tree_frame_isomorphism,
)
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.tree_rotations import recoupling_matrix_sparse


def _leaf_image(order, physical_image):
    inverse = {site: leaf for leaf, site in enumerate(order)}
    return tuple(inverse[physical_image[site]] for site in order)


def _plan_matrix(plan, target_q):
    source_states = plan.source_tree.enumerate_states(target_q)
    target_states = plan.target_tree.enumerate_states(target_q)
    target_positions = {state.labels: index for index, state in enumerate(target_states)}
    rows, columns, values = [], [], []
    for column, state in enumerate(source_states):
        mapped, phase = plan.apply(state)
        rows.append(target_positions[mapped.labels])
        columns.append(column)
        values.append(phase)
    return sparse.coo_matrix(
        (values, (rows, columns)), shape=(len(target_states), len(source_states))
    ).tocsr()


@pytest.mark.parametrize("n,target_q", ((4, 0), (4, 2), (8, 0), (8, 2)))
def test_coset_tree_translation_matches_independent_fusion_permutation(
    n: int, target_q: int
) -> None:
    order = cyclic_bit_reversed_order(n)
    tree = FusionTree.balanced(n)
    physical_image = tuple((site + 1) % n for site in range(n))
    plan = compile_tree_automorphism(tree, order, physical_image)
    actual = _plan_matrix(plan, target_q)

    leaf_image = _leaf_image(order, physical_image)
    left_action = site_permutation_fusion_matrix(leaf_image, target_q=target_q)
    left_to_tree, _ = recoupling_matrix_sparse(
        FusionTree.left_associated(n), tree, target_q
    )
    expected = (left_to_tree @ left_action @ left_to_tree.T).tocsr()
    np.testing.assert_allclose(actual.toarray(), expected.toarray(), atol=5.0e-13)
    np.testing.assert_allclose(
        (actual.T @ actual).toarray(), np.eye(actual.shape[0]), atol=1.0e-14
    )


def test_square_coset_tree_has_support_one_translations_and_two_frame_c4() -> None:
    linear_size = 4
    tree = FusionTree.balanced(linear_size**2)
    generators = square_site_generators(linear_size)
    order_x = square_translation_coset_order(linear_size, primary_axis="x")
    order_y = square_translation_coset_order(linear_size, primary_axis="y")

    tx = compile_tree_automorphism(tree, order_x, generators["tx"])
    ty = compile_tree_automorphism(tree, order_x, generators["ty"])
    c4_between_frames = compile_tree_frame_isomorphism(
        tree, tree, order_x, order_y, generators["c4"]
    )
    states = tree.enumerate_states(4)
    for plan in (tx, ty, c4_between_frames):
        images = [plan.apply(state) for state in states]
        assert {state.labels for state, _ in images} == {
            state.labels for state in states
        }
        assert {phase for _, phase in images} <= {-1.0, 1.0}

    with pytest.raises(ValueError, match="does not preserve"):
        compile_tree_automorphism(tree, order_x, generators["c4"])


def test_coset_order_rejects_non_power_of_two_and_unknown_axis() -> None:
    with pytest.raises(ValueError, match="power of two"):
        cyclic_bit_reversed_order(6)
    with pytest.raises(ValueError, match="primary_axis"):
        square_translation_coset_order(4, primary_axis="z")


def _permutation_power(permutation, exponent):
    result = tuple(range(len(permutation)))
    for _ in range(exponent):
        result = tuple(permutation[site] for site in result)
    return result


def test_six_by_six_quotient_frames_make_all_translations_support_one() -> None:
    """The order-three quotient is a frame register, not a binary automorphism."""

    linear_size = 6
    quotient_size = 3
    tree = translation_quotient_tree(linear_size, quotient_size)
    generators = square_site_generators(linear_size)
    orders = {
        (x, y): square_translation_quotient_frame_order(
            linear_size, quotient_size, frame=(x, y)
        )
        for x in range(quotient_size)
        for y in range(quotient_size)
    }
    assert tree.n_leaves == 36
    assert len(set(orders.values())) == quotient_size**2

    tx_swaps = []
    ty_swaps = []
    for x in range(quotient_size):
        for y in range(quotient_size):
            tx = compile_tree_frame_isomorphism(
                tree,
                tree,
                orders[(x, y)],
                orders[((x + 1) % quotient_size, y)],
                generators["tx"],
            )
            ty = compile_tree_frame_isomorphism(
                tree,
                tree,
                orders[(x, y)],
                orders[(x, (y + 1) % quotient_size)],
                generators["ty"],
            )
            tx_swaps.append(sum(tx.child_swaps))
            ty_swaps.append(sum(ty.child_swaps))
    assert tx_swaps == [0, 0, 0, 0, 0, 0, 9, 9, 9]
    assert ty_swaps == [0, 0, 18, 0, 0, 18, 0, 0, 18]

    reference = orders[(0, 0)]
    compile_tree_automorphism(
        tree, reference, _permutation_power(generators["tx"], quotient_size)
    )
    compile_tree_automorphism(
        tree, reference, _permutation_power(generators["ty"], quotient_size)
    )
    with pytest.raises(ValueError, match="does not preserve"):
        compile_tree_automorphism(tree, reference, generators["tx"])
    with pytest.raises(ValueError, match="does not preserve"):
        compile_tree_automorphism(tree, reference, generators["ty"])


def test_cross_frame_translation_is_not_a_fixed_basis_projector_action() -> None:
    """A support-one frame change is not ``U_tx`` in one common basis.

    At ``L=3, Q=3`` translating the physical hierarchy from frame ``(0,0)``
    to frame ``(1,0)`` leaves every abstract tree label unchanged.  The true
    physical translation represented in the fixed frame ``(0,0)`` is instead
    nontrivial.  Keeping both statements in one test prevents cross-frame
    chart/evaluation profiles from being promoted to momentum projectors.
    """

    linear_size = quotient_size = 3
    target_q = 1  # Nine spin-1/2 sites have only odd doubled-total-spin q.
    tree = translation_quotient_tree(linear_size, quotient_size)
    generators = square_site_generators(linear_size)
    reference_order = square_translation_quotient_frame_order(
        linear_size, quotient_size, frame=(0, 0)
    )
    translated_order = square_translation_quotient_frame_order(
        linear_size, quotient_size, frame=(1, 0)
    )

    cross_frame = compile_tree_frame_isomorphism(
        tree,
        tree,
        reference_order,
        translated_order,
        generators["tx"],
    )
    states = tree.enumerate_states(target_q)
    for state in states:
        mapped, phase = cross_frame.apply(state)
        assert mapped == state
        assert phase == 1.0
    cross_frame_matrix = _plan_matrix(cross_frame, target_q).toarray()
    np.testing.assert_array_equal(cross_frame_matrix, np.eye(len(states)))

    # Represent the same physical Tx while keeping the physical tree frame
    # fixed.  This independent recoupling construction is the actual operator
    # that may enter a character projector.
    leaf_image = _leaf_image(reference_order, generators["tx"])
    left_action = site_permutation_fusion_matrix(
        leaf_image, target_q=target_q
    )
    left_to_tree, _ = recoupling_matrix_sparse(
        FusionTree.left_associated(linear_size**2), tree, target_q
    )
    fixed_frame_action = (
        left_to_tree @ left_action @ left_to_tree.T
    ).toarray()
    assert np.linalg.norm(fixed_frame_action - np.eye(len(states))) > 1.0
    assert not np.allclose(cross_frame_matrix, fixed_frame_action, atol=1.0e-12)
