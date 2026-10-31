from __future__ import annotations

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.lattices.fusion_catalog import (
    DisjointMergeRoute,
    FusionSupport,
    ReducedFusionCatalog,
    group_close_merge_routes,
    group_close_supports,
    route_image,
    support_image,
)
from su2_nqs.lattices.spatial_symmetry import permutation_group_closure
from su2_nqs.lattices.spatial_symmetry import compose_site_permutations
from su2_nqs.su2.graded_recoupling import (
    canonical_child_merge_sign,
    canonical_child_merge_sign_jax,
    canonical_hubbard_child_merge_sign,
    canonical_hubbard_child_merge_sign_jax,
    validated_child_merge_sites,
)
from su2_nqs.su2.hubbard_cg_irrep import DOUBLON, DOWN, EMPTY, UP


def _two_site_group() -> tuple[tuple[int, ...], ...]:
    return permutation_group_closure(((1, 0),))


def test_support_and_group_closures_have_stable_canonical_order() -> None:
    group = _two_site_group()
    support = FusionSupport((1, 0))
    assert support.sites == (0, 1)
    assert support_image(FusionSupport((0,)), (1, 0)) == FusionSupport((1,))
    assert group_close_supports(((0,),), reversed(group)) == (
        FusionSupport((0,)),
        FusionSupport((1,)),
    )

    seed = DisjointMergeRoute(
        FusionSupport((0, 1)), FusionSupport((0,)), FusionSupport((1,))
    )
    routes = group_close_merge_routes((seed,), reversed(group))
    assert routes == (
        seed,
        DisjointMergeRoute(
            FusionSupport((0, 1)), FusionSupport((1,)), FusionSupport((0,))
        ),
    )
    assert route_image(seed, (1, 0)) == routes[1]
    for left in group:
        for right in group:
            assert route_image(route_image(seed, right), left) == route_image(
                seed, compose_site_permutations(left, right)
            )


def test_merge_route_rejects_overlap_and_missing_parent_legs() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        DisjointMergeRoute(
            FusionSupport((0, 1, 2)),
            FusionSupport((0, 1)),
            FusionSupport((1, 2)),
        )
    with pytest.raises(ValueError, match="cover the parent exactly"):
        DisjointMergeRoute(
            FusionSupport((0, 1, 2)), FusionSupport((0,)), FusionSupport((1,))
        )


def test_catalog_rejects_group_nonclosure_and_missing_dependencies() -> None:
    group = _two_site_group()
    supports = (
        FusionSupport((0,)),
        FusionSupport((1,)),
        FusionSupport((0, 1)),
    )
    forward = DisjointMergeRoute(supports[2], supports[0], supports[1])
    backward = DisjointMergeRoute(supports[2], supports[1], supports[0])
    with pytest.raises(ValueError, match="routes are not closed"):
        ReducedFusionCatalog(2, group, supports, (forward,))
    catalog = ReducedFusionCatalog(
        2,
        tuple(reversed(group)),
        tuple(reversed(supports)),
        (backward, forward),
    )
    assert catalog.supports == supports
    assert catalog.routes == (forward, backward)

    with pytest.raises(ValueError, match="must be listed"):
        ReducedFusionCatalog(
            3,
            ((0, 1, 2),),
            (
                FusionSupport((0,)),
                FusionSupport((1,)),
                FusionSupport((2,)),
                FusionSupport((0, 1, 2)),
            ),
            (
                DisjointMergeRoute(
                    FusionSupport((0, 1)),
                    FusionSupport((0,)),
                    FusionSupport((1,)),
                ),
            ),
        )

    three_site_group = permutation_group_closure(((1, 0, 2),))
    leaves = tuple(FusionSupport((site,)) for site in range(3))
    pair = FusionSupport((0, 2))
    root = FusionSupport((0, 1, 2))
    pair_route = DisjointMergeRoute(pair, leaves[0], leaves[2])
    root_route = DisjointMergeRoute(root, pair, leaves[1])
    with pytest.raises(ValueError, match="supports are not closed"):
        ReducedFusionCatalog(
            3,
            three_site_group,
            (*leaves, pair, root),
            (pair_route, root_route),
        )


def test_catalog_rejects_computed_supports_that_do_not_feed_the_root() -> None:
    leaves = tuple(FusionSupport((site,)) for site in range(4))
    dead_pair = FusionSupport((0, 1))
    live_left = FusionSupport((0, 2))
    live_right = FusionSupport((1, 3))
    root = FusionSupport((0, 1, 2, 3))
    routes = (
        DisjointMergeRoute(dead_pair, leaves[0], leaves[1]),
        DisjointMergeRoute(live_left, leaves[0], leaves[2]),
        DisjointMergeRoute(live_right, leaves[1], leaves[3]),
        DisjointMergeRoute(root, live_left, live_right),
    )
    with pytest.raises(ValueError, match="must feed the all-site root"):
        ReducedFusionCatalog(
            4,
            ((0, 1, 2, 3),),
            (*leaves, dead_pair, live_left, live_right, root),
            routes,
        )


def _occupied_modes(tokens: tuple[int, ...], sites: tuple[int, ...]) -> list[int]:
    modes = []
    for site in sites:
        token = tokens[site]
        if token in (UP, DOUBLON):
            modes.append(2 * site)
        if token in (DOWN, DOUBLON):
            modes.append(2 * site + 1)
    return modes


def _explicit_merge_sign(
    tokens: tuple[int, ...], left: tuple[int, ...], right: tuple[int, ...]
) -> int:
    concatenated = _occupied_modes(tokens, left) + _occupied_modes(tokens, right)
    inversions = sum(
        concatenated[first] > concatenated[second]
        for first in range(len(concatenated))
        for second in range(first + 1, len(concatenated))
    )
    return -1 if inversions % 2 else 1


def test_canonical_child_merge_sign_matches_all_small_hubbard_mode_inversions() -> None:
    tokens_by_site = (EMPTY, UP, DOWN, DOUBLON)
    n_sites = 4
    for membership in product((0, 1, 2), repeat=n_sites):
        left = tuple(site for site, owner in enumerate(membership) if owner == 1)
        right = tuple(site for site, owner in enumerate(membership) if owner == 2)
        if not left or not right:
            continue
        for tokens in product(tokens_by_site, repeat=n_sites):
            occupations = tuple(
                0 if token == EMPTY else 2 if token == DOUBLON else 1
                for token in tokens
            )
            assert canonical_child_merge_sign(left, right, occupations) == (
                _explicit_merge_sign(tokens, left, right)
            )


def test_canonical_child_merge_sign_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        canonical_child_merge_sign((0, 1), (1, 2), (1, 1, 1))
    with pytest.raises(ValueError, match="canonical increasing"):
        canonical_child_merge_sign((1, 0), (2,), (1, 1, 1))
    with pytest.raises(ValueError, match=r"\{0,1,2\}"):
        canonical_child_merge_sign((0,), (1,), (3, 0))


def test_hubbard_token_wrapper_does_not_confuse_down_with_a_doublon() -> None:
    assert canonical_child_merge_sign(
        (1,), (0,), occupation_counts=(1, 1)
    ) == -1
    assert canonical_hubbard_child_merge_sign((1,), (0,), (UP, DOWN)) == -1
    assert canonical_hubbard_child_merge_sign((1,), (0,), (UP, DOUBLON)) == 1
    with pytest.raises(ValueError, match="token must be one of"):
        canonical_hubbard_child_merge_sign((1,), (0,), (UP, 4))


def test_child_merge_sign_jax_kernels_support_jit_vmap_and_batches() -> None:
    left, right = validated_child_merge_sites((1, 3), (0, 2), 4)
    counts = np.asarray(
        ((1, 1, 0, 1), (2, 1, 1, 0), (0, 2, 1, 1)), dtype=np.int32
    )
    expected = np.asarray(
        [canonical_child_merge_sign(left, right, row) for row in counts]
    )
    single_kernel = jax.jit(
        lambda row: canonical_child_merge_sign_jax(left, right, row)
    )
    vmapped = jax.jit(jax.vmap(single_kernel))(jnp.asarray(counts))
    batched = jax.jit(
        lambda rows: canonical_child_merge_sign_jax(left, right, rows)
    )(jnp.asarray(counts))
    np.testing.assert_array_equal(np.asarray(vmapped), expected)
    np.testing.assert_array_equal(np.asarray(batched), expected)

    tokens = jnp.asarray(
        ((UP, DOWN, EMPTY, UP), (DOUBLON, DOWN, UP, EMPTY)), dtype=jnp.int32
    )
    token_kernel = jax.jit(
        jax.vmap(
            lambda row: canonical_hubbard_child_merge_sign_jax(left, right, row)
        )
    )
    token_expected = np.asarray(
        [
            canonical_hubbard_child_merge_sign(left, right, tuple(map(int, row)))
            for row in np.asarray(tokens)
        ]
    )
    np.testing.assert_array_equal(np.asarray(token_kernel(tokens)), token_expected)
