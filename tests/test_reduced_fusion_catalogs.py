from __future__ import annotations

from collections import Counter

import pytest

from su2_nqs.lattices.reduced_fusion_catalogs import (
    catalog_signature,
    diamond_fusion_catalog,
    diamond_seed_routes,
    permute_fusion_catalog,
)
from su2_nqs.lattices.geometry_tree import morton_order
from su2_nqs.lattices.spatial_symmetry import rectangular_translation_group


def test_diamond_seed_has_true_intermediate_route_mixing() -> None:
    routes = diamond_seed_routes(tuple(range(8)), resonating=True)
    parents = Counter(route.parent for route in routes)
    mixed = [support for support, count in parents.items() if count > 1]
    assert {support.sites for support in mixed} == {
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    }
    root = next(route.parent for route in routes if len(route.parent) == 8)
    assert parents[root] == 1
    assert all(len(support) < 8 for support in mixed)


def test_diamond_catalog_closes_under_physical_translations() -> None:
    group = rectangular_translation_group(2, 4)
    catalog = diamond_fusion_catalog(tuple(range(8)), group=group)
    assert catalog.group == group
    assert len(catalog.routes) > len(diamond_seed_routes(tuple(range(8))))


def test_permuted_catalog_is_exactly_capacity_matched() -> None:
    catalog = diamond_fusion_catalog(tuple(range(8)))
    scrambled = permute_fusion_catalog(catalog, (0, 3, 6, 1, 4, 7, 2, 5))
    assert catalog_signature(scrambled) == catalog_signature(catalog)
    assert scrambled != catalog


def test_diamond_builder_rejects_malformed_site_orders() -> None:
    with pytest.raises(ValueError, match="multiple of four"):
        diamond_fusion_catalog(tuple(range(6)))
    with pytest.raises(ValueError, match="permutation"):
        diamond_fusion_catalog((0, 1, 2, 3, 4, 5, 6, 6))


def test_n16_morton_diamond_remains_the_frozen_two_pairing_control() -> None:
    catalog = diamond_fusion_catalog(morton_order(4, 4), resonating=True)
    assert catalog_signature(catalog) == {
        "n_sites": 16,
        "group_order": 1,
        "support_count": 39,
        "route_count": 27,
        "counts_by_support_size": {
            "1": {"supports": 16, "routes": 0},
            "2": {"supports": 16, "routes": 16},
            "4": {"supports": 4, "routes": 8},
            "8": {"supports": 2, "routes": 2},
            "16": {"supports": 1, "routes": 1},
        },
        "expanded_diagram_upper_bound": 16,
    }
