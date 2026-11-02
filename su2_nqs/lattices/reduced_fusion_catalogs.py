"""Deterministic small catalogs for reduced-fusion-DAG experiments.

The builders in this module are geometry only.  They create binary diagrams
whose every merge is a disjoint exact cover and optionally close those
diagrams under a registered finite site group.  Four-site blocks contain two
pair-pair decompositions of the same support, so their route channels can mix
*before* a later root merge.  This is the smallest useful feature-level test;
it is not a terminal ensemble of complete trees.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

from .fusion_catalog import (
    DisjointMergeRoute,
    FusionSupport,
    ReducedFusionCatalog,
    group_close_merge_routes,
    route_image,
)
from .spatial_symmetry import (
    SitePermutation,
    compose_site_permutations,
    inverse_site_permutation,
    validate_site_permutation,
    validated_permutation_group,
)


def _validated_site_order(site_order: Sequence[int]) -> tuple[int, ...]:
    raw = tuple(site_order)
    if any(isinstance(site, bool) or not isinstance(site, Integral) for site in raw):
        raise TypeError("site_order entries must be integers")
    order = tuple(int(site) for site in raw)
    if len(order) < 8 or len(order) % 4:
        raise ValueError("site_order length must be a multiple of four and at least eight")
    if tuple(sorted(order)) != tuple(range(len(order))):
        raise ValueError("site_order must be a permutation of range(n_sites)")
    return order


def _route(
    parent: Sequence[int], left: Sequence[int], right: Sequence[int]
) -> DisjointMergeRoute:
    return DisjointMergeRoute(
        FusionSupport(tuple(parent)),
        FusionSupport(tuple(left)),
        FusionSupport(tuple(right)),
    )


def diamond_seed_routes(
    site_order: Sequence[int], *, resonating: bool = True
) -> tuple[DisjointMergeRoute, ...]:
    """Return a topologically complete set of four-site-block seed routes.

    For a block ``(a,b,c,d)``, the first diagram is
    ``(a,b) x (c,d)``.  With ``resonating=True`` the same four-site support
    also receives ``(a,c) x (b,d)``.  All four-site supports are subsequently
    fused in a balanced, deterministic hierarchy.  Consequently every route
    mixed in a block is consumed by at least one later disjoint merge.
    """

    order = _validated_site_order(site_order)
    routes: list[DisjointMergeRoute] = []
    blocks: list[FusionSupport] = []
    for start in range(0, len(order), 4):
        a, b, c, d = order[start : start + 4]
        ab = FusionSupport((a, b))
        cd = FusionSupport((c, d))
        block = FusionSupport((a, b, c, d))
        routes.extend(
            (
                _route(ab.sites, (a,), (b,)),
                _route(cd.sites, (c,), (d,)),
                _route(block.sites, ab.sites, cd.sites),
            )
        )
        if resonating:
            ac = FusionSupport((a, c))
            bd = FusionSupport((b, d))
            routes.extend(
                (
                    _route(ac.sites, (a,), (c,)),
                    _route(bd.sites, (b,), (d,)),
                    _route(block.sites, ac.sites, bd.sites),
                )
            )
        blocks.append(block)

    current = blocks
    while len(current) > 1:
        following: list[FusionSupport] = []
        for index in range(0, len(current), 2):
            if index + 1 == len(current):
                following.append(current[index])
                continue
            left, right = current[index : index + 2]
            parent = FusionSupport(left.sites + right.sites)
            routes.append(_route(parent.sites, left.sites, right.sites))
            following.append(parent)
        current = following
    if current != [FusionSupport(tuple(range(len(order))))]:
        raise RuntimeError("site order did not produce the all-site root")
    return tuple(routes)


def diamond_fusion_catalog(
    site_order: Sequence[int],
    *,
    group: Sequence[Sequence[int]] | None = None,
    resonating: bool = True,
) -> ReducedFusionCatalog:
    """Build and optionally finite-group-close a diamond fusion catalog."""

    order = _validated_site_order(site_order)
    n_sites = len(order)
    if group is None:
        checked_group: tuple[SitePermutation, ...] = (tuple(range(n_sites)),)
    else:
        checked_group = validated_permutation_group(group)
        if any(len(image) != n_sites for image in checked_group):
            raise ValueError("group and site_order disagree on n_sites")
    routes = group_close_merge_routes(
        diamond_seed_routes(order, resonating=resonating), checked_group
    )
    supports = {
        FusionSupport((site,)) for site in range(n_sites)
    }
    for merge in routes:
        supports.update((merge.parent, merge.left, merge.right))
    return ReducedFusionCatalog(
        n_sites,
        checked_group,
        tuple(supports),
        routes,
    )


def permute_fusion_catalog(
    catalog: ReducedFusionCatalog, image: Sequence[int]
) -> ReducedFusionCatalog:
    """Relabel a catalog and conjugate its group without changing capacity.

    This operation is useful for a strictly matched geometry-scrambled
    control: support sizes, incoming-route counts, parameter shapes, and the
    number of implicitly expanded diagrams are identical.  The conjugated
    group is metadata for the relabelled object; a physical-space projector
    must still use the actual Hamiltonian group in an energy benchmark.
    """

    checked = validate_site_permutation(image, n_sites=catalog.n_sites)
    inverse = inverse_site_permutation(checked)
    conjugated = tuple(
        compose_site_permutations(
            checked, compose_site_permutations(element, inverse)
        )
        for element in catalog.group
    )
    supports = tuple(
        FusionSupport(tuple(checked[site] for site in support.sites))
        for support in catalog.supports
    )
    routes = tuple(route_image(route, checked) for route in catalog.routes)
    return ReducedFusionCatalog(
        catalog.n_sites,
        validated_permutation_group(conjugated),
        supports,
        routes,
    )


def catalog_signature(catalog: ReducedFusionCatalog) -> dict[str, object]:
    """Return capacity-relevant counts for benchmark audit records."""

    incoming = {
        support: sum(route.parent == support for route in catalog.routes)
        for support in catalog.supports
    }
    by_size: dict[int, dict[str, int]] = {}
    for support in catalog.supports:
        size = len(support)
        row = by_size.setdefault(size, {"supports": 0, "routes": 0})
        row["supports"] += 1
        row["routes"] += incoming[support]
    expanded = 1
    for support in catalog.supports:
        if len(support) > 1:
            expanded *= max(incoming[support], 1)
    return {
        "n_sites": catalog.n_sites,
        "group_order": len(catalog.group),
        "support_count": len(catalog.supports),
        "route_count": len(catalog.routes),
        "counts_by_support_size": {
            str(size): by_size[size] for size in sorted(by_size)
        },
        # This product is a simple upper bound.  Shared dependencies can make
        # the exact symbolic expansion smaller, so never label it an equality.
        "expanded_diagram_upper_bound": expanded,
    }


__all__ = [
    "catalog_signature",
    "diamond_fusion_catalog",
    "diamond_seed_routes",
    "permute_fusion_catalog",
]
