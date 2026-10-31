"""Finite-group-closed support catalogs for prospective reduced fusion DAGs.

This module contains geometry and validation only.  It deliberately does not
evaluate amplitudes: a catalog says which *disjoint* physical supports may be
fused and how a registered finite site group permutes those routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Sequence

from .spatial_symmetry import (
    SitePermutation,
    validate_site_permutation,
    validated_permutation_group,
)


@dataclass(frozen=True, order=True)
class FusionSupport:
    """A nonempty, canonically ordered set of physical site indices."""

    sites: tuple[int, ...]

    def __post_init__(self) -> None:
        raw = tuple(self.sites)
        if not raw:
            raise ValueError("a fusion support must be nonempty")
        if any(isinstance(site, bool) or not isinstance(site, Integral) for site in raw):
            raise TypeError("fusion-support entries must be integers")
        checked = tuple(int(site) for site in raw)
        if any(site < 0 for site in checked):
            raise ValueError("fusion-support entries must be nonnegative")
        if len(set(checked)) != len(checked):
            raise ValueError("a fusion support cannot contain repeated sites")
        object.__setattr__(self, "sites", tuple(sorted(checked)))

    def __len__(self) -> int:
        return len(self.sites)


def _as_support(value: FusionSupport | Sequence[int]) -> FusionSupport:
    return value if isinstance(value, FusionSupport) else FusionSupport(tuple(value))


@dataclass(frozen=True, order=True)
class DisjointMergeRoute:
    """An oriented merge ``left tensor right -> parent``.

    Child orientation is retained because SU(2) coupling conventions and
    fermionic reordering signs are orientation sensitive.
    """

    parent: FusionSupport
    left: FusionSupport
    right: FusionSupport

    def __post_init__(self) -> None:
        parent = _as_support(self.parent)
        left = _as_support(self.left)
        right = _as_support(self.right)
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        left_sites, right_sites = set(left.sites), set(right.sites)
        if left_sites & right_sites:
            raise ValueError("merge-route children must be disjoint")
        if left_sites | right_sites != set(parent.sites):
            raise ValueError("merge-route children must cover the parent exactly")


def support_image(
    support: FusionSupport | Sequence[int], image: Sequence[int]
) -> FusionSupport:
    """Return the canonical image of one support under an active site map."""

    checked_support = _as_support(support)
    checked_image = validate_site_permutation(image)
    if checked_support.sites[-1] >= len(checked_image):
        raise ValueError("fusion support lies outside the site permutation")
    return FusionSupport(tuple(checked_image[site] for site in checked_support.sites))


def route_image(
    route: DisjointMergeRoute, image: Sequence[int]
) -> DisjointMergeRoute:
    """Return the oriented image of a disjoint merge route."""

    return DisjointMergeRoute(
        support_image(route.parent, image),
        support_image(route.left, image),
        support_image(route.right, image),
    )


def group_close_supports(
    supports: Iterable[FusionSupport | Sequence[int]],
    group: Iterable[Sequence[int]],
) -> tuple[FusionSupport, ...]:
    """Close support seeds under a validated finite permutation group."""

    checked_group = validated_permutation_group(group)
    seeds = tuple(_as_support(support) for support in supports)
    if not seeds:
        raise ValueError("at least one support seed is required")
    return tuple(
        sorted(
            {
                support_image(support, image)
                for support in seeds
                for image in checked_group
            },
            key=lambda support: (len(support), support.sites),
        )
    )


def group_close_merge_routes(
    routes: Iterable[DisjointMergeRoute],
    group: Iterable[Sequence[int]],
) -> tuple[DisjointMergeRoute, ...]:
    """Close oriented disjoint-merge seeds under a finite site group."""

    checked_group = validated_permutation_group(group)
    seeds = tuple(routes)
    if not seeds:
        raise ValueError("at least one merge-route seed is required")
    return tuple(
        sorted(
            {
                route_image(route, image)
                for route in seeds
                for image in checked_group
            },
            key=_route_sort_key,
        )
    )


def _route_sort_key(
    route: DisjointMergeRoute,
) -> tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        len(route.parent),
        route.parent.sites,
        route.left.sites,
        route.right.sites,
    )


@dataclass(frozen=True)
class ReducedFusionCatalog:
    """A complete, deterministic, finite-group-closed fusion-DAG catalog.

    This is a Phase-0 contract, not an amplitude model.  Every physical leaf
    and the all-site root must be present.  Every route must reference listed
    supports, and both supports and oriented routes must be closed under the
    registered finite group.
    """

    n_sites: int
    group: tuple[SitePermutation, ...]
    supports: tuple[FusionSupport, ...]
    routes: tuple[DisjointMergeRoute, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_sites, bool)
            or not isinstance(self.n_sites, Integral)
            or self.n_sites < 2
        ):
            raise ValueError("n_sites must be an integer of at least two")
        n_sites = int(self.n_sites)
        group = validated_permutation_group(self.group)
        if any(len(image) != n_sites for image in group):
            raise ValueError("catalog group and n_sites disagree")
        supports = tuple(_as_support(support) for support in self.supports)
        routes = tuple(self.routes)
        if len(set(supports)) != len(supports):
            raise ValueError("catalog supports must be duplicate free")
        if len(set(routes)) != len(routes):
            raise ValueError("catalog routes must be duplicate free")
        if any(support.sites[-1] >= n_sites for support in supports):
            raise ValueError("catalog support contains a site outside n_sites")
        support_set = set(supports)
        required = {
            FusionSupport((site,)) for site in range(n_sites)
        } | {FusionSupport(tuple(range(n_sites)))}
        if not required <= support_set:
            raise ValueError("catalog must contain every leaf and the all-site root")
        for route in routes:
            if (
                route.parent not in support_set
                or route.left not in support_set
                or route.right not in support_set
            ):
                raise ValueError("every merge-route support must be listed in the catalog")
        route_parents = {route.parent for route in routes}
        if any(len(support) > 1 and support not in route_parents for support in supports):
            raise ValueError("every composite support must have an incoming merge route")
        root = FusionSupport(tuple(range(n_sites)))
        root_dependencies = {root}
        while True:
            previous_count = len(root_dependencies)
            for route in routes:
                if route.parent in root_dependencies:
                    root_dependencies.update((route.left, route.right))
            if len(root_dependencies) == previous_count:
                break
        if root_dependencies != support_set:
            dead = tuple(
                sorted(
                    support_set - root_dependencies,
                    key=lambda support: (len(support), support.sites),
                )
            )
            raise ValueError(
                "every catalog support must feed the all-site root; "
                f"unreachable supports: {dead!r}"
            )
        for support in supports:
            for image in group:
                if support_image(support, image) not in support_set:
                    raise ValueError("catalog supports are not closed under the finite group")
        route_set = set(routes)
        for route in routes:
            for image in group:
                if route_image(route, image) not in route_set:
                    raise ValueError("catalog merge routes are not closed under the finite group")
        object.__setattr__(self, "n_sites", n_sites)
        object.__setattr__(self, "group", group)
        object.__setattr__(
            self,
            "supports",
            tuple(sorted(supports, key=lambda support: (len(support), support.sites))),
        )
        object.__setattr__(self, "routes", tuple(sorted(routes, key=_route_sort_key)))


def support_orbit_indices(catalog: ReducedFusionCatalog) -> tuple[int, ...]:
    """Return deterministic finite-group orbit labels for catalog supports.

    Labels are consecutive integers assigned by the first support in the
    catalog's canonical ordering.  The result aligns with ``catalog.supports``.
    """

    if not isinstance(catalog, ReducedFusionCatalog):
        raise TypeError("catalog must be a ReducedFusionCatalog")
    positions = {support: index for index, support in enumerate(catalog.supports)}
    labels = [-1] * len(catalog.supports)
    next_label = 0
    for index, support in enumerate(catalog.supports):
        if labels[index] >= 0:
            continue
        orbit = {support_image(support, image) for image in catalog.group}
        for member in orbit:
            labels[positions[member]] = next_label
        next_label += 1
    return tuple(labels)


def route_orbit_indices(catalog: ReducedFusionCatalog) -> tuple[int, ...]:
    """Return deterministic finite-group orbit labels for oriented routes.

    Child orientation is retained.  Labels align with ``catalog.routes`` and
    therefore may be used as static parameter-sharing indices.
    """

    if not isinstance(catalog, ReducedFusionCatalog):
        raise TypeError("catalog must be a ReducedFusionCatalog")
    positions = {route: index for index, route in enumerate(catalog.routes)}
    labels = [-1] * len(catalog.routes)
    next_label = 0
    for index, route in enumerate(catalog.routes):
        if labels[index] >= 0:
            continue
        orbit = {route_image(route, image) for image in catalog.group}
        for member in orbit:
            labels[positions[member]] = next_label
        next_label += 1
    return tuple(labels)


__all__ = [
    "DisjointMergeRoute",
    "FusionSupport",
    "ReducedFusionCatalog",
    "group_close_merge_routes",
    "group_close_supports",
    "route_orbit_indices",
    "route_image",
    "support_orbit_indices",
    "support_image",
]
