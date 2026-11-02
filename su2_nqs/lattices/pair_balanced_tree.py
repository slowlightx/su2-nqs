"""Fusion trees that preserve an exact adjacent-singlet product channel."""

from __future__ import annotations

from numbers import Integral

from ..su2.fusion_tree import FusionTree


def pair_balanced_fusion_tree(n_sites: int) -> FusionTree:
    """Return a balanced tree over adjacent two-site blocks.

    The ordinary leaf-balanced split can cut through a pair when ``N/2`` is
    odd (notably ``N=36``).  This topology first groups ``(0,1)``, ``(2,3)``,
    and so on, then balances the list of pairs.  Consequently the exact
    product of adjacent singlets is one explicit all-``q=0`` internal channel.
    """

    if (
        isinstance(n_sites, bool)
        or not isinstance(n_sites, Integral)
        or int(n_sites) < 2
        or int(n_sites) % 2
    ):
        raise ValueError("n_sites must be an even integer of at least two")
    pairs: list[object] = [
        (site, site + 1) for site in range(0, int(n_sites), 2)
    ]

    def balance(nodes: list[object]) -> object:
        if len(nodes) == 1:
            return nodes[0]
        middle = len(nodes) // 2
        return (balance(nodes[:middle]), balance(nodes[middle:]))

    return FusionTree.from_nested(balance(pairs))


__all__ = ["pair_balanced_fusion_tree"]
