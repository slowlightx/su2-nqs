"""Coset-hierarchical site orders for translation-covariant fusion trees."""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

from ..su2.fusion_tree import FusionTree


def _balanced_nested(values: Sequence[object]) -> object:
    if not values:
        raise ValueError("cannot build a binary tree from an empty sequence")
    if len(values) == 1:
        return values[0]
    middle = len(values) // 2
    return (
        _balanced_nested(values[:middle]),
        _balanced_nested(values[middle:]),
    )


def cyclic_bit_reversed_order(length: int) -> tuple[int, ...]:
    """Order ``Z_length`` by a binary subgroup chain.

    For ``length=4`` this returns ``(0,2,1,3)``.  A cyclic translation then
    acts as an automorphism of the complete balanced binary tree on this order.
    """

    if (
        isinstance(length, bool)
        or not isinstance(length, Integral)
        or length < 1
        or int(length) & (int(length) - 1)
    ):
        raise ValueError("length must be a positive power of two")
    length = int(length)
    bits = max(length.bit_length() - 1, 0)

    def reverse_bits(value: int) -> int:
        result = 0
        for bit in range(bits):
            result |= ((value >> bit) & 1) << (bits - bit - 1)
        return result

    return tuple(sorted(range(length), key=reverse_bits))


def square_translation_coset_order(
    linear_size: int, *, primary_axis: str = "x"
) -> tuple[int, ...]:
    """Return a subgroup-chain order on a square power-of-two torus."""

    coordinate_order = cyclic_bit_reversed_order(linear_size)
    if primary_axis == "x":
        return tuple(
            y * int(linear_size) + x
            for x in coordinate_order
            for y in coordinate_order
        )
    if primary_axis == "y":
        return tuple(
            y * int(linear_size) + x
            for y in coordinate_order
            for x in coordinate_order
        )
    raise ValueError("primary_axis must be 'x' or 'y'")


def translation_quotient_tree(
    linear_size: int, quotient_size: int
) -> FusionTree:
    """Build a binary tree for ``Z_L^2 = Z_Q^2 x Z_{L/Q}^2`` frames.

    Each of the ``Q^2`` consecutive top-level blocks contains a power-of-two
    translation subgroup of size ``(L/Q)^2``.  The quotient itself need not be
    binary.  For ``L=6, Q=3`` this gives nine four-site blocks: translations
    by three lattice spacings are exact tree automorphisms, while one-site
    translations move between nine frames.

    The tree uses abstract leaves in order.  Attach it to physical sites with
    :func:`square_translation_quotient_frame_order`.
    """

    if (
        isinstance(linear_size, bool)
        or not isinstance(linear_size, Integral)
        or int(linear_size) < 1
    ):
        raise ValueError("linear_size must be a positive integer")
    if (
        isinstance(quotient_size, bool)
        or not isinstance(quotient_size, Integral)
        or int(quotient_size) < 1
    ):
        raise ValueError("quotient_size must be a positive integer")
    length = int(linear_size)
    quotient = int(quotient_size)
    if length % quotient:
        raise ValueError("quotient_size must divide linear_size")
    subgroup = length // quotient
    if subgroup & (subgroup - 1):
        raise ValueError("linear_size / quotient_size must be a power of two")

    block_size = subgroup * subgroup
    blocks = []
    for block in range(quotient * quotient):
        start = block * block_size
        blocks.append(_balanced_nested(tuple(range(start, start + block_size))))
    return FusionTree.from_nested(_balanced_nested(tuple(blocks)))  # type: ignore[arg-type]


def square_translation_quotient_frame_order(
    linear_size: int,
    quotient_size: int,
    *,
    frame: tuple[int, int] = (0, 0),
    primary_axis: str = "x",
) -> tuple[int, ...]:
    """Map a quotient-tree frame to row-major sites on a periodic torus.

    Coordinates are grouped by their residue in ``Z_Q^2`` and ordered within
    each block by a binary subgroup chain.  ``frame`` is reduced modulo
    ``quotient_size``.  The construction only concerns Bravais-cell
    translations, so it applies equally to square and triangular lattices
    represented on an ``L x L`` coordinate torus.
    """

    tree = translation_quotient_tree(linear_size, quotient_size)
    length = int(linear_size)
    quotient = int(quotient_size)
    subgroup = length // quotient
    if (
        not isinstance(frame, tuple)
        or len(frame) != 2
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in frame)
    ):
        raise ValueError("frame must contain two integer coordinates")
    if primary_axis not in {"x", "y"}:
        raise ValueError("primary_axis must be 'x' or 'y'")
    frame_x, frame_y = (int(frame[0]) % quotient, int(frame[1]) % quotient)
    subgroup_order = cyclic_bit_reversed_order(subgroup)
    quotient_pairs = (
        ((x, y) for x in range(quotient) for y in range(quotient))
        if primary_axis == "x"
        else ((x, y) for y in range(quotient) for x in range(quotient))
    )
    inner_pairs = (
        ((x, y) for x in subgroup_order for y in subgroup_order)
        if primary_axis == "x"
        else ((x, y) for y in subgroup_order for x in subgroup_order)
    )
    # Materialize because the inner order is reused for every quotient block.
    inner = tuple(inner_pairs)
    order = tuple(
        ((qy + frame_y + quotient * hy) % length) * length
        + ((qx + frame_x + quotient * hx) % length)
        for qx, qy in quotient_pairs
        for hx, hy in inner
    )
    if len(order) != tree.n_leaves or tuple(sorted(order)) != tuple(
        range(length * length)
    ):
        raise RuntimeError("quotient frame did not generate every site exactly once")
    return order


__all__ = [
    "cyclic_bit_reversed_order",
    "square_translation_coset_order",
    "square_translation_quotient_frame_order",
    "translation_quotient_tree",
]
