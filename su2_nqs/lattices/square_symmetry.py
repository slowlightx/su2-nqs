"""Deterministic site maps and momentum labels for square tori."""

from __future__ import annotations

from numbers import Integral

SitePermutation = tuple[int, ...]


def square_site_generators(linear_size: int) -> dict[str, SitePermutation]:
    """Return active translations, rotation and reflection in row-major order."""

    if (
        isinstance(linear_size, bool)
        or not isinstance(linear_size, Integral)
        or linear_size < 2
    ):
        raise ValueError("linear_size must be an integer at least two")
    length = int(linear_size)

    def image(transform) -> SitePermutation:
        result = []
        for y in range(length):
            for x in range(length):
                new_x, new_y = transform(x, y)
                result.append((new_y % length) * length + (new_x % length))
        permutation = tuple(result)
        if tuple(sorted(permutation)) != tuple(range(length * length)):
            raise RuntimeError("square generator is not a site permutation")
        return permutation

    return {
        "identity": tuple(range(length * length)),
        "tx": image(lambda x, y: (x + 1, y)),
        "ty": image(lambda x, y: (x, y + 1)),
        "c4": image(lambda x, y: (-y, x)),
        "mx": image(lambda x, y: (-x, y)),
    }


def square_momentum_label(
    momentum: tuple[int, int], linear_size: int
) -> str:
    """Return a finite-torus momentum name without conflating generic points."""

    if linear_size < 2:
        raise ValueError("linear_size must be at least two")
    key = (int(momentum[0]) % linear_size, int(momentum[1]) % linear_size)
    if key == (0, 0):
        return "Gamma"
    if linear_size % 2 == 0:
        half = linear_size // 2
        if key == (half, half):
            return "M"
        if key == (half, 0):
            return "X"
        if key == (0, half):
            return "Y"
    return f"k({key[0]},{key[1]})[2pi/L]"
