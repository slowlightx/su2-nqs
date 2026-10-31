"""Finite triangular tori defined by column Hermite-normal-form cells.

Coordinates use the repository's 120-degree primitive basis.  A supercell
``((a, b), (0, d))`` stores the translations ``(a, 0)`` and ``(b, d)`` as
columns, so sites are the quotient ``Z^2 / A Z^2``.  Site permutations are
active maps: ``image[old_site] == new_site``.
"""

from __future__ import annotations

from numbers import Integral
from typing import Iterable


IntVector = tuple[int, int]
IntMatrix = tuple[tuple[int, int], tuple[int, int]]
SitePermutation = tuple[int, ...]

IDENTITY: IntMatrix = ((1, 0), (0, 1))
C6: IntMatrix = ((1, -1), (1, 0))
MX: IntMatrix = ((1, -1), (0, -1))

J1_POSITIVE: tuple[IntVector, ...] = ((1, 0), (0, 1), (1, 1))
J2_POSITIVE: tuple[IntVector, ...] = ((1, -1), (1, 2), (2, 1))


def matrix_product(left: IntMatrix, right: IntMatrix) -> IntMatrix:
    """Multiply two two-by-two integer matrices."""

    return (
        (
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ),
        (
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ),
    )


def matrix_vector(matrix: IntMatrix, vector: IntVector) -> IntVector:
    """Apply a two-by-two integer matrix to a column vector."""

    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    )


def matrix_power(matrix: IntMatrix, exponent: int) -> IntMatrix:
    """Return a non-negative integer power of an integer matrix."""

    if isinstance(exponent, bool) or not isinstance(exponent, Integral):
        raise TypeError("exponent must be an integer")
    if exponent < 0:
        raise ValueError("exponent must be non-negative")
    result = IDENTITY
    factor = matrix
    value = int(exponent)
    while value:
        if value & 1:
            result = matrix_product(factor, result)
        factor = matrix_product(factor, factor)
        value >>= 1
    return result


POINT_GROUP: tuple[tuple[str, IntMatrix], ...] = tuple(
    [(f"C6^{power}", matrix_power(C6, power)) for power in range(6)]
    + [
        (f"Mx*C6^{power}", matrix_product(MX, matrix_power(C6, power)))
        for power in range(6)
    ]
)


def validate_hnf(supercell: IntMatrix) -> IntMatrix:
    """Validate and return the canonical column HNF ``((a,b),(0,d))``."""

    raw = tuple(tuple(row) for row in supercell)
    if len(raw) != 2 or any(len(row) != 2 for row in raw):
        raise ValueError("supercell must be a two-by-two matrix")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for row in raw
        for value in row
    ):
        raise TypeError("supercell entries must be integers")
    matrix: IntMatrix = (
        (int(raw[0][0]), int(raw[0][1])),
        (int(raw[1][0]), int(raw[1][1])),
    )
    horizontal, tilt = matrix[0]
    lower_left, vertical = matrix[1]
    if lower_left != 0 or horizontal <= 0 or vertical <= 0:
        raise ValueError("supercell must have HNF form ((a,b),(0,d)) with a,d>0")
    if not 0 <= tilt < horizontal:
        raise ValueError("HNF tilt must satisfy 0 <= b < a")
    return matrix


def hnf_volume(supercell: IntMatrix) -> int:
    """Return the number of sites in an HNF torus."""

    matrix = validate_hnf(supercell)
    return matrix[0][0] * matrix[1][1]


def reduce_hnf(vector: IntVector, supercell: IntMatrix) -> IntVector:
    """Reduce an integer coordinate to the deterministic quotient site."""

    matrix = validate_hnf(supercell)
    horizontal, tilt = matrix[0]
    vertical = matrix[1][1]
    quotient_y = int(vector[1]) // vertical
    reduced_y = int(vector[1]) - quotient_y * vertical
    reduced_x = (int(vector[0]) - quotient_y * tilt) % horizontal
    return reduced_x, reduced_y


def hnf_sites(supercell: IntMatrix) -> tuple[IntVector, ...]:
    """Enumerate quotient representatives in deterministic row-major order."""

    matrix = validate_hnf(supercell)
    return tuple(
        (x, y)
        for y in range(matrix[1][1])
        for x in range(matrix[0][0])
    )


def point_operation_preserved(supercell: IntMatrix, operation: IntMatrix) -> bool:
    """Return whether an integer point operation preserves the HNF lattice."""

    matrix = validate_hnf(supercell)
    transformed = matrix_product(operation, matrix)
    determinant = hnf_volume(matrix)
    adjugate: IntMatrix = (
        (matrix[1][1], -matrix[0][1]),
        (0, matrix[0][0]),
    )
    numerators = matrix_product(adjugate, transformed)
    return all(value % determinant == 0 for row in numerators for value in row)


def hnf_site_permutation(
    supercell: IntMatrix,
    *,
    translation: IntVector = (0, 0),
    operation: IntMatrix = IDENTITY,
) -> SitePermutation:
    """Return the active affine site map ``r -> operation*r+translation``."""

    matrix = validate_hnf(supercell)
    if not point_operation_preserved(matrix, operation):
        raise ValueError("point operation does not preserve the supercell lattice")
    sites = hnf_sites(matrix)
    positions = {site: index for index, site in enumerate(sites)}
    image = tuple(
        positions[
            reduce_hnf(
                (
                    matrix_vector(operation, site)[0] + int(translation[0]),
                    matrix_vector(operation, site)[1] + int(translation[1]),
                ),
                matrix,
            )
        ]
        for site in sites
    )
    if tuple(sorted(image)) != tuple(range(len(sites))):
        raise RuntimeError("affine HNF action is not a site permutation")
    return image


def hnf_translation_permutations(
    supercell: IntMatrix,
) -> tuple[tuple[IntVector, SitePermutation], ...]:
    """Return one active site permutation for every quotient translation."""

    matrix = validate_hnf(supercell)
    return tuple(
        (translation, hnf_site_permutation(matrix, translation=translation))
        for translation in hnf_sites(matrix)
    )


def hnf_point_group_permutations(
    supercell: IntMatrix,
) -> tuple[tuple[str, IntMatrix, SitePermutation], ...]:
    """Return all triangular D6 operations retained by the finite torus."""

    matrix = validate_hnf(supercell)
    return tuple(
        (name, operation, hnf_site_permutation(matrix, operation=operation))
        for name, operation in POINT_GROUP
        if point_operation_preserved(matrix, operation)
    )


def hnf_character_numerators(supercell: IntMatrix) -> tuple[IntVector, ...]:
    """Enumerate all quotient characters on a common denominator ``N`` grid."""

    matrix = validate_hnf(supercell)
    volume = hnf_volume(matrix)
    return tuple(
        (first, second)
        for first in range(volume)
        for second in range(volume)
        if all(
            (
                first * matrix[0][column]
                + second * matrix[1][column]
            )
            % volume
            == 0
            for column in range(2)
        )
    )


def triangular_hnf_j1j2_bonds(
    supercell: IntMatrix,
    *,
    j1: float = 1.0,
    j2: float = 0.0,
) -> tuple[tuple[int, int, float], ...]:
    """Return periodic triangular J1-J2 bonds on a general HNF torus.

    Distinct periodic images connecting the same site pair are combined.  A
    production geometry gate should separately reject such aliases.
    """

    matrix = validate_hnf(supercell)
    sites = hnf_sites(matrix)
    positions = {site: index for index, site in enumerate(sites)}
    weights: dict[tuple[int, int], float] = {}

    def add_shell(displacements: Iterable[IntVector], coupling: float) -> None:
        if coupling == 0.0:
            return
        for first, site in enumerate(sites):
            for displacement in displacements:
                target = reduce_hnf(
                    (site[0] + displacement[0], site[1] + displacement[1]),
                    matrix,
                )
                second = positions[target]
                if first == second:
                    continue
                edge = tuple(sorted((first, second)))
                weights[edge] = weights.get(edge, 0.0) + float(coupling)

    add_shell(J1_POSITIVE, j1)
    add_shell(J2_POSITIVE, j2)
    return tuple(
        (first, second, coupling)
        for (first, second), coupling in sorted(weights.items())
    )
