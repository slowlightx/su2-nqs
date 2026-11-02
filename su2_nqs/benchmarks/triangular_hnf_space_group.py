"""Exact finite-group projectors for triangular HNF tori.

These routines are intentionally dense and intended only for exact small-
system gates.  Production-sized states should apply the same group averages
matrix-free through local recoupling rows.
"""

from __future__ import annotations

from fractions import Fraction
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from ..lattices.triangular_hnf import (
    POINT_GROUP,
    IntMatrix,
    IntVector,
    hnf_character_numerators,
    hnf_point_group_permutations,
    hnf_translation_permutations,
    hnf_volume,
    triangular_hnf_j1j2_bonds,
    validate_hnf,
)
from ..operators.permutation import site_permutation_fusion_matrix
from ..su2.fusion_path import count_paths


D6_IRREP_DIMENSIONS: dict[str, int] = {
    "A1": 1,
    "A2": 1,
    "B1": 1,
    "B2": 1,
    "E1": 2,
    "E2": 2,
}

FractionMomentum = tuple[Fraction, Fraction]


def permutation_cycle_lengths(image: Sequence[int]) -> tuple[int, ...]:
    """Return the sorted cycle lengths of one exact site permutation."""

    values = tuple(int(value) for value in image)
    if sorted(values) != list(range(len(values))):
        raise ValueError("image must be a permutation")
    visited: set[int] = set()
    lengths: list[int] = []
    for start in range(len(values)):
        if start in visited:
            continue
        site = start
        length = 0
        while site not in visited:
            visited.add(site)
            length += 1
            site = values[site]
        lengths.append(length)
    return tuple(sorted(lengths))


def fixed_weight_permutation_trace(image: Sequence[int], n_up: int) -> int:
    """Trace of a site permutation in a fixed computational-Sz sector."""

    lengths = permutation_cycle_lengths(image)
    n_sites = sum(lengths)
    if isinstance(n_up, bool) or not isinstance(n_up, Integral):
        raise TypeError("n_up must be an integer")
    if not 0 <= int(n_up) <= n_sites:
        return 0
    coefficients = [1] + [0] * n_sites
    for length in lengths:
        for weight in range(n_sites, length - 1, -1):
            coefficients[weight] += coefficients[weight - length]
    return int(coefficients[int(n_up)])


def spin_multiplicity_permutation_character(
    image: Sequence[int], target_q: int
) -> int:
    """Character of a permutation on one doubled-total-spin multiplicity."""

    values = tuple(image)
    n_sites = len(values)
    if (
        isinstance(target_q, bool)
        or not isinstance(target_q, Integral)
        or int(target_q) < 0
        or int(target_q) > n_sites
        or (n_sites + int(target_q)) % 2
    ):
        raise ValueError("invalid doubled target spin")
    n_up = (n_sites + int(target_q)) // 2
    return fixed_weight_permutation_trace(
        values, n_up
    ) - fixed_weight_permutation_trace(values, n_up + 1)


def exact_gamma_a1_spin_multiplicity_dimension(
    supercell: IntMatrix, *, target_q: int = 0
) -> dict[str, object]:
    """Return the exact ``Gamma.A1`` multiplicity dimension by characters.

    This performs only a finite permutation-cycle character average over the
    translation-by-D6 space group.  It never constructs a Hilbert-space vector
    or projector, so N36 and N48 resource dimensions remain inexpensive to
    audit exactly.
    """

    matrix = validate_hnf(supercell)
    translations = hnf_translation_permutations(matrix)
    point_group = hnf_point_group_permutations(matrix)
    if len(point_group) != 12:
        raise ValueError("Gamma.A1 dimension requires a full-D6 HNF torus")
    characters_by_point_operation = {name: 0 for name, _, _ in point_group}
    character_sum = 0
    for _, translation_image in translations:
        for name, _, point_image in point_group:
            combined = tuple(
                point_image[translation_image[index]]
                for index in range(len(translation_image))
            )
            character = spin_multiplicity_permutation_character(
                combined, int(target_q)
            )
            characters_by_point_operation[name] += character
            character_sum += character
    group_order = len(translations) * len(point_group)
    if character_sum % group_order:
        raise RuntimeError("Gamma.A1 character average is not integral")
    return {
        "supercell_hnf": [list(matrix[0]), list(matrix[1])],
        "n_sites": hnf_volume(matrix),
        "target_q": int(target_q),
        "full_spin_multiplicity_dimension": count_paths(
            hnf_volume(matrix), int(target_q)
        ),
        "space_group_order": group_order,
        "character_sum": character_sum,
        "characters_summed_over_translations": characters_by_point_operation,
        "dimension": character_sum // group_order,
    }


def parse_fraction_momentum(raw: Sequence[int]) -> FractionMomentum:
    """Parse ``(n1,d1,n2,d2)`` into a reciprocal-basis momentum.

    Components are reduced modulo one.  This representation is deliberately
    exact: floating-point coordinates must not decide whether a momentum is a
    character of a finite HNF quotient.
    """

    if len(raw) != 4:
        raise ValueError("a momentum requires (n1,d1,n2,d2)")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
        raise TypeError("fraction-momentum entries must be integers")
    first_numerator, first_denominator, second_numerator, second_denominator = (
        int(value) for value in raw
    )
    if first_denominator <= 0 or second_denominator <= 0:
        raise ValueError("fraction-momentum denominators must be positive")
    return (
        Fraction(first_numerator, first_denominator) % 1,
        Fraction(second_numerator, second_denominator) % 1,
    )


def fraction_momentum_json(momentum: FractionMomentum) -> list[int]:
    """Serialize an exact reciprocal momentum as four integers."""

    return [
        momentum[0].numerator,
        momentum[0].denominator,
        momentum[1].numerator,
        momentum[1].denominator,
    ]


def reciprocal_momentum_image(
    operation: IntMatrix, momentum: FractionMomentum
) -> FractionMomentum:
    """Apply the contragredient action ``k -> operation**(-T) k``."""

    (first_row, second_row) = operation
    a, b = int(first_row[0]), int(first_row[1])
    c, d = int(second_row[0]), int(second_row[1])
    determinant = a * d - b * c
    if determinant not in (-1, 1):
        raise ValueError("a reciprocal point operation must be unimodular")
    first, second = momentum
    return (
        ((d * first - c * second) / determinant) % 1,
        ((-b * first + a * second) / determinant) % 1,
    )


def hnf_fraction_momentum_allowed(
    supercell: IntMatrix, momentum: FractionMomentum
) -> bool:
    """Return whether ``exp(2*pi*i*k.r)`` descends to ``Z^2/A Z^2``."""

    matrix = validate_hnf(supercell)
    first, second = momentum
    first_wrap = matrix[0][0] * first + matrix[1][0] * second
    second_wrap = matrix[0][1] * first + matrix[1][1] * second
    return first_wrap.denominator == 1 and second_wrap.denominator == 1


def hnf_fraction_character_numerator(
    supercell: IntMatrix, momentum: FractionMomentum
) -> IntVector:
    """Return the existing ``numerator/N`` label for an exact momentum."""

    matrix = validate_hnf(supercell)
    if not hnf_fraction_momentum_allowed(matrix, momentum):
        raise ValueError("momentum is not an allowed character of the HNF torus")
    volume = hnf_volume(matrix)
    scaled = (volume * momentum[0], volume * momentum[1])
    if scaled[0].denominator != 1 or scaled[1].denominator != 1:
        raise ValueError("allowed momentum cannot be represented in numerator/N form")
    numerator = (int(scaled[0]) % volume, int(scaled[1]) % volume)
    if numerator not in hnf_character_numerators(matrix):
        raise RuntimeError("fractional and numerator/N character tests disagree")
    return numerator


def reciprocal_little_group(
    supercell: IntMatrix, momentum: FractionMomentum
) -> tuple[str, ...]:
    """Return the preserved point operations fixing one momentum modulo one."""

    matrix = validate_hnf(supercell)
    preserved = hnf_point_group_permutations(matrix)
    return tuple(
        name
        for name, operation, _ in preserved
        if reciprocal_momentum_image(operation, momentum) == momentum
    )


def reciprocal_star_audit(
    supercell: IntMatrix,
    representative: FractionMomentum,
    expected_star: Sequence[FractionMomentum],
) -> dict[str, object]:
    """Audit one exact momentum star without constructing a spin Hilbert space."""

    matrix = validate_hnf(supercell)
    point_group = hnf_point_group_permutations(matrix)
    expected = tuple(expected_star)
    expected_set = set(expected)
    orbit: list[FractionMomentum] = []
    for _, operation, _ in point_group:
        image = reciprocal_momentum_image(operation, representative)
        if image not in orbit:
            orbit.append(image)
    sewing: dict[str, list[int]] = {}
    for name, operation, _ in point_group:
        images = [reciprocal_momentum_image(operation, momentum) for momentum in expected]
        sewing[name] = [expected.index(image) if image in expected_set else -1 for image in images]
    allowed = [hnf_fraction_momentum_allowed(matrix, momentum) for momentum in expected]
    numerators = [
        list(hnf_fraction_character_numerator(matrix, momentum))
        if is_allowed
        else None
        for momentum, is_allowed in zip(expected, allowed, strict=True)
    ]
    return {
        "supercell_hnf": [list(matrix[0]), list(matrix[1])],
        "n_sites": hnf_volume(matrix),
        "representative_fraction": fraction_momentum_json(representative),
        "expected_star_fractions": [fraction_momentum_json(value) for value in expected],
        "computed_star_fractions": [fraction_momentum_json(value) for value in orbit],
        "expected_star_size": len(expected),
        "computed_star_size": len(orbit),
        "all_arms_allowed": all(allowed),
        "all_arms_distinct": len(expected_set) == len(expected),
        "computed_star_matches_expected": set(orbit) == expected_set,
        "D6_sewing_closed": all(-1 not in permutation for permutation in sewing.values()),
        "D6_sewing_bijective": all(
            sorted(permutation) == list(range(len(expected)))
            for permutation in sewing.values()
        ),
        "character_numerators_over_N": numerators,
        "little_group_stabilizer": list(reciprocal_little_group(matrix, representative)),
        "sewing_permutations": sewing,
    }


def _regular_permutation_matrix(image: Sequence[int]) -> np.ndarray:
    dimension = len(image)
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    result[np.asarray(image, dtype=np.int64), np.arange(dimension)] = 1.0
    return result


def regular_character_projector_audit(
    supercell: IntMatrix,
    momentum: FractionMomentum,
    stabilizer_characters: Mapping[str, int],
    *,
    j2: float = 0.125,
) -> dict[str, object]:
    """Audit a momentum/little-group projector in the regular site space.

    The matrices have dimension ``N`` rather than a spin or fusion Hilbert
    dimension.  This makes the routine suitable for geometry/character
    preflights on an N=36 torus while explicitly providing no many-body phase
    evidence.
    """

    matrix = validate_hnf(supercell)
    if not hnf_fraction_momentum_allowed(matrix, momentum):
        raise ValueError("momentum is not allowed on the HNF torus")
    point_group = {
        name: (operation, image)
        for name, operation, image in hnf_point_group_permutations(matrix)
    }
    little_group = reciprocal_little_group(matrix, momentum)
    if set(stabilizer_characters) != set(little_group):
        raise ValueError("stabilizer characters must cover the exact little group")
    if any(character not in (-1, 1) for character in stabilizer_characters.values()):
        raise ValueError("only real one-dimensional characters are supported")

    volume = hnf_volume(matrix)
    momentum_projector = np.zeros((volume, volume), dtype=np.complex128)
    for translation, image in hnf_translation_permutations(matrix):
        phase = np.exp(
            2j
            * np.pi
            * float(momentum[0] * translation[0] + momentum[1] * translation[1])
        )
        momentum_projector += phase * _regular_permutation_matrix(image)
    momentum_projector /= volume

    little_projector = np.zeros_like(momentum_projector)
    for name in little_group:
        little_projector += (
            int(stabilizer_characters[name])
            * _regular_permutation_matrix(point_group[name][1])
        )
    little_projector /= len(little_group)
    combined = momentum_projector @ little_projector

    hamiltonian = np.zeros_like(combined)
    for first, second, coupling in triangular_hnf_j1j2_bonds(
        matrix, j1=1.0, j2=float(j2)
    ):
        hamiltonian[first, second] += coupling
        hamiltonian[second, first] += coupling

    sewing_errors: dict[str, float] = {}
    for name, operation, image in hnf_point_group_permutations(matrix):
        group = _regular_permutation_matrix(image)
        target_momentum = reciprocal_momentum_image(operation, momentum)
        target = np.zeros_like(momentum_projector)
        for translation, translation_image in hnf_translation_permutations(matrix):
            phase = np.exp(
                2j
                * np.pi
                * float(
                    target_momentum[0] * translation[0]
                    + target_momentum[1] * translation[1]
                )
            )
            target += phase * _regular_permutation_matrix(translation_image)
        target /= volume
        sewing_errors[name] = float(
            np.linalg.norm(group @ momentum_projector @ group.conj().T - target)
        )

    hermitian = 0.5 * (combined + combined.conj().T)
    return {
        "space": "single-particle_regular_site_representation_not_fusion",
        "supercell_hnf": [list(matrix[0]), list(matrix[1])],
        "n_sites": volume,
        "momentum_fraction": fraction_momentum_json(momentum),
        "little_group_stabilizer": list(little_group),
        "stabilizer_characters": {
            name: int(stabilizer_characters[name]) for name in little_group
        },
        "projector_rank": int(np.count_nonzero(np.linalg.eigvalsh(hermitian) > 0.5)),
        "momentum_projector_hermiticity_error": float(
            np.linalg.norm(momentum_projector - momentum_projector.conj().T)
        ),
        "momentum_projector_idempotency_error": float(
            np.linalg.norm(momentum_projector @ momentum_projector - momentum_projector)
        ),
        "little_group_projector_idempotency_error": float(
            np.linalg.norm(little_projector @ little_projector - little_projector)
        ),
        "momentum_little_group_commutator_error": float(
            np.linalg.norm(momentum_projector @ little_projector - little_projector @ momentum_projector)
        ),
        "combined_projector_hermiticity_error": float(
            np.linalg.norm(combined - combined.conj().T)
        ),
        "combined_projector_idempotency_error": float(
            np.linalg.norm(combined @ combined - combined)
        ),
        "Hamiltonian_projector_commutator_error": float(
            np.linalg.norm(hamiltonian @ combined - combined @ hamiltonian)
        ),
        "maximum_D6_momentum_sewing_error": max(sewing_errors.values()),
        "D6_momentum_sewing_errors": sewing_errors,
    }


def momentum_projector_matrix(
    supercell: IntMatrix,
    numerator: IntVector,
    *,
    target_q: int,
) -> np.ndarray:
    """Return the translation projector for ``k=2*pi*numerator/N``.

    With active translations the selected convention is
    ``U_t psi(k)=exp(-i k.t) psi(k)``, hence the group-average coefficient is
    ``exp(+i k.t)``.
    """

    matrix = validate_hnf(supercell)
    volume = hnf_volume(matrix)
    allowed = hnf_character_numerators(matrix)
    reduced_numerator = (int(numerator[0]) % volume, int(numerator[1]) % volume)
    if reduced_numerator not in allowed:
        raise ValueError("numerator is not a character of the quotient torus")
    dimension = count_paths(volume, int(target_q))
    projector = np.zeros((dimension, dimension), dtype=np.complex128)
    for translation, image in hnf_translation_permutations(matrix):
        phase = np.exp(
            2j
            * np.pi
            * (
                reduced_numerator[0] * translation[0]
                + reduced_numerator[1] * translation[1]
            )
            / volume
        )
        projector += phase * site_permutation_fusion_matrix(
            image, target_q=int(target_q)
        ).toarray()
    return projector / volume


def momentum_projector_audit(
    supercell: IntMatrix,
    *,
    target_q: int,
) -> dict[str, object]:
    """Audit the complete translation-character resolution of identity."""

    matrix = validate_hnf(supercell)
    volume = hnf_volume(matrix)
    dimension = count_paths(volume, int(target_q))
    projectors = {
        numerator: momentum_projector_matrix(
            matrix, numerator, target_q=int(target_q)
        )
        for numerator in hnf_character_numerators(matrix)
    }
    identity = np.eye(dimension, dtype=np.complex128)
    maximum_orthogonality = 0.0
    maximum_hermiticity = 0.0
    maximum_idempotency = 0.0
    ranks: dict[str, int] = {}
    items = tuple(projectors.items())
    for index, (numerator, projector) in enumerate(items):
        maximum_hermiticity = max(
            maximum_hermiticity,
            float(np.linalg.norm(projector - projector.conj().T)),
        )
        maximum_idempotency = max(
            maximum_idempotency,
            float(np.linalg.norm(projector @ projector - projector)),
        )
        ranks[f"{numerator[0]},{numerator[1]}"] = int(
            np.count_nonzero(np.linalg.eigvalsh(0.5 * (projector + projector.conj().T)) > 0.5)
        )
        for _, other in items[index + 1 :]:
            maximum_orthogonality = max(
                maximum_orthogonality,
                float(np.linalg.norm(projector @ other)),
            )
    total = sum(projectors.values(), start=np.zeros_like(identity))
    return {
        "supercell_hnf": [list(matrix[0]), list(matrix[1])],
        "n_sites": volume,
        "target_q": int(target_q),
        "dimension": dimension,
        "character_count": len(projectors),
        "ranks": ranks,
        "rank_sum": sum(ranks.values()),
        "completeness_error": float(np.linalg.norm(total - identity)),
        "maximum_orthogonality_error": maximum_orthogonality,
        "maximum_hermiticity_error": maximum_hermiticity,
        "maximum_idempotency_error": maximum_idempotency,
    }


def _d6_character(irrep: str, element_name: str) -> float:
    """Real character of one D6 irrep in the ``Mx*C6^p`` convention."""

    if irrep not in D6_IRREP_DIMENSIONS:
        raise ValueError(f"unknown D6 irrep: {irrep}")
    reflected = element_name.startswith("Mx")
    power = int(element_name.rsplit("^", 1)[1])
    if irrep == "A1":
        return 1.0
    if irrep == "A2":
        return -1.0 if reflected else 1.0
    if irrep == "B1":
        return (-1.0) ** power
    if irrep == "B2":
        return (-1.0) ** (power + (1 if reflected else 0))
    if reflected:
        return 0.0
    angular_momentum = 1 if irrep == "E1" else 2
    return float(2.0 * np.cos(angular_momentum * power * np.pi / 3.0))


def d6_character_projectors(
    supercell: IntMatrix,
    *,
    target_q: int,
) -> dict[str, np.ndarray]:
    """Return all D6 isotypic projectors for a full-D6 HNF torus."""

    matrix = validate_hnf(supercell)
    elements = hnf_point_group_permutations(matrix)
    if len(elements) != len(POINT_GROUP):
        raise ValueError("supercell does not preserve the full D6 point group")
    dimension = count_paths(hnf_volume(matrix), int(target_q))
    result: dict[str, np.ndarray] = {}
    for irrep, irrep_dimension in D6_IRREP_DIMENSIONS.items():
        projector = np.zeros((dimension, dimension), dtype=np.complex128)
        for name, _, image in elements:
            projector += _d6_character(irrep, name) * site_permutation_fusion_matrix(
                image, target_q=int(target_q)
            ).toarray()
        result[irrep] = projector * irrep_dimension / len(elements)
    return result


def d6_projector_audit(
    supercell: IntMatrix,
    *,
    target_q: int,
) -> dict[str, object]:
    """Audit D6 character orthogonality, idempotency, and completeness."""

    matrix = validate_hnf(supercell)
    dimension = count_paths(hnf_volume(matrix), int(target_q))
    projectors = d6_character_projectors(matrix, target_q=int(target_q))
    identity = np.eye(dimension, dtype=np.complex128)
    maximum_orthogonality = 0.0
    maximum_hermiticity = 0.0
    maximum_idempotency = 0.0
    ranks: dict[str, int] = {}
    items = tuple(projectors.items())
    for index, (irrep, projector) in enumerate(items):
        hermitian = 0.5 * (projector + projector.conj().T)
        ranks[irrep] = int(np.count_nonzero(np.linalg.eigvalsh(hermitian) > 0.5))
        maximum_hermiticity = max(
            maximum_hermiticity,
            float(np.linalg.norm(projector - projector.conj().T)),
        )
        maximum_idempotency = max(
            maximum_idempotency,
            float(np.linalg.norm(projector @ projector - projector)),
        )
        for _, other in items[index + 1 :]:
            maximum_orthogonality = max(
                maximum_orthogonality,
                float(np.linalg.norm(projector @ other)),
            )
    total = sum(projectors.values(), start=np.zeros_like(identity))
    return {
        "supercell_hnf": [list(matrix[0]), list(matrix[1])],
        "n_sites": hnf_volume(matrix),
        "target_q": int(target_q),
        "dimension": dimension,
        "ranks": ranks,
        "rank_sum": sum(ranks.values()),
        "completeness_error": float(np.linalg.norm(total - identity)),
        "maximum_orthogonality_error": maximum_orthogonality,
        "maximum_hermiticity_error": maximum_hermiticity,
        "maximum_idempotency_error": maximum_idempotency,
    }
