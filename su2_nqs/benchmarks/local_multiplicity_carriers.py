"""Exact-small symmetry-adapted local SU(2) multiplicity carriers.

This module is a mathematical feasibility audit, not a trainable ansatz.  It
answers which *independent* reduced channels exist before a larger fusion DAG
is designed.  In particular, the three valence-bond perfect matchings of four
spin halves have rank two, so adding a literal third plaquette route cannot by
itself create a new singlet carrier.

The square carrier resolves the four-spin singlet multiplicity as ``A1+B1``
under local ``D4``.  The triangular carrier instead uses a three-site triangle:
its two spin-1/2 multiplicity channels furnish the irreducible ``E`` doublet
of ``D3``.  All spins and projections use doubled-integer labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from ..lattices.spatial_symmetry import validate_site_permutation
from ..su2.fusion_tree import FusionTree
from ..su2.quantum_numbers import validate_m, validate_q
from ..su2.recoupling import tree_basis_matrix
from .triangular_locality import physical_dimer_product


SQUARE_C4 = (1, 3, 0, 2)
SQUARE_MX = (1, 0, 3, 2)
TRIANGLE_C3 = (1, 2, 0)
TRIANGLE_MIRROR = (0, 2, 1)


@dataclass(frozen=True)
class LocalMultiplicityCarrier:
    """One orthonormal fixed-``(q,m)`` local multiplicity space.

    ``vectors`` contains full computational-basis columns, with site zero the
    most significant bit.  ``generator_matrices`` are the corresponding
    physical site-permutation representations in that same column basis.
    """

    name: str
    n_sites: int
    target_q: int
    target_m2: int
    irrep_decomposition: str
    channel_labels: tuple[str, ...]
    vectors: np.ndarray
    generator_names: tuple[str, ...]
    generator_images: tuple[tuple[int, ...], ...]
    generator_matrices: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("carrier name must be nonempty")
        if (
            isinstance(self.n_sites, bool)
            or not isinstance(self.n_sites, Integral)
            or self.n_sites < 2
        ):
            raise ValueError("n_sites must be an integer of at least two")
        n_sites = int(self.n_sites)
        target_q = validate_q(self.target_q, name="target_q")
        target_m2 = validate_m(target_q, self.target_m2, name="target_m2")
        labels = tuple(str(value) for value in self.channel_labels)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("channel_labels must be nonempty and duplicate free")
        vectors = np.array(self.vectors, dtype=np.complex128, copy=True)
        if vectors.shape != (1 << n_sites, len(labels)):
            raise ValueError("carrier vectors have the wrong shape")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("carrier vectors must be finite")
        orthogonality = vectors.conj().T @ vectors
        if np.max(np.abs(orthogonality - np.eye(len(labels)))) > 2.0e-13:
            raise ValueError("carrier vectors must be orthonormal")
        names = tuple(str(value) for value in self.generator_names)
        images = tuple(
            validate_site_permutation(image, n_sites=n_sites)
            for image in self.generator_images
        )
        matrices = tuple(
            np.array(matrix, dtype=np.complex128, copy=True)
            for matrix in self.generator_matrices
        )
        if not names or len(set(names)) != len(names):
            raise ValueError("generator_names must be nonempty and duplicate free")
        if not (len(names) == len(images) == len(matrices)):
            raise ValueError("generator names, images, and matrices must align")
        if any(matrix.shape != (len(labels), len(labels)) for matrix in matrices):
            raise ValueError("generator matrices have the wrong shape")
        vectors.setflags(write=False)
        for matrix in matrices:
            matrix.setflags(write=False)
        object.__setattr__(self, "n_sites", n_sites)
        object.__setattr__(self, "target_q", target_q)
        object.__setattr__(self, "target_m2", target_m2)
        object.__setattr__(self, "channel_labels", labels)
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "generator_names", names)
        object.__setattr__(self, "generator_images", images)
        object.__setattr__(self, "generator_matrices", matrices)

    @property
    def multiplicity_rank(self) -> int:
        return self.vectors.shape[1]

    def generator_matrix(self, name: str) -> np.ndarray:
        """Return a read-only generator representation by registered name."""

        try:
            index = self.generator_names.index(name)
        except ValueError as error:
            raise KeyError(name) from error
        return self.generator_matrices[index]


def _permutation_indices(n_sites: int, image: Sequence[int]) -> np.ndarray:
    checked = validate_site_permutation(image, n_sites=n_sites)
    shifts = np.arange(n_sites - 1, -1, -1, dtype=np.int64)
    target = np.arange(1 << n_sites, dtype=np.int64)
    bits = (target[:, None] >> shifts[None, :]) & 1
    source_bits = bits[:, np.asarray(checked, dtype=np.int64)]
    indices = source_bits @ (1 << shifts)
    result = np.asarray(indices, dtype=np.int64)
    result.setflags(write=False)
    return result


def apply_site_permutation(
    vectors: np.ndarray, image: Sequence[int]
) -> np.ndarray:
    """Apply an active physical-spin site map to one or more full vectors."""

    states = np.asarray(vectors, dtype=np.complex128)
    if states.ndim not in (1, 2):
        raise ValueError("vectors must be a vector or a column matrix")
    dimension = states.shape[0]
    n_sites = dimension.bit_length() - 1
    if 1 << n_sites != dimension:
        raise ValueError("the leading vector dimension must be a power of two")
    return states[_permutation_indices(n_sites, image)]


def site_permutation_in_carrier(
    carrier: LocalMultiplicityCarrier, image: Sequence[int]
) -> tuple[np.ndarray, float]:
    """Return the restricted permutation and its outside-carrier residual."""

    if not isinstance(carrier, LocalMultiplicityCarrier):
        raise TypeError("carrier must be a LocalMultiplicityCarrier")
    transformed = apply_site_permutation(carrier.vectors, image)
    matrix = carrier.vectors.conj().T @ transformed
    residual = float(np.linalg.norm(transformed - carrier.vectors @ matrix))
    return np.asarray(matrix, dtype=np.complex128), residual


def _build_carrier(
    *,
    name: str,
    n_sites: int,
    target_q: int,
    target_m2: int,
    irrep_decomposition: str,
    channel_labels: tuple[str, ...],
    vectors: np.ndarray,
    generators: tuple[tuple[str, tuple[int, ...]], ...],
) -> LocalMultiplicityCarrier:
    matrices = []
    residuals = []
    for _, image in generators:
        transformed = apply_site_permutation(vectors, image)
        matrix = vectors.conj().T @ transformed
        matrices.append(matrix)
        residuals.append(np.linalg.norm(transformed - vectors @ matrix))
    if max(residuals, default=0.0) > 2.0e-13:
        raise RuntimeError("requested multiplicity space is not symmetry invariant")
    return LocalMultiplicityCarrier(
        name=name,
        n_sites=n_sites,
        target_q=target_q,
        target_m2=target_m2,
        irrep_decomposition=irrep_decomposition,
        channel_labels=channel_labels,
        vectors=vectors,
        generator_names=tuple(value[0] for value in generators),
        generator_images=tuple(value[1] for value in generators),
        generator_matrices=tuple(matrices),
    )


def square_plaquette_singlet_carrier() -> LocalMultiplicityCarrier:
    """Return orthonormal local ``A1`` and ``B1`` four-spin singlets."""

    horizontal = physical_dimer_product(4, ((0, 1), (2, 3)))
    vertical = physical_dimer_product(4, ((0, 2), (1, 3)))
    a1 = (horizontal + vertical) / np.sqrt(3.0)
    b1 = horizontal - vertical
    return _build_carrier(
        name="square_2x2_q0_A1_plus_B1",
        n_sites=4,
        target_q=0,
        target_m2=0,
        irrep_decomposition="A1_plus_B1",
        channel_labels=("A1", "B1"),
        vectors=np.column_stack((a1, b1)),
        generators=(("c4", SQUARE_C4), ("mx", SQUARE_MX)),
    )


def square_plaquette_spin_one_carrier() -> LocalMultiplicityCarrier:
    """Return the rank-three four-spin ``q=2`` multiplicity block.

    A highest-weight ``m2=2`` slice is sufficient to represent multiplicity;
    magnetic partners are fixed by SU(2).  Under local D4 this block restricts
    as ``B2+E``.
    """

    vectors, states = tree_basis_matrix(
        FusionTree.balanced(4), target_q=2, target_m=2
    )
    labels = tuple("q=" + ",".join(str(q) for q in state.labels) for state in states)
    return _build_carrier(
        name="square_2x2_q2_B2_plus_E",
        n_sites=4,
        target_q=2,
        target_m2=2,
        irrep_decomposition="B2_plus_E",
        channel_labels=labels,
        vectors=vectors,
        generators=(("c4", SQUARE_C4), ("mx", SQUARE_MX)),
    )


def triangular_triangle_spin_half_carrier() -> LocalMultiplicityCarrier:
    """Return the rank-two triangular ``q=1`` multiplicity ``E`` doublet."""

    vectors, states = tree_basis_matrix(
        FusionTree.left_associated(3), target_q=1, target_m=1
    )
    labels = tuple("q=" + ",".join(str(q) for q in state.labels) for state in states)
    return _build_carrier(
        name="triangular_triangle_q1_E",
        n_sites=3,
        target_q=1,
        target_m2=1,
        irrep_decomposition="E",
        channel_labels=labels,
        vectors=vectors,
        generators=(("c3", TRIANGLE_C3), ("mirror", TRIANGLE_MIRROR)),
    )


def heisenberg_bond_matrix(
    n_sites: int, first: int, second: int
) -> np.ndarray:
    """Return exact full-basis ``S_first dot S_second`` for spin halves."""

    if (
        isinstance(n_sites, bool)
        or not isinstance(n_sites, Integral)
        or n_sites < 2
    ):
        raise ValueError("n_sites must be an integer of at least two")
    n_sites = int(n_sites)
    if any(
        isinstance(site, bool) or not isinstance(site, Integral)
        for site in (first, second)
    ):
        raise TypeError("bond sites must be integers")
    first, second = int(first), int(second)
    if not 0 <= first < n_sites or not 0 <= second < n_sites or first == second:
        raise ValueError("bond sites must be distinct and in range")
    image = list(range(n_sites))
    image[first], image[second] = image[second], image[first]
    permutation = np.eye(1 << n_sites, dtype=np.complex128)[
        _permutation_indices(n_sites, image)
    ]
    result = 0.5 * permutation - 0.25 * np.eye(
        1 << n_sites, dtype=np.complex128
    )
    result.setflags(write=False)
    return result


def heisenberg_bond_in_carrier(
    carrier: LocalMultiplicityCarrier, first: int, second: int
) -> np.ndarray:
    """Project one exact Heisenberg bond into a multiplicity carrier."""

    operator = heisenberg_bond_matrix(carrier.n_sites, first, second)
    return carrier.vectors.conj().T @ operator @ carrier.vectors


def _spin_squared_residual(carrier: LocalMultiplicityCarrier) -> float:
    n_sites = carrier.n_sites
    coefficient = 3.0 * n_sites / 4.0 - n_sites * (n_sites - 1) / 4.0
    operator = coefficient * np.eye(1 << n_sites, dtype=np.complex128)
    for first in range(n_sites):
        for second in range(first + 1, n_sites):
            swap = 2.0 * heisenberg_bond_matrix(n_sites, first, second)
            swap += 0.5 * np.eye(1 << n_sites, dtype=np.complex128)
            operator += swap
    eigenvalue = carrier.target_q * (carrier.target_q + 2) / 4.0
    return float(
        np.linalg.norm(operator @ carrier.vectors - eigenvalue * carrier.vectors)
    )


def _real_matrix(value: np.ndarray) -> list[list[float]]:
    array = np.asarray(value)
    if np.max(np.abs(np.imag(array)), initial=0.0) > 2.0e-13:
        raise RuntimeError("audit matrix unexpectedly became complex")
    return np.real(array).tolist()


def local_multiplicity_carrier_audit() -> dict[str, object]:
    """Return a JSON-ready exact audit of the square and triangle carriers."""

    horizontal = physical_dimer_product(4, ((0, 1), (2, 3)))
    vertical = physical_dimer_product(4, ((0, 2), (1, 3)))
    diagonal = physical_dimer_product(4, ((0, 3), (1, 2)))
    valence_bonds = np.column_stack((horizontal, vertical, diagonal))
    gram = valence_bonds.conj().T @ valence_bonds
    gram_eigenvalues = np.linalg.eigvalsh(gram)
    valence_rank = int(
        np.count_nonzero(gram_eigenvalues > 1.0e-12 * gram_eigenvalues[-1])
    )
    null_relation = horizontal - vertical + diagonal
    first_two_pairs = {(0, 1), (2, 3), (0, 2), (1, 3)}
    third_pairs = {(0, 3), (1, 2)}
    all_three_pairs = first_two_pairs | third_pairs
    square_j1_pairs = {(0, 1), (2, 3), (0, 2), (1, 3)}
    square_j2_pairs = {(0, 3), (1, 2)}
    triangular_j1_pairs = {
        (0, 1),
        (2, 3),
        (0, 2),
        (1, 3),
        (0, 3),
    }
    triangular_non_j1_pair = {(1, 2)}

    square_q0 = square_plaquette_singlet_carrier()
    square_q2 = square_plaquette_spin_one_carrier()
    triangle_q1 = triangular_triangle_spin_half_carrier()
    q0_c4 = square_q0.generator_matrix("c4")
    q0_mx = square_q0.generator_matrix("mx")
    q2_c4 = square_q2.generator_matrix("c4")
    q2_mx = square_q2.generator_matrix("mx")
    c3 = triangle_q1.generator_matrix("c3")
    mirror = triangle_q1.generator_matrix("mirror")

    square_j1 = sum(
        heisenberg_bond_in_carrier(square_q0, first, second)
        for first, second in ((0, 1), (2, 3), (0, 2), (1, 3))
    )
    square_j2 = sum(
        heisenberg_bond_in_carrier(square_q0, first, second)
        for first, second in ((0, 3), (1, 2))
    )
    triangle_bonds = {
        f"{first}{second}": heisenberg_bond_in_carrier(
            triangle_q1, first, second
        )
        for first, second in ((0, 1), (1, 2), (0, 2))
    }
    triangle_sum = sum(triangle_bonds.values())

    return {
        "schema_version": 1,
        "scope": "exact_small_local_multiplicity_feasibility_no_training",
        "conventions": {
            "quantum_numbers": "doubled_integer_q_and_m2",
            "computational_basis": "bit0_down_bit1_up_site0_most_significant",
            "square_site_order": "top_left_top_right_bottom_left_bottom_right",
        },
        "literal_three_valence_bond_pairings": {
            "labels": ["horizontal", "vertical", "diagonal"],
            "gram_matrix": _real_matrix(gram),
            "gram_eigenvalues": np.real(gram_eigenvalues).tolist(),
            "rank": valence_rank,
            "null_relation": "horizontal - vertical + diagonal = 0",
            "null_relation_residual_norm": float(np.linalg.norm(null_relation)),
            "coordinates_in_A1_B1_rows": _real_matrix(
                square_q0.vectors.conj().T @ valence_bonds
            ),
            "third_pairing_new_q0_rank": 0,
            "shell_coverage": {
                "square": {
                    "first_two_J1_covered_over_total": [
                        len(first_two_pairs & square_j1_pairs),
                        len(square_j1_pairs),
                    ],
                    "first_two_J2_covered_over_total": [
                        len(first_two_pairs & square_j2_pairs),
                        len(square_j2_pairs),
                    ],
                    "all_three_J1_covered_over_total": [
                        len(all_three_pairs & square_j1_pairs),
                        len(square_j1_pairs),
                    ],
                    "all_three_J2_covered_over_total": [
                        len(all_three_pairs & square_j2_pairs),
                        len(square_j2_pairs),
                    ],
                },
                "triangular": {
                    "first_two_J1_covered_over_total": [
                        len(first_two_pairs & triangular_j1_pairs),
                        len(triangular_j1_pairs),
                    ],
                    "third_J1_covered_over_total": [
                        len(third_pairs & triangular_j1_pairs),
                        len(triangular_j1_pairs),
                    ],
                    "third_non_J1_J2_pairs": len(
                        third_pairs & triangular_non_j1_pair
                    ),
                    "all_three_J1_covered_over_total": [
                        len(all_three_pairs & triangular_j1_pairs),
                        len(triangular_j1_pairs),
                    ],
                },
                "interpretation": (
                    "square third matching is purely J2; triangular third "
                    "matching combines one J1 and one J2 bond"
                ),
            },
        },
        "square_plaquette": {
            "q0_multiplicity_rank": square_q0.multiplicity_rank,
            "q0_irrep_decomposition": square_q0.irrep_decomposition,
            "q0_c4_matrix_in_A1_B1": _real_matrix(q0_c4),
            "q0_mx_matrix_in_A1_B1": _real_matrix(q0_mx),
            "q0_spin_squared_residual_norm": _spin_squared_residual(square_q0),
            "q0_J1_shell_matrix": _real_matrix(square_j1),
            "q0_J2_shell_matrix": _real_matrix(square_j2),
            "q0_local_energies": {
                "A1": "-2*J1 + 0.5*J2",
                "B1": "-1.5*J2",
            },
            "q2_multiplicity_rank": square_q2.multiplicity_rank,
            "q2_irrep_decomposition": square_q2.irrep_decomposition,
            "q2_c4_trace": float(np.trace(q2_c4).real),
            "q2_mx_trace": float(np.trace(q2_mx).real),
            "q2_spin_squared_residual_norm": _spin_squared_residual(square_q2),
            "d4_generator_relation_error": float(
                max(
                    np.linalg.norm(np.linalg.matrix_power(q0_c4, 4) - np.eye(2)),
                    np.linalg.norm(q0_mx @ q0_mx - np.eye(2)),
                    np.linalg.norm(q0_mx @ q0_c4 @ q0_mx - q0_c4.conj().T),
                    np.linalg.norm(np.linalg.matrix_power(q2_c4, 4) - np.eye(3)),
                    np.linalg.norm(q2_mx @ q2_mx - np.eye(3)),
                    np.linalg.norm(q2_mx @ q2_c4 @ q2_mx - q2_c4.conj().T),
                )
            ),
        },
        "triangular_triangle": {
            "q1_multiplicity_rank": triangle_q1.multiplicity_rank,
            "q1_irrep_decomposition": triangle_q1.irrep_decomposition,
            "q1_c3_matrix": _real_matrix(c3),
            "q1_mirror_matrix": _real_matrix(mirror),
            "q1_c3_trace": float(np.trace(c3).real),
            "q1_mirror_trace": float(np.trace(mirror).real),
            "q1_spin_squared_residual_norm": _spin_squared_residual(triangle_q1),
            "bond_matrices": {
                name: _real_matrix(matrix)
                for name, matrix in triangle_bonds.items()
            },
            "bond_sum_matrix": _real_matrix(triangle_sum),
            "d3_generator_relation_error": float(
                max(
                    np.linalg.norm(np.linalg.matrix_power(c3, 3) - np.eye(2)),
                    np.linalg.norm(mirror @ mirror - np.eye(2)),
                    np.linalg.norm(mirror @ c3 @ mirror - c3.conj().T),
                )
            ),
            "physical_picture": (
                "the three bond operators are 120-degree pseudospin axes in "
                "the E doublet and sum to -3/4 times the identity"
            ),
        },
        "decision": {
            "literal_third_four_site_pairing": "reject_as_redundant_q0_mechanism",
            "square_candidate": "carry_orthonormal_A1_B1_multiplicity_labels",
            "triangular_candidate": "carry_triangle_q1_E_doublet",
            "production_catalog_or_training_started": False,
        },
    }


__all__ = [
    "LocalMultiplicityCarrier",
    "SQUARE_C4",
    "SQUARE_MX",
    "TRIANGLE_C3",
    "TRIANGLE_MIRROR",
    "apply_site_permutation",
    "heisenberg_bond_in_carrier",
    "heisenberg_bond_matrix",
    "local_multiplicity_carrier_audit",
    "site_permutation_in_carrier",
    "square_plaquette_singlet_carrier",
    "square_plaquette_spin_one_carrier",
    "triangular_triangle_spin_half_carrier",
]
