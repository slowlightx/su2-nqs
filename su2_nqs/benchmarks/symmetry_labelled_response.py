"""Exact-small symmetry-labelled response loops in the Gamma-A1 space.

The ground-state ansatz must remain in ``Gamma-A1``.  Consequently a bond
nematic or chiral operator cannot be added to it linearly: it first creates a
state in a nontrivial point-group irrep.  This module closes that excursion
with the positive scalar

``K_Gamma = sum_mu O_(Gamma,mu)^dagger O_(Gamma,mu)``.

The resulting matrices act only on SU(2) singlet multiplicity coordinates.
They are accompanied by exact fixed-``Sz`` certificates for the intermediate
irrep, singlet closure, and the return to ``Gamma-A1``.  A single orthogonal
conjugation inside the same A1 space provides an isospectral control.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

from ..lattices.square_symmetry import square_site_generators
from ..lattices.triangular_e2 import e2_orientation_representation
from ..lattices.triangular_hnf import (
    C6,
    MX,
    hnf_site_permutation,
)
from ..operators.triangular_resonance import triangular_e2_actions
from .clebsch_tree_exact import physical_permutation_index_table
from .four_cell_schur_frontier import TRIANGULAR_N12_SUPERCELL
from .physical_interpretation import (
    FixedSzPairActions,
    uniform_chirality_action,
)


VectorAction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SymmetryLabelledResponseA1:
    """Response-return operators and their matched in-A1 control."""

    labels: tuple[str, ...]
    operators: np.ndarray
    scrambled_operators: np.ndarray
    moment_operators: np.ndarray
    diagnostics: dict[str, object]


def _haar_orthogonal(dimension: int, seed: int) -> np.ndarray:
    raw = np.random.default_rng(seed).normal(size=(dimension, dimension))
    orthogonal, triangular = np.linalg.qr(raw)
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    return orthogonal * signs[None, :]


def _apply_columns(action: VectorAction, vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.complex128)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a matrix of column vectors")
    return np.column_stack([action(matrix[:, index]) for index in range(matrix.shape[1])])


def _projector_leakage(vectors: np.ndarray, basis: np.ndarray) -> float:
    value = np.asarray(vectors, dtype=np.complex128)
    coordinates = basis.conj().T @ value
    residual = value - basis @ coordinates
    return float(np.linalg.norm(residual) / max(np.linalg.norm(value), 1.0e-30))


def _normalized_response_operator(moment: np.ndarray) -> np.ndarray:
    value = 0.5 * (moment + moment.conj().T)
    dimension = value.shape[0]
    value = value - np.trace(value) / dimension * np.eye(dimension)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-13:
        raise RuntimeError("a symmetry-labelled response loop is scalar identity only")
    value = value / norm
    if np.max(np.abs(value.imag)) < 2.0e-12:
        value = value.real.astype(np.float64)
    return np.asarray(value)


def _response_loop(
    physical_a1: np.ndarray,
    singlet_basis: np.ndarray,
    actions: Sequence[VectorAction],
) -> tuple[np.ndarray, np.ndarray, dict[str, float], tuple[np.ndarray, ...]]:
    if not actions:
        raise ValueError("a response loop needs at least one irrep component")
    first = tuple(_apply_columns(action, physical_a1) for action in actions)
    moment = sum((image.conj().T @ image for image in first), start=np.zeros(
        (physical_a1.shape[1], physical_a1.shape[1]), dtype=np.complex128
    ))
    returned = sum(
        (_apply_columns(action, image) for action, image in zip(actions, first, strict=True)),
        start=np.zeros_like(physical_a1, dtype=np.complex128),
    )
    compressed = physical_a1.conj().T @ returned
    return_error = float(np.linalg.norm(compressed - moment))
    diagnostics = {
        "first_order_norm": float(math.sqrt(sum(np.linalg.norm(image) ** 2 for image in first))),
        "singlet_relative_leakage": _projector_leakage(
            np.column_stack(first), singlet_basis
        ),
        "scalar_a1_relative_leakage": _projector_leakage(returned, physical_a1),
        "gram_vs_direct_return_error": return_error,
        "minimum_moment_eigenvalue": float(np.min(np.linalg.eigvalsh(0.5 * (moment + moment.conj().T)))),
    }
    return _normalized_response_operator(moment), moment, diagnostics, first


def _square_families(linear_size: int) -> dict[str, tuple[tuple[int, int], ...]]:
    length = int(linear_size)
    if length < 3:
        raise ValueError("square response families require L >= 3")

    def site(x: int, y: int) -> int:
        return (y % length) * length + (x % length)

    horizontal = []
    vertical = []
    diagonal_plus = []
    diagonal_minus = []
    triangles = []
    for y in range(length):
        for x in range(length):
            a = site(x, y)
            b = site(x + 1, y)
            c = site(x + 1, y + 1)
            d = site(x, y + 1)
            horizontal.append((a, b))
            vertical.append((a, d))
            diagonal_plus.append((a, c))
            diagonal_minus.append((b, d))
            # The cyclic four-triple sum is invariant under C4 and odd under
            # reflection.  A fixed diagonal triangulation would spuriously
            # break C4 before the response is squared.
            triangles.extend(
                ((a, b, c), (b, c, d), (c, d, a), (d, a, b))
            )
    return {
        "horizontal": tuple(horizontal),
        "vertical": tuple(vertical),
        "diagonal_plus": tuple(diagonal_plus),
        "diagonal_minus": tuple(diagonal_minus),
        "triangles": tuple(triangles),
    }


def _bond_difference_action(
    pair_actions: FixedSzPairActions,
    positive: Sequence[tuple[int, int]],
    negative: Sequence[tuple[int, int]],
) -> VectorAction:
    scale = 1.0 / math.sqrt(len(positive) + len(negative))

    def action(state: np.ndarray) -> np.ndarray:
        vector = np.asarray(state, dtype=np.complex128)
        answer = np.zeros_like(vector)
        for first, second in positive:
            answer += pair_actions.pair(vector, first, second)
        for first, second in negative:
            answer -= pair_actions.pair(vector, first, second)
        return scale * answer

    return action


def _chirality_action(
    pair_actions: FixedSzPairActions,
    triangles: Sequence[tuple[int, int, int]],
) -> VectorAction:
    scale = 1.0 / math.sqrt(len(triangles))

    def action(state: np.ndarray) -> np.ndarray:
        vector = np.asarray(state, dtype=np.complex128)
        answer = np.zeros_like(vector)
        for triangle in triangles:
            answer += pair_actions.chirality(vector, triangle)
        return scale * answer

    return action


def _character_residual(
    images: np.ndarray,
    basis_configurations: Sequence[Sequence[int]],
    site_image: Sequence[int],
    character: float,
) -> float:
    indices = physical_permutation_index_table(
        basis_configurations, (site_image,)
    )[0]
    pulled = images[indices]
    return float(
        np.linalg.norm(pulled - character * images)
        / max(np.linalg.norm(images), 1.0e-30)
    )


def _build_square(
    physical_a1: np.ndarray,
    singlet_basis: np.ndarray,
    configurations: Sequence[Sequence[int]],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, dict[str, object]]:
    pair_actions = FixedSzPairActions(configurations)
    families = _square_families(4)
    actions = {
        "B1_J1_nematic_return": _bond_difference_action(
            pair_actions, families["horizontal"], families["vertical"]
        ),
        "B2_J2_nematic_return": _bond_difference_action(
            pair_actions, families["diagonal_plus"], families["diagonal_minus"]
        ),
        "A2_chirality_return": _chirality_action(
            pair_actions, families["triangles"]
        ),
    }
    characters = {
        "B1_J1_nematic_return": {"tx": 1.0, "ty": 1.0, "c4": -1.0, "mx": 1.0},
        "B2_J2_nematic_return": {"tx": 1.0, "ty": 1.0, "c4": -1.0, "mx": -1.0},
        "A2_chirality_return": {"tx": 1.0, "ty": 1.0, "c4": 1.0, "mx": -1.0},
    }
    generators = square_site_generators(4)
    labels = tuple(actions)
    operators = []
    moments = []
    diagnostics: dict[str, object] = {}
    for label in labels:
        operator, moment, row, first = _response_loop(
            physical_a1, singlet_basis, (actions[label],)
        )
        image = first[0]
        row["irrep_covariance_max_relative_error"] = max(
            _character_residual(
                image, configurations, generators[name], character
            )
            for name, character in characters[label].items()
        )
        row["generator_characters"] = characters[label]
        diagnostics[label] = row
        operators.append(operator)
        moments.append(moment)
    return labels, np.stack(operators), np.stack(moments), diagnostics


def _build_triangular(
    physical_a1: np.ndarray,
    singlet_basis: np.ndarray,
    configurations: Sequence[Sequence[int]],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, dict[str, object]]:
    pair_actions = FixedSzPairActions(configurations)
    supercell = TRIANGULAR_N12_SUPERCELL

    def e2_component(component: int) -> VectorAction:
        return lambda state: triangular_e2_actions(
            state, pair_actions.pair, supercell
        )[component]

    triangle_count = 2 * len(configurations[0])

    def chirality(state: np.ndarray) -> np.ndarray:
        return uniform_chirality_action(
            state, pair_actions, supercell
        ) / math.sqrt(triangle_count)

    definitions = {
        "E2_J1_nematic_return": (e2_component(0), e2_component(1)),
        "A2_chirality_return": (chirality,),
    }
    labels = tuple(definitions)
    operators = []
    moments = []
    diagnostics: dict[str, object] = {}
    for label, response_actions in definitions.items():
        operator, moment, row, first = _response_loop(
            physical_a1, singlet_basis, response_actions
        )
        if label.startswith("E2"):
            doublet = np.stack(first)
            covariance = []
            for operation in (C6, MX):
                site_image = hnf_site_permutation(supercell, operation=operation)
                indices = physical_permutation_index_table(
                    configurations, (site_image,)
                )[0]
                pulled = doublet[:, indices, :]
                representation = e2_orientation_representation(operation)
                represented = np.einsum(
                    "ab,bnd->and", representation, pulled
                )
                represented_transpose = np.einsum(
                    "ab,bnd->and", representation.T, pulled
                )
                covariance.append(
                    min(
                        float(np.linalg.norm(doublet - represented)),
                        float(np.linalg.norm(doublet - represented_transpose)),
                    )
                    / max(float(np.linalg.norm(doublet)), 1.0e-30)
                )
            row["irrep_covariance_max_relative_error"] = max(covariance)
            row["irrep_dimension"] = 2
        else:
            image = first[0]
            row["irrep_covariance_max_relative_error"] = max(
                _character_residual(
                    image,
                    configurations,
                    hnf_site_permutation(supercell, operation=C6),
                    1.0,
                ),
                _character_residual(
                    image,
                    configurations,
                    hnf_site_permutation(supercell, operation=MX),
                    -1.0,
                ),
            )
            row["generator_characters"] = {"C6": 1.0, "Mx": -1.0}
        diagnostics[label] = row
        operators.append(operator)
        moments.append(moment)
    return labels, np.stack(operators), np.stack(moments), diagnostics


def build_symmetry_labelled_response_a1(
    case_name: str,
    physical_a1_basis: np.ndarray,
    singlet_basis: np.ndarray,
    configurations: Sequence[Sequence[int]],
) -> SymmetryLabelledResponseA1:
    """Build response-return loops for one registered exact-small case."""

    physical_a1 = np.asarray(physical_a1_basis, dtype=np.complex128)
    singlet = np.asarray(singlet_basis, dtype=np.complex128)
    checked = tuple(tuple(int(bit) for bit in row) for row in configurations)
    if physical_a1.ndim != 2 or singlet.ndim != 2:
        raise ValueError("basis arrays must be rank two")
    if physical_a1.shape[0] != len(checked) or singlet.shape[0] != len(checked):
        raise ValueError("basis arrays and configurations disagree")
    if case_name == "triangular_n12_j2_0125":
        labels, operators, moments, channels = _build_triangular(
            physical_a1, singlet, checked
        )
        scramble_seed = 2026092501
    elif case_name == "square_n16_j2_05":
        labels, operators, moments, channels = _build_square(
            physical_a1, singlet, checked
        )
        scramble_seed = 2026092502
    else:
        raise ValueError(f"unknown response-loop case: {case_name}")
    dimension = physical_a1.shape[1]
    rotation = _haar_orthogonal(dimension, scramble_seed)
    scrambled = np.einsum("ai,kab,bj->kij", rotation, operators, rotation)
    hermiticity_error = float(
        max(
            np.max(np.abs(operators - operators.conj().transpose(0, 2, 1))),
            np.max(np.abs(scrambled - scrambled.conj().transpose(0, 2, 1))),
        )
    )
    spectrum_error = max(
        float(np.max(np.abs(np.linalg.eigvalsh(left) - np.linalg.eigvalsh(right))))
        for left, right in zip(operators, scrambled, strict=True)
    )
    maximum_exact_error = max(
        hermiticity_error,
        spectrum_error,
        max(
            max(
                float(row["singlet_relative_leakage"]),
                float(row["scalar_a1_relative_leakage"]),
                float(row["gram_vs_direct_return_error"]),
                float(row["irrep_covariance_max_relative_error"]),
            )
            for row in channels.values()
        ),
    )
    if maximum_exact_error > 2.0e-10:
        raise RuntimeError(
            f"symmetry-labelled response certificate failed: {maximum_exact_error:.3e}"
        )
    for array in (operators, scrambled, moments):
        array.setflags(write=False)
    return SymmetryLabelledResponseA1(
        labels=labels,
        operators=operators,
        scrambled_operators=scrambled,
        moment_operators=moments,
        diagnostics={
            "case_name": case_name,
            "a1_dimension": dimension,
            "response_operator_count": len(labels),
            "maximum_hermiticity_error": hermiticity_error,
            "maximum_scrambled_spectrum_error": spectrum_error,
            "maximum_exact_certificate_error": maximum_exact_error,
            "channels": channels,
            "mechanism": (
                "A1 -> symmetry-labelled response irrep -> A1 via "
                "sum_mu O_Gamma,mu^dagger O_Gamma,mu"
            ),
        },
    )


__all__ = [
    "SymmetryLabelledResponseA1",
    "build_symmetry_labelled_response_a1",
]
