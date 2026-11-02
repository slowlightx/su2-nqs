"""Sampled component algebra for the two-layer SU(2)/A1 Operator-ViT.

The functions in this module contain no sampler or optimizer.  They define
the ordered scalar words and convert their raw on-demand actions into the
centered, normalized component convention used by the exact-support model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

import jax.numpy as jnp
import numpy as np

from ..models.su2_a1_operator_vit import (
    SU2A1OperatorViTConfig,
    su2_a1_operator_vit_component_weights,
)
from ..operators.triangular_a1_scalar import TRIANGULAR_A1_SCALAR_LABELS


ScalarWord = tuple[str, ...]


def centered_component_coefficients_to_raw_words(
    coefficients: Sequence[float],
    *,
    labels: Sequence[str],
    means: Sequence[float],
    scales: Sequence[float],
) -> dict[ScalarWord, float]:
    """Convert a centered depth-two polynomial to physical raw-word weights.

    This is useful for transferring a physics-selected operator polynomial
    between sizes whose carrier means and response scales differ.  Returned
    words retain operator order; no commutativity is assumed.
    """

    checked_labels = tuple(str(label) for label in labels)
    count = len(checked_labels)
    vector = np.asarray(coefficients, dtype=np.float64)
    means_array = np.asarray(means, dtype=np.float64)
    scales_array = np.asarray(scales, dtype=np.float64)
    if vector.shape != (1 + count + count * count,):
        raise ValueError("coefficients do not match the depth-two component count")
    if means_array.shape != (count,) or scales_array.shape != (count,):
        raise ValueError("means and scales must match labels")
    if np.any(~np.isfinite(vector)) or np.any(~np.isfinite(means_array)):
        raise ValueError("coefficients and means must be finite")
    if np.any(~np.isfinite(scales_array)) or np.any(scales_array <= 0.0):
        raise ValueError("scales must be positive and finite")

    raw: dict[ScalarWord, float] = {(): float(vector[0])}

    def add(word: ScalarWord, value: float) -> None:
        raw[word] = raw.get(word, 0.0) + float(value)

    for index, label in enumerate(checked_labels):
        value = vector[1 + index] / scales_array[index]
        add((label,), value)
        add((), -means_array[index] * value)
    offset = 1 + count
    for outer, outer_label in enumerate(checked_labels):
        for inner, inner_label in enumerate(checked_labels):
            value = vector[offset + count * outer + inner] / (
                scales_array[outer] * scales_array[inner]
            )
            add((outer_label, inner_label), value)
            add((outer_label,), -means_array[inner] * value)
            add((inner_label,), -means_array[outer] * value)
            add((), means_array[outer] * means_array[inner] * value)
    return {
        word: value for word, value in raw.items() if abs(value) > 1.0e-15
    }


def krylov_control_basis(
    *,
    hamiltonian_weights: Sequence[float],
    means: Sequence[float],
    scales: Sequence[float],
) -> np.ndarray:
    """Represent ``I,H,H^2`` in a centered scalar-token component basis.

    Component columns are ordered as ``I, O_a, O_b O_a``.  The returned
    ``(1+n+n^2,3)`` matrix maps Krylov coefficients into that basis and is valid for
    an arbitrary linear combination ``H=sum_a h_a O_a``.  This lets a
    shell-resolved ``J1,J2`` model contain exactly the same two-Lanczos control
    without collapsing the two physical tokens into one.
    """

    weights = np.asarray(hamiltonian_weights, dtype=np.float64)
    means_array = np.asarray(means, dtype=np.float64)
    scales_array = np.asarray(scales, dtype=np.float64)
    if weights.ndim != 1 or len(weights) < 2:
        raise ValueError("the control basis requires at least two tokens")
    if any(array.shape != weights.shape for array in (means_array, scales_array)):
        raise ValueError("weights, means, and scales must have matching shapes")
    if np.any(~np.isfinite(weights)) or np.any(~np.isfinite(means_array)):
        raise ValueError("Hamiltonian weights and means must be finite")
    if np.any(~np.isfinite(scales_array)) or np.any(scales_array <= 0.0):
        raise ValueError("response scales must be positive and finite")

    count = len(weights)
    basis = np.zeros((1 + count + count * count, 3), dtype=np.float64)
    basis[0, 0] = 1.0
    basis[0, 1] = float(weights @ means_array)
    basis[1 : 1 + count, 1] = weights * scales_array
    for outer in range(count):
        for inner in range(count):
            coefficient = weights[outer] * weights[inner]
            basis[0, 2] += coefficient * means_array[outer] * means_array[inner]
            basis[1 + outer, 2] += (
                coefficient * means_array[inner] * scales_array[outer]
            )
            basis[1 + inner, 2] += (
                coefficient * means_array[outer] * scales_array[inner]
            )
            basis[1 + count + count * outer + inner, 2] += (
                coefficient * scales_array[outer] * scales_array[inner]
            )
    if np.linalg.matrix_rank(basis) != 3:
        raise ValueError("Hamiltonian control basis is rank deficient")
    return basis


def scalar_component_words(labels: Sequence[str]) -> tuple[ScalarWord, ...]:
    """Return ``I``, one-token, and ordered two-token words."""

    checked = tuple(str(label) for label in labels)
    if len(checked) < 1 or len(set(checked)) != len(checked):
        raise ValueError("scalar labels must be nonempty and unique")
    return (
        (),
        *((label,) for label in checked),
        *((outer, inner) for outer in checked for inner in checked),
    )


def operator_vit_component_words() -> tuple[ScalarWord, ...]:
    """Return ``I``, ``O_a``, and ``O_b O_a`` in model component order."""

    return scalar_component_words(TRIANGULAR_A1_SCALAR_LABELS)


def expanded_base_words(
    effective_words: Sequence[ScalarWord],
    expansions: Mapping[str, Mapping[str, float]],
) -> tuple[ScalarWord, ...]:
    """Expand linear-combination tokens into unique registered base words."""

    result: list[ScalarWord] = []
    for word in effective_words:
        if not word:
            result.append(())
            continue
        choices = []
        for label in word:
            terms = expansions.get(label)
            if not terms:
                raise ValueError(f"effective token has no expansion: {label}")
            choices.append(tuple(str(base) for base in terms))
        result.extend(tuple(selection) for selection in product(*choices))
    return tuple(dict.fromkeys(result))


def effective_word_actions(
    base_actions: np.ndarray,
    base_words: Sequence[ScalarWord],
    effective_words: Sequence[ScalarWord],
    expansions: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    """Combine registered base-word actions into effective-token actions."""

    actions = np.asarray(base_actions, dtype=np.complex128)
    checked_base = tuple(tuple(word) for word in base_words)
    positions = {word: index for index, word in enumerate(checked_base)}
    if actions.ndim != 2 or actions.shape[1] != len(checked_base):
        raise ValueError("base_actions and base_words have inconsistent shapes")
    columns = []
    for word in effective_words:
        if not word:
            columns.append(actions[:, positions[()]])
            continue
        choices = []
        for label in word:
            terms = expansions.get(label)
            if not terms:
                raise ValueError(f"effective token has no expansion: {label}")
            choices.append(tuple(terms.items()))
        column = np.zeros((len(actions),), dtype=np.complex128)
        for selected in product(*choices):
            base_word = tuple(str(item[0]) for item in selected)
            coefficient = float(np.prod([float(item[1]) for item in selected]))
            try:
                column += coefficient * actions[:, positions[base_word]]
            except KeyError as error:
                raise ValueError(f"required base word is absent: {base_word}") from error
        columns.append(column)
    return np.column_stack(columns)


def centered_scalar_component_matrices(
    raw_actions: np.ndarray,
    words: Sequence[ScalarWord],
    *,
    labels: Sequence[str],
    hamiltonian_label: str | None = None,
    hamiltonian_weights: Sequence[float] | None = None,
    means: Sequence[float],
    scales: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Center generic scalar tokens when one token is the Hamiltonian."""

    actions = np.asarray(raw_actions, dtype=np.complex128)
    checked_words = tuple(tuple(word) for word in words)
    positions = {word: index for index, word in enumerate(checked_words)}
    checked_labels = tuple(str(label) for label in labels)
    if (hamiltonian_label is None) == (hamiltonian_weights is None):
        raise ValueError(
            "provide exactly one of hamiltonian_label or hamiltonian_weights"
        )
    if hamiltonian_label is not None and hamiltonian_label not in checked_labels:
        raise ValueError("hamiltonian_label must be one of the scalar labels")
    if actions.ndim != 2 or actions.shape[1] != len(checked_words):
        raise ValueError("raw_actions and words have inconsistent shapes")
    means_array = np.asarray(means, dtype=np.float64)
    scales_array = np.asarray(scales, dtype=np.float64)
    count = len(checked_labels)
    if means_array.shape != (count,) or scales_array.shape != (count,):
        raise ValueError("means and scales must match the scalar labels")
    if np.any(~np.isfinite(means_array)) or np.any(scales_array <= 0.0):
        raise ValueError("operator standardization is invalid")
    if hamiltonian_weights is None:
        h_weights = np.zeros((count,), dtype=np.float64)
        h_weights[checked_labels.index(str(hamiltonian_label))] = 1.0
    else:
        h_weights = np.asarray(hamiltonian_weights, dtype=np.float64)
        if h_weights.shape != (count,) or np.any(~np.isfinite(h_weights)):
            raise ValueError("hamiltonian_weights must match labels and be finite")

    def raw(word: ScalarWord) -> np.ndarray:
        try:
            return actions[:, positions[word]]
        except KeyError as error:
            raise ValueError(f"required scalar word is absent: {word}") from error

    def hraw(word: ScalarWord) -> np.ndarray:
        result = np.zeros((len(actions),), dtype=np.complex128)
        for coefficient, label in zip(h_weights, checked_labels, strict=True):
            if coefficient != 0.0:
                result += coefficient * raw((label, *word))
        return result

    carrier, h_carrier = raw(()), hraw(())
    components = [carrier]
    h_components = [h_carrier]
    for index, label in enumerate(checked_labels):
        components.append((raw((label,)) - means_array[index] * carrier) / scales_array[index])
        h_components.append((hraw((label,)) - means_array[index] * h_carrier) / scales_array[index])
    for outer_index, outer in enumerate(checked_labels):
        for inner_index, inner in enumerate(checked_labels):
            denominator = scales_array[outer_index] * scales_array[inner_index]
            components.append(
                (
                    raw((outer, inner))
                    - means_array[inner_index] * raw((outer,))
                    - means_array[outer_index] * raw((inner,))
                    + means_array[outer_index] * means_array[inner_index] * carrier
                )
                / denominator
            )
            h_components.append(
                (
                    hraw((outer, inner))
                    - means_array[inner_index] * hraw((outer,))
                    - means_array[outer_index] * hraw((inner,))
                    + means_array[outer_index] * means_array[inner_index] * h_carrier
                )
                / denominator
            )
    return np.column_stack(components), np.column_stack(h_components)


def operator_vit_required_words() -> tuple[ScalarWord, ...]:
    """Return the unique words needed for components and their H action."""

    components = operator_vit_component_words()
    candidates = (
        *components,
        *((TRIANGULAR_A1_SCALAR_LABELS[0], *word) for word in components),
        *((TRIANGULAR_A1_SCALAR_LABELS[1], *word) for word in components),
    )
    return tuple(dict.fromkeys(candidates))


def carrier_standardization(
    raw_actions: np.ndarray,
    words: Sequence[ScalarWord],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Estimate real centering means and response norms from carrier samples."""

    actions = np.asarray(raw_actions, dtype=np.complex128)
    checked_words = tuple(tuple(word) for word in words)
    if actions.ndim != 2 or actions.shape[1] != len(checked_words):
        raise ValueError("raw_actions and words have inconsistent shapes")
    positions = {word: index for index, word in enumerate(checked_words)}
    if len(positions) != len(checked_words) or () not in positions:
        raise ValueError("words must be unique and include the empty word")
    centers = actions[:, positions[()]]
    if np.any(np.abs(centers) <= 1.0e-14):
        raise FloatingPointError("carrier sample contains an amplitude node")
    ratios = np.column_stack(
        [
            actions[:, positions[(label,)]] / centers
            for label in TRIANGULAR_A1_SCALAR_LABELS
        ]
    )
    complex_means = np.mean(ratios, axis=0)
    means = np.real(complex_means)
    scales = np.sqrt(np.mean(np.abs(ratios - means[None, :]) ** 2, axis=0))
    if np.any(~np.isfinite(scales)) or np.any(scales <= 1.0e-12):
        raise FloatingPointError("a sampled scalar token has zero response norm")
    return means, scales, {
        "maximum_abs_imaginary_sample_mean": float(
            np.max(np.abs(np.imag(complex_means)))
        ),
        "minimum_response_scale": float(np.min(scales)),
        "maximum_response_scale": float(np.max(scales)),
    }


def carrier_standardization_for_labels(
    raw_actions: np.ndarray,
    words: Sequence[ScalarWord],
    labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Generic carrier-Born standardization for effective scalar tokens."""

    actions = np.asarray(raw_actions, dtype=np.complex128)
    checked_words = tuple(tuple(word) for word in words)
    checked_labels = tuple(str(label) for label in labels)
    positions = {word: index for index, word in enumerate(checked_words)}
    if actions.ndim != 2 or actions.shape[1] != len(checked_words):
        raise ValueError("raw_actions and words have inconsistent shapes")
    if len(positions) != len(checked_words) or () not in positions:
        raise ValueError("words must be unique and include the empty word")
    if not checked_labels or any((label,) not in positions for label in checked_labels):
        raise ValueError("every scalar label must have a one-token word")
    centers = actions[:, positions[()]]
    if np.any(np.abs(centers) <= 1.0e-14):
        raise FloatingPointError("carrier sample contains an amplitude node")
    ratios = np.column_stack(
        [actions[:, positions[(label,)]] / centers for label in checked_labels]
    )
    complex_means = np.mean(ratios, axis=0)
    means = np.real(complex_means)
    scales = np.sqrt(np.mean(np.abs(ratios - means[None, :]) ** 2, axis=0))
    if np.any(~np.isfinite(scales)) or np.any(scales <= 1.0e-12):
        raise FloatingPointError("a sampled scalar token has zero response norm")
    return means, scales, {
        "maximum_abs_imaginary_sample_mean": float(
            np.max(np.abs(np.imag(complex_means)))
        ),
        "minimum_response_scale": float(np.min(scales)),
        "maximum_response_scale": float(np.max(scales)),
    }


def centered_operator_vit_matrices(
    raw_actions: np.ndarray,
    words: Sequence[ScalarWord],
    *,
    coupling: float,
    means: Sequence[float],
    scales: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build centered component and Hamiltonian-component matrices.

    Column order exactly matches :func:`operator_vit_component_words` and
    :func:`su2_a1_operator_vit_component_weights`.
    """

    actions = np.asarray(raw_actions, dtype=np.complex128)
    checked_words = tuple(tuple(word) for word in words)
    positions = {word: index for index, word in enumerate(checked_words)}
    if actions.ndim != 2 or actions.shape[1] != len(checked_words):
        raise ValueError("raw_actions and words have inconsistent shapes")
    if len(positions) != len(checked_words):
        raise ValueError("words must be unique")
    means_array = np.asarray(means, dtype=np.float64)
    scales_array = np.asarray(scales, dtype=np.float64)
    if means_array.shape != (5,) or scales_array.shape != (5,):
        raise ValueError("means and scales must contain five values")
    if np.any(~np.isfinite(means_array)) or np.any(scales_array <= 0.0):
        raise ValueError("operator standardization is invalid")
    coupling = float(coupling)
    if not np.isfinite(coupling):
        raise ValueError("coupling must be finite")

    def raw(word: ScalarWord) -> np.ndarray:
        try:
            return actions[:, positions[word]]
        except KeyError as error:
            raise ValueError(f"required scalar word is absent: {word}") from error

    def hraw(word: ScalarWord) -> np.ndarray:
        return raw((TRIANGULAR_A1_SCALAR_LABELS[0], *word)) + coupling * raw(
            (TRIANGULAR_A1_SCALAR_LABELS[1], *word)
        )

    carrier = raw(())
    h_carrier = hraw(())
    components = [carrier]
    h_components = [h_carrier]
    for index, label in enumerate(TRIANGULAR_A1_SCALAR_LABELS):
        components.append((raw((label,)) - means_array[index] * carrier) / scales_array[index])
        h_components.append((hraw((label,)) - means_array[index] * h_carrier) / scales_array[index])
    for outer_index, outer in enumerate(TRIANGULAR_A1_SCALAR_LABELS):
        for inner_index, inner in enumerate(TRIANGULAR_A1_SCALAR_LABELS):
            denominator = scales_array[outer_index] * scales_array[inner_index]
            components.append(
                (
                    raw((outer, inner))
                    - means_array[inner_index] * raw((outer,))
                    - means_array[outer_index] * raw((inner,))
                    + means_array[outer_index]
                    * means_array[inner_index]
                    * carrier
                )
                / denominator
            )
            h_components.append(
                (
                    hraw((outer, inner))
                    - means_array[inner_index] * hraw((outer,))
                    - means_array[outer_index] * hraw((inner,))
                    + means_array[outer_index]
                    * means_array[inner_index]
                    * h_carrier
                )
                / denominator
            )
    component_matrix = np.column_stack(components)
    hamiltonian_matrix = np.column_stack(h_components)
    if component_matrix.shape[1] != 31 or np.any(~np.isfinite(component_matrix)):
        raise FloatingPointError("invalid Operator-ViT component matrix")
    if hamiltonian_matrix.shape != component_matrix.shape or np.any(
        ~np.isfinite(hamiltonian_matrix)
    ):
        raise FloatingPointError("invalid Operator-ViT Hamiltonian matrix")
    return component_matrix, hamiltonian_matrix


def sampled_operator_vit_energy(
    params: Mapping[str, object],
    component_matrix: jnp.ndarray,
    hamiltonian_matrix: jnp.ndarray,
    config: SU2A1OperatorViTConfig,
) -> jnp.ndarray:
    """Return the self-normalized carrier-Born sampled energy."""

    weights = su2_a1_operator_vit_component_weights(dict(params), config)
    state = jnp.asarray(component_matrix) @ weights
    action = jnp.asarray(hamiltonian_matrix) @ weights
    norm = jnp.mean(jnp.abs(state) ** 2)
    return jnp.real(jnp.mean(jnp.conj(state) * action) / norm)


def sampled_operator_vit_summary(
    params: Mapping[str, object],
    component_matrix: np.ndarray,
    hamiltonian_matrix: np.ndarray,
    config: SU2A1OperatorViTConfig,
) -> dict[str, float]:
    """Return energy, variance, and carrier-proposal ESS for fixed params."""

    weights = np.asarray(
        su2_a1_operator_vit_component_weights(dict(params), config)
    )
    state = np.asarray(component_matrix) @ weights
    action = np.asarray(hamiltonian_matrix) @ weights
    probability = np.abs(state) ** 2
    norm = float(np.mean(probability))
    energy = complex(np.mean(np.conj(state) * action) / norm)
    variance = float(np.mean(np.abs(action - energy * state) ** 2) / norm)
    normalized = probability / np.sum(probability)
    ess = float(1.0 / np.sum(normalized**2))
    return {
        "energy_total": float(energy.real),
        "variance_total": variance,
        "importance_ess_fraction": ess / len(state),
        "maximum_normalized_weight": float(np.max(normalized)),
        "absolute_imaginary_energy": abs(float(energy.imag)),
    }


__all__ = [
    "carrier_standardization",
    "carrier_standardization_for_labels",
    "centered_operator_vit_matrices",
    "centered_scalar_component_matrices",
    "effective_word_actions",
    "expanded_base_words",
    "krylov_control_basis",
    "operator_vit_component_words",
    "operator_vit_required_words",
    "sampled_operator_vit_energy",
    "sampled_operator_vit_summary",
    "scalar_component_words",
]
