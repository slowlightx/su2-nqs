"""Born-sampled response of physical scalar-operator tangent spaces.

For a real-coefficient variational deformation

``|psi(c)> = (1 + sum_a c_a O_a)|psi>``

the local ratios ``R_a(s) = <s|O_a|psi>/<s|psi>`` determine both the
Fubini--Study metric and the first energy derivative.  This module evaluates
those quantities without constructing a Hilbert-space vector or sparse
operator.  It is therefore a small-system exact oracle and a scalable VMC
diagnostic using precisely the same formula.

The first two registered triangular tokens are the J1 and J2 shells.  Hence
``R_H = R_J1 + (J2/J1) R_J2`` is also the Hamiltonian local energy.  The
remaining tokens test whether symmetry-allowed resonance physics supplies a
descent direction independent of ordinary Lanczos/Krylov improvement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral

import numpy as np

from ..lattices.triangular_hnf import IntMatrix, validate_hnf
from ..operators.triangular_a1_scalar import (
    TRIANGULAR_A1_SCALAR_LABELS,
    triangular_a1_scalar_token_connectivity,
)


BatchAmplitude = Callable[[np.ndarray], np.ndarray]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def triangular_a1_operator_actions(
    amplitude_fn: BatchAmplitude,
    configurations: np.ndarray,
    supercell: IntMatrix,
    *,
    center_batch_size: int = 8,
    amplitude_batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Evaluate all five exact scalar-token actions on supplied states.

    Connected configurations are deduplicated within each center microbatch.
    The routine never forms a fixed-``Sz`` basis and its storage grows only
    with the sampled Hamiltonian/operator neighbourhood.
    """

    matrix = validate_hnf(supercell)
    center_batch_size = _positive_integer(center_batch_size, "center_batch_size")
    amplitude_batch_size = _positive_integer(
        amplitude_batch_size, "amplitude_batch_size"
    )
    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("configurations must be a nonempty rank-two array")
    actions = np.zeros(
        (len(states), len(TRIANGULAR_A1_SCALAR_LABELS)), dtype=np.complex128
    )
    connected_requests = 0
    unique_evaluations = 0
    maximum_union_size = 0
    for start in range(0, len(states), center_batch_size):
        stop = min(len(states), start + center_batch_size)
        records: list[
            list[tuple[tuple[tuple[int, ...], ...], np.ndarray]]
        ] = []
        union: dict[tuple[int, ...], int] = {}
        for state in states[start:stop]:
            tokens = triangular_a1_scalar_token_connectivity(state, matrix)
            row = []
            for label in TRIANGULAR_A1_SCALAR_LABELS:
                token_states, values = tokens[label]
                row.append((token_states, values))
                connected_requests += len(token_states)
                for connected in token_states:
                    if connected not in union:
                        union[connected] = len(union)
            records.append(row)
        ordered = np.asarray(tuple(union), dtype=np.int8)
        values = []
        for amplitude_start in range(0, len(ordered), amplitude_batch_size):
            batch = ordered[
                amplitude_start : amplitude_start + amplitude_batch_size
            ]
            observed = np.asarray(amplitude_fn(batch), dtype=np.complex128)
            if observed.shape != (len(batch),) or np.any(~np.isfinite(observed)):
                raise FloatingPointError(
                    "amplitude_fn returned invalid connected amplitudes"
                )
            values.append(observed)
        connected_amplitudes = np.concatenate(values)
        unique_evaluations += len(ordered)
        maximum_union_size = max(maximum_union_size, len(ordered))
        for local_row, row in enumerate(records):
            for token_index, (token_states, elements) in enumerate(row):
                positions = np.fromiter(
                    (union[connected] for connected in token_states),
                    dtype=np.int64,
                    count=len(token_states),
                )
                actions[start + local_row, token_index] = np.dot(
                    elements, connected_amplitudes[positions]
                )
    if np.any(~np.isfinite(actions)):
        raise FloatingPointError("non-finite scalar-token action")
    return actions, {
        "sample_count": len(states),
        "connected_configuration_requests": connected_requests,
        "unique_connected_amplitude_evaluations": unique_evaluations,
        "deduplication_fraction": 1.0
        - unique_evaluations / max(1, connected_requests),
        "maximum_center_batch_union_size": maximum_union_size,
    }


def triangular_a1_local_ratios(
    amplitude_fn: BatchAmplitude,
    configurations: np.ndarray,
    center_amplitudes: np.ndarray,
    supercell: IntMatrix,
    *,
    center_batch_size: int = 8,
    amplitude_batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Evaluate all five exact scalar-token local ratios on sampled states."""

    centers = np.asarray(center_amplitudes, dtype=np.complex128)
    if centers.shape != (len(configurations),):
        raise ValueError("center_amplitudes must contain one value per state")
    if np.any(~np.isfinite(centers)) or np.any(np.abs(centers) <= 0.0):
        raise FloatingPointError("sampled center amplitudes must be finite and nonzero")
    actions, ledger = triangular_a1_operator_actions(
        amplitude_fn,
        configurations,
        supercell,
        center_batch_size=center_batch_size,
        amplitude_batch_size=amplitude_batch_size,
    )
    ratios = actions / centers[:, None]
    if np.any(~np.isfinite(ratios)):
        raise FloatingPointError("non-finite scalar-token local ratio")
    return ratios, ledger


def triangular_a1_shell_word_actions(
    amplitude_fn: BatchAmplitude,
    configurations: np.ndarray,
    supercell: IntMatrix,
    *,
    amplitude_batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Evaluate ``O_i O_j psi`` for shell ``i=J1,J2`` and all ``j``."""

    matrix = validate_hnf(supercell)
    amplitude_batch_size = _positive_integer(
        amplitude_batch_size, "amplitude_batch_size"
    )
    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("configurations must be a nonempty rank-two array")
    words = np.zeros((len(states), 2, 5), dtype=np.complex128)
    raw_terminal_requests = 0
    unique_terminal_evaluations = 0
    maximum_terminal_union_size = 0
    for row_index, state in enumerate(states):
        outer = triangular_a1_scalar_token_connectivity(state, matrix)
        accumulated = [[{} for _ in range(5)] for _ in range(2)]
        inner_cache: dict[
            tuple[int, ...],
            dict[str, tuple[tuple[tuple[int, ...], ...], np.ndarray]],
        ] = {}
        for shell_index, shell_label in enumerate(
            TRIANGULAR_A1_SCALAR_LABELS[:2]
        ):
            middle_states, outer_elements = outer[shell_label]
            for middle, outer_element in zip(
                middle_states, outer_elements, strict=True
            ):
                inner = inner_cache.get(middle)
                if inner is None:
                    inner = triangular_a1_scalar_token_connectivity(middle, matrix)
                    inner_cache[middle] = inner
                for token_index, token_label in enumerate(
                    TRIANGULAR_A1_SCALAR_LABELS
                ):
                    terminal_states, inner_elements = inner[token_label]
                    target = accumulated[shell_index][token_index]
                    raw_terminal_requests += len(terminal_states)
                    for terminal, inner_element in zip(
                        terminal_states, inner_elements, strict=True
                    ):
                        target[terminal] = target.get(terminal, 0.0) + float(
                            outer_element * inner_element
                        )
        union: dict[tuple[int, ...], int] = {}
        for shell in accumulated:
            for token in shell:
                for terminal, coefficient in token.items():
                    if abs(coefficient) > 1.0e-14 and terminal not in union:
                        union[terminal] = len(union)
        ordered = np.asarray(tuple(union), dtype=np.int8)
        observed = []
        for start in range(0, len(ordered), amplitude_batch_size):
            batch = ordered[start : start + amplitude_batch_size]
            values = np.asarray(amplitude_fn(batch), dtype=np.complex128)
            if values.shape != (len(batch),) or np.any(~np.isfinite(values)):
                raise FloatingPointError(
                    "amplitude_fn returned invalid terminal amplitudes"
                )
            observed.append(values)
        terminal_amplitudes = np.concatenate(observed)
        unique_terminal_evaluations += len(ordered)
        maximum_terminal_union_size = max(
            maximum_terminal_union_size, len(ordered)
        )
        for shell_index, shell in enumerate(accumulated):
            for token_index, token in enumerate(shell):
                words[row_index, shell_index, token_index] = sum(
                    coefficient * terminal_amplitudes[union[terminal]]
                    for terminal, coefficient in token.items()
                    if abs(coefficient) > 1.0e-14
                )
    if np.any(~np.isfinite(words)):
        raise FloatingPointError("non-finite depth-two scalar word action")
    return words, {
        "sample_count": len(states),
        "raw_terminal_configuration_requests": raw_terminal_requests,
        "unique_terminal_amplitude_evaluations": unique_terminal_evaluations,
        "terminal_deduplication_fraction": 1.0
        - unique_terminal_evaluations / max(1, raw_terminal_requests),
        "maximum_single_center_terminal_union_size": maximum_terminal_union_size,
    }


def triangular_a1_word_actions(
    amplitude_fn: BatchAmplitude,
    configurations: np.ndarray,
    supercell: IntMatrix,
    words: Sequence[Sequence[str]],
    *,
    amplitude_batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Evaluate arbitrary ordered scalar-operator words on sampled states.

    A word ``(outer, inner)`` means ``outer @ inner @ |psi>``.  All terminal
    configurations from every requested word are combined before amplitudes
    are evaluated, making depth-three G1 Operator-ViT audits possible without
    constructing a fixed-``Sz`` Hilbert space.
    """

    matrix = validate_hnf(supercell)
    amplitude_batch_size = _positive_integer(
        amplitude_batch_size, "amplitude_batch_size"
    )
    states = np.asarray(configurations, dtype=np.int8)
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("configurations must be a nonempty rank-two array")
    checked_words = tuple(tuple(str(label) for label in word) for word in words)
    if not checked_words:
        raise ValueError("at least one scalar-operator word is required")
    allowed = set(TRIANGULAR_A1_SCALAR_LABELS)
    if any(label not in allowed for word in checked_words for label in word):
        raise ValueError("scalar-operator word contains an unknown token")

    actions = np.zeros((len(states), len(checked_words)), dtype=np.complex128)
    raw_terminal_requests = 0
    unique_terminal_evaluations = 0
    maximum_terminal_union_size = 0
    maximum_word_depth = max(map(len, checked_words))
    for row_index, state_array in enumerate(states):
        center = tuple(int(value) for value in state_array)
        connectivity_cache: dict[
            tuple[int, ...],
            dict[str, tuple[tuple[tuple[int, ...], ...], np.ndarray]],
        ] = {}

        def expand(word: tuple[str, ...]) -> dict[tuple[int, ...], float]:
            frontier = {center: 1.0}
            for label in word:
                following: dict[tuple[int, ...], float] = {}
                for current, coefficient in frontier.items():
                    tokens = connectivity_cache.get(current)
                    if tokens is None:
                        tokens = triangular_a1_scalar_token_connectivity(
                            current, matrix
                        )
                        connectivity_cache[current] = tokens
                    connected, elements = tokens[label]
                    for target, element in zip(connected, elements, strict=True):
                        following[target] = following.get(target, 0.0) + float(
                            coefficient * element
                        )
                frontier = {
                    target: coefficient
                    for target, coefficient in following.items()
                    if abs(coefficient) > 1.0e-14
                }
            return frontier

        expansions = [expand(word) for word in checked_words]
        union: dict[tuple[int, ...], int] = {}
        for expansion in expansions:
            raw_terminal_requests += len(expansion)
            for terminal in expansion:
                if terminal not in union:
                    union[terminal] = len(union)
        ordered = np.asarray(tuple(union), dtype=np.int8)
        observed = []
        for start in range(0, len(ordered), amplitude_batch_size):
            batch = ordered[start : start + amplitude_batch_size]
            values = np.asarray(amplitude_fn(batch), dtype=np.complex128)
            if values.shape != (len(batch),) or np.any(~np.isfinite(values)):
                raise FloatingPointError(
                    "amplitude_fn returned invalid terminal amplitudes"
                )
            observed.append(values)
        terminal_amplitudes = np.concatenate(observed)
        unique_terminal_evaluations += len(ordered)
        maximum_terminal_union_size = max(
            maximum_terminal_union_size, len(ordered)
        )
        for word_index, expansion in enumerate(expansions):
            actions[row_index, word_index] = sum(
                coefficient * terminal_amplitudes[union[terminal]]
                for terminal, coefficient in expansion.items()
            )
    if np.any(~np.isfinite(actions)):
        raise FloatingPointError("non-finite scalar-word action")
    return actions, {
        "sample_count": len(states),
        "word_count": len(checked_words),
        "maximum_word_depth": maximum_word_depth,
        "raw_terminal_configuration_requests": raw_terminal_requests,
        "unique_terminal_amplitude_evaluations": unique_terminal_evaluations,
        "terminal_deduplication_fraction": 1.0
        - unique_terminal_evaluations / max(1, raw_terminal_requests),
        "maximum_single_center_terminal_union_size": maximum_terminal_union_size,
    }


def triangular_a1_shell_word_local_ratios(
    amplitude_fn: BatchAmplitude,
    configurations: np.ndarray,
    center_amplitudes: np.ndarray,
    supercell: IntMatrix,
    *,
    amplitude_batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Evaluate ``(O_i O_j psi)/psi`` for shell ``i=J1,J2`` and all ``j``.

    These ten depth-two words are the minimum information needed to compare a
    one-step Hamiltonian Krylov space with the one-layer five-token physical
    operator space.  Terminal configurations are deduplicated separately for
    each sampled center to keep peak memory bounded at ``N=36``.
    """

    centers = np.asarray(center_amplitudes, dtype=np.complex128)
    if centers.shape != (len(configurations),) or np.any(np.abs(centers) <= 0.0):
        raise ValueError("center_amplitudes must be nonzero and match states")
    actions, ledger = triangular_a1_shell_word_actions(
        amplitude_fn,
        configurations,
        supercell,
        amplitude_batch_size=amplitude_batch_size,
    )
    words = actions / centers[:, None, None]
    if np.any(~np.isfinite(words)):
        raise FloatingPointError("non-finite depth-two scalar word ratio")
    return words, ledger


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, values, axes=(0, 0))


def _subspace_response(
    metric: np.ndarray,
    force: np.ndarray,
    indices: Sequence[int],
    *,
    relative_cutoff: float,
) -> dict[str, object]:
    selected = np.asarray(tuple(int(index) for index in indices), dtype=np.int64)
    local_metric = metric[np.ix_(selected, selected)]
    local_force = force[selected]
    eigenvalues, eigenvectors = np.linalg.eigh(local_metric)
    cutoff = max(1.0e-15, relative_cutoff * max(1.0, float(eigenvalues[-1])))
    keep = eigenvalues > cutoff
    rank = int(np.count_nonzero(keep))
    inverse = np.zeros_like(eigenvalues)
    inverse[keep] = 1.0 / eigenvalues[keep]
    coefficients = -(eigenvectors * inverse[None, :]) @ (
        eigenvectors.T @ local_force
    )
    squared_slope = float(max(0.0, -np.dot(local_force, coefficients)))
    condition = (
        float(eigenvalues[keep][-1] / eigenvalues[keep][0]) if rank else None
    )
    return {
        "indices": selected.tolist(),
        "rank": rank,
        "condition_number": condition,
        "natural_descent_coefficients": coefficients.tolist(),
        "squared_energy_slope_per_unit_fs_norm": squared_slope,
        "energy_slope_per_unit_fs_norm": -float(np.sqrt(squared_slope)),
    }


def scalar_operator_tangent_statistics(
    local_ratios: np.ndarray,
    couplings: Sequence[float],
    *,
    weights: np.ndarray | None = None,
    relative_cutoff: float = 1.0e-10,
) -> dict[str, object]:
    """Return metric and energy-response diagnostics for scalar tokens.

    ``weights=None`` means an ordinary Monte Carlo average.  Supplying exact
    normalized Born weights turns the same routine into an exhaustive oracle.
    Reported slopes are derivatives per unit Fubini--Study tangent norm, not
    finite-step energy predictions.
    """

    ratios = np.asarray(local_ratios, dtype=np.complex128)
    if ratios.ndim != 2 or ratios.shape[1] != len(TRIANGULAR_A1_SCALAR_LABELS):
        raise ValueError("local_ratios must have shape (samples, 5)")
    if len(ratios) < 2 or np.any(~np.isfinite(ratios)):
        raise ValueError("at least two finite local-ratio rows are required")
    checked_couplings = np.asarray(tuple(couplings), dtype=np.float64)
    if (
        checked_couplings.ndim != 1
        or len(checked_couplings) < 1
        or np.any(~np.isfinite(checked_couplings))
    ):
        raise ValueError("couplings must be a nonempty finite sequence")
    if not np.isfinite(relative_cutoff) or relative_cutoff <= 0.0:
        raise ValueError("relative_cutoff must be positive and finite")
    if weights is None:
        probability = np.full((len(ratios),), 1.0 / len(ratios))
    else:
        probability = np.asarray(weights, dtype=np.float64)
        if (
            probability.shape != (len(ratios),)
            or np.any(~np.isfinite(probability))
            or np.any(probability < 0.0)
            or float(np.sum(probability)) <= 0.0
        ):
            raise ValueError("weights must be finite nonnegative sample weights")
        probability = probability / np.sum(probability)

    means = _weighted_mean(ratios, probability)
    centered = ratios - means[None, :]
    metric = np.real(centered.conj().T @ (probability[:, None] * centered))
    metric = (metric + metric.T) / 2.0
    rows = []
    for coupling in checked_couplings:
        hamiltonian_direction = np.asarray(
            (1.0, float(coupling), 0.0, 0.0, 0.0), dtype=np.float64
        )
        local_energy = ratios @ hamiltonian_direction
        mean_energy = complex(_weighted_mean(local_energy, probability))
        centered_energy = local_energy - mean_energy
        force = 2.0 * np.real(
            centered.conj().T @ (probability * centered_energy)
        )
        h_metric = float(hamiltonian_direction @ metric @ hamiltonian_direction)
        h_force = float(np.dot(hamiltonian_direction, force))
        h_squared_slope = 0.0 if h_metric <= 1.0e-15 else h_force**2 / h_metric
        hamiltonian = {
            "rank": int(h_metric > 1.0e-15),
            "squared_energy_slope_per_unit_fs_norm": h_squared_slope,
            "energy_slope_per_unit_fs_norm": -float(np.sqrt(h_squared_slope)),
            "energy_force_along_positive_h": h_force,
            "fs_metric_along_h": h_metric,
        }
        shell = _subspace_response(
            metric, force, (0, 1), relative_cutoff=relative_cutoff
        )
        resonance = _subspace_response(
            metric, force, (2, 3, 4), relative_cutoff=relative_cutoff
        )
        full = _subspace_response(
            metric, force, range(5), relative_cutoff=relative_cutoff
        )
        full_squared = float(full["squared_energy_slope_per_unit_fs_norm"])
        shell_squared = float(shell["squared_energy_slope_per_unit_fs_norm"])
        rows.append(
            {
                "j2_over_j1": float(coupling),
                "mean_energy_total": float(mean_energy.real),
                "maximum_abs_imaginary_local_energy": float(
                    np.max(np.abs(local_energy.imag))
                ),
                "variance_total": float(
                    _weighted_mean(np.abs(centered_energy) ** 2, probability).real
                ),
                "energy_force": force.tolist(),
                "hamiltonian_lanczos_direction": hamiltonian,
                "two_shell_subspace": shell,
                "resonance_only_subspace": resonance,
                "full_five_token_subspace": full,
                "full_to_hamiltonian_squared_slope_ratio": (
                    full_squared / h_squared_slope
                    if h_squared_slope > 1.0e-15
                    else None
                ),
                "incremental_squared_slope_fraction_beyond_two_shells": (
                    max(0.0, full_squared - shell_squared) / full_squared
                    if full_squared > 1.0e-15
                    else 0.0
                ),
            }
        )
    eigenvalues = np.linalg.eigvalsh(metric)
    return {
        "token_labels": list(TRIANGULAR_A1_SCALAR_LABELS),
        "sample_count": len(ratios),
        "weighted": weights is not None,
        "local_ratio_means": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in means
        ],
        "fubini_study_metric": metric.tolist(),
        "metric_eigenvalues": eigenvalues.tolist(),
        "couplings": rows,
        "interpretation": (
            "instantaneous physical-operator tangent response; not a finite-step "
            "trained variational energy"
        ),
    }


def _sampled_generalized_ritz(
    components: np.ndarray,
    hamiltonian_components: np.ndarray,
    weights: np.ndarray,
    *,
    relative_cutoff: float,
) -> dict[str, object]:
    raw_gram = components.conj().T @ (weights[:, None] * components)
    raw_hamiltonian = components.conj().T @ (
        weights[:, None] * hamiltonian_components
    )
    raw_gram = (raw_gram + raw_gram.conj().T) / 2.0
    raw_hamiltonian = (raw_hamiltonian + raw_hamiltonian.conj().T) / 2.0
    scales = np.sqrt(np.maximum(np.real(np.diag(raw_gram)), 1.0e-300))
    gram = raw_gram / (scales[:, None] * scales[None, :])
    hamiltonian = raw_hamiltonian / (scales[:, None] * scales[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    cutoff = max(1.0e-14, relative_cutoff * float(eigenvalues[-1]))
    keep = eigenvalues > cutoff
    if not np.any(keep):
        raise FloatingPointError("sampled component Gram matrix has zero rank")
    whitener = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    projected = whitener.conj().T @ hamiltonian @ whitener
    energies, vectors = np.linalg.eigh((projected + projected.conj().T) / 2.0)
    scaled_coefficients = whitener @ vectors[:, 0]
    coefficients = scaled_coefficients / scales
    state_ratio = components @ coefficients
    action_ratio = hamiltonian_components @ coefficients
    norm = float(np.sum(weights * np.abs(state_ratio) ** 2).real)
    energy = float(energies[0].real)
    variance = float(
        np.sum(weights * np.abs(action_ratio - energy * state_ratio) ** 2).real
        / norm
    )
    return {
        "energy_total": energy,
        "variance_total": variance,
        "gram_rank": int(np.count_nonzero(keep)),
        "gram_condition_number": float(eigenvalues[keep][-1] / eigenvalues[keep][0]),
        "gram_eigenvalues": eigenvalues.tolist(),
        "component_norm_scales": scales.tolist(),
        "coefficients_real": np.real(coefficients).tolist(),
        "coefficients_imag": np.imag(coefficients).tolist(),
    }


def _fixed_coefficient_audit(
    components: np.ndarray,
    hamiltonian_components: np.ndarray,
    coefficients: np.ndarray,
) -> dict[str, float]:
    state = components @ coefficients
    action = hamiltonian_components @ coefficients
    norm = float(np.mean(np.abs(state) ** 2).real)
    if not np.isfinite(norm) or norm <= 1.0e-15:
        raise FloatingPointError("held-out Ritz state has zero or nonfinite norm")
    energy = complex(np.mean(np.conj(state) * action) / norm)
    variance = float(np.mean(np.abs(action - energy * state) ** 2).real / norm)
    return {
        "energy_total": float(energy.real),
        "variance_total": variance,
        "maximum_abs_imaginary_energy": abs(float(energy.imag)),
    }


def _ritz_matrices(
    ratios: np.ndarray, words: np.ndarray, coupling: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    one = np.ones((len(ratios),), dtype=np.complex128)
    local_h = ratios[:, 0] + coupling * ratios[:, 1]
    h_on_tokens = words[:, 0, :] + coupling * words[:, 1, :]
    local_h2 = (
        words[:, 0, 0]
        + coupling * (words[:, 0, 1] + words[:, 1, 0])
        + coupling**2 * words[:, 1, 1]
    )
    return (
        np.column_stack((one, local_h)),
        np.column_stack((local_h, local_h2)),
        np.column_stack((one, ratios)),
        np.column_stack((local_h, h_on_tokens)),
    )


def sampled_one_layer_operator_ritz(
    local_ratios: np.ndarray,
    shell_word_ratios: np.ndarray,
    couplings: Sequence[float],
    *,
    weights: np.ndarray | None = None,
    relative_cutoff: float = 1.0e-10,
) -> dict[str, object]:
    """Compare one Hamiltonian Lanczos step to one five-token operator layer.

    Both Ritz problems use the identical Born samples and Hamiltonian matrix
    elements.  The comparison is finite-step and variational in the sampled
    generalized eigenproblem, unlike an instantaneous gradient diagnostic.
    """

    ratios = np.asarray(local_ratios, dtype=np.complex128)
    words = np.asarray(shell_word_ratios, dtype=np.complex128)
    if ratios.ndim != 2 or ratios.shape[1] != 5:
        raise ValueError("local_ratios must have shape (samples, 5)")
    if words.shape != (len(ratios), 2, 5):
        raise ValueError("shell_word_ratios must have shape (samples, 2, 5)")
    if np.any(~np.isfinite(ratios)) or np.any(~np.isfinite(words)):
        raise ValueError("local and word ratios must be finite")
    if weights is None:
        probability = np.full((len(ratios),), 1.0 / len(ratios))
    else:
        probability = np.asarray(weights, dtype=np.float64)
        if probability.shape != (len(ratios),) or np.any(probability < 0.0):
            raise ValueError("weights must be nonnegative and match samples")
        probability = probability / np.sum(probability)
    rows = []
    for coupling in couplings:
        coupling = float(coupling)
        krylov_components, krylov_action, operator_components, operator_action = (
            _ritz_matrices(ratios, words, coupling)
        )
        krylov = _sampled_generalized_ritz(
            krylov_components,
            krylov_action,
            probability,
            relative_cutoff=relative_cutoff,
        )
        operator = _sampled_generalized_ritz(
            operator_components,
            operator_action,
            probability,
            relative_cutoff=relative_cutoff,
        )
        rows.append(
            {
                "j2_over_j1": coupling,
                "carrier_energy_total": float(
                    np.sum(probability * krylov_action[:, 0]).real
                ),
                "one_lanczos_krylov": krylov,
                "one_layer_five_token_operator": operator,
                "operator_minus_krylov_energy_total": float(
                    operator["energy_total"] - krylov["energy_total"]
                ),
            }
        )
    return {
        "token_labels": list(TRIANGULAR_A1_SCALAR_LABELS),
        "sample_count": len(ratios),
        "couplings": rows,
        "comparison": (
            "finite sampled Ritz: span{psi,O_a psi} versus span{psi,H psi}"
        ),
    }


def sampled_resonance_channel_ritz(
    local_ratios: np.ndarray,
    shell_word_ratios: np.ndarray,
    couplings: Sequence[float],
    *,
    weights: np.ndarray | None = None,
    relative_cutoff: float = 1.0e-10,
) -> dict[str, object]:
    """Compare one Lanczos step with each single registered resonance token."""

    ratios = np.asarray(local_ratios, dtype=np.complex128)
    words = np.asarray(shell_word_ratios, dtype=np.complex128)
    if ratios.ndim != 2 or ratios.shape[1] != 5:
        raise ValueError("local_ratios must have shape (samples, 5)")
    if words.shape != (len(ratios), 2, 5):
        raise ValueError("shell_word_ratios must have shape (samples, 2, 5)")
    if weights is None:
        probability = np.full((len(ratios),), 1.0 / len(ratios))
    else:
        probability = np.asarray(weights, dtype=np.float64)
        if probability.shape != (len(ratios),) or np.any(probability < 0.0):
            raise ValueError("weights must be nonnegative and match samples")
        probability = probability / np.sum(probability)
    rows = []
    for coupling_value in couplings:
        coupling = float(coupling_value)
        krylov_components, krylov_action, _, operator_action = _ritz_matrices(
            ratios, words, coupling
        )
        baseline = _sampled_generalized_ritz(
            krylov_components,
            krylov_action,
            probability,
            relative_cutoff=relative_cutoff,
        )
        channels = []
        for token_index in range(2, 5):
            components = np.column_stack(
                (krylov_components, ratios[:, token_index])
            )
            actions = np.column_stack(
                (krylov_action, operator_action[:, token_index + 1])
            )
            augmented = _sampled_generalized_ritz(
                components,
                actions,
                probability,
                relative_cutoff=relative_cutoff,
            )
            channels.append(
                {
                    "token_index": token_index,
                    "token_label": TRIANGULAR_A1_SCALAR_LABELS[token_index],
                    "augmented_ritz": augmented,
                    "augmented_minus_krylov_energy_total": float(
                        augmented["energy_total"] - baseline["energy_total"]
                    ),
                }
            )
        rows.append(
            {
                "j2_over_j1": coupling,
                "one_lanczos_krylov": baseline,
                "resonance_channels": channels,
            }
        )
    return {"couplings": rows, "selection": "all three channels reported"}


def cross_validated_resonance_channel_ritz(
    local_ratio_replicates: Sequence[np.ndarray],
    shell_word_replicates: Sequence[np.ndarray],
    couplings: Sequence[float],
    *,
    relative_cutoff: float = 1.0e-10,
) -> dict[str, object]:
    """Leave-one-replicate-out audit for every single resonance channel."""

    ratios = tuple(np.asarray(value, dtype=np.complex128) for value in local_ratio_replicates)
    words = tuple(np.asarray(value, dtype=np.complex128) for value in shell_word_replicates)
    if len(ratios) < 2 or len(ratios) != len(words):
        raise ValueError("matching data from at least two replicates required")
    rows = []
    for coupling_value in couplings:
        coupling = float(coupling_value)
        channel_folds = {index: [] for index in range(2, 5)}
        for held_out in range(len(ratios)):
            train_ratios = np.concatenate(
                [value for index, value in enumerate(ratios) if index != held_out]
            )
            train_words = np.concatenate(
                [value for index, value in enumerate(words) if index != held_out]
            )
            trained = sampled_resonance_channel_ritz(
                train_ratios,
                train_words,
                (coupling,),
                relative_cutoff=relative_cutoff,
            )["couplings"][0]
            audit_matrices = _ritz_matrices(
                ratios[held_out], words[held_out], coupling
            )
            baseline_coefficients = np.asarray(
                trained["one_lanczos_krylov"]["coefficients_real"]
            ) + 1j * np.asarray(
                trained["one_lanczos_krylov"]["coefficients_imag"]
            )
            baseline = _fixed_coefficient_audit(
                audit_matrices[0], audit_matrices[1], baseline_coefficients
            )
            for channel in trained["resonance_channels"]:
                token_index = int(channel["token_index"])
                coefficients = np.asarray(
                    channel["augmented_ritz"]["coefficients_real"]
                ) + 1j * np.asarray(
                    channel["augmented_ritz"]["coefficients_imag"]
                )
                components = np.column_stack(
                    (audit_matrices[0], ratios[held_out][:, token_index])
                )
                actions = np.column_stack(
                    (audit_matrices[1], audit_matrices[3][:, token_index + 1])
                )
                augmented = _fixed_coefficient_audit(
                    components, actions, coefficients
                )
                channel_folds[token_index].append(
                    {
                        "held_out_replicate": held_out,
                        "one_lanczos_krylov": baseline,
                        "augmented_ritz": augmented,
                        "augmented_minus_krylov_energy_total": float(
                            augmented["energy_total"] - baseline["energy_total"]
                        ),
                    }
                )
        rows.append(
            {
                "j2_over_j1": coupling,
                "resonance_channels": [
                    {
                        "token_index": token_index,
                        "token_label": TRIANGULAR_A1_SCALAR_LABELS[token_index],
                        "folds": channel_folds[token_index],
                    }
                    for token_index in range(2, 5)
                ],
            }
        )
    return {
        "couplings": rows,
        "protocol": "leave-one-independent-replicate-out single-channel audit",
    }


def one_lanczos_and_resonance_state_actions(
    carrier_amplitudes: np.ndarray,
    token_actions: np.ndarray,
    shell_word_actions: np.ndarray,
    coupling: float,
    token_index: int,
    krylov_coefficients: np.ndarray,
    resonance_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one-Lanczos and one-resonance states and their H actions.

    The returned arrays are ``(psi_K, H psi_K, psi_O, H psi_O)`` on the
    supplied configurations.  Inputs are un-divided operator actions, so the
    routine remains valid on exact nodes of the carrier.
    """

    carrier = np.asarray(carrier_amplitudes, dtype=np.complex128)
    actions = np.asarray(token_actions, dtype=np.complex128)
    words = np.asarray(shell_word_actions, dtype=np.complex128)
    krylov = np.asarray(krylov_coefficients, dtype=np.complex128)
    resonance = np.asarray(resonance_coefficients, dtype=np.complex128)
    if carrier.ndim != 1 or actions.shape != (len(carrier), 5):
        raise ValueError("token_actions must have shape (samples, 5)")
    if words.shape != (len(carrier), 2, 5):
        raise ValueError("shell_word_actions must have shape (samples, 2, 5)")
    if token_index not in range(2, 5):
        raise ValueError("token_index must identify a registered resonance")
    if krylov.shape != (2,) or resonance.shape != (3,):
        raise ValueError("coefficient shapes must be (2,) and (3,)")
    coupling = float(coupling)
    h_carrier = actions[:, 0] + coupling * actions[:, 1]
    h2_carrier = (
        words[:, 0, 0]
        + coupling * (words[:, 0, 1] + words[:, 1, 0])
        + coupling**2 * words[:, 1, 1]
    )
    h_token = words[:, 0, token_index] + coupling * words[:, 1, token_index]
    krylov_state = krylov[0] * carrier + krylov[1] * h_carrier
    krylov_action = krylov[0] * h_carrier + krylov[1] * h2_carrier
    resonance_state = (
        resonance[0] * carrier
        + resonance[1] * h_carrier
        + resonance[2] * actions[:, token_index]
    )
    resonance_action = (
        resonance[0] * h_carrier
        + resonance[1] * h2_carrier
        + resonance[2] * h_token
    )
    return krylov_state, krylov_action, resonance_state, resonance_action


def cross_validated_one_layer_operator_ritz(
    local_ratio_replicates: Sequence[np.ndarray],
    shell_word_replicates: Sequence[np.ndarray],
    couplings: Sequence[float],
    *,
    relative_cutoff: float = 1.0e-10,
) -> dict[str, object]:
    """Fit Ritz coefficients on all other replicates and audit held-out energy."""

    ratios = tuple(np.asarray(value, dtype=np.complex128) for value in local_ratio_replicates)
    words = tuple(np.asarray(value, dtype=np.complex128) for value in shell_word_replicates)
    if len(ratios) < 2 or len(ratios) != len(words):
        raise ValueError("matching ratio/word data from at least two replicates required")
    for local, word in zip(ratios, words, strict=True):
        if local.ndim != 2 or local.shape[1] != 5 or word.shape != (len(local), 2, 5):
            raise ValueError("invalid replicate local-ratio or word shape")
    rows = []
    for coupling_value in couplings:
        coupling = float(coupling_value)
        folds = []
        for held_out in range(len(ratios)):
            train_ratios = np.concatenate(
                [value for index, value in enumerate(ratios) if index != held_out]
            )
            train_words = np.concatenate(
                [value for index, value in enumerate(words) if index != held_out]
            )
            trained = sampled_one_layer_operator_ritz(
                train_ratios,
                train_words,
                (coupling,),
                relative_cutoff=relative_cutoff,
            )["couplings"][0]
            matrices = _ritz_matrices(ratios[held_out], words[held_out], coupling)
            krylov_coefficients = np.asarray(
                trained["one_lanczos_krylov"]["coefficients_real"]
            ) + 1j * np.asarray(
                trained["one_lanczos_krylov"]["coefficients_imag"]
            )
            operator_coefficients = np.asarray(
                trained["one_layer_five_token_operator"]["coefficients_real"]
            ) + 1j * np.asarray(
                trained["one_layer_five_token_operator"]["coefficients_imag"]
            )
            krylov_audit = _fixed_coefficient_audit(
                matrices[0], matrices[1], krylov_coefficients
            )
            operator_audit = _fixed_coefficient_audit(
                matrices[2], matrices[3], operator_coefficients
            )
            folds.append(
                {
                    "held_out_replicate": held_out,
                    "training_sample_count": len(train_ratios),
                    "audit_sample_count": len(ratios[held_out]),
                    "one_lanczos_krylov": krylov_audit,
                    "one_layer_five_token_operator": operator_audit,
                    "operator_minus_krylov_energy_total": (
                        operator_audit["energy_total"] - krylov_audit["energy_total"]
                    ),
                }
            )
        rows.append({"j2_over_j1": coupling, "folds": folds})
    return {
        "couplings": rows,
        "protocol": "leave-one-independent-replicate-out coefficient audit",
    }


__all__ = [
    "cross_validated_one_layer_operator_ritz",
    "cross_validated_resonance_channel_ritz",
    "one_lanczos_and_resonance_state_actions",
    "sampled_resonance_channel_ritz",
    "sampled_one_layer_operator_ritz",
    "scalar_operator_tangent_statistics",
    "triangular_a1_word_actions",
    "triangular_a1_local_ratios",
    "triangular_a1_operator_actions",
    "triangular_a1_shell_word_actions",
    "triangular_a1_shell_word_local_ratios",
]
