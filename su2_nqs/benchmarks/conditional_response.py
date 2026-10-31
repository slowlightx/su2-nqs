"""Model-independent conditional response to a physical Hamiltonian layer.

The useful autoregressive trainability proxy is not the entropy of uniformly
enumerated fusion paths.  Given two normalized Born distributions on the same
exact action strings, the chain rule resolves ``KL(target || base)`` into one
non-negative contribution per autoregressive decision.  For a target made by
one or more local Hamiltonian actions, this locates where the physical update
changes conditional probabilities without referring to neural parameters.

This is a probability-space diagnostic.  Quantum phase response is reported
separately by :func:`complex_response_metrics`; conditional KL must never be
interpreted as the complete wavefunction distance.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def _probability_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    answer = np.asarray(values, dtype=np.float64)
    if answer.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if np.any(~np.isfinite(answer)) or np.any(answer < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    total = float(np.sum(answer))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return answer / total


def _action_array(
    action_sequences: Sequence[Sequence[int]] | np.ndarray,
    size: int,
) -> np.ndarray:
    actions = np.asarray(action_sequences)
    if actions.ndim != 2 or len(actions) != size:
        raise ValueError("action_sequences must have one row per probability")
    if not np.issubdtype(actions.dtype, np.integer):
        raise TypeError("autoregressive actions must be integers")
    if np.any(actions < 0):
        raise ValueError("autoregressive action labels must be non-negative")
    keys = [tuple(int(value) for value in row) for row in actions]
    if len(set(keys)) != len(keys):
        raise ValueError("complete action sequences must identify states uniquely")
    return actions.astype(np.int32, copy=False)


def _safe_kl_bits(target: np.ndarray, reference: np.ndarray) -> float:
    active = target > 0.0
    if np.any(reference[active] <= 0.0):
        return float("inf")
    return float(
        np.sum(target[active] * (np.log2(target[active]) - np.log2(reference[active])))
    )


def conditional_kl_decomposition(
    reference_probabilities: Sequence[float] | np.ndarray,
    target_probabilities: Sequence[float] | np.ndarray,
    action_sequences: Sequence[Sequence[int]] | np.ndarray,
) -> dict[str, object]:
    """Resolve ``KL(target || reference)`` over an exact AR filtration.

    Rows of ``action_sequences`` must be unique complete strings.  At decision
    ``d`` the contribution is

    ``sum_prefix q(prefix) KL[q(a|prefix) || p(a|prefix)]``.

    Logarithms are base two, so all KL and entropy fields are measured in bits.
    A target action with zero reference conditional probability produces an
    infinite contribution rather than an epsilon-smoothed answer.
    """

    reference = _probability_vector(reference_probabilities, "reference_probabilities")
    target = _probability_vector(target_probabilities, "target_probabilities")
    if reference.shape != target.shape:
        raise ValueError("reference and target probabilities must share shape")
    actions = _action_array(action_sequences, len(reference))

    rows: list[dict[str, float]] = []
    for decision in range(actions.shape[1]):
        groups: dict[tuple[int, ...], dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(lambda: [0.0, 0.0])
        )
        for p_value, q_value, action_row in zip(
            reference, target, actions, strict=True
        ):
            prefix = tuple(int(value) for value in action_row[:decision])
            action = int(action_row[decision])
            groups[prefix][action][0] += float(p_value)
            groups[prefix][action][1] += float(q_value)

        contribution = 0.0
        target_entropy = 0.0
        target_cross_entropy = 0.0
        mean_reachable = 0.0
        maximum_reachable = 0
        target_prefix_count = 0
        infinite = False
        for by_action in groups.values():
            p_prefix = float(sum(value[0] for value in by_action.values()))
            q_prefix = float(sum(value[1] for value in by_action.values()))
            if q_prefix <= 0.0:
                continue
            target_prefix_count += 1
            reachable = sum(value[0] > 0.0 for value in by_action.values())
            mean_reachable += q_prefix * reachable
            maximum_reachable = max(maximum_reachable, reachable)
            for p_joint, q_joint in by_action.values():
                if q_joint <= 0.0:
                    continue
                q_conditional = q_joint / q_prefix
                target_entropy -= q_joint * np.log2(q_conditional)
                if p_prefix <= 0.0 or p_joint <= 0.0:
                    infinite = True
                    target_cross_entropy = float("inf")
                    continue
                p_conditional = p_joint / p_prefix
                contribution += q_joint * np.log2(q_conditional / p_conditional)
                if np.isfinite(target_cross_entropy):
                    target_cross_entropy -= q_joint * np.log2(p_conditional)
        if infinite:
            contribution = float("inf")
        rows.append(
            {
                "decision": float(decision),
                "kl_contribution_bits": float(contribution),
                "target_conditional_entropy_bits": float(target_entropy),
                "target_cross_entropy_bits": float(target_cross_entropy),
                "target_mean_reference_reachable_actions": float(mean_reachable),
                "maximum_reference_reachable_actions": float(maximum_reachable),
                "positive_target_prefix_count": float(target_prefix_count),
            }
        )

    full_kl = _safe_kl_bits(target, reference)
    chain_kl = float(sum(row["kl_contribution_bits"] for row in rows))
    if np.isinf(full_kl) and np.isinf(chain_kl):
        chain_rule_error = 0.0
    else:
        chain_rule_error = float(abs(chain_kl - full_kl))
    target_entropy = float(-np.sum(target[target > 0.0] * np.log2(target[target > 0.0])))
    conditional_entropy = float(
        sum(row["target_conditional_entropy_bits"] for row in rows)
    )
    cumulative = 0.0
    for row in rows:
        cumulative += row["kl_contribution_bits"]
        row["cumulative_kl_bits"] = float(cumulative)
        row["kl_fraction"] = (
            float(row["kl_contribution_bits"] / full_kl)
            if np.isfinite(full_kl) and full_kl > 0.0
            else 0.0
        )
    return {
        "kl_target_reference_bits": full_kl,
        "chain_kl_bits": chain_kl,
        "chain_rule_error": chain_rule_error,
        "target_entropy_bits": target_entropy,
        "conditional_entropy_sum_bits": conditional_entropy,
        "entropy_chain_rule_error": float(abs(target_entropy - conditional_entropy)),
        "decisions": rows,
    }


def complex_response_metrics(
    reference: Sequence[complex] | np.ndarray,
    target: Sequence[complex] | np.ndarray,
) -> dict[str, float]:
    """Separate Born-amplitude and phase response of two state vectors."""

    initial = np.asarray(reference, dtype=np.complex128)
    changed = np.asarray(target, dtype=np.complex128)
    if initial.ndim != 1 or changed.shape != initial.shape:
        raise ValueError("reference and target must be equal-length vectors")
    if np.any(~np.isfinite(initial)) or np.any(~np.isfinite(changed)):
        raise ValueError("state amplitudes must be finite")
    norm_initial = float(np.linalg.norm(initial))
    norm_changed = float(np.linalg.norm(changed))
    if norm_initial <= 0.0 or norm_changed <= 0.0:
        raise ValueError("state vectors must have positive norm")
    initial = initial / norm_initial
    changed = changed / norm_changed
    p = np.abs(initial) ** 2
    q = np.abs(changed) ** 2
    overlap_squared = float(abs(np.vdot(initial, changed)) ** 2)
    classical_amplitude = float(np.sum(np.sqrt(p * q)))
    amplitude_fidelity_squared = classical_amplitude**2
    support_leakage = float(np.sum(q[p == 0.0]))
    if support_leakage > 0.0:
        second_moment = float("inf")
    else:
        second_moment = float(
            np.sum(np.divide(q**2, p, out=np.zeros_like(q), where=p > 0.0))
        )
    return {
        "overlap_squared": overlap_squared,
        "amplitude_fidelity_squared": amplitude_fidelity_squared,
        "conditional_phase_coherence": (
            overlap_squared / amplitude_fidelity_squared
            if amplitude_fidelity_squared > 0.0
            else 0.0
        ),
        "probability_kl_target_reference_bits": _safe_kl_bits(q, p),
        "probability_kl_reference_target_bits": _safe_kl_bits(p, q),
        "born_reweighting_ess_fraction": (
            1.0 / second_moment if np.isfinite(second_moment) and second_moment > 0.0 else 0.0
        ),
        "target_probability_outside_reference_support": support_leakage,
    }


def complex_martingale_decomposition(
    reference: Sequence[complex] | np.ndarray,
    target: Sequence[complex] | np.ndarray,
    action_sequences: Sequence[Sequence[int]] | np.ndarray,
) -> dict[str, object]:
    """Decompose the full complex response by autoregressive filtration decision.

    Let ``p(x)=|reference(x)|**2`` and ``R(x)=target(x)/reference(x)``.
    Conditional expectations of ``R`` along the autoregressive filtration
    form a complex Doob martingale.  Its increments are mutually orthogonal in
    the conventional complex inner product
    ``<f,g>_p = sum_x p(x) conj(f(x)) g(x)`` and reconstruct the
    *wavefunction*, including phase, rather than only its Born probabilities::

        target = reference * (overlap + sum_d increment_d).

    Here ``overlap = <reference|target>`` (not its complex conjugate).  This
    decomposition has no arbitrary per-decision phase gauge.  It does require
    target support to be contained *exactly* in reference support, as is true
    for the normalized full-support NQS benchmarks.

    Although the increment at decision ``d`` is prefix-measurable, its table
    is computed by contracting the complete target over every compatible
    suffix.  It is therefore a suffix-dependent exhaustive oracle, not a
    revealed-prefix production feature or a scalable training representation.
    """

    initial = np.asarray(reference, dtype=np.complex128)
    changed = np.asarray(target, dtype=np.complex128)
    if initial.ndim != 1 or changed.shape != initial.shape:
        raise ValueError("reference and target must be equal-length vectors")
    if np.any(~np.isfinite(initial)) or np.any(~np.isfinite(changed)):
        raise ValueError("state amplitudes must be finite")
    norm_initial = float(np.linalg.norm(initial))
    norm_changed = float(np.linalg.norm(changed))
    if norm_initial <= 0.0 or norm_changed <= 0.0:
        raise ValueError("state vectors must have positive norm")
    initial = initial / norm_initial
    changed = changed / norm_changed
    actions = _action_array(action_sequences, len(initial))

    probabilities = np.abs(initial) ** 2
    active = probabilities > 0.0
    leakage = float(np.sum(np.abs(changed[~active]) ** 2))
    if leakage > 0.0:
        raise ValueError(
            "target support must be contained in reference support for a "
            "wavefunction-ratio martingale"
        )
    ratio = np.divide(
        changed,
        initial,
        out=np.zeros_like(changed),
        where=active,
    )
    overlap = complex(np.sum(probabilities * ratio))
    previous = np.full(len(initial), overlap, dtype=np.complex128)
    increments = np.zeros((actions.shape[1], len(initial)), dtype=np.complex128)
    rows: list[dict[str, float]] = []
    total_response = max(1.0 - abs(overlap) ** 2, 0.0)

    for decision in range(actions.shape[1]):
        masses: dict[tuple[int, ...], float] = defaultdict(float)
        numerators: dict[tuple[int, ...], complex] = defaultdict(complex)
        prefix_length = decision + 1
        for probability, value, action_row in zip(
            probabilities, ratio, actions, strict=True
        ):
            if probability <= 0.0:
                continue
            key = tuple(int(action) for action in action_row[:prefix_length])
            masses[key] += float(probability)
            numerators[key] += probability * value
        means = {
            key: numerators[key] / mass
            for key, mass in masses.items()
            if mass > 0.0
        }
        current = np.asarray(
            [
                means.get(
                    tuple(int(action) for action in row[:prefix_length]), 0.0j
                )
                for row in actions
            ],
            dtype=np.complex128,
        )
        increment = current - previous
        increment[~active] = 0.0j
        increments[decision] = increment
        contribution = float(np.sum(probabilities * np.abs(increment) ** 2))
        rows.append(
            {
                "decision": float(decision),
                "wavefunction_response_norm": contribution,
                "wavefunction_response_fraction": (
                    contribution / total_response if total_response > 0.0 else 0.0
                ),
            }
        )
        previous = current

    reconstructed_ratio = overlap + np.sum(increments, axis=0)
    reconstructed = initial * reconstructed_ratio
    weighted_increments = increments * np.sqrt(probabilities)[None, :]
    gram = weighted_increments.conj() @ weighted_increments.T
    off_diagonal = gram - np.diag(np.diag(gram))
    increment_means = increments @ probabilities
    explained = float(
        abs(overlap) ** 2
        + sum(row["wavefunction_response_norm"] for row in rows)
    )
    return {
        "overlap": overlap,
        "increments": increments,
        "rows": rows,
        "target_support_leakage": leakage,
        "total_wavefunction_response_norm": total_response,
        "normalization_sum": explained,
        "normalization_sum_error": float(abs(1.0 - explained)),
        "maximum_increment_orthogonality_error": (
            float(np.max(np.abs(off_diagonal))) if off_diagonal.size else 0.0
        ),
        "maximum_increment_mean_error": (
            float(np.max(np.abs(increment_means)))
            if increment_means.size
            else 0.0
        ),
        "maximum_state_reconstruction_error": float(
            np.max(np.abs(reconstructed - changed))
        ),
    }


def complex_martingale_intervention(
    reference: Sequence[complex] | np.ndarray,
    decomposition: dict[str, object],
    selected_decisions: Sequence[int],
) -> np.ndarray:
    """Return the normalized state containing selected complex increments.

    The empty selection is defined to be the reference state, including when
    ``<reference|target> == 0``; the complete selection is the target state.
    Intermediate selections retain the orthogonal projection constant and
    the chosen increments.  They are target/suffix-dependent filtration
    oracle diagnostics, not deployable causal features, neural states, or
    variational optima.
    """

    initial = np.asarray(reference, dtype=np.complex128)
    if initial.ndim != 1 or np.any(~np.isfinite(initial)):
        raise ValueError("reference must be a finite one-dimensional vector")
    norm_initial = float(np.linalg.norm(initial))
    if norm_initial <= 0.0:
        raise ValueError("reference must have positive norm")
    initial = initial / norm_initial
    increments = np.asarray(decomposition["increments"], dtype=np.complex128)
    if increments.ndim != 2 or increments.shape[1] != len(initial):
        raise ValueError("decomposition increments do not match reference")
    raw_selected = tuple(selected_decisions)
    if any(
        not isinstance(decision, (int, np.integer))
        or isinstance(decision, (bool, np.bool_))
        for decision in raw_selected
    ):
        raise TypeError("selected_decisions must contain integers")
    selected = tuple(int(decision) for decision in raw_selected)
    if len(set(selected)) != len(selected):
        raise ValueError("selected_decisions must not contain duplicates")
    if any(decision < 0 or decision >= len(increments) for decision in selected):
        raise ValueError("a selected decision lies outside the decomposition")
    if not selected:
        return initial.copy()
    multiplier = np.full(
        len(initial), complex(decomposition["overlap"]), dtype=np.complex128
    )
    multiplier += np.sum(increments[np.asarray(selected, dtype=np.int32)], axis=0)
    answer = initial * multiplier
    norm = float(np.linalg.norm(answer))
    if norm <= 0.0:
        raise ValueError("selected martingale intervention has zero norm")
    return answer / norm
