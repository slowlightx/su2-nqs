"""Hamiltonian-connected proposals for exact importance-sampling audits.

The sparse Hamiltonian convention is ``H[new_state, old_state]``.  Therefore
each column defines the states reachable from one base Born sample.  This
module deliberately stays independent of the neural optimizer: it constructs
and validates the proposal before it is used by sampled distillation or SR.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def hamiltonian_connected_transition(
    hamiltonian: Any,
    *,
    amplitude_power: float = 1.0,
    zero_tolerance: float = 0.0,
) -> Any:
    """Return the column-stochastic local-action kernel ``K[new, old]``.

    For an old basis state ``x`` and a distinct connected state ``y``,

    ``K[y, x] = |H[y, x]|**amplitude_power / sum(z != x) |H[z, x]|**amplitude_power``.

    Diagonal matrix elements are excluded because they do not expose a new
    autoregressive path.  A column without an off-diagonal neighbor falls back
    to ``K[x, x] = 1``.  The returned object is a SciPy CSC matrix, imported
    locally so that SciPy remains an optional benchmark dependency.
    """

    from scipy import sparse

    if not np.isfinite(amplitude_power) or amplitude_power <= 0.0:
        raise ValueError("amplitude_power must be positive and finite")
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be nonnegative and finite")
    if len(hamiltonian.shape) != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be square")
    dimension = int(hamiltonian.shape[0])
    matrix = sparse.csc_matrix(hamiltonian, dtype=np.complex128)
    matrix.sum_duplicates()
    if np.any(~np.isfinite(matrix.data)):
        raise ValueError("hamiltonian entries must be finite")

    rows: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    counts = np.empty(dimension, dtype=np.int64)
    for column in range(dimension):
        start, stop = matrix.indptr[column : column + 2]
        local_rows = matrix.indices[start:stop]
        local_values = matrix.data[start:stop]
        keep = (local_rows != column) & (np.abs(local_values) > zero_tolerance)
        local_rows = np.asarray(local_rows[keep], dtype=np.int32)
        if len(local_rows) == 0:
            local_rows = np.asarray([column], dtype=np.int32)
            local_probabilities = np.asarray([1.0], dtype=np.float64)
        else:
            local_probabilities = np.asarray(
                np.abs(local_values[keep]) ** amplitude_power,
                dtype=np.float64,
            )
            normalizer = float(np.sum(local_probabilities))
            if not np.isfinite(normalizer) or normalizer <= 0.0:
                raise ValueError("connected transition has an invalid normalization")
            local_probabilities /= normalizer
        rows.append(local_rows)
        probabilities.append(local_probabilities)
        counts[column] = len(local_rows)

    indptr = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(counts)))
    transition = sparse.csc_matrix(
        (np.concatenate(probabilities), np.concatenate(rows), indptr),
        shape=(dimension, dimension),
    )
    transition.sort_indices()
    return transition


def connected_proposal_distribution(
    base_probabilities: np.ndarray,
    transition: Any,
) -> np.ndarray:
    """Push a normalized base distribution through ``K[new, old]``."""

    base = _normalized_probabilities(base_probabilities, name="base_probabilities")
    if transition.shape != (len(base), len(base)):
        raise ValueError("transition and base distribution dimensions do not match")
    column_sums = np.asarray(transition.sum(axis=0)).reshape(-1)
    if not np.allclose(column_sums, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("transition must be column stochastic")
    if np.any(np.asarray(transition.data) < 0.0):
        raise ValueError("transition probabilities must be nonnegative")
    connected = np.asarray(transition @ base, dtype=np.float64).reshape(-1)
    connected = np.maximum(connected, 0.0)
    connected /= np.sum(connected)
    return connected


def mixed_connected_proposal_distribution(
    base_probabilities: np.ndarray,
    transition: Any,
    *,
    mixture_weight: float,
) -> np.ndarray:
    """Return ``r_eta = (1-eta) p + eta K p`` exactly."""

    if not np.isfinite(mixture_weight) or not 0.0 <= mixture_weight <= 1.0:
        raise ValueError("mixture_weight must lie in [0, 1]")
    base = _normalized_probabilities(base_probabilities, name="base_probabilities")
    connected = connected_proposal_distribution(base, transition)
    proposal = (1.0 - mixture_weight) * base + mixture_weight * connected
    proposal /= np.sum(proposal)
    return proposal


def mixed_connected_probabilities_at_indices(
    target_indices: np.ndarray,
    transition: Any,
    base_probability_fn: Callable[[np.ndarray], np.ndarray],
    *,
    mixture_weight: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Evaluate ``[(1-eta)p + eta Kp](y)`` from reverse local neighbors.

    Only base probabilities for the requested targets and nonzero entries in
    the corresponding *rows* of ``K[new,old]`` are evaluated.  This is the
    probability-evaluation counterpart of
    :func:`sample_mixed_connected_indices`: it avoids a full probability
    vector while retaining the exact normalized-autoregressive ``p/q``
    weight.  The sparse transition may later be replaced by an on-the-fly
    reverse-connectivity provider without changing the formula.
    """

    proposal, _, diagnostics = mixed_connected_probabilities_and_base_at_indices(
        target_indices,
        transition,
        base_probability_fn,
        mixture_weight=mixture_weight,
    )
    return proposal, diagnostics


def mixed_connected_probabilities_and_base_at_indices(
    target_indices: np.ndarray,
    transition: Any,
    base_probability_fn: Callable[[np.ndarray], np.ndarray],
    *,
    mixture_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return local proposal and base probabilities with one NQS evaluation.

    This is the shared-work form of
    :func:`mixed_connected_probabilities_at_indices`.  In addition to
    ``q(target)``, it returns ``p(target)`` from the same evaluation of the
    union of targets and reverse neighbors.  Importance samplers therefore do
    not need a second autoregressive pass over the sampled targets.
    """

    if not np.isfinite(mixture_weight) or not 0.0 <= mixture_weight <= 1.0:
        raise ValueError("mixture_weight must lie in [0, 1]")
    targets = np.asarray(target_indices)
    if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("target_indices must be a one-dimensional integer array")
    dimension = int(transition.shape[0])
    if transition.shape != (dimension, dimension):
        raise ValueError("transition must be square")
    if np.any((targets < 0) | (targets >= dimension)):
        raise ValueError("target index lies outside the transition space")
    if len(targets) == 0:
        empty = np.empty(0, dtype=np.float64)
        return (
            empty,
            empty.copy(),
            {
                "unique_target_count": 0,
                "unique_probability_evaluation_count": 0,
                "reverse_connection_count": 0,
            },
        )

    rows = transition.tocsr()
    unique_targets, inverse = np.unique(targets.astype(np.int64), return_inverse=True)
    reverse_indices = []
    reverse_values = []
    reverse_connection_count = 0
    required = set(int(value) for value in unique_targets)
    for target in unique_targets:
        start, stop = rows.indptr[int(target) : int(target) + 2]
        indices = np.asarray(rows.indices[start:stop], dtype=np.int64)
        values = np.asarray(rows.data[start:stop], dtype=np.float64)
        reverse_indices.append(indices)
        reverse_values.append(values)
        reverse_connection_count += len(indices)
        required.update(int(value) for value in indices)
    required_indices = np.asarray(sorted(required), dtype=np.int64)
    required_probabilities = np.asarray(
        base_probability_fn(required_indices), dtype=np.float64
    )
    if required_probabilities.shape != required_indices.shape:
        raise ValueError("base_probability_fn returned the wrong shape")
    if np.any(~np.isfinite(required_probabilities)) or np.any(
        required_probabilities < 0.0
    ):
        raise ValueError("base probabilities must be finite and nonnegative")
    lookup = {
        int(index): float(probability)
        for index, probability in zip(
            required_indices, required_probabilities, strict=True
        )
    }
    unique_proposal = np.empty(len(unique_targets), dtype=np.float64)
    for position, (target, indices, values) in enumerate(
        zip(unique_targets, reverse_indices, reverse_values, strict=True)
    ):
        connected = sum(
            float(value) * lookup[int(index)]
            for index, value in zip(indices, values, strict=True)
        )
        unique_proposal[position] = (
            (1.0 - mixture_weight) * lookup[int(target)]
            + mixture_weight * connected
        )
    if np.any(~np.isfinite(unique_proposal)) or np.any(unique_proposal < 0.0):
        raise FloatingPointError("local connected proposal probability is invalid")
    return (
        unique_proposal[inverse],
        np.asarray([lookup[int(target)] for target in targets], dtype=np.float64),
        {
            "unique_target_count": int(len(unique_targets)),
            "unique_probability_evaluation_count": int(len(required_indices)),
            "reverse_connection_count": int(reverse_connection_count),
        },
    )


def stratified_connected_proposal_distribution(
    base_probabilities: np.ndarray,
    transitions: tuple[Any, ...],
    *,
    mixture_weights: tuple[float, ...],
) -> np.ndarray:
    """Return a mixture retaining distinct Hamiltonian connection channels.

    The proposal is

    ``r = (1-sum_a eta_a) p + sum_a eta_a K_a p``.

    Keeping bond shells or operator channels separate prevents the column
    normalization of one channel from suppressing another before sampling.
    """

    if len(transitions) == 0 or len(transitions) != len(mixture_weights):
        raise ValueError("transitions and mixture_weights must have equal positive length")
    weights = np.asarray(mixture_weights, dtype=np.float64)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("mixture weights must be finite and nonnegative")
    total_weight = float(np.sum(weights))
    if total_weight > 1.0 + 1e-14:
        raise ValueError("mixture weights must sum to at most one")
    base = _normalized_probabilities(base_probabilities, name="base_probabilities")
    proposal = max(1.0 - total_weight, 0.0) * base
    for weight, transition in zip(weights, transitions, strict=True):
        if weight > 0.0:
            proposal += weight * connected_proposal_distribution(base, transition)
    proposal /= np.sum(proposal)
    return proposal


def sample_mixed_connected_indices(
    base_indices: np.ndarray,
    transition: Any,
    *,
    mixture_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the mixture kernel to indices already sampled from the base NQS.

    If ``base_indices`` are distributed as ``p``, the returned indices are
    distributed exactly as ``(1-eta) p + eta K p``.  The function uses only
    column-local sparse data, so it does not require enumerating the Hilbert
    space once a base autoregressive sampler has produced its samples.
    """

    if not np.isfinite(mixture_weight) or not 0.0 <= mixture_weight <= 1.0:
        raise ValueError("mixture_weight must lie in [0, 1]")
    parents = np.asarray(base_indices)
    if parents.ndim != 1 or not np.issubdtype(parents.dtype, np.integer):
        raise ValueError("base_indices must be a one-dimensional integer array")
    dimension = int(transition.shape[0])
    if transition.shape != (dimension, dimension):
        raise ValueError("transition must be square")
    if np.any((parents < 0) | (parents >= dimension)):
        raise ValueError("base index lies outside the transition space")
    transition = transition.tocsc()
    result = np.asarray(parents, dtype=np.int64).copy()
    move = rng.random(len(result)) < mixture_weight
    _sample_transition_columns(
        parents,
        np.flatnonzero(move),
        transition,
        rng=rng,
        result=result,
    )
    return result


def sample_stratified_connected_indices(
    base_indices: np.ndarray,
    transitions: tuple[Any, ...],
    *,
    mixture_weights: tuple[float, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a stay/connected-channel mixture from base-distributed indices."""

    if len(transitions) == 0 or len(transitions) != len(mixture_weights):
        raise ValueError("transitions and mixture_weights must have equal positive length")
    weights = np.asarray(mixture_weights, dtype=np.float64)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("mixture weights must be finite and nonnegative")
    if float(np.sum(weights)) > 1.0 + 1e-14:
        raise ValueError("mixture weights must sum to at most one")
    parents = np.asarray(base_indices)
    if parents.ndim != 1 or not np.issubdtype(parents.dtype, np.integer):
        raise ValueError("base_indices must be a one-dimensional integer array")
    dimension = int(transitions[0].shape[0])
    if np.any((parents < 0) | (parents >= dimension)):
        raise ValueError("base index lies outside the transition space")
    concrete_transitions = []
    for transition in transitions:
        if transition.shape != (dimension, dimension):
            raise ValueError("all transitions must share one square shape")
        concrete_transitions.append(transition.tocsc())

    result = np.asarray(parents, dtype=np.int64).copy()
    branch_draws = rng.random(len(result))
    lower = 0.0
    for weight, transition in zip(weights, concrete_transitions, strict=True):
        upper = lower + float(weight)
        selected = np.flatnonzero((branch_draws >= lower) & (branch_draws < upper))
        _sample_transition_columns(
            parents,
            selected,
            transition,
            rng=rng,
            result=result,
        )
        lower = upper
    return result


def _sample_transition_columns(
    parents: np.ndarray,
    positions: np.ndarray,
    transition: Any,
    *,
    rng: np.random.Generator,
    result: np.ndarray,
) -> None:
    """Sample each occupied sparse column in one vectorized group."""

    if len(positions) == 0:
        return
    selected_parents = parents[positions]
    for column in np.unique(selected_parents):
        local_positions = positions[selected_parents == column]
        start, stop = transition.indptr[int(column) : int(column) + 2]
        choices = transition.indices[start:stop]
        probabilities = transition.data[start:stop]
        if len(choices) == 0:
            raise ValueError("transition contains an empty column")
        result[local_positions] = rng.choice(
            choices,
            size=len(local_positions),
            p=probabilities,
        )


def proposal_importance_diagnostics(
    proposal_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    *,
    sample_count: int,
) -> dict[str, float | int]:
    """Return exact full-support importance and finite-pool tail diagnostics.

    The normalized importance weight is ``w(y)=q(y)/r(y)``.  Besides the
    standard asymptotic ESS fraction, ``expected_unseen_target_mass`` measures
    how much teacher probability remains unseen on average after
    ``sample_count`` independent proposal draws.  The analogous second-moment
    quantity is sensitive to rare states that dominate variance.
    """

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    proposal = _normalized_probabilities(
        proposal_probabilities, name="proposal_probabilities"
    )
    target = _normalized_probabilities(target_probabilities, name="target_probabilities")
    if proposal.shape != target.shape:
        raise ValueError("proposal and target dimensions do not match")
    support = proposal > 0.0
    leaked = float(np.sum(target[~support]))
    positive_target = target > 0.0
    if np.any(positive_target & ~support):
        return {
            "effective_sample_size_fraction": 0.0,
            "target_weight_outside_proposal_support": leaked,
            "maximum_normalized_weight": float("inf"),
            "maximum_weight_index": int(np.flatnonzero(positive_target & ~support)[0]),
            "maximum_weight_proposal_probability": 0.0,
            "expected_maximum_weight_state_hits": 0.0,
            "probability_of_at_least_one_maximum_weight_state_hit": 0.0,
            "expected_unseen_target_mass": leaked,
            "expected_unseen_second_moment_fraction": 1.0,
        }

    positive_indices = np.flatnonzero(positive_target)
    log_weights = np.log(target[positive_target]) - np.log(proposal[positive_target])
    log_second_terms = np.log(target[positive_target]) + log_weights
    maximum_log_second = float(np.max(log_second_terms))
    scaled_second_terms = np.exp(log_second_terms - maximum_log_second)
    scaled_second_sum = float(np.sum(scaled_second_terms))
    log_second_moment = maximum_log_second + np.log(scaled_second_sum)
    maximum_local_index = int(np.argmax(log_weights))
    maximum_index = int(positive_indices[maximum_local_index])
    maximum_probability = float(proposal[maximum_index])
    miss_probability = np.ones_like(proposal)
    stochastic = (proposal > 0.0) & (proposal < 1.0)
    miss_probability[stochastic] = np.exp(
        sample_count * np.log1p(-proposal[stochastic])
    )
    miss_probability[proposal >= 1.0] = 0.0
    expected_unseen_target = float(np.sum(target * miss_probability))
    expected_unseen_second = float(
        np.sum(
            scaled_second_terms * miss_probability[positive_target]
        )
        / scaled_second_sum
    )
    hit_probability = (
        1.0
        if maximum_probability >= 1.0
        else float(-np.expm1(sample_count * np.log1p(-maximum_probability)))
    )
    return {
        "effective_sample_size_fraction": float(np.exp(-log_second_moment)),
        "target_weight_outside_proposal_support": leaked,
        "maximum_normalized_weight": float(np.exp(log_weights[maximum_local_index])),
        "maximum_weight_index": maximum_index,
        "maximum_weight_proposal_probability": maximum_probability,
        "expected_maximum_weight_state_hits": sample_count * maximum_probability,
        "probability_of_at_least_one_maximum_weight_state_hit": hit_probability,
        "expected_unseen_target_mass": expected_unseen_target,
        "expected_unseen_second_moment_fraction": expected_unseen_second,
    }


def _normalized_probabilities(values: np.ndarray, *, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total probability")
    return probabilities / total
