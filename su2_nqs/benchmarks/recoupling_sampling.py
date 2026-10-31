"""Exact moments of a path-local stochastic recoupling estimator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..operators.permutation import (
    adjacent_swap_path_connectivity,
    site_permutation_fusion_row,
    site_permutation_to_adjacent_swap_word,
)
from ..su2.fusion_path import FusionPath


def local_choice_recoupling_moments(
    output_path: Sequence[int],
    image: Sequence[int],
    amplitude: Callable[[FusionPath], complex],
    *,
    target_q: int,
) -> dict[str, Any]:
    """Return exact first/second moments of sequential local-F sampling.

    At each adjacent swap, channel ``b`` is drawn with probability
    ``p_b=|F_b|^2/sum_c|F_c|^2`` and the walker weight is multiplied by
    ``F_b/p_b``.  The estimator is unbiased for the selected permuted
    amplitude.  Histories ending at the same source path are deliberately
    kept incoherent in the second moment, exposing the recoupling sign problem.
    Dynamic programming combines identical endpoints without enumerating the
    exponentially many histories.
    """

    checked_path = tuple(int(value) for value in output_path)
    checked_image = tuple(int(value) for value in image)
    sources, coefficients = site_permutation_fusion_row(
        checked_path, checked_image, target_q=target_q
    )
    mean = sum(
        coefficient * complex(amplitude(path))
        for path, coefficient in zip(sources, coefficients, strict=True)
    )

    second_weights: dict[FusionPath, float] = {checked_path: 1.0}
    history_counts: dict[FusionPath, int] = {checked_path: 1}
    maximum_local_normalization_error = 0.0
    for bond in reversed(site_permutation_to_adjacent_swap_word(checked_image)):
        updated_second: dict[FusionPath, float] = {}
        updated_counts: dict[FusionPath, int] = {}
        for path, outer_second in second_weights.items():
            connected, values = adjacent_swap_path_connectivity(
                path, bond, target_q=target_q
            )
            normalization = float(np.sum(np.abs(values) ** 2))
            maximum_local_normalization_error = max(
                maximum_local_normalization_error, abs(normalization - 1.0)
            )
            for new_path in connected:
                # p_b |F_b/p_b|^2 = sum_c |F_c|^2 for p_b normalized
                # from |F_b|^2.  The local norm is unity analytically.
                updated_second[new_path] = (
                    updated_second.get(new_path, 0.0)
                    + outer_second * normalization
                )
        for path, count in history_counts.items():
            connected, _ = adjacent_swap_path_connectivity(
                path, bond, target_q=target_q
            )
            for new_path in connected:
                updated_counts[new_path] = updated_counts.get(new_path, 0) + count
        second_weights = updated_second
        history_counts = updated_counts

    second_moment = float(
        sum(weight * abs(complex(amplitude(path))) ** 2 for path, weight in second_weights.items())
    )
    variance = float(max(second_moment - abs(mean) ** 2, 0.0))
    denominator = max(abs(mean) ** 2, np.finfo(float).tiny)
    return {
        "mean": complex(mean),
        "second_moment": second_moment,
        "variance": variance,
        "relative_variance": variance / denominator,
        "terminal_support": len(second_weights),
        "total_history_count": int(sum(history_counts.values())),
        "maximum_endpoint_history_multiplicity": int(
            max(history_counts.values(), default=0)
        ),
        "maximum_local_normalization_error": maximum_local_normalization_error,
    }


__all__ = ["local_choice_recoupling_moments"]
