"""Cross-lattice diagnostics for the frozen spin benchmark program.

The synthesis deliberately separates Hilbert-space compression, target-flow
tail coverage, variational tangent error, and Fisher conditioning.  All
ratios are reported as ``Sz=0 / fusion`` so values above one favor the fusion
basis for the named cost or error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


PREDICTORS = (
    "dimension",
    "condition_number",
    "exact_samples_for_99_percent_target_weight",
    "exact_tangent_residual_fraction",
)
RESPONSES = (
    "born_worst_tangent_error",
    "cutoff_worst_tangent_error",
)


def _positive_column(
    rows: Sequence[Mapping[str, object]], key: str
) -> np.ndarray:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("correlations require at least three rows")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{key} must contain finite positive values")
    return values


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks with deterministic tie handling."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator == 0.0:
        raise ValueError("correlation is undefined for a constant vector")
    return float(np.dot(first_centered, second_centered) / denominator)


def log_correlation_table(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Correlate registered predictors and sampled errors on logarithmic scale."""

    table: list[dict[str, object]] = []
    for response in RESPONSES:
        response_values = np.log10(_positive_column(rows, response))
        response_ranks = _average_ranks(response_values)
        for predictor in PREDICTORS:
            predictor_values = np.log10(_positive_column(rows, predictor))
            table.append(
                {
                    "response": response,
                    "predictor": predictor,
                    "log_pearson": _correlation(
                        predictor_values, response_values
                    ),
                    "spearman": _correlation(
                        _average_ranks(predictor_values), response_ranks
                    ),
                }
            )
    return table


def paired_basis_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Pair fusion and ``Sz=0`` rows for each lattice/cluster/ordering."""

    grouped: dict[tuple[str, int, str], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        key = (
            str(row["lattice"]),
            int(row["n_sites"]),
            str(row["ordering"]),
        )
        basis = str(row["basis_kind"])
        if basis not in ("fusion", "sz_zero"):
            raise ValueError("basis_kind must be fusion or sz_zero")
        if basis in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate {basis} row for {key}")
        grouped[key][basis] = row

    paired: list[dict[str, object]] = []
    ratio_metrics = {
        "dimension_compression": "dimension",
        "condition_gain": "condition_number",
        "exact_residual_gain": "exact_tangent_residual_fraction",
        "target_tail_n99_gain": "exact_samples_for_99_percent_target_weight",
        "born_worst_error_gain": "born_worst_tangent_error",
        "cutoff_worst_error_gain": "cutoff_worst_tangent_error",
    }
    for key in sorted(grouped):
        bases = grouped[key]
        if set(bases) != {"fusion", "sz_zero"}:
            raise ValueError(f"incomplete basis pair for {key}")
        fusion = bases["fusion"]
        sz_zero = bases["sz_zero"]
        item: dict[str, object] = {
            "lattice": key[0],
            "n_sites": key[1],
            "ordering": key[2],
            "maximum_cut_width": int(fusion["maximum_cut_width"]),
        }
        for output, source in ratio_metrics.items():
            fusion_value = float(fusion[source])
            sz_value = float(sz_zero[source])
            if fusion_value <= 0.0 or sz_value <= 0.0:
                raise ValueError(f"{source} must be positive")
            item[output] = sz_value / fusion_value
        paired.append(item)
    return paired


def summarize_paired_basis(
    paired: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize how often symmetry compression becomes a practical gain."""

    if not paired:
        raise ValueError("at least one paired basis row is required")
    metrics = (
        "condition_gain",
        "exact_residual_gain",
        "target_tail_n99_gain",
        "born_worst_error_gain",
        "cutoff_worst_error_gain",
    )
    compressions = [float(row["dimension_compression"]) for row in paired]
    return {
        "pair_count": len(paired),
        "dimension_compression_range": [min(compressions), max(compressions)],
        "fusion_better_counts": {
            metric: sum(float(row[metric]) > 1.0 for row in paired)
            for metric in metrics
        },
        "target_tail_counterexamples": [
            {
                "lattice": row["lattice"],
                "n_sites": row["n_sites"],
                "ordering": row["ordering"],
                "target_tail_n99_gain": row["target_tail_n99_gain"],
            }
            for row in paired
            if float(row["target_tail_n99_gain"]) < 1.0
        ],
    }


__all__ = [
    "PREDICTORS",
    "RESPONSES",
    "log_correlation_table",
    "paired_basis_rows",
    "summarize_paired_basis",
]
