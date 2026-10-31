from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from su2_nqs.benchmarks.sign_frame_forensic import (
    decode_exchange_support,
    exact_fraction_minimum,
    exact_graph_costs,
    exact_matrix_costs,
    expected_oriented_multiplicity,
    naive_ratio_forward_error_bound,
    positive_transition_counts,
    stable_fsum_matrix_costs,
)
from su2_nqs.benchmarks.sign_frame_obstruction import (
    all_configuration_gauge_diagonals,
    enumerate_site_z2_gauges,
)
from su2_nqs.operators.sz_graph_sparse import heisenberg_sz_graph_sparse
from su2_nqs.su2.sz_basis import enumerate_sz_configs


def _small_reference():
    n_sites = 4
    weights = {
        (0, 1): Fraction(1),
        (1, 2): Fraction(1, 2),
        (0, 2): Fraction(1, 4),
    }
    bonds = tuple((i, j, float(value)) for (i, j), value in weights.items())
    matrix, basis = heisenberg_sz_graph_sparse(n_sites, bonds, n_up=2)
    gauges = enumerate_site_z2_gauges(n_sites)
    diagonals = all_configuration_gauge_diagonals(basis, gauges)
    return weights, matrix, basis, gauges, diagonals


def test_actual_sparse_support_decodes_every_exchange_and_reverse() -> None:
    weights, matrix, basis, _, _ = _small_reference()
    support = decode_exchange_support(matrix, basis, weights)
    assert support.edges == ((0, 1), (0, 2), (1, 2))
    assert support.expected_oriented_multiplicity == 4
    assert expected_oriented_multiplicity(4, 2) == 4
    assert support.multiplicities == (4, 4, 4)
    assert len(support.values) == 12
    assert support.reverse_missing_count == 0
    assert support.duplicate_directed_count == 0
    assert support.maximum_imaginary_value == 0.0
    assert support.maximum_exchange_value_error == 0.0
    assert support.recovered_weights == (
        Fraction(1),
        Fraction(1, 4),
        Fraction(1, 2),
    )


@pytest.mark.parametrize("power", (1, 2))
def test_fraction_matrix_recovery_equals_independent_graph_for_all_gauges(
    power: int,
) -> None:
    weights, matrix, basis, gauges, diagonals = _small_reference()
    support = decode_exchange_support(matrix, basis, weights)
    counts, mismatch_count = positive_transition_counts(
        support, gauges, diagonals, chunk_size=3
    )
    assert mismatch_count == 0
    multiplicities = np.asarray(support.multiplicities)
    assert np.all((counts == 0) | (counts == multiplicities[None, :]))
    graph = exact_graph_costs(
        gauges,
        support.edges,
        tuple(weights[edge] for edge in support.edges),
        weight_power=power,
    )
    recovered = exact_matrix_costs(
        counts,
        support.multiplicities,
        support.recovered_weights,
        weight_power=power,
    )
    assert graph == recovered
    assert graph[-1] == 1
    assert recovered[-1] == 1
    assert exact_fraction_minimum(graph, gauges) == exact_fraction_minimum(
        recovered, gauges
    )
    stable = stable_fsum_matrix_costs(support, counts, weight_power=power)
    np.testing.assert_allclose(
        stable, np.asarray([float(value) for value in graph]), atol=2.3e-16
    )


def test_forward_error_bound_is_conservative_and_rejects_invalid_count() -> None:
    audit = naive_ratio_forward_error_bound(18144)
    assert audit["gamma_n"] > 2.0e-12
    assert audit["conservative_ratio_absolute_error_bound"] > 4.0e-12
    assert 1.4166445794216997e-13 < audit[
        "conservative_ratio_absolute_error_bound"
    ]
    with pytest.raises(ValueError, match="positive"):
        naive_ratio_forward_error_bound(0)


def test_decoder_rejects_an_unregistered_active_edge() -> None:
    weights, matrix, basis, _, _ = _small_reference()
    missing = dict(weights)
    del missing[(0, 2)]
    with pytest.raises(ValueError, match="unregistered edge"):
        decode_exchange_support(matrix, basis, missing)

