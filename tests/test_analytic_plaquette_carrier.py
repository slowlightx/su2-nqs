import numpy as np

from su2_nqs.benchmarks.analytic_plaquette_carrier import (
    aligned_plaquette_a1_local_data,
    sample_aligned_plaquette_a1_physical,
    sample_aligned_plaquette_a1_schur,
)
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout


def test_local_a1_schur_coefficients_follow_registered_convention():
    physical, coefficients = aligned_plaquette_a1_local_data()
    np.testing.assert_allclose(coefficients, (np.sqrt(3.0) / 2.0, 0.5))
    assert abs(np.vdot(physical, physical).real - 1.0) < 2.0e-13


def test_n36_analytic_carrier_samplers_are_legal_and_reproducible():
    layout = rectangular_four_spin_layout(6, 6, triangular_coordinates=True)
    first_bits, first_phases = sample_aligned_plaquette_a1_physical(
        layout, sample_count=128, seed=7
    )
    second_bits, second_phases = sample_aligned_plaquette_a1_physical(
        layout, sample_count=128, seed=7
    )
    np.testing.assert_array_equal(first_bits, second_bits)
    np.testing.assert_array_equal(first_phases, second_phases)
    assert np.all(np.sum(first_bits, axis=1) == 18)
    labels, phases = sample_aligned_plaquette_a1_schur(
        layout, sample_count=128, seed=8
    )
    decoded = layout.decode(labels)
    assert np.all(decoded.cell_q == 0)
    assert np.all(decoded.intercell_q == 0)
    assert set(np.unique(decoded.cell_a)) == {0, 1}
    assert np.all(phases == 0.0)
