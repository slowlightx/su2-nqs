from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.coherent_plaquette_frames import distinct_partition_orbit
from su2_nqs.benchmarks.coherent_plaquette_sampling import (
    build_linear_resonance_sampled_action,
    build_resonance_bank_sampled_action,
    coherent_b1_count_channels,
    importance_diagnostics,
    intercell_casimir_support,
    linear_resonance_energy_variance,
    ordered_frame_orbit_with_weights,
    resonance_bank_energy_variance,
    sample_coherent_plaquette_frames,
)
from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.coherent_plaquette_frames import (
    coherent_plaquette_frame_columns,
)
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.spatial_symmetry import square_gamma_a1_group
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout


def test_aligned_frame_has_zero_nonroot_intercell_casimir() -> None:
    layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=False)
    batch = sample_coherent_plaquette_frames(
        (layout.physical_cells,), sample_count=128, seed=2026092521
    )
    diagnostics = importance_diagnostics(batch)
    support = intercell_casimir_support(
        batch, (layout.physical_cells,), layout
    )
    assert diagnostics["importance_effective_sample_fraction"] == 1.0
    assert diagnostics["zero_target_fraction"] == 0.0
    assert abs(support["mean_intercell_casimir"]) < 2.0e-14


def test_coherent_square_frame_mixture_has_valid_importance_weights() -> None:
    layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=False)
    frames = distinct_partition_orbit(
        layout.physical_cells, square_gamma_a1_group(4)
    )
    assert len(frames) == 4
    batch = sample_coherent_plaquette_frames(
        frames, sample_count=128, seed=2026092522
    )
    diagnostics = importance_diagnostics(batch)
    assert np.all(batch.proposal_probability > 0.0)
    assert np.all(batch.importance_weight >= 0.0)
    assert 0.0 < diagnostics["importance_effective_sample_fraction"] <= 1.0


def test_b1_count_channels_match_grouped_exact_frame_columns() -> None:
    layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=False)
    group = square_gamma_a1_group(4)
    basis = fixed_m_spin_basis(16, 0)
    exact = coherent_plaquette_frame_columns(
        basis, layout.physical_cells, group
    )
    ordered, weights = ordered_frame_orbit_with_weights(
        layout.physical_cells, group
    )
    actual = coherent_b1_count_channels(
        np.asarray(basis, dtype=np.int8), ordered, weights
    )
    expected = np.column_stack(
        [
            np.sum(
                exact.columns[
                    :, [sum(label) == count for label in exact.labels]
                ],
                axis=1,
            )
            for count in range(len(layout.physical_cells) + 1)
        ]
    )
    assert np.max(np.abs(actual - expected)) < 2.0e-14


def test_complete_bank_contains_old_pair_b1_action() -> None:
    layout = rectangular_four_spin_layout(4, 4, triangular_coordinates=False)
    group = square_gamma_a1_group(4)
    proposal_frames = distinct_partition_orbit(layout.physical_cells, group)
    ordered, weights = ordered_frame_orbit_with_weights(
        layout.physical_cells, group
    )
    batch = sample_coherent_plaquette_frames(
        proposal_frames, sample_count=64, seed=2026092523
    )
    bonds = square_j1j2_bonds(4, 4, j1=1.0, j2=0.5, periodic=True)
    pair = build_linear_resonance_sampled_action(
        batch, ordered, weights, bonds
    )
    bank = build_resonance_bank_sampled_action(
        batch, ordered, weights, bonds
    )
    assert np.max(np.abs(bank.amplitudes[:, 0] - pair.base_amplitude)) < 2.0e-14
    assert np.max(np.abs(bank.amplitudes[:, 2] - pair.response_amplitude)) < 2.0e-14
    assert np.max(np.abs(bank.h_amplitudes[:, 0] - pair.h_base_amplitude)) < 2.0e-14
    assert np.max(np.abs(bank.h_amplitudes[:, 2] - pair.h_response_amplitude)) < 2.0e-14
    coefficient = 0.037
    bank_coefficients = np.zeros(bank.amplitudes.shape[1])
    bank_coefficients[0] = 1.0
    bank_coefficients[2] = coefficient
    old = linear_resonance_energy_variance(pair, coefficient)
    new = resonance_bank_energy_variance(bank, bank_coefficients)
    assert abs(old["energy_total"] - new["energy_total"]) < 2.0e-13
    assert abs(old["variance_total"] - new["variance_total"]) < 2.0e-12
