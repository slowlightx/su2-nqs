from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.long_range_loop_rvb import (
    _alternating_resonance_present,
    _primary_columns,
    build_long_range_loop_rvb_data,
)
from su2_nqs.benchmarks.rvb_covering_oracle import lowest_ritz_state


ROOT = Path(__file__).resolve().parents[1]
QSL = ROOT / "projects/clebsch_tree_nqs/results/spin_qsl_symmetry_diagnostics_v1/states.npz"
A1 = ROOT / "projects/clebsch_tree_nqs/results/spin_carrier_cell_scalar_circuit_v1/states.npz"


def test_alternating_resonance_requires_connected_disjoint_partner() -> None:
    current = ((0, 1), (2, 3))
    resonant = frozenset({(0, 1), (2, 3), (0, 2), (1, 3)})
    disconnected = frozenset(current)
    assert _alternating_resonance_present(current, resonant)
    assert not _alternating_resonance_present(current, disconnected)


@pytest.fixture(scope="module")
def data():
    with np.load(QSL) as qsl, np.load(A1) as a1:
        return build_long_range_loop_rvb_data(
            qsl["basis"],
            a1["physical_a1_basis"],
            a1["coupling_0.125_h_a1"],
            qsl["optimized_pi_flux_jastrow_two_lanczos"],
            qsl["exact_ground"],
        )


def test_registered_covering_support_and_features_are_nontrivial(data) -> None:
    assert data.matching_count == 15360
    assert data.orbit_count == 103
    assert data.nonzero_pi_flux_orbit_count == 92
    assert data.pi_flux_orbit_columns.shape == (14, 103)
    assert data.paired_z2_orbit_columns.shape == (14, 103)
    assert np.array_equal(np.unique(data.raw_loop_features[:, 0]), [0, 1])
    assert np.all(np.std(data.standardized_loop_features, axis=0) > 0.99)
    assert set(np.unique(data.loop_parity)) == {-1.0, 1.0}


def test_zero_loop_state_is_exact_singlet_and_gamma_a1(data) -> None:
    columns = _primary_columns(
        np.zeros(3), data, data.standardized_loop_features
    )
    state = lowest_ritz_state(columns, data.hamiltonian_a1).state
    physical = data.a1_basis @ state
    projected = np.mean(physical[data.permutation_indices], axis=0)
    assert spin_raising_relative_norm(physical, data.configurations) < 1.0e-12
    assert np.linalg.norm(projected - physical) < 1.0e-12
    assert np.linalg.matrix_rank(columns, tol=1.0e-11) == 2

