from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_n16_spectral_error_tomography import (
    _validate,
)
from su2_nqs.benchmarks.spectral_error_tomography import (
    krylov_columns,
    lowest_subspace_state,
    orthogonal_weight,
    spectral_state_profile,
    state_metrics,
)


def test_registered_spectral_tomography_contract() -> None:
    path = Path(
        "projects/clebsch_tree_nqs/configs/validation/"
        "spin_n16_spectral_error_tomography.json"
    )
    spec = json.loads(path.read_text())
    _validate(spec)
    assert spec["diagnostic_only"] is True


def test_spectral_profile_reconstructs_energy_and_variance() -> None:
    hamiltonian = np.diag((-2.0, -0.5, 3.0))
    state = np.asarray((np.sqrt(0.8), np.sqrt(0.15), np.sqrt(0.05)))
    profile = spectral_state_profile(state, hamiltonian)
    metrics = state_metrics(state, hamiltonian)
    np.testing.assert_allclose(profile["energy_total"], metrics["energy_total"])
    np.testing.assert_allclose(profile["variance_total"], metrics["variance_total"])
    np.testing.assert_allclose(profile["ground_state_fidelity"], 0.8)
    assert profile["upper_half_variance_fraction"] > 0.0


def test_matched_rank_hybrid_detects_non_krylov_direction() -> None:
    hamiltonian = np.diag((-3.0, -1.0, 0.5, 2.0))
    carrier = np.asarray((1.0, 1.0, 1.0, 0.0))
    columns = krylov_columns(carrier, hamiltonian, degree=1)
    outside = np.asarray((0.0, 0.0, 0.0, 1.0))
    assert orthogonal_weight(outside, columns) > 0.99
    complementary_low_energy = np.asarray((1.0, 0.0, 0.0, 0.0))
    assert orthogonal_weight(complementary_low_energy, columns) > 1.0e-3
    state, rank = lowest_subspace_state(
        np.column_stack((columns, complementary_low_energy)),
        hamiltonian,
    )
    assert rank == 3
    np.testing.assert_allclose(
        state_metrics(state, hamiltonian)["energy_total"], -3.0, atol=1.0e-12
    )
