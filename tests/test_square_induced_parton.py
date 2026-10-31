from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.square_induced_parton import (
    build_square_fermi_sea_orbit,
    full_reynolds_square_fermi_sea_reference,
    projected_square_fermi_sea_a1_amplitudes,
    square_a1_coefficients,
    square_nearest_neighbor_hopping,
)


def _states(n_sites: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(count):
        state = np.asarray([1] * (n_sites // 2) + [0] * (n_sites // 2), dtype=np.int8)
        rng.shuffle(state)
        result.append(state)
    return np.asarray(result)


def test_square_hopping_has_only_registered_mixed_closed_shells() -> None:
    orbit = build_square_fermi_sea_orbit(4)
    assert np.min(orbit.one_body_gaps) > 1.0
    assert abs(orbit.excluded_pp_gap) < 1.0e-12
    assert abs(orbit.excluded_aa_gap) < 1.0e-12
    for pairing in orbit.pairings:
        assert np.all(np.isfinite(pairing))
    hopping = square_nearest_neighbor_hopping(4, boundary_x="A", boundary_y="P")
    assert np.max(np.abs(hopping - hopping.T)) < 1.0e-14


def test_compact_square_a1_equals_full_reynolds_oracle() -> None:
    orbit = build_square_fermi_sea_orbit(4)
    states = _states(16, 12, 20260926)
    compact = projected_square_fermi_sea_a1_amplitudes(states, orbit=orbit)
    full = full_reynolds_square_fermi_sea_reference(states, orbit=orbit)
    assert np.max(np.abs(compact.amplitudes - np.asarray(full.amplitudes))) < 3.0e-13


def test_square_six_by_six_compact_formula_remains_exact() -> None:
    orbit = build_square_fermi_sea_orbit(6)
    states = _states(36, 4, 20260927)
    compact = projected_square_fermi_sea_a1_amplitudes(states, orbit=orbit)
    full = full_reynolds_square_fermi_sea_reference(
        states, orbit=orbit, microbatch_size=2
    )
    assert np.max(np.abs(compact.amplitudes - np.asarray(full.amplitudes))) < 5.0e-13


def test_square_a1_boundary_phase_tracks_projected_fermion_parity() -> None:
    assert np.array_equal(square_a1_coefficients(4), np.asarray((0.5, 0.5)))
    assert np.array_equal(square_a1_coefficients(6), np.asarray((0.5, -0.5)))
    orbit = build_square_fermi_sea_orbit(8)
    states = _states(64, 1, 20260928)
    compact = projected_square_fermi_sea_a1_amplitudes(states, orbit=orbit)
    full = full_reynolds_square_fermi_sea_reference(
        states, orbit=orbit, microbatch_size=1
    )
    scale = max(float(np.max(np.abs(full.amplitudes))), 1.0e-30)
    assert (
        np.max(np.abs(compact.amplitudes - np.asarray(full.amplitudes))) / scale
        < 2.0e-12
    )
