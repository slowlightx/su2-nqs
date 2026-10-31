from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.triangular_induced_parton_n64 import (
    A1_COEFFICIENTS,
    C6_REPRESENTATION,
    MIRROR_REPRESENTATION,
    TX_REPRESENTATION,
    TY_REPRESENTATION,
    build_n64_pi_zero_flux_orbit,
    full_reynolds_n64_pi_zero_flux_reference,
    projected_n64_pi_zero_flux_a1_amplitudes,
)
from su2_nqs.lattices.triangular_hnf import C6, MX, hnf_site_permutation


def _random_half_filled(count: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    states = np.zeros((count, 64), dtype=np.int8)
    for state in states:
        state[generator.choice(64, 32, replace=False)] = 1
    return states


def test_n64_closed_shell_orbit_and_induced_generator_action() -> None:
    orbit = build_n64_pi_zero_flux_orbit()
    assert orbit.pairings.shape == (3, 64 * 63 // 2)
    assert np.min(orbit.one_body_gaps) > 2.0
    assert orbit.excluded_pp_gap < 1.0e-12
    states = _random_half_filled(48, 2026092032)
    components = projected_n64_pi_zero_flux_a1_amplitudes(
        states, orbit=orbit
    ).components
    supercell = ((8, 0), (0, 8))
    generators = (
        hnf_site_permutation(supercell, translation=(1, 0)),
        hnf_site_permutation(supercell, translation=(0, 1)),
        hnf_site_permutation(supercell, operation=C6),
        hnf_site_permutation(supercell, operation=MX),
    )
    matrices = (
        TX_REPRESENTATION,
        TY_REPRESENTATION,
        C6_REPRESENTATION,
        MIRROR_REPRESENTATION,
    )
    for image, matrix in zip(generators, matrices, strict=True):
        transformed = projected_n64_pi_zero_flux_a1_amplitudes(
            states[:, np.asarray(image)], orbit=orbit
        ).components
        relative_error = np.linalg.norm(transformed - components @ matrix) / np.linalg.norm(
            transformed
        )
        assert relative_error < 3.0e-13
        np.testing.assert_allclose(
            matrix @ A1_COEFFICIENTS, A1_COEFFICIENTS, atol=1.0e-14
        )


def test_n64_compressed_a1_matches_full_768_image_reynolds() -> None:
    orbit = build_n64_pi_zero_flux_orbit()
    held_out = _random_half_filled(16, 2026092033)
    compressed = projected_n64_pi_zero_flux_a1_amplitudes(
        held_out, orbit=orbit, microbatch_size=16
    ).amplitudes
    full = np.asarray(
        full_reynolds_n64_pi_zero_flux_reference(
            held_out,
            orbit=orbit,
            microbatch_size=16,
        ).amplitudes
    )
    relative_error = np.linalg.norm(compressed - full) / np.linalg.norm(full)
    assert relative_error < 3.0e-13
