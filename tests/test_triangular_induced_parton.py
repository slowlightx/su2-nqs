from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.benchmarks.triangular_induced_parton import (
    A1_COEFFICIENTS,
    C6_REPRESENTATION,
    MIRROR_REPRESENTATION,
    PiZeroFluxOrbit,
    TX_REPRESENTATION,
    TY_REPRESENTATION,
    build_n36_pi_zero_flux_orbit,
    closed_shell_pairing_from_hopping,
    collective_tower_base_amplitudes,
    compressed_collective_tower_a1,
    full_reynolds_pi_zero_flux_reference,
    induced_representation_audit,
    projected_pi_zero_flux_a1_amplitudes,
    projected_pi_zero_flux_a1_amplitudes_jax,
    project_collective_tower_a1,
    triangular_elementary_flux_products,
    triangular_pi_zero_flux_hopping,
)
from su2_nqs.lattices.triangular_hnf import C6, MX, hnf_site_permutation
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.projected_singlet_agp import (
    ProjectedSingletAGPConfig,
    projected_singlet_agp_amplitudes,
)


def _random_half_filled(
    count: int, n_sites: int, seed: int
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    states = np.zeros((count, n_sites), dtype=np.int8)
    for row in states:
        row[generator.choice(n_sites, n_sites // 2, replace=False)] = 1
    return states


def test_pi_zero_flux_pattern_and_closed_shell_boundary_orbit() -> None:
    for boundary_x, boundary_y in (("P", "P"), ("A", "P"), ("P", "A")):
        hopping = triangular_pi_zero_flux_hopping(
            6, boundary_x=boundary_x, boundary_y=boundary_y
        )
        down, up = triangular_elementary_flux_products(hopping, 6)
        np.testing.assert_array_equal(down, -np.ones(36))
        np.testing.assert_array_equal(up, np.ones(36))

    orbit = build_n36_pi_zero_flux_orbit()
    np.testing.assert_allclose(
        orbit.one_body_gaps, 2.0 * np.sqrt(2.0), atol=2.0e-14
    )
    assert orbit.excluded_aa_gap < 1.0e-12
    assert orbit.pairings.shape == (3, 36 * 35 // 2)


def test_closed_shell_pairing_is_an_exact_small_system_singlet() -> None:
    n_sites = 8
    generator = np.random.default_rng(2026091308)
    hopping = generator.normal(size=(n_sites, n_sites))
    hopping = (hopping + hopping.T) / 2.0
    pairing, gap = closed_shell_pairing_from_hopping(hopping)
    assert gap > 1.0e-10
    spin_squared, integer_basis = spin_squared_matrix(
        n_sites, n_up=n_sites // 2
    )
    states = np.asarray(
        [
            [(integer >> (n_sites - 1 - site)) & 1 for site in range(n_sites)]
            for integer in integer_basis
        ],
        dtype=np.int8,
    )
    identity = (tuple(range(n_sites)),)
    config = ProjectedSingletAGPConfig(n_sites=n_sites, group_images=identity)
    amplitudes = np.asarray(
        projected_singlet_agp_amplitudes(
            {"pairing_real": pairing}, states, config
        ).amplitudes
    )
    residual = np.linalg.norm(spin_squared @ amplitudes) / np.linalg.norm(amplitudes)
    assert residual < 3.0e-13


def test_induced_matrices_obey_group_relations_and_have_unique_a1() -> None:
    audit = induced_representation_audit()
    for key in (
        "tx_identity_error",
        "ty_identity_error",
        "c6_order_error",
        "mirror_order_error",
        "dihedral_error",
        "projector_idempotency_error",
        "a1_vector_error",
    ):
        assert float(audit[key]) < 1.0e-14
    assert audit["projector_rank"] == 1
    projector = np.asarray(audit["projector"])
    np.testing.assert_allclose(
        projector @ A1_COEFFICIENTS, A1_COEFFICIENTS, atol=1.0e-14
    )


def test_induced_matrices_act_on_raw_boundary_amplitudes() -> None:
    states = _random_half_filled(48, 36, 2026091309)
    components = projected_pi_zero_flux_a1_amplitudes(states).components
    supercell = ((6, 0), (0, 6))
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
        transformed = projected_pi_zero_flux_a1_amplitudes(
            states[:, np.asarray(image)]
        ).components
        relative_error = np.linalg.norm(transformed - components @ matrix) / np.linalg.norm(
            components
        )
        assert relative_error < 2.0e-13


def test_three_state_a1_matches_full_432_reynolds_on_512_held_out() -> None:
    # This seed is intentionally distinct from the discovery/calibration samples.
    held_out = _random_half_filled(512, 36, 2026091310)
    induced = projected_pi_zero_flux_a1_amplitudes(
        held_out, microbatch_size=64
    ).amplitudes
    full = np.asarray(
        full_reynolds_pi_zero_flux_reference(
            held_out, microbatch_size=64
        ).amplitudes
    )
    assert np.linalg.norm(full) > 1.0e-10
    relative_error = np.linalg.norm(induced - full) / np.linalg.norm(full)
    assert relative_error < 2.0e-13


def test_traceable_induced_carrier_matches_host_evaluator() -> None:
    states = _random_half_filled(17, 36, 2026091917)
    expected = projected_pi_zero_flux_a1_amplitudes(states).amplitudes
    actual = np.asarray(projected_pi_zero_flux_a1_amplitudes_jax(states))
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-15)


def test_collective_tower_compression_matches_full_432_on_new_heldout() -> None:
    # Prospectively frozen in the independent compression/closure audit config;
    # it was not one of the 128/512-state discovery samples.
    held_out = _random_half_filled(257, 36, 2026091577)
    for family, expected_terms in (("120", 1), ("stripe", 3)):
        compressed = compressed_collective_tower_a1(
            held_out, family=family
        )
        full = project_collective_tower_a1(held_out, family=family)
        assert compressed.semantic_term_count == expected_terms
        assert compressed.components.shape == (257, expected_terms)
        relative_error = np.linalg.norm(
            compressed.amplitudes - full.amplitudes
        ) / np.linalg.norm(full.amplitudes)
        assert relative_error < 2.0e-12
        np.testing.assert_allclose(
            compressed.cancellation_ratio,
            full.cancellation_ratio,
            rtol=0.0,
            atol=2.0e-12,
        )
    np.testing.assert_array_equal(
        compressed_collective_tower_a1(
            held_out, family="120"
        ).components[:, 0],
        collective_tower_base_amplitudes(held_out, family="120"),
    )


def test_ordinary_adjacency_heat_dressing_breaks_three_state_closure() -> None:
    # This is a regression counterexample, not a candidate carrier.  Applying
    # the same untwisted physical adjacency to the three twisted-boundary pair
    # kernels does not intertwine their induced space-group action.
    states = _random_half_filled(257, 36, 2026091577)
    orbit = build_n36_pi_zero_flux_orbit()
    adjacency = np.zeros((36, 36), dtype=np.float64)
    for first, second, coupling in triangular_j1j2_bonds(
        6, 6, j1=1.0, j2=0.0, periodic=True
    ):
        adjacency[first, second] = coupling
        adjacency[second, first] = coupling
    eigenvalues, eigenvectors = np.linalg.eigh(adjacency)
    upper = np.triu_indices(36, 1)
    generators = (
        hnf_site_permutation(((6, 0), (0, 6)), translation=(1, 0)),
        hnf_site_permutation(((6, 0), (0, 6)), translation=(0, 1)),
        hnf_site_permutation(((6, 0), (0, 6)), operation=C6),
        hnf_site_permutation(((6, 0), (0, 6)), operation=MX),
    )
    representations = (
        TX_REPRESENTATION,
        TY_REPRESENTATION,
        C6_REPRESENTATION,
        MIRROR_REPRESENTATION,
    )
    for tau in (0.037, 0.113):
        heat_kernel = (
            eigenvectors * np.exp(tau * eigenvalues)[None, :]
        ) @ eigenvectors.T
        dressed = []
        for packed in orbit.pairings:
            pairing = np.zeros((36, 36), dtype=np.float64)
            pairing[upper] = packed
            pairing[(upper[1], upper[0])] = packed
            pairing = heat_kernel @ pairing @ heat_kernel.T
            # Onsite pairs vanish after one-particle-per-site projection.
            np.fill_diagonal(pairing, 0.0)
            dressed.append(pairing[upper])
        adversarial_orbit = PiZeroFluxOrbit(
            pairings=np.stack(dressed),
            one_body_gaps=orbit.one_body_gaps,
            excluded_aa_gap=orbit.excluded_aa_gap,
        )

        batch = projected_pi_zero_flux_a1_amplitudes(
            states, orbit=adversarial_orbit
        )
        components = batch.components
        for image, representation in zip(
            generators, representations, strict=True
        ):
            transformed = projected_pi_zero_flux_a1_amplitudes(
                states[:, np.asarray(image)], orbit=adversarial_orbit
            ).components
            fixed_error = np.linalg.norm(
                transformed - components @ representation
            ) / np.linalg.norm(transformed)
            fitted = np.linalg.lstsq(
                components[:128], transformed[:128], rcond=None
            )[0]
            heldout_closure_error = np.linalg.norm(
                transformed[128:] - components[128:] @ fitted
            ) / np.linalg.norm(transformed[128:])
            assert fixed_error > 1.0e-3
            assert heldout_closure_error > 1.0e-3

        full = np.asarray(
            full_reynolds_pi_zero_flux_reference(
                states, orbit=adversarial_orbit
            ).amplitudes
        )
        relative_oracle_error = np.linalg.norm(
            batch.amplitudes - full
        ) / np.linalg.norm(full)
        assert relative_oracle_error > 1.0e-3
