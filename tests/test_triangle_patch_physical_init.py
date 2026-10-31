from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.exact_diagonalization import (
    heisenberg_sz_matrix,
    spin_squared_matrix,
)
from su2_nqs.frontier.triangle_patch_physical_init import (
    init_triangle_patch_cluster_singlet,
    sample_triangle_patch_cluster_product_chains,
    triangle_patch_cluster_reference,
    triangle_patch_cluster_six_site_probabilities,
)
from su2_nqs.lattices.spatial_symmetry import (
    rectangular_translation_group,
    triangular_hnf_translation_group,
)
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector,
    spatial_projected_amplitudes,
)
from su2_nqs.models.triangle_patch_clebsch_tree import (
    triangle_patch_6x6_canary_plan,
    triangle_patch_amplitude,
    triangle_patch_amplitudes,
    triangle_patch_grid_plan,
    triangle_patch_parameter_count,
)


jax.config.update("jax_enable_x64", True)


def _configurations(n_sites: int, *, n_up: int | None = None) -> np.ndarray:
    return np.asarray(
        [
            [(state >> (n_sites - site - 1)) & 1 for site in range(n_sites)]
            for state in range(1 << n_sites)
            if n_up is None or state.bit_count() == n_up
        ],
        dtype=np.int32,
    )


def _internal_two_cluster_bonds(reference) -> tuple[tuple[int, int, float], ...]:
    maps = ((0, 1, 2, 6, 7, 8), (3, 4, 5, 9, 10, 11))
    return tuple(
        (mapping[first], mapping[second], coupling)
        for mapping in maps
        for first, second, coupling in reference.bonds
    )


@pytest.mark.parametrize(
    ("geometry", "ratio", "expected_energy"),
    (
        ("square", 0.5, -2.6778862533179946),
        ("triangular", 0.125, -2.656913725802567),
    ),
)
def test_cluster_reference_is_exact_normalized_singlet(
    geometry: str,
    ratio: float,
    expected_energy: float,
) -> None:
    reference = triangle_patch_cluster_reference(geometry, ratio)
    hamiltonian, basis = heisenberg_sz_matrix(6, reference.bonds)
    spin_squared, spin_basis = spin_squared_matrix(6)
    assert basis == spin_basis == tuple(range(64))
    np.testing.assert_allclose(reference.energy, expected_energy, atol=3.0e-14)
    np.testing.assert_allclose(
        np.linalg.norm(reference.computational_state), 1.0, atol=2.0e-14
    )
    assert reference.basis_orthonormality_error < 1.0e-14
    assert reference.eigen_residual < 3.0e-14
    assert reference.singlet_gap > 0.8
    assert reference.coefficients[np.argmax(np.abs(reference.coefficients))] > 0.0
    np.testing.assert_allclose(
        hamiltonian @ reference.computational_state,
        reference.energy * reference.computational_state,
        atol=4.0e-14,
        rtol=0.0,
    )
    assert np.linalg.norm(spin_squared @ reference.computational_state) < 4.0e-14


@pytest.mark.parametrize(
    ("geometry", "ratio"),
    (("square", 0.5), ("triangular", 0.125)),
)
def test_n12_initializer_is_exact_cluster_product_plus_orthogonal_floor(
    geometry: str,
    ratio: float,
) -> None:
    plan = triangle_patch_grid_plan(
        geometry, 2, 6, feature_dim=2, maximum_intermediate_q=10
    )
    key = jax.random.PRNGKey(701)
    physical = init_triangle_patch_cluster_singlet(
        key, plan, j2_over_j1=ratio, background_scale=0.0
    )
    mixed = init_triangle_patch_cluster_singlet(
        key, plan, j2_over_j1=ratio, background_scale=0.05
    )
    configurations = jnp.asarray(_configurations(12, n_up=6))
    physical_state = np.asarray(
        triangle_patch_amplitudes(physical, configurations, plan)
    )
    mixed_state = np.asarray(triangle_patch_amplitudes(mixed, configurations, plan))
    background = mixed_state - physical_state
    np.testing.assert_allclose(np.linalg.norm(physical_state), 1.0, atol=2.0e-13)
    np.testing.assert_allclose(np.linalg.norm(background), 0.05, atol=2.0e-13)
    np.testing.assert_allclose(
        np.vdot(physical_state, background), 0.0, atol=2.0e-14
    )
    np.testing.assert_allclose(
        np.linalg.norm(mixed_state), np.sqrt(1.0 + 0.05**2), atol=2.0e-13
    )
    assert np.count_nonzero(mixed_state == 0.0) == 0

    reference = triangle_patch_cluster_reference(geometry, ratio)
    hamiltonian, _ = heisenberg_sz_matrix(
        12, _internal_two_cluster_bonds(reference), n_up=6
    )
    np.testing.assert_allclose(
        hamiltonian @ physical_state,
        2.0 * reference.energy * physical_state,
        atol=8.0e-14,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("geometry", "ratio"),
    (("square", 0.5), ("triangular", 0.125)),
)
def test_n36_physical_initializer_has_generic_m0_support_and_finite_jvp(
    geometry: str,
    ratio: float,
) -> None:
    plan = triangle_patch_6x6_canary_plan(geometry)
    params = init_triangle_patch_cluster_singlet(
        jax.random.PRNGKey(801),
        plan,
        j2_over_j1=ratio,
        background_scale=0.05,
    )
    assert triangle_patch_parameter_count(params) == 1252
    rng = np.random.default_rng(802)
    configurations = []
    for _ in range(512):
        row = np.zeros(36, dtype=np.int32)
        row[rng.choice(36, size=18, replace=False)] = 1
        configurations.append(row)
    states = jnp.asarray(configurations)
    tangent = jax.tree_util.tree_map(jnp.ones_like, params)
    values, derivatives = jax.jvp(
        lambda candidate: triangle_patch_amplitudes(candidate, states, plan),
        (params,),
        (tangent,),
    )
    values = np.asarray(values)
    derivatives = np.asarray(derivatives)
    assert np.count_nonzero(values == 0.0) == 0
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(derivatives))
    assert np.min(np.abs(values)) > 0.0


@pytest.mark.parametrize(
    ("geometry", "ratio"),
    (("square", 0.5), ("triangular", 0.125)),
)
def test_cluster_product_chain_seed_is_deterministic_in_sector_and_orbit_support(
    geometry: str,
    ratio: float,
) -> None:
    probabilities = triangle_patch_cluster_six_site_probabilities(
        geometry, ratio
    )
    local_up = np.asarray([state.bit_count() for state in range(64)])
    np.testing.assert_allclose(np.sum(probabilities), 1.0, atol=2.0e-15)
    assert np.all(probabilities >= 0.0)
    assert np.count_nonzero(probabilities[local_up != 3]) == 0

    plan = triangle_patch_6x6_canary_plan(geometry)
    key = jax.random.PRNGKey(901)
    untranslated = sample_triangle_patch_cluster_product_chains(
        key,
        plan,
        j2_over_j1=ratio,
        n_chains=16,
        translate=False,
    )
    repeated = sample_triangle_patch_cluster_product_chains(
        key,
        plan,
        j2_over_j1=ratio,
        n_chains=16,
        translate=False,
    )
    np.testing.assert_array_equal(untranslated, repeated)
    host = np.asarray(untranslated)
    np.testing.assert_array_equal(host.sum(axis=1), np.full(16, 18))
    for row in range(0, 6, 2):
        for column in range(0, 6, 3):
            sites = (
                row * 6 + column,
                row * 6 + column + 1,
                row * 6 + column + 2,
                (row + 1) * 6 + column,
                (row + 1) * 6 + column + 1,
                (row + 1) * 6 + column + 2,
            )
            np.testing.assert_array_equal(host[:, sites].sum(axis=1), np.full(16, 3))

    translated = sample_triangle_patch_cluster_product_chains(
        key,
        plan,
        j2_over_j1=ratio,
        n_chains=8,
        translate=True,
    )
    np.testing.assert_array_equal(np.asarray(translated).sum(axis=1), np.full(8, 18))
    group = (
        rectangular_translation_group(6, 6)
        if geometry == "square"
        else triangular_hnf_translation_group(((6, 0), (0, 6)))
    )
    projector = PhysicalSpatialProjector(group, orbit_chunk_size=6)
    physical = init_triangle_patch_cluster_singlet(
        jax.random.PRNGKey(902),
        plan,
        j2_over_j1=ratio,
        background_scale=0.0,
    )
    projected = np.asarray(
        spatial_projected_amplitudes(
            physical,
            translated,
            plan,
            projector,
            amplitude_fn=triangle_patch_amplitude,
        )
    )
    assert np.all(np.isfinite(projected))
    assert np.min(np.abs(projected)) > 1.0e-14


def test_cluster_initializer_rejects_ambiguous_or_unphysical_contracts() -> None:
    plan = triangle_patch_grid_plan(
        "square", 2, 6, feature_dim=2, maximum_intermediate_q=10
    )
    with pytest.raises(ValueError, match="nonnegative"):
        init_triangle_patch_cluster_singlet(
            jax.random.PRNGKey(1),
            plan,
            j2_over_j1=0.5,
            background_scale=-0.1,
        )
    wrong_width = triangle_patch_grid_plan(
        "square", 2, 6, feature_dim=3, maximum_intermediate_q=10
    )
    with pytest.raises(ValueError, match="feature_dim=2"):
        init_triangle_patch_cluster_singlet(
            jax.random.PRNGKey(1), wrong_width, j2_over_j1=0.5
        )
