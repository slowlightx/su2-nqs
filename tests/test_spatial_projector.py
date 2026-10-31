from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.clebsch_tree_exact import (
    physical_permutation_index_table,
    reynolds_project_state,
)
from su2_nqs.benchmarks.hubbard_space_group import (
    fermionic_permutation_index_sign_tables,
    fermionic_reynolds_project_state,
)
from su2_nqs.benchmarks.spin_sector import spin_squared_sz_sparse
from su2_nqs.lattices.spatial_symmetry import (
    rectangular_gamma_a1_group,
    rectangular_translation_group,
    square_gamma_a1_group,
    triangular_hnf_gamma_a1_group,
    triangular_hnf_retained_gamma_a1_group,
)
from su2_nqs.lattices.triangular_hnf import triangular_hnf_j1j2_bonds
from su2_nqs.models.physical_tree import (
    PhysicalHubbardTreeConfig,
    PhysicalSpinTreeConfig,
    init_physical_hubbard_tree,
    init_physical_spin_tree,
    physical_hubbard_tree_amplitude,
    physical_hubbard_tree_amplitudes,
    physical_spin_tree_amplitude,
    physical_spin_tree_amplitudes,
)
from su2_nqs.models.shared_cut import SharedCutConfig, shared_cut_amplitudes
from su2_nqs.models.spatial_projector import (
    PhysicalSpatialProjector,
    fermionic_pullback_orbit,
    spatial_projected_amplitude,
    spatial_projected_amplitudes,
)
from su2_nqs.operators.hubbard_fock import enumerate_hubbard_tokens
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.vmc.spin_physical import padded_spin_connectivity, spin_physical_hpsi
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.sz_basis import enumerate_sz_configs


jax.config.update("jax_enable_x64", True)


def test_physical_space_group_orders() -> None:
    assert len(rectangular_translation_group(2, 4)) == 8
    assert len(rectangular_gamma_a1_group(2, 4)) == 16
    assert len(square_gamma_a1_group(4)) == 128
    assert len(triangular_hnf_gamma_a1_group(((4, 0), (0, 4)))) == 192
    assert len(triangular_hnf_retained_gamma_a1_group(((6, 0), (0, 4)))) == 48
    assert (
        triangular_hnf_retained_gamma_a1_group(((4, 0), (0, 4)))
        == triangular_hnf_gamma_a1_group(((4, 0), (0, 4)))
    )


def test_triangular_4x6_retained_group_preserves_j1j2_hamiltonian_graph() -> None:
    supercell = ((6, 0), (0, 4))
    group = triangular_hnf_retained_gamma_a1_group(supercell)
    bonds = triangular_hnf_j1j2_bonds(supercell, j1=1.0, j2=0.125)
    reference = {
        tuple(sorted((first, second))): coupling
        for first, second, coupling in bonds
    }
    for image in group:
        transformed = {
            tuple(sorted((image[first], image[second]))): coupling
            for first, second, coupling in bonds
        }
        assert transformed == reference


def test_spin_on_demand_projector_matches_exact_gather() -> None:
    basis = enumerate_sz_configs(4, 2)
    states = jnp.asarray(basis, dtype=jnp.int32)
    config = PhysicalSpinTreeConfig(
        FusionTree.balanced(4),
        target_q=0,
        target_m2=0,
        feature_dim=2,
        maximum_intermediate_q=4,
        initialization_scale=0.2,
    )
    params = init_physical_spin_tree(jax.random.PRNGKey(4), config)
    group = square_gamma_a1_group(2)
    projector = PhysicalSpatialProjector(group)
    raw = physical_spin_tree_amplitudes(params, states, config)
    indices = physical_permutation_index_table(basis, group)
    gathered = reynolds_project_state(raw, indices)
    on_demand = spatial_projected_amplitudes(
        params,
        states,
        config,
        projector,
        amplitude_fn=physical_spin_tree_amplitude,
    )
    np.testing.assert_allclose(on_demand, gathered, atol=2.0e-13, rtol=0.0)
    chunked = spatial_projected_amplitudes(
        params,
        states,
        config,
        PhysicalSpatialProjector(group, orbit_chunk_size=3),
        amplitude_fn=physical_spin_tree_amplitude,
    )
    np.testing.assert_allclose(chunked, gathered, atol=2.0e-13, rtol=0.0)
    rematerialized = spatial_projected_amplitudes(
        params,
        states,
        config,
        PhysicalSpatialProjector(
            group, orbit_chunk_size=3, rematerialize_orbit=True
        ),
        amplitude_fn=physical_spin_tree_amplitude,
    )
    np.testing.assert_allclose(rematerialized, gathered, atol=2.0e-13, rtol=0.0)


def test_hubbard_signed_on_demand_projector_matches_exact_gather() -> None:
    basis = enumerate_hubbard_tokens(4, 4, total_m2=0)
    states = jnp.asarray(basis, dtype=jnp.int32)
    config = PhysicalHubbardTreeConfig(
        n_sites=4,
        n_electrons=4,
        target_q=0,
        target_m2=0,
        feature_dim=1,
        initialization_scale=0.2,
    )
    params = init_physical_hubbard_tree(jax.random.PRNGKey(7), config)
    group = square_gamma_a1_group(2)
    projector = PhysicalSpatialProjector(group, fermionic=True)
    raw = physical_hubbard_tree_amplitudes(params, states, config)
    indices, signs = fermionic_permutation_index_sign_tables(basis, group)
    gathered = fermionic_reynolds_project_state(raw, indices, signs)
    on_demand = spatial_projected_amplitudes(
        params,
        states,
        config,
        projector,
        amplitude_fn=physical_hubbard_tree_amplitude,
    )
    np.testing.assert_allclose(on_demand, gathered, atol=2.0e-13, rtol=0.0)

    pulled, on_demand_signs = fermionic_pullback_orbit(states, projector)
    positions = {state: index for index, state in enumerate(basis)}
    pulled_indices = np.asarray(
        [[positions[tuple(row)] for row in orbit] for orbit in np.asarray(pulled)]
    )
    np.testing.assert_array_equal(pulled_indices.T, indices)
    np.testing.assert_array_equal(np.asarray(on_demand_signs).T, signs)

    chunked = spatial_projected_amplitudes(
        params,
        states,
        config,
        PhysicalSpatialProjector(group, fermionic=True, orbit_chunk_size=3),
        amplitude_fn=physical_hubbard_tree_amplitude,
    )
    np.testing.assert_allclose(chunked, gathered, atol=2.0e-13, rtol=0.0)


def test_projector_reorders_explicit_characters_with_group_images() -> None:
    translations = rectangular_translation_group(1, 4)
    phase_by_image = {
        image: np.exp(2.0j * np.pi * image[0] / 4.0) for image in translations
    }
    reversed_group = tuple(reversed(translations))
    projector = PhysicalSpatialProjector(
        reversed_group,
        tuple(phase_by_image[image] for image in reversed_group),
    )
    assert projector.characters == tuple(
        phase_by_image[image] for image in projector.site_permutations
    )
    with pytest.raises(ValueError, match="one-dimensional representation"):
        PhysicalSpatialProjector(translations, (1.0, 1.0, 1.0, -1.0))


def test_shared_site_orders_add_no_parameters_and_preserve_exact_singlet() -> None:
    basis = enumerate_sz_configs(4, 2)
    states = jnp.asarray(basis, dtype=jnp.int32)
    config = PhysicalSpinTreeConfig(
        FusionTree.balanced(4),
        target_q=0,
        target_m2=0,
        feature_dim=2,
        maximum_intermediate_q=4,
        initialization_scale=0.2,
    )
    params = init_physical_spin_tree(jax.random.PRNGKey(9), config)
    identity = tuple(range(4))
    one = SharedCutConfig(config, (identity,))
    one_values = shared_cut_amplitudes(
        params, states, one, amplitude_fn=physical_spin_tree_amplitude
    )
    raw = physical_spin_tree_amplitudes(params, states, config)
    np.testing.assert_allclose(one_values, raw, atol=2.0e-13, rtol=0.0)

    two = SharedCutConfig(config, (identity, (0, 2, 1, 3)))
    values = np.asarray(
        shared_cut_amplitudes(
            params, states, two, amplitude_fn=physical_spin_tree_amplitude
        )
    )
    spin_squared, spin_basis = spin_squared_sz_sparse(4, n_up=2)
    assert tuple(spin_basis) == basis
    assert np.linalg.norm(spin_squared @ values) < 2.0e-12


def test_projected_hpsi_uses_projector_for_centers_and_connected_states() -> None:
    basis = np.asarray(enumerate_sz_configs(4, 2), dtype=np.int32)
    samples = basis[[1, 4]]
    connected, elements, valid, _ = padded_spin_connectivity(
        samples, square_j1j2_bonds(2, 2, j1=1.0, j2=0.2, periodic=True)
    )
    config = PhysicalSpinTreeConfig(
        FusionTree.balanced(4),
        target_q=0,
        target_m2=0,
        feature_dim=2,
        maximum_intermediate_q=4,
        initialization_scale=0.2,
    )
    params = init_physical_spin_tree(jax.random.PRNGKey(31), config)
    projector = PhysicalSpatialProjector(square_gamma_a1_group(2))

    def projected(candidate, state, model_config):
        return spatial_projected_amplitude(
            candidate,
            state,
            model_config,
            projector,
            amplitude_fn=physical_spin_tree_amplitude,
        )

    centers, action = spin_physical_hpsi(
        params,
        samples,
        connected,
        elements,
        valid,
        config,
        amplitude_fn=projected,
    )
    expected_centers = jax.vmap(lambda row: projected(params, row, config))(samples)
    expected_connected = jax.vmap(
        lambda row: projected(params, row, config)
    )(jnp.asarray(connected).reshape((-1, 4))).reshape(elements.shape)
    expected_action = jnp.sum(
        jnp.where(valid, elements * expected_connected, 0.0), axis=1
    )
    np.testing.assert_allclose(centers, expected_centers, atol=2.0e-13, rtol=0.0)
    np.testing.assert_allclose(action, expected_action, atol=2.0e-13, rtol=0.0)
