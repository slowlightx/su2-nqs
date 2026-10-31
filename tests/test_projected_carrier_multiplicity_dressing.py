from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.benchmarks.projected_singlet_agp import square_gamma_a1_group
from su2_nqs.benchmarks.support_aware_spatial_cg import spatial_cg_route_spec
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.lattices.multiplicity_spatial_features import (
    multiplicity_spatial_features,
)
from su2_nqs.models.projected_carrier_multiplicity_dressing import (
    ProjectedCarrierMultiplicityDressingConfig,
    certified_carrier_projected_residual_amplitudes,
    init_projected_carrier_multiplicity_dressing,
    projected_carrier_multiplicity_dressing_amplitude,
    projected_carrier_multiplicity_dressing_amplitudes,
)
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.models.spatial_projector import spatial_projected_amplitude
from su2_nqs.models.spin_clebsch_tree import (
    SpinClebschTreeConfig,
    init_spin_clebsch_tree,
    spin_clebsch_tree_amplitudes,
)
from su2_nqs.models.su2_multiplicity_spatial_attention import (
    SU2MultiplicitySpatialAttentionConfig,
)
from su2_nqs.models.su2_route_vit import SU2RouteViTConfig, init_su2_route_vit
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.vmc.physical_exchange import run_exchange_sweeps


jax.config.update("jax_enable_x64", True)


def _setup():
    spatial, _ = spatial_cg_route_spec(
        "square",
        2,
        2,
        feature_dim=2,
        maximum_intermediate_q=4,
        initialization_scale=0.25,
    )
    route = SU2RouteViTConfig(
        spatial=spatial,
        embedding_dim=4,
        head_count=1,
        depth=1,
        modulation_rank=1,
        modulation_scale=0.15,
        initialization_scale=0.2,
    )
    features = multiplicity_spatial_features(
        spatial.backbone.tree,
        spatial.route_orders,
        square_j1j2_bonds(2, 2, j1=1.0, j2=0.0, periodic=True),
        square_j1j2_bonds(2, 2, j1=0.0, j2=1.0, periodic=True),
    )
    multiplicity = SU2MultiplicitySpatialAttentionConfig(
        base=route,
        features=features,
        embedding_dim=4,
        head_count=1,
        modulation_scale=0.15,
        initialization_scale=0.2,
    )
    config = ProjectedCarrierMultiplicityDressingConfig(
        multiplicity=multiplicity,
        projector=PhysicalSpatialProjector(square_gamma_a1_group(2)),
    )
    frozen = init_su2_route_vit(jax.random.PRNGKey(1), route)
    params = init_projected_carrier_multiplicity_dressing(
        jax.random.PRNGKey(2),
        frozen,
        config,
        initial_residual_coefficient=0.17,
    )
    carrier_config = SpinClebschTreeConfig(
        FusionTree.left_associated(4),
        target_q=0,
        target_m2=0,
        feature_dim=2,
        maximum_intermediate_q=4,
        initialization_scale=0.3,
    )
    carrier_params = init_spin_clebsch_tree(
        jax.random.PRNGKey(3), carrier_config
    )

    def carrier(state):
        return spin_clebsch_tree_amplitudes(
            carrier_params, state[None, :], carrier_config
        )[0]

    return config, frozen, params, carrier


def _spin_squared(basis: tuple[tuple[int, ...], ...]) -> np.ndarray:
    matrix, integers = spin_squared_matrix(4, n_up=2)
    positions = {value: index for index, value in enumerate(integers)}
    order = [
        positions[sum(bit << (3 - site) for site, bit in enumerate(row))]
        for row in basis
    ]
    return matrix[np.ix_(order, order)]


def _assert_a1(vector, basis, group, *, atol=2.0e-12):
    positions = {row: index for index, row in enumerate(basis)}
    concrete = np.asarray(vector)
    for image in group:
        pulled = [
            positions[tuple(row[image[old]] for old in range(4))]
            for row in basis
        ]
        np.testing.assert_allclose(concrete[pulled], concrete, atol=atol, rtol=0.0)


def test_state_and_parameter_tangent_remain_exact_singlet_gamma_a1() -> None:
    config, frozen, params, carrier = _setup()
    basis = fixed_m_spin_basis(4, 0)
    states = jnp.asarray(basis, dtype=jnp.int32)

    def state(candidate):
        return projected_carrier_multiplicity_dressing_amplitudes(
            candidate,
            frozen,
            states,
            config,
            carrier_amplitude_fn=carrier,
        )

    values = state(params)
    tangent = jax.tree_util.tree_map(jnp.zeros_like, params)
    tangent["multiplicity"]["control_head"] = jnp.ones_like(
        params["multiplicity"]["control_head"]
    )
    tangent["residual_coefficient"] = jnp.asarray(0.31, dtype=jnp.float64)
    _, tangent_values = jax.jvp(state, (params,), (tangent,))

    spin_squared = _spin_squared(basis)
    for vector in (np.asarray(values), np.asarray(tangent_values)):
        assert np.linalg.norm(vector) > 1.0e-18
        assert np.linalg.norm(spin_squared @ vector) / np.linalg.norm(vector) < 2.0e-11
        _assert_a1(vector, basis, config.projector.site_permutations)


def test_exchange_sampling_uses_only_projected_amplitudes() -> None:
    config, frozen, params, carrier = _setup()
    basis = fixed_m_spin_basis(4, 0)
    initial = jnp.asarray(basis, dtype=jnp.int32)

    def amplitude(candidate, state):
        return projected_carrier_multiplicity_dressing_amplitude(
            candidate,
            frozen,
            state,
            config,
            carrier_amplitude_fn=carrier,
        )

    sampled, cached, _, diagnostics = run_exchange_sweeps(
        params,
        jax.random.PRNGKey(9),
        initial,
        amplitude_fn=amplitude,
        particle_kind="spin",
        sweeps=5,
        exchange_edges=((0, 1), (1, 2), (2, 3), (0, 3)),
        bond_proposal_probability=0.7,
        lazy_probability=0.2,
    )
    np.testing.assert_array_equal(np.sum(np.asarray(sampled), axis=1), 2)
    direct = jax.vmap(lambda row: amplitude(params, row))(sampled)
    np.testing.assert_allclose(cached, direct, atol=2.0e-13, rtol=0.0)
    for image in config.projector.site_permutations:
        pulled = sampled[:, jnp.asarray(image, dtype=jnp.int32)]
        transformed = jax.vmap(lambda row: amplitude(params, row))(pulled)
        np.testing.assert_allclose(transformed, direct, atol=2.0e-12, rtol=0.0)
    assert np.isfinite(float(diagnostics["accepted_move_fraction"]))


def test_certified_carrier_fast_path_equals_projecting_the_coherent_sum() -> None:
    config, frozen, params, raw_carrier = _setup()
    states = jnp.asarray(fixed_m_spin_basis(4, 0), dtype=jnp.int32)

    def carrier_raw(_params, state, _config):
        return raw_carrier(state)

    def projected_carrier(state):
        return spatial_projected_amplitude(
            None,
            state,
            None,
            config.projector,
            amplitude_fn=carrier_raw,
        )

    full = projected_carrier_multiplicity_dressing_amplitudes(
        params,
        frozen,
        states,
        config,
        carrier_amplitude_fn=raw_carrier,
    )
    fast = certified_carrier_projected_residual_amplitudes(
        params,
        frozen,
        states,
        config,
        carrier_amplitude_fn=projected_carrier,
    )
    np.testing.assert_allclose(fast, full, atol=3.0e-13, rtol=0.0)


def test_config_rejects_nontrivial_character_mislabeled_as_gamma_a1() -> None:
    config, _, _, _ = _setup()
    characters = list(config.projector.characters)
    characters[1] = -1.0
    try:
        projector = PhysicalSpatialProjector(
            config.projector.site_permutations,
            tuple(characters),
        )
    except ValueError:
        # Most one-sign mutations already violate the representation law.
        return
    try:
        ProjectedCarrierMultiplicityDressingConfig(
            multiplicity=config.multiplicity,
            projector=projector,
        )
    except ValueError as error:
        assert "trivial spatial character" in str(error)
    else:
        raise AssertionError("a nontrivial character was mislabeled Gamma-A1")
