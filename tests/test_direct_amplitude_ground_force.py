from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.projected_singlet_agp import permutation_group_closure
from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
from su2_nqs.models.resonating_su2_scalar_dressing import (
    ResonatingSU2ScalarDressingConfig,
    init_resonating_su2_scalar_dressing,
    packed_resonating_pairing_bank,
    projected_resonating_amplitudes,
)
from su2_nqs.su2.sz_basis import enumerate_sz_configs
from su2_nqs.vmc.direct_amplitude_ground_force import (
    direct_amplitude_ground_horizontal_force,
    prepare_direct_amplitude_ground_force_reference,
)
from su2_nqs.vmc.projected_agp import ProjectedAGPHeisenbergConfig
from su2_nqs.vmc.resonating_projected_agp import (
    ResonatingProjectedAGPAction,
    resonating_projected_agp_action,
    solve_three_root_resonating_subspace,
)


def _group(n_sites: int):
    translation = tuple((site + 1) % n_sites for site in range(n_sites))
    reflection = tuple((-site) % n_sites for site in range(n_sites))
    return permutation_group_closure((translation, reflection))


def _problem():
    n_sites = 8
    # Use the identity spatial group here so the three-column bank does not
    # accidentally span the entire tiny dihedral-A1 singlet sector.  Exact
    # space-group projection is tested independently in the dressing tests;
    # this test targets the nonlinear horizontal-force derivative.
    carrier_config = ProjectedSingletAGPConfig(
        n_sites, (tuple(range(n_sites)),)
    )
    graph = ProjectedAGPHeisenbergConfig(
        n_sites,
        tuple((site, (site + 1) % n_sites, 1.0) for site in range(n_sites))
        + tuple((site, (site + 2) % n_sites, 0.3) for site in range(n_sites)),
    )
    config = ResonatingSU2ScalarDressingConfig(
        rows=2, columns=4, embedding_dim=8, head_count=2, component_count=2
    )
    parameters = init_resonating_su2_scalar_dressing(jax.random.key(11), config)
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(n_sites, n_sites))
    base = 0.5 * (raw + raw.T)
    np.fill_diagonal(base, 0.0)
    directions = []
    for distance in (1, 2, 3):
        value = np.zeros_like(base)
        for site in range(n_sites):
            value[site, (site + distance) % n_sites] = 1.0
            value[(site + distance) % n_sites, site] = 1.0
        directions.append(value)
    directions = np.stack(directions)
    states = np.asarray(enumerate_sz_configs(n_sites, 4), dtype=np.int8)
    upper = np.triu_indices(n_sites, 1)
    fixed = resonating_projected_agp_action(
        {"pairing_real": jnp.asarray(base[upper])[None, :]},
        states,
        carrier_config,
        graph,
    )
    return parameters, base, directions, states, carrier_config, graph, config, fixed


def _combined_action(parameters, base, directions, states, carrier, graph, config, fixed):
    trainable = resonating_projected_agp_action(
        packed_resonating_pairing_bank(parameters, base, directions, config),
        states,
        carrier,
        graph,
    )
    return ResonatingProjectedAGPAction(
        amplitudes=jnp.column_stack((fixed.amplitudes[:, 0], trainable.amplitudes)),
        hamiltonian_actions=jnp.column_stack(
            (fixed.hamiltonian_actions[:, 0], trainable.hamiltonian_actions)
        ),
        center_exact_nodes=jnp.column_stack(
            (fixed.center_exact_nodes[:, 0], trainable.center_exact_nodes)
        ),
        minimum_connected_cancellation_ratio=jnp.column_stack(
            (
                fixed.minimum_connected_cancellation_ratio[:, 0],
                trainable.minimum_connected_cancellation_ratio,
            )
        ),
    )


def test_ground_force_is_finite_horizontal_and_matches_exhaustive_directional_difference() -> None:
    parameters, base, directions, states, carrier, graph, config, fixed = _problem()
    action = _combined_action(
        parameters, base, directions, states, carrier, graph, config, fixed
    )
    proposal = np.full(len(states), 1.0 / len(states))
    reference = prepare_direct_amplitude_ground_force_reference(
        action, proposal, observation_weights=proposal
    )

    def amplitudes_fn(candidate):
        trainable = projected_resonating_amplitudes(
            candidate, states, base, directions, config, carrier
        )
        return jnp.column_stack((fixed.amplitudes[:, 0], trainable))

    force = direct_amplitude_ground_horizontal_force(
        parameters, reference, amplitudes_fn
    )
    assert np.isfinite(force.energy)
    assert force.gradient_norm > 1.0e-10
    assert reference.maximum_horizontality_error < 1.0e-11
    assert reference.state_weight_ess_fraction > 0.0

    direction = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    direction = {**direction, "pair_gates": jnp.asarray((0.7, -0.4))}
    predicted = sum(
        float(np.sum(np.asarray(gradient) * np.asarray(delta)))
        for gradient, delta in zip(
            jax.tree_util.tree_leaves(force.gradient),
            jax.tree_util.tree_leaves(direction),
            strict=True,
        )
    )

    def energy_at(candidate):
        candidate_action = _combined_action(
            candidate, base, directions, states, carrier, graph, config, fixed
        )
        return solve_three_root_resonating_subspace(
            candidate_action, proposal, observation_weights=proposal
        ).energies[0]

    epsilon = 2.0e-5
    plus = jax.tree_util.tree_map(
        lambda value, delta: value + epsilon * delta, parameters, direction
    )
    minus = jax.tree_util.tree_map(
        lambda value, delta: value - epsilon * delta, parameters, direction
    )
    observed = (energy_at(plus) - energy_at(minus)) / (2.0 * epsilon)
    np.testing.assert_allclose(predicted, observed, rtol=2.0e-4, atol=2.0e-7)
