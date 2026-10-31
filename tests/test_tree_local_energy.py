import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.models.compact_tree_gru import (
    CompactTreeGRUConfig,
    compact_tree_log_psi,
    init_compact_tree_gru,
)
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.tree_rotations import recoupling_matrix_sparse
from su2_nqs.vmc.tree_local_energy import (
    build_padded_tree_connectivity,
    compact_tree_local_energy_batch,
    compact_tree_local_energy_padded_batch,
    evaluate_padded_compact_tree_local_energy,
    padded_compact_tree_importance_vmc_energy_gradient,
    padded_compact_tree_vmc_energy_gradient,
)


def test_compact_tree_local_energy_matches_complete_recoupled_action():
    n, target_q = 8, 0
    tree = FusionTree.from_nested(((0, (1, 2)), ((3, 4), (5, (6, 7)))))
    config = CompactTreeGRUConfig(tree, target_q, 3, 7)
    params = init_compact_tree_gru(jax.random.PRNGKey(812), config)
    states = tree.enumerate_states(target_q)
    labels = jnp.asarray([state.labels for state in states], dtype=jnp.int32)
    logs = np.asarray(
        jax.vmap(lambda state: compact_tree_log_psi(params, state, config))(labels)
    )
    amplitudes = np.exp(logs)
    bonds = tuple(
        (site, (site + 1) % n, 1.0 + 0.03 * site) for site in range(n)
    ) + ((0, 5, -0.27), (2, 7, 0.41))
    recoupling, _ = recoupling_matrix_sparse(
        FusionTree.left_associated(n), tree, target_q
    )
    hamiltonian = (
        recoupling
        @ heisenberg_graph_fusion_matrix(n, bonds, target_q=target_q)
        @ recoupling.T
    ).toarray()
    expected = hamiltonian @ amplitudes / amplitudes

    result = compact_tree_local_energy_batch(params, labels, config, bonds)
    np.testing.assert_allclose(result.local_energies, expected, atol=2.0e-12)
    probabilities = np.abs(amplitudes) ** 2
    expected_energy = np.vdot(amplitudes, hamiltonian @ amplitudes)
    actual_energy = np.sum(probabilities * result.local_energies)
    np.testing.assert_allclose(actual_energy, expected_energy, atol=2.0e-12)
    assert result.total_connectivity_entries == int(
        np.sum(result.connectivity_widths)
    )
    assert result.unique_state_count <= result.total_connectivity_entries


def test_compact_tree_local_energy_accepts_duplicate_sampled_states():
    tree = FusionTree.balanced(4)
    config = CompactTreeGRUConfig(tree, 0, 2, 4)
    params = init_compact_tree_gru(jax.random.PRNGKey(413), config)
    labels = np.asarray(tree.enumerate_states(0)[0].labels, dtype=np.int32)
    batch = np.stack((labels, labels))
    result = compact_tree_local_energy_batch(
        params, batch, config, ((0, 1, 1.0), (2, 3, 1.0))
    )
    np.testing.assert_allclose(
        result.local_energies[0], result.local_energies[1], atol=1.0e-14
    )


def test_padded_tree_local_energy_matches_variable_exact_workload():
    tree = FusionTree.from_nested(((0, 1), (2, (3, (4, 5)))))
    config = CompactTreeGRUConfig(tree, 0, 3, 6)
    params = init_compact_tree_gru(jax.random.PRNGKey(673), config)
    labels = np.asarray(
        [state.labels for state in tree.enumerate_states(0)], dtype=np.int32
    )
    bonds = (
        (0, 1, 1.0),
        (1, 2, 0.7),
        (2, 3, -0.2),
        (3, 4, 0.5),
        (4, 5, 1.1),
        (5, 0, 0.9),
        (1, 4, 0.3),
    )
    variable = compact_tree_local_energy_batch(params, labels, config, bonds)
    maximum_width = int(np.max(variable.connectivity_widths)) + 3
    connectivity = build_padded_tree_connectivity(
        labels, config, bonds, maximum_width
    )
    assert connectivity.maximum_width == maximum_width
    assert np.all(np.sum(connectivity.mask, axis=1) == variable.connectivity_widths)
    assert np.all(connectivity.coefficients[~connectivity.mask] == 0.0)
    padded = evaluate_padded_compact_tree_local_energy(
        params, connectivity, config
    )
    np.testing.assert_allclose(
        padded.local_energies, variable.local_energies, atol=2.0e-12
    )
    replay = evaluate_padded_compact_tree_local_energy(
        params, connectivity, config
    )
    np.testing.assert_allclose(
        replay.local_energies, variable.local_energies, atol=2.0e-12
    )


def test_padded_tree_local_energy_refuses_connectivity_truncation():
    tree = FusionTree.balanced(6)
    config = CompactTreeGRUConfig(tree, 0, 2, 4)
    params = init_compact_tree_gru(jax.random.PRNGKey(721), config)
    labels = np.asarray(
        [state.labels for state in tree.enumerate_states(0)], dtype=np.int32
    )
    bonds = tuple((site, (site + 1) % 6, 1.0) for site in range(6))
    variable = compact_tree_local_energy_batch(params, labels, config, bonds)
    required = int(np.max(variable.connectivity_widths))
    with np.testing.assert_raises_regex(ValueError, "exceeds registered"):
        compact_tree_local_energy_padded_batch(
            params, labels, config, bonds, required - 1
        )


def test_exhaustive_padded_vmc_gradient_matches_dense_energy_gradient():
    tree = FusionTree.balanced(6)
    config = CompactTreeGRUConfig(tree, 0, 3, 5)
    params = init_compact_tree_gru(jax.random.PRNGKey(887), config)
    states = tree.enumerate_states(0)
    labels = jnp.asarray([state.labels for state in states], dtype=jnp.int32)
    bonds = tuple((site, (site + 1) % 6, 1.0) for site in range(6)) + (
        (0, 3, 0.4),
        (1, 4, -0.2),
    )
    left_hamiltonian = heisenberg_graph_fusion_matrix(
        6, bonds, target_q=0
    )
    recoupling, _ = recoupling_matrix_sparse(
        FusionTree.left_associated(6), tree, 0
    )
    hamiltonian = jnp.asarray(
        (recoupling @ left_hamiltonian @ recoupling.T).toarray()
    )
    connectivity = build_padded_tree_connectivity(
        labels, config, bonds, maximum_width=64
    )
    actual = padded_compact_tree_vmc_energy_gradient(
        params, connectivity, config, born_weighted=True
    )

    def dense_energy(candidate):
        logs = jax.vmap(
            lambda state: compact_tree_log_psi(candidate, state, config)
        )(labels)
        amplitudes = jnp.exp(logs)
        return jnp.real(jnp.vdot(amplitudes, hamiltonian @ amplitudes))

    expected_energy, expected_gradient = jax.value_and_grad(dense_energy)(params)
    expected_logs = jax.vmap(
        lambda state: compact_tree_log_psi(params, state, config)
    )(labels)
    expected_amplitudes = jnp.exp(expected_logs)
    expected_local = hamiltonian @ expected_amplitudes / expected_amplitudes
    expected_variance = jnp.sum(
        jnp.abs(expected_amplitudes) ** 2
        * jnp.abs(expected_local - expected_energy) ** 2
    )
    np.testing.assert_allclose(actual.energy, expected_energy, atol=2.0e-12)
    np.testing.assert_allclose(actual.variance, expected_variance, atol=2.0e-12)
    np.testing.assert_allclose(actual.enumerated_normalization, 1.0, atol=2.0e-12)
    for observed, expected in zip(
        jax.tree_util.tree_leaves(actual.gradient),
        jax.tree_util.tree_leaves(expected_gradient),
        strict=True,
    ):
        np.testing.assert_allclose(observed, expected, atol=2.0e-11)


def test_importance_weighted_padded_gradient_matches_exact_born_weights():
    tree = FusionTree.balanced(6)
    config = CompactTreeGRUConfig(tree, 0, 3, 5)
    params = init_compact_tree_gru(jax.random.PRNGKey(977), config)
    labels = jnp.asarray(
        [state.labels for state in tree.enumerate_states(0)], dtype=jnp.int32
    )
    bonds = tuple((site, (site + 1) % 6, 1.0) for site in range(6))
    connectivity = build_padded_tree_connectivity(
        labels, config, bonds, maximum_width=64
    )
    logs = jax.vmap(
        lambda state: compact_tree_log_psi(params, state, config)
    )(labels)
    probabilities = jnp.exp(2.0 * jnp.real(logs))
    expected = padded_compact_tree_vmc_energy_gradient(
        params, connectivity, config, born_weighted=True
    )
    actual = padded_compact_tree_importance_vmc_energy_gradient(
        params, connectivity, config, probabilities
    )
    np.testing.assert_allclose(actual.energy, expected.energy, atol=2.0e-12)
    np.testing.assert_allclose(actual.variance, expected.variance, atol=2.0e-12)
    for observed, reference in zip(
        jax.tree_util.tree_leaves(actual.gradient),
        jax.tree_util.tree_leaves(expected.gradient),
        strict=True,
    ):
        np.testing.assert_allclose(observed, reference, atol=2.0e-11)
