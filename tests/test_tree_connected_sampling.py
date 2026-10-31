import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.connected_importance import (
    hamiltonian_connected_transition,
    mixed_connected_proposal_distribution,
)
from su2_nqs.models.compact_tree_gru import (
    CompactTreeGRUConfig,
    compact_tree_log_psi,
    init_compact_tree_gru,
)
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.tree_rotations import recoupling_matrix_sparse
from su2_nqs.vmc.tree_connected_sampling import (
    TreeConnectedTransitionCache,
    mixed_tree_connected_probabilities,
    sample_mixed_tree_connected,
)


def _system():
    tree = FusionTree.from_nested(((0, 1), (2, (3, (4, 5)))))
    config = CompactTreeGRUConfig(tree, 0, 3, 6)
    params = init_compact_tree_gru(jax.random.PRNGKey(1201), config)
    states = tree.enumerate_states(0)
    labels = jnp.asarray([state.labels for state in states], dtype=jnp.int32)
    bonds = (
        (0, 1, 1.0),
        (1, 2, 0.7),
        (2, 3, -0.2),
        (3, 4, 0.5),
        (4, 5, 1.1),
        (5, 0, 0.9),
        (1, 4, 0.3),
    )
    recoupling, _ = recoupling_matrix_sparse(
        FusionTree.left_associated(6), tree, 0
    )
    hamiltonian = (
        recoupling
        @ heisenberg_graph_fusion_matrix(6, bonds, target_q=0)
        @ recoupling.T
    ).tocsc()
    return tree, config, params, states, labels, bonds, hamiltonian


def test_reverse_local_tree_proposal_matches_complete_transition():
    tree, config, params, states, labels, bonds, hamiltonian = _system()
    logs = jax.vmap(
        lambda state: compact_tree_log_psi(params, state, config)
    )(labels)
    base = np.asarray(jnp.exp(2.0 * jnp.real(logs)))
    transition = hamiltonian_connected_transition(
        hamiltonian, amplitude_power=2.0
    )
    expected = mixed_connected_proposal_distribution(
        base, transition, mixture_weight=0.5
    )
    cache = TreeConnectedTransitionCache(
        tree, bonds, target_q=0, amplitude_power=2.0
    )
    proposal, observed_base, diagnostics = mixed_tree_connected_probabilities(
        params,
        [state.labels for state in states],
        config,
        cache,
        mixture_weight=0.5,
    )
    np.testing.assert_allclose(observed_base, base, atol=2.0e-14)
    np.testing.assert_allclose(proposal, expected, atol=2.0e-13)
    assert diagnostics["unique_probability_evaluation_count"] == len(states)
    assert cache.cached_state_count == len(states)


def test_tree_connected_sampler_has_exact_weights_and_frequencies():
    tree, config, params, states, labels, bonds, hamiltonian = _system()
    logs = jax.vmap(
        lambda state: compact_tree_log_psi(params, state, config)
    )(labels)
    base = np.asarray(jnp.exp(2.0 * jnp.real(logs)))
    expected = mixed_connected_proposal_distribution(
        base,
        hamiltonian_connected_transition(hamiltonian, amplitude_power=2.0),
        mixture_weight=0.5,
    )
    cache = TreeConnectedTransitionCache(
        tree, bonds, target_q=0, amplitude_power=2.0
    )
    result = sample_mixed_tree_connected(
        params,
        jax.random.PRNGKey(1202),
        30_000,
        config,
        cache,
        mixture_weight=0.5,
    )
    positions = {state.labels: index for index, state in enumerate(states)}
    counts = np.zeros(len(states), dtype=np.int64)
    for row in result.labels:
        counts[positions[tuple(map(int, row))]] += 1
    frequencies = counts / np.sum(counts)
    np.testing.assert_allclose(frequencies, expected, atol=0.012)
    np.testing.assert_allclose(
        result.importance_weights,
        result.target_probabilities / result.proposal_probabilities,
        atol=1.0e-14,
    )
    assert result.maximum_importance_weight <= 2.0 + 1.0e-12
    assert result.effective_sample_size_fraction > 0.0
    assert cache.cached_state_count == len(states)
