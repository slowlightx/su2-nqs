import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.causal_residual import compact_tree_action_sequences
from su2_nqs.models.compact_tree_gru import CompactTreeGRUConfig
from su2_nqs.models.hamiltonian_conditioned_tree import (
    HamiltonianBond,
    HamiltonianConditionedTreeGRUConfig,
    fixed_conditioned_path_data,
    hamiltonian_conditioned_tree_probabilities,
    init_hamiltonian_conditioned_tree_gru,
)
from su2_nqs.operators.tree_boundary_svd import (
    Q_OP_VECTOR,
    compile_tree_boundary_svd,
    permute_tree_boundary_bonds,
)
from su2_nqs.su2.fusion_tree import FusionTree


def _coupling_matrix(n, bonds, shell, channel_shells=None):
    matrix = np.zeros((n, n), dtype=np.float64)
    for bond in bonds:
        bond_shell = (
            bond.channel if channel_shells is None else channel_shells[bond.channel]
        )
        if bond_shell == shell:
            matrix[bond.first, bond.second] += bond.coupling
            matrix[bond.second, bond.first] += bond.coupling
    return matrix


def test_tree_boundary_svd_reconstructs_shells_and_uses_doubled_vector_rank():
    tree = FusionTree.balanced(4)
    bonds = (
        HamiltonianBond(0, 2, 1.0, 0),
        HamiltonianBond(1, 3, 1.0, 0),
        HamiltonianBond(0, 1, 0.4, 1),
        HamiltonianBond(2, 3, 0.4, 1),
    )
    decomposition = compile_tree_boundary_svd(tree, bonds)
    assert decomposition.maximum_reconstruction_error < 1.0e-14
    assert all(channel.q_op == Q_OP_VECTOR == 2 for channel in decomposition.channels)
    channel_shells = {
        channel.channel: channel.shell for channel in decomposition.channels
    }
    for shell in (0, 1):
        np.testing.assert_allclose(
            _coupling_matrix(
                4, decomposition.bonds, shell, channel_shells=channel_shells
            ),
            _coupling_matrix(4, bonds, shell),
            atol=1.0e-14,
        )
    root = [
        row for row in decomposition.group_rows
        if row["tree_depth"] == 0 and row["shell"] == 0
    ]
    assert len(root) == 1
    assert root[0]["spatial_rank"] == 2


def test_endpoint_permutation_preserves_parameter_channels_and_coefficients():
    tree = FusionTree.balanced(4)
    decomposition = compile_tree_boundary_svd(
        tree,
        (
            HamiltonianBond(0, 2, 1.0, 0),
            HamiltonianBond(1, 3, -0.3, 0),
        ),
    )
    control = permute_tree_boundary_bonds(decomposition, (2, 0, 3, 1))
    assert [bond.channel for bond in control] == [
        bond.channel for bond in decomposition.bonds
    ]
    np.testing.assert_array_equal(
        [bond.coupling for bond in control],
        [bond.coupling for bond in decomposition.bonds],
    )


def test_rank_cut_conditioner_is_normalized_and_prefix_causal():
    tree = FusionTree.balanced(8)
    raw = tuple(
        HamiltonianBond(site, (site + 1) % 8, 1.0, 0) for site in range(8)
    )
    decomposition = compile_tree_boundary_svd(tree, raw)
    base = CompactTreeGRUConfig(tree, 0, embedding_dim=3, hidden_dim=6)
    config = HamiltonianConditionedTreeGRUConfig(
        base,
        decomposition.bonds,
        n_channels=decomposition.n_channels,
        local_embedding_dim=3,
        local_hidden_dim=5,
    )
    params = init_hamiltonian_conditioned_tree_gru(jax.random.PRNGKey(991), config)
    params["local"]["probability_head"] = 0.2 * jnp.ones_like(
        params["local"]["probability_head"]
    )
    labels = np.asarray([state.labels for state in tree.enumerate_states(0)])
    probabilities = hamiltonian_conditioned_tree_probabilities(
        params, jnp.asarray(labels), config
    )
    np.testing.assert_allclose(float(probabilities.sum()), 1.0, atol=2.0e-12)

    actions = compact_tree_action_sequences(tree, 0, labels)
    path_data = jax.vmap(
        lambda state: fixed_conditioned_path_data(params, state, config)
    )(jnp.asarray(labels))
    features = np.asarray(path_data[-1])
    maximum_error = 0.0
    for decision in range(actions.shape[1]):
        references = {}
        for state_index, action_row in enumerate(actions):
            prefix = tuple(int(value) for value in action_row[: decision + 1])
            value = features[state_index, decision]
            if prefix in references:
                maximum_error = max(
                    maximum_error,
                    float(np.max(np.abs(value - references[prefix]), initial=0.0)),
                )
            else:
                references[prefix] = value
    assert maximum_error < 1.0e-13
