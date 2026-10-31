import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.causal_residual import compact_tree_action_sequences
from su2_nqs.benchmarks.doob_tangent import decompose_doob_tangent
from su2_nqs.benchmarks.second_order_causal_plan import (
    compile_second_order_causal_pair_plan,
)
from su2_nqs.benchmarks.sparse_causal_recurrence import (
    compile_sparse_causal_recurrence_plan,
)
from su2_nqs.models.compact_tree_gru import CompactTreeGRUConfig
from su2_nqs.models.k2_pair_carrier_tree import (
    K2PairCarrierTreeGRUConfig,
    init_k2_pair_carrier_tree_gru,
    k2_pair_carrier_tree_log_psi,
)
from su2_nqs.su2.fusion_path import enumerate_paths
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.vmc.tangent_quotient import (
    build_exact_tangent_quotient,
    build_matrix_free_tangent_quotient,
)


jax.config.update("jax_enable_x64", True)


def test_matrix_doob_increments_reconstruct_and_are_cross_decision_orthogonal():
    actions = np.asarray(
        ((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.int32
    )
    weights = np.asarray((0.11, 0.23, 0.29, 0.37))
    generator = np.random.default_rng(1201)
    values = generator.normal(size=(4, 3)) + 1j * generator.normal(size=(4, 3))
    result = decompose_doob_tangent(values, weights, actions)
    assert result.reconstruction_maximum_error < 3.0e-15
    assert result.increment_mean_maximum_error < 3.0e-15
    assert result.cross_decision_gram_maximum_error < 3.0e-15
    np.testing.assert_allclose(
        result.mean + np.sum(result.increments, axis=0), values, atol=3.0e-15
    )
    assert result.cumulative_ranks[-1] == np.linalg.matrix_rank(
        np.concatenate((values.real - result.mean.real, values.imag - result.mean.imag))
    )
    assert sum(result.innovation_rank_gains) == result.cumulative_ranks[-1]


def _physical_problem(n: int = 6):
    tree = FusionTree.left_associated(n)
    base = CompactTreeGRUConfig(tree, 0, embedding_dim=4, hidden_dim=8)
    bonds = tuple((site, site + 1, 1.0) for site in range(n - 1)) + tuple(
        (site, site + 2, 0.41) for site in range(n - 2)
    )
    first = compile_sparse_causal_recurrence_plan(n, bonds)
    second = compile_second_order_causal_pair_plan(first)
    config = K2PairCarrierTreeGRUConfig(base, first, second)
    params = init_k2_pair_carrier_tree_gru(jax.random.PRNGKey(1202), config)
    states = jnp.asarray([path[1:] for path in enumerate_paths(n, 0)])
    return config, params, states


def test_exact_quotient_removes_null_space_and_whitens_physical_scores():
    config, params, states = _physical_problem()

    def evaluate(decoder, state):
        return k2_pair_carrier_tree_log_psi(
            {"base": params["base"], "decoder": decoder}, state, config
        )

    logs = jax.vmap(lambda state: evaluate(params["decoder"], state))(states)
    weights = jnp.exp(2.0 * jnp.real(logs))
    quotient = build_exact_tangent_quotient(
        params["decoder"], states, weights, evaluate
    )
    assert quotient.parameter_count > quotient.quotient_rank
    assert quotient.quotient_rank == 2 * (len(states) - 1)
    assert quotient.nullity == quotient.parameter_count - quotient.quotient_rank
    assert quotient.whitening_maximum_error < 2.0e-11

    zero = jnp.zeros((quotient.quotient_rank,), dtype=jnp.float64)
    mapped = quotient.parameters(zero)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(mapped),
        jax.tree_util.tree_leaves(params["decoder"]),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=0.0)

    def quotient_logs_realified(coordinates):
        decoder = quotient.parameters(coordinates)
        values = jax.vmap(lambda state: evaluate(decoder, state))(states)
        return jnp.concatenate((jnp.real(values), jnp.imag(values)))

    realified_score = np.asarray(jax.jacrev(quotient_logs_realified)(zero))
    score = realified_score[: len(states)] + 1j * realified_score[len(states) :]
    actions = compact_tree_action_sequences(
        config.tree, config.target_q, np.asarray(states)
    )
    doob = decompose_doob_tangent(score, np.asarray(weights), actions)
    assert doob.reconstruction_maximum_error < 2.0e-12
    assert doob.increment_mean_maximum_error < 2.0e-12
    assert doob.cross_decision_gram_maximum_error < 2.0e-12
    assert sum(doob.decision_ranks) == quotient.quotient_rank
    assert doob.cumulative_ranks[-1] == quotient.quotient_rank
    assert sum(doob.innovation_rank_gains) == quotient.quotient_rank


def test_matrix_free_quotient_recovers_complete_support_tangent_subspace():
    config, params, states = _physical_problem()

    def evaluate(decoder, state):
        return k2_pair_carrier_tree_log_psi(
            {"base": params["base"], "decoder": decoder}, state, config
        )

    logs = jax.vmap(lambda state: evaluate(params["decoder"], state))(states)
    weights = jnp.exp(2.0 * jnp.real(logs))
    exact = build_exact_tangent_quotient(
        params["decoder"], states, weights, evaluate
    )
    randomized = build_matrix_free_tangent_quotient(
        params["decoder"],
        states,
        weights,
        evaluate,
        jax.random.PRNGKey(1203),
        rank=exact.quotient_rank,
        oversampling=4,
        subspace_iterations=2,
    )
    assert randomized.quotient_rank == exact.quotient_rank
    assert randomized.sampled_whitening_maximum_error < 3.0e-8

    def centered_features(decoder):
        values = jax.vmap(lambda state: evaluate(decoder, state))(states)
        probability = weights / jnp.sum(weights)
        centered = values - jnp.sum(probability * values)
        root = jnp.sqrt(2.0 * probability)
        return jnp.concatenate((root * jnp.real(centered), root * jnp.imag(centered)))

    zero = jnp.zeros((exact.quotient_rank,), dtype=jnp.float64)
    exact_output = np.asarray(
        jax.jacrev(lambda q: centered_features(exact.parameters(q)))(zero)
    )
    randomized_output = np.asarray(
        jax.jacrev(lambda q: centered_features(randomized.parameters(q)))(zero)
    )
    projector_error = np.max(
        np.abs(
            exact_output @ exact_output.T
            - randomized_output @ randomized_output.T
        )
    )
    assert projector_error < 3.0e-8
