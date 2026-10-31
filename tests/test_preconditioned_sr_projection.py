import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

jax.config.update("jax_enable_x64", True)

from su2_nqs.benchmarks.sparse_causal_recurrence import (
    compile_sparse_causal_recurrence_plan,
)
from su2_nqs.models.compact_tree_gru import (
    CompactTreeGRUConfig,
    init_compact_tree_gru,
)
from su2_nqs.models.sparse_causal_tree import (
    SparseCausalTreeGRUConfig,
    condition_existing_compact_tree_gru_sparse,
    merge_sparse_causal_parameters,
    partition_sparse_causal_parameters,
    sparse_causal_tree_log_psi,
)
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.vmc.fisher_preconditioner import FisherBlockSpec
from su2_nqs.vmc.preconditioned_sr_projection import (
    centered_complex_log_distance,
    centered_log_derivative_scores,
    preconditioned_conjugate_gradient,
    train_preconditioned_sr_projection,
)


def test_centered_complex_log_distance_removes_global_complex_shift():
    ratios = jnp.asarray([1.0 + 0.2j, -0.5 + 0.4j, 0.1 - 0.3j])
    weights = jnp.asarray([1.0, 2.0, 3.0])
    shifted = ratios + (7.0 - 2.0j)
    observed = centered_complex_log_distance(ratios, weights)
    np.testing.assert_allclose(
        centered_complex_log_distance(shifted, weights), observed, rtol=1e-13
    )
    normalized = np.asarray(weights) / float(jnp.sum(weights))
    centered = np.asarray(ratios) - np.sum(normalized * np.asarray(ratios))
    expected = np.sqrt(2.0 * np.sum(normalized * np.abs(centered) ** 2))
    np.testing.assert_allclose(observed, expected, rtol=1e-13)


def test_pcg_matches_dense_and_jax_identity_cg_with_true_residual():
    generator = np.random.default_rng(921)
    rotation, _ = np.linalg.qr(generator.normal(size=(12, 12)))
    eigenvalues = np.geomspace(3.0, 2.0e-5, 12)
    matrix = rotation @ np.diag(eigenvalues) @ rotation.T
    rhs = generator.normal(size=12)
    matrix_jax = jnp.asarray(matrix)
    rhs_jax = jnp.asarray(rhs)
    expected = np.linalg.solve(matrix, rhs)

    identity = jax.jit(
        lambda value: preconditioned_conjugate_gradient(
            lambda vector: matrix_jax @ vector,
            value,
            tolerance=1.0e-11,
            max_iterations=80,
        )
    )(rhs_jax)
    jax_solution, _ = jax.scipy.sparse.linalg.cg(
        lambda vector: matrix_jax @ vector,
        rhs_jax,
        tol=1.0e-11,
        maxiter=80,
    )
    inverse = jnp.asarray(np.linalg.inv(matrix))
    whitened = preconditioned_conjugate_gradient(
        lambda vector: matrix_jax @ vector,
        rhs_jax,
        preconditioner=lambda vector: inverse @ vector,
        tolerance=1.0e-11,
        max_iterations=80,
    )

    np.testing.assert_allclose(identity.solution, expected, rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(jax_solution, expected, rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(whitened.solution, expected, rtol=3e-10, atol=3e-10)
    direct_residual = np.linalg.norm(matrix @ np.asarray(identity.solution) - rhs)
    np.testing.assert_allclose(
        identity.true_residual_norm, direct_residual, rtol=3e-2, atol=1e-13
    )
    assert bool(identity.converged)
    assert not bool(identity.breakdown)
    assert int(whitened.iterations) <= 2
    assert int(whitened.iterations) < int(identity.iterations)


def _n6_problem():
    n_sites = 6
    bonds = tuple((site, site + 1, 1.0, 0) for site in range(n_sites - 1)) + tuple(
        (site, site + 2, 0.5, 1) for site in range(n_sites - 2)
    )
    tree = FusionTree.left_associated(n_sites)
    plan = compile_sparse_causal_recurrence_plan(n_sites, bonds)
    base_config = CompactTreeGRUConfig(
        tree, 0, embedding_dim=2, hidden_dim=4
    )
    base_params = init_compact_tree_gru(jax.random.PRNGKey(600), base_config)
    config = SparseCausalTreeGRUConfig(
        base_config, plan, message_hidden_dim=3
    )
    full_params = condition_existing_compact_tree_gru_sparse(
        base_params, jax.random.PRNGKey(601), config
    )
    trainable, frozen = partition_sparse_causal_parameters(full_params, config)
    states = jnp.asarray(
        [state.labels for state in tree.enumerate_states(0)], dtype=jnp.int32
    )

    def log_one(candidate, state):
        return sparse_causal_tree_log_psi(
            merge_sparse_causal_parameters(candidate, frozen), state, config
        )

    return trainable, states, log_one


def _centered_features(flat, unravel, samples, weights, log_one):
    normalized = weights / jnp.sum(weights)
    square_root = jnp.sqrt(2.0 * normalized)
    logs = jax.vmap(lambda state: log_one(unravel(flat), state))(samples)
    centered = logs - jnp.sum(normalized * logs)
    return jnp.concatenate(
        (square_root * jnp.real(centered), square_root * jnp.imag(centered))
    )


def test_n6_exact_physical_and_identity_projection_match_dense_normal_solve():
    trainable, states, log_one = _n6_problem()
    flat, unravel = ravel_pytree(trainable)
    base_logs = jax.vmap(lambda state: log_one(trainable, state))(states)
    weights = jnp.exp(2.0 * jnp.real(base_logs))
    weights = weights / jnp.sum(weights)
    target_ratios = jnp.asarray(
        [
            0.04 + 0.01j,
            -0.02 + 0.03j,
            0.01 - 0.04j,
            -0.03 - 0.01j,
            0.02 + 0.025j,
        ],
        dtype=jnp.complex128,
    )
    damping = 8.0e-2
    features = lambda values: _centered_features(
        values, unravel, states, weights, log_one
    )
    current = np.asarray(features(flat))
    target_logs = base_logs + target_ratios
    target_centered = target_logs - jnp.sum(weights * target_logs)
    target_features = np.concatenate(
        [
            np.asarray(jnp.sqrt(2.0 * weights) * jnp.real(target_centered)),
            np.asarray(jnp.sqrt(2.0 * weights) * jnp.imag(target_centered)),
        ]
    )
    jacobian = np.asarray(jax.jacrev(features)(flat))
    rhs = jacobian.T @ (target_features - current)
    normal = jacobian.T @ jacobian + damping * np.eye(flat.size)
    dense_direction = np.linalg.solve(normal, rhs)

    common = dict(
        sample_pool=states,
        pool_importance_weights=weights,
        pool_base_logs=base_logs,
        pool_target_log_ratios=target_ratios,
        log_psi_fn=log_one,
        steps=1,
        batch_size=len(states),
        damping=damping,
        step_size=0.05,
        cg_tolerance=2.0e-11,
        cg_max_iterations=200,
        backtracking_steps=4,
        random_batches=False,
    )
    identity_params, identity_history = train_preconditioned_sr_projection(
        trainable,
        jax.random.PRNGKey(602),
        solver_control="identity",
        **common,
    )
    physical_params, physical_history = train_preconditioned_sr_projection(
        trainable,
        jax.random.PRNGKey(602),
        solver_control="physical_block",
        **common,
    )
    identity_flat, _ = ravel_pytree(identity_params)
    physical_flat, _ = ravel_pytree(physical_params)
    identity_step = identity_history[0]["effective_step_size"]
    physical_step = physical_history[0]["effective_step_size"]
    assert identity_step > 0.0
    assert physical_step > 0.0
    identity_direction = (np.asarray(identity_flat) - np.asarray(flat)) / identity_step
    physical_direction = (np.asarray(physical_flat) - np.asarray(flat)) / physical_step

    np.testing.assert_allclose(identity_direction, dense_direction, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(physical_direction, dense_direction, rtol=2e-8, atol=2e-9)
    for row, control in (
        (identity_history[0], "identity"),
        (physical_history[0], "physical_block"),
    ):
        assert row["solver_control"] == control
        assert row["cg_converged"]
        assert not row["cg_breakdown"]
        assert row["true_relative_normal_residual"] < 2.0e-11
        assert row["feature_loss_after"] < row["feature_loss_before"]
    assert identity_history[0]["preconditioner_stored_matrix_elements"] == 0
    assert physical_history[0]["preconditioner_block_sizes"] == {
        "qop2_endpoint_site_delta": [18, 6],
        "closed_scalar_heads": [1, 6],
        "hidden_scalar_heads": [1, 6],
    }


def test_weighted_scores_are_centered_and_energy_backtracking_tests_smaller_steps():
    params = jnp.zeros(2, dtype=jnp.float64)
    samples = jnp.asarray([[0], [1], [0], [1]], dtype=jnp.int32)
    weights = jnp.asarray([0.1, 0.2, 0.3, 0.4])

    def log_psi(logits, sample):
        return 0.5 * jax.nn.log_softmax(logits)[sample[0]] + 0j

    scores = centered_log_derivative_scores(params, samples, weights, log_psi)
    np.testing.assert_allclose(
        np.sum(np.asarray(weights)[:, None] * np.asarray(scores), axis=0),
        0.0,
        atol=2e-16,
    )
    score_calls = []

    def energy_score(logits):
        score_calls.append(np.asarray(logits))
        if len(score_calls) == 1:
            return 0.0
        if len(score_calls) == 2:
            return 1.0
        return -1.0

    trained, history = train_preconditioned_sr_projection(
        params,
        jax.random.PRNGKey(603),
        sample_pool=samples,
        pool_importance_weights=jnp.ones(4),
        pool_base_logs=jnp.zeros(4, dtype=jnp.complex128),
        pool_target_log_ratios=jnp.asarray(
            [0.4, -0.4, 0.4, -0.4], dtype=jnp.complex128
        ),
        log_psi_fn=log_psi,
        steps=1,
        batch_size=4,
        solver_control="physical_block",
        block_specs=(FisherBlockSpec(name="all", paths=((),)),),
        damping=1.0e-3,
        step_size=4.0,
        cg_max_iterations=20,
        backtracking_steps=10,
        acceptance_score_fn=energy_score,
        random_batches=False,
    )
    assert history[0]["accepted"]
    assert history[0]["energy_backtracking_evaluations"] > 1
    assert 0.0 < history[0]["effective_step_size"] < 4.0
    assert np.linalg.norm(np.asarray(trained)) > 0.0


def test_output_trust_region_caps_actual_centered_log_displacement():
    params = jnp.zeros(2, dtype=jnp.float64)
    samples = jnp.asarray([[0], [1]], dtype=jnp.int32)

    def log_psi(logits, sample):
        return 0.5 * jax.nn.log_softmax(logits)[sample[0]] + 0j

    radius = 1.0e-2
    trained, history = train_preconditioned_sr_projection(
        params,
        jax.random.PRNGKey(604),
        sample_pool=samples,
        pool_importance_weights=jnp.ones(2),
        pool_base_logs=jnp.zeros(2, dtype=jnp.complex128),
        pool_target_log_ratios=jnp.asarray(
            [4.0, -4.0], dtype=jnp.complex128
        ),
        log_psi_fn=log_psi,
        steps=1,
        batch_size=2,
        solver_control="identity",
        damping=1.0e-6,
        step_size=100.0,
        cg_max_iterations=20,
        backtracking_steps=12,
        maximum_feature_distance=radius,
        random_batches=False,
    )
    row = history[0]
    assert row["output_trust_scale"] < 1.0
    assert row["raw_full_step_feature_norm"] > radius
    assert row["selected_feature_distance"] <= radius * (1.0 + 1.0e-10)
    assert row["maximum_feature_distance"] == radius
    assert row["feature_loss_after"] < row["feature_loss_before"]
    assert np.linalg.norm(np.asarray(trained)) > 0.0


def test_physical_block_step_is_explicitly_one_approximate_direction():
    params = jnp.zeros(2, dtype=jnp.float64)
    samples = jnp.asarray([[0], [1]], dtype=jnp.int32)

    def log_psi(logits, sample):
        return 0.5 * jax.nn.log_softmax(logits)[sample[0]] + 0j

    trained, history = train_preconditioned_sr_projection(
        params,
        jax.random.PRNGKey(606),
        sample_pool=samples,
        pool_importance_weights=jnp.ones(2),
        pool_base_logs=jnp.zeros(2, dtype=jnp.complex128),
        pool_target_log_ratios=jnp.asarray(
            [0.2, -0.2], dtype=jnp.complex128
        ),
        log_psi_fn=log_psi,
        steps=1,
        batch_size=2,
        solver_control="physical_block_step",
        block_specs=(FisherBlockSpec(name="all", paths=((),)),),
        damping=1.0e-3,
        step_size=0.1,
        backtracking_steps=1,
        random_batches=False,
    )
    row = history[0]
    assert row["solver_control"] == "physical_block_step"
    assert row["cg_iterations"] == 1
    assert not row["cg_breakdown"]
    assert np.isfinite(row["true_relative_normal_residual"])
    assert row["feature_loss_after"] < row["feature_loss_before"]
    assert np.linalg.norm(np.asarray(trained)) > 0.0


def test_deflated_physical_projection_matches_dense_normal_solve():
    trainable, states, log_one = _n6_problem()
    flat, unravel = ravel_pytree(trainable)
    base_logs = jax.vmap(lambda state: log_one(trainable, state))(states)
    weights = jnp.exp(2.0 * jnp.real(base_logs))
    weights = weights / jnp.sum(weights)
    target_ratios = jnp.asarray(
        [
            0.04 + 0.01j,
            -0.02 + 0.03j,
            0.01 - 0.04j,
            -0.03 - 0.01j,
            0.02 + 0.025j,
        ],
        dtype=jnp.complex128,
    )
    damping = 8.0e-2
    features = lambda values: _centered_features(
        values, unravel, states, weights, log_one
    )
    current = np.asarray(features(flat))
    target_logs = base_logs + target_ratios
    target_centered = target_logs - jnp.sum(weights * target_logs)
    target_features = np.concatenate(
        [
            np.asarray(jnp.sqrt(2.0 * weights) * jnp.real(target_centered)),
            np.asarray(jnp.sqrt(2.0 * weights) * jnp.imag(target_centered)),
        ]
    )
    jacobian = np.asarray(jax.jacrev(features)(flat))
    rhs = jacobian.T @ (target_features - current)
    normal = jacobian.T @ jacobian + damping * np.eye(flat.size)
    dense_direction = np.linalg.solve(normal, rhs)

    trained, history = train_preconditioned_sr_projection(
        trainable,
        jax.random.PRNGKey(607),
        sample_pool=states,
        pool_importance_weights=weights,
        pool_base_logs=base_logs,
        pool_target_log_ratios=target_ratios,
        log_psi_fn=log_one,
        steps=1,
        batch_size=len(states),
        solver_control="physical_block_deflated",
        damping=damping,
        step_size=0.05,
        cg_tolerance=2.0e-11,
        cg_max_iterations=100,
        deflation_rank=8,
        backtracking_steps=4,
        random_batches=False,
    )
    trained_flat, _ = ravel_pytree(trained)
    effective_step = history[0]["effective_step_size"]
    direction = (np.asarray(trained_flat) - np.asarray(flat)) / effective_step
    np.testing.assert_allclose(direction, dense_direction, rtol=2e-8, atol=2e-9)
    assert history[0]["cg_converged"]
    assert not history[0]["cg_breakdown"]
    assert history[0]["true_relative_normal_residual"] < 2.0e-11
    assert history[0]["deflation_rank"] == 8
    assert history[0]["deflation_active_rank"] > 0
    assert history[0]["deflation_minimum_coarse_eigenvalue"] > 0.0
    assert history[0]["deflation_stored_matrix_elements"] > 0

    refined, refined_history = train_preconditioned_sr_projection(
        trainable,
        jax.random.PRNGKey(607),
        sample_pool=states,
        pool_importance_weights=weights,
        pool_base_logs=base_logs,
        pool_target_log_ratios=target_ratios,
        log_psi_fn=log_one,
        steps=1,
        batch_size=len(states),
        solver_control="physical_block_deflated_refined",
        damping=damping,
        step_size=0.05,
        cg_tolerance=1.0e-8,
        cg_max_iterations=20,
        deflation_rank=8,
        refinement_cycles=6,
        backtracking_steps=4,
        random_batches=False,
    )
    refined_flat, _ = ravel_pytree(refined)
    refined_step = refined_history[0]["effective_step_size"]
    refined_direction = (
        np.asarray(refined_flat) - np.asarray(flat)
    ) / refined_step
    np.testing.assert_allclose(
        refined_direction, dense_direction, rtol=3e-6, atol=1e-8
    )
    assert refined_history[0]["cg_converged"]
    assert refined_history[0]["true_relative_normal_residual"] < 1.0e-8
    assert refined_history[0]["refinement_cycles"] >= 1
    assert refined_history[0]["refinement_true_relative_residuals"][-1] < 1.0e-8


def test_output_trust_region_rejects_invalid_radius():
    with np.testing.assert_raises_regex(ValueError, "maximum_feature_distance"):
        train_preconditioned_sr_projection(
            jnp.zeros(1),
            jax.random.PRNGKey(605),
            sample_pool=jnp.zeros((1, 1), dtype=jnp.int32),
            pool_importance_weights=jnp.ones(1),
            pool_base_logs=jnp.zeros(1, dtype=jnp.complex128),
            pool_target_log_ratios=jnp.zeros(1, dtype=jnp.complex128),
            log_psi_fn=lambda p, x: p[0] + 0j,
            steps=1,
            batch_size=1,
            solver_control="identity",
            maximum_feature_distance=0.0,
        )


def test_deflated_gmres_projection_uses_original_residual_gate():
    params = jnp.zeros(2, dtype=jnp.float64)
    samples = jnp.asarray([[0], [1]], dtype=jnp.int32)

    def log_psi(logits, sample):
        return 0.5 * jax.nn.log_softmax(logits)[sample[0]] + 0j

    trained, history = train_preconditioned_sr_projection(
        params,
        jax.random.PRNGKey(608),
        sample_pool=samples,
        pool_importance_weights=jnp.ones(2),
        pool_base_logs=jnp.zeros(2, dtype=jnp.complex128),
        pool_target_log_ratios=jnp.asarray(
            [0.2, -0.2], dtype=jnp.complex128
        ),
        log_psi_fn=log_psi,
        steps=1,
        batch_size=2,
        solver_control="physical_block_deflated_gmres",
        block_specs=(FisherBlockSpec(name="all", paths=((),)),),
        damping=1.0e-3,
        step_size=0.1,
        cg_tolerance=1.0e-10,
        deflation_rank=2,
        gmres_restart=2,
        gmres_cycles=2,
        backtracking_steps=1,
        random_batches=False,
    )
    row = history[0]
    assert row["solver_control"] == "physical_block_deflated_gmres"
    assert row["cg_converged"]
    assert not row["cg_breakdown"]
    assert row["true_relative_normal_residual"] < 1.0e-10
    assert row["gmres_restart"] == 2
    assert row["gmres_cycles"] == 2
    assert row["gmres_info"] >= 0
    assert np.linalg.norm(np.asarray(trained)) > 0.0
