from functools import lru_cache

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from su2_nqs.benchmarks.sparse_causal_recurrence import (
    compile_sparse_causal_recurrence_plan,
)
from su2_nqs.models.compact_tree_gru import CompactTreeGRUConfig
from su2_nqs.models.sparse_causal_tree import (
    SparseCausalTreeGRUConfig,
    init_sparse_causal_tree_gru,
    merge_sparse_causal_parameters,
    partition_sparse_causal_parameters,
    sparse_causal_tree_log_psi,
)
from su2_nqs.operators.permutation import heisenberg_graph_fusion_matrix
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.vmc.lsqr_projection import (
    centered_real_log_features,
    solve_linearized_projection_direction,
    train_lsqr_projection,
)
from su2_nqs.vmc.sampled_sr import train_sampled_sr_projection


@lru_cache(maxsize=1)
def _n6_full_support_problem():
    n = 6
    j2 = 0.47
    bonds = tuple((site, site + 1, 1.0) for site in range(n - 1)) + tuple(
        (site, site + 2, j2) for site in range(n - 2)
    )
    tagged_bonds = tuple(
        (left, right, coupling, int(right - left == 2))
        for left, right, coupling in bonds
    )
    plan = compile_sparse_causal_recurrence_plan(n, tagged_bonds)
    base = CompactTreeGRUConfig(
        FusionTree.left_associated(n), 0, embedding_dim=1, hidden_dim=2
    )
    config = SparseCausalTreeGRUConfig(base, plan, message_hidden_dim=1)
    params = init_sparse_causal_tree_gru(jax.random.PRNGKey(20260901), config)
    trainable, frozen = partition_sparse_causal_parameters(params, config)
    states = jnp.asarray(
        [state.labels for state in config.tree.enumerate_states(0)],
        dtype=jnp.int32,
    )

    def log_one(candidate, state):
        full = merge_sparse_causal_parameters(candidate, frozen)
        return sparse_causal_tree_log_psi(full, state, config)

    log_all = jax.jit(
        lambda candidate: jax.vmap(lambda state: log_one(candidate, state))(states)
    )
    base_logs = np.asarray(log_all(trainable), dtype=np.complex128)
    base_state = np.exp(base_logs)
    np.testing.assert_allclose(np.linalg.norm(base_state), 1.0, atol=2e-12)
    hamiltonian = heisenberg_graph_fusion_matrix(
        n, bonds, target_q=0
    ).toarray()
    target = base_state - 0.035 * (hamiltonian @ base_state)
    target /= np.linalg.norm(target)
    assert np.all(np.abs(target) > 1e-12)
    target_logs = np.log(np.abs(target)) + 1j * np.angle(target)
    target_log_ratios = target_logs - base_logs
    np.testing.assert_allclose(
        np.exp(base_logs + target_log_ratios), target, rtol=2e-14, atol=2e-14
    )
    weights = np.abs(base_state) ** 2
    weights /= np.sum(weights)
    return (
        trainable,
        states,
        jnp.asarray(weights),
        jnp.asarray(base_logs),
        jnp.asarray(target_log_ratios),
        log_one,
        hamiltonian,
    )


def _target_features(base_logs, target_log_ratios, weights):
    normalized_weights = weights / jnp.sum(weights)
    target_logs = base_logs + target_log_ratios
    centered = target_logs - jnp.sum(normalized_weights * target_logs)
    scale = jnp.sqrt(2.0 * normalized_weights)
    return jnp.concatenate((scale * jnp.real(centered), scale * jnp.imag(centered)))


def test_n6_full_support_lsqr_direction_matches_dense_and_normal_equation_cg():
    (
        trainable,
        states,
        weights,
        base_logs,
        target_log_ratios,
        log_one,
        _,
    ) = _n6_full_support_problem()
    flat, unravel = ravel_pytree(trainable)
    feature_fn = lambda values: centered_real_log_features(
        unravel(values), states, weights, log_one
    )
    target = _target_features(base_logs, target_log_ratios, weights)
    damping = 1.0e-3

    lsqr = solve_linearized_projection_direction(
        flat,
        feature_fn,
        target,
        damping=damping,
        relative_tolerance=1e-12,
        absolute_tolerance=1e-14,
        max_iterations=30,
    )
    jacobian = np.asarray(jax.jacfwd(feature_fn)(flat), dtype=np.float64)
    residual = np.asarray(target - feature_fn(flat), dtype=np.float64)
    augmented = np.vstack(
        (
            jacobian,
            np.sqrt(damping) * np.eye(jacobian.shape[1], dtype=np.float64),
        )
    )
    augmented_rhs = np.concatenate(
        (residual, np.zeros(jacobian.shape[1], dtype=np.float64))
    )
    dense_direction = np.linalg.lstsq(augmented, augmented_rhs, rcond=None)[0]

    _, pullback = jax.vjp(feature_fn, flat)

    def normal_product(vector):
        _, tangent = jax.jvp(feature_fn, (flat,), (vector,))
        return pullback(tangent)[0] + damping * vector

    cg_rhs = pullback(target - feature_fn(flat))[0]
    cg_direction, _ = jax.scipy.sparse.linalg.cg(
        normal_product, cg_rhs, tol=1e-12, maxiter=30
    )

    assert bool(lsqr.converged)
    assert not bool(lsqr.breakdown)
    np.testing.assert_allclose(
        np.asarray(lsqr.solution), dense_direction, rtol=2e-8, atol=2e-9
    )
    np.testing.assert_allclose(
        np.asarray(lsqr.solution), np.asarray(cg_direction), rtol=3e-8, atol=3e-9
    )
    exact_primal = np.linalg.norm(jacobian @ dense_direction - residual)
    exact_normal = np.linalg.norm(
        jacobian.T @ (jacobian @ dense_direction - residual)
        + damping * dense_direction
    )
    np.testing.assert_allclose(
        float(lsqr.primal_residual_norm), exact_primal, rtol=2e-10, atol=2e-12
    )
    np.testing.assert_allclose(
        float(lsqr.normal_residual_norm), exact_normal, rtol=2e-2, atol=2e-13
    )


def test_n6_lsqr_and_existing_cg_use_same_target_and_energy_backtracking_contract():
    (
        trainable,
        states,
        weights,
        base_logs,
        target_log_ratios,
        log_one,
        hamiltonian,
    ) = _n6_full_support_problem()

    def exact_energy(candidate):
        logs = jax.vmap(lambda state: log_one(candidate, state))(states)
        vector = np.asarray(jnp.exp(logs))
        return float(
            np.real(np.vdot(vector, hamiltonian @ vector) / np.vdot(vector, vector))
        )

    common = {
        "sample_pool": states,
        "pool_importance_weights": weights,
        "pool_base_logs": base_logs,
        "pool_target_log_ratios": target_log_ratios,
        "log_psi_fn": log_one,
        "steps": 1,
        "batch_size": len(states),
        "damping": 1.0e-3,
        "step_size": 0.25,
        "backtracking_steps": 8,
        "acceptance_score_fn": exact_energy,
        "random_batches": False,
    }
    lsqr_params, lsqr_history = train_lsqr_projection(
        trainable,
        jax.random.PRNGKey(71),
        **common,
        lsqr_tolerance=1e-12,
        lsqr_absolute_tolerance=1e-14,
        lsqr_max_iterations=30,
    )
    cg_params, cg_history = train_sampled_sr_projection(
        trainable,
        jax.random.PRNGKey(71),
        **common,
        cg_tolerance=1e-12,
        cg_max_iterations=30,
    )

    assert lsqr_history[0]["linear_solver"] == "matrix_free_golub_kahan_lsqr"
    assert lsqr_history[0]["lsqr_converged"]
    assert not lsqr_history[0]["lsqr_breakdown"]
    assert lsqr_history[0]["accepted"] == cg_history[0]["accepted"]
    assert (
        lsqr_history[0]["effective_step_size"]
        == cg_history[0]["effective_step_size"]
    )
    assert lsqr_history[0]["feature_loss_after"] < lsqr_history[0]["feature_loss_before"]
    assert len(lsqr_history[0]["primal_residual_history"]) == (
        lsqr_history[0]["lsqr_iterations"] + 1
    )
    assert len(lsqr_history[0]["normal_residual_history"]) == (
        lsqr_history[0]["lsqr_iterations"] + 1
    )
    for lsqr_leaf, cg_leaf in zip(
        jax.tree_util.tree_leaves(lsqr_params),
        jax.tree_util.tree_leaves(cg_params),
        strict=True,
    ):
        np.testing.assert_allclose(lsqr_leaf, cg_leaf, rtol=4e-8, atol=4e-9)


def test_projection_rejects_zero_damping():
    with np.testing.assert_raises_regex(ValueError, "damping"):
        train_lsqr_projection(
            jnp.zeros(2),
            jax.random.PRNGKey(0),
            sample_pool=jnp.zeros((1, 1), dtype=jnp.int32),
            pool_importance_weights=jnp.ones(1),
            pool_base_logs=jnp.zeros(1, dtype=jnp.complex128),
            pool_target_log_ratios=jnp.zeros(1, dtype=jnp.complex128),
            log_psi_fn=lambda p, x: p[0] + 0j,
            steps=1,
            batch_size=1,
            damping=0.0,
        )
