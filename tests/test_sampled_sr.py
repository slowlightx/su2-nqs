import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.vmc.sampled_sr import train_sampled_sr_projection


def test_sampled_matrix_free_sr_moves_logits_toward_teacher():
    samples = jnp.asarray([[0], [1], [0], [1]], dtype=jnp.int32)
    base = jnp.asarray([0.0, 0.0])
    target_ratio = jnp.asarray([0.4, -0.4, 0.4, -0.4])

    def log_psi(logits, sample):
        return 0.5 * jax.nn.log_softmax(logits)[sample[0]] + 0j

    trained, history = train_sampled_sr_projection(
        base,
        jax.random.PRNGKey(0),
        sample_pool=samples,
        pool_importance_weights=jnp.ones(4),
        pool_base_logs=jnp.zeros(4, dtype=jnp.complex64),
        pool_target_log_ratios=target_ratio.astype(jnp.complex64),
        log_psi_fn=log_psi,
        steps=2,
        batch_size=4,
        damping=1e-3,
        step_size=0.5,
        cg_max_iterations=10,
    )
    assert trained[0] > trained[1]
    assert history[-1]["feature_loss_after"] < history[0]["feature_loss_before"]
    assert 0.0 <= history[-1]["effective_step_size"] <= 0.5
    assert np.isfinite(history[-1]["cg_relative_residual"])
    assert history[-1]["cg_relative_residual"] < 1.0e-4


def test_sampled_sr_rejects_zero_damping():
    try:
        train_sampled_sr_projection(
            jnp.zeros(2), jax.random.PRNGKey(0),
            sample_pool=jnp.zeros((1, 1), dtype=jnp.int32),
            pool_importance_weights=jnp.ones(1),
            pool_base_logs=jnp.zeros(1, dtype=jnp.complex64),
            pool_target_log_ratios=jnp.zeros(1, dtype=jnp.complex64),
            log_psi_fn=lambda p, x: p[0] + 0j,
            steps=1, batch_size=1, damping=0.0,
        )
    except ValueError:
        return
    raise AssertionError("zero damping was accepted")


def test_identity_preconditioner_keyword_preserves_default_outputs_bitwise():
    samples = jnp.asarray([[0], [1], [0], [1]], dtype=jnp.int32)
    target_ratio = jnp.asarray([0.3, -0.3, 0.3, -0.3], dtype=jnp.complex64)

    def log_psi(logits, sample):
        return 0.5 * jax.nn.log_softmax(logits)[sample[0]] + 0j

    kwargs = dict(
        sample_pool=samples,
        pool_importance_weights=jnp.ones(4),
        pool_base_logs=jnp.zeros(4, dtype=jnp.complex64),
        pool_target_log_ratios=target_ratio,
        log_psi_fn=log_psi,
        steps=2,
        batch_size=4,
        damping=1.0e-3,
        cg_max_iterations=10,
    )
    default_params, default_history = train_sampled_sr_projection(
        jnp.zeros(2), jax.random.PRNGKey(71), **kwargs
    )
    identity_params, identity_history = train_sampled_sr_projection(
        jnp.zeros(2),
        jax.random.PRNGKey(71),
        fisher_preconditioner="identity",
        **kwargs,
    )
    np.testing.assert_array_equal(default_params, identity_params)
    assert default_history == identity_history


def test_opt_in_exact_carrier_head_blocks_use_bounded_score_chunks():
    context_width = 2
    params = {
        "probability_endpoint_head": jnp.zeros((2, 3, context_width)),
        "phase_endpoint_head": jnp.zeros((2, 3, context_width)),
        "probability_closed_head": jnp.zeros((2, context_width)),
        "phase_closed_head": jnp.zeros((2, context_width)),
    }
    samples = jnp.arange(8, dtype=jnp.int32)[:, None]

    def log_psi(decoder, sample):
        index = sample[0].astype(jnp.float32) + 1.0
        endpoint_feature = jnp.sin(
            index * (jnp.arange(12, dtype=jnp.float32) + 1.0) * 0.13
        ).reshape(2, 3, context_width)
        closed_feature = jnp.cos(
            index * (jnp.arange(4, dtype=jnp.float32) + 1.0) * 0.19
        ).reshape(2, context_width)
        probability = (
            jnp.sum(decoder["probability_endpoint_head"] * endpoint_feature)
            + jnp.sum(decoder["probability_closed_head"] * closed_feature)
        )
        phase = (
            jnp.sum(decoder["phase_endpoint_head"] * endpoint_feature)
            + jnp.sum(decoder["phase_closed_head"] * closed_feature)
        )
        return probability + 1j * phase

    target = jnp.asarray(
        [0.08, -0.03, 0.04, -0.07, 0.02, 0.06, -0.05, 0.01],
        dtype=jnp.complex64,
    ) * (1.0 + 0.4j)
    projected, history = train_sampled_sr_projection(
        params,
        jax.random.PRNGKey(72),
        sample_pool=samples,
        pool_importance_weights=jnp.linspace(0.5, 1.2, len(samples)),
        pool_base_logs=jnp.zeros(len(samples), dtype=jnp.complex64),
        pool_target_log_ratios=target,
        log_psi_fn=log_psi,
        steps=1,
        batch_size=len(samples),
        damping=2.0e-2,
        step_size=0.1,
        cg_tolerance=1.0e-10,
        cg_max_iterations=80,
        random_batches=False,
        fisher_preconditioner="physical_block",
        fisher_score_chunk_size=3,
    )
    row = history[0]
    parameter_count = sum(value.size for value in params.values())
    complex_itemsize = 2 * params["probability_endpoint_head"].dtype.itemsize
    assert row["solver_control"] == "physical_block"
    assert row["preconditioner_score_chunk_shape"] == [3, parameter_count]
    assert (
        row["preconditioner_score_chunk_bytes"]
        == 3 * parameter_count * complex_itemsize
    )
    assert row["preconditioner_real_imag_jacobian_shape"] == [
        3,
        2,
        parameter_count,
    ]
    assert (
        row["preconditioner_estimated_peak_score_workspace_bytes"]
        == 6 * parameter_count * complex_itemsize
    )
    assert row["preconditioner_largest_block_dimension"] == 2 * context_width
    assert row["preconditioner_block_sizes"] == {
        "exact_qop2_endpoint_site_delta": [6, 2 * context_width],
        "exact_closed_decision": [2, 2 * context_width],
    }
    assert np.isfinite(row["cg_relative_residual"])
    assert row["feature_loss_after"] < row["feature_loss_before"]
    assert any(
        np.linalg.norm(np.asarray(value)) > 0.0
        for value in projected.values()
    )


def test_energy_trust_region_continues_to_smaller_feature_improving_step():
    samples = jnp.asarray([[0], [1], [0], [1]], dtype=jnp.int32)

    def log_psi(logits, sample):
        return 0.5 * jax.nn.log_softmax(logits)[sample[0]] + 0j

    score_calls = []

    def score(logits):
        score_calls.append(np.asarray(logits))
        # Call one initializes the accepted score.  Reject the first tested
        # feature-improving candidate and accept the next smaller one.
        if len(score_calls) == 1:
            return 0.0
        if len(score_calls) == 2:
            return 1.0
        return -1.0

    trained, history = train_sampled_sr_projection(
        jnp.zeros(2),
        jax.random.PRNGKey(1),
        sample_pool=samples,
        pool_importance_weights=jnp.ones(4),
        pool_base_logs=jnp.zeros(4, dtype=jnp.complex64),
        pool_target_log_ratios=jnp.asarray([0.4, -0.4, 0.4, -0.4], dtype=jnp.complex64),
        log_psi_fn=log_psi,
        steps=1,
        batch_size=4,
        damping=1e-3,
        step_size=4.0,
        cg_max_iterations=10,
        backtracking_steps=10,
        acceptance_score_fn=score,
    )
    assert history[0]["accepted"]
    assert history[0]["energy_backtracking_evaluations"] > 1
    assert 0.0 < history[0]["effective_step_size"] < 4.0


def test_output_trust_region_is_exposed_by_sampled_sr_api():
    samples = jnp.asarray([[0], [1]], dtype=jnp.int32)

    def log_psi(logits, sample):
        return 0.5 * jax.nn.log_softmax(logits)[sample[0]] + 0j

    radius = 1.0e-2
    trained, history = train_sampled_sr_projection(
        jnp.zeros(2, dtype=jnp.float64),
        jax.random.PRNGKey(73),
        sample_pool=samples,
        pool_importance_weights=jnp.ones(2),
        pool_base_logs=jnp.zeros(2, dtype=jnp.complex128),
        pool_target_log_ratios=jnp.asarray([4.0, -4.0], dtype=jnp.complex128),
        log_psi_fn=log_psi,
        steps=1,
        batch_size=2,
        damping=1.0e-6,
        step_size=100.0,
        cg_max_iterations=20,
        backtracking_steps=12,
        maximum_feature_distance=radius,
        random_batches=False,
    )
    row = history[0]
    assert row["solver_control"] == "identity"
    assert row["output_trust_scale"] < 1.0
    assert row["selected_feature_distance"] <= radius * (1.0 + 1.0e-10)
    assert row["feature_loss_after"] < row["feature_loss_before"]
    assert np.linalg.norm(np.asarray(trained)) > 0.0
