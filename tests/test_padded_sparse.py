import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from su2_nqs.models.hubbard_fusion_gru import (
    HubbardFusionGRUConfig,
    hubbard_log_psi,
    init_hubbard_fusion_gru,
    sample_hubbard_fusion,
)
from su2_nqs.operators.hubbard import hubbard_reduced_fusion_sparse
from su2_nqs.su2.hubbard_fusion import encode_hubbard_fusion_state
from su2_nqs.vmc.padded_sparse import (
    padded_connectivity,
    padded_sparse_importance_sr_step_diagnostics,
    train_padded_sparse_importance_sr,
    train_padded_sparse_sr,
)


def test_padded_connectivity_preserves_columns():
    matrix = sparse.csr_matrix(np.asarray([[1.0, 0.0], [2.0, 3.0]]))
    neighbors, elements, valid = padded_connectivity(matrix)
    np.testing.assert_array_equal(neighbors[0, valid[0]], [0, 1])
    np.testing.assert_allclose(elements[0, valid[0]], [1.0, 2.0])
    np.testing.assert_array_equal(neighbors[1, valid[1]], [1])
    np.testing.assert_allclose(elements[1, valid[1]], [3.0])


def test_padded_sparse_sampled_sr_lowers_exact_hubbard_energy():
    matrix, states = hubbard_reduced_fusion_sparse(
        2, 2, 0, ((0, 1),), interaction=4.0
    )
    basis = tuple(encode_hubbard_fusion_state(state) for state in states)
    config = HubbardFusionGRUConfig(2, 2, 0, embedding_dim=3, hidden_dim=5)
    params = init_hubbard_fusion_gru(jax.random.PRNGKey(30), config)
    log_fn = lambda candidate, state: hubbard_log_psi(candidate, state, config)
    sample_fn = lambda candidate, key, count: sample_hubbard_fusion(
        candidate, key, count, config
    )

    def exact_energy(candidate):
        logs = jax.vmap(lambda state: log_fn(candidate, state))(jnp.asarray(basis))
        vector = np.exp(np.asarray(logs))
        return float(np.real(np.vdot(vector, matrix @ vector)))

    initial_energy = exact_energy(params)
    observed_steps = []
    sampled_steps = []

    def observer(candidate, step):
        observed_steps.append(step)
        return {"exact_energy": exact_energy(candidate)}

    def sample_observer(_candidate, samples, step):
        sampled_steps.append(step)
        return {"sample_count": int(samples.shape[0])}

    trained, history = train_padded_sparse_sr(
        params,
        jax.random.PRNGKey(31),
        matrix=matrix,
        basis=basis,
        sample_fn=sample_fn,
        log_psi_fn=log_fn,
        steps=5,
        batch_size=256,
        learning_rate=0.1,
        damping=1e-3,
        cg_max_iterations=100,
        observer=observer,
        observe_every=2,
        sample_observer=sample_observer,
    )
    assert exact_energy(trained) < initial_energy
    assert len(history) == 5
    assert all(np.isfinite(item["cg_relative_residual"]) for item in history)
    assert all(0.0 < item["unique_sample_fraction"] <= 1.0 for item in history)
    assert observed_steps == [2, 4, 5]
    assert sampled_steps == [1, 2, 3, 4, 5]
    assert all(item["sample_observation"]["sample_count"] == 256 for item in history)
    assert [int(item["step"]) for item in history if "observation" in item] == [2, 4, 5]
    assert all(
        np.isfinite(item["observation"]["exact_energy"])
        for item in history
        if "observation" in item
    )


def test_padded_sparse_sr_rejects_update_that_fails_residual_gate():
    matrix, states = hubbard_reduced_fusion_sparse(
        2, 2, 0, ((0, 1),), interaction=4.0
    )
    basis = tuple(encode_hubbard_fusion_state(state) for state in states)
    config = HubbardFusionGRUConfig(2, 2, 0, embedding_dim=3, hidden_dim=5)
    params = init_hubbard_fusion_gru(jax.random.PRNGKey(40), config)
    log_fn = lambda candidate, state: hubbard_log_psi(candidate, state, config)
    sample_fn = lambda candidate, key, count: sample_hubbard_fusion(
        candidate, key, count, config
    )
    trained, history = train_padded_sparse_sr(
        params,
        jax.random.PRNGKey(41),
        matrix=matrix,
        basis=basis,
        sample_fn=sample_fn,
        log_psi_fn=log_fn,
        steps=5,
        batch_size=256,
        learning_rate=0.1,
        damping=1e-3,
        cg_max_iterations=1,
        maximum_cg_relative_residual=1e-30,
    )
    assert len(history) == 1
    assert not history[0]["update_accepted"]
    assert history[0]["termination_reason"] == "cg_relative_residual_gate"
    for initial_leaf, trained_leaf in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(trained)
    ):
        np.testing.assert_array_equal(trained_leaf, initial_leaf)


def test_importance_sr_uses_target_over_proposal_weights():
    matrix = sparse.diags([0.0, 1.0], format="csr")
    basis = ((0,), (1,))
    params = {"theta": jnp.asarray(-0.5, dtype=jnp.float64)}

    def log_fn(candidate, state):
        logits = jnp.asarray([candidate["theta"], -candidate["theta"]])
        return 0.5 * jax.nn.log_softmax(logits)[state[0]] + 0.0j

    def proposal_sample_fn(candidate, _key, count):
        assert count == 2
        samples = jnp.asarray(basis, dtype=jnp.int32)
        probabilities = jnp.exp(
            2.0 * jnp.real(jax.vmap(lambda state: log_fn(candidate, state))(samples))
        )
        # Each support point appears once, hence the empirical proposal is
        # uniform and p/q = 2p. Self-normalization must recover exact p.
        return samples, 2.0 * probabilities, {"kind": "exhaustive_uniform"}

    def exact_energy(candidate):
        samples = jnp.asarray(basis, dtype=jnp.int32)
        probabilities = jnp.exp(
            2.0 * jnp.real(jax.vmap(lambda state: log_fn(candidate, state))(samples))
        )
        return float(probabilities[1])

    initial_energy = exact_energy(params)
    trained, history = train_padded_sparse_importance_sr(
        params,
        jax.random.PRNGKey(50),
        matrix=matrix,
        basis=basis,
        proposal_sample_fn=proposal_sample_fn,
        log_psi_fn=log_fn,
        steps=5,
        batch_size=2,
        learning_rate=0.1,
        damping=1e-3,
        cg_max_iterations=100,
        maximum_cg_relative_residual=1e-8,
        maximum_predicted_feature_step=0.02,
        maximum_actual_feature_step=0.02,
        feature_backtracking_steps=8,
    )
    assert exact_energy(trained) < initial_energy
    assert len(history) == 5
    assert all(item["update_accepted"] for item in history)
    assert all(item["proposal_observation"]["kind"] == "exhaustive_uniform" for item in history)
    assert all(1.0 <= item["importance_effective_sample_size"] <= 2.0 for item in history)
    assert all(item["predicted_feature_step"] <= 0.02 + 1e-12 for item in history)
    assert all(item["actual_feature_step"] <= 0.02 + 1e-12 for item in history)
    assert all(item["trust_region_step_found"] for item in history)
    assert any(item["step_scale"] < 1.0 for item in history)


def test_importance_sr_step_diagnostics_is_invariant_to_weight_scale():
    matrix = sparse.diags([0.0, 1.0], format="csr")
    basis = ((0,), (1,))
    params = {"theta": jnp.asarray(-0.5, dtype=jnp.float64)}

    def log_fn(candidate, state):
        logits = jnp.asarray([candidate["theta"], -candidate["theta"]])
        return 0.5 * jax.nn.log_softmax(logits)[state[0]] + 0.0j

    samples = jnp.asarray(basis, dtype=jnp.int32)
    common = dict(
        matrix=matrix,
        basis=basis,
        samples=samples,
        log_psi_fn=log_fn,
        learning_rate=0.1,
        damping=1e-3,
        cg_tolerance=1e-10,
        cg_max_iterations=100,
        maximum_predicted_feature_step=0.02,
        maximum_actual_feature_step=0.02,
    )
    first_params, first = padded_sparse_importance_sr_step_diagnostics(
        params, importance_weights=jnp.asarray([0.4, 1.6]), **common
    )
    second_params, second = padded_sparse_importance_sr_step_diagnostics(
        params, importance_weights=jnp.asarray([4.0, 16.0]), **common
    )
    for name in ("gradient", "fisher_probe_product", "natural_direction"):
        np.testing.assert_allclose(first[name], second[name], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        first_params["theta"], second_params["theta"], atol=1e-12, rtol=1e-12
    )


def test_importance_sr_step_diagnostics_accepts_a_separate_fisher_sample():
    matrix = sparse.csr_matrix(np.asarray([[0.0, 0.3], [0.3, 1.0]]))
    basis = ((0,), (1,))
    params = {"theta": jnp.asarray(-0.5, dtype=jnp.float64)}

    def log_fn(candidate, state):
        logits = jnp.asarray([candidate["theta"], -candidate["theta"]])
        return 0.5 * jax.nn.log_softmax(logits)[state[0]] + 0.0j

    samples = jnp.asarray(((0,), (1,), (0,), (1,)), dtype=jnp.int32)
    weights = jnp.asarray([0.4, 1.6, 0.4, 1.6], dtype=jnp.float64)
    common = dict(
        matrix=matrix,
        basis=basis,
        samples=samples,
        importance_weights=weights,
        log_psi_fn=log_fn,
        learning_rate=0.1,
        damping=1e-3,
        cg_tolerance=1e-10,
        cg_max_iterations=100,
    )
    ordinary_params, ordinary = padded_sparse_importance_sr_step_diagnostics(
        params, **common
    )
    split_params, split = padded_sparse_importance_sr_step_diagnostics(
        params,
        fisher_samples=samples,
        fisher_importance_weights=weights,
        **common,
    )
    for name in ("gradient", "fisher_probe_product", "natural_direction"):
        np.testing.assert_allclose(
            ordinary[name], split[name], atol=1e-12, rtol=1e-12
        )
    np.testing.assert_allclose(
        ordinary_params["theta"], split_params["theta"], atol=1e-12, rtol=1e-12
    )
    assert split["gradient_sample_count"] == 4
    assert split["fisher_sample_count"] == 4
