"""Multi-environment generator tests and an independent N12 Pauli oracle."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.multicontext_resonance_attention import (
    MulticontextResonanceAttentionConfig, complete_environment_labels,
    free_multicontext_generators, init_multicontext_resonance_attention,
    interaction_contrast, make_multicontext_reference, multicontext_active_filters,
    multicontext_attention_for_labels, multicontext_attention_raw, multicontext_parameter_count,
    multicontext_resonance_generators, permute_environment_positions,
    restrict_multicontext_generators,
)
from su2_nqs.models.conditional_carrier_resonance import local_scalar_components


def _setup():
    cfg = MulticontextResonanceAttentionConfig()
    p = init_multicontext_resonance_attention(jax.random.PRNGKey(10201), cfg)
    positions = np.array([[-2., 0.], [0., 2.]])
    ref = make_multicontext_reference(p, positions, cfg)
    rng = np.random.default_rng(10202)
    direction = {k: rng.normal(size=v.shape) for k, v in p.items()}
    norm = np.sqrt(sum(np.sum(x*x) for x in direction.values()))
    direction = {k: jnp.asarray(v / norm) for k, v in direction.items()}
    moved = {k: v + .1 * direction[k] for k, v in p.items()}
    return cfg, p, moved, positions, ref, direction


def test_complete_contexts_identity_free_table_and_two_environment_dependence():
    cfg, p, moved, positions, ref, _ = _setup()
    labels = complete_environment_labels(cfg)
    assert labels.shape == (36, 2, 2)
    assert len({tuple(x.reshape(-1)) for x in labels}) == 36
    assert multicontext_parameter_count(cfg) == 730
    assert not ref.flags.writeable
    for control in ("full", "one_environment", "additive"):
        np.testing.assert_array_equal(multicontext_resonance_generators(
            p, ref, positions, cfg, control=control), np.zeros((6, 6, 10)))
    filters = np.asarray(multicontext_active_filters(np.zeros(360), cfg))
    np.testing.assert_array_equal(filters, np.broadcast_to(np.eye(16), (6, 6, 16, 16)))
    table = np.asarray(multicontext_resonance_generators(moved, ref, positions, cfg))
    np.testing.assert_array_equal(free_multicontext_generators(table.reshape(360), cfg), table)
    assert np.linalg.norm(table - table.mean(axis=0, keepdims=True)) > 1e-5
    assert np.linalg.norm(table - table.mean(axis=1, keepdims=True)) > 1e-5
    values = np.asarray(multicontext_attention_for_labels(moved, labels, positions, cfg))
    np.testing.assert_array_equal(values.reshape(6, 6, 10), multicontext_attention_raw(moved, positions, cfg))


def test_genuine_nonadditive_generator_survives_anchor_and_stronger_additive_control():
    cfg, _, moved, positions, ref, _ = _setup()
    full = np.asarray(multicontext_resonance_generators(moved, ref, positions, cfg))
    additive = np.asarray(restrict_multicontext_generators(full, cfg, control="additive"))
    contrast = np.asarray(interaction_contrast(full, cfg))
    direct = full - full.mean(axis=0, keepdims=True) - full.mean(axis=1, keepdims=True) + full.mean(axis=(0, 1), keepdims=True)
    np.testing.assert_allclose(contrast, direct, atol=1e-17)
    np.testing.assert_allclose(full, additive + contrast, atol=1e-17)
    np.testing.assert_allclose(contrast.mean(axis=0), 0., atol=1e-17)
    np.testing.assert_allclose(contrast.mean(axis=1), 0., atol=1e-17)
    ratio = np.linalg.norm(contrast) / np.linalg.norm(full)
    assert ratio > 1e-3
    assert abs(np.vdot(additive, contrast)) < 1e-15
    np.testing.assert_allclose(interaction_contrast(additive, cfg), 0., atol=1e-17)
    for index in (0, 1):
        one = np.asarray(restrict_multicontext_generators(full, cfg, control="one_environment", environment_index=index))
        np.testing.assert_allclose(one, np.broadcast_to(one.mean(axis=1-index, keepdims=True), one.shape), atol=1e-17)
        np.testing.assert_allclose(restrict_multicontext_generators(one, cfg, control="additive"), one, atol=1e-17)
        np.testing.assert_allclose(interaction_contrast(one, cfg), 0., atol=1e-17)
    print("anchored generator interaction contrast fraction", ratio)


def test_token_order_covariance_is_not_the_position_assignment_control():
    cfg, p, moved, positions, ref, _ = _setup()
    labels = complete_environment_labels(cfg)
    permuted_positions = permute_environment_positions(positions, (1, 0))
    ordinary = np.asarray(multicontext_attention_for_labels(moved, labels, positions, cfg))
    relabelled = np.asarray(multicontext_attention_for_labels(moved, labels[:, ::-1], permuted_positions, cfg))
    np.testing.assert_allclose(ordinary, relabelled, atol=2e-15)
    control_ref = make_multicontext_reference(p, permuted_positions, cfg)
    np.testing.assert_array_equal(multicontext_resonance_generators(p, control_ref, permuted_positions, cfg), 0.)
    physical = np.asarray(multicontext_resonance_generators(moved, ref, positions, cfg))
    assignment = np.asarray(multicontext_resonance_generators(moved, control_ref, permuted_positions, cfg))
    assert np.linalg.norm(assignment - physical) > 1e-5
    np.testing.assert_allclose(assignment, physical.swapaxes(0, 1), atol=2e-15)


def test_joint_context_jvp_and_backbone_gradients_use_frozen_reference():
    cfg, p, _, positions, ref, direction = _setup()
    weights = jnp.asarray(np.random.default_rng(10203).normal(size=(6, 6, 10)))
    objective = lambda theta: jnp.sum(weights * interaction_contrast(
        multicontext_resonance_generators(theta, ref, positions, cfg), cfg))
    grad = jax.grad(objective)(p)
    for name in ("query_kernel", "key_kernel", "value_kernel", "generator_kernel",
                 "hidden_kernel", "q_embedding", "a_embedding", "position_kernel"):
        assert np.linalg.norm(np.asarray(grad[name])) > 1e-6
    exact = sum(float(jnp.sum(grad[k] * direction[k])) for k in p)
    eps = 1e-6
    plus = {k: v + eps * direction[k] for k, v in p.items()}
    minus = {k: v - eps * direction[k] for k, v in p.items()}
    finite = float((objective(plus) - objective(minus)) / (2 * eps))
    np.testing.assert_allclose(finite, exact, atol=5e-9, rtol=3e-7)
    print("mixed-context gradient FD absolute error", abs(finite - exact))


def _raising(vector, n_sites):
    output = np.zeros_like(vector)
    states = np.arange(1 << n_sites)
    for site in range(n_sites):
        mask = 1 << (n_sites - 1 - site)
        old = states[(states & mask) == 0]
        output[old | mask] += vector[old]
    return output


def test_complete_multiplet_controlled_operator_is_scalar_with_independent_n12_pauli_actions():
    cfg, _, moved, positions, ref, _ = _setup()
    table = multicontext_resonance_generators(moved, ref, positions, cfg)
    active = np.asarray(multicontext_active_filters(table, cfg))
    projectors = local_scalar_components().spectator_projectors
    def action(vector):
        # Three disjoint four-site cells A,B,D; no dense4096x4096 matrix.
        return np.einsum("zij,zwkl,wmn,jln->ikm", projectors, active,
                         projectors, vector.reshape(16, 16, 16), optimize=True).reshape(-1)
    rng = np.random.default_rng(10204)
    vector = rng.normal(size=4096) + 1j * rng.normal(size=4096)
    vector /= np.linalg.norm(vector)
    raised_error = np.max(abs(action(_raising(vector, 12)) - _raising(action(vector), 12)))
    # Sz is diagonal with doubled integer labels underlying each bit.
    sz = np.asarray([int(s).bit_count() - 6 for s in range(4096)])
    z_error = np.max(abs(action(sz * vector) - sz * action(vector)))
    assert raised_error < 5e-14 and z_error < 5e-14
    identity = np.einsum("zij,kl,wmn,jln->ikm", projectors, np.eye(16),
                         projectors, vector.reshape(16, 16, 16), optimize=True).reshape(-1)
    np.testing.assert_allclose(identity, vector, atol=3e-16)
    print("N12 controlled-operator Pauli commutator max", max(raised_error, z_error))


def test_more_than_two_environments_and_explicit_enumeration_budget():
    cfg = MulticontextResonanceAttentionConfig(num_environments=3)
    p = init_multicontext_resonance_attention(jax.random.PRNGKey(10205), cfg)
    positions = np.array([[-2., 0.], [0., 2.], [2., 0.]])
    table = multicontext_attention_raw(p, positions, cfg)
    assert table.shape == (6, 6, 6, 10)
    assert multicontext_parameter_count(cfg) == 730
    additive = restrict_multicontext_generators(table, cfg, control="additive")
    np.testing.assert_allclose(interaction_contrast(additive, cfg), 0., atol=2e-15)
    limited = MulticontextResonanceAttentionConfig(num_environments=3, max_table_contexts=36)
    with pytest.raises(ValueError):
        multicontext_attention_raw(p, positions, limited)
    assert multicontext_attention_for_labels(p, complete_environment_labels(cfg)[:2], positions, limited).shape == (2, 10)


@pytest.mark.parametrize("labels", [np.zeros((1, 2, 3), dtype=int),
    np.zeros((1, 2, 2), dtype=float), np.array([[[1, 0], [0, 0]]]),
    np.array([[[0, 2], [0, 0]]]), np.array([[[4, 1], [0, 0]]])])
def test_invalid_quantum_labels_fail_closed(labels):
    cfg, p, _, positions, _, _ = _setup()
    with pytest.raises(ValueError):
        multicontext_attention_for_labels(p, labels, positions, cfg)


def test_invalid_dynamic_labels_propagate_nan_not_a_clipped_valid_sector():
    cfg, p, _, positions, _, _ = _setup()
    evaluate = jax.jit(lambda labels: multicontext_attention_for_labels(p, labels, positions, cfg))
    assert np.isnan(np.asarray(evaluate(jnp.array([[[5, 0], [0, 0]]])))).all()
    with pytest.raises(ValueError):
        free_multicontext_generators(np.zeros(359), cfg)
    with pytest.raises(ValueError):
        free_multicontext_generators(np.full(360, np.nan), cfg)
    with pytest.raises(ValueError):
        MulticontextResonanceAttentionConfig(num_environments=1)
    with pytest.raises(ValueError):
        permute_environment_positions(positions, (0, 0))
