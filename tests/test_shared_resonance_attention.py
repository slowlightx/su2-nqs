"""Exact small checks of shared reduced-label attention, not training."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.shared_resonance_attention import (
    ResonanceAttentionConfig, SPECTATOR_LABELS,
    init_shared_resonance_attention, make_resonance_attention_reference,
    resonance_attention_raw, restrict_resonance_generators,
    shared_resonance_generators, shared_resonance_parameter_count,
)
from su2_nqs.models.conditional_carrier_resonance import (
    ACTIVE_DIAGONAL_COORDINATES, conditional_filter_matrix,
    diagonal_conditioned_to_conditional, factorized_to_conditional,
)


def _setup():
    cfg = ResonanceAttentionConfig()
    p = init_shared_resonance_attention(jax.random.PRNGKey(901), cfg)
    offsets = np.array([[2., 0.], [0., 2.]])
    ref = make_resonance_attention_reference(p, offsets, cfg)
    return cfg, p, offsets, ref


def test_exact_identity_with_a_nonzero_head_and_persisted_anchor():
    cfg, p, offsets, ref = _setup()
    assert SPECTATOR_LABELS == ((0, 0), (0, 1), (2, 0), (2, 1), (2, 2), (4, 0))
    assert not ref.flags.writeable
    assert np.linalg.norm(ref) > .01
    assert np.linalg.norm(p["generator_kernel"]) > .01
    assert shared_resonance_parameter_count(cfg) == 730
    assert sum(np.asarray(x).size for x in p.values()) == 730
    for arm in ("full", "diagonal_conditioned", "factorized"):
        values = np.asarray(shared_resonance_generators(p, ref, offsets, cfg, restriction=arm))
        np.testing.assert_array_equal(values, np.zeros((2, 6, 10)))
        np.testing.assert_array_equal(conditional_filter_matrix(values[0]), np.eye(256))
    changed = dict(p)
    changed["generator_bias"] = p["generator_bias"].at[2].add(.05)
    delta = np.asarray(shared_resonance_generators(changed, ref, offsets, cfg))
    assert np.linalg.norm(delta) > .1
    # Re-anchoring would erase the change. The production API requires the
    # initialization reference to be retained, not recomputed each step.
    new_ref = make_resonance_attention_reference(changed, offsets, cfg)
    np.testing.assert_array_equal(shared_resonance_generators(changed, new_ref, offsets, cfg), 0.)


def test_shared_parameters_equal_offsets_and_gate_permutation_covariance():
    cfg, p, offsets, _ = _setup()
    raw = np.asarray(resonance_attention_raw(p, offsets, cfg))
    equal = np.asarray(resonance_attention_raw(p, np.array([[2., 0.], [2., 0.]]), cfg))
    np.testing.assert_array_equal(equal[0], equal[1])
    np.testing.assert_array_equal(resonance_attention_raw(p, offsets[::-1], cfg), raw[::-1])
    assert np.linalg.norm(raw[0] - raw[1]) > 1e-4
    assert np.linalg.norm(raw[:, 0] - raw[:, 1]) > 1e-4


def test_query_key_value_and_every_backbone_leaf_have_nonzero_identity_gradient():
    cfg, p, offsets, ref = _setup()
    weight = jnp.asarray(np.random.default_rng(902).normal(size=(2, 6, 10)))
    def objective(parameters):
        generators = shared_resonance_generators(parameters, ref, offsets, cfg)
        return jnp.sum(weight * generators)
    gradients = jax.grad(objective)(p)
    norms = {name: float(jnp.linalg.norm(value)) for name, value in gradients.items()}
    assert min(norms.values()) > 1e-7
    print("shared attention identity gradient norms", norms)
    # Independent finite differences with the same immutable reference catch
    # a spurious stop_gradient(f_current) anchor and verify Q/K separately.
    for name in ("query_kernel", "key_kernel", "value_kernel", "generator_kernel"):
        direction = np.random.default_rng(len(name)).normal(size=p[name].shape)
        direction /= np.linalg.norm(direction)
        eps = 1e-6
        plus, minus = dict(p), dict(p)
        plus[name] = p[name] + eps * direction
        minus[name] = p[name] - eps * direction
        finite = float((objective(plus) - objective(minus)) / (2 * eps))
        exact = float(jnp.sum(gradients[name] * direction))
        np.testing.assert_allclose(finite, exact, atol=3e-9, rtol=3e-8)


def test_restrictions_are_nested_and_embed_into_the_existing_strong_nulls():
    values = np.random.default_rng(903).normal(size=(2, 6, 10))
    full = np.asarray(restrict_resonance_generators(values, "full"))
    middle = np.asarray(restrict_resonance_generators(values, "diagonal_conditioned"))
    low = np.asarray(restrict_resonance_generators(values, "factorized"))
    np.testing.assert_array_equal(full, values)
    np.testing.assert_allclose(restrict_resonance_generators(middle, "diagonal_conditioned"), middle, atol=1e-15)
    np.testing.assert_allclose(restrict_resonance_generators(low, "diagonal_conditioned"), low, atol=1e-15)
    np.testing.assert_allclose(restrict_resonance_generators(middle, "factorized"), low, atol=1e-15)
    np.testing.assert_allclose(restrict_resonance_generators(low, "factorized"), low, atol=1e-15)
    diag = np.asarray(ACTIVE_DIAGONAL_COORDINATES)
    for gate in range(2):
        base = low[gate].mean(axis=0)
        scalar = (low[gate] - base)[:, diag].mean(axis=1)
        free_low = np.concatenate((base, scalar))
        np.testing.assert_allclose(factorized_to_conditional(free_low), low[gate], atol=1e-15)
        base = middle[gate].mean(axis=0)
        offsets = (middle[gate] - base)[:, diag]
        free_middle = np.concatenate((base, offsets.reshape(-1)))
        np.testing.assert_allclose(diagonal_conditioned_to_conditional(free_middle), middle[gate], atol=1e-15)


def test_attention_generated_filter_commutes_with_independent_pauli_raising():
    cfg, p, offsets, ref = _setup()
    changed = {name: value + .01 * jnp.sin(jnp.arange(value.size).reshape(value.shape))
               for name, value in p.items()}
    generators = shared_resonance_generators(changed, ref, offsets, cfg)
    raising = np.zeros((256, 256))
    for old in range(256):
        for site in range(8):
            bit = 1 << (7 - site)
            if not old & bit:
                raising[old | bit, old] = 1.
    maximum = 0.
    for gate in generators:
        matrix = np.asarray(conditional_filter_matrix(gate))
        maximum = max(maximum, float(np.max(abs(matrix @ raising - raising @ matrix))))
    assert maximum < 2e-13
    print("shared attention generated scalar commutator max", maximum)


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.zeros((0, 2)), np.zeros((2, 2), dtype=complex), np.full((2, 2), np.nan)])
def test_invalid_positions_fail_closed(bad):
    cfg, p, _, _ = _setup()
    with pytest.raises(ValueError):
        resonance_attention_raw(p, bad, cfg)


def test_invalid_reference_parameters_and_restriction_fail_closed():
    cfg, p, offsets, ref = _setup()
    with pytest.raises(ValueError):
        shared_resonance_generators(p, ref[:, :, :9], offsets, cfg)
    with pytest.raises(ValueError):
        shared_resonance_generators(p, ref, offsets, cfg, restriction="magnetic")
    bad = dict(p)
    bad["query_kernel"] = np.zeros((3, 3))
    with pytest.raises(ValueError):
        resonance_attention_raw(bad, offsets, cfg)
    with pytest.raises(ValueError):
        ResonanceAttentionConfig(width=1)
