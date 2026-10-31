"""Independent full-bit-space Reynolds references for the query compiler."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.conditional_carrier_resonance import (
    compile_conditional_carrier_queries, contract_conditional_carrier_queries,
    evaluate_conditional_carrier_queries, conditional_carrier_amplitudes,
)
from su2_nqs.models.conditional_carrier_resonance import conditional_filter_matrix


def _bits(n):
    return ((np.arange(1 << n)[:, None] >> np.arange(n-1, -1, -1)) & 1).astype(np.int8)


def _integers(bits):
    return np.asarray(bits)@(1 << np.arange(bits.shape[-1]-1, -1, -1))


def _group(n):
    return tuple(sorted({tuple((sign*np.arange(n)+shift) % n) for sign in (-1, 1) for shift in range(n)}))


def _project(vector, bits, group):
    return np.mean(np.stack([vector[_integers(bits[:, g])] for g in group]), axis=0)


def _local_dense_action(vector, matrix, sites, n):
    # Independent tensor-axis contraction, not packed connectivity/gather.
    order = tuple(sites)+tuple(i for i in range(n) if i not in sites)
    front = np.transpose(vector.reshape((2,)*n), order).reshape(256, -1)
    result = (matrix@front).reshape((2,)*n)
    return np.transpose(result, np.argsort(order)).reshape(-1)


def test_n8_full_reynolds_dense_amplitudes_and_all_output_deduplication():
    n = 8; bits = _bits(n); group = _group(n)
    rng = np.random.default_rng(2026101005)
    carrier = _project(rng.normal(size=256)+1j*rng.normal(size=256), bits, group)
    active = (4, 1, 6, 3); spectator = (0, 5, 2, 7)
    plan = compile_conditional_carrier_queries(bits, group, active_sites=active, spectator_sites=spectator)
    seen = []
    def callback(x):
        seen.extend(map(tuple, x)); return carrier[_integers(x)]
    response = evaluate_conditional_carrier_queries(plan, callback, batch_size=19)
    assert len(seen) == len(set(seen)) == 256
    assert plan.diagnostics['unique_carrier_queries'] == 256
    assert plan.diagnostics['duplicate_requests_removed'] > 0
    theta = rng.normal(scale=.13, size=60)
    matrix = np.asarray(conditional_filter_matrix(theta))
    expected = _project(_local_dense_action(carrier, matrix, active+spectator, n), bits, group)
    actual = np.asarray(conditional_carrier_amplitudes(theta, response.carrier_amplitudes, response.responses))
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=2e-13)
    np.testing.assert_allclose(response.projected_carrier_amplitudes, carrier, atol=2e-15)
    np.testing.assert_array_equal(conditional_carrier_amplitudes(jnp.zeros(60), response.carrier_amplitudes, response.responses), carrier)


def test_n12_nonconsecutive_support_keeps_spectator_external_bits_exact():
    n = 12; all_bits = _bits(n)
    # Complete order-two group, with nontrivial pullback outside the local8.
    group = (tuple(range(n)), tuple(reversed(range(n))))
    active = (0, 4, 8, 11); spectator = (2, 3, 6, 9)
    rng = np.random.default_rng(2026101006)
    carrier = _project(rng.normal(size=1 << n)+1j*rng.normal(size=1 << n), all_bits, group)
    selected = np.array([0, 117, 731, 1799, 2666, 4095]); bits = all_bits[selected]
    plan = compile_conditional_carrier_queries(bits, group, active_sites=active, spectator_sites=spectator)
    response = contract_conditional_carrier_queries(plan, carrier[_integers(plan.configurations)])
    theta = rng.normal(scale=.09, size=60)
    expected = _project(_local_dense_action(carrier, np.asarray(conditional_filter_matrix(theta)), active+spectator, n), all_bits, group)[selected]
    actual = conditional_carrier_amplitudes(theta, response.carrier_amplitudes, response.responses)
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=2e-13)


def test_jvp_reuses_response_bank_and_matches_independent_dense_finite_difference():
    n = 8; bits = _bits(n); group = _group(n)
    rng = np.random.default_rng(2026101007)
    carrier = _project(rng.normal(size=256), bits, group)
    outputs = np.array([3, 15, 37, 59, 130, 221])
    plan = compile_conditional_carrier_queries(bits[outputs], group)
    response = contract_conditional_carrier_queries(plan, carrier[_integers(plan.configurations)])
    theta = jnp.asarray(rng.normal(scale=.1, size=60)); direction = jnp.asarray(rng.normal(size=60))
    _, tangent = jax.jvp(lambda p:conditional_carrier_amplitudes(p, response.carrier_amplitudes, response.responses), (theta,), (direction,))
    eps = 1e-6
    def dense(p): return _project(np.asarray(conditional_filter_matrix(p))@carrier, bits, group)[outputs]
    finite = (dense(theta+eps*direction)-dense(theta-eps*direction))/(2*eps)
    assert np.linalg.norm(tangent) > 1e-8
    np.testing.assert_allclose(tangent, finite, atol=1e-8, rtol=3e-8)


def test_exact_carrier_nodes_are_not_replaced_by_floors():
    bits = _bits(8); group = (tuple(range(8)),)
    plan = compile_conditional_carrier_queries(bits, group)
    carrier = np.zeros(256, dtype=complex); carrier[85] = 1j
    response = contract_conditional_carrier_queries(plan, carrier[_integers(plan.configurations)])
    assert response.diagnostics['unique_exact_carrier_nodes'] == 255
    np.testing.assert_array_equal(response.carrier_amplitudes, carrier)
    theta = np.zeros(60); theta[2] = .2
    np.testing.assert_allclose(conditional_carrier_amplitudes(theta, carrier, response.responses),
        np.asarray(conditional_filter_matrix(theta))@carrier, atol=2e-15)


def test_query_compiler_has_no_machine_word_site_limit():
    bits = np.zeros((1, 72), dtype=np.int8); bits[0, 1::2] = 1
    plan = compile_conditional_carrier_queries(bits, (tuple(range(72)),))
    assert plan.configurations.shape == (36, 72)
    assert np.all(plan.configurations[:, 8:] == bits[:, 8:])


def test_invalid_group_support_and_callback_fail_closed():
    bits = _bits(8)[[15]]
    with pytest.raises(ValueError): compile_conditional_carrier_queries(bits, (tuple(range(8)),), spectator_sites=(0, 5, 6, 7))
    with pytest.raises(ValueError): compile_conditional_carrier_queries(bits, (tuple(range(8)), tuple(np.roll(np.arange(8), 1))))
    with pytest.raises(ValueError): compile_conditional_carrier_queries(bits.astype(float), (tuple(range(8)),))
    plan = compile_conditional_carrier_queries(bits, (tuple(range(8)),))
    with pytest.raises(ValueError): contract_conditional_carrier_queries(plan, np.ones(len(plan.configurations)+1))
    with pytest.raises(FloatingPointError): evaluate_conditional_carrier_queries(plan, lambda x:np.full(len(x), np.nan))
