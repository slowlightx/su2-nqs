"""Independent tensor-axis tests of overlapping conditional carrier filters."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.overlapping_carrier_queries import (
    local_overlap_connections, overlap_filter_rows, compile_overlapping_carrier_queries,
    evaluate_overlapping_carrier_queries, overlapping_carrier_amplitudes,
)
from su2_nqs.models.conditional_carrier_resonance import conditional_filter_matrix
from su2_nqs.models.shared_resonance_attention import (
    ResonanceAttentionConfig, init_shared_resonance_attention,
    make_resonance_attention_reference, shared_resonance_generators,
)


def _bits(n): return ((np.arange(1 << n)[:, None] >> np.arange(n-1, -1, -1)) & 1).astype(np.int8)
def _indices(bits): return bits@(1 << np.arange(bits.shape[-1]-1, -1, -1))


def _apply(vector, matrix, sites):
    n = int(np.log2(len(vector))); order = tuple(sites)+tuple(i for i in range(n) if i not in sites)
    front = np.transpose(vector.reshape((2,)*n), order).reshape(256, -1)
    return np.transpose((matrix@front).reshape((2,)*n), np.argsort(order)).reshape(-1)


def _project(vector, bits, group):
    return np.mean(np.stack([vector[_indices(bits[:, image])] for image in group]), axis=0)


def test_local_rows_contract_intermediate_B_coherently_and_identity_exact():
    rng = np.random.default_rng(2026101201)
    first, second = rng.normal(scale=.15, size=(2, 60))
    carrier = rng.normal(size=4096)+1j*rng.normal(size=4096)
    selected = np.array([0, 4095, 1365, 2218, 2389, 399, 995, 3087], dtype=np.int32)
    k1, k2 = map(np.asarray, (conditional_filter_matrix(first), conditional_filter_matrix(second)))
    expected = _apply(_apply(carrier, k1, (4, 5, 6, 7, 0, 1, 2, 3)), k2, (8, 9, 10, 11, 4, 5, 6, 7))
    no_cascade = _apply(carrier, k1, (4, 5, 6, 7, 0, 1, 2, 3))+_apply(carrier, k2, (8, 9, 10, 11, 4, 5, 6, 7))-carrier
    rows = np.asarray(overlap_filter_rows(first, second, selected))
    null_rows = np.asarray(overlap_filter_rows(first, second, selected, mode='no_cascade'))
    for i, row in enumerate(selected):
        columns, valid, identity = local_overlap_connections(int(row))
        assert valid.sum() <= 216
        np.testing.assert_allclose(rows[i]@carrier[columns], expected[row], atol=2e-14)
        np.testing.assert_allclose(null_rows[i]@carrier[columns], no_cascade[row], atol=2e-14)
        zero = np.asarray(overlap_filter_rows(np.zeros(60), np.zeros(60), np.asarray([row])))[0]
        exact = np.zeros(216); exact[identity] = 1
        np.testing.assert_array_equal(zero, exact)
    assert np.linalg.norm(expected-no_cascade) > 1e-3


def test_complete_group_projection_once_after_product_and_query_dedup():
    rng = np.random.default_rng(2026101202); bits = _bits(12)
    group = tuple(tuple((np.arange(12)+shift) % 12) for shift in range(12))
    carrier = _project(rng.normal(size=4096)+1j*rng.normal(size=4096), bits, group)
    selected = np.array([27, 4095, 2730, 1365, 995, 2751])
    cells = ((0, 2, 4, 6), (1, 3, 5, 7), (8, 9, 10, 11))
    plan = compile_overlapping_carrier_queries(bits[selected], group, cells=cells)
    seen = []
    def callback(x): seen.extend(map(tuple, x)); return carrier[_indices(x)]
    values = evaluate_overlapping_carrier_queries(plan, callback, batch_size=37)
    assert len(seen) == len(set(seen)) == len(plan.configurations)
    assert plan.diagnostics['duplicate_requests_removed'] > 0
    first, second = rng.normal(scale=.2, size=(2, 60))
    k1, k2 = map(np.asarray, (conditional_filter_matrix(first), conditional_filter_matrix(second)))
    first_state = _apply(carrier, k1, cells[1]+cells[0])
    final = _project(_apply(first_state, k2, cells[2]+cells[1]), bits, group)
    bad = _project(_apply(_project(first_state, bits, group), k2, cells[2]+cells[1]), bits, group)
    observed = overlapping_carrier_amplitudes(first, second, plan, values)
    np.testing.assert_allclose(observed, final[selected], atol=2e-14)
    assert np.linalg.norm((final-bad)[selected]) > 1e-5
    np.testing.assert_allclose(overlapping_carrier_amplitudes(np.zeros(60), np.zeros(60), plan, values), carrier[selected], atol=2e-15)


def test_both_filter_gradients_are_live_at_nonzero_parameters_and_match_dense_FD():
    rng = np.random.default_rng(2026101203); bits = _bits(12)
    group = (tuple(range(12)), tuple(reversed(range(12))))
    carrier = _project(rng.normal(size=4096), bits, group)
    selected = np.array([85, 173, 1349, 1965, 2730, 2971])
    plan = compile_overlapping_carrier_queries(bits[selected], group)
    values = carrier[_indices(plan.configurations)]
    theta = jnp.asarray(rng.normal(scale=.13, size=(2, 60)))
    weights = rng.normal(size=len(selected))
    def loss(p): return jnp.real(jnp.vdot(jnp.asarray(weights), overlapping_carrier_amplitudes(p[0], p[1], plan, values)))
    gradient = np.asarray(jax.jit(jax.grad(loss))(theta))
    assert np.isfinite(gradient).all() and min(np.linalg.norm(row) for row in gradient) > 1e-6
    def dense_loss(p):
        first, second = map(np.asarray, (conditional_filter_matrix(p[0]), conditional_filter_matrix(p[1])))
        dressed = _apply(_apply(carrier, first, tuple(range(4, 8))+tuple(range(4))), second, tuple(range(8, 12))+tuple(range(4, 8)))
        return np.vdot(weights, _project(dressed, bits, group)[selected]).real
    for layer in (0, 1):
        direction = np.zeros((2, 60)); direction[layer] = rng.normal(size=60)
        eps = 1e-6
        finite = (dense_loss(theta+eps*direction)-dense_loss(theta-eps*direction))/(2*eps)
        np.testing.assert_allclose(np.sum(gradient*direction), finite, rtol=2e-8, atol=2e-8)


def test_exact_nodes_and_query_caps_are_not_silently_repaired():
    bits = _bits(12); group = (tuple(range(12)),)
    selected = bits[[1365, 2730]]
    plan = compile_overlapping_carrier_queries(selected, group)
    carrier = np.zeros(4096); carrier[1365] = 1
    values = carrier[_indices(plan.configurations)]
    zero = overlapping_carrier_amplitudes(np.zeros(60), np.zeros(60), plan, values)
    np.testing.assert_array_equal(zero, carrier[[1365, 2730]])
    for keyword in ('max_unique_queries', 'max_terminal_requests', 'max_plan_bytes'):
        with pytest.raises(MemoryError): compile_overlapping_carrier_queries(selected, group, **{keyword:1})
    with pytest.raises(ValueError): compile_overlapping_carrier_queries(selected, group, cells=((0,1,2,3),(3,4,5,6),(8,9,10,11)))
    with pytest.raises(ValueError): overlap_filter_rows(np.zeros(60), np.zeros(60), np.array([0]), mode='project_between')


def test_arbitrary_site_count_without_uint64_encoding_or_external_changes():
    state = np.zeros((1, 72), dtype=np.int8); state[:, 1::2] = 1
    plan = compile_overlapping_carrier_queries(state, (tuple(range(72)),))
    assert plan.configurations.shape == (216, 72)
    assert np.all(plan.configurations[:, 12:] == state[:, 12:])


def test_row_permutations_and_n16_complex_carrier_with_untouched_sites():
    rng = np.random.default_rng(2026101204); bits = _bits(16)
    group = (tuple(range(16)), tuple(reversed(range(16))))
    carrier = _project(rng.normal(size=1 << 16)+1j*rng.normal(size=1 << 16), bits, group)
    cells = ((0, 5, 10, 15), (1, 4, 11, 14), (3, 6, 9, 12))
    first, second = rng.normal(scale=.13, size=(2, 60))
    rows = np.array([1365, 2751, 991, 2751, 0, 4095], dtype=np.int32)
    order = np.array([4, 2, 0, 5, 1, 3])
    np.testing.assert_array_equal(overlap_filter_rows(first, second, rows[order]),
                                  np.asarray(overlap_filter_rows(first, second, rows))[order])
    selected = np.array([173, 1775, 19741, 32771, 43690, 65531])
    plan = compile_overlapping_carrier_queries(bits[selected], group, cells=cells)
    values = carrier[_indices(plan.configurations)]
    expected = _project(_apply(_apply(carrier, np.asarray(conditional_filter_matrix(first)), cells[1]+cells[0]),
        np.asarray(conditional_filter_matrix(second)), cells[2]+cells[1]), bits, group)
    actual = overlapping_carrier_amplitudes(first, second, plan, values)
    np.testing.assert_allclose(actual, expected[selected], atol=2e-14, rtol=2e-13)
    # For identity-group query compilation the four external bits cannot change.
    one = compile_overlapping_carrier_queries(bits[selected[:1]], (tuple(range(16)),), cells=cells)
    external = sorted(set(range(16))-set(sum(cells, ())))
    assert np.all(one.configurations[:, external] == bits[selected[:1], external])


def test_actual_shared_attention_full_H_JVP_at_nonzero_generators_matches_dense():
    rng = np.random.default_rng(2026101205); bits = _bits(12)
    group = (tuple(range(12)), tuple(reversed(range(12))))
    carrier = _project(rng.normal(size=4096)+1j*rng.normal(size=4096), bits, group)
    selected = np.array([1365, 2730, 1366])
    bonds = ((0, 1, 1.), (3, 7, .125), (8, 11, -.2))
    requested = set(map(int, selected))
    swaps = []
    for a, b, coupling in bonds:
        exchanged = bits.copy(); exchanged[:, a], exchanged[:, b] = bits[:, b], bits[:, a]
        indices = _indices(exchanged); swaps.append((a, b, coupling, indices))
        requested.update(map(int, indices[selected]))
    centers = np.asarray(sorted(requested)); position = {int(value):i for i, value in enumerate(centers)}
    stencil = np.zeros((len(selected), len(centers)))
    for row, configuration in enumerate(selected):
        for a, b, coupling, indices in swaps:
            opposite = bits[configuration, a] != bits[configuration, b]
            stencil[row, position[int(configuration)]] += coupling*(-.25 if opposite else .25)
            if opposite: stencil[row, position[int(indices[configuration])]] += .5*coupling
    plan = compile_overlapping_carrier_queries(bits[centers], group)
    values = carrier[_indices(plan.configurations)]
    config = ResonanceAttentionConfig(width=4, hidden_width=6)
    initial = init_shared_resonance_attention(jax.random.PRNGKey(2026101205), config)
    offsets = jnp.asarray([[1., 0.], [0., 1.]])
    reference = make_resonance_attention_reference(initial, offsets, config)
    params = jax.tree.map(lambda value:value+jnp.asarray(rng.normal(scale=.025, size=value.shape)), initial)
    direction = jax.tree.map(lambda value:jnp.asarray(rng.normal(size=value.shape)), initial)
    assert np.linalg.norm(shared_resonance_generators(params, reference, offsets, config)) > 1e-5
    def action(p):
        generators = shared_resonance_generators(p, reference, offsets, config)
        amplitudes = overlapping_carrier_amplitudes(generators[0], generators[1], plan, values)
        return amplitudes[jnp.asarray([position[int(i)] for i in selected])], jnp.asarray(stencil)@amplitudes
    (amplitude, haction), (jvp, hjvp) = jax.jvp(jax.jit(action), (params,), (direction,))
    def dense(p):
        g = shared_resonance_generators(p, reference, offsets, config)
        dressed = _apply(_apply(carrier, np.asarray(conditional_filter_matrix(g[0])), tuple(range(4, 8))+tuple(range(4))),
            np.asarray(conditional_filter_matrix(g[1])), tuple(range(8, 12))+tuple(range(4, 8)))
        projected = _project(dressed, bits, group)
        full_h = np.zeros(4096, dtype=complex)
        for a, b, coupling, indices in swaps:
            opposite = bits[:, a] != bits[:, b]
            full_h += coupling*(np.where(opposite, -.25, .25)*projected+.5*opposite*projected[indices])
        return projected[selected], full_h[selected]
    expected, hexpected = dense(params)
    np.testing.assert_allclose(amplitude, expected, atol=3e-14)
    np.testing.assert_allclose(haction, hexpected, atol=3e-14)
    eps = 2e-6
    plus = dense(jax.tree.map(lambda p, d:p+eps*d, params, direction))
    minus = dense(jax.tree.map(lambda p, d:p-eps*d, params, direction))
    np.testing.assert_allclose(jvp, (plus[0]-minus[0])/(2*eps), atol=2e-7, rtol=3e-7)
    np.testing.assert_allclose(hjvp, (plus[1]-minus[1])/(2*eps), atol=2e-7, rtol=3e-7)
    assert np.linalg.norm(hjvp) > 1e-5
