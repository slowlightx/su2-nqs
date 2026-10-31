"""Exact scalar-commutant feature compression, including dynamic gradients."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.multicontext_resonance_features import (
    commutant_components, multicontext_feature_coefficients, features_from_rows,
    features_from_plan, amplitudes_from_features,
)
from su2_nqs.benchmarks.multicontext_resonance_queries import (
    multicontext_overlap_filter_rows, multicontext_carrier_amplitudes,
)
from su2_nqs.benchmarks.overlapping_carrier_queries import (
    compile_overlapping_carrier_queries, local_overlap_connections,
)
from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries, gather_physical_h_action


def _bits(n): return ((np.arange(1 << n)[:, None] >> np.arange(n-1, -1, -1)) & 1).astype(np.int8)
def _indices(bits): return bits @ (1 << np.arange(bits.shape[-1]-1, -1, -1))


def test_matrix_units_keep_magnetic_dimension_and_exact_scalar_algebra():
    parts = commutant_components(); labels = parts.unit_labels; units = parts.matrix_units
    assert len(labels) == 14 and parts.identity_coefficients.shape == (1176,)
    np.testing.assert_allclose(np.einsum('rij,sij->rs', units, units),
        np.diag([q+1 for q, _, _ in labels]), atol=9e-16)
    raising = np.zeros((16, 16))
    for old in range(16):
        for site in range(4):
            bit = 1 << site
            if not old & bit: raising[old | bit, old] += 1
    for index, (q, a, b) in enumerate(labels):
        reverse = labels.index((q, b, a))
        np.testing.assert_allclose(units[index].T, units[reverse], atol=1e-15)
        np.testing.assert_allclose(units[index]@raising, raising@units[index], atol=1e-15)
    np.testing.assert_allclose(sum(units[i] for i, (_, a, b) in enumerate(labels) if a == b),
                               np.eye(16), atol=7e-16)


def test_rows_coefficients_reproduce_dynamic_two_filter_action_and_zero_identity():
    rng = np.random.default_rng(2026101501)
    rows = np.array([0, 4095, 1, 16, 17, 511, 1365, 1787, 2751, 2973], dtype=np.int32)
    full_carrier = rng.normal(size=4096)+1j*rng.normal(size=4096)
    terminals = np.stack([full_carrier[local_overlap_connections(int(row))[0]] for row in rows])
    # The sentinel is deliberately huge: padded entries must never contribute.
    for i, row in enumerate(rows): terminals[i, ~local_overlap_connections(int(row))[1]] = 1e9+3e8j
    features = jax.jit(features_from_rows)(rows, terminals)
    first, second = rng.normal(scale=.07, size=(2, 6, 6, 10))
    for mode in ('cascade', 'no_cascade'):
        weights = np.asarray(multicontext_overlap_filter_rows(first, second, rows, mode=mode))
        reference = np.sum(weights*terminals, axis=1)
        values = amplitudes_from_features(first, second, features, full_carrier[rows], mode=mode)
        np.testing.assert_allclose(values, reference, atol=8e-15, rtol=2e-13)
        raw = np.asarray(features)@np.asarray(multicontext_feature_coefficients(first, second, centered=False, mode=mode))
        np.testing.assert_allclose(raw, reference, atol=8e-15, rtol=2e-13)
    zero = np.zeros((6, 6, 10))
    np.testing.assert_array_equal(multicontext_feature_coefficients(zero, zero), np.zeros(1176))
    np.testing.assert_array_equal(amplitudes_from_features(zero, zero, features, full_carrier[rows]), full_carrier[rows])


def test_final_full_projection_feature_cache_and_both_gate_JVPs():
    rng = np.random.default_rng(2026101502); bits = _bits(12)
    group = tuple(tuple((np.arange(12)+shift) % 12) for shift in range(12))
    selected = np.array([17, 73, 1365, 1873, 2730, 4095])
    # Deliberately not A1: identity must return projected C, not unprojected C.
    carrier = rng.normal(size=4096)+1j*rng.normal(size=4096)
    plan = compile_overlapping_carrier_queries(bits[selected], group)
    values = carrier[_indices(plan.configurations)]
    cached = features_from_plan(plan, values, image_batch_size=17)
    assert cached.diagnostics['valid_images'] == len(selected)*len(group)
    assert cached.diagnostics['new_carrier_calls'] == 0
    assert cached.diagnostics['retained_feature_bytes'] < 150_000
    assert cached.diagnostics['identity_absolute_error'] < 1e-14
    projected = np.mean(carrier[_indices(bits[selected][:, np.asarray(group)])], axis=1)
    np.testing.assert_allclose(cached.projected_carrier, projected, atol=2e-16)
    parameters = jnp.asarray(rng.normal(scale=.045, size=(2, 6, 6, 10)))
    direct = lambda p:multicontext_carrier_amplitudes(p[0], p[1], plan, values)
    compressed = lambda p:amplitudes_from_features(p[0], p[1], cached.features, cached.projected_carrier)
    np.testing.assert_allclose(compressed(parameters), direct(parameters), atol=3e-15)
    for layer in (0, 1):
        direction = np.zeros_like(parameters); direction[layer] = rng.normal(size=(6, 6, 10))
        dt = jax.jvp(compressed, (parameters,), (jnp.asarray(direction),))[1]
        dt_ref = jax.jvp(direct, (parameters,), (jnp.asarray(direction),))[1]
        np.testing.assert_allclose(dt, dt_ref, atol=5e-14, rtol=2e-12)
        eps = 1e-6
        finite = (np.asarray(direct(parameters+eps*direction))-np.asarray(direct(parameters-eps*direction)))/(2*eps)
        np.testing.assert_allclose(dt, finite, atol=3e-10, rtol=2e-7)


def test_full_H_features_and_two_sided_nonuniform_Q_energy_gradient():
    rng = np.random.default_rng(2026101503); bits = _bits(12)
    centers = bits[[1365, 1873, 2730, 995]]
    bonds = ((0, 1, 1.), (3, 7, .125), (8, 11, -.2))
    hplan = compile_physical_h_queries(centers, bonds)
    group = (tuple(range(12)), tuple(reversed(range(12))))
    plan = compile_overlapping_carrier_queries(hplan.configurations, group)
    carrier = rng.normal(size=4096)+1j*rng.normal(size=4096)
    values = carrier[_indices(plan.configurations)]
    cached = features_from_plan(plan, values, image_batch_size=11)
    centers_features = cached.features[hplan.center_indices]
    hfeatures = np.sum(np.where(hplan.valid[..., None],
        cached.features[hplan.connected_indices]*hplan.elements[..., None], 0.), axis=1)
    c, hc = gather_physical_h_action(hplan, jnp.asarray(cached.projected_carrier))
    q = jnp.asarray([.7, 1.3, 2.9, .21])
    parameters = jnp.asarray(rng.normal(scale=.07, size=(2, 6, 6, 10)))
    def loss_compressed(p):
        psi = amplitudes_from_features(p[0], p[1], centers_features, c)
        hp = amplitudes_from_features(p[0], p[1], hfeatures, hc)
        return jnp.real(jnp.sum(psi.conj()*hp/q))/jnp.sum(abs(psi)**2/q)
    def loss_direct(p):
        psi, hp = gather_physical_h_action(hplan, multicontext_carrier_amplitudes(p[0], p[1], plan, values))
        return jnp.real(jnp.sum(psi.conj()*hp/q))/jnp.sum(abs(psi)**2/q)
    val, gradient = jax.jit(jax.value_and_grad(loss_compressed))(parameters)
    ref, reference_gradient = jax.jit(jax.value_and_grad(loss_direct))(parameters)
    np.testing.assert_allclose(val, ref, atol=2e-14)
    np.testing.assert_allclose(gradient, reference_gradient, atol=3e-14, rtol=3e-11)
    assert np.isfinite(gradient).all() and min(np.linalg.norm(g) for g in gradient) > 1e-4


def test_invalid_feature_shapes_fail_closed():
    with pytest.raises(ValueError): features_from_rows(np.array([4096]), np.ones((1, 216)))
    with pytest.raises(ValueError): features_from_rows(np.array([0]), np.ones((1, 215)))
    with pytest.raises(ValueError): amplitudes_from_features(np.zeros((6, 6, 10)), np.zeros((6, 6, 10)),
                                                           np.zeros((2, 1175)), np.zeros(2))
    with pytest.raises(ValueError): multicontext_feature_coefficients(np.zeros((6, 6, 10)), np.zeros((6, 6, 10)), mode='middle_projection')
