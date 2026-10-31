"""State, full-H and AD equivalence of the optional local-feature cache."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.triangular_clebsch_feature_lut import (
    prepare_feature_lut_bank, make_feature_lut_amplitude, make_feature_lut_projected_amplitude,
)
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank, init_arity_pair
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, init_mixed_clebsch_family, mixed_clebsch_bank,
    make_mixed_clebsch_amplitude, make_mixed_projected_amplitude,
)
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.su2.triangle_blocks import triangle_block_tables
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action

jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='module')
def setup():
    cfg = MixedClebschConfig(multiplicity_cap=9, maximum_q=3)
    family = init_mixed_clebsch_family(jax.random.PRNGKey(17881), cfg)
    bank = mixed_clebsch_bank(family['ternary_c3'], cfg, 'ternary_c3')
    pair, _ = init_arity_pair(bank, cfg, 2)
    rng = np.random.default_rng(17882)
    rows = np.asarray([rng.permutation([0]*18+[1]*18) for _ in range(12)], np.int32)
    return cfg, pair, rows


def test_every_local_512_pattern_and_N36_raw_amplitude_Ward(setup):
    cfg, pair, rows = setup
    raw_old = make_mixed_clebsch_amplitude(cfg)
    raw_new = make_feature_lut_amplitude(cfg)
    for arm, params in pair.items():
        bank = arity_clebsch_bank(params, cfg, arm, 2)
        prepared = jax.jit(lambda b: prepare_feature_lut_bank(b, cfg))(bank)
        for node, cached in zip(bank['local'], prepared['local_features'], strict=True):
            for q, weights, table in zip(cfg.local_q, node, cached, strict=True):
                expected = np.asarray(triangle_block_tables()['nine'][q]) @ np.asarray(weights)
                np.testing.assert_allclose(table, expected, atol=2e-15, rtol=2e-13)
        old = jax.jit(jax.vmap(raw_old, in_axes=(None, 0)))(bank, rows)
        evaluate = jax.jit(jax.vmap(raw_new, in_axes=(None, 0)))
        np.testing.assert_allclose(evaluate(prepared, rows), old, atol=2e-16, rtol=2e-11)
        np.testing.assert_array_equal(evaluate(prepared, np.zeros((1, 36), np.int32)), 0.)
        bra = rows[0].copy(); bra[np.flatnonzero(bra == 0)[0]] = 1
        lowering = np.tile(bra, (19, 1)); lowering[np.arange(19), np.flatnonzero(bra)] = 0
        ward = np.asarray(evaluate(prepared, lowering))
        assert abs(ward.sum())/np.linalg.norm(ward) < 2e-11


def test_full_A1_H_and_independent_graph_connections(setup):
    cfg, pair, rows = setup
    bank = arity_clebsch_bank(pair['ternary_joint'], cfg, 'ternary_joint', 2)
    prepared = prepare_feature_lut_bank(bank, cfg)
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    old = make_mixed_projected_amplitude(cfg, images, chunk=4)
    new = make_feature_lut_projected_amplitude(cfg, images, chunk=4)
    np.testing.assert_allclose(jax.jit(jax.vmap(new, in_axes=(None, 0)))(prepared, rows[:2]),
        jax.jit(jax.vmap(old, in_axes=(None, 0)))(bank, rows[:2]), atol=2e-16, rtol=2e-11)
    np.testing.assert_allclose(new(prepared, jnp.asarray(rows[0, images[17]])), new(prepared, jnp.asarray(rows[0])),
        atol=2e-16, rtol=2e-11)
    bonds = tuple(triangular_j1j2_bonds(6, 6, j2=.125, periodic=True))
    a, h = make_amplitude_heisenberg_action(new, bonds, n_sites=36)(prepared, rows[:1])
    a0, h0 = make_amplitude_heisenberg_action(old, bonds, n_sites=36)(bank, rows[:1])
    np.testing.assert_allclose((a, h), (a0, h0), atol=2e-16, rtol=2e-11)
    connected, elements = SzHeisenbergGraph(36, bonds, n_up=18).get_conn(tuple(rows[0]))
    evaluate_old = jax.jit(lambda bits: jax.lax.map(lambda row: old(bank, row), bits, batch_size=4))
    oracle = np.dot(elements, np.asarray(evaluate_old(jnp.asarray(connected))))
    np.testing.assert_allclose(h[0], oracle, atol=2e-16, rtol=2e-11)


@pytest.mark.parametrize('arm', ['binary_truncated', 'ternary_joint'])
def test_parameter_JVP_and_full_gradient_include_LUT_preparation(setup, arm):
    cfg, pair, rows = setup
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    old = make_mixed_projected_amplitude(cfg, images, chunk=4)
    new = make_feature_lut_projected_amplitude(cfg, images, chunk=4)
    row = jnp.asarray(rows[0])
    before = lambda p: old(arity_clebsch_bank(p, cfg, arm, 2), row)
    after = lambda p: new(prepare_feature_lut_bank(arity_clebsch_bank(p, cfg, arm, 2), cfg), row)
    params = pair[arm]
    direction = jax.tree.map(lambda w: .001*jnp.cos(jnp.arange(w.size).reshape(w.shape)), params)
    d0 = jax.jit(lambda p, v: jax.jvp(before, (p,), (v,))[1])(params, direction)
    d1 = jax.jit(lambda p, v: jax.jvp(after, (p,), (v,))[1])(params, direction)
    np.testing.assert_allclose(d1, d0, atol=2e-16, rtol=2e-11)
    g0, g1 = jax.jit(jax.grad(before))(params), jax.jit(jax.grad(after))(params)
    for original, cached in zip(jax.tree.leaves(g0), jax.tree.leaves(g1), strict=True):
        np.testing.assert_allclose(cached, original, atol=2e-16, rtol=2e-11)


def test_invalid_prepared_shapes_and_projector_are_rejected(setup):
    cfg, pair, _ = setup
    bank = arity_clebsch_bank(pair['ternary_joint'], cfg, 'ternary_joint', 2)
    with pytest.raises(ValueError, match='hierarchy'):
        prepare_feature_lut_bank(dict(bank, local=bank['local'][:-1]), cfg)
    with pytest.raises(ValueError, match='permutations'):
        make_feature_lut_projected_amplitude(cfg, np.zeros((4, 36), int))


def test_preparation_is_outside_query_trace_not_hidden_per_configuration(setup):
    cfg, pair, rows = setup
    bank = arity_clebsch_bank(pair['ternary_joint'], cfg, 'ternary_joint', 2)
    preparation = jax.make_jaxpr(lambda b: prepare_feature_lut_bank(b, cfg))(bank)
    # One512-row precontraction per block and q, before the query is traced.
    dots = [e for e in preparation.jaxpr.eqns if e.primitive.name == 'dot_general']
    assert len(dots) == len(bank['local'])*len(cfg.local_q)
    assert all(e.invars[0].aval.shape[0] == 512 for e in dots)
    prepared = prepare_feature_lut_bank(bank, cfg)
    new = jax.make_jaxpr(make_feature_lut_amplitude(cfg))(prepared, rows[0])
    old = jax.make_jaxpr(make_mixed_clebsch_amplitude(cfg))(bank, rows[0])
    # The cached query cannot re-do U@W: no U[512,multiplicity] constant
    # remains in its graph, only dynamic feature-table inputs and gathers.
    assert not any(getattr(v, 'ndim', 0) == 2 and v.shape[0] == 512 for v in new.consts)
    assert any(getattr(v, 'ndim', 0) == 2 and v.shape[0] == 512 for v in old.consts)
    assert any(e.primitive.name in ('gather', 'dynamic_slice') for e in new.jaxpr.eqns)
