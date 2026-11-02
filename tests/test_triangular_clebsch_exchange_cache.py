"""Independent unchanged-state checks for ancestor-union exchange reuse."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank, init_arity_pair
from su2_nqs.models.triangular_clebsch_exchange_cache import (
    exchange_query_plan, make_feature_lut_exchange_queries,
)
from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, init_mixed_clebsch_family, mixed_clebsch_bank,
    make_mixed_clebsch_amplitude, make_mixed_projected_amplitude,
)
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.triangular_clebsch_cached_action import (
    make_cached_projected_heisenberg_action, make_cached_projected_heisenberg_bridge,
)

jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='module')
def system():
    config = MixedClebschConfig(multiplicity_cap=9, maximum_q=3)
    initial = init_mixed_clebsch_family(jax.random.PRNGKey(178931), config)
    bank = mixed_clebsch_bank(initial['ternary_c3'], config, 'ternary_c3')
    pair, _ = init_arity_pair(bank, config, 2)
    rng = np.random.default_rng(178932)
    rows = np.asarray([rng.permutation([0]*18+[1]*18) for _ in range(3)], np.int32)
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    bonds = tuple(triangular_j1j2_bonds(6, 6, j2=.125, periodic=True))
    return config, pair, rows, images, bonds


def assert_relative(actual, expected, tolerance=2e-11):
    a, b = np.asarray(actual), np.asarray(expected)
    scale = max(np.max(abs(a), initial=0.), np.max(abs(b), initial=0.), 1e-300)
    error = np.linalg.norm((a-b)/scale)/max(np.linalg.norm(b/scale), 1e-280)
    assert error < tolerance, error


@pytest.mark.parametrize('arm', ['binary_truncated', 'ternary_joint'])
def test_raw_every_bond_and_null_swap_repeated_pair_and_exact_node(system, arm):
    cfg, pair, rows, _, bonds = system
    edges = [(a, b) for a, b, _ in bonds]+[(0, 1), (0, 1)]
    bank = arity_clebsch_bank(pair[arm], cfg, arm, 2)
    prepared = prepare_feature_lut_bank(bank, cfg)
    query = jax.jit(make_feature_lut_exchange_queries(cfg, edges))
    raw = jax.jit(jax.vmap(make_mixed_clebsch_amplitude(cfg), in_axes=(None, 0)))
    for row in rows:
        actual, swaps = query(prepared, row)
        candidates = np.tile(row, (len(edges), 1))
        for index, (a, b) in enumerate(edges):
            candidates[index, a], candidates[index, b] = row[b], row[a]
        expected = raw(bank, np.vstack((row, candidates)))
        assert_relative(np.r_[actual, swaps], expected)
        np.testing.assert_array_equal(swaps[-2], swaps[-1])
    psi, swapped = query(prepared, np.zeros(36, np.int32))
    assert psi == 0.
    np.testing.assert_array_equal(swapped, 0.)
    # Invalid physical bits remain an exact zero, not silently repaired inputs.
    invalid = rows[0].copy(); invalid[0] = 2
    psi, swapped = query(prepared, invalid)
    assert psi == 0.
    np.testing.assert_array_equal(swapped, 0.)


def test_full_A1_full_H_and_independent_Sz_oracle(system):
    cfg, pair, rows, images, bonds = system
    bank = arity_clebsch_bank(pair['ternary_joint'], cfg, 'ternary_joint', 2)
    prepared = prepare_feature_lut_bank(bank, cfg)
    projected = make_mixed_projected_amplitude(cfg, images, chunk=4)
    legacy = make_amplitude_heisenberg_action(projected, bonds, n_sites=36)
    cached = make_cached_projected_heisenberg_action(cfg, images, bonds)
    actual = cached(prepared, rows[:1])
    expected = legacy(bank, rows[:1])
    assert_relative(actual, expected)
    connected, elements = SzHeisenbergGraph(36, bonds, n_up=18).get_conn(tuple(rows[0]))
    evaluate = jax.jit(lambda bits: jax.lax.map(lambda x: projected(bank, x), bits, batch_size=4))
    oracle = np.dot(elements, np.asarray(evaluate(jnp.asarray(connected))))
    assert_relative(actual[1], [oracle])
    rotated = cached(prepared, rows[:1, images[23]])
    assert_relative(rotated, actual)


def test_raw_node_can_enter_support_after_cross_cluster_exchange(system):
    cfg, pair, _, _, bonds = system
    bank = arity_clebsch_bank(pair['ternary_joint'], cfg, 'ternary_joint', 2)
    blocks = triangular_schur_hierarchy(6).blocks
    row = np.zeros(36, np.int32)
    for block, up in zip(blocks, (7, 3, 4, 4), strict=True):
        row[list(block.sites[:up])] = 1
    # First block has doubled m=5>qmax=3, but one exchange can lower it to3.
    assert row.sum() == 18
    query = jax.jit(make_feature_lut_exchange_queries(cfg, [(a, b) for a, b, _ in bonds]))
    raw, others = query(prepare_feature_lut_bank(bank, cfg), row)
    assert raw == 0.
    assert np.max(abs(np.asarray(others))) > 1e-20
    images = np.arange(36, dtype=np.int32)[None]
    legacy_raw = make_mixed_clebsch_amplitude(cfg)
    expected = make_amplitude_heisenberg_action(legacy_raw, bonds, n_sites=36)(bank, row[None])
    actual = make_cached_projected_heisenberg_action(cfg, images, bonds, projection_chunk=1)(
        prepare_feature_lut_bank(bank, cfg), row[None])
    assert expected[0][0] == actual[0][0] == 0.
    assert abs(expected[1][0]) > 1e-20
    assert_relative(actual, expected)


def test_full_H_parameter_JVP_and_gradient_include_rebuilt_cache(system):
    cfg, pair, rows, images, bonds = system
    params = pair['ternary_joint']
    projected = make_mixed_projected_amplitude(cfg, images, chunk=4)
    legacy = make_amplitude_heisenberg_action(projected, bonds, n_sites=36)
    cached = make_cached_projected_heisenberg_action(cfg, images, bonds)
    before = lambda p: legacy(arity_clebsch_bank(p, cfg, 'ternary_joint', 2), rows[:1])[1].sum()
    after = lambda p: cached(prepare_feature_lut_bank(
        arity_clebsch_bank(p, cfg, 'ternary_joint', 2), cfg), rows[:1])[1].sum()
    direction = jax.tree.map(lambda x: .001*jnp.cos(jnp.arange(x.size).reshape(x.shape)), params)
    j0 = jax.jit(lambda p, v: jax.jvp(before, (p,), (v,))[1])(params, direction)
    j1 = jax.jit(lambda p, v: jax.jvp(after, (p,), (v,))[1])(params, direction)
    assert_relative([j1], [j0])
    g0, g1 = jax.jit(jax.grad(before))(params), jax.jit(jax.grad(after))(params)
    assert_relative(np.concatenate([np.ravel(x) for x in jax.tree.leaves(g1)]),
                    np.concatenate([np.ravel(x) for x in jax.tree.leaves(g0)]))


def test_weighted_H_symmetry_including_negative_duplicate_and_zero_edges(system):
    cfg, pair, rows, _, _ = system
    # Cancellation is harmless even if that artificial zero edge is not mapped
    # onto itself. Equal total weights, not edge ordering, define H invariance.
    images = np.vstack((np.arange(36), np.roll(np.arange(36), 18)))
    bonds = [(0, 1, -.3), (0, 1, .1), (18, 19, -.2),
             (2, 3, 1.), (2, 3, -1.), (5, 6, 0.)]
    # Use exactly representable equal negative weights (avoid a false mismatch
    # of -0.3+0.1 and -0.2 when testing the underlying formula).
    bonds[:3] = [(0, 1, -.5), (0, 1, .25), (18, 19, -.25)]
    bank = arity_clebsch_bank(pair['ternary_joint'], cfg, 'ternary_joint', 2)
    projected = make_mixed_projected_amplitude(cfg, images, chunk=2)
    before = make_amplitude_heisenberg_action(projected, bonds, n_sites=36)(bank, rows[:1])
    after = make_cached_projected_heisenberg_action(cfg, images, bonds, projection_chunk=2)(
        prepare_feature_lut_bank(bank, cfg), rows[:1])
    assert_relative(after, before)
    with pytest.raises(ValueError, match='weighted H'):
        make_cached_projected_heisenberg_action(cfg, images, [(0, 1, 1.)], projection_chunk=2)


def test_plan_recomputes_only_ancestor_union_and_rejects_bad_inputs(system):
    cfg, _, _, _, bonds = system
    plan = exchange_query_plan(cfg, [(a, b) for a, b, _ in bonds])
    assert sum(len(group['indices']) for group in plan) == len(bonds)
    assert any(len(group['ancestors']) < 3 for group in plan)
    assert any(len(group['leaves']) == 1 for group in plan)
    for invalid in ([], [(0, 0)], [(0, 36)], [(0., 1.)]):
        with pytest.raises(ValueError, match='integer site pairs'):
            exchange_query_plan(cfg, invalid)


def test_cached_bridge_matches_direct_coherent_physical_exchange_and_legacy_H(system):
    cfg, pair, rows, _, bonds = system
    images = np.vstack((np.arange(36), np.roll(np.arange(36), 18))).astype(np.int32)
    bank = arity_clebsch_bank(pair['ternary_joint'], cfg, 'ternary_joint', 2)
    prepared = prepare_feature_lut_bank(bank, cfg)
    probabilities = np.abs([value for _, _, value in bonds])
    probabilities /= probabilities.sum()
    bridge, metadata = make_cached_projected_heisenberg_bridge(cfg, images, bonds,
        probabilities, projection_chunk=2)
    psi, hp, push = bridge(prepared, rows[:1])
    legacy = make_cached_projected_heisenberg_action(cfg, images, bonds, projection_chunk=2)
    old_psi, old_hp = legacy(prepared, rows[:1])
    assert_relative(psi, old_psi)
    assert_relative(hp, old_hp)
    projected = make_mixed_projected_amplitude(cfg, images, chunk=2)
    changed = np.tile(rows[0], (len(bonds), 1))
    for k, (a, b, _) in enumerate(bonds):
        changed[k, a], changed[k, b] = rows[0, b], rows[0, a]
    direct = jax.jit(lambda x: jax.lax.map(lambda row: projected(bank, row), x, batch_size=4))(changed)
    expected = np.dot(probabilities, abs(np.asarray(direct)) ** 2)
    assert_relative(push, [expected])
    assert metadata['raw_query_unique_edges'] == metadata['physical_unique_edges'] == len(bonds)
    # No coherent-state support is invented for the forbidden all-down sector.
    node = bridge(prepared, np.zeros((1, 36), np.int32))
    for value in node:
        np.testing.assert_array_equal(value, 0.)
