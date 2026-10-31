"""Full-support IID proposal, independently enumerated at doubled m=0."""
import json
from itertools import combinations

import numpy as np
import pytest

from su2_nqs.vmc.full_support_proposal import (
    draw_full_support_sz0, full_support_sz0_constants, full_support_sz0_density,
)


def _oracle():
    rows = np.zeros((6, 4), np.int32)
    for row, ones in zip(rows, combinations(range(4), 2), strict=True):
        row[list(ones)] = 1
    index = {tuple(row): i for i, row in enumerate(rows)}
    # A delta raw distribution followed by a complete cyclic group average.
    orbit = np.array([sum(np.array_equal(row, np.roll(rows[0], g)) for g in range(4))/4
                      for row in rows])
    edges = [(0, 1), (1, 2), (1, 2), (2, 3), (3, 0)]
    weights = np.array([.1, .2, .15, .25, .3])
    push = np.zeros(6)
    for i, row in enumerate(rows):
        for (a, b), weight in zip(edges, weights, strict=True):
            swapped = row.copy(); swapped[a], swapped[b] = row[b], row[a]
            push[i] += weight*orbit[index[tuple(swapped)]]
    bridge = (orbit+push)/2

    def draw(count, seed):
        rng = np.random.default_rng(seed)
        labels = rng.choice(len(rows), size=count, p=bridge)
        return rows[labels], dict(latent=labels, Born_replay=np.zeros(count),
                                  rng_state=rng.bit_generator.state)
    return rows, orbit, push, bridge, draw


def test_exact_normalization_and_node_support_with_duplicate_null_bond_labels():
    rows, orbit, push, _, _ = _oracle()
    q = full_support_sz0_density(orbit, push, n_sites=4)
    np.testing.assert_allclose([orbit.sum(), push.sum(), q.sum()], 1., atol=2e-16, rtol=0)
    # A second singular proposal has no bridge support on five configurations.
    point = np.array([1., 0, 0, 0, 0, 0])
    covered = full_support_sz0_density(point, point, n_sites=4)
    assert np.all(covered > 0)
    np.testing.assert_array_equal(covered[1:], np.repeat(1/48, 5))
    assert full_support_sz0_constants(36)["minimum_density"] == (1/8)/9075135300
    assert np.all(rows.sum(1) == 2)


def test_empirical_IID_frequencies_match_fixed_density_not_observed_branch_share():
    basis, orbit, push, _, draw = _oracle()
    rows, report = draw_full_support_sz0(draw, 60000, n_sites=4, seed=71027)
    q = full_support_sz0_density(orbit, push, n_sites=4)
    empirical = np.array([np.mean(np.all(rows == row, axis=1)) for row in basis])
    assert np.max(abs(empirical-q)) < .006
    assert abs(report["uniform_records"]/len(rows)-1/8) < .006
    assert report["orbit_weight"] == report["Hpush_weight"] == 7/16
    assert report["empirical_branch_weights_used"] is False
    assert report["work"]["full_H_queries"] == report["work"]["target_amplitude_queries"] == 0
    assert report["base_records"]+report["uniform_records"] == len(rows)
    np.testing.assert_array_equal(report["callback_row_index"][report["bridge_row_indices"]],
                                  np.arange(report["base_records"]))
    assert np.all(report["callback_row_index"][report["uniform_row_indices"]] == -1)


def test_checkpointed_three_stream_RNG_replays_next_batch_and_latents():
    _, _, _, _, draw = _oracle()
    _, first = draw_full_support_sz0(draw, 137, n_sites=4, seed=271027)
    saved = json.loads(json.dumps(first["rng_after"]))
    rows, second = draw_full_support_sz0(draw, 233, n_sites=4, rng_state=saved)
    again, replay = draw_full_support_sz0(draw, 233, n_sites=4, rng_state=saved)
    np.testing.assert_array_equal(rows, again)
    for name in ["is_uniform", "bridge_row_indices", "uniform_row_indices", "callback_row_index"]:
        np.testing.assert_array_equal(second[name], replay[name])
    np.testing.assert_array_equal(second["base_draw"]["latent"], replay["base_draw"]["latent"])
    assert second["rng_after"] == replay["rng_after"]
    assert saved == first["rng_after"]
    assert second["rng_after"]["records"] == 370
    assert second["rng_after"]["batches"] == 2
    assert {tuple(x["spawn_key"]) for x in saved["stream_identities"].values()} == {(0,), (1,), (2,)}


def test_existing_full_orbit_H_kernel_plus_uniform_keeps_new_tangent_support():
    import jax
    import jax.numpy as jnp
    from su2_nqs.vmc.exchange_bridge_importance import make_orbit_exchange_bridge_products
    jax.config.update("jax_enable_x64", True)
    basis = np.zeros((6, 4), np.int32)
    for row, ones in zip(basis, combinations(range(4), 2), strict=True): row[list(ones)] = 1
    powers = jnp.array([8, 4, 2, 1])
    def factory(edges):
        edges = jnp.asarray(edges)
        def query(table, bits):
            changed = jnp.broadcast_to(bits, (len(edges), 4)); ids = jnp.arange(len(edges))
            changed = changed.at[ids, edges[:, 0]].set(bits[edges[:, 1]])
            changed = changed.at[ids, edges[:, 1]].set(bits[edges[:, 0]])
            return table[bits@powers], table[changed@powers]
        return query
    products, _ = make_orbit_exchange_bridge_products(factory, np.arange(4)[None, :],
        [(0, 1, 1.)], [1.], n_sites=4, projection_chunk=1)
    # The only bridge label acts as an equal-spin null swap on0011.
    raw = np.zeros(16); raw[3] = 1
    phi, hphi, orbit, push = map(np.asarray, products(jnp.asarray(raw), jnp.asarray(basis)))
    q = full_support_sz0_density(orbit, push, n_sites=4)
    assert np.sum((orbit+push) == 0) == 5 and np.all(q > 0)
    np.testing.assert_allclose(q.sum(), 1., atol=2e-16, rtol=0)
    np.testing.assert_array_equal(phi, raw[np.asarray(basis)@np.array([8, 4, 2, 1])])
    np.testing.assert_array_equal(hphi, .25*phi)
    # A native tangent can appear where both old-state guide components vanish.
    tangent = np.all(basis == [1, 1, 0, 0], axis=1).astype(float)
    assert float(np.sum(q*tangent**2/q)) == 1.
    assert float(np.sum(tangent[(orbit+push) > 0]**2)) == 0.


def test_uniform_branch_is_independent_of_callback_consumption():
    _, _, _, _, draw = _oracle()
    def consume_more(count, seed):
        bits, info = draw(count, seed)
        other = np.random.default_rng(seed); other.random(100000)
        return bits, dict(info, unrelated_rng=other.bit_generator.state)
    a, x = draw_full_support_sz0(draw, 200, n_sites=4, seed=71028)
    b, y = draw_full_support_sz0(consume_more, 200, n_sites=4, seed=71028)
    np.testing.assert_array_equal(x["is_uniform"], y["is_uniform"])
    np.testing.assert_array_equal(a[x["is_uniform"]], b[y["is_uniform"]])
    assert x["rng_after"] == y["rng_after"]


def test_all_uniform_draw_does_not_invoke_empty_callback():
    def forbidden(count, seed):
        raise AssertionError("zero-count callback was invoked")
    rows, report = draw_full_support_sz0(forbidden, 1, n_sites=4, seed=1, epsilon=.999999)
    assert rows.shape == (1, 4) and rows.sum() == 2
    assert report["uniform_records"] == 1 and report["base_draw"] == {}


def test_invalid_density_sector_callback_and_restore_fail_closed():
    _, _, _, _, draw = _oracle()
    for n in [True, 0, 3, 4.5]:
        with pytest.raises(ValueError): full_support_sz0_constants(n)
    for epsilon in [0, 1, np.nan, True, "0.125"]:
        with pytest.raises(ValueError): full_support_sz0_constants(4, epsilon=epsilon)
    for a, b in [([-.1], [1.1]), ([np.nan], [0]), ([1j], [0]), ([1, 0], [1])]:
        with pytest.raises(ValueError): full_support_sz0_density(a, b, n_sites=4)
    with pytest.raises(ValueError): draw_full_support_sz0(draw, 20, n_sites=4)
    with pytest.raises(ValueError): draw_full_support_sz0(draw, 0, n_sites=4, seed=1)
    _, report = draw_full_support_sz0(draw, 20, n_sites=4, seed=1)
    with pytest.raises(ValueError):
        draw_full_support_sz0(draw, 20, n_sites=6, rng_state=report["rng_after"])
    with pytest.raises(ValueError):
        draw_full_support_sz0(lambda n,s: (np.ones((n,4), int), {}), 20, n_sites=4, seed=1)
