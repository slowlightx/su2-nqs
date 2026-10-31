"""Independent physical-amplitude checks for whole-q two-node retractions."""
from dataclasses import replace
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_decoder_cost import physical_config
from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.models.finite_schur_decoder import (
    FiniteSchurDecoderConfig, free_reduced_bank, make_finite_schur_amplitude,
    reduced_entry_metadata)
from su2_nqs.models.schur_two_site import (
    embed_reduced_bank, internal_edges, two_site_layout, join_reduced_pair,
    split_reduced_pair, make_two_site_features)
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.projected_spin_linear import make_projected_linear_products

jax.config.update('jax_enable_x64', True)


def config(rows=4, columns=4, chi=2):
    tree = replace(physical_config(rows, columns, chi), maximum_intermediate_q=4)
    return FiniteSchurDecoderConfig(tree, rows//2, columns//2, 8, 1)


def bank(cfg, seed=512):
    entries = np.random.default_rng(seed).normal(size=(reduced_entry_metadata(cfg)['complex_entries'], 2))*.2
    return free_reduced_bank({'entries': jnp.asarray(entries)}, cfg)


def states(n):
    rows = np.zeros((int(__import__('math').comb(n, n//2)), n), np.int32)
    for i, up in enumerate(combinations(range(n), n//2)):
        rows[i, list(up)] = 1
    return rows


def test_nested_chi_embedding_every_n12_amplitude_and_n16_independent_probes():
    for rows, columns in ((2, 6), (4, 4)):
        a, b = config(rows, columns, 2), config(rows, columns, 4)
        old = bank(a); new = embed_reduced_bank(old, a, b)
        bits = states(a.tree.n_sites)
        if a.tree.n_sites == 16:
            bits = bits[np.random.default_rng(514).choice(len(bits), 61, replace=False)]
        va = jax.jit(jax.vmap(make_finite_schur_amplitude(a), in_axes=(None, 0)))(old, bits)
        vb = jax.jit(jax.vmap(make_finite_schur_amplitude(b), in_axes=(None, 0)))(new, bits)
        np.testing.assert_allclose(va, vb, rtol=3e-13, atol=2e-15)


def test_left_and_right_join_feature_and_exact_split_replay():
    cfg = config(); b = bank(cfg); bits = states(16)[::123]
    raw = jax.jit(jax.vmap(make_finite_schur_amplitude(cfg), in_axes=(None, 0)))
    for pair in internal_edges(cfg):
        layout = two_site_layout(cfg, *pair); joined = join_reduced_pair(b, layout)
        f = jax.jit(jax.vmap(make_two_site_features(layout), in_axes=(None, 0)))(b, bits)
        np.testing.assert_allclose(f@joined, raw(b, bits), rtol=3e-12, atol=3e-15)
        reconstructed, info = split_reduced_pair(joined, b, layout)
        np.testing.assert_allclose(join_reduced_pair(reconstructed, layout), joined, rtol=4e-12, atol=3e-15)
        np.testing.assert_allclose(raw(reconstructed, bits), raw(b, bits), rtol=4e-12, atol=3e-15)
        assert not info['physical_Schmidt_spectrum']


def test_joint_update_activates_new_copies_and_every_linear_feature_stays_singlet():
    small, large = config(2, 6, 2), config(2, 6, 4)
    b = embed_reduced_bank(bank(small), small, large)
    layout = two_site_layout(large, *internal_edges(large)[0])
    joined = join_reduced_pair(b, layout)
    arbitrary = joined + .02*(np.random.default_rng(515).normal(size=layout.size)
                               + 1j*np.random.default_rng(516).normal(size=layout.size))
    bits = states(12)
    f = np.asarray(jax.jit(jax.vmap(make_two_site_features(layout), in_axes=(None, 0)))(b, bits))
    assert spin_raising_relative_norm(f@arbitrary, bits) < 4e-13
    two, a = split_reduced_pair(arbitrary, b, layout, rank=2)
    four, d = split_reduced_pair(arbitrary, b, layout, rank=4)
    assert np.linalg.norm(np.asarray(four['nodes'][layout.child])[:, 2:]) > 1e-5
    err2 = np.linalg.norm(join_reduced_pair(two, layout)-arbitrary)
    err4 = np.linalg.norm(join_reduced_pair(four, layout)-arbitrary)
    assert err4 < err2
    raw = jax.jit(jax.vmap(make_finite_schur_amplitude(large), in_axes=(None, 0)))
    assert spin_raising_relative_norm(np.asarray(raw(four, bits)), bits) < 4e-13


def test_batched_linear_full_h_and_a1_against_independent_n12_graph():
    cfg = config(2, 6); b = bank(cfg)
    layout = two_site_layout(cfg, *internal_edges(cfg)[0]); f = make_two_site_features(layout)
    z = join_reduced_pair(b, layout)
    coefficients = np.column_stack((z, np.random.default_rng(517).normal(size=layout.size)))
    # Cyclic translations: exact A1 with nontrivial image multiplicities.
    images = tuple(tuple(np.roll(np.arange(12), k)) for k in range(12))
    projector = PhysicalSpatialProjector(images)
    bonds = tuple((i, (i+1)%12, 1.) for i in range(12))
    plan, action, projected = make_projected_linear_products(f, projector, bonds, chunk_size=16)
    bits = states(12)
    all_p = np.asarray(jax.jit(projected)(b, bits))@coefficients
    select = np.array([31, 81, 400]); prepared = jax.jit(plan)(bits[select])
    h, p, *_ = jax.jit(action)(b, jnp.asarray(coefficients), prepared)
    graph = SzHeisenbergGraph(12, bonds, n_up=6)
    indices = {tuple(x): i for i, x in enumerate(bits)}
    expected = []
    for x in bits[select]:
        connected, values = graph.get_conn(tuple(map(int, x)))
        expected.append(np.asarray(values)@all_p[[indices[tuple(y)] for y in connected]])
    np.testing.assert_allclose(p, all_p[select], rtol=4e-12, atol=4e-14)
    np.testing.assert_allclose(h, expected, rtol=4e-12, atol=5e-14)
    assert spin_raising_relative_norm(all_p[:, 1], bits) < 5e-13


def test_invalid_embedding_edges_and_rank_are_rejected():
    cfg = config(); b = bank(cfg)
    with pytest.raises(ValueError):
        embed_reduced_bank(b, cfg, config(4, 4, 1))
    with pytest.raises(ValueError):
        two_site_layout(cfg, 0, 1)
    layout = two_site_layout(cfg, *internal_edges(cfg)[0])
    with pytest.raises(ValueError):
        split_reduced_pair(join_reduced_pair(b, layout), b, layout, rank=True)
