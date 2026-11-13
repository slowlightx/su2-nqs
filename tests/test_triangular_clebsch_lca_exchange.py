"""The LCA-environment exchange query must equal the legacy root-recompute query."""
from __future__ import annotations

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank
from su2_nqs.models.triangular_clebsch_lca_exchange import lca_exchange_plan, make_lca_exchange_queries
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, init_mixed_clebsch_family
from su2_nqs.vmc.triangular_clebsch_cached_action import (
    make_cached_projected_heisenberg_action, make_cached_projected_heisenberg_bridge,
)


def _prepared(config, seed):
    bank = init_mixed_clebsch_family(jax.random.PRNGKey(seed), config)['ternary_free']
    return prepare_feature_lut_bank(bank, config)


@pytest.mark.parametrize('length,cap,qmax', [(6, 9, 6), (12, 9, 4)])
def test_lca_query_equals_legacy_query_on_every_bond(length, cap, qmax):
    config = MixedClebschConfig(length=length, multiplicity_cap=cap, maximum_q=qmax)
    prepared = _prepared(config, length)
    edges = np.asarray([(a, b) for a, b, _ in triangular_j1j2_bonds(length, length, j2=.125, periodic=True)],
                       np.int32)
    plan = lca_exchange_plan(config, edges)
    assert sorted(i for g in plan for i in g['indices']) == list(range(len(edges)))
    legacy = jax.jit(make_feature_lut_exchange_queries(config, edges))
    fast = jax.jit(make_lca_exchange_queries(config, edges))
    rng = np.random.default_rng(length)
    n = length**2
    for _ in range(2):
        bits = jnp.asarray(rng.permutation(np.repeat([0, 1], n//2)).astype(np.int32))
        p0, s0 = (np.asarray(v) for v in legacy(prepared, bits))
        p1, s1 = (np.asarray(v) for v in fast(prepared, bits))
        scale = max(np.max(np.abs(s0)), abs(p0))
        assert abs(p0-p1) <= 1e-14*scale
        assert np.max(np.abs(s0-s1)) <= 1e-12*scale
    # Illegal total magnetization gives zero, as in the legacy factory.
    bits = jnp.asarray(np.r_[np.ones(n//2+1), np.zeros(n//2-1)].astype(np.int32))
    _, s = fast(prepared, bits)
    assert np.all(np.asarray(s) == 0.)


def test_lca_projected_action_equals_cached_action():
    config = MixedClebschConfig(length=6, multiplicity_cap=9, maximum_q=6)
    prepared = _prepared(config, 3)
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    bonds = tuple(triangular_j1j2_bonds(6, 6, j2=.125, periodic=True))
    rng = np.random.default_rng(5)
    rows = jnp.asarray(np.stack([rng.permutation(np.repeat([0, 1], 18)) for _ in range(2)]).astype(np.int32))
    p0, h0 = (np.asarray(v) for v in make_cached_projected_heisenberg_action(
        config, images, bonds, exchange_kernel='legacy')(prepared, rows))
    p1, h1 = (np.asarray(v) for v in make_cached_projected_heisenberg_action(
        config, images, bonds, exchange_kernel='lca')(prepared, rows))
    np.testing.assert_allclose(p1, p0, rtol=1e-13, atol=0)
    np.testing.assert_allclose(h1, h0, rtol=1e-11, atol=0)
    # The bridge factory exposes the same kernel choice with identical outputs.
    probabilities = np.full(len(bonds), 1/len(bonds))
    legacy_bridge, _ = make_cached_projected_heisenberg_bridge(config, images, bonds, probabilities,
                                                               exchange_kernel='legacy')
    lca_bridge, _ = make_cached_projected_heisenberg_bridge(config, images, bonds, probabilities,
                                                            exchange_kernel='lca')
    for a, b in zip(legacy_bridge(prepared, rows), lca_bridge(prepared, rows), strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-11, atol=0)


def test_lca_kernel_rejects_active_widths_and_unknown_names():
    config = MixedClebschConfig(length=6, multiplicity_cap=9, maximum_q=6)
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    bonds = tuple(triangular_j1j2_bonds(6, 6, j2=.125, periodic=True))
    with pytest.raises(ValueError):
        make_cached_projected_heisenberg_action(config, images, bonds, exchange_kernel='lca',
                                                active_top_widths=({0: 1},))
    with pytest.raises(ValueError):
        make_cached_projected_heisenberg_action(config, images, bonds, exchange_kernel='fast')


def test_lca_query_exact_for_complex_banks():
    config = MixedClebschConfig(length=12, multiplicity_cap=9, maximum_q=4)
    real = init_mixed_clebsch_family(jax.random.PRNGKey(1), config)['ternary_free']
    imag = init_mixed_clebsch_family(jax.random.PRNGKey(2), config)['ternary_free']
    bank = jax.tree_util.tree_map(lambda a, b: a+.3j*b, real, imag)
    prepared = prepare_feature_lut_bank(bank, config)
    edges = np.asarray([(a, b) for a, b, _ in triangular_j1j2_bonds(12, 12, j2=.125, periodic=True)], np.int32)
    bits = jnp.asarray(np.random.default_rng(3).permutation(np.repeat([0, 1], 72)).astype(np.int32))
    p0, s0 = (np.asarray(v) for v in jax.jit(make_feature_lut_exchange_queries(config, edges))(prepared, bits))
    p1, s1 = (np.asarray(v) for v in jax.jit(make_lca_exchange_queries(config, edges))(prepared, bits))
    assert np.iscomplexobj(s1)
    scale = max(np.max(np.abs(s0)), abs(p0))
    assert abs(p0-p1) <= 1e-14*scale
    assert np.max(np.abs(s0-s1)) <= 1e-12*scale


def test_default_auto_kernel_selects_lca_and_falls_back_for_active_widths(monkeypatch):
    import su2_nqs.vmc.triangular_clebsch_cached_action as module
    config = MixedClebschConfig(length=6, multiplicity_cap=9, maximum_q=6)
    chosen = []
    monkeypatch.setattr(module, 'make_lca_exchange_queries',
                        lambda *a, **k: chosen.append('lca') or make_lca_exchange_queries(*a, **k))
    monkeypatch.setattr(module, 'make_feature_lut_exchange_queries',
                        lambda *a, **k: chosen.append('legacy') or make_feature_lut_exchange_queries(*a, **k))
    edges = np.asarray([[0, 1]], np.int32)
    module._exchange_queries(config, edges, None, 'auto')
    assert chosen == ['lca']
    module._validate_kernel(({0: 1},), 'auto')  # auto never rejects active widths
