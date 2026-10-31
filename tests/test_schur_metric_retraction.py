"""Whitened truncation versus exhaustive physical amplitudes, not tensor SVD."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.models.finite_schur_decoder import make_finite_schur_amplitude
from su2_nqs.models.schur_metric_retraction import split_physical_reduced_pair
from su2_nqs.models.schur_two_site import (
    embed_reduced_bank, internal_edges, join_reduced_pair,
    make_two_site_features, split_reduced_pair, two_site_layout,
)
from test_schur_two_site import bank, config, states

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('shape', [(2, 6), (4, 4)])
def test_exact_unprojected_metric_error_and_best_rank_against_all_sz_amplitudes(shape):
    cfg = config(*shape); b = bank(cfg, 1710); bits = states(cfg.tree.n_sites)
    for pair in internal_edges(cfg):
        layout = two_site_layout(cfg, *pair)
        rng = np.random.default_rng(1711 + pair[1])
        z = rng.normal(size=layout.size) + 1j*rng.normal(size=layout.size)
        f = np.asarray(jax.jit(jax.vmap(make_two_site_features(layout), in_axes=(None, 0)))(b, bits))
        target = f@z
        coordinate, _ = split_reduced_pair(z, b, layout, rank=1)
        actual, info = split_physical_reduced_pair(z, b, layout, rank=1,
            comparison_entries=join_reduced_pair(coordinate,layout))
        result = f@join_reduced_pair(actual, layout)
        coordinate_result = f@join_reduced_pair(coordinate, layout)
        np.testing.assert_allclose(info['physical_norm_squared'], np.vdot(target, target).real, rtol=2e-12)
        error = np.linalg.norm(target-result)**2
        np.testing.assert_allclose(info['discarded_physical_norm_squared'], error, rtol=3e-12, atol=2e-13)
        assert error <= np.linalg.norm(target-coordinate_result)**2 + 2e-12*np.vdot(target,target).real
        np.testing.assert_allclose(info['comparison_discarded_physical_norm_squared'],
                                  np.linalg.norm(target-coordinate_result)**2,rtol=4e-12)
        assert spin_raising_relative_norm(result, bits) < 8e-13
        assert info['full_projected_state_still_requires_actual_amplitude_check']
        assert 'UNPROJECTED' in info['metric_scope']


def test_zero_padded_gram_nullspace_preserves_all_n12_physical_amplitudes():
    small, large = config(2,6,2), config(2,6,4)
    b = embed_reduced_bank(bank(small,1712), small, large)
    layout = two_site_layout(large, *internal_edges(large)[0])
    z = join_reduced_pair(b, layout)
    result, info = split_physical_reduced_pair(z, b, layout)
    raw = jax.jit(jax.vmap(make_finite_schur_amplitude(large),in_axes=(None,0)))
    bits = states(12)
    before, after = np.asarray(raw(b,bits)), np.asarray(raw(result,bits))
    np.testing.assert_allclose(after,before,rtol=4e-12,atol=3e-15)
    # Exact physical-amplitude error, versus eigen-SVD roundoff in a Gram
    # with numerical null eigenvalues. Never relabel that floor as a rank loss.
    measured = np.linalg.norm(after-before)**2/np.linalg.norm(before)**2
    assert measured < 2e-24
    assert abs(info['relative_physical_error']-measured) < 16*np.finfo(float).eps
    assert any(v['row_metric']['resolved_rank'] < v['row_metric']['dimension'] for v in info['q_blocks'].values())


def test_sibling_coordinate_gauge_does_not_change_physical_rank_one_retraction():
    cfg=config();b=bank(cfg,1713);layout=two_site_layout(cfg,2,0)
    rng=np.random.default_rng(1714)
    z=rng.normal(size=layout.size)+1j*rng.normal(size=layout.size)
    transform=np.array([[3.+.3j,.5j],[.2, .4-.2j]])
    inverse=np.linalg.inv(transform)
    arrays=[np.asarray(v).copy() for v in b['nodes']]
    arrays[1]=np.einsum('ab,cblm->calm',transform,arrays[1])
    arrays[2]=np.einsum('cals,st->calt',arrays[2],inverse)
    gauge={'nodes':tuple(jnp.asarray(v) for v in arrays)}
    zg=z.copy()
    for block in layout.blocks:
        zg[block.start:block.stop]=np.einsum('kabs,st->kabt',z[block.start:block.stop].reshape(block.shape),inverse).reshape(-1)
    original, info=split_physical_reduced_pair(z,b,layout,rank=1)
    changed, other=split_physical_reduced_pair(zg,gauge,layout,rank=1)
    bits=states(16)[::23]
    f=np.asarray(jax.jit(jax.vmap(make_two_site_features(layout),in_axes=(None,0)))(b,bits))
    fg=np.asarray(jax.jit(jax.vmap(make_two_site_features(layout),in_axes=(None,0)))(gauge,bits))
    np.testing.assert_allclose(fg@zg,f@z,rtol=5e-12,atol=2e-14)
    np.testing.assert_allclose(fg@join_reduced_pair(changed,layout),f@join_reduced_pair(original,layout),rtol=8e-11,atol=2e-13)
    np.testing.assert_allclose(info['discarded_physical_norm_squared'],other['discarded_physical_norm_squared'],rtol=8e-12)


def test_invalid_rank_and_cutoff_are_rejected():
    cfg=config();b=bank(cfg);layout=two_site_layout(cfg,2,0);z=join_reduced_pair(b,layout)
    for rank in (0,3,True):
        with pytest.raises(ValueError):split_physical_reduced_pair(z,b,layout,rank=rank)
    for cutoff in (0.,1.,float('nan'),True):
        with pytest.raises(ValueError):split_physical_reduced_pair(z,b,layout,metric_cutoff=cutoff)
