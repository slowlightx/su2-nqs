"""Size transfer is an initializer, not a hidden change to existing states."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.literature_spatial_vit import (
    LiteratureSpatialViTConfig, init_literature_spatial_vit,
    literature_spatial_vit_log_amplitude,
)
from su2_nqs.models.spatial_vit_transfer import transfer_spatial_vit_size
from projects.clebsch_tree_nqs.workflows.spin import _groups
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import model_log

jax.config.update('jax_enable_x64', True)


def test_equal_size_replays_every_leaf_and_other_architecture_changes_fail():
    c = LiteratureSpatialViTConfig(geometry='triangular', rows=6, columns=6,
        patch_rows=3, patch_columns=3, embedding_dim=4, head_count=2, depth=1)
    p = init_literature_spatial_vit(jax.random.PRNGKey(1), c)
    actual = transfer_spatial_vit_size(p, c, c)
    for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match='architecture'):
        transfer_spatial_vit_size(p, c, replace(c, depth=2))


def test_periodic_interpolation_inherits_coincident_entries_and_all_other_weights():
    c = LiteratureSpatialViTConfig(geometry='triangular', rows=6, columns=6,
        patch_rows=3, patch_columns=3, embedding_dim=4, head_count=2, depth=1)
    p = init_literature_spatial_vit(jax.random.PRNGKey(2), c)
    target = replace(c, rows=12, columns=12)
    q = transfer_spatial_vit_size(p, c, target)
    old = np.asarray(p['layers'][0]['relative_factor']).reshape(2, 2, 2)
    new = np.asarray(q['layers'][0]['relative_factor']).reshape(2, 4, 4)
    np.testing.assert_array_equal(new[:, ::2, ::2], old)
    np.testing.assert_allclose(new[:, 1, 1], old.mean((1, 2)), atol=1e-15)
    assert new.min() >= old.min() and new.max() <= old.max()
    for name in p['layers'][0]:
        if name != 'relative_factor':
            for a, b in zip(jax.tree.leaves(q['layers'][0][name]),
                            jax.tree.leaves(p['layers'][0][name]), strict=True):
                np.testing.assert_array_equal(a, b)
    bits = jnp.asarray(np.random.default_rng(3).permutation([0]*72+[1]*72))
    assert np.isfinite(literature_spatial_vit_log_amplitude(q, bits, target))


@pytest.mark.parametrize('rule', ['normalized_bilinear', 'folded_spatial_kernel'])
def test_transferred_target_keeps_complete_coherent_a1_and_legal_m0_support(rule):
    c = LiteratureSpatialViTConfig(geometry='square', rows=2, columns=2,
        patch_rows=1, patch_columns=1, embedding_dim=4, head_count=2, depth=1)
    target = replace(c, rows=4, columns=4)
    p = transfer_spatial_vit_size(init_literature_spatial_vit(jax.random.PRNGKey(4), c), c, target, rule=rule)
    group = _groups('square', 4, 4)['gamma_a1']
    spec = dict(model_family='literature', coordinate_convention='physical120', spin_inversion_even=True)
    log_fn, images = model_log(spec, target, group)
    bits = jnp.asarray(np.random.default_rng(5).permutation([0]*8+[1]*8))
    transformed = bits[jnp.asarray(group)[jnp.asarray([0, 7, 29, -1])]]
    values = np.asarray(jax.jit(jax.vmap(log_fn, in_axes=(None, 0)))(p, transformed))
    np.testing.assert_allclose(np.exp(values-values[0]), 1., atol=2e-13)
    assert images == 2*len(group)//target.token_count
    # Independent full Reynolds sum, including both spin-inversion images.
    rows = jnp.concatenate((bits[jnp.asarray(group)], 1-bits[jnp.asarray(group)]))
    raw = np.asarray(jax.vmap(lambda x: literature_spatial_vit_log_amplitude(p, x, target))(rows))
    shift = raw.real.max()
    np.testing.assert_allclose(np.exp(values[0]-shift), np.exp(raw-shift).mean(), atol=2e-13)
    assert np.isneginf(log_fn(p, jnp.zeros(16, dtype=jnp.int32)).real)


@pytest.mark.parametrize('geometry', ['square', 'triangular'])
def test_folded_kernel_preserves_filter_on_tiled_inputs(geometry):
    from su2_nqs.models.literature_spatial_vit import literature_relative_geometry
    source = LiteratureSpatialViTConfig(geometry=geometry, rows=6, columns=6,
        patch_rows=3, patch_columns=3, embedding_dim=4, head_count=2, depth=2,
        layer_norm_epsilon=1e-14)
    target = replace(source, rows=12, columns=18)
    p = init_literature_spatial_vit(jax.random.PRNGKey(21), source)
    q = transfer_spatial_vit_size(p, source, target, rule='folded_spatial_kernel')
    # Explicit NumPy exponential weights and spatial residue classes provide
    # an independent linear-filter oracle, without using transfer internals.
    alias = np.array([(r % 2)*2+c % 2 for r in range(4) for c in range(6)])
    d0 = literature_relative_geometry(source)[1][0]
    d1 = literature_relative_geometry(target)[1][0]
    for before, after in zip(p['layers'], q['layers'], strict=True):
        decay = np.logaddexp(0, np.asarray(before['raw_distance_decay']))[:, None]
        a = np.exp(-decay*d0); a /= a.sum(axis=1, keepdims=True)
        b = np.exp(-decay*d1); b /= b.sum(axis=1, keepdims=True)
        actual = np.asarray(after['relative_factor'])*b
        folded = np.stack([actual[:, alias == i].sum(axis=1) for i in range(4)], axis=1)
        np.testing.assert_allclose(folded, np.asarray(before['relative_factor'])*a,
                                   rtol=3e-14, atol=3e-15)
    row = np.random.default_rng(29).permutation([0]*18+[1]*18)
    tiled = np.tile(row.reshape(6, 6), (2, 3)).ravel()
    f0 = literature_spatial_vit_log_amplitude(p, row, source)
    f1 = literature_spatial_vit_log_amplitude(q, tiled, target)
    # Tiny epsilon isolates filter preservation from finite-epsilon pooling
    # normalization. This is a repeated configuration, not a product state.
    np.testing.assert_allclose(np.exp(f1-f0), 1., rtol=2e-11, atol=2e-11)


def test_folded_kernel_rejects_unsupported_resize_without_changing_old_default():
    c = LiteratureSpatialViTConfig(geometry='triangular', rows=6, columns=6,
        patch_rows=3, patch_columns=3, embedding_dim=4, head_count=2, depth=1)
    p = init_literature_spatial_vit(jax.random.PRNGKey(11), c)
    with pytest.raises(ValueError, match='integer size multiples'):
        transfer_spatial_vit_size(p, c, replace(c, rows=9, columns=12), rule='folded_spatial_kernel')
    with pytest.raises(ValueError, match='unknown'):
        transfer_spatial_vit_size(p, c, c, rule='unspecified')
