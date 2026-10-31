"""Exact inclusion, full physical projection/H, and live new-capacity checks."""
from dataclasses import replace
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.literature_spatial_vit import (
    LiteratureSpatialViTConfig, init_literature_spatial_vit,
    literature_spatial_vit_log_amplitude,
)
from su2_nqs.models.spatial_vit_capacity import expand_spatial_vit_capacity
from su2_nqs.benchmarks.spatial_vit_sampling import make_full_heisenberg_ratios
from projects.clebsch_tree_nqs.workflows.spin import _groups
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import model_log

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('depth,ratio', [(1, 4), (2, 2), (2, 4)])
def test_inclusion_matches_every_amplitude_full_H_and_projected_state(depth, ratio):
    source = LiteratureSpatialViTConfig(geometry='square', rows=2, columns=2,
        patch_rows=1, patch_columns=1, embedding_dim=4, head_count=2, depth=1, mlp_ratio=2)
    target = replace(source, depth=depth, mlp_ratio=ratio)
    params = init_literature_spatial_vit(jax.random.PRNGKey(319), source)
    params = jax.tree.map(lambda a: a+.017, params)  # Not an initialization-only coincidence.
    expanded = expand_spatial_vit_capacity(params, source, target, jax.random.PRNGKey(327))
    bits = jnp.asarray([[int(i in subset) for i in range(4)]
                        for subset in combinations(range(4), 2)])
    group = _groups('square', 2, 2)['gamma_a1']
    spec = dict(model_family='literature', spin_inversion_even=True)
    old, _ = model_log(spec, source, group)
    new, _ = model_log(spec, target, group)
    raw_old = jax.vmap(lambda x: literature_spatial_vit_log_amplitude(params, x, source))(bits)
    raw_new = jax.vmap(lambda x: literature_spatial_vit_log_amplitude(expanded, x, target))(bits)
    np.testing.assert_allclose(np.exp(raw_new-raw_old), 1, atol=2e-12, rtol=0)
    a = np.exp(jax.vmap(old, in_axes=(None, 0))(params, bits))
    b = np.exp(jax.vmap(new, in_axes=(None, 0))(expanded, bits))
    np.testing.assert_allclose(b, a, atol=2e-12, rtol=2e-12)
    bonds = [(0, 1, 1.), (0, 2, 1.), (1, 3, 1.), (2, 3, 1.), (0, 3, .125)]
    h = np.zeros((6, 6)); lookup = {tuple(x): i for i, x in enumerate(np.asarray(bits))}
    for k, x in enumerate(np.asarray(bits)):
        for i, j, weight in bonds:
            h[k, k] += weight*(.25 if x[i] == x[j] else -.25)
            if x[i] != x[j]:
                y = x.copy(); y[i], y[j] = y[j], y[i]
                h[lookup[tuple(y)], k] += weight*.5
    expected = h@a/a
    actual = make_full_heisenberg_ratios(new, bonds)(expanded, bits)
    np.testing.assert_allclose(actual, expected, atol=4e-12, rtol=2e-12)


def test_opened_output_channels_have_non_gauge_physical_tangent():
    source = LiteratureSpatialViTConfig(geometry='triangular', rows=4, columns=4,
        patch_rows=2, patch_columns=2, embedding_dim=4, head_count=2, depth=1, mlp_ratio=2)
    target = replace(source, mlp_ratio=4)
    params = init_literature_spatial_vit(jax.random.PRNGKey(917), source)
    expanded = expand_spatial_vit_capacity(params, source, target, jax.random.PRNGKey(921))
    rows = jnp.asarray(np.stack([np.random.default_rng(seed).permutation([0]*8+[1]*8)
                               for seed in range(10, 26)]))
    log_fn, _ = model_log(dict(model_family='literature', spin_inversion_even=True),
                         target, _groups('triangular', 4, 4)['gamma_a1'])
    direction = jax.tree.map(jnp.zeros_like, expanded)
    layer = direction['layers'][0]
    layer['mlp_out'] = layer['mlp_out'].at[source.mlp_dim:].set(
        jax.random.normal(jax.random.PRNGKey(925), (source.mlp_dim, source.embedding_dim)))
    tangent = jax.jit(lambda p, d: jax.jvp(lambda q: jax.vmap(log_fn,
        in_axes=(None, 0))(q, rows), (p,), (d,))[1])(expanded, direction)
    assert np.std(np.asarray(tangent)) > 1e-7  # Not a uniform normalization/phase gauge.
    delta = 1e-5
    plus = jax.tree.map(lambda a, b: a+delta*b, expanded, direction)
    minus = jax.tree.map(lambda a, b: a-delta*b, expanded, direction)
    derivative = (jax.vmap(log_fn, in_axes=(None, 0))(plus, rows)
                  -jax.vmap(log_fn, in_axes=(None, 0))(minus, rows))/(2*delta)
    np.testing.assert_allclose(tangent, derivative, atol=2e-7, rtol=2e-6)


def test_invalid_geometry_shrink_or_parent_shape_is_rejected():
    c = LiteratureSpatialViTConfig(geometry='square', rows=2, columns=2,
        patch_rows=1, patch_columns=1, embedding_dim=4, head_count=2, depth=2)
    params = init_literature_spatial_vit(jax.random.PRNGKey(47), c)
    for target in (replace(c, depth=1), replace(c, rows=4), replace(c, embedding_dim=8)):
        with pytest.raises(ValueError):
            expand_spatial_vit_capacity(params, c, target, jax.random.PRNGKey(51))
    bad = dict(params, patch_embedding=params['patch_embedding'][:, :-1])
    with pytest.raises(ValueError, match='parent arrays'):
        expand_spatial_vit_capacity(bad, c, c, jax.random.PRNGKey(51))
