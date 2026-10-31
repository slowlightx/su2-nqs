"""Independent author-aligned architecture, projection and full-H checks."""
from dataclasses import replace
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.literature_spatial_vit import (
    LiteratureSpatialViTConfig, init_literature_spatial_vit,
    literature_spatial_vit_log_amplitude, literature_relative_geometry,
)
from su2_nqs.models.fixed_sz_spatial_vit import fixed_sz_spatial_vit_parameter_count
from su2_nqs.benchmarks.spatial_vit_sampling import make_full_heisenberg_ratios
from projects.clebsch_tree_nqs.workflows.spin import _groups
from projects.clebsch_tree_nqs.workflows.spin_spatial_vit_quality import model_log


def numpy_oracle(p, bits, c):
    """Explicit patch slicing and head loops, without using model helpers."""
    p = jax.tree_util.tree_map(np.asarray, p)
    grid = 2*np.asarray(bits).reshape(c.rows, c.columns)-1
    patches = np.array([grid[r:r+c.patch_rows, s:s+c.patch_columns].ravel()
                       for r in range(0, c.rows, c.patch_rows)
                       for s in range(0, c.columns, c.patch_columns)])
    coords = [(r, s) for r in range(c.patch_grid_rows) for s in range(c.patch_grid_columns)]
    def ln(x, weights):
        mean = np.mean(x, axis=-1, keepdims=True)
        return ((x-mean)/np.sqrt(np.mean((x-mean)**2, axis=-1, keepdims=True)
                                +c.layer_norm_epsilon))*weights['scale']+weights['bias']
    x = patches@p['patch_embedding']+p['patch_embedding_bias']
    for l in p['layers']:
        v = (ln(x, l['norm_spatial'])@l['value']+l['value_bias']).reshape(c.token_count, c.head_count, c.head_dim)
        mixed = np.zeros_like(v)
        for i, (r, s) in enumerate(coords):
            for h in range(c.head_count):
                alpha, distances = [], []
                for t, u in coords:
                    alpha.append(l['relative_factor'][h, ((t-r)%c.patch_grid_rows)*c.patch_grid_columns+(u-s)%c.patch_grid_columns])
                    candidates = []
                    for a in (-1, 0, 1):
                        for b in (-1, 0, 1):
                            dr, dc = (t-r)*c.patch_rows+a*c.rows, (u-s)*c.patch_columns+b*c.columns
                            candidates.append(dr*dr+dc*dc-(dr*dc if c.geometry == 'triangular' else 0))
                    distances.append(np.sqrt(min(candidates)))
                w = np.array(alpha)
                if c.use_distance_envelope:
                    decay = np.logaddexp(0, l['raw_distance_decay'][h])
                    envelope = np.exp(-decay*np.array(distances)); envelope /= envelope.sum()
                    w *= envelope
                mixed[i, h] = w@v[:, h]
        x = x+mixed.reshape(c.token_count, c.embedding_dim)@l['output']+l['output_bias']
        z = ln(x, l['norm_mlp'])@l['mlp_in']+l['mlp_in_bias']
        gelu = .5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+.044715*z**3)))
        x = x+gelu@l['mlp_out']+l['mlp_out_bias']
    pooled = ln(x.sum(axis=0), p['final_norm']) if c.pooling == 'authors' else ln(x, p['final_norm']).sum(axis=0)
    h = p['log_cosh_head']
    z = ln(pooled@h['weight_real']+h['bias_real'], h['norm_real'])+1j*ln(pooled@h['weight_imag']+h['bias_imag'], h['norm_imag'])
    return np.prod(np.cosh(z))


@pytest.mark.parametrize('geometry', ['square', 'triangular'])
@pytest.mark.parametrize('envelope', [True, False])
@pytest.mark.parametrize('pooling', ['authors', 'tokenwise_control'])
def test_explicit_numpy_encoder_oracle(geometry, envelope, pooling):
    c = LiteratureSpatialViTConfig(geometry=geometry, rows=4, columns=4,
        embedding_dim=6, head_count=2, depth=2, pooling=pooling,
        use_distance_envelope=envelope)
    p = init_literature_spatial_vit(jax.random.PRNGKey(1201), c)
    # Nonzero LN and Dense biases ensure equivalence is not an initialization accident.
    p = jax.tree_util.tree_map(lambda v: v+.013, p)
    bits = np.random.default_rng(32).permutation([0]*8+[1]*8)
    expected = numpy_oracle(p, bits, c)
    actual = np.exp(literature_spatial_vit_log_amplitude(p, bits, c))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_tutorial_parameter_count_and_initializer_and_mask():
    c = LiteratureSpatialViTConfig(geometry='square', rows=10, columns=10,
        embedding_dim=60, head_count=10, depth=4, use_distance_envelope=False)
    p = init_literature_spatial_vit(jax.random.PRNGKey(82), c)
    assert fixed_sz_spatial_vit_parameter_count(p) == 155620
    assert abs(p['patch_embedding']).max() <= np.sqrt(6/(4+60))
    assert np.isneginf(literature_spatial_vit_log_amplitude(p, jnp.zeros(100), c).real)
    assert np.isneginf(literature_spatial_vit_log_amplitude(p, jnp.full(100, 2), c).real)
    np.testing.assert_array_equal(p['layers'][0]['value_bias'], 0.)
    small = replace(c, rows=4, columns=4)
    with pytest.raises(ValueError):
        literature_spatial_vit_log_amplitude(p, jnp.zeros(15), small)


def test_physical_metric_and_coarse_translation_invariance():
    c = LiteratureSpatialViTConfig(geometry='triangular', rows=6, columns=6,
        patch_rows=1, patch_columns=1, embedding_dim=4, head_count=2)
    _, distance = literature_relative_geometry(c)
    np.testing.assert_allclose(distance[0, [1, 6, 7]], 1.)
    p = init_literature_spatial_vit(jax.random.PRNGKey(2), c)
    bits = np.random.default_rng(53).permutation([0]*18+[1]*18)
    f = jax.jit(lambda b: literature_spatial_vit_log_amplitude(p, b, c))
    shifted = np.roll(bits.reshape(6, 6), (1, 2), (0, 1)).ravel()
    np.testing.assert_allclose(np.exp(f(bits)-f(shifted)), 1., atol=1e-12)


def test_full_reynolds_parity_H_and_gradient_independently():
    c = LiteratureSpatialViTConfig(geometry='square', rows=2, columns=2,
        patch_rows=1, patch_columns=1, embedding_dim=4, head_count=2)
    p = init_literature_spatial_vit(jax.random.PRNGKey(128), c)
    group = np.asarray(_groups('square', 2, 2)['gamma_a1'])
    fn, _ = model_log(dict(model_family='literature', spin_inversion_even=True), c, group)
    bits = np.array([[int(i in subset) for i in range(4)] for subset in combinations(range(4), 2)])
    amplitudes = np.exp(jax.vmap(fn, in_axes=(None, 0))(p, bits))
    expected = np.array([np.mean([numpy_oracle(p, x[g], c) for g in group]
                         +[numpy_oracle(p, 1-x[g], c) for g in group]) for x in bits])
    np.testing.assert_allclose(amplitudes, expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(jax.vmap(fn, in_axes=(None, 0))(p, 1-bits),
                               jax.vmap(fn, in_axes=(None, 0))(p, bits), atol=2e-12)
    bonds = [(0, 1, 1.), (1, 3, 1.), (2, 3, 1.), (0, 2, 1.), (0, 3, .125)]
    h = np.zeros((6, 6))
    lookup = {tuple(x): i for i, x in enumerate(bits)}
    for j, x in enumerate(bits):
        for a, b, coupling in bonds:
            h[j, j] += coupling*(.25 if x[a] == x[b] else -.25)
            if x[a] != x[b]:
                y = x.copy(); y[a], y[b] = y[b], y[a]
                h[lookup[tuple(y)], j] += .5*coupling
    local = make_full_heisenberg_ratios(fn, bonds)(p, jnp.asarray(bits))
    np.testing.assert_allclose(local, h@amplitudes/amplitudes, atol=3e-12)
    direction = jax.tree_util.tree_map(lambda v: jnp.ones_like(v)*.01, p)
    f = lambda w: fn(w, bits[0])
    tangent = jax.jvp(f, (p,), (direction,))[1]
    plus = jax.tree_util.tree_map(lambda a, b: a+1e-5*b, p, direction)
    minus = jax.tree_util.tree_map(lambda a, b: a-1e-5*b, p, direction)
    np.testing.assert_allclose(tangent, (f(plus)-f(minus))/2e-5, atol=1e-8, rtol=1e-7)


def test_optional_flax_dense_and_normalization_parity():
    """Explicit-array oracle; Flax remains optional for the main environment."""
    nn = pytest.importorskip('flax.linen')
    from su2_nqs.models.literature_spatial_vit import literature_layer_norm
    c = LiteratureSpatialViTConfig(geometry='triangular', rows=6, columns=6,
                                  embedding_dim=12, head_count=3, depth=2)
    p = init_literature_spatial_vit(jax.random.PRNGKey(13), c)
    x = jax.random.normal(jax.random.PRNGKey(31), (4, 12), dtype=jnp.float64)
    l = p['layers'][0]
    # Test both pooling orders and nontrivial trained LN parameters.
    norm = dict(scale=l['norm_spatial']['scale']*.8, bias=jnp.arange(12)*.017)
    flax_ln = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
    flax_dense = nn.Dense(12, dtype=jnp.float64, param_dtype=jnp.float64)
    for values in (x, x.sum(axis=0)):
        expected = flax_dense.apply({'params': {'kernel': l['value'], 'bias': l['value_bias']}},
            flax_ln.apply({'params': norm}, values))
        actual = literature_layer_norm(values, norm)@l['value']+l['value_bias']
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
