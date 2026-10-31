"""Independent dense-ridge oracles and fixed-score workspace checks."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.parameter_block_sr import solve_parameter_block_sr

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('components', [1, 2])
@pytest.mark.parametrize('parameters,chunk', [(1, 8), (17, 1), (17, 6), (17, 17), (17, 64), (34, 8)])
def test_parameter_block_matches_independent_dense_primal(components, parameters, chunk):
    rng = np.random.default_rng(71021+parameters+components)
    scores = rng.normal(size=(7, components, parameters))
    scores[:, :, 0] = 1.  # Exact normalization / phase gauge must vanish.
    force = rng.normal(size=7*components)
    ridge, scale = .13, np.sqrt(2/7)
    centered = scores-scores.mean(axis=0, keepdims=True)
    a = centered.transpose(1, 0, 2).reshape(7*components, parameters)*scale
    gradient = a.T@force
    expected = np.linalg.solve(a.T@a+ridge*np.eye(parameters), gradient)
    function = jax.jit(partial(solve_parameter_block_sr, damping=ridge,
        score_scale=scale, parameter_chunk_size=chunk))
    direction, actual_g, info = function(jnp.asarray(scores), jnp.asarray(force))
    np.testing.assert_allclose(direction, expected, atol=3e-13, rtol=3e-12)
    np.testing.assert_allclose(actual_g, gradient, atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(info['raw_fisher_norm'], np.linalg.norm(a@expected), atol=3e-13, rtol=3e-13)
    residual = a.T@(a@np.asarray(direction))+ridge*np.asarray(direction)-gradient
    assert np.linalg.norm(residual)/max(np.linalg.norm(gradient), 1e-30) < 3e-12
    assert float(info['relative_linear_residual']) < 3e-12
    assert int(info['maximum_normalized_score_block_entries']) == 7*components*min(chunk, parameters)


def test_parameter_block_collinear_small_ridge_and_energy_shift():
    rng = np.random.default_rng(7102104)
    scores = rng.normal(size=(9, 2, 23))
    scores[:, :, 4:] = scores[:, :, 1:2]+1e-7*rng.normal(size=(9, 2, 19))
    e = rng.normal(size=9)+1j*rng.normal(size=9)
    outputs = []
    for energy in (e, e+5+3j):
        centered = energy-energy.mean()
        force = np.sqrt(2/9)*np.concatenate((centered.real, centered.imag))
        outputs.append(solve_parameter_block_sr(jnp.asarray(scores), jnp.asarray(force),
            damping=.05, score_scale=np.sqrt(2/9), parameter_chunk_size=7))
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=1e-13, rtol=1e-12)
    assert float(outputs[0][2]['relative_linear_residual']) < 1e-12


@pytest.mark.parametrize('change', ['shape', 'force', 'complex', 'chunk_bool', 'chunk_zero', 'ridge', 'scale'])
def test_parameter_block_invalid_contract_rejected(change):
    scores, force = jnp.ones((4, 2, 7)), jnp.ones(8)
    options = dict(damping=.1, score_scale=.5, parameter_chunk_size=3)
    if change == 'shape': scores = scores[0]
    if change == 'force': force = force[:-1]
    if change == 'complex': scores = scores.astype(jnp.complex128)
    if change == 'chunk_bool': options['parameter_chunk_size'] = True
    if change == 'chunk_zero': options['parameter_chunk_size'] = 0
    if change == 'ridge': options['damping'] = 0.
    if change == 'scale': options['score_scale'] = float('nan')
    with pytest.raises(ValueError): solve_parameter_block_sr(scores, force, **options)


def test_compiled_workspace_does_not_require_full_centered_score_copy():
    # Enough parameter columns to resolve score-sized copies in compiler
    # memory analysis, without allocating N144-sized tensors on a laptop.
    count, components, parameters, chunk = 16, 2, 131071, 1024
    function = jax.jit(partial(solve_parameter_block_sr, damping=.1,
        score_scale=np.sqrt(2/count), parameter_chunk_size=chunk))
    compiled = function.lower(jax.ShapeDtypeStruct((count, components, parameters), jnp.float64),
        jax.ShapeDtypeStruct((count*components,), jnp.float64)).compile()
    memory = compiled.memory_analysis()
    raw_bytes = count*components*parameters*8
    assert memory.temp_size_in_bytes < raw_bytes*.5, memory
