"""Independent node-safe products and the sealed fixed-budget stage boundary."""
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_finite_schur_optimize import (
    CONFIG, centered_products, validate_spec)
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.vmc.projected_spin_products import make_projected_heisenberg_products

jax.config.update('jax_enable_x64', True)


def test_centered_additive_products_agree_with_independent_dense_H_at_nodes():
    basis = np.asarray([x for x in itertools.product((0, 1), repeat=4) if sum(x) == 2], dtype=np.int32)
    codes = jnp.asarray([8, 4, 2, 1])
    bonds = ((0, 1, 1.), (1, 2, .3), (2, 3, -.2))
    projector = PhysicalSpatialProjector((tuple(range(4)), (3, 2, 1, 0)))
    raw = lambda bank, x: bank[jnp.dot(x, codes)]
    _, _, action = make_projected_heisenberg_products(raw, projector, bonds, chunk_size=4)
    indices = basis@np.asarray(codes)
    phi = jnp.arange(16, dtype=jnp.float64).astype(jnp.complex128).at[3].set(0).at[12].set(0)
    ref = jnp.cos(jnp.arange(16))+.1j*jnp.sin(jnp.arange(16))
    c = jnp.ones(16, dtype=jnp.complex128)
    hp, p, _, _ = jax.jit(action)(phi, basis)
    hr, r, _, _ = jax.jit(action)(ref, basis)
    hc, carrier, _, _ = jax.jit(action)(c, basis)
    images = np.asarray(projector.site_permutations)
    expected = np.mean(np.asarray(c)[basis[:, images]@np.asarray(codes)], axis=1)+.2*(
        np.mean(np.asarray(phi)[basis[:, images]@np.asarray(codes)], axis=1)-
        np.mean(np.asarray(ref)[basis[:, images]@np.asarray(codes)], axis=1))
    psi, hpsi = centered_products(carrier, hc, p, hp, r, hr, .2)
    np.testing.assert_allclose(psi, expected, atol=2e-14, rtol=0)
    np.testing.assert_allclose(hpsi, SzHeisenbergGraph(4, bonds, n_up=2).to_dense()@expected, atol=2e-14, rtol=0)
    assert p[0] == 0 and hp[0] != 0
    replay, replay_h = centered_products(carrier, hc, r, hr, r, hr, .2)
    np.testing.assert_array_equal(replay, carrier)
    np.testing.assert_array_equal(replay_h, hc)


def test_registered_optimization_has_independent_seeds_and_fixed_matched_budget():
    spec = json.loads(CONFIG.read_text()); validate_spec(spec)
    assert spec['steps']*spec['chains']*spec['records_per_step'] == 256
    assert spec['audit']['chains']*spec['audit']['records'] == 128
    assert spec['maximum_q'] == 4 and spec['multiplicity_width'] == 2
    bad = {**spec, 'seeds': {**spec['seeds'], 'audit': spec['seeds']['training']}}
    with pytest.raises(ValueError, match='distinct'): validate_spec(bad)
    with pytest.raises(ValueError, match='microbatch'): validate_spec({**spec, 'H_microbatch': 3})
    with pytest.raises(ValueError,match='initializer'):
        validate_spec({**spec,'free_initialization':'independent_uniform_layer_standardized'})
    validate_spec({**spec,'free_initialization':'independent_uniform_layer_standardized',
        'initialization_calibration_rows':1000,'seeds':{**spec['seeds'],'initialization':2026101499}})
