"""Exact checks for the unbiased image-subsampled coherent-A1 action."""
from __future__ import annotations

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, init_mixed_clebsch_family
from su2_nqs.vmc.image_subsampled_local_energy import (
    conditional_estimator_variance, make_image_resolved_heisenberg,
    make_subsampled_projected_action,
)
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action

CONFIG = MixedClebschConfig(length=6, multiplicity_cap=9, maximum_q=6)


@pytest.fixture(scope='module')
def system():
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    bonds = tuple(triangular_j1j2_bonds(6, 6, j2=.125, periodic=True))
    bank = init_mixed_clebsch_family(jax.random.PRNGKey(7), CONFIG)['ternary_free']
    prepared = prepare_feature_lut_bank(bank, CONFIG)
    rng = np.random.default_rng(11)
    rows = np.stack([rng.permutation(np.repeat([0, 1], 18)) for _ in range(3)]).astype(np.int32)
    exact = make_cached_projected_heisenberg_action(CONFIG, images, bonds)
    psi, hpsi = (np.asarray(v) for v in exact(prepared, jnp.asarray(rows)))
    return dict(images=images, bonds=bonds, prepared=prepared, rows=rows, psi=psi, hpsi=hpsi)


def _full_image_data(system, row):
    parts = make_image_resolved_heisenberg(CONFIG, system['images'], system['bonds'])
    indices = jnp.arange(len(system['images']))
    raw = np.asarray(parts['raw_amplitudes'](system['prepared'], jnp.asarray(row)))
    psi_g, hpsi_g = (np.asarray(v) for v in parts['image_actions'](system['prepared'], jnp.asarray(row), indices))
    return parts, raw, psi_g, hpsi_g


def test_image_resolved_data_reproduce_exact_action(system):
    parts, raw, psi_g, hpsi_g = _full_image_data(system, system['rows'][0])
    np.testing.assert_allclose(raw, psi_g, rtol=0, atol=1e-15)
    scale = np.max(np.abs(system['hpsi'][0]))
    assert abs(raw.mean()-system['psi'][0]) <= 1e-13*np.max(np.abs(raw))
    assert abs(hpsi_g.mean()-system['hpsi'][0]) <= 1e-12*scale
    # Projected per-bond off-diagonal terms plus the invariant diagonal.
    bits = system['rows'][0]
    terms = np.asarray(parts['bond_terms'](system['prepared'], jnp.asarray(bits)))
    diag = sum(J*(.25 if bits[a] == bits[b] else -.25) for a, b, J in system['bonds'])
    assert abs(diag*raw.mean()+terms.sum()-system['hpsi'][0]) <= 1e-12*scale


@pytest.mark.parametrize('beta', [0., -20.])
def test_complete_uniform_selection_is_exact(system, beta):
    count = len(system['images'])
    action = make_subsampled_projected_action(CONFIG, system['images'], system['bonds'],
                                              image_samples=count, scheme='uniform')
    keys = jax.random.split(jax.random.PRNGKey(3), len(system['rows']))
    psi, hpsi = (np.asarray(v) for v in action(system['prepared'], jnp.asarray(system['rows']), keys, beta))
    reference = make_feature_lut_projected_amplitude(CONFIG, system['images'])
    for row, value in zip(system['rows'], psi, strict=True):
        assert abs(value-float(reference(system['prepared'], jnp.asarray(row)))) <= 1e-13*abs(value)+1e-300
    np.testing.assert_allclose(psi, system['psi'], rtol=1e-12, atol=0)
    np.testing.assert_allclose(hpsi, system['hpsi'], rtol=1e-11, atol=0)


@pytest.mark.parametrize('scheme,samples', [('uniform', 8), ('amplitude', 4)])
def test_monte_carlo_unbiased_with_predicted_variance(system, scheme, samples):
    row = system['rows'][1]
    _, raw, _, hpsi_g = _full_image_data(system, row)
    beta = -18.
    predicted = conditional_estimator_variance(raw, hpsi_g, beta, samples, scheme)
    draws = 3000
    action = make_subsampled_projected_action(CONFIG, system['images'], system['bonds'],
                                              image_samples=samples, scheme=scheme)
    keys = jax.random.split(jax.random.PRNGKey(29), draws)
    batch = 100
    rows = jnp.asarray(np.repeat(row[None], batch, axis=0))
    pieces = [action(system['prepared'], rows, keys[i:i+batch], beta) for i in range(0, draws, batch)]
    psi = np.concatenate([np.asarray(p) for p, _ in pieces])
    hpsi = np.concatenate([np.asarray(h) for _, h in pieces])
    assert np.all(psi == psi[0])
    exact = system['hpsi'][1]
    assert abs(hpsi.mean()-exact) <= 5*np.sqrt(predicted/draws)
    ratio = hpsi.var(ddof=1)/predicted
    assert .85 < ratio < 1.18, ratio


def test_invalid_arguments_rejected(system):
    with pytest.raises(ValueError):
        make_subsampled_projected_action(CONFIG, system['images'], system['bonds'], image_samples=0)
    with pytest.raises(ValueError):
        make_subsampled_projected_action(CONFIG, system['images'], system['bonds'],
                                         image_samples=len(system['images'])+1, scheme='uniform')
    with pytest.raises(ValueError):
        make_subsampled_projected_action(CONFIG, system['images'], system['bonds'],
                                         image_samples=2, defensive=0.)
