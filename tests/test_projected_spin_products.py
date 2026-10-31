"""Residual nodes are legal in a centered additive state: Hphi stays finite."""
import itertools
from dataclasses import replace
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.projected_spin_products import make_projected_heisenberg_products
from projects.clebsch_tree_nqs.workflows.spin import _groups, _bonds
from projects.clebsch_tree_nqs.workflows.spin_decoder_cost import physical_config
from su2_nqs.frontier.parameter_artifact import load_parameter_artifact
from su2_nqs.models.finite_schur_decoder import (
    FiniteSchurDecoderConfig, init_finite_schur_network, neural_reduced_bank,
    make_finite_schur_amplitude,
)

jax.config.update('jax_enable_x64', True)


def test_exact_residual_node_has_nonzero_finite_H_product_and_correct_additive_ratio():
    # Explicit non-group-invariant H: no illicit H/projector commutation used.
    projector = PhysicalSpatialProjector((tuple(range(4)), (3, 2, 1, 0)))
    bonds = ((0, 1, 1.), (1, 2, .3), (0, 3, -.4), (1, 0, .2))
    codes = jnp.asarray([8, 4, 2, 1])
    raw = lambda p, x: p[jnp.dot(x, codes)]
    plan, evaluate, action = make_projected_heisenberg_products(raw, projector, bonds, chunk_size=3)
    basis = np.asarray([x for x in itertools.product((0, 1), repeat=4) if sum(x) == 2], dtype=np.int32)
    phi = jnp.arange(16, dtype=jnp.float64)+.03j*jnp.arange(16)**2
    # 0011 and its reversed image 1100 are both nodes, but Hphi is NOT zero.
    phi = phi.at[3].set(0j).at[12].set(0j)
    prepared = jax.jit(plan)(basis)
    actual, psi, _, _ = jax.jit(evaluate)(phi, prepared)
    images = np.asarray(projector.site_permutations)
    v = np.asarray(phi)[np.sum(basis[:, images]*np.asarray(codes), axis=-1)].mean(axis=1)
    graph = SzHeisenbergGraph(4, bonds, n_up=2)
    expected = graph.to_dense()@v
    np.testing.assert_allclose(psi, v, atol=1e-14, rtol=0)
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
    assert psi[0] == 0j
    # Direct hand calculation, including the opposite-sign exchange bonds.
    np.testing.assert_allclose(actual[0], -.375-.09375j, atol=1e-14, rtol=0)
    assert np.isfinite(actual).all()
    c = jnp.ones(16, dtype=jnp.complex128)
    hc, carrier, _, _ = jax.jit(action)(c, basis)
    scale = .23
    combined_ratio = (hc+scale*actual)/(carrier+scale*psi)
    combined = np.ones(6)+scale*v
    np.testing.assert_allclose(combined_ratio, graph.to_dense()@combined/combined, atol=2e-14, rtol=0)
    # Identically zero reference bank is also valid, with zero products.
    hzero, zero, _, _ = jax.jit(action)(jnp.zeros(16, dtype=jnp.complex128), basis)
    np.testing.assert_array_equal(hzero, 0j); np.testing.assert_array_equal(zero, 0j)


@pytest.mark.parametrize('geometry,ratio', (('triangular', .125), ('square', .5)))
def test_N36_division_free_products_replay_sealed_full_H_and_zero_reference(geometry, ratio):
    root = Path(__file__).resolve().parents[1]
    folder = root/'projects/clebsch_tree_nqs/results/spin_finite_schur_cost_v1'/geometry
    cfg = FiniteSchurDecoderConfig(replace(physical_config(6, 6, 2), maximum_intermediate_q=4), 3, 3, 16, 2)
    template = init_finite_schur_network(jax.random.PRNGKey(0), cfg)
    params = load_parameter_artifact(folder/'network.json', template)
    bank = neural_reduced_bank(params, cfg)
    raw = make_finite_schur_amplitude(cfg)
    projector = PhysicalSpatialProjector(_groups(geometry, 6, 6)['gamma_a1'])
    plan, evaluate, _ = make_projected_heisenberg_products(raw, projector, _bonds(geometry, 6, 6, ratio))
    with np.load(folder/'values.npz', allow_pickle=False) as archive:
        prepared = jax.jit(plan)(jnp.asarray(archive['states']))
        products, psi, _, _ = jax.jit(evaluate)(bank, prepared)
        expected = archive['network_full_H_b4']*archive['network_A1_b4']
        np.testing.assert_allclose(products, expected, atol=2e-17, rtol=2e-12)
        np.testing.assert_allclose(psi, archive['network_A1_b4'], atol=1e-18, rtol=2e-12)
    empty = jax.tree.map(jnp.zeros_like, bank)
    zero_h, zero, _, _ = jax.jit(evaluate)(empty, prepared)
    np.testing.assert_array_equal(zero_h, 0j); np.testing.assert_array_equal(zero, 0j)
