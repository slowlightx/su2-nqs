import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from su2_nqs.benchmarks.clebsch_tree_exact import fixed_m_spin_basis
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix
from su2_nqs.lattices.geometry_tree import square_geometry_tree
from su2_nqs.lattices.j1j2 import square_j1j2_bonds
from su2_nqs.models.clebsch_tree_live_expansion import initialize_live_clebsch_expansion
from su2_nqs.models.clebsch_tree_projected import make_projected_clebsch
from su2_nqs.models.clebsch_tree_transfer import transfer_clebsch_parameters
from su2_nqs.models.literature_clebsch_tree import (
    LiteratureClebschTreeConfig, init_literature_clebsch_tree,
    make_clebsch_amplitudes, make_clebsch_features,
)
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from projects.clebsch_tree_nqs.workflows.spin import _groups

jax.config.update("jax_enable_x64", True)


def _models():
    tree, order = square_geometry_tree(2, 4)
    narrow = LiteratureClebschTreeConfig(tree, multiplicity_cap=1, maximum_q=4)
    wide = LiteratureClebschTreeConfig(tree, multiplicity_cap=4, maximum_q=4)
    parent, _ = init_literature_clebsch_tree(jax.random.PRNGKey(12), narrow)
    live, report = initialize_live_clebsch_expansion(
        narrow, parent, wide, jax.random.PRNGKey(13))
    return narrow, wide, order, parent, live, report


def test_every_physical_amplitude_old_feature_and_inherited_jvp_is_preserved():
    narrow, wide, _, parent, live, report = _models()
    x = jnp.asarray(fixed_m_spin_basis(8, 0))
    old, new = make_clebsch_amplitudes(narrow), make_clebsch_amplitudes(wide)
    np.testing.assert_allclose(old(parent, x), new(live, x), atol=2e-15, rtol=2e-13)
    a, b = make_clebsch_features(narrow)(parent, x), make_clebsch_features(wide)(live, x)
    for first, second in zip(a, b, strict=True):
        np.testing.assert_allclose(first, second[..., :1], atol=2e-15, rtol=2e-13)
    direction = jax.tree.map(jnp.ones_like, parent)
    target_zero = jax.tree.map(jnp.zeros_like, live)
    inherited, _ = transfer_clebsch_parameters(narrow, direction, wide, target_zero, mode="embedding")
    aa = jax.jvp(lambda p: old(p, x), (parent,), (direction,))[1]
    bb = jax.jvp(lambda p: new(p, x), (live,), (inherited,))[1]
    np.testing.assert_allclose(aa, bb, atol=2e-14, rtol=2e-13)
    # Differentiate all inherited coordinates through the explicit lift,
    # rather than checking only one hand-picked update vector.
    old_grad = jax.grad(lambda p: jnp.sum(old(p, x) ** 2))(parent)
    inherited_grad = jax.grad(lambda p: jnp.sum(new(
        initialize_live_clebsch_expansion(narrow, p, wide, jax.random.PRNGKey(13))[0], x) ** 2))(parent)
    for aa, bb in zip(jax.tree.leaves(old_grad), jax.tree.leaves(inherited_grad), strict=True):
        np.testing.assert_allclose(aa, bb, atol=2e-14, rtol=2e-13)
    assert report["target_parameters"] > report["source_parameters"]
    assert not report["layer_rescaling_performed"]


def test_new_root_readout_is_first_order_live_but_zero_padding_is_dead():
    narrow, wide, _, parent, live, _ = _models()
    x = jnp.asarray(fixed_m_spin_basis(8, 0))
    evaluate = make_clebsch_amplitudes(wide)
    empty = jax.tree.map(jnp.zeros_like, live)
    padded, _ = transfer_clebsch_parameters(narrow, parent, wide, empty, mode="embedding")
    direction = dict(merges=empty["merges"], readout=jnp.zeros_like(live["readout"]).at[-1].set(1.))
    new = jax.jvp(lambda p: evaluate(p, x), (live,), (direction,))[1]
    dead = jax.jvp(lambda p: evaluate(p, x), (padded,), (direction,))[1]
    assert float(jnp.linalg.norm(new)) > 1e-8
    np.testing.assert_array_equal(dead, np.zeros(len(x)))
    # The live output is not just the same state with an overall scale.
    psi = evaluate(live, x)
    residual = new - psi * jnp.vdot(psi, new) / jnp.vdot(psi, psi)
    assert float(jnp.linalg.norm(residual)) > 1e-8


def test_live_features_open_tangent_directions_not_reached_by_zero_padding():
    narrow, wide, order, parent, live, _ = _models()
    empty = jax.tree.map(jnp.zeros_like, live)
    padded, _ = transfer_clebsch_parameters(narrow, parent, wide, empty, mode="embedding")
    x = jnp.asarray(fixed_m_spin_basis(8, 0))
    ranks = []
    for config, params in ((narrow, parent), (wide, padded), (wide, live)):
        flat, unravel = ravel_pytree(params)
        evaluate = make_clebsch_amplitudes(config)
        jacobian = np.asarray(jax.jacfwd(lambda p: evaluate(unravel(p), x))(flat))
        values = np.linalg.svd(jacobian, compute_uv=False)
        ranks.append(int(np.count_nonzero(values > values[0] * 1e-8)))
    assert ranks[0] == ranks[1] == 6
    assert ranks[2] > ranks[1]
    # Important negative control: this small full-A1 sector is already
    # saturated by the narrow model. Raw-basis rank growth is NOT a gain
    # of projected-state capacity or evidence of a variational improvement.
    projected_ranks = []
    group = _groups("square", 2, 4)["gamma_a1"]
    for config, params in ((narrow, parent), (wide, padded), (wide, live)):
        flat, unravel = ravel_pytree(params)
        evaluate, _ = make_projected_clebsch(config, order, group, orbit_chunk=2)
        jacobian = np.asarray(jax.jit(jax.jacfwd(lambda p: evaluate(unravel(p), x)))(flat))
        values = np.linalg.svd(jacobian, compute_uv=False)
        projected_ranks.append(int(np.count_nonzero(values > values[0] * 1e-8)))
    assert projected_ranks == [5, 5, 5]


def test_singlet_full_A1_and_complete_H_are_preserved():
    narrow, wide, order, parent, live, _ = _models()
    states = np.asarray(fixed_m_spin_basis(8, 0))
    group = _groups("square", 2, 4)["gamma_a1"]
    old, _ = make_projected_clebsch(narrow, order, group, orbit_chunk=2)
    new, _ = make_projected_clebsch(wide, order, group, orbit_chunk=2)
    first, second = np.asarray(old(parent, states)), np.asarray(new(live, states))
    np.testing.assert_allclose(first, second, atol=2e-15, rtol=2e-13)
    np.testing.assert_allclose(new(live, states[:, group[3]]), second, atol=2e-15, rtol=2e-13)
    s2, integers = spin_squared_matrix(8, n_up=4)
    lookup = {v: i for i, v in enumerate(integers)}
    indices = [lookup[int("".join(map(str, row)), 2)] for row in states]
    assert np.linalg.norm(s2[np.ix_(indices, indices)] @ second) / np.linalg.norm(second) < 2e-12
    bonds = square_j1j2_bonds(2, 4, j2=.5, periodic=True)
    f = make_amplitude_heisenberg_action(lambda p, x: old(p, x[None])[0], bonds, n_sites=8)
    g = make_amplitude_heisenberg_action(lambda p, x: new(p, x[None])[0], bonds, n_sites=8)
    for a, b in zip(f(parent, states[:2]), g(live, states[:2]), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-14, rtol=2e-12)


def test_input_contract_disallows_cutoff_tree_target_truncation_and_invalid_scale():
    narrow, wide, _, parent, _, _ = _models()
    key = jax.random.PRNGKey(15)
    for changed in (LiteratureClebschTreeConfig(wide.tree, maximum_q=6),
                    LiteratureClebschTreeConfig(wide.tree, target_q=2, maximum_q=4)):
        with pytest.raises(ValueError, match="same tree"):
            initialize_live_clebsch_expansion(narrow, parent, changed, key)
    for scale in (0., -1., float("nan")):
        with pytest.raises(ValueError, match="positive"):
            initialize_live_clebsch_expansion(narrow, parent, wide, key, new_feature_scale=scale)
    with pytest.raises(ValueError, match="cannot remove"):
        initialize_live_clebsch_expansion(wide, init_literature_clebsch_tree(key, wide)[0], narrow, key)
