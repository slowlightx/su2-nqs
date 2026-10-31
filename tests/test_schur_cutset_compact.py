"""Equivalent compact contraction: no spin cutoff or spatial-image removal."""
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin import _groups
from projects.clebsch_tree_nqs.workflows.spin_decoder_cost import ROOT, physical_config
from projects.clebsch_tree_nqs.workflows.spin_vap_decoder import init_decoder
from su2_nqs.models.cell_multiplicity_clebsch_tree import cell_multiplicity_layouts
from su2_nqs.models.schur_cutset_attention import (
    CutsetAttentionConfig, cutset_raw_amplitude, init_cutset_attention, make_branch_bank,
)
from su2_nqs.models.schur_cutset_compact import (
    compact_cg_storage, make_compact_cutset_amplitude, make_full_a1_batch, packed_cg_table,
)
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.vmc.projected_spin_work import make_chunked_a1_amplitudes

jax.config.update('jax_enable_x64', True)


def _config(rows=2, columns=4):
    tree = physical_config(rows, columns, 2)
    coords = tuple((float(np.mean(np.asarray(c) % columns)),
                    float(np.mean(np.asarray(c) // columns))) for c in tree.cells)
    return CutsetAttentionConfig(tree, coords)


def _bank(config):
    return make_branch_bank(init_cutset_attention(
        jax.random.PRNGKey(781), init_decoder(780, config.tree), config), config)


def test_n8_exhaustive_amplitudes_and_jvp_equal_legacy_contraction():
    config = _config(); bank = _bank(config)
    states = jnp.asarray((np.arange(256)[:, None] >> np.arange(7, -1, -1)) & 1, dtype=jnp.int32)
    compact = jax.jit(jax.vmap(make_compact_cutset_amplitude(config), in_axes=(None, 0)))
    legacy = jax.jit(jax.vmap(lambda b, x: cutset_raw_amplitude(b, x, config), in_axes=(None, 0)))
    actual = np.asarray(compact(bank, states)); expected = np.asarray(legacy(bank, states))
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=5e-16)
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 3e-13
    np.testing.assert_array_equal(actual[np.asarray(states).sum(axis=1) != 4], 0)
    flat, unravel = ravel_pytree(bank)
    direction = jnp.sin(jnp.arange(flat.size)+.913); direction /= jnp.linalg.norm(direction)
    old_jvp = jax.jvp(lambda p: legacy(unravel(p), states), (flat,), (direction,))[1]
    new_jvp = jax.jvp(lambda p: compact(unravel(p), states), (flat,), (direction,))[1]
    np.testing.assert_allclose(new_jvp, old_jvp, rtol=5e-12, atol=5e-16)
    assert np.linalg.norm(np.asarray(new_jvp-old_jvp))/np.linalg.norm(old_jvp) < 3e-12


def test_n8_full_projection_amplitudes_and_reverse_gradient_equal_legacy():
    config = _config(); bank = _bank(config)
    all_states = (np.arange(256)[:, None] >> np.arange(7, -1, -1)) & 1
    states = jnp.asarray(all_states[all_states.sum(axis=1) == 4], dtype=jnp.int32)
    projector = PhysicalSpatialProjector(_groups('square', 2, 4)['gamma_a1'])
    old = make_chunked_a1_amplitudes(lambda b, x: cutset_raw_amplitude(b, x, config), projector, chunk_size=7)
    new = make_full_a1_batch(make_compact_cutset_amplitude(config), projector, chunk_size=7)
    expected = jax.jit(old)(bank, states); actual = jax.jit(new)(bank, states)
    np.testing.assert_allclose(actual, expected, atol=5e-16, rtol=5e-12)
    flat, unravel = ravel_pytree(bank)
    old_gradient = jax.jit(jax.grad(lambda p: jnp.sum(abs(old(unravel(p), states))**2)))(flat)
    new_gradient = jax.jit(jax.grad(lambda p: jnp.sum(abs(new(unravel(p), states))**2)))(flat)
    np.testing.assert_allclose(new_gradient, old_gradient, rtol=5e-11, atol=5e-18)
    assert np.linalg.norm(np.asarray(new_gradient-old_gradient))/np.linalg.norm(old_gradient) < 3e-11


def test_n36_two_frozen_raw_probes_equal_legacy_without_state_change():
    config = _config(6, 6); bank = _bank(config)
    with np.load(ROOT/'projects/clebsch_tree_nqs/results/spin_decoder_physical_init_v1/triangular/values.npz') as archive:
        states = jnp.asarray(archive['states'][:2], dtype=jnp.int32)
    compact = jax.jit(jax.vmap(make_compact_cutset_amplitude(config), in_axes=(None, 0)))
    legacy = jax.jit(jax.vmap(lambda b, x: cutset_raw_amplitude(b, x, config), in_axes=(None, 0)))
    expected = np.asarray(legacy(bank, states)); actual = np.asarray(compact(bank, states))
    np.testing.assert_allclose(actual, expected, atol=1e-20, rtol=5e-11)
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 5e-11


def test_n64_root_cg_closed_formula_and_compact_storage_without_legacy_table(monkeypatch):
    config = _config(8, 8)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('legacy dense CG allocation is forbidden in compact evaluator')
    monkeypatch.setattr('su2_nqs.models.schur_cutset_attention.spin_cg_tensor', forbidden)
    function = make_compact_cutset_amplitude(config)
    assert callable(function)
    report = compact_cg_storage(config)
    assert not report['allowed_channels_truncated']
    assert report['packed_bytes'] == 5043600
    assert report['legacy_dense_bytes'] == 2478948768
    root = cell_multiplicity_layouts(config.tree)[-1]
    assert (root.left_size, root.right_size) == (32, 32)
    assert root.keys == tuple((q, q, 0) for q in range(0, 33, 2))
    table = packed_cg_table(root.left_size, root.right_size, root.keys)
    assert table.shape == (17, 65, 65) and table.nbytes == 574600
    expected = np.zeros_like(table)
    for index, (q, _, _) in enumerate(root.keys):
        for m in range(-q, q+1, 2):
            expected[index, m+32, -m+32] = (-1.)**((q-m)//2)/np.sqrt(q+1)
    np.testing.assert_allclose(table, expected, atol=3e-15, rtol=3e-14)
    assert not table.flags.writeable
    # Another reachable, non-root coefficient is independent of singlet formula.
    middle = next(row for row in cell_multiplicity_layouts(config.tree)
                  if row.left_size == row.right_size == 16)
    coefficients = packed_cg_table(16, 16, middle.keys)
    channel = middle.keys.index((16, 16, 16))
    assert coefficients[channel, 16, 16] == pytest.approx(clebsch_gordan(16, 0, 16, 0, 16, 0), abs=1e-15)


def test_full_n64_a1_scan_has_no_uint64_guard_and_keeps_every_image():
    group = _groups('triangular', 8, 8)['gamma_a1']
    projector = PhysicalSpatialProjector(group)
    assert projector.group_order == 768
    states = jnp.asarray([([1]*32+[0]*32), ([1, 0]*32)], dtype=jnp.int32)
    raw = lambda p, x: p['scale']*(jnp.dot(jnp.arange(64), x)+.1j*jnp.dot(jnp.arange(64)**2, x))
    params = {'scale': jnp.asarray(.3)}
    fn = make_full_a1_batch(raw, projector, chunk_size=17)
    actual = jax.jit(fn)(params, states)
    expected = jax.vmap(lambda x: jax.vmap(lambda image: raw(params, x[image]))(
        jnp.asarray(group)).mean())(states)
    np.testing.assert_allclose(actual, expected, atol=5e-12, rtol=2e-14)
    gradient = jax.grad(lambda s: jnp.real(fn({'scale': s}, states)).sum())(.3)
    assert float(gradient) == pytest.approx(float(jnp.real(expected).sum()/.3), rel=3e-14)


def test_packed_lookup_and_key_free_wrapper_reject_invalid_metadata():
    with pytest.raises(ValueError):
        packed_cg_table(4, 4, ((0, 0, 2),))
    with pytest.raises(ValueError):
        packed_cg_table(4, 4, ((1, 1, 0),))
    with pytest.raises(ValueError):
        packed_cg_table(4, 4, ((0, 0, 0), (0, 0, 0)))
    with pytest.raises(TypeError):
        packed_cg_table(4, 4, ((0., 0, 0),))
    projector = PhysicalSpatialProjector((tuple(range(64)),))
    with pytest.raises(ValueError):
        make_full_a1_batch(lambda p, x: 0j, projector, chunk_size=0)
    fn = make_full_a1_batch(lambda p, x: 0j, projector)
    with pytest.raises(ValueError):
        fn({}, jnp.zeros((1, 63)))
