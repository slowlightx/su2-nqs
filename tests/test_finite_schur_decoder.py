"""Exact references for finite multiplet truncation and untied tensor control."""
from dataclasses import replace
import itertools

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin import _groups, _bonds
from projects.clebsch_tree_nqs.workflows.spin_decoder_cost import physical_config
from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.benchmarks.schur_tensor_decoder import four_cell_tensor_decoder_rows
from su2_nqs.models.finite_schur_decoder import (
    FiniteSchurDecoderConfig, reduced_entry_metadata, free_reduced_bank,
    init_finite_schur_network, neural_reduced_entries, neural_reduced_bank,
    make_finite_schur_amplitude, centered_additive_amplitudes,
    init_layer_standardized_free_tensors,
    shifted_neural_reduced_bank,
)
from su2_nqs.models.cell_multiplicity_clebsch_tree import cell_multiplicity_layouts
from su2_nqs.models.schur_cutset_compact import make_full_a1_batch
from su2_nqs.models.spatial_projector import PhysicalSpatialProjector
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.su2.local_schur_fusion import four_cell_schur_states
from su2_nqs.vmc.projected_spin_work import make_deduplicated_a1_heisenberg

jax.config.update('jax_enable_x64', True)


def config(rows=2, columns=4, chi=2):
    tree = replace(physical_config(rows, columns, chi), maximum_intermediate_q=4)
    return FiniteSchurDecoderConfig(tree, rows//2, columns//2, embedding_dim=8, depth=1)


def m0_states(n):
    return np.asarray([row for row in itertools.product((0, 1), repeat=n) if sum(row) == n//2], dtype=np.int32)


def test_n16_amplitudes_equal_independent_complete_schur_decoder_with_explicit_q_cutoff():
    cfg = config(4, 4); meta = reduced_entry_metadata(cfg)
    entries = np.random.default_rng(2341).normal(size=(meta['complex_entries'], 2))
    bank = free_reduced_bank({'entries': jnp.asarray(entries)}, cfg)
    layouts = cell_multiplicity_layouts(cfg.tree)
    arrays = jax.tree.map(np.asarray, bank)
    coefficients = []
    for state in four_cell_schur_states('square_plaquette'):
        if state.pair_q > 4:
            coefficients.append(0j); continue
        q, a, t = state.cell_q, state.cell_a, state.pair_q
        left = arrays['nodes'][0][layouts[0].keys.index((q[0], q[1], t)), :, a[0], a[1]]
        right = arrays['nodes'][1][layouts[1].keys.index((q[2], q[3], t)), :, a[2], a[3]]
        root = arrays['nodes'][2][layouts[2].keys.index((t, t, 0))]
        coefficients.append(np.einsum('l,r,klr->', left, right, root)/np.sqrt(cfg.tree.feature_dim))
    rng = np.random.default_rng(2342)
    states = np.stack([rng.permutation([0]*8+[1]*8) for _ in range(17)]).astype(np.int32)
    decoder = four_cell_tensor_decoder_rows(states, 'square_plaquette', cfg.tree.cells)
    expected = decoder@np.asarray(coefficients)
    actual = jax.jit(jax.vmap(make_finite_schur_amplitude(cfg), in_axes=(None, 0)))(bank, states)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 2e-13


def test_n8_all_amplitudes_and_full_neural_tangents_are_singlets_and_free_control_matches():
    cfg = config(); p = init_finite_schur_network(jax.random.PRNGKey(310), cfg)
    rows = jnp.asarray(m0_states(8)); raw = make_finite_schur_amplitude(cfg)
    fn = jax.jit(lambda x: jax.vmap(raw, in_axes=(None, 0))(neural_reduced_bank(x, cfg), rows))
    flat, unravel = ravel_pytree(p)
    direction = jnp.sin(jnp.arange(flat.size)+.3); direction /= jnp.linalg.norm(direction)
    v, tangent = jax.jvp(lambda x: fn(unravel(x)), (flat,), (direction,))
    assert np.linalg.norm(v) > 1e-4 and np.linalg.norm(tangent) > 1e-6
    assert spin_raising_relative_norm(np.asarray(v), np.asarray(rows)) < 2e-13
    assert spin_raising_relative_norm(np.asarray(tangent), np.asarray(rows)) < 2e-13
    epsilon = 1e-5
    np.testing.assert_allclose((fn(unravel(flat+epsilon*direction))-fn(unravel(flat-epsilon*direction)))/(2*epsilon),
                               tangent, atol=1e-9, rtol=2e-6)
    entries = neural_reduced_entries(p, cfg)
    free = {'entries': jnp.stack((entries.real, entries.imag), axis=1)}
    # Different JIT boundaries may fuse numeric operations differently.
    np.testing.assert_allclose(fn(p), jax.jit(jax.vmap(raw, in_axes=(None, 0)))(free_reduced_bank(free, cfg), rows),
                               atol=2e-16, rtol=2e-14)
    assert cfg.tree.feature_dim == 2
    # Same-size left and right nodes at N16 are independent in the strong control.
    assert len(reduced_entry_metadata(config(4, 4))['blocks']) == 3


def test_complete_a1_full_h_and_additive_reference_replay_including_gradients():
    cfg = config(); p = init_finite_schur_network(jax.random.PRNGKey(312), cfg)
    raw = make_finite_schur_amplitude(cfg)
    group = _groups('square', 2, 4)['gamma_a1']; projector = PhysicalSpatialProjector(group)
    projected = make_full_a1_batch(raw, projector, chunk_size=8)
    bank = neural_reduced_bank(p, cfg); states = m0_states(8)
    full = jax.jit(projected)(bank, jnp.asarray(states))
    assert spin_raising_relative_norm(np.asarray(full), states) < 2e-13
    all_images = states[:3, np.asarray(group)].reshape(-1, 8)
    images = np.asarray(jax.jit(projected)(bank, all_images)).reshape(3, len(group))
    np.testing.assert_allclose(images, np.asarray(full[:3])[:, None]*np.ones_like(images), atol=3e-15, rtol=2e-13)
    bonds = _bonds('square', 2, 4, .5)
    _, _, action = make_deduplicated_a1_heisenberg(raw, projector, bonds, chunk_size=16)
    ratios, psi, _, _ = jax.jit(action)(bank, states[:3])
    graph = SzHeisenbergGraph(8, tuple(bonds), n_up=4)
    for i in range(3):
        connected, elements = graph.get_conn(tuple(map(int, states[i])))
        values = jax.jit(projected)(bank, jnp.asarray(connected, dtype=jnp.int32))
        np.testing.assert_allclose(ratios[i]*psi[i], np.dot(elements, values), atol=3e-14, rtol=1e-12)
    # Add a independently certified singlet/A1 carrier without node lock-in.
    carrier = lambda bits: projected(bank, bits)
    additive = centered_additive_amplitudes(projected, carrier, bank, scale=.7)
    # Preserve the original exact physical replay gate. The current GPU
    # reduction discrepancy remains an unresolved failed gate, not something
    # to accept by changing the tolerance. The polynomial oracle below helps
    # isolate algebraic cancellation from contraction/compiler behavior.
    np.testing.assert_array_equal(additive(bank, states), full)
    delta = jax.tree.map(jnp.ones_like, bank)
    _, tangent = jax.jvp(lambda b: additive(b, states), (bank,), (delta,))
    assert np.linalg.norm(tangent) > 1e-6
    assert spin_raising_relative_norm(np.asarray(tangent), states) < 2e-13


def test_centered_additive_reduction_free_exact_anchor_and_nonzero_tangent():
    reference={'v':jnp.array([.5,-.25])}
    states=jnp.array([[0,1],[1,0]],dtype=jnp.int32)
    residual=lambda bank,x: bank['v'][0]*x[:,0]+bank['v'][1]*x[:,1]
    carrier=lambda x: 2.*x[:,0]-3.*x[:,1]
    additive=centered_additive_amplitudes(residual,carrier,reference,scale=.5)
    np.testing.assert_array_equal(additive(reference,states),carrier(states))
    np.testing.assert_array_equal(jax.jit(additive)(reference,states),carrier(states))
    direction={'v':jnp.array([1.,2.])}
    _,tangent=jax.jvp(lambda bank:additive(bank,states),(reference,),(direction,))
    np.testing.assert_array_equal(tangent,jnp.array([1.,.5]))


def test_identical_reduced_bank_projection_is_bitwise_repeatable():
    cfg=config();bank=neural_reduced_bank(init_finite_schur_network(jax.random.PRNGKey(917),cfg),cfg)
    projector=PhysicalSpatialProjector(_groups('square',2,4)['gamma_a1'])
    amplitude=jax.jit(make_full_a1_batch(make_finite_schur_amplitude(cfg),projector,chunk_size=8))
    bits=jnp.asarray(m0_states(8));expected=np.asarray(amplitude(bank,bits))
    for _ in range(6):
        repeated=amplitude(jax.tree.map(jnp.copy,bank),bits)
        np.testing.assert_array_equal(repeated,expected)


def test_metadata_invalid_contracts_and_runtime_validation():
    cfg = config()
    with pytest.raises(ValueError):
        FiniteSchurDecoderConfig(replace(cfg.tree, maximum_intermediate_q=2), 1, 2)
    with pytest.raises(ValueError):
        replace(cfg, cell_columns=3)
    with pytest.raises(ValueError):
        free_reduced_bank({'entries': jnp.zeros((1, 2))}, cfg)
    raw = make_finite_schur_amplitude(cfg)
    bank = neural_reduced_bank(init_finite_schur_network(jax.random.PRNGKey(1), cfg), cfg)
    with pytest.raises(TypeError):
        raw(bank, jnp.zeros(8))
    assert raw(bank, jnp.asarray([0, 1, 0, 1, 0, 1, 0, 2])) == 0j
    assert raw(bank, jnp.zeros(8, dtype=jnp.int32)) == 0j


def test_independent_uniform_layer_calibration_preserves_the_ray_and_singlet():
    cfg = config(4,4)
    bits=np.stack([np.random.default_rng(370+i).permutation([0]*8+[1]*8)
                   for i in range(32)]).astype(np.int32)
    key=jax.random.PRNGKey(370)
    free,report=init_layer_standardized_free_tensors(key,cfg,bits)
    original={'entries':jax.random.uniform(key,free['entries'].shape,
        dtype=jnp.float64,minval=-1.,maxval=1.)}
    raw=jax.jit(jax.vmap(make_finite_schur_amplitude(cfg),in_axes=(None,0)))
    old=np.asarray(raw(free_reduced_bank(original,cfg),bits))
    new=np.asarray(raw(free_reduced_bank(free,cfg),bits))
    np.testing.assert_allclose(new,old*report['raw_global_amplitude_scale'],atol=2e-13,rtol=3e-13)
    np.testing.assert_allclose(report['activation_std_after'],1.,atol=3e-13,rtol=0)
    assert report['physical_ray_unchanged_by_scaling']
    assert not report['magnetic_index_dependent_weights']
    cfg8=config()
    rows=m0_states(8)
    small,_=init_layer_standardized_free_tensors(key,cfg8,rows)
    values=jax.jit(jax.vmap(make_finite_schur_amplitude(cfg8),in_axes=(None,0)))(free_reduced_bank(small,cfg8),rows)
    assert spin_raising_relative_norm(np.asarray(values),rows)<3e-13
    with pytest.raises(ValueError,match='calibration'):
        init_layer_standardized_free_tensors(key,cfg,bits[:1])


def test_rich_tensor_offset_preserves_full_neural_gradients_and_exact_spin():
    cfg=config(); bits=m0_states(8)
    p=init_finite_schur_network(jax.random.PRNGKey(390),cfg)
    reference=neural_reduced_entries(p,cfg)
    free,_=init_layer_standardized_free_tensors(jax.random.PRNGKey(391),cfg,bits)
    offset=free['entries'][:,0]+1j*free['entries'][:,1]
    make_bank=lambda params:shifted_neural_reduced_bank(params,cfg,
        offset_entries=offset,reference_entries=reference)
    raw=jax.vmap(make_finite_schur_amplitude(cfg),in_axes=(None,0))
    baseline=np.asarray(raw(free_reduced_bank(free,cfg),bits))
    actual=np.asarray(jax.jit(lambda params:raw(make_bank(params),bits))(p))
    np.testing.assert_allclose(actual,baseline,atol=3e-13,rtol=3e-13)
    flat,unravel=ravel_pytree(p); direction=jnp.sin(jnp.arange(flat.size)); direction/=jnp.linalg.norm(direction)
    fn=jax.jit(lambda z:raw(make_bank(unravel(z)),bits))
    _,tangent=jax.jvp(fn,(flat,),(direction,))
    assert np.linalg.norm(tangent)>1e-6
    assert spin_raising_relative_norm(np.asarray(tangent),bits)<3e-13
    epsilon=1e-5
    np.testing.assert_allclose((fn(flat+epsilon*direction)-fn(flat-epsilon*direction))/(2*epsilon),
                               tangent,atol=2e-9,rtol=2e-6)
