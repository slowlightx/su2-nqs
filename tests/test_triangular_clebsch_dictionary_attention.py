"""Static neural tensor generation, no change to CG or coherent projection."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.triangular_clebsch_dictionary_attention import (
    TriangleDictionaryAttentionConfig, dictionary_layout, dictionary_geometry,
    dictionary_attention_bank, dictionary_parameter_count, init_dictionary_attention_from_bank,
)
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, init_mixed_clebsch_family, mixed_clebsch_bank,
    make_mixed_clebsch_amplitude, make_mixed_projected_amplitude,
)
from su2_nqs.models.triangular_clebsch_feature_lut import (
    prepare_feature_lut_bank, make_feature_lut_projected_amplitude,
)
from su2_nqs.su2.triangle_blocks import triangle_block_tables, nine_site_output, real_multiplicity_frame
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action

jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='module')
def parent():
    cfg = MixedClebschConfig(multiplicity_cap=9, maximum_q=3)
    family = init_mixed_clebsch_family(jax.random.PRNGKey(17901), cfg)
    return cfg, mixed_clebsch_bank(family['ternary_c3'], cfg, 'ternary_c3')


@pytest.mark.parametrize('architecture', ['attention', 'linear'])
@pytest.mark.parametrize('symmetry', ['free_joint', 'C3'])
def test_exact_parent_replay_has_no_hidden_full_tensor_residual(parent, architecture, symmetry):
    model, bank = parent
    cfg = TriangleDictionaryAttentionConfig(model=model, embedding_dim=12,
        architecture=architecture, local_symmetry=symmetry)
    params, note = init_dictionary_attention_from_bank(jax.random.PRNGKey(17902), bank, cfg)
    generated = jax.jit(lambda p: dictionary_attention_bank(p, cfg))(params)
    assert set(params) == {'encoder', 'dictionaries', 'coefficient_heads'}
    assert note['exact_same_bank_expected'] and not note['virtual_leg_rank_factorization']
    assert not note['full_per_node_residual']
    for actual, expected in zip(jax.tree.leaves(generated), jax.tree.leaves(bank), strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=2e-11)
    assert dictionary_parameter_count(params) == sum(w.size for w in jax.tree.leaves(params))
    assert all(b.shape == (item['rank'], item['vector_size']) for b, item in
        zip(params['dictionaries'], dictionary_layout(cfg), strict=True))
    tokens, distances = dictionary_geometry(cfg)
    assert tokens.shape[0] == 7 and distances.shape == (7, 7)
    np.testing.assert_allclose(distances, distances.T, atol=2e-15, rtol=0)


def test_dictionary_rank_is_across_nodes_not_a_hidden_virtual_leg_rank(parent):
    model, bank = parent
    cfg = TriangleDictionaryAttentionConfig(model=model, dictionary_rank=1, architecture='linear')
    rng = np.random.default_rng(17903)
    free = dict(bank, local=tuple(tuple(jnp.asarray(rng.normal(size=w.shape)) for w in node)
                                 for node in bank['local']))
    params, note = init_dictionary_attention_from_bank(jax.random.PRNGKey(17904), free, cfg)
    generated = dictionary_attention_bank(params, cfg)
    assert max(note['per_type_node_sharing_approximation_errors']) > .01
    # R=1 does NOT force any reduced virtual matrix to rank1.
    assert np.linalg.matrix_rank(np.asarray(generated['local'][0][0]), tol=1e-11) == 9


def test_C3_survives_arbitrary_scalar_coefficients_and_nonC3_parent_is_rejected(parent):
    model, bank = parent
    cfg = TriangleDictionaryAttentionConfig(model=model, local_symmetry='C3')
    params, _ = init_dictionary_attention_from_bank(jax.random.PRNGKey(17905), bank, cfg)
    perturbed = jax.tree.map(lambda w: w+.02*jnp.sin(jnp.arange(w.size).reshape(w.shape)), params)
    generated = dictionary_attention_bank(perturbed, cfg)
    data = triangle_block_tables()
    assert data['three'][1].shape[1] == 2
    for node in generated['local']:
        for q, weights in zip(model.local_q, node, strict=True):
            representation = nine_site_output(q, model.multiplicity_cap)
            real = real_multiplicity_frame(representation.time_reversal_multiplicity())
            output = (real.conj().T @ representation.rotation() @ real).real
            np.testing.assert_allclose(data['rotations'][q] @ weights, weights @ output,
                atol=3e-13, rtol=2e-11)
    first = list(bank['local'][0]); first[0] = first[0].at[0, 0].add(.1)
    invalid = dict(bank, local=(tuple(first), *bank['local'][1:]))
    with pytest.raises(ValueError, match='not in the requested local C3'):
        init_dictionary_attention_from_bank(jax.random.PRNGKey(17905), invalid, cfg)


def test_N36_full_A1_H_JVP_gradient_and_Splus_Ward(parent):
    model, bank = parent
    cfg = TriangleDictionaryAttentionConfig(model=model, embedding_dim=12)
    params, _ = init_dictionary_attention_from_bank(jax.random.PRNGKey(17906), bank, cfg)
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    old = make_mixed_projected_amplitude(model, images, chunk=4)
    cached = make_feature_lut_projected_amplitude(model, images, chunk=4)
    generated = dictionary_attention_bank(params, cfg)
    rng = np.random.default_rng(17907)
    rows = jnp.asarray([rng.permutation([0]*18+[1]*18) for _ in range(3)], jnp.int32)
    evaluate = jax.jit(lambda p, x: jax.vmap(cached, in_axes=(None, 0))(
        prepare_feature_lut_bank(dictionary_attention_bank(p, cfg), model), x))
    np.testing.assert_allclose(evaluate(params, rows), jax.jit(jax.vmap(old, in_axes=(None, 0)))(bank, rows),
        atol=2e-16, rtol=2e-11)
    np.testing.assert_allclose(evaluate(params, rows[:1, images[19]]), evaluate(params, rows[:1]),
        atol=2e-16, rtol=2e-11)
    bonds = tuple(triangular_j1j2_bonds(6, 6, j2=.125, periodic=True))
    actual = make_amplitude_heisenberg_action(cached, bonds, n_sites=36)(prepare_feature_lut_bank(generated, model), rows[:1])
    expected = make_amplitude_heisenberg_action(old, bonds, n_sites=36)(bank, rows[:1])
    np.testing.assert_allclose(actual, expected, atol=2e-16, rtol=2e-11)
    baseline = lambda p: old(dictionary_attention_bank(p, cfg), rows[0])
    candidate = lambda p: evaluate(p, rows[:1])[0]
    direction = jax.tree.map(lambda w: .001*jnp.sin(jnp.arange(w.size).reshape(w.shape)+.2), params)
    d0 = jax.jit(lambda p, v: jax.jvp(baseline, (p,), (v,))[1])(params, direction)
    d1 = jax.jit(lambda p, v: jax.jvp(candidate, (p,), (v,))[1])(params, direction)
    np.testing.assert_allclose(d1, d0, atol=2e-16, rtol=2e-11)
    g0, g1 = jax.jit(jax.grad(baseline))(params), jax.jit(jax.grad(candidate))(params)
    for a, b in zip(jax.tree.leaves(g0), jax.tree.leaves(g1), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-16, rtol=2e-11)
    assert np.linalg.norm(np.asarray(g1['encoder']['embedding'])) > 1e-20
    bra = np.asarray(rows[0]).copy(); bra[np.flatnonzero(bra == 0)[0]] = 1
    lowering = np.tile(bra, (19, 1)); lowering[np.arange(19), np.flatnonzero(bra)] = 0
    raw = make_mixed_clebsch_amplitude(model)
    ward = jax.jit(jax.vmap(raw, in_axes=(None, 0)))(generated, jnp.asarray(lowering))
    assert abs(np.asarray(ward).sum())/np.linalg.norm(ward) < 2e-11


def test_invalid_config_rejected():
    for options in ({'dictionary_rank': 0}, {'embedding_dim': 7, 'heads': 2},
                    {'local_symmetry': 'keep_one_chirality'}, {'architecture': 'physical_bit_vit'},
                    {'geometry_harmonics': 0}, {'geometry_harmonics': True}):
        with pytest.raises(ValueError):
            TriangleDictionaryAttentionConfig(**options)


def test_N144_full_dictionary_rejects_insufficient_attention_design_rank():
    model = MixedClebschConfig(length=12, multiplicity_cap=9, maximum_q=3)
    config = TriangleDictionaryAttentionConfig(model=model, embedding_dim=8)
    with pytest.raises(ValueError, match='attention feature rank'):
        init_dictionary_attention_from_bank(jax.random.PRNGKey(17911), {}, config)


@pytest.mark.parametrize('architecture', ['linear', 'attention'])
def test_N144_geometry_readout_replays_every_free_node_without_rank_loss(architecture):
    model = MixedClebschConfig(length=12, multiplicity_cap=9, maximum_q=3)
    family = init_mixed_clebsch_family(jax.random.PRNGKey(17912), model)
    bank = mixed_clebsch_bank(family['ternary_free'], model, 'ternary_free')
    config = TriangleDictionaryAttentionConfig(model=model, architecture=architecture,
        embedding_dim=32, geometry_harmonics=3)
    params, note = init_dictionary_attention_from_bank(jax.random.PRNGKey(17913), bank, config)
    reconstructed = dictionary_attention_bank(params, config)
    assert note['feature_readout_ranks'] == [16, 8, 4, 2, 1]
    assert note['exact_same_bank_expected']
    for actual, expected in zip(jax.tree.leaves(reconstructed), jax.tree.leaves(bank), strict=True):
        relative = np.linalg.norm(np.asarray(actual)-np.asarray(expected))/np.linalg.norm(expected)
        assert relative < 2e-11


def test_static_attention_prepare_cached_H_parameter_JVP_matches_legacy(parent):
    model, bank = parent
    cfg = TriangleDictionaryAttentionConfig(model=model, embedding_dim=8)
    params, _ = init_dictionary_attention_from_bank(jax.random.PRNGKey(17908), bank, cfg)
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    bonds = tuple(triangular_j1j2_bonds(6, 6, j2=.125, periodic=True))
    old = make_amplitude_heisenberg_action(make_mixed_projected_amplitude(model, images, chunk=4), bonds, n_sites=36)
    new = make_cached_projected_heisenberg_action(model, images, bonds)
    row = jnp.asarray(np.random.default_rng(17909).permutation([0]*18+[1]*18)[None], jnp.int32)
    legacy = lambda p: old(dictionary_attention_bank(p, cfg), row)
    cached = lambda p: new(prepare_feature_lut_bank(dictionary_attention_bank(p, cfg), model), row)
    direction = jax.tree.map(lambda w: .001*jnp.cos(jnp.arange(w.size).reshape(w.shape)+.7), params)
    expected = jax.jit(lambda p, v: jax.jvp(legacy, (p,), (v,)))(params, direction)
    actual = jax.jit(lambda p, v: jax.jvp(cached, (p,), (v,)))(params, direction)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=3e-16, rtol=2e-11)
    assert abs(np.asarray(actual[1][1])).max() > 1e-20
