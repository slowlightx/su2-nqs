"""Independent physical and rank checks for the finite binary control."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.models.triangular_arity_clebsch import (
    arity_clebsch_bank, binary_intermediate_widths, init_arity_pair,
    six_edge_unfoldings,
)
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, init_mixed_clebsch_family, mixed_clebsch_bank,
    mixed_parameter_count, make_mixed_clebsch_amplitude,
    make_mixed_projected_amplitude,
)
from su2_nqs.su2.triangle_blocks import triangle_block_tables
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.quantum_numbers import coupling_range
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action

jax.config.update('jax_enable_x64', True)


def parent_bank(config):
    family = init_mixed_clebsch_family(jax.random.PRNGKey(16101), config)
    return mixed_clebsch_bank(family['ternary_c3'], config, 'ternary_c3')


def test_complete_binary_is_equivalent_but_finite_rank_is_a_real_restriction():
    cfg = MixedClebschConfig()
    bank = parent_bank(cfg)
    complete, _ = init_arity_pair(bank, cfg, 9)
    for old, new in zip(jax.tree.leaves(bank), jax.tree.leaves(complete['ternary_joint']), strict=True):
        np.testing.assert_allclose(old, new, atol=7e-15, rtol=5e-13)
    pair, note = init_arity_pair(bank, cfg, 4)
    assert binary_intermediate_widths(4) == {0: 4, 2: 4, 4: 4, 6: 1}
    assert not note['joint_tensor_factorized']
    assert note['same_physical_initial_state'] and not note['parent_preserved_exactly']
    # Matricization joins ALL output q/copies. A separate SVD per output
    # would accidentally give a larger, inconsistent binary state family.
    for matrices in six_edge_unfoldings(pair['ternary_joint'], cfg):
        for q, matrix in matrices.items():
            assert np.linalg.matrix_rank(matrix, tol=2e-12) <= binary_intermediate_widths(4)[q]
    rng = np.random.default_rng(16102)
    unconstrained = dict(local=tuple(tuple(jnp.asarray(rng.normal(size=w.shape)) for w in node)
                                     for node in bank['local']), top=bank['top'])
    assert np.linalg.matrix_rank(six_edge_unfoldings(unconstrained, cfg)[0][2], tol=2e-12) == 9
    # Same output and upper-tree widths; no dummy trainable padding.
    counts = {arm: mixed_parameter_count(p) for arm, p in pair.items()}
    assert counts == {'binary_truncated': 350424, 'ternary_joint': 353028}
    assert abs(counts['binary_truncated']/counts['ternary_joint'] - 1) < .01
    matched, _ = init_arity_pair(bank, cfg, 8)
    assert mixed_parameter_count(matched['binary_truncated']) == 353072
    local_binary = mixed_parameter_count({k:v for k,v in matched['binary_truncated'].items() if k!='top'})
    local_joint = mixed_parameter_count(dict(local=matched['ternary_joint']['local']))
    assert (local_binary, local_joint) == (7792, 7748)
    assert abs(local_binary/local_joint-1) < .01
    for matrices in six_edge_unfoldings(matched['ternary_joint'], cfg):
        assert np.linalg.matrix_rank(matrices[2], tol=2e-12) <= 8


def test_folded_ternary_equals_literal_binary_CG_contraction_all_512_rows():
    cfg = MixedClebschConfig()
    pair, _ = init_arity_pair(parent_bank(cfg), cfg, 4)
    p = pair['binary_truncated']
    folded = arity_clebsch_bank(p, cfg, 'binary_truncated', 4)
    data = triangle_block_tables()
    widths = binary_intermediate_widths(4)
    for q, w9 in zip(cfg.local_q, p['local'][0], strict=True):
        expected = np.zeros((512, w9.shape[1]))
        for packed in range(512):
            six_bits, three_bits = packed >> 3, packed & 7
            ml, mr = 2*six_bits.bit_count()-6, 2*three_bits.bit_count()-3
            offset = 0
            for q6 in (0, 2, 4, 6):
                for q3 in (1, 3):
                    if q not in coupling_range(q6, q3):
                        continue
                    count = widths[q6]*data['three'][q3].shape[1]
                    weights = np.asarray(w9[offset:offset+count]).reshape(widths[q6], -1, w9.shape[1])
                    offset += count
                    if abs(ml) > q6 or abs(mr) > q3 or abs(ml+mr) > q:
                        continue
                    left = data['six'][q6][six_bits] @ np.asarray(p['six'][0][q6//2])
                    right = data['three'][q3][three_bits]
                    expected[packed] += clebsch_gordan(q6, ml, q3, mr, q, ml+mr)*np.einsum('a,b,abk->k', left, right, weights)
        actual = data['nine'][q] @ np.asarray(folded['local'][0][cfg.local_q.index(q)])
        np.testing.assert_allclose(actual, expected, atol=3e-15, rtol=8e-12)


def test_same_N36_singlet_full_H_A1_and_trainable_binary_gradient():
    cfg = MixedClebschConfig(multiplicity_cap=9, maximum_q=3)
    pair, _ = init_arity_pair(parent_bank(cfg), cfg, 2)
    banks = {mode: arity_clebsch_bank(p, cfg, mode, 2) for mode, p in pair.items()}
    rng = np.random.default_rng(16103)
    rows = np.asarray([rng.permutation([0]*18+[1]*18) for _ in range(4)], np.int32)
    raw = make_mixed_clebsch_amplitude(cfg)
    evaluate = jax.jit(jax.vmap(raw, in_axes=(None, 0)))
    reference = evaluate(banks['ternary_joint'], rows)
    assert np.linalg.norm(reference) > 1e-15
    np.testing.assert_allclose(evaluate(banks['binary_truncated'], rows), reference, atol=3e-17, rtol=3e-12)
    bra = rng.permutation([0]*17+[1]*19)
    lowering = np.tile(bra, (19, 1))
    lowering[np.arange(19), np.flatnonzero(bra)] = 0
    for bank in banks.values():
        ward = np.asarray(evaluate(bank, lowering))
        assert abs(ward.sum())/np.linalg.norm(ward) < 3e-12
    bonds = tuple(triangular_j1j2_bonds(6, 6, j2=.125, periodic=True))
    h = make_amplitude_heisenberg_action(raw, bonds, n_sites=36)
    _, hp = jax.jit(h)(banks['binary_truncated'], rows[:1])
    conn, elements = SzHeisenbergGraph(36, bonds, n_up=18).get_conn(tuple(rows[0]))
    expected = np.dot(elements, np.asarray(evaluate(banks['binary_truncated'], np.asarray(conn))))
    np.testing.assert_allclose(hp[0], expected, atol=2e-16, rtol=3e-12)
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    projected = jax.jit(make_mixed_projected_amplitude(cfg, images, chunk=4))
    bank = banks['binary_truncated']
    np.testing.assert_allclose(projected(bank, rows[0]), projected(bank, rows[0, images[31]]), atol=2e-17, rtol=3e-12)
    p = pair['binary_truncated']
    loss = lambda z: projected(arity_clebsch_bank(z, cfg, 'binary_truncated', 2), rows[0])
    tangent = jax.tree.map(lambda w: .001*jnp.cos(w), p)
    d = jax.jit(lambda z, v: jax.jvp(loss, (z,), (v,))[1])(p, tangent)
    plus = jax.tree.map(lambda z, v: z+1e-4*v, p, tangent)
    minus = jax.tree.map(lambda z, v: z-1e-4*v, p, tangent)
    fd = (loss(plus)-loss(minus))/(2e-4)
    np.testing.assert_allclose(d, fd, atol=3e-12, rtol=3e-6)


@pytest.mark.parametrize('rank', [0, -1, True, 2.5])
def test_invalid_binary_rank_rejected(rank):
    with pytest.raises(ValueError):
        binary_intermediate_widths(rank)
