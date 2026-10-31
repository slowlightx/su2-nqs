"""Same-state live multiplicity expansion, not a change of CG conventions."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_arity_clebsch import arity_clebsch_bank, init_arity_pair
from su2_nqs.models.triangular_clebsch_expansion import initialize_live_triangular_expansion
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, init_mixed_clebsch_family, make_mixed_clebsch_amplitude,
    make_mixed_projected_amplitude, mixed_clebsch_bank, mixed_parameter_count,
)
from su2_nqs.vmc.exchange_bridge_importance import make_amplitude_heisenberg_action

jax.config.update('jax_enable_x64', True)


def _parent(cap=9, maximum_q=3):
    cfg = MixedClebschConfig(multiplicity_cap=cap, maximum_q=maximum_q)
    family = init_mixed_clebsch_family(jax.random.PRNGKey(19101), cfg)
    bank = mixed_clebsch_bank(family['ternary_c3'], cfg, 'ternary_c3')
    pair, _ = init_arity_pair(bank, cfg, 8)
    return cfg, pair


@pytest.mark.parametrize('mode', ['binary_truncated', 'ternary_joint'])
@pytest.mark.parametrize('old_cap,new_cap', [(9, 10), (20, 28)])
def test_raw_amplitudes_inherited_jvp_and_new_root_tangent(mode, old_cap, new_cap):
    old, pair = _parent(cap=old_cap)
    new = MixedClebschConfig(multiplicity_cap=new_cap, maximum_q=3)
    p = pair[mode]
    expanded, report = initialize_live_triangular_expansion(old, p, new,
        jax.random.PRNGKey(19102), mode=mode)
    rng = np.random.default_rng(19103)
    rows = np.asarray([rng.permutation([0]*18+[1]*18) for _ in range(12)], np.int32)
    old_raw, new_raw = make_mixed_clebsch_amplitude(old), make_mixed_clebsch_amplitude(new)
    first = jax.jit(jax.vmap(lambda z, x: old_raw(arity_clebsch_bank(z, old, mode, 8), x), in_axes=(None, 0)))
    second = jax.jit(jax.vmap(lambda z, x: new_raw(arity_clebsch_bank(z, new, mode, 8), x), in_axes=(None, 0)))
    psi = np.asarray(first(p, rows))
    np.testing.assert_allclose(second(expanded, rows), psi, atol=2e-17, rtol=3e-12)
    direction = jax.tree.map(lambda w: jnp.cos(w)*.01, p)
    lift = lambda z: initialize_live_triangular_expansion(old, z, new,
        jax.random.PRNGKey(19102), mode=mode)[0]
    inherited = jax.jvp(lift, (p,), (direction,))[1]
    a = jax.jvp(lambda z: first(z, rows), (p,), (direction,))[1]
    b = jax.jvp(lambda z: second(z, rows), (expanded,), (inherited,))[1]
    np.testing.assert_allclose(a, b, atol=2e-17, rtol=3e-12)
    # Root's new-child coefficients are initially zero, but their tangent is
    # nonzero because the corresponding child features were not zero-padded.
    tangent = jax.tree.map(jnp.zeros_like, expanded)
    root = list(tangent['top'][-1])
    root[0] = root[0].at[:, :, -1, :].set(1.)
    tangent['top'] = tuple(tangent['top'][:-1])+(tuple(root),)
    derivative = np.asarray(jax.jvp(lambda z: second(z, rows), (expanded,), (tangent,))[1])
    perpendicular = derivative-psi*np.vdot(psi, derivative)/np.vdot(psi, psi)
    assert np.linalg.norm(perpendicular) > 1e-10
    assert report['exact_same_physical_state'] and report['new_optimizer_registration_required']
    if mode == 'binary_truncated':
        for a, b in zip(jax.tree.leaves(p['six']), jax.tree.leaves(expanded['six']), strict=True):
            np.testing.assert_array_equal(a, b)


def test_complete_A1_full_H_and_singlet_Ward_survive_width_change():
    old, pair = _parent()
    new = MixedClebschConfig(multiplicity_cap=10, maximum_q=3)
    p = pair['ternary_joint']
    expanded, _ = initialize_live_triangular_expansion(old, p, new,
        jax.random.PRNGKey(19104), mode='ternary_joint')
    rng = np.random.default_rng(19105)
    rows = np.asarray([rng.permutation([0]*18+[1]*18) for _ in range(2)], np.int32)
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    a = make_mixed_projected_amplitude(old, images, chunk=4)
    b = make_mixed_projected_amplitude(new, images, chunk=4)
    first, second = jax.jit(a), jax.jit(b)
    np.testing.assert_allclose(first(p, rows[0]), second(expanded, rows[0]), atol=2e-17, rtol=3e-12)
    np.testing.assert_allclose(second(expanded, rows[0, images[31]]), second(expanded, rows[0]), atol=2e-17, rtol=3e-12)
    # Full H without projected-query duplication: every connected amplitude
    # is itself unchanged, so linear projection also commutes with this check.
    bonds = triangular_j1j2_bonds(6, 6, j2=.125, periodic=True)
    h0 = jax.jit(make_amplitude_heisenberg_action(make_mixed_clebsch_amplitude(old), bonds, n_sites=36))
    h1 = jax.jit(make_amplitude_heisenberg_action(make_mixed_clebsch_amplitude(new), bonds, n_sites=36))
    for x, y in zip(h0(p, rows), h1(expanded, rows), strict=True):
        np.testing.assert_allclose(x, y, atol=2e-16, rtol=3e-12)
    bra = rng.permutation([0]*17+[1]*19)
    lowering = np.tile(bra, (19, 1))
    lowering[np.arange(19), np.flatnonzero(bra)] = 0
    ward = np.asarray(jax.jit(jax.vmap(make_mixed_clebsch_amplitude(new), in_axes=(None, 0)))(expanded, lowering))
    assert abs(ward.sum())/np.linalg.norm(ward) < 3e-12


def test_production_widths_keep_binary_rank_and_native_parameter_counts():
    old, pair = _parent(cap=16, maximum_q=10)
    counts = {'binary_truncated': [353072, 479840, 633952],
              'ternary_joint': [353028, 479852, 634020]}
    for mode in counts:
        assert mixed_parameter_count(pair[mode]) == counts[mode][0]
        for index, cap in enumerate((18, 20), 1):
            expanded, note = initialize_live_triangular_expansion(old, pair[mode],
                MixedClebschConfig(multiplicity_cap=cap, maximum_q=10),
                jax.random.PRNGKey(19110+cap), mode=mode)
            assert mixed_parameter_count(expanded) == counts[mode][index]
            assert note['binary_rank'] == (8 if mode == 'binary_truncated' else None)


def test_width_inclusion_rejects_arity_cutoff_cover_rank_and_bad_scale():
    old, pair = _parent()
    p = pair['binary_truncated']; key = jax.random.PRNGKey(19120)
    for new in (old, MixedClebschConfig(length=12, multiplicity_cap=10, maximum_q=3),
                MixedClebschConfig(multiplicity_cap=10, maximum_q=4)):
        with pytest.raises(ValueError):
            initialize_live_triangular_expansion(old, p, new, key, mode='binary_truncated')
    new = MixedClebschConfig(multiplicity_cap=10, maximum_q=3)
    for scale in (0., -1., np.nan):
        with pytest.raises(ValueError, match='positive'):
            initialize_live_triangular_expansion(old, p, new, key, mode='binary_truncated', new_feature_scale=scale)
    with pytest.raises(ValueError, match='rank'):
        initialize_live_triangular_expansion(old, p, new, key, mode='binary_truncated', binary_rank=4)
