"""Fixed-state audits are unchanged by exact inactive upper-path removal."""
import jax
import numpy as np
import pytest

from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_clebsch_active_layout import mixed_active_layout, repack_active_clebsch_bank
from su2_nqs.models.triangular_clebsch_exchange_cache import make_feature_lut_exchange_queries
from su2_nqs.models.triangular_clebsch_feature_lut import (
    make_feature_lut_amplitude, make_feature_lut_projected_amplitude, prepare_feature_lut_bank,
)
from su2_nqs.models.triangular_mixed_clebsch import (
    MixedClebschConfig, init_mixed_clebsch_family, mixed_clebsch_layout,
)
from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action

jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='module')
def active_case():
    cfg = MixedClebschConfig(multiplicity_cap=9, maximum_q=3)
    bank = init_mixed_clebsch_family(jax.random.PRNGKey(9381), cfg)['ternary_free']
    native = jax.tree.map(lambda a: np.asarray(a).copy(), bank)
    layout = mixed_clebsch_layout(cfg)
    spec = tuple({q: (1 if i == len(layout)-1 else 2) for q in row['widths']}
                 for i, row in enumerate(layout))
    # Inactive first-level outputs are deliberately NONZERO. Only their parent
    # (the root here) discards them; exact closure must permit this situation.
    for weights in native['top'][-1]:
        weights[:, :, 2:, :] = 0.
        weights[:, :, :, 2:] = 0.
    packed = repack_active_clebsch_bank(native, cfg, spec)
    rng = np.random.default_rng(12783)
    bits = np.asarray([rng.permutation([0]*18+[1]*18) for _ in range(5)], np.int32)
    return cfg, native, packed, spec, bits


def test_active_layout_preserves_raw_and_exchange_amplitudes(active_case):
    cfg, native, packed, spec, bits = active_case
    a, b = (prepare_feature_lut_bank(bank, cfg) for bank in (native, packed))
    ordinary = jax.jit(jax.vmap(make_feature_lut_amplitude(cfg), in_axes=(None, 0)))
    compact = jax.jit(jax.vmap(make_feature_lut_amplitude(cfg, active_top_widths=spec), in_axes=(None, 0)))
    x, y = np.asarray(ordinary(a, bits)), np.asarray(compact(b, bits))
    assert np.linalg.norm(x) > 1e-12
    assert np.max(abs(x-y))/np.linalg.norm(x) < 2e-13
    pairs = np.asarray([(0, 1), (0, 20), (5, 30), (7, 22)], int)
    old = jax.jit(make_feature_lut_exchange_queries(cfg, pairs))(a, bits[0])
    new = jax.jit(make_feature_lut_exchange_queries(cfg, pairs, active_top_widths=spec))(b, bits[0])
    np.testing.assert_allclose(new[0], old[0], atol=1e-18, rtol=2e-12)
    np.testing.assert_allclose(new[1], old[1], atol=1e-18, rtol=2e-12)
    assert all(np.asarray(old).shape == np.asarray(new).shape for old, new in zip(native['local'][0], packed['local'][0]))
    assert sum(w.size for node in packed['top'] for w in node) < sum(w.size for node in native['top'] for w in node)


def test_active_layout_preserves_complete_A1_and_full_H_action(active_case):
    cfg, native, packed, spec, bits = active_case
    a, b = (prepare_feature_lut_bank(bank, cfg) for bank in (native, packed))
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    old = jax.jit(make_feature_lut_projected_amplitude(cfg, images))
    new = jax.jit(make_feature_lut_projected_amplitude(cfg, images, active_top_widths=spec))
    x, y = np.asarray(old(a, bits[0])), np.asarray(new(b, bits[0]))
    np.testing.assert_allclose(y, x, atol=1e-18, rtol=2e-12)
    np.testing.assert_allclose(new(b, bits[0, images[21]]), y, atol=1e-18, rtol=2e-12)
    bonds = tuple(triangular_j1j2_bonds(6, 6, j2=.125, periodic=True))
    old_h = make_cached_projected_heisenberg_action(cfg, images, bonds)(a, bits[:1])
    new_h = make_cached_projected_heisenberg_action(cfg, images, bonds, active_top_widths=spec)(b, bits[:1])
    np.testing.assert_allclose(new_h, old_h, atol=1e-18, rtol=2e-12)


def test_active_layout_requires_exact_parent_closure_and_valid_widths(active_case):
    cfg, native, _, spec, _ = active_case
    bad = jax.tree.map(lambda a: np.asarray(a).copy(), native)
    bad['top'][-1][0][0, 0, 2, 0] = 1e-100
    with pytest.raises(ValueError, match='nonzero discarded-child'):
        repack_active_clebsch_bank(bad, cfg, spec)
    with pytest.raises(ValueError, match='per upper merge'):
        mixed_active_layout(cfg, spec[:-1])
    with pytest.raises(ValueError, match='native-width subsets'):
        mixed_active_layout(cfg, ({0: 10}, *spec[1:]))
    with pytest.raises(ValueError, match='native-width subsets'):
        mixed_active_layout(cfg, ({0: True}, *spec[1:]))


def test_active_layout_regroups_unequal_widths_and_omitted_charge_paths(active_case):
    cfg, native, _, _, bits = active_case
    original = mixed_clebsch_layout(cfg)
    for spec in (({0: 1, 2: 2}, {0: 2, 2: 1}, {0: 1}),
                 ({0: 2}, {0: 2}, {0: 1})):
        bank = jax.tree.map(lambda a: np.asarray(a).copy(), native)
        for group, weights in zip(original[-1]['groups'], bank['top'][-1], strict=True):
            for k, (ql, qr, _) in enumerate(group['keys']):
                dl, dr = spec[0].get(ql, 0), spec[1].get(qr, 0)
                weights[k, :, dl:, :] = 0.; weights[k, :, :, dr:] = 0.
        packed = repack_active_clebsch_bank(bank, cfg, spec)
        old = jax.jit(make_feature_lut_amplitude(cfg))(prepare_feature_lut_bank(bank, cfg), bits[0])
        new = jax.jit(make_feature_lut_amplitude(cfg, active_top_widths=spec))(
            prepare_feature_lut_bank(packed, cfg), bits[0])
        np.testing.assert_allclose(new, old, atol=1e-18, rtol=2e-12)
