"""Node-safe terminal coordinates, with independent direct action oracles."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.projected_spin_linear import make_terminal_tensor_products

jax.config.update('jax_enable_x64', True)


def test_complex_linear_H_columns_retain_exact_nodes_and_no_64_site_packing():
    rng = np.random.default_rng(2610232110)
    phi = rng.normal(size=(4, 3))+1j*rng.normal(size=(4, 3))
    # An exact amplitude node can have a nonzero Hamiltonian action.
    phi[0] = 0
    raw = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    h = raw+raw.conj().T
    hp = h @ phi
    phi, hp = jnp.asarray(phi), jnp.asarray(hp)
    bank = dict(top=((jnp.array([.2, -.7]), jnp.array([.4])),), marker=jnp.array(13.))

    def action(b, rows):
        coeff = jnp.r_[b['top'][-1][0], b['top'][-1][1]]
        index = 2*rows[:, 0]+rows[:, 1]
        return phi[index] @ coeff, hp[index] @ coeff

    old, replace, columns = make_terminal_tensor_products(action, bank)
    rows = np.zeros((4, 144), np.int32)
    rows[:, :2] = [[0, 0], [0, 1], [1, 0], [1, 1]]
    f, hf = columns(bank, rows)
    np.testing.assert_allclose(f, phi, atol=2e-14)
    np.testing.assert_allclose(hf, hp, atol=2e-14)
    assert np.linalg.norm(hf[0]) > 1 and np.linalg.norm(f[0]) == 0
    for coeff in (old, jnp.array([.8, -.3, .7]), jnp.zeros(3)):
        p, a = action(replace(bank, coeff), rows)
        np.testing.assert_allclose(f @ coeff, p, atol=2e-14)
        np.testing.assert_allclose(hf @ coeff, a, atol=2e-14)
    assert float(bank['marker']) == 13.
    np.testing.assert_array_equal(bank['top'][-1][0], [.2, -.7])


def test_actual_N36_complete_A1_full_H_terminal_columns_replay():
    from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank
    from su2_nqs.models.triangular_mixed_clebsch import (
        MixedClebschConfig, init_mixed_clebsch_family, mixed_clebsch_bank,
    )
    from su2_nqs.vmc.triangular_clebsch_cached_action import make_cached_projected_heisenberg_action

    cfg = MixedClebschConfig(multiplicity_cap=9, maximum_q=3)
    family = init_mixed_clebsch_family(jax.random.PRNGKey(2610232111), cfg)
    bank = mixed_clebsch_bank(family['ternary_c3'], cfg, 'ternary_c3')
    bank = prepare_feature_lut_bank(bank, cfg)
    images = np.asarray(triangular_hnf_gamma_a1_group(((6, 0), (0, 6))), np.int32)
    action = make_cached_projected_heisenberg_action(cfg, images,
        triangular_j1j2_bonds(6, 6, j2=.125, periodic=True))
    old, replace, columns = make_terminal_tensor_products(action, bank)
    rng = np.random.default_rng(2610232112)
    rows = np.asarray([rng.permutation([0]*18+[1]*18)], np.int32)
    f, hf = columns(bank, rows)
    for coeff in (old, old+.1*jnp.arange(len(old))):
        expected = np.asarray(action(replace(bank, coeff), rows))
        observed = np.asarray([f @ coeff, hf @ coeff])
        scale = max(np.max(abs(expected)), np.finfo(float).tiny)
        assert np.max(abs(observed-expected))/scale < 3e-11
    np.testing.assert_array_equal(np.asarray(columns(bank, np.zeros((1, 36), np.int32))), 0.)
    # Same complete A1 columns on a translated physical configuration.
    g, hg = columns(bank, rows[:, images[17]])
    np.testing.assert_allclose((g, hg), (f, hf), atol=1e-15, rtol=3e-11)


@pytest.mark.parametrize('bad', [dict(), dict(top=()), dict(top=(jnp.ones(2, dtype=jnp.float32),)),
    dict(top=(jnp.array([np.nan]),)), dict(top=(jnp.ones(2, dtype=jnp.complex128),))])
def test_invalid_coordinate_banks_fail_closed(bad):
    with pytest.raises(ValueError):
        make_terminal_tensor_products(lambda *_: None, bad)
