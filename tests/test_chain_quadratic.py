"""Independent ordered-pair and physical-singlet oracles for quadratic data."""
import itertools

import numpy as np
import pytest

from su2_nqs.vmc.chain_quadratic import whole_chain_squared_mean
from su2_nqs.benchmarks.spin_partition_readout import _three_plus_three_singlet_columns


def test_complex_whole_chain_oracle_and_no_snapshot_pseudoreplication():
    rng = np.random.default_rng(2026101701)
    x = rng.normal(size=(15, 4))+1j*rng.normal(size=(15, 4))
    ids = np.repeat(np.arange(5), 3)
    means = x.reshape(5, 3, 4).mean(1)
    expected = np.mean([np.vdot(means[a], means[b]).real
                        for a in range(5) for b in range(5) if a != b])
    result = whole_chain_squared_mean(x, ids)
    assert result['value'] == pytest.approx(expected, abs=5e-16)
    replicated = whole_chain_squared_mean(np.repeat(x, 4, axis=0), np.repeat(ids, 4))
    assert replicated['value'] == pytest.approx(result['value'], abs=5e-16)
    assert replicated['whole_chain_jackknife_SE'] == pytest.approx(result['whole_chain_jackknife_SE'], abs=1e-15)


def test_noise_only_enumeration_is_unbiased_and_negative_not_clipped():
    results = [whole_chain_squared_mean(np.array(signs)[:, None], np.arange(4))
               for signs in itertools.product((-1., 1.), repeat=4)]
    assert np.mean([x['value'] for x in results]) == pytest.approx(0., abs=1e-16)
    assert any(x['negative_estimate_retained'] for x in results)
    assert np.mean([x['plug_in_value'] for x in results]) == pytest.approx(.25)


def test_same_screening_occupations_do_not_fix_singlet_coherence():
    c = _three_plus_three_singlet_columns((0, 1, 2))
    psi = (c[:, 0]+1j*c[:, 4])/np.sqrt(2)
    coherent = np.outer(psi, psi.conj())
    mixed = (np.outer(c[:, 0], c[:, 0].conj())+np.outer(c[:, 4], c[:, 4].conj()))/2
    blocks = [c.conj().T@rho@c for rho in (coherent, mixed)]
    np.testing.assert_allclose(np.diag(blocks[0]), np.diag(blocks[1]), atol=1e-15, rtol=0)
    for block, expected in zip(blocks, (.25, 0.)):
        value = whole_chain_squared_mean(np.tile(block[:4, 4], (8, 1)), np.repeat(np.arange(4), 2))
        assert value['value'] == pytest.approx(expected, abs=1e-15)
    rng = np.random.default_rng(2026101702)
    u, _ = np.linalg.qr(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))
    rotated = whole_chain_squared_mean(np.tile(u@blocks[0][:4, 4], (4, 1)), np.arange(4))
    assert rotated['value'] == pytest.approx(.25, abs=1e-15)


def test_screening_readout_retains_physical_occupation_and_complex_coherence():
    from projects.clebsch_tree_nqs.workflows.spin_singlet_screening import screening_chain_readout
    c = _three_plus_three_singlet_columns((0, 1, 2))
    psi = (c[:, 0]+1j*c[:, 4])/np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    result = screening_chain_readout(np.tile(rho,(4,1,1)),np.arange(4),(0,1,2))
    assert result['occupations']['p11_singlet']['value'] == pytest.approx(.5,abs=1e-15)
    assert result['occupations']['p33_singlet']['value'] == pytest.approx(.5,abs=1e-15)
    assert result['cross_screening_coherence_squared']['value'] == pytest.approx(.25,abs=1e-15)
    split=result['q33_linear_decomposition']
    assert split['full']['value'] == pytest.approx(split['low']['value']+split['high']['value'],abs=1e-15)
    assert not result['strict_entanglement_spectrum_reported']
    with pytest.raises(ValueError):
        screening_chain_readout(np.tile(2*rho,(4,1,1)),np.arange(4),(0,1,2))


@pytest.mark.parametrize('x,ids', [
    (np.ones((3, 1)), [0, 0, 1]),
    (np.ones((4, 1)), [0, 0, 1, 2]),
    (np.ones((3, 1)), [0., 1., 2.]),
    (np.array([[1.], [np.nan], [0.]]), [0, 1, 2]),
    (np.ones(3), [0, 1, 2]),
])
def test_invalid_inputs_are_not_repaired(x, ids):
    with pytest.raises(ValueError):
        whole_chain_squared_mean(x, np.asarray(ids))
