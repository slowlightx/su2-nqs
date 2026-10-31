"""Independent physical-spin oracles; no Hamiltonian eigensolve or training."""
import itertools
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from su2_nqs.benchmarks.spin_partition_readout import (
    spin_partition_readout, spin_partition_distance, singlet_resonance_readout,
    _three_plus_three_singlet_columns,
)


def _dimers(n, pairs):
    bits = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    v = np.ones(2**n)
    for i, j in pairs:
        v *= (bits[:, i]-bits[:, j])/np.sqrt(2)
    return np.outer(v, v)


def _pair(n, i, j):
    bits = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    op = np.diag((bits[:, i]-.5)*(bits[:, j]-.5))
    for k, b in enumerate(bits):
        if b[i] != b[j]:
            op[k ^ ((1 << (n-1-i)) | (1 << (n-1-j))), k] = .5
    return op


def test_singlets_with_different_block_spin_structure():
    internal = spin_partition_readout(_dimers(6, [(0, 1), (2, 3), (4, 5)]), (0, 1, 2))
    crossing = spin_partition_readout(_dimers(6, [(0, 3), (1, 4), (2, 5)]), (0, 1, 2))
    assert internal['p_total']['0'] == pytest.approx(1., abs=3e-14, rel=0.)
    assert crossing['p_total']['0'] == pytest.approx(1., abs=3e-14, rel=0.)
    assert internal['singlet_pair_channels']['1'] == pytest.approx(1., abs=3e-14, rel=0.)
    assert crossing['singlet_pair_channels']['1'] == pytest.approx(.5, abs=3e-14, rel=0.)
    assert crossing['singlet_pair_channels']['3'] == pytest.approx(.5, abs=3e-14, rel=0.)
    distance = spin_partition_distance(internal, crossing)
    assert distance['joint_total_variation'] == pytest.approx(.5, abs=3e-14, rel=0.)
    assert distance['total_spin_total_variation'] < 3e-14


def test_arbitrary_broken_state_matches_independent_casimir_projectors():
    n = 4
    rng = np.random.default_rng(2026100601)
    v = rng.normal(size=16)+1j*rng.normal(size=16)
    v /= np.linalg.norm(v)
    rho = np.outer(v, v.conj())
    left, right = (0, 2), (1, 3)
    measured = spin_partition_readout(rho, left)
    assert not measured['strict_su2_eligible']
    assert 'invariant_readout' not in measured
    casimir = lambda sites: 3*len(sites)*np.eye(16)/4 + 2*sum(_pair(n, i, j) for i, j in itertools.combinations(sites, 2))
    cl, cr, ct = casimir(left), casimir(right), casimir(range(n))
    # Polynomial spectral projectors built from bit-basis S.S, independent of CG.
    def projector(c, q, allowed):
        p = np.eye(16)
        for other in allowed:
            if other != q:
                p = p @ (c-other*(other+2)*np.eye(16)/4)/((q*(q+2)-other*(other+2))/4)
        return p
    for w in measured['weights']:
        p = projector(cl, w['q_left'], (0, 2)) @ projector(cr, w['q_right'], (0, 2)) @ projector(ct, w['q_total'], (0, 2, 4))
        assert np.trace(rho@p).real == pytest.approx(w['probability'], abs=3e-14, rel=0.)
    expected = sum(np.trace(rho@_pair(n, i, j)).real for i in left for j in right)
    assert measured['cross_block_spin_dot'] == pytest.approx(expected, abs=3e-14, rel=0.)
    perm = (2, 0, 3, 1)
    reordered = rho.reshape([2]*8).transpose(perm+tuple(n+i for i in perm)).reshape(16, 16)
    other = spin_partition_readout(reordered, (0, 1))
    np.testing.assert_allclose([w['probability'] for w in measured['weights']],
                               [w['probability'] for w in other['weights']], atol=3e-14, rtol=0.)


def test_maximally_mixed_state_counts_multiplicities_and_invalid_inputs():
    r = spin_partition_readout(np.eye(64)/64, (0, 1, 2))
    multiplicities = {1: 2, 3: 1}
    for w in r['weights']:
        rank = multiplicities[w['q_left']]*multiplicities[w['q_right']]*(w['q_total']+1)
        assert w['probability'] == pytest.approx(rank/64, abs=3e-14, rel=0.)
    assert r['measured_spin_mutual_information_nats'] == pytest.approx(0., abs=3e-14, rel=0.)
    for left in ((), (0, 1, 2, 3), (0, 0), (True,), (1.5,), (4,)):
        with pytest.raises(ValueError):
            spin_partition_readout(np.eye(16)/16, left)
    with pytest.raises(ValueError):
        spin_partition_readout(np.eye(16), (0, 1))


def test_saved_NQS_joint_readout_replays_without_DMRG_or_projection():
    root = Path(__file__).resolve().parents[1]
    directory = root/'projects/clebsch_tree_nqs/results/spin_partition_coupling_readout_v1'
    registration = json.loads((directory/'registration.json').read_text())
    result = json.loads((directory/'results.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for path, digest in registration['inputs'].items():
        assert sha(root/path) == digest
    for path, digest in registration['sources'].items():
        assert sha(directory/'source_snapshots'/path) == digest
    assert not result['phase_claim'] and not result['size_evolution_released']
    assert all(result[k] == 0 for k in ('new_ED', 'new_training', 'new_samples', 'new_DMRG', 'new_full_H_calls'))
    trusted = root/registration['config']['trusted_archive']
    with np.load(trusted/'readout.npz') as z:
        for name in ('triangular_j0', 'triangular_j0.125', 'dimer_internal'):
            replay = spin_partition_readout(z[name+'_strip_6_rho'], (0, 2, 4))
            recorded = result['states'][name]['regions']['strip_6']
            np.testing.assert_allclose([w['probability'] for w in replay['weights']],
                                       [w['probability'] for w in recorded['weights']], atol=3e-14, rtol=0.)
    assert sum(any(r['exceeds_each_refinement_sum'] for r in p['regions'].values())
               for p in result['primary_pair_comparisons']) == 0
    assert max(result['validation'].values()) < 1e-10


def test_singlet_resonance_columns_against_independent_casimir_polynomials():
    n = 6
    for left in ((0, 1, 2), (0, 3, 4)):
        c = _three_plus_three_singlet_columns(left)
        total = 3*n*np.eye(64)/4+2*sum(_pair(n, i, j) for i, j in itertools.combinations(range(n), 2))
        p0 = np.eye(64)
        for spin in (1, 2, 3):
            p0 = p0@(np.eye(64)-total/(spin*(spin+1)))
        np.testing.assert_allclose(c@c.conj().T, p0, atol=3e-14, rtol=0)
        cl = 9*np.eye(64)/4+2*sum(_pair(n, i, j) for i, j in itertools.combinations(left, 2))
        np.testing.assert_allclose(c.conj().T@cl@c, np.diag([.75]*4+[3.75]), atol=3e-14, rtol=0)
        # Independent explicit symmetric Dicke spin-3/2 singlet; no CG call.
        dicke = [np.array([float(int(k).bit_count()==r) for k in range(8)])/
                  np.sqrt([1, 3, 3, 1][r]) for r in range(4)]
        v = sum((-1)**(3-r)*np.kron(dicke[r], dicke[3-r])/2 for r in range(4))
        bits = (np.arange(64)[:, None] >> np.arange(5, -1, -1)) & 1
        order = left+tuple(i for i in range(6) if i not in left)
        v = v[bits[:, order]@(2**np.arange(5, -1, -1))]
        np.testing.assert_allclose(np.outer(c[:, 4], c[:, 4].conj()), np.outer(v, v), atol=2e-14, rtol=0)


def test_singlet_resonance_distinguishes_occupation_from_coherence():
    c = _three_plus_three_singlet_columns((0, 1, 2))
    v = (c[:, 0]+1j*c[:, 4])/np.sqrt(2)
    pure = np.outer(v, v.conj())
    mixed = (np.outer(c[:, 0], c[:, 0].conj())+np.outer(c[:, 4], c[:, 4].conj()))/2
    a, block = singlet_resonance_readout(pure, (0, 1, 2))
    b, _ = singlet_resonance_readout(mixed, (0, 1, 2))
    assert a['p11_singlet'] == pytest.approx(b['p11_singlet'], abs=2e-14)
    assert a['p33_singlet'] == pytest.approx(b['p33_singlet'], abs=2e-14)
    assert a['cross_q1_q3_coherence_norm'] == pytest.approx(.5, abs=2e-14)
    assert b['cross_q1_q3_coherence_norm'] < 2e-14
    assert a['cross_q1_q3_normalized_coherence'] == pytest.approx(1., abs=2e-14)
    assert abs(block[0, 4]+.5j) < 2e-14
    # Broken-SU2 mixture still yields legal singlet operator data, not strict ES.
    up = np.eye(64)[63]
    r, _ = singlet_resonance_readout(.7*pure+.3*np.outer(up, up), (0, 1, 2))
    assert r['p_total_singlet'] == pytest.approx(.7, abs=2e-14)
    assert not r['full_entanglement_spectrum']
    with pytest.raises(ValueError):
        singlet_resonance_readout(np.eye(64)/64, (0, 1))
