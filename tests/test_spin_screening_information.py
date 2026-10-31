"""Independent exact checks of spin-screening conditional information."""
import itertools

import numpy as np
import pytest
from scipy.linalg import expm

from su2_nqs.benchmarks.spin_screening_information import (
    pseudoscalar_pair_split, screening_columns, screening_information_readout,
    screening_multiplicity_information,
)


def test_cg_counts_and_full_spin_squared():
    bank = np.column_stack([v for group in screening_columns((0, 1, 2)) for v in group])
    np.testing.assert_allclose(bank.conj().T@bank, np.eye(16), atol=2e-14)
    plus = np.zeros((64, 64), complex)
    for bits in range(64):
        for i in range(6):
            if not bits & (1 << i): plus[bits | (1 << i), bits] += 1
    z = np.diag([int(i).bit_count()-3 for i in range(64)])
    s2 = z@z+(plus@plus.conj().T+plus.conj().T@plus)/2
    np.testing.assert_allclose(bank.conj().T@s2@bank, np.diag([0.]*4+[2.]*12), atol=2e-14)


def test_product_screening_and_resonance_zero_information():
    groups = screening_columns((0, 1, 2)); rho = np.zeros((64, 64), complex)
    sigma = np.diag([.1, .2, .3, .4])
    for probability, columns in zip((.3, .7), groups):
        rho += sum(probability*v@sigma@v.conj().T/len(columns) for v in columns)
    r = screening_multiplicity_information(rho, (0, 1, 2))
    assert abs(r['holevo_nats']) < 2e-14
    assert abs(r['p11']-1) < 2e-14


def test_maximal_classical_screening_multiplicity_correlation():
    groups = screening_columns((0, 1, 2))
    rho = np.outer(groups[0][0][:, 0], groups[0][0][:, 0].conj())/2
    rho += sum(np.outer(v[:, 3], v[:, 3].conj())/6 for v in groups[1])
    r = screening_multiplicity_information(rho, (0, 1, 2))
    assert abs(r['holevo_nats']-np.log(2)) < 2e-14
    assert abs(r['conditional_trace_distance']-1) < 2e-14


def test_rotation_invariance_on_noninvariant_input():
    rng = np.random.default_rng(10613)
    v = rng.normal(size=64)+1j*rng.normal(size=64); v /= np.linalg.norm(v)
    rho = np.outer(v, v.conj()); sx = np.array([[0., .5], [.5, 0.]])
    rot = np.array([[1.]], complex)
    for _ in range(6): rot = np.kron(rot, expm(.61j*sx))
    a = screening_information_readout(rho, (0, 1, 2))
    b = screening_information_readout(rot@rho@rot.conj().T, (0, 1, 2))
    for key in ('p11', 'holevo_nats', 'pseudoscalar_product', 'pseudoscalar_high'):
        assert abs(a[key]-b[key]) < 3e-14


def test_chirality_independent_cartesian_and_support():
    spin = [np.array([[0., .5], [.5, 0.]]), np.array([[0., .5j], [-.5j, 0.]]), np.diag([-.5, .5])]
    local = np.zeros((8, 8), complex)
    for p in itertools.permutations(range(3)):
        sign = 1 if p in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] else -1
        local += sign*np.kron(np.kron(spin[p[0]], spin[p[1]]), spin[p[2]])
    split = pseudoscalar_pair_split((0, 1, 2))
    np.testing.assert_allclose(split['raw'], np.kron(local, local), atol=2e-14)
    assert split['diagnostics']['raw_vs_pair_product_span_max'] < 2e-14
    assert split['diagnostics']['high_vs_declared_span_max'] < 2e-13
    assert split['diagnostics']['span_rank'] == 62
    assert split['diagnostics']['high_relative_frobenius'] > .1
    # EVERY arbitrary Pauli string of support <=4, not just scalar bonds.
    pauli = [np.eye(2), *[2*x for x in spin]]
    for support in range(5):
        for sites in itertools.combinations(range(6), support):
            for axes in itertools.product((1, 2, 3), repeat=support):
                which = dict(zip(sites, axes)); op = np.array([[1.]], complex)
                for i in range(6): op = np.kron(op, pauli[which.get(i, 0)])
                assert abs(np.vdot(op, split['raw'])) < 2e-14


def test_zero_probability_and_invalid_input():
    v = np.zeros(64); v[-1] = 1
    r = screening_multiplicity_information(np.outer(v, v), (0, 1, 2))
    assert r['holevo_nats'] is None and r['weighted_holevo_nats'] == 0
    for left in [(0, 0, 1), (0, 1, 6), (True, 1, 2)]:
        with pytest.raises(ValueError): screening_columns(left)
    with pytest.raises(ValueError): screening_information_readout(np.eye(64), (0, 1, 2))


def test_strict_singlet_ordered_counterexample_not_qsl_classifier():
    # Independent physical S=0 polynomial, not a CG/RVB state constructor.
    # Each colour remains maximally polarized before its global singlet restoration.
    n = 12
    states = np.arange(2**n)
    bits = (states[:, None] >> np.arange(n-1, -1, -1)) & 1
    phase = np.exp(2j*np.pi*(np.arange(n) % 3)/3)
    v = np.prod(np.where(bits, 1., phase), axis=1)/np.sqrt(2**n)

    def ladder(x, raise_spin):
        out = np.zeros_like(x)
        for site in range(n):
            mask = 1 << (n-1-site)
            old = states[((states & mask) == 0) == raise_spin]
            out[old ^ mask] += x[old]
        return out

    z = bits.sum(axis=1)-n/2

    def s2(x):
        return z*z*x+(ladder(ladder(x, False), True)+ladder(ladder(x, True), False))/2

    for spin in range(n//2, 0, -1):
        v -= s2(v)/(spin*(spin+1))
    v /= np.linalg.norm(v)
    assert np.linalg.norm(ladder(v, True)) < 2e-12
    assert np.linalg.norm(s2(v)) < 2e-12
    for far_site in (3, 6, 9):
        action = (bits[:, 0]-.5)*(bits[:, far_site]-.5)*v
        old = states[bits[:, 0] != bits[:, far_site]]
        flip = (1 << (n-1)) | (1 << (n-1-far_site))
        action[old ^ flip] += .5*v[old]
        assert abs(np.vdot(v, action)-.25) < 2e-13
    coefficient = v.reshape(64, 64)
    rho = coefficient@coefficient.conj().T
    report = screening_information_readout(rho, (0, 1, 2))
    assert abs(report['weighted_holevo_nats']-.52826142910452) < 2e-12
    assert abs(report['p11']-7/9) < 2e-13
    assert report['original_block_entanglement_certificate'] is False
