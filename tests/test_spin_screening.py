"""Independent CG and ordinary crossed-dimer screening calibrations."""
import numpy as np
import pytest

from su2_nqs.benchmarks.spin_screening import block_spin_z, screening_weight_bound
from su2_nqs.su2.quantum_numbers import coupling_range, magnetic_numbers
from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.benchmarks.regional_su2_readout import regional_schur_columns


def test_lande_means_and_extrema_equal_explicit_cg_probabilities():
    for qt, minimum in ((2, -1.5), (4, -2.), (6, -2.25)):
        values = []
        for ql in range(0, 13, 2):
            for qr in range(0, 7, 2):
                if qt not in coupling_range(ql, qr): continue
                weights = [(mr/2, clebsch_gordan(ql, qt-mr, qr, mr, qt, qt)**2)
                    for mr in magnetic_numbers(qr) if qt-mr in magnetic_numbers(ql)]
                assert abs(sum(p for _, p in weights)-1) < 3e-13
                expected = sum(z*p for z, p in weights)
                actual = block_spin_z(ql, qr, qt, qt)
                assert abs(expected-actual) < 3e-13
                values.append(actual)
        assert min(values) == pytest.approx(minimum)


def test_bound_is_about_weight_not_probability_equal_to_polarization():
    b = screening_weight_bound(n_left=12,n_right=6,q_total=2,m2=2,
        right_spin_z=-.2679269324,casimir=2.,casimir_absolute_error=1e-12)
    assert b['nominal_negative_channel_weight_lower_bound'] == pytest.approx(.2679269324/1.5)
    assert .17861 < b['precision_adjusted_negative_channel_weight_lower_bound'] < .17862
    assert not b['qsl_specific'] and not b['independent_new_observable']
    assert [(r['q_left'],r['q_right']) for r in b['negative_channels']] == [(4,2),(6,4),(8,6)]
    with pytest.raises(ValueError,match='highest-weight'):
        screening_weight_bound(n_left=12,n_right=6,q_total=2,m2=0,right_spin_z=0.,casimir=2.)
    with pytest.raises(ValueError,match='Casimir below'):
        screening_weight_bound(n_left=12,n_right=6,q_total=2,m2=2,right_spin_z=0.,casimir=1.)


def test_trivial_boundary_triplet_with_crossed_dimers_already_has_screening_channels():
    # Active part of the existing18-spin control: two boundary ups and two
    # dimers crossing outer/middle. All other pairs are internal spin0.
    psi = np.zeros(64)
    for a in (0,1):
        for b in (0,1):
            bits = (1,1,a,b,1-a,1-b)
            psi[int(''.join(map(str,bits)),2)] = (1 if a else -1)*(1 if b else -1)/2
    ul, ll = regional_schur_columns(4); ur, lr = regional_schur_columns(2)
    coupled = ul.conj().T@psi.reshape(16,4)@ur.conj()
    weights = {}
    for i,(ql,_,_) in enumerate(ll):
        for j,(qr,_,_) in enumerate(lr):
            weights[ql,qr] = weights.get((ql,qr),0)+abs(coupled[i,j])**2
    assert weights[4,2] == pytest.approx(5/12)
    assert weights[2,0] == pytest.approx(1/4)
    assert weights[0,2] == pytest.approx(1/12)
    assert weights[2,2] == pytest.approx(1/4)
    # Nonzero negative-channel population cancels positive channels: middleSz0.
    mean = sum(p*block_spin_z(ql,qr,2,2) for (ql,qr),p in weights.items()
               if 2 in coupling_range(ql,qr))
    assert abs(mean) < 3e-14
