"""Independent axial-rotor and boundary-spin controls for the family screen."""
import numpy as np
import pytest

from su2_nqs.benchmarks.rotor_family_scan import scan_quadratic_rotor_families
from su2_nqs.benchmarks.rotor_family_scan import axial_spatial_rotor_prediction,compare_frozen_spatial_rotor


def _axial_levels(q, a, c):
    # Independent analytic symmetric-top spectrum, doubled magnetic labels.
    m2 = np.arange(-q, q+1, 2)
    return np.sort(a*q*(q+2)/4+(c-a)*m2**2/4)


def test_higher_rotor_is_found_even_with_a_softer_boundary_triplet():
    a, c, eps = .2, .3, .001
    energies = {0: [0.0]}
    for q in (2, 4, 6):
        actual = _axial_levels(q, a, c)
        energies[q] = actual.tolist()
    energies[2] = [eps]+energies[2]
    result = scan_quadratic_rotor_families(energies)
    best = result['candidates'][0]
    assert best['spin1_start'] == 1
    assert best['maximum_predictive_error'] < 2e-15
    np.testing.assert_allclose(best['coefficients'], [a, a, c], atol=2e-16)
    assert not result['phase_claim'] and not result['higher_rotor_excluded']


def test_distant_spin2_levels_do_not_match_the_predicted_family():
    energies = {q: _axial_levels(q, .2, .3).tolist() for q in (0, 2, 4, 6)}
    energies[4] = (np.asarray(energies[4])+2).tolist()
    result = scan_quadratic_rotor_families(energies)
    assert result['candidates'][0]['maximum_predictive_error'] > 1


def test_missing_or_invalid_input_is_not_a_negative_phase_result():
    result = scan_quadratic_rotor_families({0: [0], 2: [.1], 4: [], 6: []})
    assert result['status'] == 'insufficient_observed_multiplets'
    assert not result['phase_claim']
    for energies in ({0: [0], 2: [.2, .1], 4: [], 6: []},
                     {0: [0], 2: [np.nan], 4: [], 6: []},
                     {False: [0], 2: [.1], 4: [], 6: []},
                     {0: [0], 2: [.1+0j], 4: [], 6: []}):
        with pytest.raises(ValueError):
            scan_quadratic_rotor_families(energies)


def _spatial_reference(a,c,spin2_shift=0.):
    # Independent symmetric-top polynomials and physical mirror embedding.
    data={0:[(0.,'k0_mirror-1')],
        2:[(2*a,'k0_mirror+1'),(a+c,'K_star'),(a+c,'K_star')],
        4:[(6*a,'k0_mirror-1'),(5*a+c,'K_star'),(5*a+c,'K_star'),
           (2*a+4*c,'K_star'),(2*a+4*c,'K_star')],
        6:[(12*a,'k0_mirror+1'),(11*a+c,'K_star'),(11*a+c,'K_star'),
           (8*a+4*c,'K_star'),(8*a+4*c,'K_star'),
           (3*a+9*c,'k0_mirror+1'),(3*a+9*c,'k0_mirror-1')]}
    sectors={};spatial={}
    for q,entries in data.items():
        entries=sorted(entries)
        sectors[str(q)]=dict(leading_levels=[dict(entanglement_energy=1.2+e+(spin2_shift if q==4 else 0),
            physical_casimir_expectation=q*(q+2)/4,higher_spin_weight_upper_bound_nominal=0.) for e,_ in entries])
        spatial[str(q)]=dict(momentum_weights=[[int(channel.startswith('k0')),0,
            .5*int(channel=='K_star'),0,.5*int(channel=='K_star'),0] for _,channel in entries],
            gamma_parallel_even_weights=[int(channel=='k0_mirror+1') for _,channel in entries],
            gamma_parallel_odd_weights=[int(channel=='k0_mirror-1') for _,channel in entries])
    return dict(leading_spin_content=dict(sectors=sectors,raising_transitions={}),leading_spatial_content=spatial)


@pytest.mark.parametrize('a,c',[(.2,.3),(.4,.1)])
def test_spatial_rotor_predicts_five_and_seven_multiplets_without_refit(a,c):
    result=compare_frozen_spatial_rotor(_spatial_reference(a,c))
    np.testing.assert_allclose(result['frozen_axial_coefficients'],[a,c],atol=3e-16,rtol=0)
    assert not result['fit_uses_S2_or_S3']
    for q in (4,6):
        row=result['comparisons'][str(q)]
        assert row['required_independent_multiplets']==q+1
        assert row['maximum_energy_prediction_error']<2e-15
        assert row['minimum_matched_spatial_weight']==1
    for q in (4,6):
        predicted=axial_spatial_rotor_prediction(a,c,q,-1)
        assert all(isinstance(x['body_abs_m2'],int) and x['body_abs_m2']%2==0 for x in predicted)
        np.testing.assert_allclose([x['gap'] for x in predicted],_axial_levels(q,a,c),atol=2e-15,rtol=0)


def test_higher_spin_mismatch_cannot_change_frozen_inertias():
    reference=compare_frozen_spatial_rotor(_spatial_reference(.2,.3))
    shifted=compare_frozen_spatial_rotor(_spatial_reference(.2,.3,.4))
    np.testing.assert_array_equal(shifted['frozen_axial_coefficients'],reference['frozen_axial_coefficients'])
    assert shifted['comparisons']['4']['maximum_energy_prediction_error']>.39
    assert not shifted['phase_claim']


def test_row_staggered_bond_spin_probe_has_rank_two_and_alternating_mirror_parity():
    """Independent dense algebra for the proposed spin-two/pi selector.

    Six-spin row, bit0=down, site0=MSB. No learned state or Schmidt-vector
    convention supplies the operator oracle. This establishes a selection
    rule, not measured brightness of the actual MPS branch.
    """
    from itertools import product
    bits=list(product((0,1),repeat=6));position={row:i for i,row in enumerate(bits)}
    identity=np.eye(2);sp=np.array([[0.,0.],[1.,0.]])
    def local(site):
        op=np.array([[1.]])
        for j in range(6):op=np.kron(op,sp if j==site else identity)
        return op
    plus=[local(j) for j in range(6)]
    q_pi=sum((-1)**c*(plus[c]@plus[(c+1)%6]) for c in range(6))
    for row in (0,1,2):
        reflection=np.zeros((64,64))
        for old,b in enumerate(bits):
            new=[0]*6
            for c in range(6):new[(row-c)%6]=b[c]
            reflection[position[tuple(new)],old]=1.
        np.testing.assert_array_equal(reflection@q_pi@reflection.T,(-1)**(row-1)*q_pi)
    singlet=np.array([0.,-1.,1.,0.])/np.sqrt(2)
    parent=np.kron(np.kron(singlet,singlet),singlet)
    state=q_pi@parent
    assert np.vdot(state,state).real>.1  # Not a kinematic zero on singlets.
    state/=np.linalg.norm(state)
    raising=sum(plus);lowering=raising.T
    sz=np.diag([(sum(b)-3) for b in bits])
    casimir=sz@sz+.5*(raising@lowering+lowering@raising)
    np.testing.assert_allclose(sz@state,2*state,atol=2e-14,rtol=0)  # doubled m=4
    np.testing.assert_allclose(raising@state,0.,atol=2e-14,rtol=0)
    np.testing.assert_allclose(casimir@state,6*state,atol=2e-14,rtol=0)  # doubled q=4
