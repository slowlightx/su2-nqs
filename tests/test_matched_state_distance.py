"""Independent physical-distance and two-candidate common-parent checks."""
import numpy as np
import pytest
from su2_nqs.vmc.matched_state_distance import (
    ratio_distance, match_ray_distance, candidate_contrast, paired_candidate_intervals,
)


def test_phase_ray_hits_exact_physical_distance_with_no_energy_callback():
    fn = lambda scale: 1j*.4*scale*np.array([1, -1])
    r = match_ray_distance(fn, target=.02, bracket_scales=[0, .25, .5, 1, 2, 4],
                           tolerance=1e-8, iterations=30)
    assert r['matched'] and r['error'] <= 1e-8
    assert r['scale'] == pytest.approx(np.arcsin(.02)/.4, abs=3e-8)
    assert ratio_distance(fn(r['scale'])+1000+2j) == pytest.approx(r['distance'], abs=2e-13)
    no = match_ray_distance(lambda x: np.zeros(4), target=.02, bracket_scales=[0, 1, 2])
    assert not no['matched'] and no['evaluations'] == 3
    with pytest.raises(ValueError):
        match_ray_distance(fn, target=.02, bracket_scales=[0, 2, 1])
    with pytest.raises(ValueError):
        match_ray_distance(lambda x: fn(1+x), target=.02, bracket_scales=[0, 1, 2])


def test_two_reweighted_candidates_match_dense_complex_expectations():
    rng = np.random.default_rng(96)
    a = rng.normal(size=(8, 8))+1j*rng.normal(size=(8, 8)); h = (a+a.conj().T)/2
    parent, p, q = [rng.normal(size=8)+1j*rng.normal(size=8) for _ in range(3)]
    weights = abs(parent)**2/np.vdot(parent, parent).real
    lr = [np.log(x/parent) for x in (p, q)]
    es = [h@x/x for x in (p, q)]
    measured = candidate_contrast(lr[0], es[0], lr[1], es[1], n_sites=4, reference_weights=weights)
    moments = []
    for x in (p, q):
        norm = np.vdot(x, x).real; mean = np.vdot(x, h@x)/norm
        var = np.vdot(h@x, h@x).real/norm-abs(mean)**2
        fs = np.sqrt(1-abs(np.vdot(parent, x))**2/(np.vdot(parent, parent).real*norm))
        moments.append(np.array([mean.real/4, var/4, fs]))
    np.testing.assert_allclose(list(measured.values()), moments[1]-moments[0], atol=2e-13)
    shifted = candidate_contrast(lr[0]+1000+1j, es[0], lr[1]-1000+2j, es[1],
                                 n_sites=4, reference_weights=weights)
    np.testing.assert_allclose(list(shifted.values()), list(measured.values()), atol=2e-11)


def test_paired_candidate_bootstrap_retains_chain_dependence_and_sign():
    rng = np.random.default_rng(16); shape = (2, 7, 6)
    ra, rb = [.1*rng.normal(size=shape)+.03j*rng.normal(size=shape) for _ in range(2)]
    ea, eb = [rng.normal(size=shape)+.1j*rng.normal(size=shape) for _ in range(2)]
    kw = dict(n_sites=4, seed=32, replicates=128, family_tests=24)
    a = paired_candidate_intervals(ra, ea, rb, eb, **kw)
    repeated = paired_candidate_intervals(*(np.repeat(x, 2, axis=1) for x in (ra, ea, rb, eb)), **kw)
    reverse = paired_candidate_intervals(rb, eb, ra, ea, **kw)
    for k in a:
        for field in a[k]:
            np.testing.assert_allclose(a[k][field], repeated[k][field], atol=1e-13)
        assert reverse[k]['difference'] == pytest.approx(-a[k]['difference'], abs=1e-13)
        np.testing.assert_allclose(reverse[k]['family_adjusted_interval'], -np.array(a[k]['family_adjusted_interval'])[::-1], atol=1e-13)
    same = paired_candidate_intervals(ra, ea, ra, ea, **kw)
    for k in same:
        assert same[k]['difference'] == same[k]['bootstrap_se'] == 0
