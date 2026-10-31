"""Analytic same-group covariance/ratio oracles; no wavefunction queries."""
import numpy as np
import pytest

from projects.clebsch_tree_nqs.explorations.ct_A1_frozen_cache_postprocess_20261028 import paired_statistics


def test_proportional_observables_have_zero_conditional_ratio_error():
    denominator=np.repeat(np.linspace(.1,.8,32),16)
    numerator=.3*denominator
    report=paired_statistics(np.stack([numerator,denominator],axis=1).astype(complex),
        np.ones(512),('a','b'),dict(ratio=('a','b')))
    ratio=report['conditional_ratios']['ratio']
    np.testing.assert_allclose(ratio['value'],.3,atol=2e-15)
    assert ratio['paired_group_jackknife_SE']<2e-14
    assert report['observables']['a']['real_group_jackknife_SE']>1e-3


def test_paired_difference_error_matches_group_sample_variance():
    a=np.linspace(.1,.8,32)
    b=.8*a+.02*np.cos(np.arange(32))
    values=np.repeat(np.stack([a,b,a-b],axis=1),16,axis=0)
    report=paired_statistics(values.astype(complex),np.ones(512),('a','b','delta'),{})
    expected=np.std(a-b,ddof=1)/np.sqrt(32)
    np.testing.assert_allclose(report['observables']['delta']['real_group_jackknife_SE'],expected,atol=2e-15)
    covariance=np.asarray(report['linear_same_group_covariance'])
    np.testing.assert_allclose(covariance[2,2],covariance[0,0]+covariance[1,1]-2*covariance[0,1],atol=2e-15)


def test_vanishing_delete_group_norm_is_rejected_not_floored():
    mass=np.zeros(512);mass[:16]=1
    with pytest.raises(ValueError,match='vanished'):
        paired_statistics(np.ones((512,1),complex),mass,('a',),{})


def test_nonpositive_sector_denominator_withholds_ratio():
    report=paired_statistics(np.stack([np.ones(512),-np.ones(512)],axis=1).astype(complex),
        np.ones(512),('a','b'),dict(ratio=('a','b')))
    assert report['conditional_ratios']['ratio']['withheld']
