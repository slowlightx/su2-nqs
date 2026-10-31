"""Exact two-state controls: reject high-energy/gauge changes, keep nodes."""
import numpy as np
import pytest

from su2_nqs.vmc.carrier_protection import CarrierProtection, physical_moments, carrier_quality_gate


def _data(vector, matrix, q=(.3,2.)):
    # Four exact repetitions with fixed positive, unnormalized Q quadrature.
    q=np.tile(q,4);p=np.tile(vector,4)*np.sqrt(q)
    hp=np.tile(matrix@np.asarray(vector),4)*np.sqrt(q)
    return p,hp,q,np.repeat(np.arange(4),2)


def test_node_safe_moments_match_independent_normalized_dense_H():
    h=np.array([[0.,1.],[1.,2.]])
    p,hp,q,g=_data(np.array([1.,0.]),h)
    m=physical_moments(p,hp,q,1)
    assert m['energy_per_site']==0 and m['variance_per_site']==pytest.approx(1.,abs=2e-15)
    assert m['exact_node_samples']==4 and np.all(hp[p==0]!=0)
    for scale in (7.3, .02):
        mm=physical_moments(p,hp,scale*q,1)
        assert mm['energy_per_site']==pytest.approx(m['energy_per_site'])
        assert mm['variance_per_site']==pytest.approx(m['variance_per_site'])


def test_small_descent_passes_and_high_energy_singlet_analogue_is_rejected():
    h=np.array([[0.,1.],[1.,2.]])
    c,hc,q,g=_data(np.array([1.,0.]),h)
    p,hp,_,_=_data(np.array([1.,-.005]),h)
    concentrated=carrier_quality_gate(c,hc,c,hc,p,hp,q,g,1,CarrierProtection())
    assert not concentrated['checks']['importance_concentration']
    # Half of this exact two-state quadrature consists of carrier nodes.
    # Declare the toy ESS bound explicitly; the production .6 bound is unchanged.
    r=carrier_quality_gate(c,hc,c,hc,p,hp,q,g,1,CarrierProtection(minimum_ess_fraction=.4))
    assert r['accepted'] and r['step_distance']['fs']==pytest.approx(np.arctan(.005),abs=1e-13)
    assert r['candidate']['energy_per_site']==pytest.approx((-2*.005+2*.005**2)/(1+.005**2),abs=2e-15)
    assert r['candidate']['variance_per_site']<r['carrier']['variance_per_site']
    bad,hbad,_,_=_data(np.array([1.,.005]),h)
    rejected=carrier_quality_gate(c,hc,c,hc,bad,hbad,q,g,1,CarrierProtection(minimum_ess_fraction=.4))
    assert not rejected['accepted'] and 'energy_vs_carrier' in rejected['rejected_by']
    assert rejected['checks']['step_distance']  # small distance alone is insufficient


def test_cumulative_radius_and_variance_fail_independently_of_step_radius():
    h=np.array([[0.,1.],[1.,2.]])
    c,hc,q,g=_data(np.array([1.,0.]),h)
    parent,hparent,_,_=_data(np.array([1.,-.039]),h)
    p,hp,_,_=_data(np.array([1.,-.045]),h)
    r=carrier_quality_gate(c,hc,parent,hparent,p,hp,q,g,1,CarrierProtection(minimum_ess_fraction=.4))
    assert r['checks']['step_distance'] and r['checks']['energy_vs_parent']
    assert not r['checks']['carrier_distance'] and not r['accepted']
    h=np.diag([0.,100.]);c,hc,q,g=_data(np.array([1.,0.]),h)
    p,hp,_,_=_data(np.array([1.,.005]),h)
    r=carrier_quality_gate(c,hc,c,hc,p,hp,q,g,1,CarrierProtection(energy_tolerance_per_site=1.))
    assert r['checks']['energy_vs_carrier'] and not r['checks']['variance_envelope']


def test_gauge_is_not_counted_as_learning_and_invalid_Q_fails_closed():
    h=np.array([[0.,1.],[1.,2.]])
    c,hc,q,g=_data(np.array([1.,0.]),h)
    r=carrier_quality_gate(c,hc,c,hc,2j*c,2j*hc,q,g,1,CarrierProtection())
    assert not r['checks']['nonzero_update'] and not r['accepted']
    with pytest.raises(ValueError): physical_moments(c,hc,q*0,1)
    with pytest.raises(ValueError): physical_moments(c,hc,q,True)
    with pytest.raises(ValueError): CarrierProtection(maximum_carrier_fs=.001)
    with pytest.raises(ValueError): carrier_quality_gate(c,hc,c,hc,c,hc,q,g*0,1,CarrierProtection())
