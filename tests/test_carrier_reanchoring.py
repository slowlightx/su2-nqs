"""Independent dense-H reanchoring, including carrier nodes and gauges."""
import numpy as np
import pytest
from su2_nqs.vmc.carrier_reanchoring import polynomial_carrier_products, carrier_span_diagnostics


def test_polynomial_parent_and_H_product_match_independent_dense_H():
    rng=np.random.default_rng(51);a=rng.normal(size=(7,7))+1j*rng.normal(size=(7,7))
    h=(a+a.conj().T)/2;c=rng.normal(size=7)+1j*rng.normal(size=7);c[0]=0
    powers=np.column_stack((c,h@c,h@h@c,h@h@h@c))
    for coeff in (np.array([.916,-.0045]),np.array([1.,.2+.1j,-.03])):
        p,hp=polynomial_carrier_products(powers,coeff)
        expected=sum(v*np.linalg.matrix_power(h,i)@c for i,v in enumerate(coeff))
        np.testing.assert_allclose(p,expected,atol=2e-14,rtol=2e-14)
        np.testing.assert_allclose(hp,h@expected,atol=3e-14,rtol=3e-14)
    assert abs(polynomial_carrier_products(powers,[.916,-.0045])[0][0])>0
    with pytest.raises(ValueError):polynomial_carrier_products(powers[:,:2],[1.,.2])
    with pytest.raises(ValueError):polynomial_carrier_products(powers,[0.,0.])


def test_span_diagnostic_independent_direction_and_gauge_covariance():
    rng=np.random.default_rng(5);h=np.diag(np.arange(8,dtype=float));c=rng.normal(size=8)+1j*rng.normal(size=8)
    b=np.column_stack((c,h@c));hb=h@b;p,hp=polynomial_carrier_products(np.column_stack((c,h@c,h@h@c)),[.9,-.04])
    q=np.linspace(.4,2.,8);d=rng.normal(size=8)+1j*rng.normal(size=8);hd=h@d
    row=carrier_span_diagnostics(p,hp,d,hd,b,hb,q,8)
    assert row['centered_carrier_rank']==1 and 0<row['outside_carrier_span_fraction']<1
    phase=np.exp(.31j)
    other=carrier_span_diagnostics(phase*p,phase*hp,phase*(d+(.2+.7j)*p),phase*(hd+(.2+.7j)*hp),phase*b,phase*hb,12*q,8)
    np.testing.assert_allclose(row['outside_carrier_span_fraction'],other['outside_carrier_span_fraction'],atol=3e-14)
    inside=carrier_span_diagnostics(p,hp,b@np.array([.3,.1j]),hb@np.array([.3,.1j]),b,hb,q,8)
    assert inside['outside_carrier_span_fraction']<1e-26
    assert row['actual_direction_unchanged'] and row['no_energy_gain_guaranteed_by_linear_independence']


def test_continuation_positive_filter_ratio_certifies_energy_not_variance():
    # Spectral calculation independent of graph kernels and fitted samples.
    energies=np.linspace(-7.,7.,21);weights=np.linspace(.2,2.,21);anchor=-2.
    beta0=.5/(7-anchor);beta1=.75/(7-anchor)
    f0=1+beta0*(anchor-energies);f1=1+beta1*(anchor-energies)
    assert f1.min()>=.25 and np.all(np.diff(f1/f0)<0)
    old=weights*f0**2;new=weights*f1**2
    e0=np.dot(old,energies)/old.sum();e1=np.dot(new,energies)/new.sum()
    assert e1<e0
