"""Independent dense expansion and exact zero-mean correction numerators."""
import numpy as np
import pytest
from su2_nqs.vmc.anchored_moments import anchored_moments


def test_exhaustive_complex_nodes_H_shift_and_missing_quadrature():
    rng=np.random.default_rng(3071);a=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8));h=a+a.conj().T
    c=rng.normal(size=8)+1j*rng.normal(size=8);c[0]=0
    d=.02*(rng.normal(size=8)+1j*rng.normal(size=8));p=c+d
    hc,h2,hp=h@c,h@h@c,h@p
    got=anchored_moments(c,hc,h2,p,hp,np.ones(8),8)
    e=np.vdot(p,hp)/np.vdot(p,p);v=(np.vdot(hp,hp)/np.vdot(p,p)-abs(e)**2).real/8
    assert got['energy_per_site']==pytest.approx(e.real/8,abs=2e-14)
    assert got['variance_per_site']==pytest.approx(v,abs=2e-13)
    assert abs(got['energy_correction_per_site'])<2e-14
    assert abs(got['second_moment_correction_total'])<2e-13
    # Incomplete Q changes the estimator, NOT an exact empirical H identity.
    ids=np.array([0,1,3,7]);q=np.linspace(.4,1.4,4)
    part=anchored_moments(c[ids],hc[ids],h2[ids],p[ids],hp[ids],q,8)
    assert abs(part['energy_correction_per_site'])>1e-4
    assert abs(part['second_moment_correction_total'])>1e-3
    shifted=anchored_moments(3j*c[ids],3j*(hc+2*c)[ids],3j*(h2+4*hc+4*c)[ids],3j*p[ids],3j*(hp+2*p)[ids],11*q,8)
    assert shifted['energy_per_site']==pytest.approx(part['energy_per_site']+.25,abs=2e-13)
    assert shifted['variance_per_site']==pytest.approx(part['variance_per_site'],abs=3e-13)
    # Any exhaustive proposal weighting has exact-zero EXPECTATION of each
    # raw correction, before the ratio. Do not claim finite ratio unbiasedness.
    prob=np.arange(1,9,dtype=float);prob/=prob.sum()
    ec=np.real(d.conj()*hc-c.conj()*(h@d))/prob
    mc=2*np.real(d.conj()*h2-hc.conj()*(h@d))/prob
    assert np.dot(prob,ec)==pytest.approx(0.,abs=2e-13)
    assert np.dot(prob,mc)==pytest.approx(0.,abs=3e-13)
    # Do not use this readout as an unconstrained training objective: an
    # unchanged physical ray with a different relative anchor phase changes
    # its finite-Q correction. Exact support still removes this defect.
    phase=np.exp(.7j)
    parent=anchored_moments(c[ids],hc[ids],h2[ids],c[ids],hc[ids],q,8)
    same_ray=anchored_moments(c[ids],hc[ids],h2[ids],phase*c[ids],phase*hc[ids],q,8)
    assert same_ray['raw']['variance_per_site']==pytest.approx(parent['raw']['variance_per_site'],abs=2e-13)
    assert abs(same_ray['variance_per_site']-parent['variance_per_site'])>1e-3
    with pytest.raises(ValueError):anchored_moments(c,hc,h2[:-1],p,hp,np.ones(8),8)
