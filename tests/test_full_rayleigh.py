"""Exact finite-Q two-sided gradient, including reverse-only/null-metric axes."""
import numpy as np
import pytest
from su2_nqs.vmc.full_rayleigh import (reverse_rayleigh_weights, full_rayleigh_gradient,
    full_rayleigh_sr_direction, common_rayleigh_sr_direction)
from su2_nqs.vmc.coherent_tangent import coherent_tangent_diagnostics
from su2_nqs.vmc.importance_tangent import importance_sr_direction


@pytest.mark.parametrize('incomplete', [False, True])
def test_full_gradient_matches_independent_finite_difference_and_gauge(incomplete):
    rng = np.random.default_rng(52)
    a = rng.normal(size=(7,7))+1j*rng.normal(size=(7,7)); h = a+a.conj().T
    p = rng.normal(size=7)+1j*rng.normal(size=7); p[0] = 0
    j = rng.normal(size=(7,3))+1j*rng.normal(size=(7,3))
    j = np.column_stack((j, p, 1j*p))
    rows = np.array([0,2,5,6]) if incomplete else np.arange(7)
    q = np.array([.3,2.,.7,1.1,4.,.8,1.7])[rows] if incomplete else np.ones(7)
    hp, hj = h@p, h@j
    weights = reverse_rayleigh_weights(p[rows], q)
    reverse = np.real(weights@hj[rows])
    terms = full_rayleigh_gradient(p[rows],hp[rows],j[rows],q,reverse)
    def energy(x):
        z = p+j@x; hz = h@z
        return np.real(np.sum(z[rows].conj()*hz[rows]/q)/np.sum(abs(z[rows])**2/q))
    observed = []
    for column in range(j.shape[1]):
        step = np.eye(j.shape[1])[column]*1e-6
        observed.append((energy(step)-energy(-step))/2e-6)
    np.testing.assert_allclose(terms['gradient'], observed, atol=3e-9, rtol=3e-9)
    np.testing.assert_allclose(terms['gradient'][-2:], 0, atol=1e-14)
    if incomplete:
        assert np.linalg.norm(terms['forward']-terms['reverse']) > 1e-2
    else:
        np.testing.assert_allclose(terms['forward'], terms['reverse'], atol=2e-14)
    d, info = full_rayleigh_sr_direction(p[rows],hp[rows],j[rows],q,terms['gradient'],damping=.13)
    t = j[rows]-p[rows,None]*(np.sum(p[rows].conj()[:,None]*j[rows]/q[:,None],axis=0)/np.sum(abs(p[rows])**2/q))[None,:]
    metric = 2*np.real(t.conj().T@(t/q[:,None]))/np.sum(abs(p[rows])**2/q)
    np.testing.assert_allclose(d,np.linalg.solve(metric+.13*np.eye(j.shape[1]),terms['gradient']),atol=2e-13,rtol=2e-13)
    diagnostic = coherent_tangent_diagnostics(p[rows],hp[rows],-j[rows]@d,-hj[rows]@d,q,1)
    assert diagnostic['energy_slope_per_site'] == pytest.approx(-terms['gradient']@d,abs=2e-13)
    assert info['relative_linear_residual'] < 1e-12
    changed = full_rayleigh_gradient(p[rows],hp[rows],j[rows],q*7,reverse)
    np.testing.assert_allclose(changed['gradient'],terms['gradient'],atol=2e-14)
    if not incomplete:
        old,_ = importance_sr_direction(p,hp,j,q,damping=.13)
        np.testing.assert_allclose(old,d,atol=3e-13)


def test_reverse_only_gradient_outside_sampled_tangent_is_not_silently_removed():
    p, hp, j, q = np.array([1.,0]), np.array([0.,1.]), np.array([[0.,0.],[1.,0.]]), np.ones(2)
    g = np.array([2.,3.])
    d, info = full_rayleigh_sr_direction(p,hp,j,q,g,damping=.1)
    np.testing.assert_allclose(d,[2/2.1,30.],atol=1e-13)
    assert info['gradient_outside_sampled_tangent_fraction'] == pytest.approx(3/np.sqrt(13))
    assert info['predicted_total_energy_slope'] < 0


def test_invalid_weights_gradient_and_regularization_fail():
    p,hp,j,q=np.ones(2),np.ones(2),np.ones((2,1)),np.ones(2)
    with pytest.raises(ValueError): reverse_rayleigh_weights(p,q*0)
    with pytest.raises(ValueError): full_rayleigh_gradient(p,hp,j,q,np.ones(2))
    with pytest.raises(ValueError): full_rayleigh_sr_direction(p,hp,j,q,np.ones(1),damping=0)


def test_common_natural_direction_uses_gradients_only_and_descends_each_objective():
    p,hp,j,q=np.array([1.,0.,0.]),np.zeros(3),np.array([[0.,0.],[1.,0.],[0.,1.]]),np.ones(3)
    gradients=np.array([[.5,2.],[1.,0.],[-.2,3.]])
    # Pooled SR descends the average but the second gradient can disagree.
    d,info=common_rayleigh_sr_direction(p,hp,j,q,gradients,damping=.1)
    assert np.all(gradients@d>0)
    assert sum(info['objective_weights'])==pytest.approx(1.)
    assert info['simplex_kkt_scaled_violation']<1e-12
    assert info['weights_use_gradients_only']
    # Independent dense two-dimensional SPD-metric KKT oracle.
    energy_metric=2.1*np.dot(d,d)
    assert np.min(gradients@d)>=energy_metric-1e-13
    np.testing.assert_allclose(d,np.asarray(info['objective_weights'])@gradients/2.1,atol=1e-14)
    # Opposing objectives can have zero common descent, never fabricated win.
    zero,result=common_rayleigh_sr_direction(p,hp,j,q,np.array([[1.,0.],[-1.,0.]]))
    np.testing.assert_allclose(zero,0,atol=1e-14)
    assert not result['common_descent_at_fixed_controls']
