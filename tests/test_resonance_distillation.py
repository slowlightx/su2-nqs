"""Independent fixed-quadrature and complete-table distillation checks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.vmc.resonance_distillation import (
    table_distillation_metrics, gauge_invariant_table_loss, physical_distillation_metrics,
)

jax.config.update('jax_enable_x64', True)


def _fixture():
    rng = np.random.default_rng(16001)
    raw = rng.normal(size=(5, 5)) + 1j*rng.normal(size=(5, 5))
    h = (raw+raw.conj().T)/2
    c = np.array([1., 0., .3j, -.2, 0.])
    dt = np.array([.1j, .7, -.4+.3j, .2, -.3j])
    dp = dt+np.array([.03, -.04j, .02, .1j, .01])
    t, p = c+.1*dt, c+.1*dp
    q = np.array([.2, 1.2, .4, .7, .3])
    return p, h@p, t, h@t, c, h@c, q, h


def test_table_loss_removes_only_common_identity_and_matches_independent_weights():
    rng = np.random.default_rng(16002)
    t, p = rng.normal(size=(6, 6, 10)), rng.normal(size=(6, 6, 10))
    identity = np.zeros((6, 6, 10)); identity[:, :, [0, 1, 3, 4, 5, 9]] = 1
    result = table_distillation_metrics(p, t)
    for candidate_shift, target_shift in ((.7, 0.), (0., -.4), (1.2, -.6)):
        shifted = table_distillation_metrics(p+candidate_shift*identity, t+target_shift*identity)
        np.testing.assert_allclose(shifted.relative_mse, result.relative_mse, atol=5e-15)
        np.testing.assert_allclose(shifted.hs_relative_mse, result.hs_relative_mse, atol=5e-15)
    row = np.array([1, 1, 3, 3, 3, 5]); col = np.array([1, 1, 1, 3, 3, 3, 3, 3, 3, 5])
    weights = row[:, None, None]*row[None, :, None]*col
    assert np.sum(weights*identity**2) == 4096
    project = lambda x:x-np.sum(weights*x*identity)/np.sum(weights*identity**2)*identity
    expected = np.sum(weights*project(p-t)**2)/np.sum(weights*project(t)**2)
    np.testing.assert_allclose(result.hs_relative_mse, expected, atol=3e-15)
    # A context-dependent trace changes the physical state and is retained.
    context_shift = np.arange(36).reshape(6, 6, 1)*identity*.01
    assert float(gauge_invariant_table_loss(t+context_shift, t)) > 1e-4
    np.testing.assert_allclose(gauge_invariant_table_loss(t+.3*identity, t), 0., atol=1e-30)


def test_physical_losses_have_independent_candidate_teacher_and_carrier_ray_invariance():
    p, hp, t, ht, c, hc, q, _ = _fixture()
    expected = physical_distillation_metrics(p, hp, t, ht, c, hc, q)
    for sp, st, sc in ((.03+.07j, 1., 1.), (1., -2+.7j, 1.), (1., 1., -.3j), (2j, -.8, .03+.04j)):
        observed = physical_distillation_metrics(sp*p, sp*hp, st*t, st*ht, sc*c, sc*hc, q)
        for name in ('ray_infidelity', 'innovation_relative_mse', 'h_response_relative_mse',
                     'innovation_absolute_mse', 'h_response_absolute_mse',
                     'candidate_carrier_fidelity', 'target_carrier_fidelity', 'energy_shift'):
            np.testing.assert_allclose(getattr(observed, name), getattr(expected, name), atol=4e-14)
    changed_q = physical_distillation_metrics(p, hp, t, ht, c, hc, q*129.)
    np.testing.assert_allclose(changed_q[:7], expected[:7], atol=4e-14)
    changed_h = physical_distillation_metrics(p, hp/36, t, ht/36, c, hc/36, q)
    np.testing.assert_allclose(changed_h.h_response_relative_mse, expected.h_response_relative_mse, atol=4e-14)
    np.testing.assert_allclose(changed_h.h_response_absolute_mse, expected.h_response_absolute_mse/36**2, atol=1e-16)


def test_carrier_dominated_total_fidelity_cannot_hide_missing_innovation():
    c = np.array([1., 0., 0.]); t = c+np.array([0., 1e-4, 0.]); h = np.diag([0., 1., 2.])
    value = physical_distillation_metrics(c, h@c, t, h@t, c, h@c, np.ones(3))
    assert value.ray_infidelity < 1.1e-8
    np.testing.assert_allclose(value.innovation_relative_mse, 1., atol=1e-15)
    np.testing.assert_allclose(value.h_response_relative_mse, 1., atol=1e-15)
    assert value.chart_valid and value.innovation_defined and value.h_response_defined


def test_global_chart_keeps_configuration_nodes_and_matches_direct_formulas():
    p, hp, t, ht, c, hc, q, _ = _fixture()
    assert np.count_nonzero(c == 0) == 2 and np.any(abs(hc[c == 0]) > .1)
    p = p.copy(); p[1] = 0.  # Global chart remains valid at this candidate node.
    hp = hp.copy(); hp[1] = .3+.2j
    result = physical_distillation_metrics(p, hp, t, ht, c, hc, q)
    inner = lambda x,y:np.sum(x.conj()*y/q)
    norm = lambda x:inner(x,x).real
    ap, at = inner(c,p)/norm(c), inner(c,t)/norm(c)
    dp, dt = p/ap-c, t/at-c
    shift = inner(t,ht).real/norm(t)
    rp, rt = hp/ap-hc-shift*dp, ht/at-hc-shift*dt
    np.testing.assert_allclose(result.innovation_relative_mse, norm(dp-dt)/norm(dt), atol=2e-14)
    np.testing.assert_allclose(result.h_response_relative_mse, norm(rp-rt)/norm(rt), atol=2e-14)
    assert np.isfinite(np.asarray(result[:7])).all()


def test_candidate_gradient_has_both_state_and_H_response_dependence():
    p, hp, t, ht, c, hc, q, h = _fixture()
    directions = np.random.default_rng(16003).normal(size=(5, 2))*.01 + 1j*np.random.default_rng(16004).normal(size=(5, 2))*.01
    def loss(theta):
        candidate = jnp.asarray(p)+jnp.asarray(directions)@theta
        result = physical_distillation_metrics(candidate, jnp.asarray(h)@candidate, t, ht, c, hc, q)
        return result.innovation_relative_mse+result.h_response_relative_mse+result.ray_infidelity
    theta = jnp.array([.2, -.1]); derivative = np.asarray(jax.grad(loss)(theta)); epsilon = 1e-5
    finite = np.array([float((loss(theta+epsilon*jnp.eye(2)[k])-loss(theta-epsilon*jnp.eye(2)[k]))/(2*epsilon)) for k in range(2)])
    np.testing.assert_allclose(derivative, finite, atol=3e-9, rtol=1e-8)
    print('canonical/H-response distillation FD error', np.max(abs(derivative-finite)))
    target_gradient = jax.grad(lambda teacher: physical_distillation_metrics(p, hp, teacher, ht, c, hc, q).innovation_relative_mse)(jnp.asarray(t))
    np.testing.assert_array_equal(target_gradient, np.zeros(5, complex))
    q_gradient = jax.grad(lambda qq: physical_distillation_metrics(p, hp, t, ht, c, hc, qq).innovation_relative_mse)(jnp.asarray(q))
    np.testing.assert_array_equal(q_gradient, np.zeros(5))


def test_table_gradient_and_global_identity_direction_are_correct():
    rng = np.random.default_rng(16005); t = rng.normal(size=(6,6,10)); p = rng.normal(size=(6,6,10)); d = rng.normal(size=(6,6,10))
    exact = float(jax.jvp(lambda x:gauge_invariant_table_loss(x,t), (jnp.asarray(p),), (jnp.asarray(d),))[1])
    epsilon = 1e-5
    finite = float((gauge_invariant_table_loss(p+epsilon*d,t)-gauge_invariant_table_loss(p-epsilon*d,t))/(2*epsilon))
    np.testing.assert_allclose(exact, finite, atol=2e-10)
    identity = np.zeros((6,6,10)); identity[:,:,[0,1,3,4,5,9]] = 1
    np.testing.assert_allclose(jnp.sum(jax.grad(gauge_invariant_table_loss)(jnp.asarray(p),jnp.asarray(t))*identity), 0., atol=3e-16)


def test_zero_signal_and_orthogonal_chart_are_explicitly_undefined():
    c = np.array([1., 0.]); t = np.array([1., .1]); q = np.ones(2)
    no_signal = physical_distillation_metrics(c, c, c, c, c, c, q)
    assert no_signal.chart_valid and not no_signal.innovation_defined and not no_signal.h_response_defined
    assert np.isnan(no_signal.innovation_relative_mse) and np.isnan(no_signal.h_response_relative_mse)
    assert no_signal.ray_infidelity == 0
    orthogonal = physical_distillation_metrics(np.array([0.,1.]), np.array([0.,1.]), t, t, c, c, q)
    assert not orthogonal.chart_valid and np.isnan(orthogonal.innovation_relative_mse)
    assert np.isfinite(orthogonal.ray_infidelity)
    zero_h = physical_distillation_metrics(t, t*0, t, t*0, c, c*0, q)
    assert zero_h.innovation_defined and not zero_h.h_response_defined
    scalar = np.zeros((6,6,10)); scalar[:,:,[0,1,3,4,5,9]]=2
    trivial = table_distillation_metrics(scalar, scalar)
    assert not trivial.defined and np.isnan(trivial.relative_mse) and trivial.absolute_mse == 0


def test_invalid_data_fail_closed():
    p,hp,t,ht,c,hc,q,_ = _fixture()
    for bad in (-q,q*0,q.astype(complex)):
        with pytest.raises(ValueError):
            physical_distillation_metrics(p,hp,t,ht,c,hc,bad)
    with pytest.raises(ValueError):
        physical_distillation_metrics(p,hp,t[:-1],ht,c,hc,q)
    with pytest.raises(ValueError):
        table_distillation_metrics(np.zeros(360),np.zeros((6,6,10)))
    with pytest.raises(ValueError):
        physical_distillation_metrics(p,hp,t,ht,c,hc,q,minimum_carrier_fidelity=1.)
    value = jax.jit(physical_distillation_metrics)(p*0,hp,t,ht,c,hc,q)
    assert not value.chart_valid and np.isnan(value.ray_infidelity)
