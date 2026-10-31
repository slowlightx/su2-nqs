"""Independent dense objective/gradient and generic SR checks (float64)."""
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest
from fnqs.basis import all_bits
from fnqs.hamiltonians import dense_hamiltonian,IsingCouplings
from fnqs.pilots.engine import Options,penalty,response_loss,solve,sampled_block,init_model
from fnqs.pilots.measurements import action,exact_audit,reference
from fnqs.variants.size_features import SizeFeatureConfig,log_psi


@pytest.mark.parametrize("n",[4,6,8,10])
def test_vector_action_against_dense(n):
    rng=np.random.default_rng(n)
    v=rng.normal(size=1<<n)+1j*rng.normal(size=1<<n)
    for g in np.eye(4):
        np.testing.assert_allclose(action(v,n,g),dense_hamiltonian(n,IsingCouplings(*g))@v,atol=2e-11,rtol=1e-13)


def test_response_penalty_and_born_gradient_against_dense_finite_difference():
    cfg=SizeFeatureConfig(3)
    p=init_model(jax.random.PRNGKey(23),cfg)
    bits=jnp.asarray(all_bits(4)); g=jnp.array([1.,.1,.07,1.03])
    flat,unravel=ravel_pytree(p)
    proposal=log_psi(p,bits,g,cfg)
    h=dense_hamiltonian(4,IsingCouplings(*g))
    v=dense_hamiltonian(4,IsingCouplings(0,0,0,1))
    def independent(f):
        params=unravel(jnp.asarray(f))
        psi=np.exp(np.asarray(log_psi(params,bits,g,cfg)))
        delta=1e-5
        s=(np.asarray(log_psi(params,bits,g+jnp.array([0,0,0,delta]),cfg))-
           np.asarray(log_psi(params,bits,g-jnp.array([0,0,0,delta]),cfg)))/(2*delta)
        weights=psi**2
        weights/=weights.sum()
        energy=weights@(h@psi/psi)
        tau=psi*(s-weights@s)
        r=(h-energy*np.eye(len(psi)))@tau+v@psi
        r-=psi*(psi@r)
        source=v@psi-psi*(psi@(v@psi))
        return (r@r)/(source@source+1e-12)
    # Exhaustive importance sum requires proposal probabilities, not uniform rows.
    from fnqs.diagnostics.response import response_records
    def objective(f):
        params=unravel(f)
        rows=response_records(params,bits,g,lambda p,b,x:log_psi(p,b,x,cfg))
        return response_loss(rows,jnp.exp(2*log_psi(params,bits,g,cfg)))
    observed=float(objective(flat))
    np.testing.assert_allclose(observed,independent(np.asarray(flat)),atol=2e-9,rtol=2e-8)
    direction=np.random.default_rng(9).normal(size=len(flat)); direction/=np.linalg.norm(direction)
    grad=np.asarray(jax.grad(objective)(flat))
    delta=2e-4
    finite=(independent(np.asarray(flat)+delta*direction)-independent(np.asarray(flat)-delta*direction))/(2*delta)
    np.testing.assert_allclose(grad@direction,finite,atol=2e-7,rtol=2e-5)
    # The sampled importance estimator gradient is checked with a frozen proposal.
    def sampled_independent(f):
        params=unravel(jnp.asarray(f)); logs=np.asarray(log_psi(params,bits,g,cfg))
        psi=np.exp(logs); weights=np.exp(2*(logs-np.asarray(proposal))); weights/=weights.sum()
        s=np.asarray(jax.jvp(lambda x:log_psi(params,bits,x,cfg),(g,),(jnp.array([0.,0.,0.,1.]),))[1])
        e=h@psi/psi; hs=h@(psi*s)/psi; source=v@psi/psi
        mean=lambda x:weights@x
        r=hs-mean(s)*e-mean(e)*(s-mean(s))+source; r-=mean(r)
        return mean(r*r)/(mean((source-mean(source))**2)+1e-12)
    grad=np.asarray(jax.grad(lambda f:penalty(unravel(f),bits,g,proposal,cfg))(flat))
    finite=(sampled_independent(np.asarray(flat)+delta*direction)-sampled_independent(np.asarray(flat)-delta*direction))/(2*delta)
    np.testing.assert_allclose(grad@direction,finite,atol=2e-7,rtol=2e-5)


def test_arbitrary_force_sr_against_independent_dense():
    rng=np.random.default_rng(12); a=rng.normal(size=(7,23)); rhs=rng.normal(size=23)
    options=Options(shift=.01,tolerance=1e-10)
    got,rho,_=solve(jnp.asarray(a),jnp.asarray(rhs),options)
    expected=np.linalg.solve(a.T@a+.01*np.eye(23),rhs)
    np.testing.assert_allclose(got,expected,atol=1e-6,rtol=1e-8)
    assert rho<1e-6


def test_energy_only_force_and_normalization():
    cfg=SizeFeatureConfig(3); p=init_model(jax.random.PRNGKey(5),cfg)
    bits=jnp.asarray(all_bits(4)); g=jnp.array([1.,.1,.03,1.])
    flat,unravel=ravel_pytree(p)
    a,force,_,_=sampled_block(p,bits,g,cfg,Options(samples=16))
    scores=np.asarray(jax.jacrev(lambda f:log_psi(unravel(f),bits,g,cfg))(flat))
    psi=np.exp(np.asarray(log_psi(p,bits,g,cfg)))
    e=dense_hamiltonian(4,IsingCouplings(*g))@psi/psi
    expected=(scores-scores.mean(axis=0)).T@(e-e.mean())/64
    np.testing.assert_allclose(force,expected,atol=1e-12)
    np.testing.assert_allclose(a,np.asarray(scores-scores.mean(axis=0))/4,atol=1e-12)
    ref=reference(4,list(g)); audit=exact_audit(p,4,list(g),cfg,ref)
    assert audit['normalization_error']<1e-12
    assert ref['response_relative_residual']<1e-8


@pytest.mark.parametrize("kwargs",[{"samples":0},{"shift":0},{"response_weight":-1},{"max_step":float('nan')}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):Options(**kwargs)
