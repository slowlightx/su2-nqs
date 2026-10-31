import jax.numpy as jnp
import numpy as np
import pytest

from fnqs import training


def test_residual_refinement_recovers_independent_dense_direction(monkeypatch):
    rng = np.random.default_rng(7)
    a = rng.normal(size=(48,20))
    target = rng.normal(size=48)
    shift = .01
    operator = a.T@a + shift*np.eye(20)
    force = a.T@target
    reference = np.linalg.solve(operator,force)
    inaccurate = reference + rng.normal(size=20)*.01
    relative = np.linalg.norm(operator@inaccurate-force)/np.linalg.norm(force)
    monkeypatch.setattr(training,"_solve_sr_once",lambda *args,**kwargs:
                        (jnp.asarray(inaccurate),jnp.asarray(relative),jnp.asarray(2*np.linalg.norm(force))))
    direction,residual,_,repairs = training._solve_sr_refined(
        jnp.asarray(a),jnp.asarray(target),shift,solver="cg",tolerance=1e-11,maxiter=200)
    assert repairs == 1
    assert float(residual) < 1e-10
    np.testing.assert_allclose(direction,reference,atol=4e-10,rtol=0)


def test_successful_cg_direction_is_returned_without_changes():
    rng=np.random.default_rng(31)
    a=jnp.asarray(rng.normal(size=(40,12)));target=jnp.asarray(rng.normal(size=40))
    expected=training._solve_sr_once(a,target,.01,solver="cg",tolerance=1e-10,maxiter=200)
    actual=training._solve_sr_refined(a,target,.01,solver="cg",tolerance=1e-10,maxiter=200)
    assert actual[3] == 0
    for first,second in zip(expected,actual[:3]):np.testing.assert_array_equal(first,second)


def test_unsolved_correction_keeps_original_failure_guard(monkeypatch):
    from fnqs.hamiltonians import IsingCouplings
    from fnqs.models import GRUConfig,init_model
    import jax
    model=GRUConfig(hidden_dim=1)
    params=init_model(jax.random.PRNGKey(7),model)
    calls=[]
    monkeypatch.setattr(training,"_solve_sr_once",lambda a,*args,**kwargs:
                        (jnp.ones(a.shape[1]),jnp.asarray(1.),jnp.asarray(1.)))
    def no_correction(a,rhs,*args):
        calls.append(1)
        return jnp.zeros(a.shape[1])
    monkeypatch.setattr(training,"_sr_cg_correction",no_correction)
    with pytest.raises(FloatingPointError,match="did not converge"):
        training.ensemble_sr_direction(params,[training.Task(4,IsingCouplings())],
            [jax.random.PRNGKey(1)],model,"exact",16,solver="cg")
    assert len(calls) == 3
