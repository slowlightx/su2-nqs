"""Dense Pauli construction, independent of sparse Heisenberg connectivity."""
import itertools
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from su2_nqs.benchmarks.spin_polynomial_products import build_spin_h2_plan
from su2_nqs.benchmarks.operator_word_jax import evaluate_scalar_word_action_plan_numpy,evaluate_scalar_word_action_plan_jax

jax.config.update('jax_enable_x64',True)


def test_exact_H_squared_complex_nodes_repeated_and_negative_bonds():
    n=4;states=np.array(list(itertools.product((0,1),repeat=n)),dtype=np.int8)
    bonds=[(0,1,1.),(1,0,.3),(1,2,-.7),(2,3,.21),(3,0,.125)]
    sx=np.array([[0,1],[1,0]],complex)/2;sy=np.array([[0,1j],[-1j,0]],complex)/2;sz=np.diag([-.5,.5])
    h=np.zeros((2**n,2**n),complex)
    for a,b,c in bonds:
        for s in (sx,sy,sz):
            term=np.ones((1,1),complex)
            for k in range(n):term=np.kron(term,s if k in (a,b) else np.eye(2))
            h+=c*term
    rng=np.random.default_rng(3041);p=rng.normal(size=2**n)+1j*rng.normal(size=2**n);p[5]=0
    ids=lambda x:np.asarray(x)@(1<<np.arange(n-1,-1,-1))
    plan,work=build_spin_h2_plan(states,bonds)
    got=evaluate_scalar_word_action_plan_numpy(plan,lambda x:p[ids(x)],amplitude_batch_size=7)
    jax_got,_=evaluate_scalar_word_action_plan_jax(plan,lambda x:jnp.asarray(p)[x@jnp.asarray(1<<np.arange(n-1,-1,-1))],amplitude_batch_size=8)
    expected=np.column_stack((p,h@p,h@h@p))
    np.testing.assert_allclose(got,expected,atol=2e-14);np.testing.assert_allclose(jax_got,expected,atol=2e-14)
    assert not work['small_coefficient_clipping'] and len(plan.terminal_configurations)==16
    with pytest.raises(ValueError):build_spin_h2_plan(states,[(0,1,np.nan)])
    with pytest.raises(ValueError):build_spin_h2_plan(states,[(.2,1,1.)])
