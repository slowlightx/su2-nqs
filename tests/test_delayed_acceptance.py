"""Independent dense Markov and sampling oracles for exact cheap screening."""
import numpy as np
import pytest
from su2_nqs.vmc.delayed_acceptance import delayed_log_acceptance,screened_metropolis


def test_nonuniform_proposal_detailed_balance_and_stationary_target():
    rng=np.random.default_rng(123);p=rng.uniform(.1,2,7);p/=p.sum()
    cheap=rng.uniform(.05,3,7);T=rng.uniform(.1,2,(7,7));T/=T.sum(1,keepdims=True)
    kernel=np.zeros_like(T)
    for i in range(7):
        for j in range(7):
            if i==j:continue
            a,b=delayed_log_acceptance(np.log(cheap[j]/cheap[i]),np.log(p[j]/p[i]),np.log(T[j,i]/T[i,j]))
            kernel[i,j]=T[i,j]*np.exp(a+b)
        kernel[i,i]=1-kernel[i].sum()
    np.testing.assert_allclose(p[:,None]*kernel,(p[:,None]*kernel).T,atol=2e-16)
    np.testing.assert_allclose(p@kernel,p,atol=2e-16)


def test_equal_surrogate_target_recovers_metropolis():
    a=np.array([-3.,-.1,0.,.7,4.]);r=np.array([.2,-1.,.5,-.3,2.])
    first,second=delayed_log_acceptance(a,a,r)
    np.testing.assert_array_equal(first,np.minimum(0,a+r));np.testing.assert_array_equal(second,0.)


def test_callback_queries_only_survivors_and_rng_restore_is_exact():
    r=np.random.default_rng(918);state=r.bit_generator.state;old=np.zeros(32);cheap=np.zeros(32);trial=np.full(32,-1.)
    calls=[]
    def target(ids):calls.append(ids.copy());return np.full(len(ids),-.5)
    one=screened_metropolis(old,cheap,trial,np.zeros(32),target,r)
    assert 0<len(calls[0])<32 and one[2]['full_target_queries']==len(calls[0])
    nxt=r.bit_generator.state;r.bit_generator.state=state;two=screened_metropolis(old,cheap,trial,np.zeros(32),target,r)
    np.testing.assert_array_equal(one[0],two[0]);np.testing.assert_array_equal(one[1],two[1]);assert r.bit_generator.state==nxt


def test_zero_survivors_never_query_target():
    def forbidden(ids):raise AssertionError('no full query expected')
    result=screened_metropolis(np.zeros(4),np.zeros(4),np.full(4,-1000.),np.zeros(4),forbidden,np.random.default_rng(7))
    assert result[2]['full_target_queries']==0 and not result[0].any()


@pytest.mark.parametrize('bad',[np.inf,np.nan,-np.inf])
def test_nonfinite_density_ratios_rejected(bad):
    with pytest.raises(ValueError):delayed_log_acceptance(bad,0.,0.)
