"""Budget segmentation, stateless continuation and realized-step controls."""
from dataclasses import replace

import jax
import numpy as np
import pytest

from fnqs.pilots.coordinate_sr import train
from fnqs.pilots.engine import Options, init_model
from fnqs.scripts.run_l32_controls import segments, exclusive_state
from fnqs.variants.coupling_units import CouplingUnits
from fnqs.variants.size_features import SizeFeatureConfig


def test_nested_budget_counts_are_exclusive():
    assert segments([100,400]) == [(100,100),(400,300)]
    for invalid in ([], [400,100], [100,100], [True,400], [0,400], [100.,400]):
        with pytest.raises(ValueError):
            segments(invalid)


def test_budget_snapshots_preserve_the_continuous_optimizer_trajectory():
    cfg=SizeFeatureConfig(2,64,'legacy');units=CouplingUnits(1.,.05)
    p=init_model(jax.random.PRNGKey(301),cfg);key=jax.random.PRNGKey(302)
    schedule=[[[1.,0.,0.,.97],[1.,0.,0.,1.03]]]*4;options=Options(samples=8,learning_rate=2.)
    whole,ws,wk,_=train(p,key,4,schedule,cfg,units,options)
    snapshots={}
    def callback(params,state,rng,history):
        if state['step']==1:
            snapshots.update(params=params,key=rng,state={**state,'work':dict(state['work'])})
    last,ls,lk,_=train(p,key,4,schedule,cfg,units,options,callback)
    first,fs,fk,_=train(p,key,4,schedule[:1],cfg,units,options)
    for name in whole:
        np.testing.assert_allclose(last[name],whole[name],atol=2e-12,rtol=2e-12)
        np.testing.assert_allclose(snapshots['params'][name],first[name],atol=2e-12,rtol=2e-12)
    np.testing.assert_array_equal(lk,wk)
    np.testing.assert_array_equal(snapshots['key'],fk)
    remainder=exclusive_state(ls,snapshots['state']['work'])
    assert remainder['step']==3
    for name in ws['work']:
        assert ws['work'][name] == fs['work'][name]+remainder['work'][name]


def test_norm_cap_can_cancel_nominal_learning_rate_difference():
    cfg=SizeFeatureConfig(2,64,'legacy');units=CouplingUnits(1.,.05)
    p=init_model(jax.random.PRNGKey(303),cfg);key=jax.random.PRNGKey(304)
    options=Options(samples=8,learning_rate=2.,max_step=1e-8)
    schedule=[[[1.,0.,0.,.99]]]
    slow,_,sk,sh=train(p,key,4,schedule,cfg,units,options)
    fast,_,fk,fh=train(p,key,4,schedule,cfg,units,replace(options,learning_rate=8.))
    for name in slow:
        np.testing.assert_allclose(slow[name],fast[name],atol=1e-13,rtol=1e-13)
    np.testing.assert_array_equal(sk,fk)
    assert abs(sh[0]['step_norm']-options.max_step) < 1e-15
    assert abs(fh[0]['step_norm']-options.max_step) < 1e-15
