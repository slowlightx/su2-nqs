"""Protocol/cost split and unchanged-state parameter-direction semantics."""
import json

import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_finite_schur_protected import CONFIG, interpolate, registration
from su2_nqs.vmc.carrier_protection import CarrierProtection, physical_distance


def test_protected_protocol_has_fixed_matched_budgets_and_fresh_audit():
    spec=json.loads(CONFIG.read_text());reg=registration(spec)
    assert spec['steps']*spec['chains']*spec['records_per_step']==256
    assert spec['matching_bisections']+1==13
    assert spec['audit']['chains']*spec['audit']['records']==128
    assert spec['maximum_q']==4 and spec['multiplicity_width']==2
    assert reg['config']['protection']==spec['protection']
    bad={**spec,'seeds':{**spec['seeds'],'audit':2026100114}}
    with pytest.raises(ValueError,match='fresh'):registration(bad)


def test_step_interpolation_does_not_create_another_parameter_or_change_direction():
    old={'a':jnp.asarray([1.,2.]),'b':(jnp.asarray([3.]),)}
    new={'a':jnp.asarray([-1.,4.]),'b':(jnp.asarray([5.]),)}
    half=interpolate(old,new,.5)
    np.testing.assert_array_equal(half['a'],[0.,3.]);np.testing.assert_array_equal(half['b'][0],[4.])
    np.testing.assert_array_equal(interpolate(old,new,0.)['a'],old['a'])


def test_physical_distance_is_gauge_stable_and_Q_normalization_independent():
    old=np.array([1.,.3j,0.]);q=np.array([.2,3.,4.])
    for phase in (2j, .13*np.exp(.7j),-.03):
        assert physical_distance(old,phase*old,q)['fs']<1e-14
    new=old+np.array([0.,0.,.005])
    np.testing.assert_allclose(physical_distance(old,new,q)['fs'],physical_distance(old,new,7.3*q)['fs'],atol=2e-16)
