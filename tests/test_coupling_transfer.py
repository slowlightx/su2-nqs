"""Independent exact-small derivative and sealed-protocol checks, no ED."""
import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_coupling_transfer import (
    CONFIG, rayleigh_metrics, response_metrics, validate_protocol,
)

jax.config.update('jax_enable_x64', True)


def test_phase_gauge_does_not_change_energy_response_or_metric():
    h0=jnp.diag(jnp.array([-1.,1.]));h2=jnp.diag(jnp.array([.3,.7]))
    def real_state(p,j):return jnp.array([jnp.cos(p[0]*j),jnp.sin(p[0]*j)],dtype=jnp.complex128)
    def phase_state(p,j):return real_state(p,j)*jnp.exp(7j*j*j)
    p=jnp.array([.7]);j=.4
    a=response_metrics(real_state,p,j,h0,h2,1e-4)
    b=response_metrics(phase_state,p,j,h0,h2,1e-4)
    assert a['fidelity_susceptibility']==pytest.approx(.49,abs=1e-12)
    for key in ('energy_derivative_per_site','energy_curvature_per_site',
                'bond_response_per_site','fidelity_susceptibility','hellmann_feynman_defect'):
        assert a[key]==pytest.approx(b[key],abs=1e-12)
    assert a['energy_derivative_fd_error']<1e-8
    assert a['energy_curvature_fd_error']<1e-7
    assert a['bond_response_fd_error']<1e-8
    assert abs(a['hellmann_feynman_defect'])>.01
    assert a['hellmann_feynman_residual_identity_error']<1e-14
    assert abs(a['hellmann_feynman_defect'])<=a['hellmann_feynman_residual_bound']+1e-14


def test_constant_non_ground_state_passes_HF_but_is_not_response_certificate():
    def state(p,j):return jnp.array([1.,1.],dtype=jnp.complex128)/jnp.sqrt(2.)
    a=response_metrics(state,jnp.array([0.]),.4,jnp.diag(jnp.array([-1.,1.])),jnp.eye(2),1e-4)
    assert abs(a['hellmann_feynman_defect'])<1e-15
    assert a['fidelity_susceptibility']==0
    metrics=rayleigh_metrics(np.ones(2),np.diag([-1.,1.]))
    assert metrics['variance_per_site']==pytest.approx(1/16)


def test_HF_consistency_can_coexist_with_wrong_ground_state_response():
    def state(p,j):return jnp.array([1.,0.],dtype=jnp.complex128)
    j=.4
    a=response_metrics(state,jnp.array([0.]),j,jnp.diag(jnp.array([-1.,1.])),jnp.array([[0.,1.],[1.,0.]]),1e-4)
    assert abs(a['hellmann_feynman_defect'])<1e-15
    assert a['energy_curvature_per_site']==0
    assert a['fidelity_susceptibility']==0
    exact_curvature=-1/(16*(1+j*j)**1.5)
    exact_metric=1/(4*(1+j*j)**2)
    assert abs(a['energy_curvature_per_site']-exact_curvature)>.04
    assert abs(a['fidelity_susceptibility']-exact_metric)>.1


def test_variance_is_nonnegative_residual_and_normalization_invariant():
    p=np.array([1.,2j]);h=np.array([[.2,1j],[-1j,-.3]])
    actual=rayleigh_metrics(p,h)
    unit=p/np.linalg.norm(p);energy=np.vdot(unit,h@unit).real
    assert actual['energy_per_site']==pytest.approx(energy/16)
    assert actual['variance_per_site']==pytest.approx((np.vdot(h@unit,h@unit).real-energy**2)/16)
    assert actual==pytest.approx(rayleigh_metrics((.3+.4j)*p,h))


def test_frozen_training_budget_and_disjoint_heldout():
    spec=json.loads(CONFIG.read_text());validate_protocol(spec)
    assert set(spec['train_couplings']).isdisjoint(spec['held_out_couplings'])
    assert spec['shared_steps']*len(spec['train_couplings'])==spec['pointwise_steps']*len(spec['all_couplings'])==900
    for key in ('new_ED_allowed','new_sampling_allowed','remote_allowed','heldout_in_objective'):
        altered=copy.deepcopy(spec);altered[key]=True
        with pytest.raises(ValueError):validate_protocol(altered)


@pytest.mark.parametrize('key,value',[('shared_steps',301),('learning_rate',.003),('parents',[]),('held_out_couplings',[.4,.6])])
def test_protocol_rejects_posthoc_changes(key,value):
    spec=json.loads(CONFIG.read_text());spec[key]=value
    with pytest.raises(ValueError):validate_protocol(spec)
