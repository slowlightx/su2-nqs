"""Independent physical tests for complex local-Schur decoder compression."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin import _prepare_case
from projects.clebsch_tree_nqs.workflows.spin_vap_decoder import (
    decoder_config, init_decoder, prepare_decoder, teacher_objective, validate_decoder, tangent_rank,
)
from su2_nqs.benchmarks.krylov_anchored_full_vit import a1_orbit_coordinates


@pytest.mark.parametrize('geometry,width',[('square',1),('square',2)])
def test_local_schur_decoder_raw_and_tangent_are_singlets_before_projection(geometry,width):
    data=_prepare_case(dict(geometry=geometry,rows=2,columns=4,j2_over_j1=.125,
                            exact_energy_anchor=0.),'fixed_m0')
    config=decoder_config(2,4,width);p=init_decoder(130,config)
    problem=prepare_decoder(data,config)
    raw=np.asarray(problem['raw_fn'](p))
    assert np.linalg.norm(raw.real)>1e-10 and np.linalg.norm(raw.imag)>1e-10
    assert np.linalg.norm(data['spin_squared']@raw)/np.linalg.norm(raw)<2e-12
    state=np.asarray(problem['state_fn'](p));physical=problem['raw']['physical'](state)
    assert np.linalg.norm(data['spin_squared']@physical)<2e-12
    np.testing.assert_allclose(physical[data['tables']['gamma_a1']],np.broadcast_to(physical,data['tables']['gamma_a1'].shape),atol=2e-14)
    target=np.asarray(problem['state_fn'](init_decoder(131,config)))
    checks=validate_decoder(p,problem,data,target)
    assert max(v for k,v in checks.items() if not k.endswith('gradient_error'))<2e-12
    assert checks['fit_gradient_error']<2e-8
    assert checks['energy_gradient_error']<2e-8
    direct=np.mean(raw[data['tables']['gamma_a1']],axis=0)
    direct/=np.linalg.norm(direct)
    np.testing.assert_allclose(physical,direct,atol=2e-14)
    np.testing.assert_allclose(a1_orbit_coordinates(physical,problem['a1']),state,atol=2e-14)


def test_rank_one_teacher_objective_is_phase_invariant_and_not_physical_energy():
    target=np.array([1+2j,-3j,4.]);target/=np.linalg.norm(target)
    loss=teacher_objective(target)
    np.testing.assert_allclose(loss,loss.conj().T,atol=1e-15)
    np.testing.assert_allclose(loss@target,0,atol=3e-16)
    np.testing.assert_allclose(loss,teacher_objective(3*np.exp(.4j)*target),atol=3e-16)
    candidate=np.array([1j,.2,2.]);candidate/=np.linalg.norm(candidate)
    assert np.vdot(candidate,loss@candidate).real==pytest.approx(1-abs(np.vdot(target,candidate))**2,abs=5e-15)
    with pytest.raises(ValueError): teacher_objective(np.zeros(4))


def test_decoder_does_not_truncate_local_or_intercell_spin_or_accept_large_support():
    config=decoder_config(4,4,2)
    assert config.cells==((0,1,4,5),(2,3,6,7),(8,9,12,13),(10,11,14,15))
    assert config.maximum_intermediate_q is None
    assert config.projector is None and config.target_q==config.target_m2==0
    with pytest.raises(ValueError): decoder_config(6,6,2)


def test_projective_tangent_rank_removes_phase_and_norm():
    def state(p):
        v=p['re']+1j*p['im']
        return v/jnp.linalg.norm(v)
    p=dict(re=jnp.array([1.,.2,.3]),im=jnp.array([.2,-.1,.1]))
    result=tangent_rank(p,state,3)
    assert result['physical_projective_dimension']==4
    assert set(result['rank_lower_bounds'].values())=={4}
