"""Independent finite-Pauli, influence and non-Ising stability checks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fnqs.basis import all_bits
from fnqs.variants.size_features import SizeFeatureConfig,init_model,log_psi
from fnqs.pilots.measurements import amplitudes
from fnqs.diagnostics.confirmation import correlation_rows,joint_statistics,energy_rows,fd_rows
from fnqs.diagnostics.frozen_stability import eta_fit,local_peak
from fnqs.scripts.run_frozen_confirmation import adapt_state
from fnqs.scripts.review_frozen_confirmation import independent_local_energy


def test_actual_pauli_local_energy_and_physical_derivative():
    n=4;cfg=SizeFeatureConfig(3,64,'legacy');p=init_model(jax.random.PRNGKey(391),cfg)
    bits=all_bits(n);g=jnp.array([1.,.17,.09,.88]);psi=np.exp(np.asarray(log_psi(p,jnp.asarray(bits),g,cfg)))
    z=np.diag([-1.,1.]);x=np.array([[0.,1.],[1.,0.]])
    def op(sites):
        value=np.array([[1.]])
        for s in range(n):value=np.kron(value,sites.get(s,np.eye(2)))
        return value
    h=np.zeros((16,16))
    for s in range(n):
        h-=float(g[0])*op({s:z,(s+1)%n:z})+float(g[1])*op({s:z,(s+2)%n:z})
        h-=float(g[2])*op({s:x,(s+1)%n:x})+float(g[3])*op({s:x})
    assert np.max(abs(np.asarray(energy_rows(p,jnp.asarray(bits),g,cfg))-h@psi/psi))<2e-11
    tfim_g=np.array([1.,0.,0.,.88]);tfim_psi=np.exp(np.asarray(log_psi(p,jnp.asarray(bits),jnp.asarray(tfim_g),cfg)))
    tfim_h=sum(-op({s:z,(s+1)%n:z})-.88*op({s:x}) for s in range(n))
    assert np.max(abs(independent_local_energy(p,bits,tfim_g,cfg)-tfim_h@tfim_psi/tfim_psi))<2e-11
    _,score=amplitudes(p,jnp.asarray(bits),g,cfg)
    assert np.max(abs(np.asarray(fd_rows(p,jnp.asarray(bits),g,cfg))-np.asarray(score)[:,None]))<2e-7
    rows=correlation_rows(bits)
    for r in range(n):
        expected=sum(psi@op({s:z,(s+r)%n:z})@psi for s in range(n))/n if r else 1.
        assert abs(psi**2@rows[:,r]-expected)<2e-12


def test_joint_influence_agrees_with_probability_perturbations():
    score=np.array([-.9,.2,1.8,-.4,.7]);corr=np.array([[1.,.2],[1.,-.4],[1.,.8],[1.,-.9],[1.,.5]])
    stats=joint_statistics(score,corr);p=np.ones(5)/5;eps=1e-6
    def function(q):return np.r_[q@score**2-(q@score)**2,q@corr]
    jac=np.column_stack([(function(p+eps*(np.eye(5)[k]-p))-function(p-eps*(np.eye(5)[k]-p)))/(2*eps) for k in range(5)])
    independent=jac@np.diag(p)@jac.T/(len(p)-1)
    assert np.max(abs(stats['joint_covariance']-independent))<2e-10
    assert abs(stats['chi_total']-(p@score**2-(p@score)**2))<1e-14


@pytest.mark.parametrize('correction',[False,True])
def test_free_eta_recovers_non_ising_exponent(correction):
    sizes=[16,24,32];cs=[]
    for n in sizes:
        r=np.arange(n);chord=n/np.pi*np.sin(np.pi*r/n);chord[0]=1
        cs.append(.8*(n/16)**.02*chord**(-.37-(.41/n if correction else 0)))
    fit=eta_fit(sizes,cs,4,correction)
    assert abs(fit['eta']-.37)<2e-13
    if correction:assert abs(fit['correction']-.41)<2e-11


def test_quadratic_peak_and_unbracketed_peak():
    h=np.linspace(.8,1.2,41);y=2.-3*(h-1.037)**2
    for window in (5,7,9):
        fit=local_peak(h,y,window);assert abs(fit['h_peak']-1.037)<1e-13
        assert fit['bracketed'] and fit['quadratic_vertex_accepted']
    assert not local_peak(h,1+h,7)['bracketed']
    with pytest.raises(ValueError):local_peak(h,y,43)


def test_work_translation_preserves_optimizer_and_actual_counts():
    state=dict(step=17,m={'x':np.array([2.])},work=dict(gradient_hamiltonians=34,
        hamiltonian_evaluations=34,born_samples=4352,basis_state_evaluations=0,sr_refinement_solves=3))
    translated=adapt_state(state)
    assert translated['m'] is state['m']
    assert translated['work']==dict(updates=17,task_evaluations=34,hamiltonian_actions=34,
        source_actions=0,born_samples=4352,basis_rows=0,sr_correction_solves=3)
