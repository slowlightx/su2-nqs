"""Force, finite-distance update, audit errors, and resumed trajectory checks."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from su2_nqs.vmc.bounded_physical_pilot import (
    make_dynamic_force, initialize_pilot_adam, bounded_adam_step,
    energy_summary, independent_audit_comparison,
)
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint, load_resumable_checkpoint


def test_dynamic_force_matches_independent_dense_normalized_energy_derivative():
    # Equal-amplitude parent means uniform samples are exact Born quadrature.
    states=jnp.eye(4);theta=jnp.zeros(4)
    H=jnp.asarray([[1.,.2,0,.1],[.2,-1.,.4,0],[0,.4,2.,.3],[.1,0,.3,-.2]])
    amp=lambda p,x:jnp.exp(x@p+1j*.7*(x@p))
    vector=amp(theta,states);energies=(H@vector)/vector
    actual=make_dynamic_force(amp,4)(theta,states,energies)
    def energy(p):
        psi=amp(p,states)
        return jnp.real(jnp.vdot(psi,H@psi)/jnp.vdot(psi,psi))/4
    np.testing.assert_allclose(actual,jax.grad(energy)(theta),atol=2e-14)
    np.testing.assert_allclose(make_dynamic_force(amp,4)(theta,states,energies+7.),actual,atol=2e-14)


def test_checkpoint_restores_params_moments_chains_keys_and_same_next_update(tmp_path):
    amp=jax.jit(lambda p,x:jnp.exp(x@p).astype(jnp.complex128))
    theta=jnp.array([.02,-.03]);states=jnp.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    opt=initialize_pilot_adam(theta);key=jax.random.PRNGKey(17);history=[]
    def step(p,o,k,x):
        k,sub=jax.random.split(k);gradient=jax.random.normal(sub,p.shape)
        p,o,info=bounded_adam_step(p,gradient,o,x,amplitudes=amp,learning_rate=.2,
            clip_norm=1.,maximum_fs=.02,bisections=10)
        assert info['distance']['fubini_study_distance']<=.02+1e-12
        return p,o,k,info
    theta,opt,key,info=step(theta,opt,key,states);history.append(info)
    path=tmp_path/'pilot.pkl';identity={'model':'unit','steps':2}
    save_resumable_checkpoint(path,np.asarray(theta),optimizer_state=jax.device_get(opt),
        completed_steps=1,next_sample_index=1,run_identity=identity,history=history,
        metadata={'chains':np.asarray(states),'key':np.asarray(key)})
    restored=load_resumable_checkpoint(path,expected_run_identity=identity)
    direct=step(theta,opt,key,states)
    resumed=step(jnp.asarray(restored['params']),jax.tree_util.tree_map(jnp.asarray,restored['optimizer_state']),
        jnp.asarray(restored['metadata']['key']),jnp.asarray(restored['metadata']['chains']))
    for a,b in zip(jax.tree_util.tree_leaves(direct[:3]),jax.tree_util.tree_leaves(resumed[:3])):
        np.testing.assert_array_equal(a,b)
    assert direct[3]==resumed[3]


def test_complex_variance_and_independent_chain_errors_do_not_use_iid_snapshots():
    values=np.tile(np.array([1+1j,2-1j,4+2j,8-2j]),(8,1))
    r=energy_summary(values,2)
    assert r['variance_per_site']==np.mean(abs(values-values.mean())**2)/2
    np.testing.assert_allclose(r['energy_chain_se'],values[0].real.std(ddof=1)/4)
    c=independent_audit_comparison(values,values-1,n_sites=2,seed=11,replicates=1000)
    assert c['delta_energy_per_site']==-.5 and not c['paired']
    np.testing.assert_allclose(c['delta_energy_chain_se'],np.sqrt(2)*r['energy_chain_se'])
    assert c['delta_variance_per_site']==0 and not r['equilibrium_certified']
