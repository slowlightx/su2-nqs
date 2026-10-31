from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.analytic_plaquette_carrier import aligned_plaquette_a1_local_data
from su2_nqs.benchmarks.coherent_plaquette_sampling import plaquette_frame_component_amplitudes
from su2_nqs.vmc.frame_metropolis import frame_proposal, hastings_log_acceptance, make_frame_metropolis
from su2_nqs.vmc.resumable_checkpoint import save_resumable_checkpoint, load_resumable_checkpoint

jax.config.update('jax_enable_x64', True)
FRAMES = (((0,1,2,3),(4,5,6,7)), ((1,2,3,4),(5,6,7,0)))
WEIGHTS = [.3, .7]


def setup():
    bits = np.zeros((70,8), dtype=np.int32)
    for row, up in enumerate(combinations(range(8),4)):
        bits[row,list(up)] = 1
    local, _ = aligned_plaquette_a1_local_data()
    logq, draw = frame_proposal(FRAMES, WEIGHTS, abs(local)**2, uniform_fraction=.1)
    return bits, logq, draw


def test_full_density_normalization_support_and_independent_frequencies():
    bits, logq, draw = setup()
    q = np.exp(np.asarray(jax.jit(logq)(jnp.asarray(bits))))
    expected = .9*(abs(plaquette_frame_component_amplitudes(bits,FRAMES))**2 @ WEIGHTS)+.1/70
    np.testing.assert_allclose(q,expected,atol=1e-15,rtol=2e-14)
    assert abs(q.sum()-1) < 2e-14
    assert np.all(q >= .1/70*(1-1e-14))
    samples = np.asarray(jax.jit(lambda k:draw(k,30000))(jax.random.PRNGKey(10)))
    codes = samples @ (1 << np.arange(7,-1,-1))
    observed = np.asarray([(codes == code).mean() for code in bits @ (1 << np.arange(7,-1,-1))])
    np.testing.assert_allclose(observed,q,atol=.006,rtol=0)
    assert np.all(samples.sum(axis=1)==4)


def test_exact_independence_transition_detailed_balance_including_target_nodes():
    bits, logq, _ = setup()
    q = np.exp(np.asarray(logq(jnp.asarray(bits))))
    # Deliberately non-product target with support outside all frames and one node.
    pi = np.exp(bits @ np.linspace(-.7,.8,8)); pi[0] = 0; pi /= pi.sum()
    with np.errstate(divide='ignore'):
        logs = np.log(pi)
    accept = np.exp(np.asarray(hastings_log_acceptance(jnp.asarray(logs[:,None]),
        jnp.asarray(logs[None,:]),jnp.log(q[:,None]),jnp.log(q[None,:]))))
    transition = q[None,:]*accept
    transition[np.diag_indices(70)] += 1-transition.sum(axis=1)
    flow = pi[:,None]*transition
    assert np.max(abs(flow-flow.T)) < 2e-17
    assert np.max(abs(pi@transition-pi)) < 2e-16
    assert np.max(abs(transition.sum(axis=1)-1)) < 3e-16
    assert np.exp(float(hastings_log_acceptance(-jnp.inf,-jnp.inf,0.,0.))) == 0


@pytest.mark.parametrize('mode',['nearest','global','frame_mix'])
def test_sampler_born_frequencies_and_exact_checkpoint_resume(mode,tmp_path):
    bits, logq, draw = setup()
    field = jnp.linspace(-.25,.3,8)
    log_fn = lambda p,x: (p@x).astype(jnp.complex128)
    sample = make_frame_metropolis(log_fn,mode=mode,proposals=96,n_sites=8,
        edges=[(i,(i+1)%8) for i in range(8)],log_proposal=logq,draw_proposal=draw)
    initial = jnp.asarray(np.tile(bits,(60,1)))
    result = sample(field,jax.random.PRNGKey(42),initial)
    states,key,counts = jax.device_get(result)
    observed = np.mean(np.all(states[:,None,:]==bits[None,:,:],axis=-1),axis=0)
    expected = np.exp(2*np.asarray(bits@field)); expected /= expected.sum()
    np.testing.assert_allclose(observed,expected,atol=.009,rtol=0)
    assert counts[:,0].sum()==96*len(initial)
    assert np.all(counts[:,2]<=counts[:,1])
    assert np.all(counts[:,1]<=counts[:,0])
    file=tmp_path/'checkpoint.pkl'
    save_resumable_checkpoint(file,np.asarray(field),optimizer_state=None,completed_steps=0,
        next_sample_index=96,run_identity={'mode':mode},history=[],metadata={'states':states,'key':key})
    restored=load_resumable_checkpoint(file,expected_run_identity={'mode':mode})
    left=sample(field,key,states)
    right=sample(jnp.asarray(restored['params']),restored['metadata']['key'],restored['metadata']['states'])
    for a,b in zip(left,right):np.testing.assert_array_equal(a,b)


def test_reject_missing_support_and_nonzero_m():
    local,_=aligned_plaquette_a1_local_data()
    with pytest.raises(ValueError):frame_proposal(FRAMES,WEIGHTS,abs(local)**2,uniform_fraction=0)
    p=np.zeros(16);p[0]=1
    with pytest.raises(ValueError):frame_proposal(FRAMES,WEIGHTS,p,uniform_fraction=.1)


def test_near_node_mass_can_hide_variance_despite_small_energy_bias():
    """Two-dimensional Hermitian example, NOT a numerical claim about N36."""
    epsilon=1e-6;psi=np.array([1.,epsilon]);psi/=np.linalg.norm(psi)
    h=np.array([[-1.,.5],[.5,2.]])
    local=(h@psi)/psi;prob=psi**2
    exact_energy=float(prob@local)
    exact_variance=float(prob@(local-exact_energy)**2)
    assert prob[1]<1.1e-12
    assert abs(local[0]-exact_energy)<1e-6
    assert exact_variance>.249
    # Every finite sample confined to the dominant component reports zero variance.
    assert np.var(np.repeat(local[0],256))<1e-30
