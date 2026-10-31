import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from su2_nqs.vmc.dual_sampled_sr import make_dual_sampled_sr_update, sampled_overlap_diagnostics
from su2_nqs.vmc.physical_sampled_sr import sampled_physical_sr_direction

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('linear_solver', ['dual', 'svd'])
def test_dual_ridge_matches_dense_complex_qgt_and_existing_matrix_free_oracle(linear_solver):
    rng=np.random.default_rng(15)
    scores=jnp.asarray(rng.normal(size=(7,5))+1j*rng.normal(size=(7,5)))
    p=jnp.asarray(rng.normal(size=5))
    x=jnp.arange(7)[:,None]
    energies=jnp.asarray(rng.normal(size=7)+1j*rng.normal(size=7))
    def log_fn(p,x):
        return scores[x[0]]@p
    update=make_dual_sampled_sr_update(log_fn,p,damping=.13,step_size=.01,
        fisher_trust_radius=100.,maximum_overlap_distance=1.,minimum_ess_fraction=.01,
        score_chunk_size=3,linear_solver=linear_solver)
    new,diag=update(p,x,energies)
    centered=np.asarray(scores-scores.mean(axis=0))
    g=2*np.real(centered.conj().T@np.asarray(energies-energies.mean()))/7
    metric=2*np.real(centered.conj().T@centered)/7
    direction=np.linalg.solve(metric+.13*np.eye(5),g)
    np.testing.assert_allclose(new,np.asarray(p)-.01*direction,atol=2e-12,rtol=2e-12)
    oracle=sampled_physical_sr_direction(p,x,energy_gradient=jnp.asarray(g),
        log_amplitude_fn=jax.vmap(log_fn,in_axes=(None,0)),damping=.13,
        cg_tolerance=1e-12,cg_max_iterations=30)
    np.testing.assert_allclose((p-new)/.01,oracle.raw_direction,atol=2e-11,rtol=2e-11)
    assert float(diag['relative_linear_residual'])<1e-11
    assert int(diag['sample_matrix_dimension'])==14
    assert int(diag['score_real_entries'])==7*2*5


def test_gauge_energy_offset_and_chunk_invariance():
    x=jnp.arange(6)[:,None]
    p=jnp.asarray([.1,-.2,.3])
    def logs(p,x):
        v=x[0]-2.5
        return p[0]*v+1j*p[1]*v*v+p[2]*(1.+2j)
    energies=jnp.asarray([-.8,.4,.9,-1.,.3,.6])+1j*jnp.arange(6)*.1
    a=make_dual_sampled_sr_update(logs,p,score_chunk_size=2)
    b=make_dual_sampled_sr_update(logs,p,score_chunk_size=3)
    pa,da=a(p,x,energies)
    pb,db=b(p,x,energies+7.+3j)
    np.testing.assert_allclose(pa,pb,atol=2e-12,rtol=2e-12)
    assert float(pa[2])==pytest.approx(float(p[2]),abs=2e-12)
    assert float(da['fisher_norm'])<=.05*(1+1e-12)
    assert float(da['sampled_fs_sine'])<=.05*(1+1e-12)
    assert float(da['reweighting_ess_fraction'])>=.8


def test_small_exact_born_step_lowers_energy_and_respects_nonlinear_trust():
    x=jnp.array([[0],[1]])
    p=jnp.zeros(1,dtype=jnp.float64)
    def log_fn(p,x):
        return (2*x[0]-1)*p[0]+1j*jnp.pi*x[0]
    update=make_dual_sampled_sr_update(log_fn,p,step_size=10.,fisher_trust_radius=.3,
        maximum_overlap_distance=.05,score_chunk_size=1)
    changed,diag=update(p,x,jnp.asarray([1.,-1.]))
    prob=jax.nn.softmax(2*jax.vmap(log_fn,in_axes=(None,0))(changed,x).real)
    assert float(prob@jnp.asarray([1.,-1.]))<0
    assert int(diag['backtracking_evaluations'])>1
    assert bool(diag['accepted'])
    assert float(diag['sampled_fs_sine'])<=.05
    replay=sampled_overlap_diagnostics(jax.vmap(log_fn,in_axes=(None,0))(changed,x)-jax.vmap(log_fn,in_axes=(None,0))(p,x))
    assert float(diag['sampled_fs_sine'])==pytest.approx(float(replay['sampled_fs_sine']))


def test_overlap_stability_and_factory_guards():
    diag=sampled_overlap_diagnostics(jnp.array([1000.+2j,1000.+2j]))
    assert float(diag['sampled_fs_sine'])==0
    assert float(diag['reweighting_ess_fraction'])==1
    p=jnp.ones(1,dtype=jnp.float64)
    log_fn=lambda p,x:(p[0]*x[0]).astype(jnp.complex128)
    for kwargs in ({'damping':0},{'score_chunk_size':0},{'step_size':float('nan')},
                   {'minimum_ess_fraction':1.1},{'backtracking_steps':False}):
        with pytest.raises(ValueError):
            make_dual_sampled_sr_update(log_fn,p,**kwargs)
    with pytest.raises(ValueError,match='real float64'):
        make_dual_sampled_sr_update(log_fn,p.astype(jnp.complex128))


def test_zero_force_and_rejected_step_preserve_parameters():
    p=jnp.asarray([0.],dtype=jnp.float64)
    x=jnp.array([[0],[1]])
    def log_fn(p,x):
        return ((2*x[0]-1)*p[0]).astype(jnp.complex128)
    unchanged,zero=make_dual_sampled_sr_update(log_fn,p,score_chunk_size=1)(p,x,jnp.ones(2))
    np.testing.assert_array_equal(unchanged,p)
    assert float(zero['relative_linear_residual'])==0
    assert bool(zero['accepted'])
    reject=make_dual_sampled_sr_update(log_fn,p,step_size=1.,fisher_trust_radius=.3,
        maximum_overlap_distance=1e-8,backtracking_steps=1,score_chunk_size=1)
    unchanged,diag=reject(p,x,jnp.asarray([1.,-1.]))
    np.testing.assert_array_equal(unchanged,p)
    assert not bool(diag['accepted'])
    assert int(diag['backtracking_evaluations'])==1
    assert float(diag['parameter_step_norm'])==float(diag['applied_scale'])==0


def test_svd_ridge_retains_guard_for_ill_conditioned_rank_deficient_scores():
    rng=np.random.default_rng(121)
    scores=(rng.normal(size=(32,4))@rng.normal(size=(4,19)))*1e4
    scores+=rng.normal(size=scores.shape)*1e-4
    scores=scores+1j*rng.normal(size=scores.shape)
    x=jnp.arange(32)[:,None]; p=jnp.zeros(19,dtype=jnp.float64)
    energies=rng.normal(size=32)+1j*rng.normal(size=32)
    def log_fn(p,x):
        return jnp.asarray(scores)[x[0]]@p
    new,diag=make_dual_sampled_sr_update(log_fn,p,damping=.1,step_size=.001,
        fisher_trust_radius=100.,maximum_overlap_distance=1.,minimum_ess_fraction=.01,
        linear_solver='svd')(p,x,jnp.asarray(energies))
    center=scores-scores.mean(axis=0)
    a=np.sqrt(2/32)*np.concatenate((center.real,center.imag),axis=0)
    e=energies-energies.mean(); b=np.sqrt(2/32)*np.concatenate((e.real,e.imag))
    u,s,vh=np.linalg.svd(a,full_matrices=False)
    direction=vh.T@((s/(s*s+.1))*(u.T@b))
    np.testing.assert_allclose(new,-float(diag['applied_scale'])*direction,atol=1e-12,rtol=1e-9)
    assert float(diag['relative_linear_residual'])<1e-7
    assert bool(diag['accepted'])
    with pytest.raises(ValueError,match='linear_solver'):
        make_dual_sampled_sr_update(log_fn,p,linear_solver='weaken_guard')
