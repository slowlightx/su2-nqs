import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from fnqs.artifacts import load_checkpoint, save_checkpoint
from fnqs.hamiltonians import IsingCouplings, all_bits, dense_hamiltonian
from fnqs.models import GRUConfig, coupling_scores, init_model, log_psi, sample
from fnqs.initialization import initialize_run
from fnqs.experiments import _sampled_audit_values
from fnqs.artifacts import source_hashes
from fnqs.observables import fidelity_metric, quantum_geometric_tensor
from fnqs.training import (Task, ensemble_sr_direction, exact_statistics, init_optimizer,
                           local_energies, train, _exact_ensemble_gradient)


@pytest.fixture
def model():
    config=GRUConfig(4)
    return init_model(jax.random.PRNGKey(123),config),config


def test_shared_model_normalization_z2_and_sampling_frequencies(model):
    params,config=model
    g=jnp.asarray([1,.1,.05,1.2])
    for n in [4,6,8]:
        bits=jnp.asarray(all_bits(n))
        logs=log_psi(params,bits,g,config)
        np.testing.assert_allclose(jnp.sum(jnp.exp(2*logs)),1,atol=1e-12,rtol=0)
        np.testing.assert_allclose(logs,log_psi(params,1-bits,g,config),atol=1e-14,rtol=0)
    draws=np.asarray(sample(params,jax.random.PRNGKey(2),24000,4,g,config))
    labels=draws @ (1 << np.arange(3,-1,-1))
    p=np.asarray(jnp.exp(2*log_psi(params,jnp.asarray(all_bits(4)),g,config)))
    count=np.bincount(labels,minlength=16)
    assert np.max(abs(count/len(draws)-p)/np.sqrt(p*(1-p)/len(draws)))<5
    assert sample(params,jax.random.PRNGKey(1),4,16,g,config).shape==(4,16)
    assert np.isneginf(log_psi(params,jnp.array([0,1,2,0]),g,config))
    assert np.isneginf(log_psi(params,jnp.array([0,.5,1,0]),g,config))


def test_input_qgt_fixed_parameter_ad_against_finite_difference(model):
    params,config=model
    bits=jnp.asarray(all_bits(4))
    g=jnp.array([1,.1,.05,1.1])
    scores=np.asarray(coupling_scores(params,bits,g,config))
    delta=1e-4
    for i in range(4):
        plus=g.at[i].add(delta);minus=g.at[i].add(-delta)
        finite=(log_psi(params,bits,plus,config)-log_psi(params,bits,minus,config))/(2*delta)
        np.testing.assert_allclose(scores[:,i],finite,atol=1e-9,rtol=1e-6)
    psi=np.exp(np.asarray(log_psi(params,bits,g,config)))
    qgt=quantum_geometric_tensor(scores,psi**2)
    plus=np.exp(np.asarray(log_psi(params,bits,g.at[3].add(delta),config)))
    minus=np.exp(np.asarray(log_psi(params,bits,g.at[3].add(-delta),config)))
    np.testing.assert_allclose(qgt[3,3].real,fidelity_metric(psi,plus,minus,delta),atol=5e-8)


def test_full_operator_local_energy_and_exact_gradient(model):
    params,config=model
    bits=jnp.asarray(all_bits(4));g=IsingCouplings(j2=.2,jx=.1,h=1.1)
    psi=np.exp(np.asarray(log_psi(params,bits,jnp.asarray(g.vector()),config)))
    matrix=dense_hamiltonian(4,g)
    local=local_energies(params,bits,jnp.asarray(g.vector()),config)
    np.testing.assert_allclose(local,(matrix@psi)/psi,atol=2e-12)
    energy,variance=exact_statistics(params,4,jnp.asarray(g.vector()),config)
    np.testing.assert_allclose(energy,psi@matrix@psi,atol=2e-12)
    np.testing.assert_allclose(variance,np.linalg.norm(matrix@psi-energy*psi)**2,atol=2e-12)
    flat,unravel=ravel_pytree(params)
    direction=jnp.linspace(-.1,.1,len(flat))
    objective=lambda p: exact_statistics(unravel(p),4,jnp.asarray(g.vector()),config)[0]
    analytic=jax.jvp(objective,(flat,),(direction,))[1]
    finite=(objective(flat+1e-5*direction)-objective(flat-1e-5*direction))/2e-5
    np.testing.assert_allclose(analytic,finite,atol=2e-8)


@pytest.mark.parametrize("n",[4,6,8,10])
def test_unsigned_local_energy_matches_independent_dense_operator(model,n):
    params,config=model
    bits=all_bits(n).astype(np.uint8);g=IsingCouplings(j2=.2,jx=.1,h=1.1)
    vector=jnp.asarray(g.vector())
    psi=np.exp(np.asarray(log_psi(params,jnp.asarray(bits),vector,config)))
    expected=dense_hamiltonian(n,g)@psi/psi
    actual=local_energies(params,jnp.asarray(bits),vector,config)
    np.testing.assert_allclose(actual,expected,atol=4e-11,rtol=0)


def test_warm_start_restarts_optimizer_and_rng_without_resealing_parent(model,tmp_path):
    params,config=model
    path=tmp_path/"parent.npz"
    parent_state=init_optimizer(params);parent_state["step"]=53
    hashes=source_hashes();hashes["training.py"]="previous_training_version"
    previous_origin={"checkpoint":"grandparent.npz","parent_work":{"gradient_hamiltonians":6000}}
    save_checkpoint(path,params,parent_state,jax.random.PRNGKey(2),
                    {"seed":123,"config":{"model":{"hidden_dim":4,"size_scale":64.}},
                     "source_sha256":hashes,"warm_start":previous_origin})
    before=path.read_bytes()
    loaded,state,key,origin=initialize_run(29,config,path)
    assert path.read_bytes()==before and state["step"]==0
    assert origin["parent_step"]==53 and origin["optimizer_and_prng_restarted"]
    assert origin["parent_origin"]==previous_origin
    np.testing.assert_array_equal(key,jax.random.split(jax.random.PRNGKey(29))[0])
    for name in params:np.testing.assert_array_equal(loaded[name],params[name])
    with pytest.raises(ValueError):initialize_run(29,GRUConfig(8),path)
    hashes["models.py"]="incompatible_model"
    save_checkpoint(tmp_path/"incompatible.npz",params,parent_state,key,
                    {"seed":123,"config":{"model":{"hidden_dim":4,"size_scale":64.}},"source_sha256":hashes})
    with pytest.raises(ValueError):initialize_run(29,config,tmp_path/"incompatible.npz")


def test_dynamic_audit_kernel_matches_independent_operator_and_coupling_jacobian(model):
    params,config=model;bits=jnp.asarray(all_bits(4))
    for h in [.7,1.3]:
        g=IsingCouplings(j2=.2,jx=.1,h=h);vector=jnp.asarray(g.vector())
        psi=np.exp(np.asarray(log_psi(params,bits,vector,config)))
        expected_local=dense_hamiltonian(4,g)@psi/psi
        expected_scores=jax.jacrev(lambda v:log_psi(params,bits,v,config))(vector)
        actual_local,actual_scores=_sampled_audit_values(params,bits,vector,config)
        np.testing.assert_allclose(actual_local,expected_local,atol=3e-12,rtol=0)
        np.testing.assert_allclose(actual_scores,expected_scores,atol=3e-12,rtol=0)


@pytest.mark.parametrize("sizes",[(4,4),(4,6),(4,8)])
def test_multisystem_sr_uses_within_system_centering(model,sizes):
    params,config=model
    tasks=[Task(n,IsingCouplings(h=h)) for n,h in zip(sizes,[.7,1.3],strict=True)]
    flat,unravel=ravel_pytree(params)
    matrix=np.zeros((len(flat),len(flat)));force=np.zeros(len(flat))
    for task in tasks:
        bits=jnp.asarray(all_bits(task.n))
        g=jnp.asarray(task.couplings.vector())
        scores=np.asarray(jax.jacrev(lambda p:log_psi(unravel(p),bits,g,config))(flat))
        p=np.asarray(jnp.exp(2*log_psi(params,bits,g,config)))
        centered=scores-p@scores
        energies=np.asarray(local_energies(params,bits,g,config))/task.n
        matrix += centered.T@(p[:,None]*centered)/2
        force += centered.T@(p*(energies-p@energies))/2
    direction=ensemble_sr_direction(params,tasks,[jax.random.PRNGKey(1)]*2,config,"exact",16,.01)
    expected=np.linalg.solve(matrix+.01*np.eye(len(flat)),force)
    np.testing.assert_allclose(ravel_pytree(direction)[0],expected,atol=2e-11)
    iterative=ensemble_sr_direction(params,tasks,[jax.random.PRNGKey(1)]*2,config,"exact",16,.01,
                                    solver="cg",tolerance=1e-11,maxiter=500)
    np.testing.assert_allclose(ravel_pytree(iterative)[0],expected,atol=2e-10,rtol=1e-8)


def test_batched_exact_gradient_matches_independent_dense_hamiltonian(model):
    params,config=model
    fields=[.8,1.2]
    bits=jnp.asarray(all_bits(4))
    matrices=jnp.asarray([dense_hamiltonian(4,IsingCouplings(h=h)) for h in fields])
    couplings=jnp.asarray([IsingCouplings(h=h).vector() for h in fields])
    def independent(p):
        energies=[]
        for g,matrix in zip(couplings,matrices,strict=True):
            psi=jnp.exp(log_psi(p,bits,g,config))
            energies.append(psi@matrix@psi/4)
        return sum(energies)/len(energies)
    expected=jax.grad(independent)(params)
    actual,_=_exact_ensemble_gradient(params,4,couplings,config)
    np.testing.assert_allclose(ravel_pytree(actual)[0],ravel_pytree(expected)[0],atol=3e-12)


def test_sampled_sr_centering_has_a_nonzero_global_centering_witness(model):
    # Exhaustive normalized models have zero score means in each system, so
    # an exact-only test cannot detect incorrectly pooling those means. Finite
    # independent samples give a nonzero witness of this implementation error.
    params,config=model
    tasks=[Task(4,IsingCouplings(h=.7)),Task(6,IsingCouplings(h=1.3))]
    keys=[jax.random.PRNGKey(13),jax.random.PRNGKey(37)]
    flat,unravel=ravel_pytree(params)
    scores_list,energies_list=[],[]
    for task,key in zip(tasks,keys,strict=True):
        g=jnp.asarray(task.couplings.vector())
        bits=sample(params,key,12,task.n,g,config)
        scores=np.asarray(jax.jacrev(lambda p:log_psi(unravel(p),bits,g,config))(flat))
        basis=jnp.asarray(all_bits(task.n))
        psi=np.exp(np.asarray(log_psi(params,basis,g,config)))
        local=(dense_hamiltonian(task.n,task.couplings)@psi)/psi/task.n
        labels=np.asarray(bits)@(1<<np.arange(task.n-1,-1,-1))
        scores_list.append(scores);energies_list.append(local[labels])
    matrix=np.zeros((len(flat),len(flat)));force=np.zeros(len(flat))
    for scores,energies in zip(scores_list,energies_list,strict=True):
        centered=scores-scores.mean(axis=0)
        matrix+=centered.T@centered/24
        force+=centered.T@(energies-energies.mean())/24
    expected=np.linalg.solve(matrix+.01*np.eye(len(flat)),force)
    actual=ensemble_sr_direction(params,tasks,keys,config,"vmc",12,.01,solver="cg",tolerance=1e-11)
    np.testing.assert_allclose(ravel_pytree(actual)[0],expected,atol=2e-10,rtol=1e-8)
    pooled=np.concatenate(scores_list);pooled-=pooled.mean(axis=0)
    incorrect=np.linalg.solve(pooled.T@pooled/24+.01*np.eye(len(flat)),force)
    assert np.linalg.norm(incorrect-expected)>1e-4


def test_exact_sr_backtracking_and_task_minibatch_resume(model):
    params,config=model
    tasks=[Task(4,IsingCouplings(h=h)) for h in [.7,.9,1.1,1.3]]
    key=jax.random.PRNGKey(71)
    options=dict(optimizer="sr",sr_solver="cg",sr_shift=.01,learning_rate=100.,
                 sr_max_step_norm=.3,sr_backtracking=True,tasks_per_step=2)
    final,state,last_key,history=train(params,tasks,config,key,steps=4,**options)
    first,moments,first_key,_=train(params,tasks,config,key,steps=2,**options)
    resumed,_,resumed_key,_=train(first,tasks,config,first_key,state=moments,steps=2,**options)
    np.testing.assert_array_equal(last_key,resumed_key)
    np.testing.assert_array_equal(ravel_pytree(final)[0],ravel_pytree(resumed)[0])
    assert state["step"]==4 and all(r["tasks_evaluated"]==2 for r in history)
    assert all(r["sr_relative_residual"]<1e-7 for r in history)
    before=sum(float(exact_statistics(params,4,jnp.asarray(t.couplings.vector()),config)[0]) for t in tasks)
    after=sum(float(exact_statistics(final,4,jnp.asarray(t.couplings.vector()),config)[0]) for t in tasks)
    assert after<before
    with pytest.raises(ValueError):train(params,tasks,config,key,steps=1,tasks_per_step=5)
    with pytest.raises(ValueError):train(params,tasks,config,key,steps=1,mode="vmc",**options)


def test_training_checkpoint_resume_is_identical_and_energy_descends(model,tmp_path):
    params,config=model
    tasks=[Task(4,IsingCouplings(h=.8)),Task(6,IsingCouplings(h=1.2))]
    key=jax.random.PRNGKey(7)
    final,state,final_key,_=train(params,tasks,config,key,steps=4)
    first,first_state,first_key,_=train(params,tasks,config,key,steps=2)
    path=tmp_path/"checkpoint.npz"
    save_checkpoint(path,first,first_state,first_key,{"seed":7})
    first,first_state,first_key,metadata=load_checkpoint(path)
    resumed,resumed_state,resumed_key,_=train(first,tasks,config,first_key,state=first_state,steps=2)
    assert metadata=={"seed":7} and resumed_state["step"]==state["step"]==4
    np.testing.assert_array_equal(final_key,resumed_key)
    for a,b in zip(jax.tree.leaves(final),jax.tree.leaves(resumed),strict=True):np.testing.assert_array_equal(a,b)
    initial_energy=exact_statistics(params,4,jnp.asarray(tasks[0].couplings.vector()),config)[0]
    final_energy=exact_statistics(final,4,jnp.asarray(tasks[0].couplings.vector()),config)[0]
    assert final_energy<initial_energy


def test_vmc_step_and_stoquastic_boundary(model):
    params,config=model
    final,state,key,history=train(params,[Task(4,IsingCouplings())],config,jax.random.PRNGKey(3),steps=1,mode="vmc",n_samples=32)
    assert state["step"]==1 and np.isfinite(history[0]["energy_per_site"])
    with pytest.raises(ValueError):Task(4,IsingCouplings(jx=-.1))
    with pytest.raises(ValueError):train(params,[Task(16,IsingCouplings())],config,key,steps=1)


def test_centered_vmc_force_has_exact_variational_gradient(model):
    params,config=model
    bits=jnp.asarray(all_bits(4));g=jnp.asarray([1,.1,.1,1.1])
    p=jnp.exp(2*log_psi(params,bits,g,config))
    energies=local_energies(params,bits,g,config)
    centered=jax.lax.stop_gradient(energies-p@energies)
    weights=jax.lax.stop_gradient(p)
    force=jax.grad(lambda pars:2*jnp.sum(weights*log_psi(pars,bits,g,config)*centered))(params)
    derivative=jax.grad(lambda pars:exact_statistics(pars,4,g,config)[0])(params)
    for a,b in zip(jax.tree.leaves(force),jax.tree.leaves(derivative),strict=True):
        np.testing.assert_allclose(a,b,atol=3e-12)


def test_vmc_checkpoint_resume_uses_identical_draws(model,tmp_path):
    params,config=model;tasks=[Task(4,IsingCouplings())];key=jax.random.PRNGKey(91)
    final,state,last_key,_=train(params,tasks,config,key,steps=2,mode="vmc",n_samples=16)
    first,moments,first_key,_=train(params,tasks,config,key,steps=1,mode="vmc",n_samples=16)
    path=tmp_path/"vmc.npz";save_checkpoint(path,first,moments,first_key,{"mode":"vmc"})
    first,moments,first_key,_=load_checkpoint(path)
    resumed,_,resumed_key,_=train(first,tasks,config,first_key,state=moments,steps=1,mode="vmc",n_samples=16)
    np.testing.assert_array_equal(last_key,resumed_key)
    np.testing.assert_array_equal(ravel_pytree(final)[0],ravel_pytree(resumed)[0])
