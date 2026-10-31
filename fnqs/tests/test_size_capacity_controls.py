import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from fnqs.experiments import _audit_sampled
from fnqs.hamiltonians import all_bits, IsingCouplings
from fnqs.models import GRUConfig, coupling_scores, init_model, log_psi
from fnqs.observables import quantum_geometric_tensor, fidelity_metric
from fnqs.training import init_optimizer
from fnqs.scripts import size_capacity_controls as controls


def test_exact_aggregate_costs_parameter_counts_and_field_splits():
    protocol = controls.build_protocol([71,83,97])
    fixed = [a for name,a in protocol["arms"].items() if name.startswith("fixed")]
    shared = protocol["arms"]["shared16"]
    for metric in ["gradient_hamiltonians", "hamiltonian_evaluations", "born_samples"]:
        assert sum(a["endpoint_costs"]["1000"][metric] for a in fixed) == shared["endpoint_costs"]["1000"][metric]
    assert shared["endpoint_costs"]["2000"]["born_samples"] == 1536000
    assert sum(a["parameter_count"] for a in fixed) == 4422
    assert protocol["arms"]["shared32"]["parameter_count"] == 4482
    for width in [16,32]:
        actual = sum(p.size for p in jax.tree.leaves(init_model(jax.random.PRNGKey(71),GRUConfig(width))))
        assert actual == controls.parameter_count(width)
    groups = [set(protocol[k]) for k in ["train_fields","development_fields","reserved_confirmation_fields"]]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert not protocol["universality_established"]


def test_balanced_field_exposure_and_prefix_schedule():
    protocol=controls.build_protocol([71])
    schedule=controls.field_schedule(71,20,protocol["sizes"],protocol["train_fields"])
    np.testing.assert_array_equal(schedule[:7],controls.field_schedule(71,7,protocol["sizes"],protocol["train_fields"]))
    assert np.all(schedule[:,:,0] != schedule[:,:,1])
    for epoch in range(20):
        shared=controls.selected_tasks(protocol,protocol["arms"]["shared16"],schedule,epoch)
        fixed=sum([controls.selected_tasks(protocol,protocol["arms"][f"fixed16_L{n}"],schedule,epoch) for n in protocol["sizes"]],[])
        assert shared == fixed
        assert [t.n for t in shared] == [16,16,24,24,32,32]


def test_control_prefix_resume_preserves_optimizer_and_sampling_rng():
    protocol=controls.build_protocol([71],base_steps=2)
    protocol["sizes"]=[4,6]
    protocol["training_options"].update(n_samples=8,learning_rate=.1,sr_shift=.001,sr_maxiter=500)
    arm={"sizes":[4,6]}
    schedule=controls.field_schedule(71,4,protocol["sizes"],protocol["train_fields"])
    model=GRUConfig(3)
    params=init_model(jax.random.PRNGKey(2),model);state=init_optimizer(params);key=jax.random.PRNGKey(71)
    full,full_state,full_key,_=controls.advance(params,state,key,model,protocol,arm,schedule,2)
    first,moments,first_key,_=controls.advance(params,state,key,model,protocol,arm,schedule,1)
    resumed,resumed_state,resumed_key,_=controls.advance(first,moments,first_key,model,protocol,arm,schedule,1)
    np.testing.assert_array_equal(ravel_pytree(full)[0],ravel_pytree(resumed)[0])
    np.testing.assert_array_equal(full_key,resumed_key)
    assert full_state["work"] == resumed_state["work"]
    assert full_state["work"]["gradient_hamiltonians"] == 8


@pytest.mark.parametrize("width",[16,32])
def test_exhaustive_normalization_and_input_metric_at_actual_widths(width):
    model=GRUConfig(width);params=init_model(jax.random.PRNGKey(71),model)
    vector=jnp.array([1.,.1,.05,1.01])
    for n in [4,6]:
        bits=jnp.asarray(all_bits(n));logs=log_psi(params,bits,vector,model)
        assert abs(float(jnp.exp(2*logs).sum())-1) < 1e-12
        np.testing.assert_allclose(logs,log_psi(params,1-bits,vector,model),atol=1e-14,rtol=0)
    bits=jnp.asarray(all_bits(4));delta=1e-4
    psi=np.exp(np.asarray(log_psi(params,bits,vector,model)))
    plus=np.exp(np.asarray(log_psi(params,bits,vector.at[3].add(delta),model)))
    minus=np.exp(np.asarray(log_psi(params,bits,vector.at[3].add(-delta),model)))
    qgt=quantum_geometric_tensor(np.asarray(coupling_scores(params,bits,vector,model)),psi**2)
    np.testing.assert_allclose(qgt[3,3].real,fidelity_metric(psi,plus,minus,delta),atol=5e-8,rtol=0)


def test_compiled_sampler_audit_matches_existing_draws_and_measurement():
    model=GRUConfig(4);params=init_model(jax.random.PRNGKey(71),model)
    key=jax.random.PRNGKey(31);g=IsingCouplings(h=1.01)
    expected=_audit_sampled(params,model,4,g,key,128)
    actual=controls.audit_row(params,model,4,g.h,key,128)
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(actual[name],expected[name],atol=1e-12,rtol=0)


def test_runtime_guard_rejects_a_changed_runner(monkeypatch):
    protocol=controls.build_protocol([71])
    monkeypatch.setattr(controls,"launcher_hash",lambda:"changed")
    with pytest.raises(ValueError,match="changed since declaration"):
        controls.check_runtime(protocol)
