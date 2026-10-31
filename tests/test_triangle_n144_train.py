"""Real optimized-state checkpoint replay for the reusable native SR driver."""
import jax
import jax.numpy as jnp
import json
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_n144_train import (
    checked_training_gate, make_training_transition,
)
from su2_nqs.frontier.artifacts import canonical_sha256, pytree_sha256
from su2_nqs.frontier.provenance import file_sha256
from su2_nqs.operators.sz_heisenberg import SzHeisenbergGraph
from su2_nqs.vmc.checkpoint import load_checkpoint, save_checkpoint
from su2_nqs.vmc.physical_exchange import initialize_spin_exchange_chains

jax.config.update('jax_enable_x64', True)


def test_unreleased_training_fails_before_loading_any_source_or_allocating_model():
    for release in (False, None, 'false', 'true', 1):
        with pytest.raises(ValueError, match='unreleased'):
            checked_training_gate(dict(main_agent_training_release=release))


@pytest.mark.parametrize('changed', ['projection_chunk', 'action_chunk', 'score_chunk_size'])
def test_training_cost_release_requires_the_actually_measured_execution_chunks(tmp_path, changed):
    """A same-H/same-state chunk4 certificate cannot budget a chunk32 run."""
    cost_registration_path = tmp_path/'cost'/'registration.json'
    cost_registration_path.parent.mkdir()
    cost_registration_path.write_text(json.dumps(dict(config=dict(binary_rank=8))))
    execution = dict(projection_chunk=4, action_chunk=4, sr=dict(score_chunk_size=2),
        sampling=dict(chains=8, records_per_chain=4),
        cost_parent=dict(directory=str(cost_registration_path.parent),
            registration_sha256=file_sha256(cost_registration_path)))
    registration = dict(config=execution, sources={})
    registration['sha256'] = canonical_sha256(registration)
    model = dict(length=12, multiplicity_cap=16, maximum_q=10)
    checkpoint = tmp_path/'gate_checkpoint.pkl'
    checkpoint.write_bytes(b'immutable-execution-checkpoint-fixture')
    gate = dict(status='completed', cost_gate_passed=True, model=model, j2_over_j1=0,
        optimizer_updates_committed=0, registration_sha256=registration['sha256'],
        full_A1_images=1728, Hamiltonian_bonds=432, native_record_count=32,
        checkpoint_sha256=file_sha256(checkpoint))
    result_path = tmp_path/'results.json'
    result_path.write_text(json.dumps(gate))
    records = dict(registration=registration,
        results=gate,
        process_replay=dict(status='passed', full_next_sampler_bitwise_exact=True,
            source_results_sha256=file_sha256(result_path), checkpoint_sha256=file_sha256(checkpoint)),
        accounting=dict(status='completed'))
    request = dict(directory=str(tmp_path))
    for name, record in records.items():
        path = tmp_path/(name+'.json')
        path.write_text(json.dumps(record))
        request[name+'_sha256'] = file_sha256(path)
    spec = dict(main_agent_training_release=True, native_gate=request, model=model,
        j2_over_j1=0, binary_rank=8, projection_chunk=4, action_chunk=4, sr=dict(score_chunk_size=2),
        seconds_cap=120, training=dict(chains=8, records_per_chain=4),
        budget=dict(maximum_A100_hours=4, forecast_seconds=60))
    assert checked_training_gate(spec, root=tmp_path)['cost_gate_passed']
    if changed == 'score_chunk_size':
        spec['sr'][changed] = 4
    else:
        spec[changed] = 32
    with pytest.raises(ValueError, match='execution chunks'):
        checked_training_gate(spec, root=tmp_path)


def test_two_real_N6_MH_full_H_SR_updates_restore_the_next_complete_transition(tmp_path):
    """Use the full N6 fixed-Sz H, real MH draws and the actual SR optimizer.

    No eigensolve, fake optimizer, fixed invented update or mocked H enters.
    N36/N144 CG/A1 kernels have independent existing exact tests; here the
    optimizer/checkpoint/controller flow is checked with a cheap exact H.
    """
    n = 6
    edges = tuple((i, (i+1) % n) for i in range(n))
    graph = SzHeisenbergGraph(n, tuple((a, b, 1.) for a, b in edges), n_up=3)
    basis = jnp.asarray(graph.basis, jnp.int32)
    matrix = jnp.asarray(graph.to_dense())
    powers = jnp.asarray(1 << np.arange(n-1, -1, -1), jnp.int32)
    rng = np.random.default_rng(20261018831)
    features = jnp.asarray(rng.normal(size=(1 << n, 5)))
    fixed_sign = jnp.asarray(rng.choice([-1., 1.], size=1 << n))
    index = jnp.full((1 << n,), -1, jnp.int32).at[basis @ powers].set(
        jnp.arange(len(basis), dtype=jnp.int32))
    initial = dict(local=jnp.asarray([.03, -.02, .01]), top=jnp.asarray([-.015, .025]))

    def prepare(params):
        values = jnp.concatenate((params['local'], params['top']))
        return values+.03*values**2

    def amplitude(prepared, bits):
        packed = bits @ powers
        return fixed_sign[packed]*jnp.exp(features[packed] @ prepared)

    def exact_energy(params):
        values = jax.vmap(amplitude, in_axes=(None, 0))(prepare(params), basis)
        return jnp.vdot(values, matrix @ values).real/jnp.vdot(values, values).real

    @jax.jit
    def evaluate(prepared, samples):
        values = jax.vmap(amplitude, in_axes=(None, 0))(prepared, basis)
        hp = matrix @ values
        ids = index[samples @ powers]
        return values[ids], hp[ids]

    transition = make_training_transition(prepare, amplitude, evaluate, initial,
        n_sites=n, training=dict(chains=4, records_per_chain=8,
            initial_burn_proposals=30, refresh_proposals=6, proposals_between=6),
        sampler_options=dict(bond_proposal_probability=.75, lazy_probability=.05),
        exchange_edges=edges,
        sr_options=dict(damping=.1, step_size=.01, fisher_trust_radius=.02,
            maximum_overlap_distance=.02, minimum_ess_fraction=.8,
            backtracking_steps=8, score_chunk_size=4))
    params = initial
    optimizer = dict(completed_steps=jnp.asarray(0, jnp.int64),
        accepted_updates=jnp.asarray(0, jnp.int64), damping=jnp.asarray(.1))
    key, init = jax.random.split(jax.random.PRNGKey(20261018832))
    states = initialize_spin_exchange_chains(init, n_chains=4, n_sites=n, n_up=3)
    for step in range(2):
        params, optimizer, states, key, rows, energy, sampling, diagnostic = transition(
            params, optimizer, states, key)
        assert int(optimizer['completed_steps']) == step+1
        assert bool(diagnostic['accepted'])
        assert float(diagnostic['gradient_norm']) > 1e-8
        assert float(diagnostic['parameter_step_norm']) > 1e-8
        assert np.isfinite(np.asarray(energy)).all()
        np.testing.assert_array_equal(np.asarray(rows).sum(-1), 3)
    assert int(optimizer['accepted_updates']) == 2
    assert float(exact_energy(params)) < float(exact_energy(initial))
    assert pytree_sha256(params) != pytree_sha256(initial)
    assert all(np.linalg.norm(np.asarray(a-b)) > 1e-8 for a, b in
               zip(jax.tree.leaves(params), jax.tree.leaves(initial), strict=True))
    checkpoint = tmp_path/'trained.pkl'
    save_checkpoint(checkpoint, params, optimizer_state=optimizer,
        metadata=dict(sampler_states=states, rng=key, trained=True))
    restored = load_checkpoint(checkpoint)
    assert pytree_sha256(restored['optimizer_state']) == pytree_sha256(optimizer)
    expected = transition(params, optimizer, states, key)
    actual = transition(restored['params'], restored['optimizer_state'],
        restored['metadata']['sampler_states'], restored['metadata']['rng'])
    assert pytree_sha256(actual) == pytree_sha256(expected)
    assert int(actual[1]['completed_steps']) == 3
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(a, b)
